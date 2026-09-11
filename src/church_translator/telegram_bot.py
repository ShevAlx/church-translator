"""Telegram control for the booth Raspberry Pi — the menu bar app's job, from a phone.

The Pi lives in the church with no screen and nobody at it. What an operator
did from the 🎙️ menu (▶️/⏹, watching for ⚠️ and 🐢) happens here instead:
buttons in the chat start and stop translation, and the bot messages every
problem the menu bar would have shown. Unlike the menu bar it also fixes the
common ones itself — the operator guide's "⏹ then ▶️" — because on the Pi there
is no one to press the buttons (see config.BotConfig).

Run `church-translator-bot` from the project folder (it reads config.yaml and
.env from there); on the Pi it is the systemd user service in deploy/.

Access is limited to the Telegram user ids in TELEGRAM_ALLOWED_USER_IDS. Anyone
else gets their own id back and nothing more — which is also how you find the
id to add in the first place.

Plain urllib against the Bot API, no Telegram library: long polling and
sendMessage are all this needs, and one less dependency is one less thing to
break on a machine nobody logs into.
"""

from __future__ import annotations

import collections
import datetime as dt
import http.client
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from .audio_io import list_devices, refresh_devices, resolve_device
from .config import AppConfig, BotConfig, load_config
from .session import LiveSession, StopReport
from .usage_log import prune_dated_folders

CONFIG_PATH = Path("config.yaml")
# Survives a crash or a power cut, so the next boot can say translation died
# mid-service instead of greeting everyone as if nothing happened.
STATE_PATH = Path("logs/bot-state.json")
SERVICE_NAME = "church-translator-bot"

POLL_TIMEOUT_S = 25  # Telegram long poll; the HTTP timeout sits a little above it
WATCHDOG_TICK_S = 5.0
# A state-changing command older than this is not executed. Taps made while
# the Pi had no internet arrive in a burst once it reconnects — a ⏹ from twenty
# minutes ago must not stop the service restarted since, and a ▶️ tapped last
# Sunday must not start one when the Pi boots today.
STALE_COMMAND_S = 120
ERROR_REPORT_EVERY_S = 120.0  # pipeline errors: the first at once, then one summary at most this often
LOW_DISK_GB = 5.0
TELEGRAM_TEXT_LIMIT = 4000  # the API's hard limit is 4096

# Network trouble or Telegram having a moment: worth retrying.
TRANSIENT = (OSError, http.client.HTTPException)

BTN_START = "▶️ Старт"
BTN_STOP = "⏹ Стоп"
BTN_STATUS = "📊 Статус"
BTN_RESTART = "🔄 Перезапуск"
BTN_TEST = "🧪 Тест каналов"

KEYBOARD = {
    "keyboard": [
        [{"text": BTN_START}, {"text": BTN_STOP}],
        [{"text": BTN_STATUS}, {"text": BTN_RESTART}],
        [{"text": BTN_TEST}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
}
BUTTON_TO_COMMAND = {BTN_START: "go", BTN_STOP: "stop", BTN_STATUS: "status", BTN_RESTART: "restart", BTN_TEST: "test"}
ACTIONS = {"go", "stop", "restart", "test"}  # the commands STALE_COMMAND_S guards

COMMANDS = [  # setMyCommands — the "Menu" list next to the chat input
    ("go", "Запустить перевод"),
    ("stop", "Остановить перевод"),
    ("status", "Состояние перевода и малины"),
    ("restart", "Перезапустить перевод"),
    ("test", "Тест каналов: тоны вместо перевода, без API"),
    ("devices", "Звуковые устройства, которые видит малина"),
    ("log", "Последние строки журнала"),
    ("help", "Что умеет бот"),
]

HELP = (
    "Бот управляет переводом на малине в церкви.\n\n"
    f"{BTN_START} — запустить перевод (/go)\n"
    f"{BTN_STOP} — остановить (/stop). После службы — обязательно: распознавание оплачивается "
    "за время открытого соединения.\n"
    f"{BTN_STATUS} — идёт ли перевод, отставание, сигнал с микшера, сеть (/status)\n"
    f"{BTN_RESTART} — остановить и сразу запустить заново (/restart)\n"
    f"{BTN_TEST} — вместо перевода два разных тона по каналам: проверка, что каждый язык идёт "
    "в свой передатчик. Без API и без затрат (/test)\n\n"
    "/devices — какие звуковые устройства видит малина\n"
    "/log — последние строки журнала\n\n"
    "О проблемах бот пишет сам. Обрыв связи с распознаванием, пропавшую звуковую карту и "
    "затяжное отставание он чинит перезапуском и сообщает, что сделал."
)


# -- Telegram transport --------------------------------------------------------


class TelegramError(Exception):
    """Telegram answered and refused (bad token, blocked bot, unknown chat) —
    sending the same request again will not help."""


class TelegramAPI:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}/"

    def call(self, method: str, http_timeout: float = 15.0, **params):
        req = urllib.request.Request(
            self._base + method,
            data=json.dumps(params).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as exc:
            # Telegram explains itself in the body. 429 (flood control) and 5xx
            # pass with time; anything else is a refusal. Never let the URL into
            # a message: it carries the token.
            try:
                payload = json.load(exc)
            except ValueError:
                payload = {"description": f"HTTP {exc.code}"}
            if exc.code == 429 or exc.code >= 500:
                raise ConnectionError(f"Telegram {exc.code}: {payload.get('description')}") from None
            raise TelegramError(f"{exc.code}: {payload.get('description')}") from None
        if not payload.get("ok"):
            raise TelegramError(f"{payload.get('error_code')}: {payload.get('description')}")
        return payload["result"]


class Outbox(threading.Thread):
    """Delivers messages in order and holds them through an outage.

    The alerts that matter most — recognition lost, the line too slow — arrive
    exactly when the internet is flaky, so a failed send is retried, never
    dropped, and a message that gets out late says when it happened.
    """

    def __init__(self, api: TelegramAPI, chat_ids: list[int]):
        super().__init__(name="telegram-outbox", daemon=True)
        self._api = api
        self._chat_ids = list(chat_ids)
        self._items: collections.deque[tuple[int, str, bool, float]] = collections.deque()
        self._cv = threading.Condition()

    def send(self, text: str, chat_id: int | None = None, keyboard: bool = False) -> None:
        """chat_id=None: to everyone on the allow-list."""
        targets = [chat_id] if chat_id is not None else self._chat_ids
        with self._cv:
            for target in targets:
                self._items.append((target, text, keyboard, time.time()))
            self._cv.notify()

    def run(self) -> None:
        backoff = 2.0
        while True:
            with self._cv:
                while not self._items:
                    self._cv.wait()
                chat_id, text, keyboard, created = self._items[0]
            if time.time() - created > 60:
                text += f"\n\n(доставлено с опозданием — событие в {_clock(created)})"
            params: dict = {"chat_id": chat_id, "text": text[:TELEGRAM_TEXT_LIMIT]}
            if keyboard:
                params["reply_markup"] = KEYBOARD
            try:
                self._api.call("sendMessage", **params)
            except TRANSIENT as exc:
                print(f"[bot] send failed, retrying in {backoff:.0f}s: {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            except TelegramError as exc:
                print(f"[bot] Telegram refused a message to {chat_id}: {exc}")
            backoff = 2.0
            with self._cv:
                self._items.popleft()

    def flush(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cv:
                if not self._items:
                    return True
            time.sleep(0.2)
        return False


# -- formatting and the Pi's own vitals ----------------------------------------


def _clock(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def _duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _mode_label(mode: str) -> str:
    return {"real": "боевой режим", "mock": "тест каналов"}.get(mode, mode)


def _channels_line(config: AppConfig) -> str:
    return ", ".join(f"{lang.code} → выход {lang.output_channel + 1}" for lang in config.languages)


def _report_text(report: StopReport) -> str:
    lines = [f"Длительность {_duration(report.duration_s)} ({_mode_label(report.mode)})"]
    if report.mode == "real":
        lines.append(f"Озвучено {report.tts_chars:,} симв. в {report.tts_segments} фразах".replace(",", " "))
    if report.errors:
        lines.append(f"Ошибок перевода: {report.errors}")
    if any(report.dropped.values()):
        lines.append(
            "Пропущено фраз из-за отставания: " + ", ".join(f"{code} {n}" for code, n in report.dropped.items())
        )
    return "\n".join(lines)


def _describe(exc: Exception) -> str:
    if isinstance(exc, KeyError):  # a provider reading os.environ["..._API_KEY"]
        return f"в .env не задан {exc.args[0]}"
    return str(exc) or type(exc).__name__


def _hint(message: str) -> str:
    """What to actually do about an error, for someone standing in the church."""
    text = message.lower()
    if "в .env не задан" in text:
        return ("Допишите ключ в ~/church-translator/.env на малине и перезапустите бота: "
                "systemctl --user restart church-translator-bot")
    if "no input device" in text or "no output device" in text:
        return "Scarlett не видна: проверьте USB-кабель и что она горит. /devices — что видит малина."
    if "assemblyai" in text or "streaming api" in text:
        return "Не поднялось распознавание: проверьте интернет на малине (/status) и ключи в .env."
    if "402" in text or "credit" in text or "quota" in text:
        return "Похоже, кончились кредиты — Cartesia: включить overages на play.cartesia.ai."
    if "401" in text or "authentication" in text or "api key" in text or "api_key" in text:
        return "Сервис не принял ключ — проверьте ключи в .env на малине."
    if "unavailable" in text or "busy" in text or "paerrorcode" in text or "invalid" in text:
        return "Звуковую карту не удалось открыть (занята или не тот формат). Подробности — /log."
    return ""


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _direct_monitor() -> str | None:
    """"Mono"/"Stereo" while the Scarlett's Direct Monitor is on, else None.

    Direct Monitor mixes the inputs into the outputs in hardware. At home that
    is how you hear the original and the translation together; in church it
    puts the English into the FM transmitters under the translation, and the
    program never sees it — its own recordings hold only what it sends.
    Linux only (the scarlett2 driver's ALSA control); None wherever it can't
    be read, so the Mac and other interfaces simply get no warning.
    """
    try:
        cards = Path("/proc/asound/cards").read_text()
    except OSError:
        return None
    for line in cards.splitlines():
        fields = line.split()
        if fields and fields[0].isdigit() and "Scarlett" in line:
            state = re.search(r"Item0: '(\w+)'", _run(["amixer", "-c", fields[0], "sget", "Direct Monitor"]))
            return state.group(1) if state and state.group(1) != "Off" else None
    return None


def _direct_monitor_line() -> str | None:
    mode = _direct_monitor()
    if mode is None:
        return None
    return (f"🎧 На Scarlett включён Direct Monitor ({mode}): английский оригинал идёт в передатчики "
            "вместе с переводом. Дома для прослушки — нормально, в церкви — отожмите кнопку 🎧 на Scarlett.")


def _local_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # UDP connect sends nothing; it just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return None


def _network_line() -> str:
    links = []
    for line in _run(["nmcli", "-t", "-f", "TYPE,STATE", "dev"]).splitlines():
        kind, _, state = line.partition(":")
        if kind == "ethernet" and state.startswith("connected"):
            links.append("кабель")
    # --rescan no: a Wi-Fi scan in the middle of a service is a latency spike
    # on the very link the translation depends on.
    for line in _run(["nmcli", "-t", "-f", "ACTIVE,SIGNAL,SSID", "dev", "wifi", "list", "--rescan", "no"]).splitlines():
        if line.startswith("yes:"):
            signal_pct, _, ssid = line[4:].partition(":")
            ssid = ssid.replace("\\:", ":")
            links.append(f"Wi-Fi «{ssid}» {signal_pct}%")
            break
    ip = _local_ip()
    where = ", ".join(links) if links else "сеть"
    return f"🌐 {where}, IP {ip}" if ip else "🌐 нет сети"


def _system_lines() -> list[str]:
    lines = [_network_line()]
    vitals = []
    try:
        temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        vitals.append(f"CPU {temp:.0f}°C")
    except (OSError, ValueError):
        pass
    free_gb = shutil.disk_usage(".").free / 1e9
    vitals.append(f"свободно {free_gb:.0f} ГБ")
    if vitals:
        lines.append("🖥 " + ", ".join(vitals))
    if free_gb < LOW_DISK_GB:
        lines.append("⚠️ Мало места на карте — уменьшите logging.keep_days в config.yaml.")
    # A weak power supply is the classic cause of USB audio dropouts on a Pi.
    throttled = _run(["vcgencmd", "get_throttled"]).strip()  # "throttled=0x50000"
    if throttled.startswith("throttled="):
        flags = int(throttled.split("=", 1)[1], 16)
        if flags & 0x1:
            lines.append("⚡ Сейчас не хватает питания — нужен штатный блок питания 5V/3A.")
        elif flags & 0x10000:
            lines.append("⚡ С момента загрузки питания не хватало — проверьте блок питания.")
    return lines


def _read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(**state) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError as exc:
        print(f"[bot] could not write {STATE_PATH}: {exc}")


# -- the bot -------------------------------------------------------------------


class BoothBot:
    def __init__(self, api: TelegramAPI, allowed_ids: list[int], config_path: Path = CONFIG_PATH):
        self.api = api
        self.allowed = set(allowed_ids)
        self.config_path = config_path
        self.outbox = Outbox(api, allowed_ids)
        # Held around every start/stop: the poll thread (buttons) and the
        # watchdog (auto-recovery) both drive the same session.
        self._lock = threading.RLock()
        self.session: LiveSession | None = None
        self._stopping = threading.Event()
        self._commands_set = False

        # Auto-recovery. An "episode" starts when a problem is detected and ends
        # when translation is running again or the bot gives up on it.
        self._episodes: collections.deque[float] = collections.deque()  # start times, for max_recoveries
        self._episode_started: float | None = None
        self._retry_at: float | None = None
        self._retry_mode: str | None = None
        self._retry_attempts = 0
        self._lag_since: float | None = None
        self._lag_gave_up = False
        self._silence_warned = False

        # Pipeline errors arrive on worker threads, possibly many per minute.
        self._errors_lock = threading.Lock()
        self._last_error_report = 0.0
        self._pending_errors = 0
        self._last_error_text = ""

    # -- lifecycle -------------------------------------------------------------

    def run(self) -> None:
        self.outbox.start()
        self.outbox.send(self._boot_text(), keyboard=True)
        threading.Thread(target=self._watchdog, name="watchdog", daemon=True).start()
        try:
            self._poll()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self._stopping.set()
            with self._lock:
                if self.session is not None:
                    report = self._stop_locked()
                    self.outbox.send(f"⏹ Бот выключается — перевод остановлен.\n{_report_text(report)}")
            self.outbox.flush(timeout=5.0)

    def _boot_text(self) -> str:
        lines = [f"🟢 Малина {socket.gethostname()} включилась, бот на связи."]
        state = _read_state()
        if state.get("running"):
            since = state.get("started", "?")
            lines.append(
                f"⚠️ Прошлый запуск оборвался посреди перевода (запущен {since}): перезагрузка, "
                f"пропало питание или сбой. Сейчас перевод НЕ идёт — нажмите {BTN_START}."
            )
            _write_state(running=False)
        try:
            config = load_config(self.config_path)
        except Exception as exc:  # noqa: BLE001 — a broken config must reach the phone, not only the journal
            lines.append(f"❌ config.yaml не читается: {exc}")
        else:
            removed = sum(
                prune_dated_folders(d, config.logging.keep_days)
                for d in (config.logging.recordings_dir, config.logging.debug_audio_dir)
            )
            if removed:
                print(f"[bot] pruned {removed} dated folder(s) older than {config.logging.keep_days} days")
            with self._lock:
                lines.append(self._scarlett_line(config))
            if dm := _direct_monitor_line():
                lines.append(dm)
            lines.append(f"Режим по кнопке {BTN_START}: {_mode_label(config.pipeline.mode)}; {_channels_line(config)}")
        lines.extend(_system_lines())
        return "\n".join(lines)

    # -- Telegram polling --------------------------------------------------------

    def _poll(self) -> None:
        offset: int | None = None
        backoff = 2.0
        while not self._stopping.is_set():
            params: dict = {"timeout": POLL_TIMEOUT_S, "allowed_updates": ["message"]}
            if offset is not None:
                params["offset"] = offset
            try:
                if not self._commands_set:
                    self.api.call(
                        "setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS]
                    )
                    self._commands_set = True
                updates = self.api.call("getUpdates", http_timeout=POLL_TIMEOUT_S + 10, **params)
            except TRANSIENT as exc:
                print(f"[bot] Telegram unreachable, retrying in {backoff:.0f}s: {exc}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            except TelegramError as exc:
                # 401 = wrong token; 409 = another copy of the bot is polling
                # (e.g. still running on the Mac). Neither fixes itself quickly.
                print(f"[bot] Telegram refused getUpdates: {exc}")
                time.sleep(30)
                continue
            backoff = 2.0
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    self._handle(update)
                except Exception:  # noqa: BLE001 — one bad update must not kill the bot
                    traceback.print_exc()

    def _handle(self, update: dict) -> None:
        msg = update.get("message") or {}
        if "text" not in msg:
            return
        chat_id = msg["chat"]["id"]
        user = msg.get("from") or {}
        uid = user.get("id")
        who = user.get("first_name") or user.get("username") or str(uid)
        if uid not in self.allowed:
            print(f"[bot] ignored message from unlisted user {uid} ({who})")
            self.outbox.send(
                "⛔ Этот бот управляет переводом в церкви, доступ только по списку.\n"
                f"Ваш Telegram ID: {uid}",
                chat_id,
            )
            return

        text = msg["text"].strip()
        cmd = BUTTON_TO_COMMAND.get(text)
        if cmd is None and text.startswith("/"):
            cmd = text[1:].split()[0].split("@")[0].lower()
        age = time.time() - msg.get("date", time.time())
        if cmd in ACTIONS and age > STALE_COMMAND_S:
            self.outbox.send(
                f"⌛ «{text}» дошло с опозданием на {age / 60:.0f} мин (не было интернета?) — не выполняю. "
                "Если ещё нужно, нажмите снова.",
                chat_id, keyboard=True,
            )
            return
        print(f"[bot] {who} ({uid}): {text}")
        handlers = {
            "go": lambda: self.cmd_start(chat_id, who, mode=None),
            "test": lambda: self.cmd_start(chat_id, who, mode="mock"),
            "stop": lambda: self.cmd_stop(chat_id, who),
            "restart": lambda: self.cmd_restart(chat_id, who),
            "status": lambda: self.outbox.send(self._status_text(), chat_id, keyboard=True),
            "devices": lambda: self.cmd_devices(chat_id),
            "log": lambda: self.cmd_log(chat_id),
        }
        handler = handlers.get(cmd)
        if handler is None:  # /start, /help, or anything typed by hand
            self.outbox.send(HELP, chat_id, keyboard=True)
        else:
            handler()

    # -- commands ----------------------------------------------------------------

    def cmd_start(self, chat_id: int, who: str, mode: str | None) -> None:
        with self._lock:
            if self.session is not None:
                s = self.session
                self.outbox.send(
                    f"Перевод уже идёт: {_mode_label(s.mode)}, {_duration(s.elapsed_s)}. "
                    f"Сменить режим — сначала {BTN_STOP}.",
                    chat_id, keyboard=True,
                )
                return
            self._end_episode()
            self.outbox.send("⏳ Запускаю…", chat_id)
            try:
                session = self._start_locked(mode)
            except Exception as exc:  # noqa: BLE001 — report it, the bot stays up
                traceback.print_exc()
                why = _describe(exc)
                self.outbox.send(f"❌ Не запустилось: {why}\n{_hint(why)}".strip(), chat_id, keyboard=True)
                return
        if session.mode == "mock":
            detail = "В каждом канале свой тон, по одному на каждую услышанную фразу. Перевода и затрат нет."
        else:
            detail = _channels_line(session.config)
        if dm := _direct_monitor_line():
            detail += f"\n{dm}"
        self.outbox.send(f"▶️ Перевод запущен ({_mode_label(session.mode)}) — {who}\n{detail}", keyboard=True)

    def cmd_stop(self, chat_id: int, who: str) -> None:
        with self._lock:
            had_retry = self._retry_at is not None
            self._end_episode()
            if self.session is None:
                note = " Ожидавший автоперезапуск отменён." if had_retry else ""
                self.outbox.send(f"Перевод и так остановлен.{note}", chat_id, keyboard=True)
                return
            report = self._stop_locked()
        self.outbox.send(f"⏹ Перевод остановлен — {who}\n{_report_text(report)}", keyboard=True)

    def cmd_restart(self, chat_id: int, who: str) -> None:
        with self._lock:
            mode = self.session.mode if self.session is not None else None
            self._end_episode()
            if self.session is not None:
                self._stop_locked()
            self.outbox.send("⏳ Перезапускаю…", chat_id)
            try:
                session = self._start_locked(mode)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                why = _describe(exc)
                self.outbox.send(
                    f"❌ Остановил, но не запустилось: {why}\n{_hint(why)}".strip(), chat_id, keyboard=True
                )
                return
        self.outbox.send(f"🔄 Перевод перезапущен ({_mode_label(session.mode)}) — {who}", keyboard=True)

    def cmd_devices(self, chat_id: int) -> None:
        with self._lock:
            note = ""
            if self.session is None:
                refresh_devices()
            else:
                note = "(перевод идёт — список на момент запуска)\n"
            devices = list_devices()
        self.outbox.send(f"{note}{devices}", chat_id)

    def cmd_log(self, chat_id: int) -> None:
        # --user-unit, not `--user -u`: on the Pi there are no per-user journal
        # files ("No journal files were found"), the service's lines live in the
        # system journal, which the user can read as a member of `adm`.
        out = _run(
            ["journalctl", "--user-unit", SERVICE_NAME, "-n", "40", "--no-pager", "--no-hostname", "-o", "short"],
            timeout=10,
        )
        self.outbox.send(out[-3500:] or "Журнал пуст или недоступен (бот запущен не как systemd-сервис?).", chat_id)

    # -- start/stop, caller holds _lock --------------------------------------------

    def _start_locked(self, mode: str | None) -> LiveSession:
        config = load_config(self.config_path)  # fresh, so config.yaml edits apply without restarting the bot
        if mode:
            config.pipeline.mode = mode
        refresh_devices()  # no stream is open: self.session is None here
        session = LiveSession(config, on_error=self._on_pipeline_error)
        session.start()
        self.session = session
        self._lag_since = None
        self._lag_gave_up = False
        self._silence_warned = False
        _write_state(running=True, mode=session.mode, started=dt.datetime.now().strftime("%d.%m %H:%M"))
        return session

    def _stop_locked(self) -> StopReport:
        session, self.session = self.session, None
        report = session.stop()
        _write_state(running=False)
        with self._errors_lock:
            self._pending_errors = 0  # the report carries the total
        for d in (session.config.logging.recordings_dir, session.config.logging.debug_audio_dir):
            prune_dated_folders(d, session.config.logging.keep_days)
        return report

    def _scarlett_line(self, config: AppConfig) -> str:
        """Only while stopped: re-scanning devices would tear down a live stream."""
        refresh_devices()
        try:
            resolve_device(config.audio.input_device, kind="input")
            resolve_device(config.audio.output_device or config.audio.input_device, kind="output")
        except ValueError:
            return f"🔌 Звуковая карта «{config.audio.input_device}» не найдена — проверьте USB. /devices"
        return f"🎛 Звуковая карта «{config.audio.input_device}» на месте"

    def _status_text(self) -> str:
        with self._lock:
            s = self.session
            lines = []
            if s is not None:
                limit = s.config.audio.max_backlog_s
                lines.append(f"🔴 Перевод идёт — {_duration(s.elapsed_s)} ({_mode_label(s.mode)})")
                if s.stt_error:
                    lines.append(f"⚠️ Распознавание оборвалось: {s.stt_error}")
                lag = s.stt_lag_s
                if lag is not None:
                    icon = "🐢" if lag > limit else "⏱"
                    lines.append(f"{icon} Отставание распознавания {lag:.1f} с (порог {limit:.0f} с)")
                if not s.audio_alive:
                    lines.append("🔌 Звуковая карта не отвечает")
                quiet = s.seconds_without_signal()
                peak = s.router.input_peak
                if quiet > 10 or peak <= 0:
                    lines.append(f"🔇 С микшера тишина уже {_duration(quiet)}")
                else:
                    lines.append(f"🎚 Сигнал с микшера {20 * math.log10(peak):.0f} dBFS")
                if s.mode == "real":
                    lines.append(f"Озвучено {s.usage.tts_chars} симв., ошибок {s.usage.errors}")
                dropped = s.router.dropped_report()
                if any(dropped.values()):
                    lines.append(f"Пропущено фраз из-за отставания: {sum(dropped.values())}")
            else:
                lines.append("⏹ Перевод остановлен")
                if self._retry_at is not None:
                    wait = max(0.0, self._retry_at - time.monotonic())
                    lines.append(f"🔄 Автоперезапуск: следующая попытка через {wait:.0f} с")
                try:
                    config = load_config(self.config_path)
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"❌ config.yaml не читается: {exc}")
                else:
                    lines.append(self._scarlett_line(config))
        if dm := _direct_monitor_line():
            lines.append(dm)
        lines.append("")
        lines.extend(_system_lines())
        return "\n".join(lines)

    # -- watchdog and auto-recovery --------------------------------------------------

    def _watchdog(self) -> None:
        while not self._stopping.wait(WATCHDOG_TICK_S):
            try:
                self._flush_errors()
                self._tick()
            except Exception:  # noqa: BLE001 — the watchdog must outlive anything it watches
                traceback.print_exc()

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            s = self.session
            if s is None:
                if self._retry_at is not None and now >= self._retry_at:
                    self._retry_start()
                return
            bot = s.config.bot

            if bot.max_session_hours and s.elapsed_s > bot.max_session_hours * 3600:
                report = self._stop_locked()
                self.outbox.send(
                    f"⏹ Автостоп: перевод шёл дольше {bot.max_session_hours:g} ч — похоже, забыли остановить. "
                    f"Распознавание оплачивается поминутно, поэтому остановил.\n{_report_text(report)}"
                )
                return
            if not s.audio_alive:
                self._recover("🔌 Звуковая карта перестала отвечать — выдернут USB или пропало питание Scarlett.",
                              fatal=True)
                return
            if s.stt_error:
                self._recover(f"⚠️ Оборвалась связь с распознаванием: {s.stt_error}", fatal=True)
                return

            lag = s.stt_lag_s
            if lag is not None and lag > s.config.audio.max_backlog_s:
                if self._lag_since is None:
                    self._lag_since = now
                    self.outbox.send(
                        f"🐢 Перевод отстаёт от проповедника на {lag:.0f} с — интернет не успевает. "
                        f"Если не пройдёт за {bot.lag_recover_after_s:.0f} с, перезапущу."
                    )
                elif not self._lag_gave_up and now - self._lag_since > bot.lag_recover_after_s:
                    self._recover(f"🐢 Отставание {lag:.0f} с не проходит.", fatal=False)
                    return
            elif self._lag_since is not None:
                self._lag_since = None
                self.outbox.send("✅ Отставание ушло, перевод снова успевает.")

            if bot.silence_alert_s:
                quiet = s.seconds_without_signal()
                if quiet > bot.silence_alert_s and not self._silence_warned:
                    self._silence_warned = True
                    self.outbox.send(
                        f"🔇 С микшера нет сигнала уже {_duration(quiet)}. Если служба идёт — проверьте кабель "
                        "X32 → Scarlett и что на пульте не выключена шина, которая идёт на перевод."
                    )
                elif quiet < 2 * WATCHDOG_TICK_S and self._silence_warned:
                    self._silence_warned = False
                    self.outbox.send("🔊 Сигнал с микшера появился.")

    def _recover(self, problem: str, fatal: bool) -> None:
        """Caller holds _lock and self.session is running but unhealthy.

        `fatal`: nothing reaches the listeners any more (dead recognition, no
        sound card), so if a restart is not allowed the session is stopped.
        A lag is not fatal — late translation beats none — so then it runs on.
        """
        s = self.session
        bot = s.config.bot
        now = time.monotonic()
        while self._episodes and now - self._episodes[0] > bot.recovery_window_min * 60:
            self._episodes.popleft()
        if not bot.auto_recover or len(self._episodes) >= bot.max_recoveries:
            why = ("автоперезапуск выключен в config.yaml" if not bot.auto_recover
                   else f"уже {len(self._episodes)} перезапусков за {bot.recovery_window_min:g} мин")
            if fatal:
                report = self._stop_locked()
                self.outbox.send(
                    f"{problem}\n⛔ Перевод остановлен ({why}). Нужно разобраться на месте, потом {BTN_START}.\n"
                    f"{_report_text(report)}"
                )
            else:
                self._lag_gave_up = True
                self.outbox.send(f"{problem}\nБольше не перезапускаю ({why}) — перевод идёт с отставанием.")
            return
        self._episodes.append(now)
        self._episode_started = now
        self._retry_attempts = 0
        mode = s.mode
        self._stop_locked()
        self.outbox.send(f"{problem}\n🔄 Перезапускаю ({len(self._episodes)}/{bot.max_recoveries})…")
        self._try_start(mode, bot)

    def _retry_start(self) -> None:
        """Caller holds _lock; a restart failed earlier and its retry is due."""
        try:
            bot = load_config(self.config_path).bot
        except Exception:  # noqa: BLE001 — a broken config fails the start itself, with the error sent there
            bot = BotConfig()
        if self._episode_started is not None and time.monotonic() - self._episode_started > bot.retry_give_up_min * 60:
            attempts = self._retry_attempts
            self._end_episode()
            self.outbox.send(
                f"⛔ Перевод так и не поднялся: {attempts} попыток за {bot.retry_give_up_min:g} мин. "
                f"Остановлен. Проверьте на месте интернет и Scarlett, потом {BTN_START}."
            )
            return
        self._try_start(self._retry_mode, bot)

    def _try_start(self, mode: str | None, bot: BotConfig) -> None:
        self._retry_attempts += 1
        try:
            self._start_locked(mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[bot] recovery start #{self._retry_attempts} failed: {exc}")
            self._retry_mode = mode
            self._retry_at = time.monotonic() + bot.retry_after_s
            if self._retry_attempts == 1:  # later failures stay quiet — one line per 30s is noise
                why = _describe(exc)
                self.outbox.send(
                    f"❌ Сразу не поднялось: {why}\n{_hint(why)}\n"
                    f"Пробую каждые {bot.retry_after_s:.0f} с до {bot.retry_give_up_min:g} мин, напишу, когда получится."
                )
            return
        attempts = self._retry_attempts
        self._end_episode()
        self.outbox.send("✅ Перевод снова идёт." + (f" (с {attempts}-й попытки)" if attempts > 1 else ""))

    def _end_episode(self) -> None:
        self._episode_started = None
        self._retry_at = None
        self._retry_mode = None
        self._retry_attempts = 0

    # -- pipeline errors (called on worker threads — never take _lock here) ------------

    def _on_pipeline_error(self, language: str, note: str) -> None:
        with self._errors_lock:
            now = time.monotonic()
            if now - self._last_error_report < ERROR_REPORT_EVERY_S:
                self._pending_errors += 1
                self._last_error_text = f"[{language}] {note}"
                return
            self._last_error_report = now
        hint = _hint(note)
        self.outbox.send(
            f"❗ Ошибка перевода [{language}]: {note}\nФраза пропущена, перевод продолжается."
            + (f"\n{hint}" if hint else "")
        )

    def _flush_errors(self) -> None:
        with self._errors_lock:
            if not self._pending_errors or time.monotonic() - self._last_error_report < ERROR_REPORT_EVERY_S:
                return
            count, last = self._pending_errors, self._last_error_text
            self._pending_errors = 0
            self._last_error_report = time.monotonic()
        self.outbox.send(f"❗ Ещё ошибок перевода: {count}. Последняя: {last}")


def _parse_ids(raw: str) -> list[int]:
    ids = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                ids.append(int(part))
            except ValueError:
                print(f"[bot] TELEGRAM_ALLOWED_USER_IDS: {part!r} is not a numeric id, ignored")
    return ids


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set — add it to .env (see .env.example)")
    allowed = _parse_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    if not allowed:
        print("[bot] TELEGRAM_ALLOWED_USER_IDS is empty: the bot will answer everyone with their id "
              "and obey no one. Message the bot, put your id in .env, restart.")

    def _terminate(_signum, _frame):
        raise SystemExit(0)  # unwinds run(), which stops a live session cleanly

    signal.signal(signal.SIGTERM, _terminate)
    BoothBot(TelegramAPI(token), allowed).run()


if __name__ == "__main__":
    main()
