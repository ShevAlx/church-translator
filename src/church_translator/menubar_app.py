"""Menu-bar app for the booth operator — report §09's third requirement
("channel routing without editing code") in its simplest possible form.

Everything here calls straight into the same AudioRouter/Pipeline classes
`cli.py run` uses — no subprocess, no `uv run` wrapper, so Start/Stop are
just method calls on objects this process already owns. That sidesteps the
SIGINT-forwarding problem that bit the subprocess-based smoke tests twice
during development (2026-08-19/20): there is no signal to forward, the
button handler calls `.stop()` directly.

Launch: `uv run church-translator-app` (or `.venv/bin/church-translator-app`).
Lives in the menu bar, not the Dock — this is a booth utility, not a document
window app.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import rumps
import sounddevice as sd
import yaml

from .audio_io import AudioRouter, resolve_device
from .config import AppConfig, LanguageConfig, load_config
from .pipeline import Pipeline
from .providers.mock import MockMT, MockSTT, MockTTS
from .usage_log import UsageLogger, date_folder

CONFIG_PATH = Path("config.yaml")
DEFAULT_TEMPLATE_PATH = Path("config.church.example.yaml")


def _save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Round-trips through plain PyYAML — this rewrites the file without the
    hand-written comments in config.church.example.yaml. That's a deliberate
    trade: once you're managing channels from the app, you don't need the
    inline YAML notes any more. The file stays human-readable either way."""
    data = {
        "audio": {
            "input_device": config.audio.input_device,
            "output_device": config.audio.output_device,
            "samplerate": config.audio.samplerate,
            "blocksize": config.audio.blocksize,
            "input_channel": config.audio.input_channel,
            "max_backlog_s": config.audio.max_backlog_s,
            "catchup_start_s": config.audio.catchup_start_s,
            "max_playback_rate": config.audio.max_playback_rate,
        },
        "source_languages": config.source_languages,
        "languages": [
            {"name": lang.name, "code": lang.code, "output_channel": lang.output_channel, "voice_id": lang.voice_id}
            for lang in config.languages
        ],
        "pipeline": {
            "mode": config.pipeline.mode,
            "stt_provider": config.pipeline.stt_provider,
            "mt_provider": config.pipeline.mt_provider,
            "tts_provider": config.pipeline.tts_provider,
        },
        # Round-tripped, not defaulted: this function rewrites the whole file, so
        # anything missing here is silently erased from the operator's config the
        # first time they touch a menu.
        "stt": {
            "end_of_turn_confidence_threshold": config.stt.end_of_turn_confidence_threshold,
            "min_turn_silence_ms": (
                config.stt.min_turn_silence_ms
            ),
            "max_turn_silence_ms": config.stt.max_turn_silence_ms,
            "partial_emit": config.stt.partial_emit,
            "partial_min_words": config.stt.partial_min_words,
            "partial_max_words": config.stt.partial_max_words,
            "partial_gap_ms": config.stt.partial_gap_ms,
        },
        "tts": {"speed": config.tts.speed},
        "logging": {
            "usage_log_path": config.logging.usage_log_path,
            "debug_audio_dir": config.logging.debug_audio_dir,
            "recordings_dir": config.logging.recordings_dir,
        },
    }
    header = "# Managed by the church-translator menu-bar app — edits here get overwritten.\n"
    path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _load_or_seed_config() -> AppConfig:
    if CONFIG_PATH.exists():
        return load_config(CONFIG_PATH)
    if DEFAULT_TEMPLATE_PATH.exists():
        cfg = load_config(DEFAULT_TEMPLATE_PATH)
        _save_config(cfg)
        return cfg
    raise SystemExit(f"No {CONFIG_PATH} and no {DEFAULT_TEMPLATE_PATH} to seed from.")


class ChurchTranslatorApp(rumps.App):
    def __init__(self):
        super().__init__("🎙️", quit_button=None)
        try:
            from dotenv import load_dotenv

            load_dotenv(Path(".env"))  # GUI launches don't inherit a shell's `source .env`
        except ImportError:
            pass

        self.config = _load_or_seed_config()
        self.router: AudioRouter | None = None
        self.pipeline: Pipeline | None = None
        self.session_start: float | None = None
        self._warned_stt = False

        self.status_item = rumps.MenuItem("○ Остановлено")
        self.toggle_item = rumps.MenuItem("▶️ Запустить перевод", callback=self.toggle_run)
        self.mode_item = rumps.MenuItem(
            f"Тестовый режим (mock), сейчас: {'вкл' if self.config.pipeline.mode == 'mock' else 'выкл'}",
            callback=self.toggle_mode,
        )
        self.mode_item.state = self.config.pipeline.mode == "mock"

        self._build_menu()
        rumps.Timer(self._tick, 5).start()

    # -- menu construction ---------------------------------------------------

    def _build_menu(self) -> None:
        self.menu.clear()
        self.menu = [
            self.status_item,
            self.toggle_item,
            self.mode_item,
            None,
            rumps.MenuItem("Входное устройство", callback=None),
            *self._device_items(kind="input"),
            None,
            rumps.MenuItem("Выходное устройство", callback=None),
            *self._device_items(kind="output"),
            None,
            self._input_channel_menu(),
            self._languages_menu(),
            None,
            rumps.MenuItem("Открыть логи", callback=self.open_logs),
            rumps.MenuItem("Открыть debug-аудио", callback=self.open_debug_audio),
            rumps.MenuItem("Открыть записи служб", callback=self.open_recordings),
            None,
            rumps.MenuItem("Выход", callback=self.quit_app),
        ]

    def _device_items(self, kind: str) -> list[rumps.MenuItem]:
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        current = self.config.audio.input_device if kind == "input" else self.config.audio.output_device
        items = []
        for dev in sd.query_devices():
            if dev[channel_key] <= 0:
                continue
            label = f"   {dev['name']} ({dev[channel_key]} ch)"
            item = rumps.MenuItem(label, callback=lambda sender, k=kind, name=dev["name"]: self._set_device(k, name))
            item.state = bool(current and current.lower() in dev["name"].lower())
            items.append(item)
        return items

    def _input_channel_menu(self) -> rumps.MenuItem:
        parent = rumps.MenuItem("Входной канал")
        n_channels = self._channel_count("input", self.config.audio.input_device) or 8
        for ch in range(n_channels):
            item = rumps.MenuItem(f"Канал {ch}", callback=lambda sender, c=ch: self._set_input_channel(c))
            item.state = ch == self.config.audio.input_channel
            parent.add(item)
        return parent

    def _languages_menu(self) -> rumps.MenuItem:
        parent = rumps.MenuItem("Каналы языков")
        n_channels = self._channel_count("output", self.config.audio.output_device) or 8
        for lang in self.config.languages:
            lang_item = rumps.MenuItem(f"{lang.name} ({lang.code})")
            for ch in range(n_channels):
                item = rumps.MenuItem(
                    f"Канал {ch}", callback=lambda sender, c=ch, code=lang.code: self._set_output_channel(code, c)
                )
                item.state = ch == lang.output_channel
                lang_item.add(item)
            parent.add(lang_item)
        return parent

    def _channel_count(self, kind: str, device_substring: str | None) -> int | None:
        if not device_substring:
            return None
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        for dev in sd.query_devices():
            if device_substring.lower() in dev["name"].lower() and dev[channel_key] > 0:
                return dev[channel_key]
        return None

    # -- config-mutating callbacks --------------------------------------------

    def _set_device(self, kind: str, name: str) -> None:
        if kind == "input":
            self.config.audio.input_device = name
        else:
            self.config.audio.output_device = name
        _save_config(self.config)
        self._build_menu()
        rumps.notification("church-translator", "Устройство изменено", name)

    def _set_input_channel(self, channel: int) -> None:
        self.config.audio.input_channel = channel
        _save_config(self.config)
        self._build_menu()

    def _set_output_channel(self, lang_code: str, channel: int) -> None:
        for lang in self.config.languages:
            if lang.code == lang_code and lang.output_channel != channel:
                # swap, so two languages can never silently collide on one channel
                for other in self.config.languages:
                    if other.output_channel == channel:
                        other.output_channel = lang.output_channel
                lang.output_channel = channel
        _save_config(self.config)
        self._build_menu()

    def toggle_mode(self, sender) -> None:
        if self.running:
            rumps.alert("Сначала остановите перевод, потом меняйте режим.")
            return
        self.config.pipeline.mode = "real" if self.config.pipeline.mode == "mock" else "mock"
        sender.state = self.config.pipeline.mode == "mock"
        sender.title = f"Тестовый режим (mock), сейчас: {'вкл' if self.config.pipeline.mode == 'mock' else 'выкл'}"
        _save_config(self.config)

    # -- start/stop ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.pipeline is not None

    def toggle_run(self, sender) -> None:
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        try:
            input_idx = resolve_device(self.config.audio.input_device, kind="input")
            output_source = self.config.audio.output_device or self.config.audio.input_device
            output_idx = resolve_device(output_source, kind="output")

            self.router = AudioRouter(
                input_device=input_idx,
                output_device=output_idx,
                samplerate=self.config.audio.samplerate,
                blocksize=self.config.audio.blocksize,
                input_channel=self.config.audio.input_channel,
                output_channels=[lang.output_channel for lang in self.config.languages],
                max_backlog_s=self.config.audio.max_backlog_s,
                catchup_start_s=self.config.audio.catchup_start_s,
                max_playback_rate=self.config.audio.max_playback_rate,
            )
            usage_logger = UsageLogger(self.config.logging.usage_log_path)
            session_id = usage_logger.new_session_id()

            if self.config.logging.recordings_dir:
                recordings_dir = Path(self.config.logging.recordings_dir) / date_folder(session_id)
                for lang in self.config.languages:
                    self.router.start_recording(lang.output_channel, recordings_dir / f"{session_id}-{lang.code}.wav")
                # Same timeline as the outputs — this is what makes the delay
                # measurable after the fact, and makes the service replayable.
                self.router.start_input_recording(recordings_dir / f"{session_id}-source.wav")

            if self.config.pipeline.mode == "mock":
                stt = MockSTT(samplerate=self.config.audio.samplerate)
                mt_by_lang = {lang.code: MockMT() for lang in self.config.languages}
                tts_by_lang = {
                    lang.code: MockTTS(samplerate=self.config.audio.samplerate, base_hz=440.0 * (i + 1))
                    for i, lang in enumerate(self.config.languages)
                }
            else:
                from .providers.assemblyai_stt import AssemblyAISTT
                from .providers.cartesia_tts import CartesiaTTS
                from .providers.claude_mt import ClaudeMT

                stt = AssemblyAISTT(
                    samplerate=self.config.audio.samplerate,
                    language_codes=self.config.source_languages or None,
                    end_of_turn_confidence_threshold=self.config.stt.end_of_turn_confidence_threshold,
                    min_turn_silence_ms=(
                        self.config.stt.min_turn_silence_ms
                    ),
                    max_turn_silence_ms=self.config.stt.max_turn_silence_ms,
                    partial_emit=self.config.stt.partial_emit,
                    partial_min_words=self.config.stt.partial_min_words,
                    partial_max_words=self.config.stt.partial_max_words,
                    partial_gap_ms=self.config.stt.partial_gap_ms,
                )
                mt_by_lang = {lang.code: ClaudeMT() for lang in self.config.languages}
                tts_by_lang = {
                    lang.code: CartesiaTTS(samplerate=self.config.audio.samplerate, speed=self.config.tts.speed)
                    for lang in self.config.languages
                }

            self.pipeline = Pipeline(
                self.config, self.router, usage_logger,
                stt=stt, mt_by_language=mt_by_lang, tts_by_language=tts_by_lang,
                debug_audio_dir=self.config.logging.debug_audio_dir, session_id=session_id,
            )
            self.router.start()
            self.pipeline.start()
            self.session_start = time.monotonic()
            self._warned_stt = False

            self.toggle_item.title = "⏹ Остановить перевод"
            self.title = "🔴"
            self._tick(None)
        except Exception as exc:  # noqa: BLE001 — surface to the operator, don't just crash the menu bar
            self.router = None
            self.pipeline = None
            rumps.alert(f"Не удалось запустить: {exc}")

    def _stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        if self.router:
            underruns = self.router.underrun_report()
            dropped = self.router.dropped_report()
            self.router.stop()
        else:
            underruns, dropped = {}, {}
        self.pipeline = None
        self.router = None
        self.session_start = None
        self.toggle_item.title = "▶️ Запустить перевод"
        self.title = "🎙️"
        self.status_item.title = "○ Остановлено"
        if any(dropped.values()):
            # Shown because it is the one number that says whether listeners
            # actually heard the service or a skipped version of it.
            rumps.notification(
                "church-translator", "Остановлено",
                f"Пропущено реплик (отставание): {dropped}. Много — увеличьте max_backlog_s.",
            )
        elif any(underruns.values()):
            rumps.notification("church-translator", "Остановлено", f"Underruns по каналам: {underruns}")

    def _tick(self, _sender) -> None:
        if self.session_start is None:
            return
        elapsed = int(time.monotonic() - self.session_start)
        clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # A dead STT stream leaves everything else looking healthy — audio still
        # flows, every thread is still alive, the timer still counts up. Without
        # this the booth sees "● Работает" while the headphones are silent
        # (happened for real during the 2026-08-23 pre-service check).
        error = self.pipeline.stt_error if self.pipeline is not None else None
        if error:
            self.title = "⚠️"
            self.status_item.title = f"⚠️ Распознавание отвалилось — {clock}"
            if not self._warned_stt:
                self._warned_stt = True
                rumps.notification(
                    "church-translator", "Нет связи с распознаванием",
                    f"{error} — остановите (⏹) и запустите заново",
                )
            return
        self.status_item.title = f"● Работает — {clock}"

    # -- misc ------------------------------------------------------------------

    def open_logs(self, _sender) -> None:
        subprocess.run(["open", str(Path(self.config.logging.usage_log_path).parent)])

    def open_debug_audio(self, _sender) -> None:
        d = self.config.logging.debug_audio_dir
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", d])
        else:
            rumps.alert("logging.debug_audio_dir не задан в config.yaml")

    def open_recordings(self, _sender) -> None:
        d = self.config.logging.recordings_dir
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", d])
        else:
            rumps.alert("logging.recordings_dir не задан в config.yaml")

    def quit_app(self, _sender) -> None:
        if self.running:
            self._stop()
        rumps.quit_application()


def run() -> None:
    ChurchTranslatorApp().run()


if __name__ == "__main__":
    run()
