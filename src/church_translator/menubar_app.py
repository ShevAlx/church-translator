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
from pathlib import Path

import rumps
import sounddevice as sd

from .config import AppConfig, load_config, save_config
from .session import LiveSession

CONFIG_PATH = Path("config.yaml")
DEFAULT_TEMPLATE_PATH = Path("config.church.example.yaml")


def _save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    save_config(config, path)


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
        self.session: LiveSession | None = None
        self._warned_stt = False

        self.status_item = rumps.MenuItem("○ Stopped")
        self.toggle_item = rumps.MenuItem("▶️ Start translation", callback=self.toggle_run)
        self.mode_item = rumps.MenuItem(
            f"Test mode (mock): {'on' if self.config.pipeline.mode == 'mock' else 'off'}",
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
            rumps.MenuItem("Input device", callback=None),
            *self._device_items(kind="input"),
            None,
            rumps.MenuItem("Output device", callback=None),
            *self._device_items(kind="output"),
            None,
            self._input_channel_menu(),
            self._languages_menu(),
            None,
            rumps.MenuItem("Open logs", callback=self.open_logs),
            rumps.MenuItem("Open debug audio", callback=self.open_debug_audio),
            rumps.MenuItem("Open service recordings", callback=self.open_recordings),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
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
        parent = rumps.MenuItem("Input channel")
        n_channels = self._channel_count("input", self.config.audio.input_device) or 8
        for ch in range(n_channels):
            item = rumps.MenuItem(f"Channel {ch}", callback=lambda sender, c=ch: self._set_input_channel(c))
            item.state = ch == self.config.audio.input_channel
            parent.add(item)
        return parent

    def _languages_menu(self) -> rumps.MenuItem:
        parent = rumps.MenuItem("Language channels")
        n_channels = self._channel_count("output", self.config.audio.output_device) or 8
        for lang in self.config.languages:
            lang_item = rumps.MenuItem(f"{lang.name} ({lang.code})")
            for ch in range(n_channels):
                item = rumps.MenuItem(
                    f"Channel {ch}", callback=lambda sender, c=ch, code=lang.code: self._set_output_channel(code, c)
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
        rumps.notification("church-translator", "Device changed", name)

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
            rumps.alert("Stop translation first, then change the mode.")
            return
        self.config.pipeline.mode = "real" if self.config.pipeline.mode == "mock" else "mock"
        sender.state = self.config.pipeline.mode == "mock"
        sender.title = f"Test mode (mock): {'on' if self.config.pipeline.mode == 'mock' else 'off'}"
        _save_config(self.config)

    # -- start/stop ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.session is not None

    def toggle_run(self, sender) -> None:
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        try:
            session = LiveSession(self.config)
            session.start()
        except Exception as exc:  # noqa: BLE001 — surface to the operator, don't just crash the menu bar
            rumps.alert(f"Could not start: {exc}")
            return
        self.session = session
        self._warned_stt = False
        self.toggle_item.title = "⏹ Stop translation"
        self.title = "🔴"
        self._tick(None)

    def _stop(self) -> None:
        if self.session is not None:
            report = self.session.stop()
            underruns, dropped = report.underruns, report.dropped
        else:
            underruns, dropped = {}, {}
        self.session = None
        self.toggle_item.title = "▶️ Start translation"
        self.title = "🎙️"
        self.status_item.title = "○ Stopped"
        if any(dropped.values()):
            # Shown because it is the one number that says whether listeners
            # actually heard the service or a skipped version of it.
            rumps.notification(
                "church-translator", "Stopped",
                f"Turns skipped (lag): {dropped}. If that is many, raise max_backlog_s.",
            )
        elif any(underruns.values()):
            rumps.notification("church-translator", "Stopped", f"Underruns per channel: {underruns}")

    def _tick(self, _sender) -> None:
        if self.session is None:
            return
        elapsed = int(self.session.elapsed_s)
        clock = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # A dead STT stream leaves everything else looking healthy — audio still
        # flows, every thread is still alive, the timer still counts up. Without
        # this the booth sees "● Running" while the headphones are silent
        # (happened for real during the 2026-08-23 pre-service check).
        error = self.session.stt_error
        if error:
            self.title = "⚠️"
            self.status_item.title = f"⚠️ Recognition lost — {clock}"
            if not self._warned_stt:
                self._warned_stt = True
                rumps.notification(
                    "church-translator", "Lost connection to recognition",
                    f"{error} — stop (⏹) and start again",
                )
            return
        # Past max_backlog_s the output buffer is already skipping whole turns,
        # so this is the point where the listener is actually losing sermon.
        lag = self.session.stt_lag_s
        if lag is not None and lag > self.config.audio.max_backlog_s:
            self.title = "🐢"
            self.status_item.title = f"🐢 Lagging {lag:.0f}s — slow internet — {clock}"
            return
        self.title = "🔴"
        self.status_item.title = f"● Running — {clock}"

    # -- misc ------------------------------------------------------------------

    def open_logs(self, _sender) -> None:
        subprocess.run(["open", str(Path(self.config.logging.usage_log_path).parent)])

    def open_debug_audio(self, _sender) -> None:
        d = self.config.logging.debug_audio_dir
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", d])
        else:
            rumps.alert("logging.debug_audio_dir is not set in config.yaml")

    def open_recordings(self, _sender) -> None:
        d = self.config.logging.recordings_dir
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)
            subprocess.run(["open", d])
        else:
            rumps.alert("logging.recordings_dir is not set in config.yaml")

    def quit_app(self, _sender) -> None:
        if self.running:
            self._stop()
        rumps.quit_application()


def run() -> None:
    ChurchTranslatorApp().run()


if __name__ == "__main__":
    run()
