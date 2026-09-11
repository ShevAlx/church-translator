"""One running service, shared by every front end: the terminal `run`, the
macOS menu bar, and the Telegram bot on the booth Pi.

Building a session (device lookup, providers, recordings, start order) used to
be copied line for line into cli.py and menubar_app.py. A third copy for the
bot would have been one too many: the Pi runs with nobody watching it, and a
fix that lands in two front ends but not the third is exactly the drift nobody
notices until a service.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio_io import AudioRouter, resolve_device
from .config import AppConfig
from .pipeline import Pipeline
from .providers.mock import MockMT, MockSTT, MockTTS
from .usage_log import UsageLogger, date_folder

# No callback from PortAudio for this long means the interface is gone (USB
# pulled, power lost) — at 512 frames / 48 kHz a healthy stream calls back ~94
# times a second, so 3s of nothing is never a hiccup.
AUDIO_STALL_S = 3.0


def build_pipeline(config: AppConfig, router: AudioRouter, usage_logger: UsageLogger, session_id: str) -> Pipeline:
    mode = config.pipeline.mode
    if mode == "passthrough":
        return Pipeline(config, router, usage_logger, session_id=session_id)
    if mode == "mock":
        stt = MockSTT(samplerate=config.audio.samplerate)
        mt_by_lang = {lang.code: MockMT() for lang in config.languages}
        tts_by_lang = {
            lang.code: MockTTS(samplerate=config.audio.samplerate, base_hz=440.0 * (i + 1))
            for i, lang in enumerate(config.languages)
        }
        return Pipeline(
            config, router, usage_logger, stt=stt, mt_by_language=mt_by_lang, tts_by_language=tts_by_lang,
            debug_audio_dir=config.logging.debug_audio_dir, session_id=session_id,
        )
    if mode != "real":
        raise ValueError(f"unknown pipeline.mode: {mode!r}")

    # Milestone 2 — needs `uv sync --extra providers` plus
    # ASSEMBLYAI_API_KEY / ANTHROPIC_API_KEY / CARTESIA_API_KEY (see README).
    # RuntimeError, not SystemExit: the bot catches this and reports it, and a
    # SystemExit would sail past `except Exception` and take the bot down.
    try:
        from .providers.assemblyai_stt import AssemblyAISTT
        from .providers.cartesia_tts import CartesiaTTS
        from .providers.claude_mt import ClaudeMT
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.mode is 'real' but the provider extras aren't installed. "
            f"Run: uv sync --extra providers (import error: {exc})"
        ) from exc

    # Constrain STT to what the speaker is expected to say (config.source_languages),
    # NOT to the output/target languages — those are what MT produces, never what
    # STT needs to recognize. Conflating the two breaks as soon as source and
    # target differ: live-tested 2026-08-19, AssemblyAI's streaming API flatly
    # rejects "uk" as a source candidate (code=3006) even though Cartesia
    # synthesizes Ukrainian output fine — source and target are different sets
    # with different constraints, on purpose.
    stt = AssemblyAISTT(
        samplerate=config.audio.samplerate,
        language_codes=config.source_languages or None,
        end_of_turn_confidence_threshold=config.stt.end_of_turn_confidence_threshold,
        min_turn_silence_ms=config.stt.min_turn_silence_ms,
        max_turn_silence_ms=config.stt.max_turn_silence_ms,
        partial_emit=config.stt.partial_emit,
        partial_min_words=config.stt.partial_min_words,
        partial_max_words=config.stt.partial_max_words,
        partial_gap_ms=config.stt.partial_gap_ms,
    )
    # The STT socket is open from here on, and AssemblyAI bills it by the
    # minute — anything below that fails must not leave it running.
    try:
        mt_by_lang = {lang.code: ClaudeMT() for lang in config.languages}
        tts_by_lang = {
            lang.code: CartesiaTTS(
                samplerate=config.audio.samplerate, model=config.tts.model, speed=config.tts.speed,
                continuous=config.tts.continuous_context,
            )
            for lang in config.languages
        }
        return Pipeline(
            config, router, usage_logger, stt=stt, mt_by_language=mt_by_lang, tts_by_language=tts_by_lang,
            debug_audio_dir=config.logging.debug_audio_dir, session_id=session_id,
        )
    except Exception:
        stt.close()
        raise


class SessionUsageLogger(UsageLogger):
    """UsageLogger that also keeps this session's totals and reports pipeline
    errors as they happen. The CSV stays the record; the counters are what a
    front end without a terminal (the bot) can show at ⏹."""

    def __init__(self, path: str | Path, on_error: Callable[[str, str], None] | None = None):
        super().__init__(path)
        self._on_error = on_error
        self._counts_lock = threading.Lock()
        self.tts_chars = 0
        self.tts_segments = 0
        self.errors = 0

    def log(self, session_id, language, component, duration_s=None, chars=None, note="") -> None:
        super().log(session_id, language, component, duration_s=duration_s, chars=chars, note=note)
        with self._counts_lock:
            if component == "tts" and chars:
                self.tts_chars += chars
                self.tts_segments += 1
            elif component == "error":
                self.errors += 1
        if component == "error" and self._on_error is not None:
            self._on_error(language, note)


@dataclass
class StopReport:
    session_id: str
    mode: str
    duration_s: float
    dropped: dict[str, int]    # language code -> whole turns skipped for lag
    underruns: dict[str, int]  # language code -> blocks padded with silence
    tts_chars: int
    tts_segments: int
    errors: int
    recordings_dir: Path | None


class LiveSession:
    """Router + pipeline for one service, from ▶️ to ⏹.

    Everything that can fail on a bad key or a dead line (the AssemblyAI
    handshake) happens in __init__, before any recording file or audio stream
    exists — so a failed start leaves nothing behind to clean up.
    """

    def __init__(self, config: AppConfig, on_error: Callable[[str, str], None] | None = None):
        self.config = config
        self.mode = config.pipeline.mode
        input_idx = resolve_device(config.audio.input_device, kind="input")
        output_idx = resolve_device(config.audio.output_device or config.audio.input_device, kind="output")
        self.router = AudioRouter(
            input_device=input_idx,
            output_device=output_idx,
            samplerate=config.audio.samplerate,
            blocksize=config.audio.blocksize,
            input_channel=config.audio.input_channel,
            output_channels=[lang.output_channel for lang in config.languages],
            max_backlog_s=config.audio.max_backlog_s,
            catchup_start_s=config.audio.catchup_start_s,
            max_playback_rate=config.audio.max_playback_rate,
        )
        self.usage = SessionUsageLogger(config.logging.usage_log_path, on_error=on_error)
        self.session_id = self.usage.new_session_id()  # shared with recordings, so filenames line up with usage.csv
        self.pipeline = build_pipeline(config, self.router, self.usage, self.session_id)
        self.recordings_dir: Path | None = None
        self.started_at: float | None = None

    def start(self) -> None:
        cfg = self.config
        try:
            if cfg.logging.recordings_dir:
                self.recordings_dir = Path(cfg.logging.recordings_dir) / date_folder(self.session_id)
                for lang in cfg.languages:
                    self.router.start_recording(
                        lang.output_channel, self.recordings_dir / f"{self.session_id}-{lang.code}.wav"
                    )
                # The English feed too, on the same timeline as the outputs: without it
                # the delay can only be estimated by ear, and the service cannot be
                # replayed offline afterwards.
                self.router.start_input_recording(self.recordings_dir / f"{self.session_id}-source.wav")
            self.router.start()
            self.pipeline.start()
        except Exception:
            self.pipeline.stop()  # closes the billed STT socket
            self.router.stop()
            raise
        self.started_at = time.monotonic()

    def stop(self) -> StopReport:
        self.pipeline.stop()
        self.router.stop()
        by_code = {lang.output_channel: lang.code for lang in self.config.languages}
        return StopReport(
            session_id=self.session_id,
            mode=self.mode,
            duration_s=self.elapsed_s,
            dropped={by_code[ch]: n for ch, n in self.router.dropped_report().items()},
            underruns={by_code[ch]: n for ch, n in self.router.underrun_report().items()},
            tts_chars=self.usage.tts_chars,
            tts_segments=self.usage.tts_segments,
            errors=self.usage.errors,
            recordings_dir=self.recordings_dir,
        )

    # -- health, for whoever is watching (menu bar tick, bot watchdog) ---------

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at if self.started_at is not None else 0.0

    @property
    def stt_error(self) -> str | None:
        return self.pipeline.stt_error

    @property
    def stt_lag_s(self) -> float | None:
        return self.pipeline.stt_lag_s

    @property
    def audio_alive(self) -> bool:
        since = self.router.seconds_since_callback()
        if since is None:  # stream opened but not a single callback yet
            return self.elapsed_s < AUDIO_STALL_S
        return since < AUDIO_STALL_S

    def seconds_without_signal(self) -> float:
        """How long the mixer feed has been silent — since ▶️ if it never spoke."""
        since = self.router.seconds_since_signal()
        return self.elapsed_s if since is None else min(since, self.elapsed_s)
