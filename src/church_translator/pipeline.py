"""Wires audio_io <-> providers into the shape from report §05: one STT stage
reads the mixer feed once, and fans final transcripts out to one MT+TTS
worker per configured language. Cost and thread count scale with the number
of languages, never with how many receivers are tuned to a channel.

Three run modes (config: pipeline.mode):
  passthrough  audio routed straight to every output channel, no STT/MT/TTS.
               Proves device I/O, channel mapping, threading, silence-on-
               underrun — the part that has nothing to do with translation.
  mock         real STT->MT->TTS thread shape, fake providers (a beep per
               detected utterance). Proves the pipeline's own plumbing.
  real         Milestone 2: real provider instances passed in by the caller.
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .audio_io import AudioRouter
from .config import AppConfig, LanguageConfig
from .providers.base import MTProvider, STTProvider, TranscriptEvent, TTSProvider
from .usage_log import UsageLogger, date_folder


def _write_debug_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    """16-bit mono WAV — plain enough that QuickTime/afplay/anything opens it,
    so you can hear exactly what would have gone to the output channel
    without needing a transmitter/receiver hooked up yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(pcm16.tobytes())


class _STTStage(threading.Thread):
    """Runs once per service, regardless of language count (report §05 fan-out point)."""

    def __init__(
        self,
        router: AudioRouter,
        stt: STTProvider,
        fanout: list["queue.Queue[TranscriptEvent]"],
        usage_logger: UsageLogger,
        session_id: str,
        stop_event: threading.Event,
    ):
        super().__init__(name="stt-stage", daemon=True)
        self._router = router
        self._stt = stt
        self._fanout = fanout
        self._usage_logger = usage_logger
        self._session_id = session_id
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._router.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            event = self._stt.feed(chunk)
            if event is None or not event.is_final:
                continue
            self._usage_logger.log(
                self._session_id, event.language_code, "stt",
                duration_s=time.monotonic() - t0, chars=len(event.text),
            )
            for q in self._fanout:
                q.put(event)


class _LanguageStage(threading.Thread):
    """One per configured language: MT then TTS, writing into that language's output channel."""

    def __init__(
        self,
        lang: LanguageConfig,
        event_queue: "queue.Queue[TranscriptEvent]",
        mt: MTProvider,
        tts: TTSProvider,
        router: AudioRouter,
        usage_logger: UsageLogger,
        session_id: str,
        stop_event: threading.Event,
        debug_audio_dir: str | None = None,
    ):
        super().__init__(name=f"lang-{lang.code}", daemon=True)
        self._lang = lang
        self._queue = event_queue
        self._mt = mt
        self._tts = tts
        self._router = router
        self._usage_logger = usage_logger
        self._session_id = session_id
        self._stop_event = stop_event
        self._debug_audio_dir = Path(debug_audio_dir) / date_folder(session_id) if debug_audio_dir else None
        self._debug_counter = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t0 = time.monotonic()
                translated = self._mt.translate(event.text, event.language_code, self._lang.code)
                self._usage_logger.log(
                    self._session_id, self._lang.code, "mt",
                    duration_s=time.monotonic() - t0, chars=len(translated),
                )

                t0 = time.monotonic()
                audio = self._tts.synthesize(translated, self._lang.voice_id, self._lang.code)
                self._usage_logger.log(
                    self._session_id, self._lang.code, "tts",
                    duration_s=time.monotonic() - t0, chars=len(translated),
                )

                self._router.push_output(self._lang.output_channel, audio)

                if self._debug_audio_dir is not None and len(audio):
                    self._debug_counter += 1
                    wav_path = self._debug_audio_dir / (
                        f"{self._session_id}-{self._lang.code}-{self._debug_counter:03d}.wav"
                    )
                    _write_debug_wav(wav_path, audio, self._router.samplerate)
                    print(f"[{self._lang.code}] wrote {wav_path} ({translated!r})")
            except Exception as exc:  # noqa: BLE001 — report §09: silence, never a hang
                self._usage_logger.log(
                    self._session_id, self._lang.code, "error", note=str(exc)[:200],
                )
                print(f"[{self._lang.code}] pipeline error, dropping this utterance: {exc}")


class _PassthroughStage(threading.Thread):
    """No STT/MT/TTS at all — the raw-I/O smoke test."""

    def __init__(self, router: AudioRouter, languages: list[LanguageConfig], stop_event: threading.Event):
        super().__init__(name="passthrough", daemon=True)
        self._router = router
        self._languages = languages
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._router.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            for lang in self._languages:
                self._router.push_output(lang.output_channel, chunk)


class Pipeline:
    """Owns every worker thread for one running service."""

    def __init__(
        self,
        config: AppConfig,
        router: AudioRouter,
        usage_logger: UsageLogger,
        stt: STTProvider | None = None,
        mt_by_language: dict[str, MTProvider] | None = None,
        tts_by_language: dict[str, TTSProvider] | None = None,
        debug_audio_dir: str | None = None,
        session_id: str | None = None,
    ):
        self._config = config
        self._router = router
        self._usage_logger = usage_logger
        self._session_id = session_id or usage_logger.new_session_id()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stt = stt

        if config.pipeline.mode == "passthrough":
            self._threads.append(_PassthroughStage(router, config.languages, self._stop_event))
            return

        if stt is None:
            raise ValueError("pipeline.mode is not 'passthrough' — an STT provider instance is required")

        fanout_queues: list[queue.Queue[TranscriptEvent]] = []
        for lang in config.languages:
            q: "queue.Queue[TranscriptEvent]" = queue.Queue(maxsize=50)
            fanout_queues.append(q)
            mt = (mt_by_language or {}).get(lang.code)
            tts = (tts_by_language or {}).get(lang.code)
            if mt is None or tts is None:
                raise ValueError(f"missing mt/tts provider for language {lang.code!r}")
            self._threads.append(
                _LanguageStage(
                    lang, q, mt, tts, router, usage_logger, self._session_id, self._stop_event,
                    debug_audio_dir=debug_audio_dir,
                )
            )

        self._threads.insert(
            0, _STTStage(router, stt, fanout_queues, usage_logger, self._session_id, self._stop_event)
        )

    @property
    def stt_error(self) -> str | None:
        """Non-None once the STT stream is dead (connection dropped mid-service).

        The pipeline keeps running — audio still flows, the language threads are
        still alive — but no transcript will ever arrive again, so a caller with
        a UI (menubar_app) must say so instead of showing a green "running".
        Providers without the attribute (the mock) simply never report an error.
        """
        return getattr(self._stt, "fatal_error", None)

    def start(self) -> None:
        for t in self._threads:
            t.start()
        print(f"[pipeline] session {self._session_id} started, mode={self._config.pipeline.mode}, "
              f"{len(self._config.languages)} language(s)")

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        # AssemblyAI bills by websocket connection time, not by how much audio you
        # sent through it, so leaving the stream open after ⏹ keeps the meter
        # running for as long as the app sits in the menu bar. STTProvider.close()
        # has been part of the interface since Milestone 1 and nothing ever called
        # it (found 2026-08-23) — a stopped service was still being charged.
        if self._stt is not None:
            try:
                self._stt.close()
            except Exception as exc:  # noqa: BLE001 — shutdown must never raise
                print(f"[pipeline] STT close failed (соединение всё равно оборвётся): {exc}")
