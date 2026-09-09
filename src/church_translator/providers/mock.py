"""Milestone-1 providers: no API keys, no network. These exercise the STT ->
MT -> TTS *shape* of the pipeline (threading, timing, usage logging) without
calling anything real. They are deliberately not clever:

- MockSTT fires one placeholder transcript every `chunk_seconds` of audio.
- MockMT is the identity function.
- MockTTS turns each transcript into a short sine-wave beep, at a pitch
  fixed per language, so on a real speaker you can hear each configured
  language's channel as a distinct tone and confirm the routing is correct.

For testing raw audio routing itself (no STT/MT/TTS involved at all), use
pipeline.mode="passthrough" instead — see pipeline.py.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from .base import MTProvider, STTProvider, TranscriptEvent, TTSProvider


class MockSTT(STTProvider):
    def __init__(self, samplerate: int, chunk_seconds: float = 1.5):
        self._samples_seen = 0
        self._threshold = int(samplerate * chunk_seconds)

    def feed(self, pcm_chunk: np.ndarray) -> TranscriptEvent | None:
        self._samples_seen += len(pcm_chunk)
        if self._samples_seen < self._threshold:
            return None
        self._samples_seen = 0
        return TranscriptEvent(text="mock transcript", is_final=True, language_code="auto",
                               received_at=time.monotonic())

    def close(self) -> None:
        pass


class MockMT(MTProvider):
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        return text


class MockTTS(TTSProvider):
    """One short beep per call. Pitch is derived from `language_code` so
    different languages are audibly distinct in a smoke test."""

    def __init__(self, samplerate: int, duration_s: float = 0.35, base_hz: float | None = None):
        self.samplerate = samplerate
        self.duration_s = duration_s
        # The caller assigns one distinct pitch per language so a two-channel
        # smoke test is unambiguous. Deriving it here from the language code
        # used to go through hash(), which is salted per process: the tones
        # changed on every launch and two languages could collide on the same
        # pitch (5 buckets), making a working stereo pair sound like one
        # channel. Left as a fallback for direct construction in tests.
        self.base_hz = base_hz

    def synthesize(self, text: str, voice_id: str, language_code: str) -> np.ndarray:
        base_hz = self.base_hz
        if base_hz is None:
            base_hz = 220.0 + (zlib.crc32(language_code.encode()) % 5) * 110.0
        t = np.linspace(0, self.duration_s, int(self.samplerate * self.duration_s), endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * base_hz * t)
        fade = min(200, len(tone) // 4)
        if fade > 0:
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        return tone.astype(np.float32)
