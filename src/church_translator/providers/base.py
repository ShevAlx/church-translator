"""Provider interfaces — the seam between the audio plumbing and whichever
STT/MT/TTS vendors actually do the work (report §04: AssemblyAI/Gladia for
STT, an LLM for MT, Cartesia/ElevenLabs for TTS).

Milestone 1 ships one implementation of each: the mock in providers/mock.py.
Milestone 2 wires in the real ones (providers/assemblyai_stt.py etc.) behind
these exact same interfaces, so pipeline.py does not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    language_code: str  # what the STT engine detected this segment as (report §04.2 code-switching)
    # time.monotonic() when the speaker finished saying this. The output buffer
    # needs it to tell "the listener is genuinely behind" from "the pipeline
    # delivered this early" — those look identical if you only measure how much
    # audio is queued. See AudioRouter.push_output.
    received_at: float = 0.0
    # Diagnostics, logged to usage.csv. On 2026-09-11 recognition fell 90s
    # behind the speaker while every other stage stayed fast, and nothing on
    # screen or in the log could show it — the delay was only found by
    # aligning recordings by hand afterwards.
    #   stt_lag_s:     seconds between the speaker saying the last word of this
    #                  text and the text reaching us (upload + server together)
    #   stt_backlog_s: audio still sitting in the client's send queue
    stt_lag_s: float | None = None
    stt_backlog_s: float | None = None
    # Speaker's pace over this text (words per second of speech), from the STT
    # word timings. Drives the synthesis speed — see pipeline.choose_speed.
    words_per_s: float | None = None


class STTProvider(ABC):
    """Streaming speech-to-text with language auto-detection."""

    @abstractmethod
    def feed(self, pcm_chunk: np.ndarray) -> TranscriptEvent | None:
        """Push a chunk of mono float32 audio; returns a transcript event when one is ready, else None."""

    @abstractmethod
    def close(self) -> None: ...


class MTProvider(ABC):
    """Text translation, source language auto-detected upstream by STT."""

    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str: ...


class TTSProvider(ABC):
    """Streaming speech synthesis with a voice locked for the whole session (report §04.1 fix)."""

    @abstractmethod
    def synthesize(
        self, text: str, voice_id: str, language_code: str, speed: float | None = None
    ) -> np.ndarray:
        """Returns mono float32 PCM at the pipeline's samplerate.

        `speed` is a delivery-rate multiplier (1.0 = the voice's natural pace);
        None, or a provider that cannot vary it, means natural pace."""

    def synthesize_stream(
        self, text: str, voice_id: str, language_code: str, speed: float | None = None
    ) -> Iterator[np.ndarray]:
        """Optional: yield PCM as it is generated, so playback starts before the
        whole utterance exists. Measured 2026-08-23 against Cartesia: 0.19s to
        first audio streaming vs 1.07s waiting for the finished file, and the
        gap widens with sentence length. Default falls back to one-shot
        synthesize(), so a provider that cannot stream needs no changes.
        """
        audio = self.synthesize(text, voice_id, language_code, speed=speed)
        if len(audio):
            yield audio

    def close(self) -> None:
        """Release any long-lived connection. Called when the service stops."""
