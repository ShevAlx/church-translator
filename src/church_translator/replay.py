"""Offline replay: run a WAV file through the real pipeline, no sound card.

Why this exists. Every bug this project has had in the output path was found
*during a service* — the mid-word truncation (2026-08-23), then the dropped
middle of a thought (2026-09-06) — because the only way to hear the result was
to hold a service. That is an expensive test loop and a bad one: the booth gets
one take, on a Sunday.

This drives the exact production classes (AudioRouter._callback included, drop
logic and all) from a file at wall-clock pace, and writes what each language
channel would have played. Real STT, real MT, real TTS; the only thing not
exercised is the physical Scarlett output.

Pace matters: the callback is stepped in real time, not as fast as possible,
because AssemblyAI's endpointing and the output buffer's staleness rule are
both wall-clock quantities. Replaying at 10x would report latencies that no
service will ever reproduce.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from .audio_io import AudioRouter
from .config import AppConfig


class OfflineRouter(AudioRouter):
    """AudioRouter with the device replaced by a file in and a file out."""

    def start(self) -> None:  # no PortAudio stream to open
        pass

    def stop(self) -> None:
        for rec in self._recorders.values():
            rec.close()
        self._recorders.clear()
        if self._input_recorder is not None:
            self._input_recorder.close()
            self._input_recorder = None


def read_wav_mono(path: Path, samplerate: int) -> np.ndarray:
    """16-bit PCM at the pipeline's rate, mixed down to mono."""
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != samplerate:
            raise SystemExit(
                f"{path}: {f.getframerate()}Hz, but the pipeline runs at {samplerate}Hz — "
                f"resample first:  ffmpeg -i {path.name} -ar {samplerate} -ac 1 -c:a pcm_s16le out.wav"
            )
        if f.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM (ffmpeg -c:a pcm_s16le)")
        channels = f.getnchannels()
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    mono = raw if channels == 1 else raw.reshape(-1, channels).mean(axis=1)
    return mono.astype(np.float32) / 32768.0


def _write_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(pcm16.tobytes())


def replay(
    config: AppConfig,
    router: OfflineRouter,
    source: np.ndarray,
    out_dir: Path,
    session_id: str,
    drain_s: float = 45.0,
) -> dict:
    """Step the real callback at wall-clock pace over `source`, then let the
    tail of the translation drain, and write one WAV per language channel."""
    sr = config.audio.samplerate
    blk = config.audio.blocksize
    n_out = max(lang.output_channel for lang in config.languages) + 1
    in_ch = config.audio.input_channel

    captured: dict[int, list[np.ndarray]] = {lang.output_channel: [] for lang in config.languages}
    indata = np.zeros((blk, in_ch + 1), dtype=np.float32)
    outdata = np.zeros((blk, n_out), dtype=np.float32)

    total_blocks = int((len(source) + drain_s * sr) // blk)
    start = time.monotonic()
    for i in range(total_blocks):
        lo = i * blk
        frame = source[lo : lo + blk]
        indata[:] = 0.0
        if len(frame):
            indata[: len(frame), in_ch] = frame  # silence once the file runs out
        outdata[:] = 0.0
        router._callback(indata, outdata, blk, None, None)
        for ch in captured:
            captured[ch].append(outdata[:, ch].copy())
        # Hold wall-clock pace — the whole point of the harness.
        target = start + (i + 1) * blk / sr
        while (delay := target - time.monotonic()) > 0:
            time.sleep(min(delay, 0.005))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for lang in config.languages:
        track = np.concatenate(captured[lang.output_channel])
        path = out_dir / f"{session_id}-{lang.code}.wav"
        _write_wav(path, track, sr)
        voiced = float((np.abs(track) > 1e-4).sum()) / sr
        written[lang.code] = {"path": path, "length_s": len(track) / sr, "voiced_s": voiced}
    return written
