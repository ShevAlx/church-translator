"""Real-time duplex audio I/O — report §09 "Будка и звуковая карта".

One input channel (the mixer's AUX/direct-out feed) goes in; each configured
language writes into its own dedicated output channel on the same interface.

The hard real-time constraint lives here: sounddevice's callback runs on
PortAudio's own thread and must never block. So the callback only moves
numpy arrays through lock-protected buffers — all STT/MT/TTS work happens on
separate worker threads (see pipeline.py) that are free to take the ~0.6-1.3s
this report budgeted for translation without ever stalling the audio device.

If a language's output buffer runs dry (worker still thinking, or crashed),
the callback pads with silence rather than blocking or repeating audio —
this is the "проходной сигнал или тишина, не зависание" rule from §09.
"""

from __future__ import annotations

import queue
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd


class ChannelRecorder:
    """Continuously writes one output channel to a single growing WAV file —
    gaps and all, exactly what would have been heard live, not a splice of
    utterances back to back. Disk writes happen on their own thread; `push()`
    (called from the real-time audio callback) only ever queues, so a slow
    disk can never cause an audio glitch."""

    def __init__(self, path: Path, samplerate: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(samplerate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, block: np.ndarray) -> None:
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm16 = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16)
            self._wav.writeframes(pcm16.tobytes())

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._wav.close()


def list_devices() -> str:
    """Human-readable device list, for picking `audio.device` / channel indices in config.yaml."""
    lines = []
    for idx, dev in enumerate(sd.query_devices()):
        lines.append(
            f"[{idx}] {dev['name']!r}  in={dev['max_input_channels']} "
            f"out={dev['max_output_channels']}  default_sr={dev['default_samplerate']:.0f}"
        )
    return "\n".join(lines)


def resolve_device(name_substring: str | None, kind: str = "input") -> int | None:
    """Find a device index by case-insensitive substring match on its name,
    restricted to devices that actually have channels on the requested side.

    Needed because some USB interfaces (consumer soundbars especially) show
    up as two separate CoreAudio devices sharing a name — one input-only,
    one output-only — even though physically they're one box. A plain
    name match with no channel filter can silently pick the wrong half.
    """
    if not name_substring:
        return None
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if name_substring.lower() in dev["name"].lower() and dev[channel_key] > 0:
            return idx
    raise ValueError(
        f"no {kind} device matching {name_substring!r} with {kind} channels — "
        "try `church-translator list-devices`"
    )


class AudioRouter:
    """Owns the duplex stream. One input channel in, N output channels out."""

    def __init__(
        self,
        input_device: int | None,
        output_device: int | None,
        samplerate: int,
        blocksize: int,
        input_channel: int,
        output_channels: list[int],
        max_backlog_s: float = 4.0,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.input_channel = input_channel
        self.output_channels = output_channels
        self.max_backlog_s = max_backlog_s
        self._max_out_channel = max(output_channels) + 1

        # Consumed by the STT worker: raw mono float32 chunks from the mixer feed.
        self.input_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

        # Filled by each language's TTS worker; drained by the callback every block.
        self._out_lock = threading.Lock()
        # (utterance_id, samples). The id groups the chunks of one utterance so the
        # backlog cap can drop whole utterances instead of decapitating the current one.
        self._out_buffers: dict[int, deque[tuple[int, np.ndarray]]] = {ch: deque() for ch in output_channels}

        self._stream: sd.Stream | None = None
        self._underrun_count = {ch: 0 for ch in output_channels}
        self._dropped_count = {ch: 0 for ch in output_channels}
        self._next_utterance_id = 0
        self._recorders: dict[int, ChannelRecorder] = {}

    # -- one continuous recording per channel, for the whole session --------

    def start_recording(self, channel: int, path: Path) -> None:
        if channel not in self.output_channels:
            raise ValueError(f"channel {channel} is not one of the configured output_channels")
        self._recorders[channel] = ChannelRecorder(path, self.samplerate)

    # -- called from language worker threads --------------------------------

    def push_output(self, channel: int, samples: np.ndarray, utterance_id: int | None = None) -> None:
        """Queue synthesized audio for `channel`. Safe to call from any thread.

        The queue is capped at `max_backlog_s`. This is not a memory guard — it
        is what keeps live interpretation *live*. Synthesized speech routinely
        runs longer than the original (measured 2026-08-23 on a real service:
        107% channel load — 173s of Russian audio for 162s of speech), so an
        unbounded queue drifts further behind the speaker every minute: +11s of
        lag in the first 2.5 minutes, and it never recovers on its own. Past the
        cap the OLDEST whole utterances are dropped, never the newest — a
        listener a few seconds behind and current is useful; a listener a minute
        behind is translating the previous paragraph.
        """
        if channel not in self._out_buffers:
            raise ValueError(f"channel {channel} is not one of the configured output_channels")
        limit = int(self.max_backlog_s * self.samplerate)
        with self._out_lock:
            buf = self._out_buffers[channel]
            uid = self._next_utterance_id if utterance_id is None else utterance_id
            if utterance_id is None:
                self._next_utterance_id += 1
            buf.append((uid, samples.astype(np.float32, copy=False)))
            queued = sum(len(chunk) for _, chunk in buf)
            # Drop whole utterances, oldest first, and never the one being pushed:
            # a streaming provider delivers one utterance as many small chunks, and
            # trimming those individually would cut off the front of a sentence
            # mid-word instead of skipping a stale sentence outright.
            while queued > limit and buf[0][0] != uid:
                stale = buf[0][0]
                while buf and buf[0][0] == stale:
                    queued -= len(buf.popleft()[1])
                self._dropped_count[channel] += 1

    # -- PortAudio callback (real-time thread — must not block) -------------

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # Overflow/underflow flags from PortAudio itself — surfaced, not swallowed.
            print(f"[audio] stream status: {status}")

        try:
            self.input_queue.put_nowait(indata[:, self.input_channel].copy())
        except queue.Full:
            pass  # STT worker is behind; drop this block rather than block the callback

        outdata[:] = 0.0
        with self._out_lock:
            for ch in self.output_channels:
                buf = self._out_buffers[ch]
                filled = 0
                while buf and filled < frames:
                    uid, chunk = buf[0]
                    take = min(len(chunk), frames - filled)
                    outdata[filled : filled + take, ch] = chunk[:take]
                    if take == len(chunk):
                        buf.popleft()
                    else:
                        buf[0] = (uid, chunk[take:])
                    filled += take
                if filled < frames:
                    self._underrun_count[ch] += 1  # silence padding — expected occasionally, not a crash
                if ch in self._recorders:
                    self._recorders[ch].push(outdata[:frames, ch].copy())

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._stream = sd.Stream(
            device=(self.input_device, self.output_device),  # None = PortAudio's default for that side
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=(self.input_channel + 1, self._max_out_channel),
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        for rec in self._recorders.values():
            rec.close()
        self._recorders.clear()

    def underrun_report(self) -> dict[int, int]:
        return dict(self._underrun_count)

    def dropped_report(self) -> dict[int, int]:
        """Utterances skipped per channel to stay current (see push_output)."""
        return dict(self._dropped_count)
