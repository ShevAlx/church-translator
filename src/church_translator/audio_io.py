"""Real-time duplex audio I/O — report §09 "The booth and the audio interface".

One input channel (the mixer's AUX/direct-out feed) goes in; each configured
language writes into its own dedicated output channel on the same interface.

The hard real-time constraint lives here: sounddevice's callback runs on
PortAudio's own thread and must never block. So the callback only moves
numpy arrays through lock-protected buffers — all STT/MT/TTS work happens on
separate worker threads (see pipeline.py) that are free to take the ~0.6-1.3s
this report budgeted for translation without ever stalling the audio device.

If a language's output buffer runs dry (worker still thinking, or crashed),
the callback pads with silence rather than blocking or repeating audio —
this is the "pass the signal through or output silence, never freeze" rule from §09.
"""

from __future__ import annotations

import queue
import threading
import time
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
    """Human-readable device list, for picking `audio.input_device` / `audio.output_device` / channel indices in config.yaml."""
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
        catchup_start_s: float = 2.0,
        max_playback_rate: float = 1.12,
        time_fn=time.monotonic,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.input_channel = input_channel
        self.output_channels = output_channels
        self.max_backlog_s = max_backlog_s
        self.catchup_start_s = catchup_start_s
        self.max_playback_rate = max_playback_rate
        self._catchup_frames = 0  # blocks spent compressing, for the post-service report
        # Injectable so lag behaviour can be simulated faster than real time —
        # a drift bug takes minutes of wall clock to show up otherwise, which is
        # exactly why the last two were found during a service instead of before.
        self._now = time_fn
        self._max_out_channel = max(output_channels) + 1

        # Consumed by the STT worker: raw mono float32 chunks from the mixer feed.
        self.input_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

        # Filled by each language's TTS worker; drained by the callback every block.
        self._out_lock = threading.Lock()
        # (utterance_id, samples, source_time). The id groups every chunk of one
        # spoken turn so the backlog cap skips whole thoughts instead of cutting
        # into one; source_time is when the speaker actually said it.
        self._out_buffers: dict[int, deque[tuple[int, np.ndarray, float]]] = {
            ch: deque() for ch in output_channels
        }

        self._stream: sd.Stream | None = None
        self._underrun_count = {ch: 0 for ch in output_channels}
        self._dropped_count = {ch: 0 for ch in output_channels}
        self._next_utterance_id = 0
        self._id_lock = threading.Lock()
        # The utterance each channel is in the middle of playing. The backlog cap
        # must never drop this one — see push_output.
        self._playing_uid: dict[int, int | None] = {ch: None for ch in output_channels}
        self._recorders: dict[int, ChannelRecorder] = {}
        self._input_recorder: ChannelRecorder | None = None

    def allocate_utterance_id(self) -> int:
        """Reserve an id before the first chunk of a streaming utterance exists.

        A streaming TTS provider delivers one sentence as many small chunks;
        they all have to carry the same id or the backlog cap treats each chunk
        as its own utterance and happily drops half a sentence.
        """
        with self._id_lock:
            uid = self._next_utterance_id
            self._next_utterance_id += 1
        return uid

    # -- one continuous recording per channel, for the whole session --------

    def start_recording(self, channel: int, path: Path) -> None:
        if channel not in self.output_channels:
            raise ValueError(f"channel {channel} is not one of the configured output_channels")
        self._recorders[channel] = ChannelRecorder(path, self.samplerate)

    def start_input_recording(self, path: Path) -> None:
        """Record the mixer feed itself, alongside the output channels.

        Both are written from the same callback invocation, so the input file
        and every output file share one timeline sample for sample. That is what
        makes delay *measurable* instead of estimated: open the two in any audio
        editor and read the offset between a phrase and its translation.

        It also turns any service into replay material — `church-translator
        replay --input <this file>` re-runs it through the whole pipeline
        offline, so a fix can be checked on real speech without a service.
        """
        self._input_recorder = ChannelRecorder(path, self.samplerate)

    # -- called from language worker threads --------------------------------

    def push_output(
        self,
        channel: int,
        samples: np.ndarray,
        utterance_id: int | None = None,
        source_time: float | None = None,
    ) -> None:
        """Queue synthesized audio for `channel`. Safe to call from any thread.

        `utterance_id` groups everything belonging to one spoken turn — every
        SSE chunk, and every sentence the turn was split into for translation.
        `source_time` is time.monotonic() when the speaker finished that turn.

        Staleness is decided at playback (see _callback), not here, and this is
        the whole point. Two earlier versions got it wrong:

        1. Drop the oldest whenever the queue exceeds the cap. The oldest is the
           sentence currently in the listener's ears, so every new sentence cut
           the current one off mid-word — 55% of the 2026-09-06 service lost.
        2. Keep the playing one and the newest, drop the queue in between. But
           streaming TTS delivers a whole turn's audio in ~3s of wall clock,
           so a 23s turn lands in a 10s buffer instantly and the middle sentence
           of a single thought was dropped while the listener was not behind at
           all — measured 40% loss on a 49s test with only 78% channel load.

        Queue depth simply is not lag: it counts audio that arrived early just
        the same as audio the listener is behind on. Real lag is wall-clock
        distance from when the words were spoken, which is what _callback uses.
        """
        if channel not in self._out_buffers:
            raise ValueError(f"channel {channel} is not one of the configured output_channels")
        with self._out_lock:
            if utterance_id is None:
                with self._id_lock:
                    uid = self._next_utterance_id
                    self._next_utterance_id += 1
            else:
                uid = utterance_id
            self._out_buffers[channel].append(
                (uid, samples.astype(np.float32, copy=False), source_time if source_time is not None else 0.0)
            )

    def queued_seconds(self, channel: int) -> float:
        """Synthesized audio already waiting to play on `channel` — how long the
        next sentence would sit in the queue before the listener hears it."""
        with self._out_lock:
            return sum(len(chunk) for _, chunk, _ in self._out_buffers[channel]) / self.samplerate

    # -- PortAudio callback (real-time thread — must not block) -------------

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # Overflow/underflow flags from PortAudio itself — surfaced, not swallowed.
            print(f"[audio] stream status: {status}")

        mic = indata[:, self.input_channel].copy()
        if self._input_recorder is not None:
            self._input_recorder.push(mic)  # same block as the outputs -> aligned timelines
        try:
            self.input_queue.put_nowait(mic)
        except queue.Full:
            pass  # STT worker is behind; drop this block rather than block the callback

        outdata[:] = 0.0
        now = self._now()
        with self._out_lock:
            for ch in self.output_channels:
                buf = self._out_buffers[ch]
                # How far behind the speaker the audio about to play actually is.
                lag = now - buf[0][2] if buf and buf[0][2] else 0.0
                rate = 1.0
                if lag > self.catchup_start_s and self.max_playback_rate > 1.0:
                    rate = min(self.max_playback_rate, 1.0 + (lag - self.catchup_start_s) * 0.05)
                need = int(round(frames * rate))

                taken = self._take(ch, buf, need, now)
                if len(taken) == 0:
                    self._underrun_count[ch] += 1  # silence padding — expected occasionally, not a crash
                elif len(taken) >= need and rate > 1.0:
                    # Linear resample: `need` samples of speech compressed into
                    # `frames` of output. Cheap enough for the real-time thread,
                    # and the pitch rise stays under ~2 semitones at the cap.
                    outdata[:, ch] = np.interp(
                        np.linspace(0.0, len(taken) - 1.0, frames), np.arange(len(taken)), taken
                    )
                    self._catchup_frames += 1
                else:
                    # Buffer ran dry mid-block: play what we have at normal speed
                    # rather than stretching a short read into a full block.
                    outdata[: len(taken), ch] = taken[:frames]
                    if len(taken) < frames:
                        self._underrun_count[ch] += 1
                if ch in self._recorders:
                    self._recorders[ch].push(outdata[:frames, ch].copy())

    def _take(self, ch: int, buf, need: int, now: float) -> np.ndarray:
        """Pull up to `need` samples, skipping whole turns that are already too
        stale to be worth playing. Called with `_out_lock` held.

        Staleness is judged only when stepping into a NEW turn: a turn already
        being spoken always finishes, because cutting into one is what made the
        translation lose its thread mid-thought.
        """
        out: list[np.ndarray] = []
        filled = 0
        while buf and filled < need:
            uid, chunk, source_time = buf[0]
            if uid != self._playing_uid[ch]:
                if source_time and now - source_time > self.max_backlog_s:
                    while buf and buf[0][0] == uid:
                        buf.popleft()
                    self._dropped_count[ch] += 1
                    continue  # skip this whole thought, try the next one
                self._playing_uid[ch] = uid
            take = min(len(chunk), need - filled)
            out.append(chunk[:take])
            if take == len(chunk):
                buf.popleft()
            else:
                buf[0] = (uid, chunk[take:], source_time)
            filled += take
        if not out:
            return np.zeros(0, dtype=np.float32)
        return out[0] if len(out) == 1 else np.concatenate(out)

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
        if self._input_recorder is not None:
            self._input_recorder.close()
            self._input_recorder = None

    def underrun_report(self) -> dict[int, int]:
        return dict(self._underrun_count)

    def dropped_report(self) -> dict[int, int]:
        """Whole turns skipped per channel to stay current (see _take)."""
        return dict(self._dropped_count)

    def catchup_report(self) -> float:
        """Seconds of output that were played compressed to claw back lag."""
        return self._catchup_frames * self.blocksize / self.samplerate
