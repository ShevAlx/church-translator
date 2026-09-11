"""STT provider backed by AssemblyAI's streaming API (report §04.2: native
code-switching, sub-300ms latency). Requires ASSEMBLYAI_API_KEY.

Verified against the installed SDK (assemblyai==1.0.0, `assemblyai.streaming.v3`)
by introspecting its actual classes rather than guessing — StreamingClient
takes a StreamingClientOptions, `.connect()` takes StreamingParameters,
`.on(event, handler)` registers a `handler(client, event)` callback, and
`language_detection=True` on StreamingParameters is what turns on the
code-switching behaviour report §04.2 asked for (there's also a narrower
`language_codes=[...]` if you'd rather pin a candidate set than fully
auto-detect). If a future SDK version renames these, `feed()`'s contract
(`STTProvider` in base.py) is what pipeline.py depends on — keep that.

Live-tested 2026-08-20 on the real Scarlett 2i2, config.audio.blocksize=512
@ 48kHz: `audio_io.py`'s callback hands `feed()` one 512-sample chunk at a
time — 10.67ms. AssemblyAI's streaming API rejects anything under 50ms
outright (code=3007, "Input Duration Violation"), which killed every packet
before any transcript could come back. Fixed by buffering internally and
only calling `.stream()` once enough audio has accumulated — the audio
callback's chunk size and AssemblyAI's minimum packet size are unrelated
constraints and one must not assume they match.

Live-tested 2026-08-23, pre-service check: a transient TLS/routing failure
made the handshake fail outright. `StreamingClient.connect()` does NOT raise
on that — read its docstring: an HTTP-level rejection (bad key, quota) or an
exhausted retry chain is *dispatched to the Error handler and swallowed*,
and connect() returns normally. So the old code built a dead client, the
menu-bar app lit up "● Running", and the booth heard silence for a whole
service with nothing on screen saying why. Now the constructor waits for the
Begin frame that proves a live session and raises if it never arrives, and a
mid-session drop is recorded in `fatal_error` so the app can surface it.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

from .base import STTProvider, TranscriptEvent


STT_SAMPLERATE = 16000

# How much unsent audio may pile up in the SDK before the oldest is thrown
# away. The SDK's send queue is unbounded: on 2026-09-11 recognition fell 90s
# behind during a live test and never came back, while the same recording
# replayed offline stayed under 2s — i.e. the line, not the pipeline. For live
# interpretation a sentence 90s late is worth nothing, so losing a few seconds
# of speech to stay current beats translating the past for the rest of the
# service. Mirrors audio.max_backlog_s on the output side.
MAX_SEND_BACKLOG_S = 3.0


class _Decimator:
    """Capture rate (48 kHz) -> 16 kHz before upload.

    AssemblyAI's models run at 16 kHz, so the extra samples cost upload and buy
    no accuracy: this cuts the stream from 768 to 256 kbit/s on the same line
    that stalled. Windowed-sinc low-pass first so nothing above the new Nyquist
    folds back into the speech band; filter state carries across chunks.
    """

    def __init__(self, in_rate: int, out_rate: int = STT_SAMPLERATE, taps: int = 63):
        self.factor = in_rate // out_rate if in_rate > out_rate and in_rate % out_rate == 0 else 1
        self.out_rate = in_rate // self.factor
        self._fir: np.ndarray | None = None
        if self.factor == 1:
            return  # not an integer ratio: send the capture rate untouched
        n = np.arange(taps) - (taps - 1) / 2
        cutoff = 0.45 / self.factor  # cycles/sample, just under the new Nyquist
        fir = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        self._fir = (fir / fir.sum()).astype(np.float32)
        self._hist = np.zeros(taps - 1, dtype=np.float32)
        self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._fir is None:
            return x
        buf = np.concatenate([self._hist, x])
        y = np.convolve(buf, self._fir, mode="valid")  # one output per input sample
        self._hist = buf[-(len(self._fir) - 1) :]
        out = y[self._phase :: self.factor]
        self._phase = (self._phase - len(y)) % self.factor
        return out


class AssemblyAISTT(STTProvider):
    def __init__(
        self,
        samplerate: int,
        api_key: str | None = None,
        language_codes: list[str] | None = None,
        send_chunk_ms: float = 100.0,  # inside AssemblyAI's required 50-1000ms window, with margin
        connect_timeout_s: float = 15.0,
        end_of_turn_confidence_threshold: float | None = None,
        min_turn_silence_ms: int | None = None,
        max_turn_silence_ms: int | None = None,
        partial_emit: bool = True,
        partial_min_words: int = 8,
        partial_max_words: int = 25,
        partial_gap_ms: int = 250,
    ):
        import assemblyai.streaming.v3 as s3

        self.samplerate = samplerate
        # Set when the stream dies — read by the caller (menubar_app) to tell the
        # operator the channel is dead instead of leaving a green "running" status.
        self.fatal_error: str | None = None
        self._events: "queue.Queue[TranscriptEvent]" = queue.Queue()
        self._decimator = _Decimator(samplerate)
        # A whole number of decimation steps per packet, so no packet boundary
        # ever falls between the samples one output sample is built from.
        k = self._decimator.factor
        self._send_threshold = max(k, int(samplerate * send_chunk_ms / 1000) // k * k)
        self._send_chunk_s = self._send_threshold / samplerate
        self._t_first_feed: float | None = None  # stream time zero for word timings
        self.dropped_audio_s = 0.0  # thrown away by the backlog cap, whole session
        self.last_lag_s: float | None = None  # read by Pipeline.stt_lag_s -> menu bar
        self._send_buffer = np.zeros(0, dtype=np.float32)
        self._handshake = threading.Event()
        self._closing = False
        self._partial_emit = partial_emit
        self._partial_min_words = partial_min_words
        self._partial_max_words = partial_max_words
        self._partial_gap_ms = partial_gap_ms
        self._forced_turn: int | None = None  # turn we already sent ForceEndpoint for

        # The SDK's default open_timeout is 1s with 2 retries. Measured 2026-09-11
        # on the booth's T-Mobile line, the TLS handshake alone took 0.4-2.1s, and
        # a probe failed all three attempts outright — i.e. ▶️ could fail on a
        # perfectly working connection that is merely slow to say hello.
        self._client = s3.StreamingClient(
            s3.StreamingClientOptions(
                api_key=api_key or os.environ["ASSEMBLYAI_API_KEY"],
                connect_timeout=10.0,
                max_connection_retries=3,
            )
        )
        # Registration must precede connect() — the SDK reports handshake
        # failures through these handlers, not through an exception.
        self._client.on(s3.StreamingEvents.Begin, self._on_begin)
        self._client.on(s3.StreamingEvents.Turn, self._on_turn)
        self._client.on(s3.StreamingEvents.Error, self._on_error)
        self._client.on(s3.StreamingEvents.Termination, self._on_termination)

        params = dict(
            sample_rate=self._decimator.out_rate,
            encoding=s3.Encoding.pcm_s16le,
            format_turns=True,
        )
        # Endpointing. Left unset, the SDK uses phone-call defaults that wait for
        # the speaker to hand the floor over — a preacher never does, so turns ran
        # to 921 characters (~60s) on the 2026-09-06 service and nothing reached
        # translation until then. See config.STTConfig for the reasoning.
        if end_of_turn_confidence_threshold is not None:
            params["end_of_turn_confidence_threshold"] = end_of_turn_confidence_threshold
        if min_turn_silence_ms is not None:
            # `min_turn_silence`, not the older `min_end_of_turn_silence_when_confident` —
            # the SDK marks that one deprecated and warns on every start.
            params["min_turn_silence"] = min_turn_silence_ms
        if max_turn_silence_ms is not None:
            params["max_turn_silence"] = max_turn_silence_ms
        if language_codes:
            params["language_codes"] = language_codes  # pin a candidate set
        else:
            params["language_detection"] = True  # full auto, any supported language

        self._client.connect(s3.StreamingParameters(**params))

        # A Begin frame is the only proof the session is actually live; connect()
        # returning tells us nothing (see module docstring).
        if not self._handshake.wait(timeout=connect_timeout_s):
            raise RuntimeError(
                f"AssemblyAI: no response from the streaming API within {connect_timeout_s:.0f}s — "
                "recognition not started (check the internet connection and the API key)"
            )
        if self.fatal_error is not None:
            raise RuntimeError(f"AssemblyAI: connection not established — {self.fatal_error}")

    @property
    def is_healthy(self) -> bool:
        return self.fatal_error is None

    def _on_begin(self, client, event) -> None:
        self._handshake.set()

    def _on_turn(self, client, turn) -> None:
        """Hand each turn to translation when it ends — and end long ones early.

        Waiting for `end_of_turn` puts a hard floor under the delay equal to how
        long the speaker talks without pausing — you cannot translate a sentence
        before it has been said. Measured 2026-09-06: a 457-character turn, ~30s
        of unbroken speech, and the booth's delay tracked it exactly.

        The first fix sent clauses as AssemblyAI marked words `word_is_final`.
        Measured 2026-09-11 on a real sermon, that never fired: across 173
        in-progress messages not one word was marked final before its turn
        closed, so turns still went out whole — up to 86 words, ~37s of speech —
        and the output buffer then skipped whatever had queued up behind them.

        So the cut is made on the server instead. Once a turn holds
        partial_min_words and the speaker has paused after them (a gap of
        partial_gap_ms between two words already spoken), or it reaches
        partial_max_words regardless, ForceEndpoint closes it. The server sends
        it as a normal formatted end-of-turn — punctuation and casing intact,
        no guessing which words are settled — and the next clause starts fresh.
        """
        if turn.end_of_turn:
            if turn.transcript.strip():
                self._emit(turn.transcript, turn)
            return

        if not self._partial_emit or turn.turn_order == self._forced_turn:
            return  # off, or already asked the server to close this one
        words = turn.words
        if len(words) < self._partial_min_words:
            return
        paused = any(
            words[i + 1].start - words[i].end >= self._partial_gap_ms
            for i in range(self._partial_min_words - 1, len(words) - 1)
        )
        if paused or len(words) >= self._partial_max_words:
            self._forced_turn = turn.turn_order
            self._client.force_endpoint()

    def _emit(self, text: str, turn) -> None:
        now = time.monotonic()
        lag = None
        if turn.words and self._t_first_feed is not None:
            # Word timings count only the audio the server received. feed() runs
            # at capture pace, so adding back what the backlog cap threw away
            # gives when the word was actually spoken — and the difference is
            # how far behind the speaker recognition is running.
            spoken_at = self._t_first_feed + self.dropped_audio_s + turn.words[-1].end / 1000.0
            lag = now - spoken_at
            self.last_lag_s = lag
        wps = None
        if len(turn.words) >= 4:  # fewer words say nothing reliable about pace
            span_s = (turn.words[-1].end - turn.words[0].start) / 1000.0
            if span_s > 0.5:
                wps = len(turn.words) / span_s
        self._events.put(
            TranscriptEvent(
                text=text,
                is_final=True,
                language_code=turn.language_code or "auto",
                received_at=now,
                stt_lag_s=lag,
                stt_backlog_s=self._send_backlog_s(),
                words_per_s=wps,
            )
        )

    def _send_backlog_s(self) -> float | None:
        write_queue = getattr(self._client, "_write_queue", None)  # SDK-private
        return None if write_queue is None else write_queue.qsize() * self._send_chunk_s

    def _drop_stale_audio(self) -> None:
        """Enforce MAX_SEND_BACKLOG_S on the SDK's unbounded send queue.

        Everything queued is stale by definition once the cap is hit, so all of
        it goes and the next chunk starts current. Only audio is discarded —
        control frames (Terminate, ForceEndpoint) are put back.
        """
        write_queue = getattr(self._client, "_write_queue", None)
        if write_queue is None or write_queue.qsize() * self._send_chunk_s <= MAX_SEND_BACKLOG_S:
            return
        kept, dropped = [], 0
        while True:
            try:
                item = write_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, bytes):
                dropped += 1
            else:
                kept.append(item)
        for item in kept:
            write_queue.put(item)
        self.dropped_audio_s += dropped * self._send_chunk_s
        print(f"[assemblyai] internet too slow: dropped {dropped * self._send_chunk_s:.1f}s of audio "
              f"to stay current ({self.dropped_audio_s:.0f}s so far this service)")

    def _on_error(self, client, error) -> None:
        print(f"[assemblyai] streaming error: {error}")
        self.fatal_error = str(error)
        self._handshake.set()  # unblock a constructor still waiting on the handshake

    def _on_termination(self, client, event) -> None:
        if self._closing:
            return  # we asked for this one
        self.fatal_error = "connection closed by the server"
        self._handshake.set()

    def feed(self, pcm_chunk: np.ndarray) -> TranscriptEvent | None:
        if self._t_first_feed is None:
            self._t_first_feed = time.monotonic()
        self._send_buffer = np.concatenate([self._send_buffer, pcm_chunk])
        while len(self._send_buffer) >= self._send_threshold:
            segment, self._send_buffer = (
                self._send_buffer[: self._send_threshold],
                self._send_buffer[self._send_threshold :],
            )
            segment = self._decimator.process(segment)
            pcm16 = (np.clip(segment, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            self._drop_stale_audio()
            self._client.stream(pcm16)
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closing = True
        self._client.disconnect(terminate=True)
