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
menu-bar app lit up "● Работает" ("Running"), and the booth heard silence for a whole
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
        self._send_threshold = max(1, int(samplerate * send_chunk_ms / 1000))
        self._send_buffer = np.zeros(0, dtype=np.float32)
        self._handshake = threading.Event()
        self._closing = False
        self._partial_emit = partial_emit
        self._partial_min_words = partial_min_words
        self._partial_max_words = partial_max_words
        self._partial_gap_ms = partial_gap_ms
        self._turn_order: int | None = None
        self._emitted_words = 0

        self._client = s3.StreamingClient(
            s3.StreamingClientOptions(api_key=api_key or os.environ["ASSEMBLYAI_API_KEY"])
        )
        # Registration must precede connect() — the SDK reports handshake
        # failures through these handlers, not through an exception.
        self._client.on(s3.StreamingEvents.Begin, self._on_begin)
        self._client.on(s3.StreamingEvents.Turn, self._on_turn)
        self._client.on(s3.StreamingEvents.Error, self._on_error)
        self._client.on(s3.StreamingEvents.Termination, self._on_termination)

        params = dict(
            sample_rate=samplerate,
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
                f"AssemblyAI: нет ответа от streaming API за {connect_timeout_s:.0f}с — "
                "распознавание не запущено (проверьте интернет и ключ)"
            )
        if self.fatal_error is not None:
            raise RuntimeError(f"AssemblyAI: соединение не установлено — {self.fatal_error}")

    @property
    def is_healthy(self) -> bool:
        return self.fatal_error is None

    def _on_begin(self, client, event) -> None:
        self._handshake.set()

    def _on_turn(self, client, turn) -> None:
        """Emit clauses as they are finalized, not only when the turn closes.

        Waiting for `end_of_turn` puts a hard floor under the delay equal to how
        long the speaker talks without pausing — you cannot translate a sentence
        before it has been said. Measured 2026-09-06: a 457-character turn, ~30s
        of unbroken speech, and the booth's delay tracked it exactly (10s, then
        20s, then 30s) while MT and TTS were only costing 0.18s.

        AssemblyAI marks each word `word_is_final` once it stops revising it and
        gives millisecond timings, so a finished clause can go to translation
        while the speaker is still talking — which is what a human interpreter
        does. Cuts are placed at a real pause between words wherever one exists,
        so the fragment handed to MT is a clause and not an arbitrary slice.
        """
        if turn.turn_order != self._turn_order:
            self._turn_order = turn.turn_order
            self._emitted_words = 0

        if turn.end_of_turn:
            # The formatted transcript is the good one (punctuation, casing), so
            # the tail is taken from it — minus whatever already went out early.
            tokens = turn.transcript.split()
            tail = " ".join(tokens[self._emitted_words :])
            self._emitted_words = 0
            self._turn_order = None
            if tail.strip():
                self._emit(tail, turn)
            return

        if not self._partial_emit:
            return

        final_words = []
        for w in turn.words:
            if not w.word_is_final:
                break  # only the leading run of settled words is safe to send
            final_words.append(w)
        pending = final_words[self._emitted_words :]
        if len(pending) < self._partial_min_words:
            return

        # Prefer the last real pause; a clause boundary beats a word count.
        cut = 0
        for i in range(len(pending) - 1):
            if pending[i + 1].start - pending[i].end >= self._partial_gap_ms:
                cut = i + 1
        if cut < self._partial_min_words:
            if len(pending) < self._partial_max_words:
                return  # keep waiting for a pause
            cut = self._partial_max_words  # speaker is not pausing; cut anyway

        self._emit(" ".join(w.text for w in pending[:cut]), turn)
        self._emitted_words += cut

    def _emit(self, text: str, turn) -> None:
        self._events.put(
            TranscriptEvent(
                text=text,
                is_final=True,
                language_code=turn.language_code or "auto",
                received_at=time.monotonic(),
            )
        )

    def _on_error(self, client, error) -> None:
        print(f"[assemblyai] streaming error: {error}")
        self.fatal_error = str(error)
        self._handshake.set()  # unblock a constructor still waiting on the handshake

    def _on_termination(self, client, event) -> None:
        if self._closing:
            return  # we asked for this one
        self.fatal_error = "соединение закрыто сервером"
        self._handshake.set()

    def feed(self, pcm_chunk: np.ndarray) -> TranscriptEvent | None:
        self._send_buffer = np.concatenate([self._send_buffer, pcm_chunk])
        while len(self._send_buffer) >= self._send_threshold:
            segment, self._send_buffer = (
                self._send_buffer[: self._send_threshold],
                self._send_buffer[self._send_threshold :],
            )
            pcm16 = (np.clip(segment, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            self._client.stream(pcm16)
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closing = True
        self._client.disconnect(terminate=True)
