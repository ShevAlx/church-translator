"""TTS provider backed by Cartesia (report §04.1/§04.3: locked voice_id for a
whole session, supports cloned voices). Requires CARTESIA_API_KEY.

Verified against the installed SDK (cartesia==4.0.1) by introspecting the
real `TTSResource` rather than guessing: `.bytes()` exists but is deprecated
in favour of `.generate()`, which returns a `BinaryAPIResponse` you read with
`.read()`. `output_format` for raw PCM is `{"container": "raw", "encoding":
"pcm_s16le", "sample_rate": N}`. The model is "sonic-3" rather than
"sonic-latest": only Sonic-3 honours a numeric `generation_config.speed`, and
that is the one speed control that measurably works on the cloned voice (see
config.TTSConfig for the numbers).

Continuous context (2026-09-11). Synthesizing every clause as its own request
made each one sound like a complete sentence — falling pitch at the end, a
fresh start at the beginning — in the middle of the preacher's sentence. Over
the WebSocket API the clauses go into one Cartesia *context* in turn, so the
model hears what came before and carries the intonation on; on an A/B of the
same 34 sermon clauses that was preferred by ear, at the same 0.23s to first
audio. Measured limits the code below is built around:
  - speed is fixed per context: a later push asking for 1.3 in a context
    started at 1.0 came out no faster, so a speed change starts a new context;
  - a context survived 5s without input and was gone by 8s;
  - an idle socket died on a keepalive timeout after ~60s.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Iterator

import numpy as np

from .base import TTSProvider

CONTEXT_IDLE_S = 4.0      # start a fresh context after this long without input (expiry measured 5-8s)
CONNECTION_IDLE_S = 20.0  # reopen the socket after this long idle rather than trust it
WS_RECV_TIMEOUT_S = 8.0   # no audio for this long mid-clause = the socket is dead


class CartesiaTTS(TTSProvider):
    def __init__(
        self,
        samplerate: int,
        model: str = "sonic-3",
        api_key: str | None = None,
        speed: str = "normal",
        continuous: bool = True,
    ):
        from cartesia import Cartesia

        self._client = Cartesia(api_key=api_key or os.environ["CARTESIA_API_KEY"])
        self.samplerate = samplerate
        self._model = model
        # The pre-Sonic-3 "slow"/"normal"/"fast" knob — measured as a no-op on
        # the cloned voice (2026-09-06, again 2026-09-11). Sent only to models
        # that have no numeric speed.
        self._legacy_speed = speed
        self._continuous = continuous and model.startswith("sonic-3")
        # One socket and one live context per instance; the pipeline builds one
        # instance per language and calls it from that language's thread only.
        self._conn_manager = None
        self._conn = None
        self._ctx = None
        self._ctx_events: Iterator | None = None
        self._ctx_key: tuple | None = None
        self._last_use = 0.0

    def _output_format(self) -> dict:
        # 16-bit, not float32: half the bytes for audio no listener can tell
        # apart through an FM receiver. The download is not free — Cartesia
        # renders faster than real time, so two languages arrive as bursts of
        # several Mbit/s, base64 on top. On the booth's T-Mobile line (downlink
        # responsiveness ~1.1s under load, measured 2026-09-11) those bursts sit
        # in the same bloated queue as the ACKs of the upload to AssemblyAI.
        return {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.samplerate}

    def _speed_kwargs(self, speed: float | None) -> dict:
        if self._model.startswith("sonic-3"):
            return {"generation_config": {"speed": speed}} if speed is not None else {}
        return {"speed": self._legacy_speed}

    def synthesize(
        self, text: str, voice_id: str, language_code: str, speed: float | None = None
    ) -> np.ndarray:
        if not text.strip() or not voice_id:
            return np.zeros(0, dtype=np.float32)

        response = self._client.tts.generate(
            model_id=self._model,
            output_format=self._output_format(),
            transcript=text,
            voice=voice_id,  # str is a valid VoiceSpecifierParam — must stay the SAME id every
                              # call for a given language/session (report §04.1 "locked voice" fix)
            language=language_code,
            **self._speed_kwargs(speed),
        )
        return np.frombuffer(response.read(), dtype=np.int16).astype(np.float32) / 32768.0

    def synthesize_stream(
        self, text: str, voice_id: str, language_code: str, speed: float | None = None
    ) -> Iterator[np.ndarray]:
        if not text.strip() or not voice_id:
            return
        if self._continuous:
            started = False
            try:
                for pcm in self._stream_continuous(text, voice_id, language_code, speed):
                    started = True
                    yield pcm
                return
            except Exception as exc:  # noqa: BLE001 — any socket trouble: reconnect next time
                self._reset_connection()
                if started:
                    # Half the clause is already in the headphones; replaying it
                    # from the top would say it twice. Lose the rest instead.
                    print(f"[tts] continuous stream broke mid-clause, rest dropped: {exc}")
                    return
                print(f"[tts] continuous stream unavailable ({exc}) — one-shot for this clause")
        yield from self._stream_sse(text, voice_id, language_code, speed)

    # -- continuous context over the WebSocket API ----------------------------

    def _stream_continuous(
        self, text: str, voice_id: str, language_code: str, speed: float | None
    ) -> Iterator[np.ndarray]:
        idle = time.monotonic() - self._last_use
        if self._conn is not None and idle > CONNECTION_IDLE_S:
            self._reset_connection()
        if self._conn is None:
            self._conn_manager = self._client.tts.websocket_connect()
            self._conn = self._conn_manager.__enter__()

        key = (voice_id, language_code, speed)
        if self._ctx is None or key != self._ctx_key or idle > CONTEXT_IDLE_S:
            # The old context is simply abandoned: the server expires it within
            # seconds, and its queue stays registered so a late event for it is
            # routed there instead of into the new context's stream.
            self._ctx = self._conn.context(
                model_id=self._model,
                voice={"mode": "id", "id": voice_id},
                output_format=self._output_format(),
                language=language_code,
                generation_config={"speed": speed} if speed is not None else None,
                max_buffer_delay_ms=0,  # default 3000 would hold text back to "gather context"
                timeout=WS_RECV_TIMEOUT_S,
            )
            self._ctx_events = self._ctx.receive()
            self._ctx_key = key

        # Trailing space: the next clause continues the same speech, not a new word glued on.
        self._ctx.push(text.rstrip() + " ", flush=True)
        for event in self._ctx_events:
            if event.type == "chunk":
                data = getattr(event, "data", None)
                if data:
                    pcm = base64.b64decode(data) if isinstance(data, str) else data
                    yield np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32) / 32768.0
            elif event.type == "flush_done":
                self._last_use = time.monotonic()
                return
            elif event.type in ("done", "error"):
                self._ctx = None  # expired or refused — the next clause starts a fresh one
                raise RuntimeError(f"context ended with {event.type!r} before the clause was spoken")
        self._ctx = None
        raise RuntimeError("context stream ended before flush_done")

    def _reset_connection(self) -> None:
        self._ctx = None
        self._ctx_events = None
        self._ctx_key = None
        manager, self._conn_manager, self._conn = self._conn_manager, None, None
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 — closing a dead socket may itself fail
                pass

    def close(self) -> None:
        self._reset_connection()

    # -- one request per clause (fallback, and pre-sonic-3 models) ------------

    def _stream_sse(
        self, text: str, voice_id: str, language_code: str, speed: float | None
    ) -> Iterator[np.ndarray]:
        """Server-sent-events variant: audio starts arriving in ~0.2s instead of
        after the whole utterance is rendered. Same voice, same billing (Cartesia
        charges per character either way) — only the wait changes.

        `TTSSSEChunkEvent.data` is a base64 **str**, not raw bytes (verified
        against the live API 2026-09-06 — the first version of this method
        assumed bytes and raised TypeError on the very first chunk, which is why
        nothing had ever called it). Decode before touching the PCM.
        """
        tail = b""
        for chunk in self._client.tts.sse(
            model_id=self._model,
            transcript=text,
            voice={"mode": "id", "id": voice_id},
            language=language_code,
            output_format=self._output_format(),
            **self._speed_kwargs(speed),
        ):
            data = getattr(chunk, "data", None)
            if not data:
                continue  # non-audio events (done/timestamps) carry no `data`
            pcm = base64.b64decode(data) if isinstance(data, str) else data
            # int16 is 2 bytes; SSE chunk boundaries do not respect that, so a
            # partial sample is carried over rather than misaligning the stream.
            buf = tail + pcm
            usable = len(buf) - (len(buf) % 2)
            tail = buf[usable:]
            if usable:
                yield np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float32) / 32768.0
