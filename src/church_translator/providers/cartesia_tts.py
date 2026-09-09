"""TTS provider backed by Cartesia (report §04.1/§04.3: locked voice_id for a
whole session, supports cloned voices). Requires CARTESIA_API_KEY.

Verified against the installed SDK (cartesia==4.0.1) by introspecting the
real `TTSResource` rather than guessing: `.bytes()` exists but is deprecated
in favour of `.generate()`, which returns a `BinaryAPIResponse` you read with
`.read()`. `output_format` for raw PCM is `{"container": "raw", "encoding":
"pcm_f32le", "sample_rate": N}` — `pcm_f32le` maps straight onto numpy
float32, no int16 round-trip needed. `model_id` defaults to "sonic-latest"
rather than the "Sonic-Turbo" name from the original research pass — this
SDK's current model list is "sonic-3.5" / "sonic-3" / "sonic-latest" (plus
dated pins); "sonic-latest" is the forward-safe choice.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterator

import numpy as np

from .base import TTSProvider


class CartesiaTTS(TTSProvider):
    def __init__(
        self,
        samplerate: int,
        model: str = "sonic-latest",
        api_key: str | None = None,
        speed: str = "normal",
    ):
        from cartesia import Cartesia

        self._client = Cartesia(api_key=api_key or os.environ["CARTESIA_API_KEY"])
        self.samplerate = samplerate
        self._model = model
        # Passed through per config.TTSConfig. Note that "fast" measured as a
        # no-op on sonic-latest with the cloned voice (2026-09-06) — it is here
        # as a knob, not as a working fix for channel overload.
        self._speed = speed

    def synthesize(self, text: str, voice_id: str, language_code: str) -> np.ndarray:
        if not text.strip() or not voice_id:
            return np.zeros(0, dtype=np.float32)

        response = self._client.tts.generate(
            model_id=self._model,
            output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": self.samplerate},
            transcript=text,
            voice=voice_id,  # str is a valid VoiceSpecifierParam — must stay the SAME id every
                              # call for a given language/session (report §04.1 "locked voice" fix)
            language=language_code,
            speed=self._speed,
        )
        return np.frombuffer(response.read(), dtype=np.float32)

    def synthesize_stream(self, text: str, voice_id: str, language_code: str) -> Iterator[np.ndarray]:
        """Server-sent-events variant: audio starts arriving in ~0.2s instead of
        after the whole utterance is rendered. Same voice, same billing (Cartesia
        charges per character either way) — only the wait changes.

        `TTSSSEChunkEvent.data` is a base64 **str**, not raw bytes (verified
        against the live API 2026-09-06 — the first version of this method
        assumed bytes and raised TypeError on the very first chunk, which is why
        nothing had ever called it). Decode before touching the PCM.
        """
        if not text.strip() or not voice_id:
            return

        tail = b""
        for chunk in self._client.tts.sse(
            model_id=self._model,
            transcript=text,
            voice={"mode": "id", "id": voice_id},
            language=language_code,
            speed=self._speed,
            output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": self.samplerate},
        ):
            data = getattr(chunk, "data", None)
            if not data:
                continue  # non-audio events (done/timestamps) carry no `data`
            pcm = base64.b64decode(data) if isinstance(data, str) else data
            # float32 is 4 bytes; SSE chunk boundaries do not respect that, so a
            # partial sample is carried over rather than misaligning the stream.
            buf = tail + pcm
            usable = len(buf) - (len(buf) % 4)
            tail = buf[usable:]
            if usable:
                yield np.frombuffer(buf[:usable], dtype=np.float32)
