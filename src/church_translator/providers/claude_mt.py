"""MT provider backed by the Claude API. Requires ANTHROPIC_API_KEY.

Translation only, on purpose: no chit-chat, no explanations, so latency and
output length stay predictable — this runs on the ~0.25-0.3s slice of the
report's §06 latency budget.
"""

from __future__ import annotations

import os

from .base import MTProvider

_SYSTEM_PROMPT = (
    "You are a real-time interpreter for a live church service. Translate the "
    "given utterance into {target_language} only. Output the translation and "
    "nothing else — no notes, no quotes, no alternate phrasings. Preserve tone "
    "and register; keep numbers, names, and scripture references unchanged."
)


class ClaudeMT(MTProvider):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        import anthropic  # deferred import: only needed in pipeline.mode == "real"

        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            return ""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT.format(target_language=target_language),
            messages=[{"role": "user", "content": text}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
