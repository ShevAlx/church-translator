"""MT provider backed by the Claude API. Requires ANTHROPIC_API_KEY.

Translation only, on purpose: no chit-chat, no explanations, so latency and
output length stay predictable — this runs on the ~0.25-0.3s slice of the
report's §06 latency budget.
"""

from __future__ import annotations

import os

from .base import MTProvider

# Length is not a style question here, it is the channel budget. Russian and
# Ukrainian rendered from English ran at 95% of the service's own length
# (measured 2026-09-06), leaving no room to ever recover lost time — the booth
# had to skip whole thoughts to stay current. Asking for an interpretation
# rather than a translation gets that back: measured on real sermon text, 91%
# of source length -> 77%, which is 8% less synthesized audio, with the
# scripture reference, numbers and names intact. This is what a human
# simultaneous interpreter does, and it costs nothing at playback.
_SYSTEM_PROMPT = (
    "You are a simultaneous interpreter for a live church service, speaking into "
    "listeners' headphones while the preacher is still talking. Render the "
    "utterance into {target_language}.\n"
    "Interpret, do not transcribe: say the same thing the way a fluent "
    "{target_language} speaker would say it out loud, in as few words as carry "
    "the full meaning. Drop filler, false starts, and self-repetition. Never drop "
    "or soften: scripture references, numbers, names, commands, or the point "
    "being made. Preserve tone and register.\n"
    "You may be given a clause from the middle of a sentence — interpret just "
    "that clause, do not complete it or add context of your own.\n"
    "Output only the interpretation — no notes, no quotes, no alternatives."
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
            max_tokens=2048,
            system=_SYSTEM_PROMPT.format(target_language=target_language),
            messages=[{"role": "user", "content": text}],
        )
        # Cyrillic costs 2-3x the tokens of the English it came from, so a long
        # utterance can hit the ceiling and come back neatly cut off mid-sentence
        # with no error anywhere. Loud on stdout beats a listener wondering why
        # the sermon stopped halfway through a clause.
        if response.stop_reason == "max_tokens":
            print(
                f"[mt] translation truncated by max_tokens ({target_language}, {len(text)} chars in) "
                "— lower MAX_SEGMENT_CHARS in pipeline.py"
            )
        return "".join(block.text for block in response.content if block.type == "text").strip()
