"""What a service cost, in dollars — from the same rows usage.csv already holds.

Three bills, three different meters, and only one of them is about tokens:

- AssemblyAI streaming bills the open WebSocket by the hour, speech or silence.
  The `session` row LiveSession writes at ⏹ carries that duration.
- Claude bills input and output tokens. The API returns the exact counts; they
  ride along in the `mt` row's note as `in_tok=` / `out_tok=`.
- Cartesia bills one credit per synthesized character — the `tts` row's chars.

Rows written before any of that existed (before 2026-09-13) have no tokens and
no session row; those get an estimate and are marked as such, rather than
reported as $0.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .config import PricingConfig

# Fallback for old mt rows, calibrated on the 2026-09-11 test: the system prompt
# is ~250 tokens on every call, an English clause is ~4.5 chars per token, and
# the Cyrillic rendering ~2.5 chars per token (the logged chars are the output).
_EST_PROMPT_TOKENS = 260
_EST_SOURCE_CHARS_PER_TOKEN = 4.5
_EST_OUTPUT_CHARS_PER_TOKEN = 2.5

_TOKENS_RE = re.compile(r"in_tok=(\d+)\s+out_tok=(\d+)")


@dataclass
class SessionCost:
    session_id: str
    stt_hours: float = 0.0
    mt_calls: int = 0
    mt_input_tokens: int = 0
    mt_output_tokens: int = 0
    tts_chars: int = 0
    estimated: set[str] = field(default_factory=set)  # which parts are guesses: "stt", "mt"
    pricing: PricingConfig = field(default_factory=PricingConfig)
    billable: bool = True  # False for a mock/passthrough session: nothing reached a vendor

    @property
    def stt_usd(self) -> float:
        return self.stt_hours * self.pricing.stt_usd_per_hour if self.billable else 0.0

    @property
    def mt_usd(self) -> float:
        if not self.billable:
            return 0.0
        return (
            self.mt_input_tokens * self.pricing.mt_input_usd_per_mtok
            + self.mt_output_tokens * self.pricing.mt_output_usd_per_mtok
        ) / 1e6

    @property
    def tts_usd(self) -> float:
        return self.tts_chars * self.pricing.tts_usd_per_1k_chars / 1000 if self.billable else 0.0

    @property
    def total_usd(self) -> float:
        return self.stt_usd + self.mt_usd + self.tts_usd

    def add_mt(self, out_chars: int, note: str) -> None:
        self.mt_calls += 1
        m = _TOKENS_RE.search(note or "")
        if m:
            self.mt_input_tokens += int(m.group(1))
            self.mt_output_tokens += int(m.group(2))
        else:
            self.estimated.add("mt")
            self.mt_input_tokens += _EST_PROMPT_TOKENS + round(out_chars / _EST_SOURCE_CHARS_PER_TOKEN)
            self.mt_output_tokens += round(out_chars / _EST_OUTPUT_CHARS_PER_TOKEN)

    def lines(self) -> list[str]:
        """Plain-text breakdown, one line per bill — shared by the CLI and the bot."""
        def mark(part: str) -> str:
            return " ~" if part in self.estimated else ""

        return [
            f"AssemblyAI  {self.stt_hours * 60:6.1f} мин соединения{mark('stt')}  ${self.stt_usd:6.2f}",
            f"Claude      {self.mt_calls} запросов, {self.mt_input_tokens:,} вх. / "
            f"{self.mt_output_tokens:,} вых. токенов{mark('mt')}  ${self.mt_usd:6.2f}".replace(",", " "),
            f"Cartesia    {self.tts_chars:,} символов  ${self.tts_usd:6.2f}".replace(",", " "),
            f"Итого  ${self.total_usd:.2f}" + ("   (~ = оценка, в старых строках лога нет точных данных)"
                                             if self.estimated else ""),
        ]


def costs_from_rows(rows: Iterable[dict], pricing: PricingConfig) -> dict[str, SessionCost]:
    """usage.csv rows -> cost per session_id, in first-seen order."""
    out: dict[str, SessionCost] = {}
    span: dict[str, tuple[dt.datetime, dt.datetime]] = {}
    has_session_row: set[str] = set()
    for row in rows:
        sid = row["session_id"]
        cost = out.setdefault(sid, SessionCost(sid, pricing=pricing))
        try:
            ts = dt.datetime.fromisoformat(row["timestamp"])
            first, last = span.get(sid, (ts, ts))
            span[sid] = (min(first, ts), max(last, ts))
        except ValueError:
            pass
        component = row["component"]
        chars = int(row["chars"]) if row.get("chars") else 0
        if component == "session":
            has_session_row.add(sid)
            if "mode=real" in (row.get("note") or ""):
                cost.stt_hours += float(row["duration_s"] or 0) / 3600
            else:
                cost.billable = False
        elif component == "mt":
            cost.add_mt(chars, row.get("note", ""))
        elif component == "tts":
            cost.tts_chars += chars
    for sid, cost in out.items():
        if sid not in has_session_row and sid in span:
            # No ⏹ row: first to last logged event is the best guess, and an
            # underestimate — the socket was open before the first word and
            # after the last one.
            first, last = span[sid]
            cost.stt_hours = (last - first).total_seconds() / 3600
            cost.estimated.add("stt")
    return out


def costs_from_csv(path: str | Path, pricing: PricingConfig) -> dict[str, SessionCost]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return costs_from_rows(csv.DictReader(f), pricing)
