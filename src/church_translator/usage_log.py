"""Usage/cost logging — report §06 "How to track spend".

Every STT/MT/TTS call already tells you its own duration or character count;
this just writes that down next to a timestamp and a session id, in a plain
CSV a spreadsheet (or a Google Sheet) can ingest directly. No cloud service,
no separate database — the file lives next to the app on the booth computer.
"""

from __future__ import annotations

import csv
import datetime as dt
import shutil
import threading
from pathlib import Path

FIELDS = ["timestamp", "session_id", "language", "component", "duration_s", "chars", "note"]


class UsageLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(FIELDS)

    def log(
        self,
        session_id: str,
        language: str,
        component: str,  # "stt" | "mt" | "tts"
        duration_s: float | None = None,
        chars: int | None = None,
        note: str = "",
    ) -> None:
        row = [
            dt.datetime.now().isoformat(timespec="seconds"),
            session_id,
            language,
            component,
            f"{duration_s:.3f}" if duration_s is not None else "",
            chars if chars is not None else "",
            note,
        ]
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def new_session_id(self) -> str:
        return dt.datetime.now().strftime("svc-%Y%m%d-%H%M%S")


def date_folder(session_id: str) -> str:
    """"svc-20260820-013845" -> "2026-08-20" — one folder per day of service,
    shared by anything that names its files after a session_id (recordings,
    debug audio) so they land in the same place without a second clock read."""
    for part in session_id.split("-"):
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
    # No date in the id (a hand-made session name in a test) — one folder for
    # those beats scattering half-parsed garbage directories through logs/.
    return "undated"


def prune_dated_folders(root: str | Path | None, keep_days: int) -> int:
    """Delete the YYYY-MM-DD folders under `root` (see date_folder) older than
    `keep_days`; anything not named like a date is left alone. Returns how many
    went. keep_days <= 0 keeps everything."""
    if not root or keep_days <= 0:
        return 0
    base = Path(root)
    if not base.is_dir():
        return 0
    cutoff = dt.date.today() - dt.timedelta(days=keep_days)
    removed = 0
    for child in base.iterdir():
        try:
            day = dt.date.fromisoformat(child.name)
        except ValueError:
            continue
        if child.is_dir() and day < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
