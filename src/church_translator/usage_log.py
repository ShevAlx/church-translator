"""Usage/cost logging — report §06 "Как отслеживать расходы".

Every STT/MT/TTS call already tells you its own duration or character count;
this just writes that down next to a timestamp and a session id, in a plain
CSV a spreadsheet (or a Google Sheet) can ingest directly. No cloud service,
no separate database — the file lives next to the app on the booth computer.
"""

from __future__ import annotations

import csv
import datetime as dt
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
    date_part = session_id.split("-")[1]
    return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
