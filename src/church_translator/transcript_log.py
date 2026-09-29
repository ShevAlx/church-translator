"""What was said and what went out, line by line, for one service.

usage.csv records how long things took and how many characters moved, never
the words. So after the 2026-09-27 service, "some phrases were missing" could
only be checked by transcribing the recordings by hand: which phrase, in which
channel, and why — MT filter, lag drop, or never recognized — was guesswork.

One TSV per session next to its recordings (so logging.keep_days prunes it with
them): every segment with its source text, translation and status, and at ⏹ a
list of the turns the output buffer skipped for lag, with their text.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

FIELDS = ["time", "language", "turn", "status", "source", "translation"]


class TranscriptLog:
    """Thread-safe: every language thread writes into the same file.

    Rows are kept in memory only as turn -> source text, which is what the
    dropped-turn report at ⏹ needs; the file is the full record. Until open()
    is called (no recordings_dir configured) nothing touches the disk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._turn_text: dict[tuple[str, int], str] = {}

    def open(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._file = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file, delimiter="\t")
            if self._file.tell() == 0:
                self._writer.writerow(FIELDS)
                self._file.flush()

    def log(self, language: str, turn: int, status: str, source: str, translation: str = "") -> None:
        """status: ok | empty (MT had nothing to say) | filtered (MT reply
        rejected) | error. Written as it happens and flushed, so a crash or a
        pulled plug still leaves everything up to that point."""
        with self._lock:
            key = (language, turn)
            self._turn_text[key] = f"{self._turn_text[key]} {source}" if key in self._turn_text else source
            if self._writer is None:
                return
            self._write([time.strftime("%H:%M:%S"), language, turn, status, source, translation])

    def log_dropped(self, language: str, turns: list[int]) -> list[str]:
        """Record the turns the output buffer skipped for lag; returns their text."""
        texts = []
        with self._lock:
            for turn in turns:
                text = self._turn_text.get((language, turn), "")
                texts.append(text)
                if self._writer is not None:
                    self._write(["", language, turn, "dropped_lag", text, ""])
        return texts

    def _write(self, row: list) -> None:
        """Called with the lock held. A full SD card must cost the log, never
        the translation: this runs on the language threads, and an exception
        here would kill a channel for the rest of the service."""
        try:
            self._writer.writerow(row)
            self._file.flush()
        except OSError as exc:
            print(f"[transcript] write failed, transcript log stopped: {exc}")
            self._writer = None

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._file = self._writer = None
