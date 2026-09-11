"""Wires audio_io <-> providers into the shape from report §05: one STT stage
reads the mixer feed once, and fans final transcripts out to one MT+TTS
worker per configured language. Cost and thread count scale with the number
of languages, never with how many receivers are tuned to a channel.

Three run modes (config: pipeline.mode):
  passthrough  audio routed straight to every output channel, no STT/MT/TTS.
               Proves device I/O, channel mapping, threading, silence-on-
               underrun — the part that has nothing to do with translation.
  mock         real STT->MT->TTS thread shape, fake providers (a beep per
               detected utterance). Proves the pipeline's own plumbing.
  real         Milestone 2: real provider instances passed in by the caller.
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .audio_io import AudioRouter
from .config import AppConfig, LanguageConfig, TTSConfig
from .providers.base import MTProvider, STTProvider, TranscriptEvent, TTSProvider
from .usage_log import UsageLogger, date_folder


# A turn this long has already cost the listener more than it can ever repay:
# nothing is heard until the speaker pauses, and by then the translation is a
# paragraph behind. Measured on the 2026-09-06 service, AssemblyAI returned
# single turns of up to 921 characters (~60s of speech, synthesized to a 66s
# blob) because a preacher does not pause the way a phone caller does.
# ~140 characters is about one spoken sentence, which synthesizes to ~8-10s of
# Russian. Bigger blocks put the listener further behind before a single word
# comes out; much smaller ones strip the sentence context the translation needs.
MAX_SEGMENT_CHARS = 140

_SENTENCE_END = (". ", "! ", "? ", "; ", ": ", "… ")


def split_for_streaming(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Break one long transcript into sentence-sized segments.

    Each segment is translated and spoken on its own, so the first sentence is
    already coming out of the speaker while the rest is still being translated,
    and the backlog cap has whole sentences to drop instead of one huge blob.
    Short turns (the common case) come back unchanged as a single segment.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    segments: list[str] = []
    current = ""
    for sentence in _iter_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            segments.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current.strip():
        segments.append(current.strip())
    return segments


def _iter_sentences(text: str):
    """Yield sentence-ish pieces, falling back to a word split for a run of
    speech with no punctuation at all (a formatted transcript usually has it,
    but an unpunctuated turn must not come back as one 900-char 'sentence')."""
    start = 0
    for i in range(len(text) - 1):
        if text[i : i + 2] in _SENTENCE_END:
            yield text[start : i + 1].strip()
            start = i + 2
    tail = text[start:].strip()
    if not tail:
        return
    if len(tail) <= MAX_SEGMENT_CHARS:
        yield tail
        return
    words, chunk = tail.split(), ""
    for word in words:
        if chunk and len(chunk) + len(word) + 1 > MAX_SEGMENT_CHARS:
            yield chunk
            chunk = word
        else:
            chunk = f"{chunk} {word}".strip()
    if chunk:
        yield chunk


def trim_silence_stream(chunks, threshold: float = 0.005):
    """Drop the silent head and tail of one synthesized clause, keep the middle.

    Cartesia pads every generation with ~0.15s of silence at each end. That was
    4% of the channel when a whole turn was one generation; with clauses going
    out while the preacher is still speaking there are far more generations, and
    the padding measured 7% (2026-09-06). Seven percent of a channel running at
    95% load is the difference between recovering lost time and never recovering
    it — and unlike speeding playback up, cutting silence changes nothing about
    how the voice sounds.

    Pauses *inside* a clause are left alone: they are the speaker's rhythm, not
    padding. Only the leading and trailing silence of each generation goes.
    """
    started = False
    held: list[np.ndarray] = []
    for chunk in chunks:
        loud = np.abs(chunk) > threshold
        if not loud.any():
            if started:
                held.append(chunk)  # might be an internal pause; decided when more audio arrives
            continue
        first = int(loud.argmax())
        last = len(loud) - 1 - int(loud[::-1].argmax())
        if started and held:
            yield np.concatenate(held)  # it was internal after all — keep the rhythm
        held = []
        body = chunk[first : last + 1] if not started else chunk[: last + 1]
        started = True
        if len(body):
            yield body
        if last + 1 < len(chunk):
            held.append(chunk[last + 1 :])
    # whatever silence is still held at the end is trailing padding — dropped


def _write_debug_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    """16-bit mono WAV — plain enough that QuickTime/afplay/anything opens it,
    so you can hear exactly what would have gone to the output channel
    without needing a transmitter/receiver hooked up yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(pcm16.tobytes())


def _stt_note(event: TranscriptEvent) -> str:
    """usage.csv note for an STT row: how far behind the speaker recognition is
    running. Empty for providers that do not report it (the mock)."""
    parts = []
    if event.stt_lag_s is not None:
        parts.append(f"lag={event.stt_lag_s:.1f}s")
    if event.stt_backlog_s is not None:
        parts.append(f"backlog={event.stt_backlog_s:.1f}s")
    if event.words_per_s is not None:
        parts.append(f"wps={event.words_per_s:.2f}")
    return " ".join(parts)


class _STTStage(threading.Thread):
    """Runs once per service, regardless of language count (report §05 fan-out point)."""

    def __init__(
        self,
        router: AudioRouter,
        stt: STTProvider,
        fanout: list["queue.Queue[TranscriptEvent]"],
        usage_logger: UsageLogger,
        session_id: str,
        stop_event: threading.Event,
    ):
        super().__init__(name="stt-stage", daemon=True)
        self._router = router
        self._stt = stt
        self._fanout = fanout
        self._usage_logger = usage_logger
        self._session_id = session_id
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._router.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            event = self._stt.feed(chunk)
            if event is None or not event.is_final:
                continue
            self._usage_logger.log(
                self._session_id, event.language_code, "stt",
                duration_s=time.monotonic() - t0, chars=len(event.text), note=_stt_note(event),
            )
            for q in self._fanout:
                # Never block here. This thread is also the one feeding audio to
                # AssemblyAI; if a language worker stalls, a blocking put() would
                # back the input queue up and start losing the mixer feed itself,
                # i.e. one slow language would break recognition for both.
                try:
                    q.put_nowait(event)
                except queue.Full:
                    try:
                        q.get_nowait()  # drop the stalest pending turn, keep the newest
                        q.put_nowait(event)
                    except (queue.Empty, queue.Full):
                        pass


# Translation piling up in the headphones pushes the voice faster whatever the
# preacher's pace: ~3s queued is where the wait starts to be noticeable, and by
# ~8s the output buffer is close to skipping whole turns (audio.max_backlog_s).
QUEUE_BOOST_START_S = 3.0
QUEUE_BOOST_FULL_S = 8.0
PACE_SMOOTHING = 0.3  # weight of the newest turn in the running pace estimate
# Largest speed change from one segment to the next. A jump from 1.0 straight to
# 1.3 between two clauses of one sentence was audible on the 2026-09-11 test;
# 0.1 per segment still reaches the ceiling within three clauses.
MAX_SPEED_STEP = 0.1


def choose_speed(
    tts: "TTSConfig", pace_wps: float | None, queued_s: float, previous: float | None = None
) -> float:
    """Synthesis speed for the next segment, in [tts.speed_min, tts.speed_max].

    Two pressures, and the stronger one wins. A preacher speeding up means more
    words per second to render, so the voice has to keep pace or fall behind;
    audio already waiting in the channel means it is falling behind right now.
    The speed is fixed within a segment and moves between segments, at most
    MAX_SPEED_STEP from `previous` — a change mid-sentence is audible, and so
    is a big one between two clauses of the same sentence.
    """
    pace = 0.0
    if pace_wps is not None:
        pace = (pace_wps - tts.pace_normal_wps) / (tts.pace_fast_wps - tts.pace_normal_wps)
    queue = (queued_s - QUEUE_BOOST_START_S) / (QUEUE_BOOST_FULL_S - QUEUE_BOOST_START_S)
    pressure = min(1.0, max(0.0, pace, queue))
    target = tts.speed_min + pressure * (tts.speed_max - tts.speed_min)
    if previous is not None:
        target = min(max(target, previous - MAX_SPEED_STEP), previous + MAX_SPEED_STEP)
    # On a 0.1 grid: Cartesia fixes the speed per continuous context, so every
    # distinct value starts a new one and breaks the intonation — 1.16 then
    # 1.19 would cost a context for no audible gain.
    return round(target, 1)


class _LanguageStage(threading.Thread):
    """One per configured language: MT then TTS, writing into that language's output channel."""

    def __init__(
        self,
        lang: LanguageConfig,
        event_queue: "queue.Queue[TranscriptEvent]",
        mt: MTProvider,
        tts: TTSProvider,
        router: AudioRouter,
        usage_logger: UsageLogger,
        session_id: str,
        stop_event: threading.Event,
        debug_audio_dir: str | None = None,
        tts_config: "TTSConfig | None" = None,
    ):
        super().__init__(name=f"lang-{lang.code}", daemon=True)
        self._lang = lang
        self._queue = event_queue
        self._mt = mt
        self._tts = tts
        self._router = router
        self._usage_logger = usage_logger
        self._session_id = session_id
        self._stop_event = stop_event
        self._debug_audio_dir = Path(debug_audio_dir) / date_folder(session_id) if debug_audio_dir else None
        self._debug_counter = 0
        self._tts_config = tts_config  # None = natural speed, no pace tracking
        self._pace_wps: float | None = None  # preacher's pace, smoothed over turns
        self._last_speed: float | None = None  # previous segment's speed, for MAX_SPEED_STEP

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event.words_per_s is not None:
                # Smoothed so one hurried phrase does not jerk the voice faster
                # for the sentence after it; a sustained change still gets
                # through within three or four turns.
                self._pace_wps = event.words_per_s if self._pace_wps is None else (
                    PACE_SMOOTHING * event.words_per_s + (1 - PACE_SMOOTHING) * self._pace_wps
                )
            # One id for the whole turn, shared by every sentence it was split
            # into. Splitting exists so the first sentence can start playing
            # while the rest is still translating — but the turn is the unit of
            # *meaning*, so it has to be the unit the output buffer skips or
            # keeps. Giving each sentence its own id let the buffer drop the
            # middle of a single thought (measured 2026-09-06, 40% loss).
            turn_uid = self._router.allocate_utterance_id()
            for segment in split_for_streaming(event.text):
                if self._stop_event.is_set():
                    break
                self._handle_segment(segment, event.language_code, turn_uid, event.received_at)

    def _handle_segment(self, text: str, source_language: str, turn_uid: int, source_time: float) -> None:
        """One sentence-sized piece: translate it, then stream the synthesis
        straight into the output channel instead of waiting for the whole file.

        Streaming is the difference between the listener waiting for the *end*
        of synthesis and waiting for its *start*: measured against Cartesia,
        0.19s to first audio vs 1.07s for the finished blob, and on the
        2026-09-06 service the one-shot call hit 7.6s on the long turns.
        """
        try:
            t0 = time.monotonic()
            translated = self._mt.translate(text, source_language, self._lang.code)
            self._usage_logger.log(
                self._session_id, self._lang.code, "mt",
                duration_s=time.monotonic() - t0, chars=len(translated),
            )
            if not translated.strip():
                return

            t0 = time.monotonic()
            first_audio_s: float | None = None
            chunks: list[np.ndarray] = []
            speed = None
            if self._tts_config is not None:
                queued = self._router.queued_seconds(self._lang.output_channel)
                speed = choose_speed(self._tts_config, self._pace_wps, queued, previous=self._last_speed)
                self._last_speed = speed
            stream = self._tts.synthesize_stream(translated, self._lang.voice_id, self._lang.code, speed=speed)
            for chunk in trim_silence_stream(stream):
                if not len(chunk):
                    continue
                if first_audio_s is None:
                    first_audio_s = time.monotonic() - t0
                self._router.push_output(
                    self._lang.output_channel, chunk, utterance_id=turn_uid, source_time=source_time,
                )
                if self._debug_audio_dir is not None:
                    chunks.append(chunk)
            note = f"first_audio={first_audio_s:.2f}s" if first_audio_s is not None else "no audio"
            if speed is not None:
                note += f" speed={speed:.2f}"
            self._usage_logger.log(
                self._session_id, self._lang.code, "tts",
                duration_s=time.monotonic() - t0, chars=len(translated), note=note,
            )

            if self._debug_audio_dir is not None and chunks:
                self._debug_counter += 1
                wav_path = self._debug_audio_dir / (
                    f"{self._session_id}-{self._lang.code}-{self._debug_counter:03d}.wav"
                )
                _write_debug_wav(wav_path, np.concatenate(chunks), self._router.samplerate)
                print(f"[{self._lang.code}] wrote {wav_path} ({translated!r})")
        except Exception as exc:  # noqa: BLE001 — report §09: silence, never a hang
            self._usage_logger.log(
                self._session_id, self._lang.code, "error", note=str(exc)[:200],
            )
            print(f"[{self._lang.code}] pipeline error, dropping this utterance: {exc}")


class _PassthroughStage(threading.Thread):
    """No STT/MT/TTS at all — the raw-I/O smoke test."""

    def __init__(self, router: AudioRouter, languages: list[LanguageConfig], stop_event: threading.Event):
        super().__init__(name="passthrough", daemon=True)
        self._router = router
        self._languages = languages
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._router.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            for lang in self._languages:
                self._router.push_output(lang.output_channel, chunk)


class Pipeline:
    """Owns every worker thread for one running service."""

    def __init__(
        self,
        config: AppConfig,
        router: AudioRouter,
        usage_logger: UsageLogger,
        stt: STTProvider | None = None,
        mt_by_language: dict[str, MTProvider] | None = None,
        tts_by_language: dict[str, TTSProvider] | None = None,
        debug_audio_dir: str | None = None,
        session_id: str | None = None,
    ):
        self._config = config
        self._router = router
        self._usage_logger = usage_logger
        self._session_id = session_id or usage_logger.new_session_id()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stt = stt
        self._tts_providers = list((tts_by_language or {}).values())  # closed in stop()

        if config.pipeline.mode == "passthrough":
            self._threads.append(_PassthroughStage(router, config.languages, self._stop_event))
            return

        if stt is None:
            raise ValueError("pipeline.mode is not 'passthrough' — an STT provider instance is required")

        fanout_queues: list[queue.Queue[TranscriptEvent]] = []
        for lang in config.languages:
            q: "queue.Queue[TranscriptEvent]" = queue.Queue(maxsize=50)
            fanout_queues.append(q)
            mt = (mt_by_language or {}).get(lang.code)
            tts = (tts_by_language or {}).get(lang.code)
            if mt is None or tts is None:
                raise ValueError(f"missing mt/tts provider for language {lang.code!r}")
            self._threads.append(
                _LanguageStage(
                    lang, q, mt, tts, router, usage_logger, self._session_id, self._stop_event,
                    debug_audio_dir=debug_audio_dir, tts_config=config.tts,
                )
            )

        self._threads.insert(
            0, _STTStage(router, stt, fanout_queues, usage_logger, self._session_id, self._stop_event)
        )

    @property
    def stt_error(self) -> str | None:
        """Non-None once the STT stream is dead (connection dropped mid-service).

        The pipeline keeps running — audio still flows, the language threads are
        still alive — but no transcript will ever arrive again, so a caller with
        a UI (menubar_app) must say so instead of showing a green "running".
        Providers without the attribute (the mock) simply never report an error.
        """
        return getattr(self._stt, "fatal_error", None)

    @property
    def stt_lag_s(self) -> float | None:
        """How far recognition is running behind the speaker, as of the last
        transcript. A slow line does not break anything visibly — translation
        keeps coming, just of what was said a minute ago (2026-09-11: 90s,
        found only by aligning recordings afterwards). None until the first
        transcript, and for providers that do not measure it (the mock)."""
        return getattr(self._stt, "last_lag_s", None)

    def start(self) -> None:
        for t in self._threads:
            t.start()
        print(f"[pipeline] session {self._session_id} started, mode={self._config.pipeline.mode}, "
              f"{len(self._config.languages)} language(s)")

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        # AssemblyAI bills by websocket connection time, not by how much audio you
        # sent through it, so leaving the stream open after ⏹ keeps the meter
        # running for as long as the app sits in the menu bar. STTProvider.close()
        # has been part of the interface since Milestone 1 and nothing ever called
        # it (found 2026-08-23) — a stopped service was still being charged.
        if self._stt is not None:
            try:
                self._stt.close()
            except Exception as exc:  # noqa: BLE001 — shutdown must never raise
                print(f"[pipeline] STT close failed (the connection drops anyway): {exc}")
        # Continuous-context TTS holds a WebSocket per language; without this
        # every ⏹/▶️ in the menu bar would leave another one open.
        for tts in self._tts_providers:
            try:
                tts.close()
            except Exception as exc:  # noqa: BLE001 — shutdown must never raise
                print(f"[pipeline] TTS close failed: {exc}")
