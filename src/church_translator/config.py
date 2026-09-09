"""Config loading and validation for church-translator.

See config.example.yaml for the shape of the file. Kept as plain dataclasses
(no pydantic) so Milestone 1 has zero extra dependencies beyond audio I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    # Substring match against `list-devices` output. Kept separate because on
    # some interfaces (consumer USB soundbars especially) macOS exposes input
    # and output as two distinct CoreAudio devices even though it's one box —
    # `output_device` falls back to `input_device` when not set, which is
    # correct for real duplex interfaces (Focusrite/MOTU/RME etc).
    input_device: str | None = None
    output_device: str | None = None
    samplerate: int = 48000
    blocksize: int = 512  # frames/callback; 512 @ 48kHz ~= 10.7ms
    input_channel: int = 0  # 0-indexed channel on the input side carrying the mixer feed
    # How far behind the speaker a listener is allowed to fall before whole
    # sentences start getting skipped. See AudioRouter.push_output — without a
    # cap the lag grows for the entire service and never comes back.
    #
    # This is the operator's main trade-off knob: lower = more current but more
    # sentences skipped, higher = fewer skips but further behind. It was 4.0,
    # chosen back when a drop also decapitated the sentence being played, so it
    # had to be tight. Now that the playing sentence always finishes and only
    # stale queued ones are skipped, a tight cap just throws away translation
    # that would have been heard: on the 2026-09-06 service 55% of the
    # synthesized Russian was dropped. 10s keeps the listener within one
    # sentence of the speaker while skipping far less.
    max_backlog_s: float = 10.0

    # Catch-up playback. At 95% channel load (measured 2026-09-06: 98.4s of
    # Russian for 104s of speech) there is almost no idle time, so any lag the
    # channel picks up is permanent — skipping whole thoughts was the only way
    # back, which is why the listener heard 25% of the service go missing and
    # the delay still walked from 10s to 30s.
    #
    # Speeding playback up buys time without losing words, but it shifts pitch —
    # on a voice cloned from the pastor that is immediately audible and was
    # rejected (2026-09-06). OFF by default. The headroom is bought instead where
    # it costs nothing: a shorter interpretation (claude_mt.py) and trimmed
    # silence between clauses (pipeline.py). Left as an emergency knob only.
    catchup_start_s: float = 2.0     # start compressing once this far behind
    max_playback_rate: float = 1.0   # 1.0 = off


@dataclass
class LanguageConfig:
    name: str
    code: str
    output_channel: int
    voice_id: str = ""  # locked TTS voice id for this language (Milestone 2+)


@dataclass
class STTConfig:
    """AssemblyAI turn-endpointing. These decide how long the booth waits before
    a sentence even reaches translation, so they are the single biggest lever on
    perceived delay.

    The SDK's defaults are tuned for phone-call turn-taking, where the other
    party stops talking and waits. A preacher does not: measured on the
    2026-09-06 service, AssemblyAI held single turns of up to 921 characters —
    about a minute of speech with nothing sent downstream. Lower thresholds cut
    a turn at natural clause pauses instead, which is what live interpretation
    actually wants.
    """

    # 0-1; lower = end the turn on weaker evidence the speaker is done.
    end_of_turn_confidence_threshold: float = 0.4
    # Silence (ms) that closes a turn once the model is confident it ended.
    min_turn_silence_ms: int = 400
    # Hard ceiling (ms) on silence before a turn is closed regardless.
    max_turn_silence_ms: int = 1000

    # Send finished clauses to translation while the speaker is still talking.
    # Without this the delay can never be shorter than the speaker's longest
    # unbroken sentence — a 457-character turn on 2026-09-06 meant ~30s of delay
    # no matter how fast the rest of the pipeline was. Set False to go back to
    # translating only whole turns (better context, much worse delay).
    partial_emit: bool = True
    partial_min_words: int = 8    # never cut a fragment shorter than this
    partial_max_words: int = 25   # speaker never pauses? cut anyway at this many
    partial_gap_ms: int = 250     # a gap this long between words counts as a pause


@dataclass
class TTSConfig:
    """Cartesia synthesis settings.

    `speed` is exposed because Russian and Ukrainian render longer than the
    English they came from (107% channel load measured 2026-08-23), so shorter
    output would directly buy back lag. Measured 2026-09-06 on the cloned voice
    with sonic-latest, though, "fast" did NOT shorten anything — 9.1-9.8s vs
    8.9-9.0s for "normal" across repeats, i.e. noise. Left configurable in case
    a future model honours it; do not count on it as a latency fix.
    """

    speed: str = "normal"  # "slow" | "normal" | "fast"


@dataclass
class PipelineConfig:
    mode: str = "mock"  # "mock" | "real"
    stt_provider: str = "assemblyai"
    mt_provider: str = "claude"
    tts_provider: str = "cartesia"


@dataclass
class LoggingConfig:
    usage_log_path: str = "./logs/usage.csv"
    # When set, every synthesized utterance is also written as a .wav file
    # here — lets you verify *what audio was actually produced* by playing
    # it back on the Mac itself, independent of whether a transmitter/
    # receiver is hooked up yet. None = off (no extra disk writes).
    debug_audio_dir: str | None = None
    # When set, one continuous .wav per language for the whole session —
    # gaps included, exactly as it would be heard live. This is the "save
    # the sermon translation as one file" output, separate from the
    # per-utterance debug_audio_dir dump above.
    recordings_dir: str | None = None


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    # What the speaker is expected to say — passed to STT as its candidate
    # language_codes. Deliberately separate from `languages` below: source
    # and target are not the same set (report §04.2 case: English in, Russian
    # + Ukrainian out — STT never needs to recognize Russian or Ukrainian at
    # all). Empty list = full open language_detection, no candidate hint.
    source_languages: list[str] = field(default_factory=lambda: ["en"])
    languages: list[LanguageConfig] = field(default_factory=list)  # output/target languages
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if not self.languages:
            raise ValueError("config.languages is empty — need at least one output language")
        seen_channels = set()
        for lang in self.languages:
            if lang.output_channel in seen_channels:
                raise ValueError(f"output_channel {lang.output_channel} used by more than one language")
            seen_channels.add(lang.output_channel)
        if self.pipeline.mode not in ("mock", "real"):
            raise ValueError(f"pipeline.mode must be 'mock' or 'real', got {self.pipeline.mode!r}")
        if not 1.0 <= self.audio.max_playback_rate <= 1.5:
            raise ValueError(f"audio.max_playback_rate must be 1.0-1.5, got {self.audio.max_playback_rate}")
        if self.tts.speed not in ("slow", "normal", "fast"):
            raise ValueError(f"tts.speed must be slow/normal/fast, got {self.tts.speed!r}")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    audio = AudioConfig(**raw.get("audio", {}))
    source_languages = raw.get("source_languages", ["en"])
    languages = [LanguageConfig(**lang) for lang in raw.get("languages", [])]
    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    stt = STTConfig(**raw.get("stt", {}))
    tts = TTSConfig(**raw.get("tts", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    cfg = AppConfig(
        audio=audio, source_languages=source_languages, languages=languages,
        pipeline=pipeline, stt=stt, tts=tts, logging=logging_cfg,
    )
    cfg.validate()
    return cfg
