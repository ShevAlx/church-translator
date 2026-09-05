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
    # utterances start getting skipped. See AudioRouter.push_output — without a
    # cap the lag grows for the entire service and never comes back.
    max_backlog_s: float = 4.0


@dataclass
class LanguageConfig:
    name: str
    code: str
    output_channel: int
    voice_id: str = ""  # locked TTS voice id for this language (Milestone 2+)


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


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    audio = AudioConfig(**raw.get("audio", {}))
    source_languages = raw.get("source_languages", ["en"])
    languages = [LanguageConfig(**lang) for lang in raw.get("languages", [])]
    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    cfg = AppConfig(
        audio=audio, source_languages=source_languages, languages=languages,
        pipeline=pipeline, logging=logging_cfg,
    )
    cfg.validate()
    return cfg
