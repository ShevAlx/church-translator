"""Config loading and validation for church-translator.

See config.example.yaml for the shape of the file. Kept as plain dataclasses
(no pydantic) so Milestone 1 has zero extra dependencies beyond audio I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    # Close long turns early (ForceEndpoint) so clauses reach translation while
    # the speaker is still talking. Without this the delay can never be shorter
    # than the speaker's longest unbroken sentence — a 457-character turn on
    # 2026-09-06 meant ~30s of delay no matter how fast the rest of the pipeline
    # was. Set False to translate only turns AssemblyAI ends on its own (better
    # context, much worse delay). See AssemblyAISTT._on_turn.
    partial_emit: bool = True
    partial_min_words: int = 8    # never close a turn shorter than this
    partial_max_words: int = 25   # speaker never pauses? close anyway at this many
    partial_gap_ms: int = 250     # a gap this long between words counts as a pause


@dataclass
class TTSConfig:
    """Cartesia synthesis settings.

    Russian and Ukrainian render longer than the English they came from, and
    that — not the network, not MT — is what made the listener wait: on the
    2026-09-11 test the channel was playing 75% of the time and 68% of new
    turns arrived while the previous one was still in the headphones.

    `speed` ("slow"/"normal"/"fast") is the pre-Sonic-3 knob and measured as a
    no-op on the clone (2026-09-06, again 2026-09-11). Sonic-3 has a numeric
    generation_config.speed instead, and it works on the cloned voice without
    shifting its pitch — same 34 sermon phrases, 2026-09-11, mean of 2 runs:
    sonic-latest 100%, sonic-3 at 1.0 90%, 1.2 87%, 1.3 85%, 1.4 76%.

    The speed is chosen per segment (pipeline.choose_speed): faster when the
    preacher speeds up, and faster when translation is piling up in the
    headphones regardless of pace.
    """

    model: str = "sonic-3"   # generation_config.speed only exists on sonic-3*
    # Clauses spoken in one continuous Cartesia context, so intonation carries
    # from one to the next instead of every clause sounding like a whole
    # sentence (see providers/cartesia_tts.py). False = one request per clause,
    # the pre-2026-09-11 behaviour — the switch to flip if the socket misbehaves.
    continuous_context: bool = True
    speed: str = "normal"    # legacy knob, used only by pre-sonic-3 models
    speed_min: float = 1.0   # normal pace, nothing queued
    speed_max: float = 1.3   # ceiling; Cartesia accepts 0.6-1.5. 1.4 was audibly rushed on the clone (2026-09-11)
    # Preacher's pace in words/s (from AssemblyAI word timings, smoothed).
    # At or below pace_normal_wps the voice runs at speed_min; at pace_fast_wps
    # and above, at speed_max; linear in between.
    # Calibrated on the 2026-09-11 test: this preacher averaged 1.9 words/s and
    # peaked at 2.6 (pauses inside a turn included), so the first guess of
    # 2.5/3.5 never let pace move the speed at all — only the queue did.
    pace_normal_wps: float = 2.0
    pace_fast_wps: float = 2.8


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
    # Per-date folders under recordings_dir / debug_audio_dir older than this
    # many days are deleted by the Telegram bot at boot and after every ⏹.
    # A 2-hour service is ~2 GB of WAV (source + one file per language), which
    # a Pi's SD card cannot hold forever. 0 = keep everything (the Mac default).
    keep_days: int = 0


@dataclass
class BotConfig:
    """How the Telegram bot (telegram_bot.py) runs an unattended booth.

    Only the bot reads this section. The menu bar and `run` have a person in
    front of them who sees ⚠️/🐢 and presses ⏹/▶️; on the Pi nobody does, so the
    bot does that same restart itself and messages what happened.
    """

    # Restart the session on a dead recognition link, a vanished sound card, or
    # a lag that will not clear — the operator guide's manual "⏹ then ▶️".
    auto_recover: bool = True
    max_recoveries: int = 5          # within recovery_window_min, then a human decides
    recovery_window_min: float = 15.0
    retry_after_s: float = 30.0      # a failed restart (no internet yet) is retried this often...
    retry_give_up_min: float = 10.0  # ...for this long, then the bot stops and says so
    lag_recover_after_s: float = 60.0  # 🐢 held this long -> restart
    silence_alert_s: float = 180.0   # mixer feed silent this long while running -> message; 0 = off
    # A forgotten ▶️ stops by itself: AssemblyAI bills the open connection by
    # the minute whether anyone is speaking or not. 0 = never.
    max_session_hours: float = 3.0


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
    bot: BotConfig = field(default_factory=BotConfig)

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
        if not 0.6 <= self.tts.speed_min <= self.tts.speed_max <= 1.5:
            raise ValueError(
                f"need 0.6 <= tts.speed_min <= tts.speed_max <= 1.5 (Cartesia's range), "
                f"got {self.tts.speed_min}-{self.tts.speed_max}"
            )
        if not 0 < self.tts.pace_normal_wps < self.tts.pace_fast_wps:
            raise ValueError("need 0 < tts.pace_normal_wps < tts.pace_fast_wps")
        if self.logging.keep_days < 0:
            raise ValueError("logging.keep_days must be >= 0 (0 = keep everything)")
        if self.bot.max_recoveries < 0 or self.bot.retry_after_s <= 0:
            raise ValueError("need bot.max_recoveries >= 0 and bot.retry_after_s > 0")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    audio = AudioConfig(**raw.get("audio", {}))
    source_languages = raw.get("source_languages", ["en"])
    languages = [LanguageConfig(**lang) for lang in raw.get("languages", [])]
    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    stt = STTConfig(**raw.get("stt", {}))
    tts = TTSConfig(**raw.get("tts", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))
    bot = BotConfig(**raw.get("bot", {}))

    cfg = AppConfig(
        audio=audio, source_languages=source_languages, languages=languages,
        pipeline=pipeline, stt=stt, tts=tts, logging=logging_cfg, bot=bot,
    )
    cfg.validate()
    return cfg


def save_config(config: AppConfig, path: str | Path) -> None:
    """Rewrite the whole file from `config`, every section included.

    Built from the dataclasses themselves rather than a hand-listed dict: both
    writers of config.yaml (menubar app, `clone-voice --assign`) used to list
    fields by hand, and clone-voice's list had no `stt`/`tts`, so cloning a
    voice silently reset the endpointing tuning to defaults. A field added to
    any dataclass above now round-trips without touching this function.

    Plain PyYAML, so hand-written comments from config.church.example.yaml do
    not survive — a deliberate trade once the file is managed by the app.
    """
    header = "# Managed by the church-translator menu-bar app — edits here get overwritten.\n"
    Path(path).write_text(
        header + yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
