"""Entry point: `church-translator list-devices` / `church-translator run --config config.yaml`."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import sounddevice as sd
import yaml

from .audio_io import AudioRouter, list_devices, resolve_device
from .config import load_config
from .pipeline import Pipeline, _write_debug_wav
from .providers.mock import MockMT, MockSTT, MockTTS
from .usage_log import UsageLogger, date_folder

try:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))  # run/record-sample/clone-voice all need this without a manual `source .env`
except ImportError:
    pass


def _cmd_list_devices(_args: argparse.Namespace) -> None:
    print(list_devices())
    print(
        "\nPick the interface your mixer feed is plugged into, then set its name\n"
        "(or a distinctive substring) as `audio.device` in your config.yaml."
    )


def _build_mock_pipeline(config, router, usage_logger, session_id) -> Pipeline:
    stt = MockSTT(samplerate=config.audio.samplerate)
    mt_by_lang = {lang.code: MockMT() for lang in config.languages}
    tts_by_lang = {
        lang.code: MockTTS(samplerate=config.audio.samplerate, base_hz=440.0 * (i + 1))
        for i, lang in enumerate(config.languages)
    }
    return Pipeline(
        config, router, usage_logger, stt=stt, mt_by_language=mt_by_lang, tts_by_language=tts_by_lang,
        debug_audio_dir=config.logging.debug_audio_dir, session_id=session_id,
    )


def _build_real_pipeline(config, router, usage_logger, session_id) -> Pipeline:
    # Milestone 2 — needs `pip install "church-translator[providers]"` plus
    # ASSEMBLYAI_API_KEY / ANTHROPIC_API_KEY / CARTESIA_API_KEY (see README).
    try:
        from .providers.assemblyai_stt import AssemblyAISTT
        from .providers.cartesia_tts import CartesiaTTS
        from .providers.claude_mt import ClaudeMT
    except ImportError as exc:
        raise SystemExit(
            "pipeline.mode is 'real' but the provider extras aren't installed.\n"
            "Run: uv sync --extra providers\n"
            f"(import error: {exc})"
        ) from exc

    # Constrain STT to what the speaker is expected to say (config.source_languages),
    # NOT to the output/target languages — those are what MT produces, never what
    # STT needs to recognize. Conflating the two breaks as soon as source and
    # target differ: live-tested 2026-08-19, AssemblyAI's streaming API flatly
    # rejects "uk" as a source candidate (code=3006) even though Cartesia
    # synthesizes Ukrainian output fine — source and target are different sets
    # with different constraints, on purpose.
    stt = AssemblyAISTT(
        samplerate=config.audio.samplerate,
        language_codes=config.source_languages or None,
    )
    mt_by_lang = {lang.code: ClaudeMT() for lang in config.languages}
    tts_by_lang = {lang.code: CartesiaTTS(samplerate=config.audio.samplerate) for lang in config.languages}
    return Pipeline(
        config, router, usage_logger, stt=stt, mt_by_language=mt_by_lang, tts_by_language=tts_by_lang,
        debug_audio_dir=config.logging.debug_audio_dir, session_id=session_id,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    input_device_index = resolve_device(config.audio.input_device, kind="input")
    output_source = config.audio.output_device or config.audio.input_device
    output_device_index = resolve_device(output_source, kind="output")

    router = AudioRouter(
        input_device=input_device_index,
        output_device=output_device_index,
        samplerate=config.audio.samplerate,
        blocksize=config.audio.blocksize,
        input_channel=config.audio.input_channel,
        output_channels=[lang.output_channel for lang in config.languages],
        max_backlog_s=config.audio.max_backlog_s,
    )
    usage_logger = UsageLogger(config.logging.usage_log_path)
    session_id = usage_logger.new_session_id()  # shared with recordings below, so filenames line up with usage.csv

    if config.logging.recordings_dir:
        recordings_dir = Path(config.logging.recordings_dir) / date_folder(session_id)
        for lang in config.languages:
            router.start_recording(lang.output_channel, recordings_dir / f"{session_id}-{lang.code}.wav")
        print(f"Recording full session to {recordings_dir}/{session_id}-<lang>.wav")

    if config.pipeline.mode == "passthrough":
        pipeline = Pipeline(config, router, usage_logger, session_id=session_id)
    elif config.pipeline.mode == "mock":
        pipeline = _build_mock_pipeline(config, router, usage_logger, session_id)
    elif config.pipeline.mode == "real":
        pipeline = _build_real_pipeline(config, router, usage_logger, session_id)
    else:
        raise SystemExit(f"unknown pipeline.mode: {config.pipeline.mode!r}")

    router.start()
    pipeline.start()
    print("Ctrl+C to stop.")

    stop = False

    def _handle_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not stop:
            time.sleep(1.0)
    finally:
        pipeline.stop()
        router.stop()
        underruns = router.underrun_report()
        if any(underruns.values()):
            print(f"[audio] output underruns per channel (silence padding, not a crash): {underruns}")
        print("Stopped.")


def _cmd_record_sample(args: argparse.Namespace) -> None:
    """Capture a clean clip for Cartesia voice cloning (report §04.3/§07):
    5-10s is enough, and a clean isolated mic beats the usual "copy of Main
    L/R" input — that mix carries music the clone would pick up too."""
    config = load_config(args.config)
    input_device_index = resolve_device(config.audio.input_device, kind="input")
    samplerate = config.audio.samplerate
    channel = config.audio.input_channel

    print(f"Recording {args.seconds}s from channel {channel} in 2s... говорите чисто, без музыки за спиной.")
    time.sleep(2)
    frames = int(samplerate * args.seconds)
    recording = sd.rec(
        frames, samplerate=samplerate, channels=channel + 1, device=input_device_index, dtype="float32"
    )
    sd.wait()
    mono = recording[:, channel]

    out_path = Path(args.out)
    _write_debug_wav(out_path, mono, samplerate)
    peak = float(abs(mono).max()) if len(mono) else 0.0
    print(f"Saved {out_path} ({args.seconds}s, peak={peak:.3f}). Прослушайте перед клонированием — "
          f"{'сигнал есть' if peak > 0.02 else 'ПОДОЗРИТЕЛЬНО ТИХО, проверьте вход'}.")


def _cmd_clone_voice(args: argparse.Namespace) -> None:
    """Report §04.3/§07: clone from a clip, then (optionally) wire the
    resulting voice_id straight into config.yaml for the languages given via
    --assign — cross-lingual by default, no separate clone per target language."""
    try:
        from cartesia import Cartesia
    except ImportError as exc:
        raise SystemExit("Needs the provider extras: uv sync --extra providers") from exc

    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"{clip_path} not found — record one first with `church-translator record-sample`")

    client = Cartesia(api_key=os.environ["CARTESIA_API_KEY"])
    print(f"Cloning from {clip_path} ({clip_path.stat().st_size / 1024:.0f} KB)...")
    voice = client.voices.clone(clip=clip_path, language=args.language, name=args.name)
    print(f"Cloned voice_id: {voice.id}")

    if args.assign:
        config_path = Path(args.config)
        config = load_config(config_path)
        targets = {c.strip() for c in args.assign.split(",")}
        matched = [lang for lang in config.languages if lang.code in targets]
        if not matched:
            print(f"WARNING: none of {targets} matched a language in {config_path} — voice_id not written anywhere.")
            return
        for lang in matched:
            lang.voice_id = voice.id
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "audio": vars(config.audio),
                    "source_languages": config.source_languages,
                    "languages": [vars(lang) for lang in config.languages],
                    "pipeline": vars(config.pipeline),
                    "logging": vars(config.logging),
                },
                f, allow_unicode=True, sort_keys=False,
            )
        print(f"Assigned {voice.id} to: {', '.join(l.code for l in matched)} in {config_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="church-translator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-devices", help="list audio devices and their channel counts").set_defaults(
        func=_cmd_list_devices
    )

    run_parser = sub.add_parser("run", help="run the translation pipeline")
    run_parser.add_argument("--config", required=True, help="path to config.yaml")
    run_parser.set_defaults(func=_cmd_run)

    sample_parser = sub.add_parser("record-sample", help="record a clean clip for voice cloning")
    sample_parser.add_argument("--config", required=True, help="path to config.yaml")
    sample_parser.add_argument("--seconds", type=float, default=8.0)
    sample_parser.add_argument("--out", default="pastor_sample.wav")
    sample_parser.set_defaults(func=_cmd_record_sample)

    clone_parser = sub.add_parser("clone-voice", help="clone a voice from a clip (needs CARTESIA_API_KEY)")
    clone_parser.add_argument("--clip", required=True, help="path to a clean 5-10s wav/mp3 clip")
    clone_parser.add_argument("--name", required=True, help="label for the cloned voice in your Cartesia account")
    clone_parser.add_argument("--language", default="en", help="language spoken IN THE CLIP, e.g. en")
    clone_parser.add_argument("--assign", help="comma-separated language codes in config.yaml to point at this voice, e.g. ru,uk")
    clone_parser.add_argument("--config", default="config.yaml")
    clone_parser.set_defaults(func=_cmd_clone_voice)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
