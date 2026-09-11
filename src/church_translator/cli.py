"""Entry point: `church-translator list-devices` / `church-translator run --config config.yaml`."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import sounddevice as sd

from .audio_io import list_devices, resolve_device
from .config import load_config, save_config
from .pipeline import _write_debug_wav
from .session import LiveSession, build_pipeline
from .usage_log import UsageLogger

try:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))  # run/record-sample/clone-voice all need this without a manual `source .env`
except ImportError:
    pass


def _cmd_list_devices(_args: argparse.Namespace) -> None:
    print(list_devices())
    print(
        "\nPick the interface your mixer feed is plugged into, then set its name\n"
        "(or a distinctive substring) as `audio.input_device` / `audio.output_device`\n"
        "in your config.yaml — the same name in both is fine for a duplex interface."
    )


def _cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    try:
        session = LiveSession(config)
        session.start()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if session.recordings_dir:
        print(f"Recording full session to {session.recordings_dir}/{session.session_id}-<lang>.wav (+ -source.wav)")
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
        report = session.stop()
        if any(report.underruns.values()):
            print(f"[audio] output underruns per language (silence padding, not a crash): {report.underruns}")
        if any(report.dropped.values()):
            # The operator's tuning signal: anything above the odd one or two
            # means max_backlog_s is too tight for how fast the voice speaks.
            print(f"[audio] whole turns skipped per language (lag > {config.audio.max_backlog_s:.0f}s): {report.dropped}")
        print("Stopped.")


def _cmd_record_sample(args: argparse.Namespace) -> None:
    """Capture a clean clip for Cartesia voice cloning (report §04.3/§07):
    5-10s is enough, and a clean isolated mic beats the usual "copy of Main
    L/R" input — that mix carries music the clone would pick up too."""
    config = load_config(args.config)
    input_device_index = resolve_device(config.audio.input_device, kind="input")
    samplerate = config.audio.samplerate
    channel = config.audio.input_channel

    print(f"Recording {args.seconds}s from channel {channel} in 2s... speak clearly, no music in the background.")
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
    print(f"Saved {out_path} ({args.seconds}s, peak={peak:.3f}). Listen to it before cloning — "
          f"{'signal present' if peak > 0.02 else 'SUSPICIOUSLY QUIET, check the input'}.")


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
        save_config(config, config_path)
        print(f"Assigned {voice.id} to: {', '.join(l.code for l in matched)} in {config_path}")


def _cmd_replay(args: argparse.Namespace) -> None:
    """Run a WAV through the whole pipeline with no sound card attached.

    The point is to stop finding output-path bugs during a service. Everything
    real runs — AssemblyAI, Claude, Cartesia, and AudioRouter's own callback with
    its skip rules — only the Scarlett is replaced by a file at each end.
    """
    from .replay import OfflineRouter, read_wav_mono, replay

    config = load_config(args.config)
    source = read_wav_mono(Path(args.input), config.audio.samplerate)

    router = OfflineRouter(
        input_device=None,
        output_device=None,
        samplerate=config.audio.samplerate,
        blocksize=config.audio.blocksize,
        input_channel=config.audio.input_channel,
        output_channels=[lang.output_channel for lang in config.languages],
        max_backlog_s=config.audio.max_backlog_s,
        catchup_start_s=config.audio.catchup_start_s,
        max_playback_rate=config.audio.max_playback_rate,
    )
    usage_logger = UsageLogger(config.logging.usage_log_path)
    session_id = usage_logger.new_session_id().replace("svc-", "replay-")

    if args.voice:
        for lang in config.languages:
            lang.voice_id = args.voice  # A/B a stock voice against the clone on identical input
        print(f"voice override: every language uses {args.voice}")

    pipeline = build_pipeline(config, router, usage_logger, session_id)

    print(f"replaying {args.input} ({len(source) / config.audio.samplerate:.0f}s) at wall-clock pace, "
          f"session {session_id}")
    pipeline.start()
    try:
        written = replay(config, router, source, Path(args.out_dir), session_id, drain_s=args.drain)
    finally:
        pipeline.stop()
        router.stop()

    src_s = len(source) / config.audio.samplerate
    print()
    for code, info in written.items():
        # `voiced` is what a listener would actually have heard; the gap between
        # that and the source length is the translation that never made it out.
        print(f"  [{code}] {info['path']}  {info['voiced_s']:.0f}s heard of {src_s:.0f}s spoken "
              f"({100 * info['voiced_s'] / src_s:.0f}% channel load)")
    dropped = router.dropped_report()
    if any(dropped.values()):
        print(f"  whole turns skipped: {dropped}")
    underruns = router.underrun_report()
    print(f"  underruns (silence in pauses is normal): {underruns}")
    print("\nListen to the files above — exactly what a listener would have heard.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="church-translator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-devices", help="list audio devices and their channel counts").set_defaults(
        func=_cmd_list_devices
    )

    run_parser = sub.add_parser("run", help="run the translation pipeline")
    run_parser.add_argument("--config", required=True, help="path to config.yaml")
    run_parser.set_defaults(func=_cmd_run)

    replay_parser = sub.add_parser(
        "replay", help="run a wav file through the pipeline with no sound card, write the channel outputs"
    )
    replay_parser.add_argument("--config", required=True, help="path to config.yaml")
    replay_parser.add_argument("--input", required=True, help="16-bit wav at the config samplerate")
    replay_parser.add_argument("--out-dir", default="replay-out", help="where to write one wav per language")
    replay_parser.add_argument("--voice", help="override every language's voice_id (A/B a stock voice vs the clone)")
    replay_parser.add_argument("--drain", type=float, default=45.0,
                               help="seconds to keep playing after the input ends, so the tail is not cut")
    replay_parser.set_defaults(func=_cmd_replay)

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
