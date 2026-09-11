# church-translator

Live speech translation for church services: audio comes in from the mixer through an audio
interface, gets processed, and each language goes out on its own physical output channel → FM
transmitter → a receiver with an earpiece in the listener's hand. No website, no app, nothing
for the listener to hold but a receiver.

*(Русская версия — [README.ru.md](README.ru.md). Инструкция оператора — [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md).)*

## Three stages, three milestones

| Mode (`pipeline.mode`) | What it exercises | API keys needed? |
|---|---|---|
| `passthrough` | Capture and per-channel output, routing, behavior on dropout — the audio path itself, no translation | no |
| `mock` | The shape of the STT→MT→TTS pipeline (streams, timing, cost log) against fake providers | no |
| `real` | Full translation: AssemblyAI (STT) → Claude (MT) → Cartesia (TTS) | yes |

Milestone 1 is `passthrough` and `mock` — both work today and need no keys. Milestone 2 is
turning on `real` with live APIs. Milestone 3 is on-the-fly language detection and voice cloning
on top of the same architecture.

## Quick start

```bash
uv sync --extra providers               # installs dependencies into .venv
uv run church-translator list-devices   # see what the system sees
cp config.example.yaml config.yaml      # set device/channels for your hardware
uv run church-translator run --config config.yaml
```

**Always pass `--extra providers`, even if you're only running `mock` today.** A bare `uv sync`
is an exact-synchronization command: anything outside the default set gets removed. On the booth
machine that wipes assemblyai, anthropic, cartesia and python-dotenv (confirmed with
`uv sync --dry-run`: "Would uninstall 21 packages"), and `mode: real` stops starting. `uv run` is
safe — it installs what's missing without removing anything else.

In `mode: mock`, any two-output device (ordinary headphones will do) gives you an immediate
result: one language plays a tone on channel 0, the other a different tone on channel 1. That's
the entire smoke test at this stage — translation isn't involved yet, you're only confirming that
each language really lands on its own physical channel.

## Connecting a real audio interface

1. Run `list-devices` and find your interface. **If the same box shows up as two entries** (one
   with `in=N out=0`, another `in=0 out=N`), that's normal for some USB devices — macOS exposes
   input and output as separate CoreAudio devices. That's why `input_device` and `output_device`
   are separate config fields: you can put the same substring in both, and the code will pick the
   half that actually has the channels.
2. `audio.input_channel` — which input channel carries the signal from the mixer's AUX / direct
   out.
3. Each language gets its own `output_channel`, which then goes physically into its own FM
   transmitter.

### source_languages vs. languages

Two different lists in the config; do not conflate them. `source_languages` is what the speaker
actually speaks (passed to STT as a candidate list). `languages` is what you translate *into* on
the output side (MT + TTS + channel). For "speaker talks in English, listeners get Russian and
Ukrainian" that's `source_languages: [en]` and `languages: [ru, uk]` — genuinely different sets.

Mixing them breaks more than the logic: a live test on 2026-08-19 confirmed that AssemblyAI
streaming will not accept `uk` as a source language at all (error 3006), even though Ukrainian
works fine as an output language in Cartesia TTS.

### The real chain: Behringer X32 → Focusrite Scarlett 2i2 → 2 FM transmitters

A ready-made config for this hardware is `config.church.example.yaml`. Its limits in short (the
comments inside the file go deeper):

- **2i2 = 2 outputs = a ceiling of 2 languages.** Beyond that you need either an Aggregate Device
  built from two interfaces (Audio MIDI Setup) or a larger interface.
- **The input today is a copy of Main L/R from the X32** — the whole front-of-house mix, not an
  isolated speaker mic. It works, but for STT accuracy a dedicated AUX bus carrying only speech
  is the better setup.

## Milestone 2: real providers

```bash
uv sync --extra providers
cp .env.example .env   # fill in ASSEMBLYAI_API_KEY / ANTHROPIC_API_KEY / CARTESIA_API_KEY
```

All three providers were checked against the actually installed SDKs — not against documentation
from memory — and exercised with real keys:

- **Cartesia (`cartesia_tts.py`)** — confirmed with a live call, returns real audio.
- **AssemblyAI (`assemblyai_stt.py`)** — connection and auth work; a full speech run happens when
  you test on live audio.
- **Claude (`claude_mt.py`)** — verified end to end: a real phrase, transcribed with noise by
  AssemblyAI, came back translated correctly and sensibly *despite* errors in the transcript. LLM
  translation survives a noisy input far better than a rule-based translator would.

**Field finding, 2026-08-19:** fully open language detection (`language_detection=True`, no hint)
once returned a third, random language on ambiguous audio. With `language_codes=["ru","en"]` — the
set the church actually expects — it did not recur. `cli.py` now always passes the configured
languages to STT as a candidate list instead of leaving detection wide open.

**Field finding, 2026-09-11:** a live test fell 90s behind the speaker and stayed there, while
every stage after recognition stayed fast (MT ~1.7s, first TTS audio ~0.6s). Replaying the same
recording offline, recognition never lagged more than 1.6s — the line, not the pipeline. The booth
runs on T-Mobile home internet: plenty of bandwidth (16/36 Mbit/s) but ~0.6-1.1s of latency under
load, and the SDK's send queue is unbounded, so a stall turns into permanent delay. What changed:

- Audio goes to AssemblyAI at 16 kHz (its models' native rate), 256 instead of 768 kbit/s.
- Unsent audio older than 3s is thrown away (`MAX_SEND_BACKLOG_S`) — losing a few seconds of
  speech beats translating the past for the rest of the service.
- Cartesia returns 16-bit PCM instead of float32: half the download, same sound.
- Connect timeout 10s instead of the SDK's 1s — the TLS handshake alone measured up to 2.1s.
- `usage.csv` STT rows carry `lag=` (how far behind the speaker) and `backlog=` (unsent audio),
  and the menu bar turns 🐢 when the lag passes `audio.max_backlog_s`.

The same test showed `partial_emit` had never fired: AssemblyAI marked no word `word_is_final`
before its turn closed, so sermon turns went out whole (up to 86 words, ~37s). Long turns are now
closed on the server with `ForceEndpoint` once they hold `partial_min_words` and the speaker
pauses, or reach `partial_max_words`.

Later the same day, with the network out of the way, the remaining wait was the channel itself:
Russian speech runs longer than the English, the channel was busy 75% of the time, and new
clauses queued behind the one still playing. Also changed:

- **Dynamic speed.** TTS moved to `sonic-3`, whose numeric `generation_config.speed` measurably
  shortens the cloned voice without shifting pitch. Speed is picked per segment between
  `tts.speed_min` and `tts.speed_max` (1.0-1.3) from the preacher's pace (`pace_normal_wps` /
  `pace_fast_wps`, from AssemblyAI word timings) and from audio already queued in the channel,
  moving at most 0.1 per segment. STT rows log `wps=`, TTS rows `speed=`.
- **Continuous intonation.** Clauses go into one Cartesia WebSocket context in turn
  (`tts.continuous_context`), so the voice carries the intonation on instead of reading every
  clause as a finished sentence; preferred by ear on an A/B, same ~0.2s to first audio. A new
  context starts on a speed change or after a 4s pause; the socket is reopened after 20s idle,
  and a failed clause falls back to one request per clause.
- **No more assistant replies.** On short clauses ("listen", "give me wisdom") Claude sometimes
  answered the booth in English ("I'm ready to interpret…") and it went out in the cloned voice.
  The clause is now fenced in `<utterance>` tags with an explicit "never addressed to you", and a
  reply that is mostly Latin letters for a Cyrillic target is dropped (14/60 bad on the old prompt,
  0/174 on the new one). A clause cut mid-sentence ends with a comma, not a full stop, so the voice
  does not drop its pitch in the middle of a sentence.

If any provider SDK changes in the future, the contract used by `pipeline.py`
(`STTProvider` / `MTProvider` / `TTSProvider` in `providers/base.py`) doesn't need to move — only
the code inside that one provider file does.

Whatever you change in those three files, `voice_id` must stay **the same for a given language
for the whole service**. That constraint is the architectural fix for the "the voice keeps
drifting" complaint: if the TTS call inside `synthesize()` doesn't get the same `voice_id` every
time, the problem comes straight back.

## Cost accounting

Every STT/MT/TTS call appends a row to `logs/usage.csv` (component, language, duration/characters,
timestamp). Nothing to enable — the logging already lives inside `pipeline.py`.

## Testing changes without a service (`replay`)

All three output bugs — cutting off mid-word, losing the middle of a thought, and drift that grows
over time — were found **during an actual service**, because there was no other way to see them.
`replay` runs a WAV file through the entire pipeline with no audio interface attached and writes
out what a listener would have heard.

```bash
church-translator replay --config config.yaml --input sermon.wav --out-dir replay-out
```

Everything real is in the loop — AssemblyAI, Claude, Cartesia, and `AudioRouter` with its own skip
rules; the file replaces only the Scarlett on input and output. The run happens in real time
rather than as fast as possible, deliberately: AssemblyAI's endpointing and the buffer staleness
rule are both measured in seconds, so a fast-forwarded run would show delays that won't exist
during a service.

The report tells you how many seconds of translation actually played out of how many were spoken
(channel utilization), how many whole utterances were skipped, and how many underruns occurred.

Input must be 16-bit WAV at the sample rate from your config. If it isn't, convert:

```bash
ffmpeg -i sermon.m4a -ar 48000 -ac 1 -c:a pcm_s16le sermon.wav
```

`--voice <id>` forces one voice across all languages — that's how you compare a cloned voice
against a stock one on identical input. Cartesia's generation variance runs up to 10%, so
comparing on a clip shorter than a minute is pointless.

Replay sessions are written to `logs/` with a `replay-` prefix so they don't get mixed up with
real services (`svc-`).

## Checklist for the first test in the church

**Empty room, no congregation.** The first run of the live physical chain should not be the same
run as the first time in front of an audience.

1. Connect the Scarlett 2i2 to the Mac over USB.
2. `uv run church-translator list-devices` — find the interface's real name in the list (it may
   not match "Scarlett 2i2" character for character).
3. `cp config.church.example.yaml config.yaml`, then confirm `input_device` / `output_device`
   match what `list-devices` actually reported.
4. Route X32 AUX / direct out into the Scarlett's L/R inputs; Scarlett outputs into the two FM
   transmitters (L → transmitter 1, R → transmitter 2).
5. **First run is `pipeline.mode: mock`, not `real`.** Take two receivers, listen on headphones,
   speak or play audio through the X32 — receiver 1 should carry one tone, receiver 2 the other.
   This tests pure physics (channel routing), with no APIs and no risk.
6. Only once routing is confirmed, switch to `pipeline.mode: real` and repeat: say a phrase in
   English into the mic; after roughly 1–2 seconds receiver 1 should play the Russian translation
   and receiver 2 the Ukrainian.
7. Watch the terminal for `input overflow` / `output underrun`. If they show up regularly (not
   just once), that's a signal to adjust `blocksize` in the config rather than ignore them.

## Booth app (menu bar)

```bash
uv run church-translator-app
```

A 🎙️ icon in the menu bar — not in the Dock, because this is a booth utility, not a windowed app.
It does everything `run` does in the terminal but without one: the same `AudioRouter` / `Pipeline`
are invoked directly in-process, so start/stop is just a click, with none of the signal-handling
problems that came up twice when going through `uv run` in a terminal.

- **▶️/⏹** — start/stop translation.
- **Test mode (mock)** — toggles `pipeline.mode` between mock and real; disabled while translation
  is running.
- **Input/output device, input channel, language channels** — the real device list from this Mac
  with a checkmark on the current selection; clicking changes it and immediately saves
  `config.yaml`. Assign two languages to one channel and the code swaps them rather than silently
  producing a collision.
- **Open logs / Open debug audio** — opens Finder on the right folder.

Note that the app rewrites `config.yaml`, so the comments from `config.church.example.yaml`
disappear after the first click. That's a deliberate trade-off: once channels are configured from
the app, reading the raw YAML stops being necessary.

> The menu bar labels are in English; `OPERATOR_GUIDE.md` stays in Russian — it is written for
> the volunteer running the booth during a service and quotes the English labels as they appear.

## Recording a voice sample for cloning

```bash
uv run church-translator record-sample --config config.yaml --seconds 8 --out pastor_sample.wav
```

Records from the already-configured input channel. For a clone, cleanliness matters more than
convenience: if all you have is a "copy of Main L/R" input with music in it, it's worth asking the
pastor to say a couple of sentences into a separate clean mic for those 8 seconds rather than
taking a slice of the full mix. Cartesia's `voices.clone()` works from clips as short as 5
seconds; the resulting `voice_id` then goes into `languages[].voice_id` in the config like any
other voice — cross-language delivery works out of the box (a clone made from an English clip
speaks Russian and Ukrainian without complaint).

## What's next

Ideas measured or discussed during the 2026-09-11 tests and deliberately left for later:

- **Shorter clauses on unbroken speech.** When the preacher talks without pauses, every turn is
  closed by the word limit, not by a pause: 150-170 characters, ~11s of speech before translation
  can start. `stt.partial_max_words` 25 -> 15 and `stt.partial_gap_ms` 250 -> 200 would roughly
  halve that, at the cost of less context per translation. Left at 25 for now.
- **A better clone.** Today's clone is 8 seconds of English. 20-30 seconds of clean, expressive
  speech should sound better; a Russian-language sample (if the pastor speaks Russian) would give
  native Russian intonation instead of English prosody carried over.
- **Tighter translation.** Russian came out at 93-105% of the English length on short clauses, and
  the channel was busy 57-77% of the time. A more compressed interpretation buys headroom.
- **Wired network for the booth.** The T-Mobile line has bandwidth to spare but 0.6-1.1s of latency
  under load; Ethernet (or a separate line) removes the one failure that produced a 90s lag.
- **Log the text, not just its length.** `usage.csv` records character counts only, so a dropped or
  odd translation can be read only in the terminal of that session. Logging source and translation
  per segment (and `[mt] dropped …` events) would make every test reviewable afterwards.
- **Calibrate the pace thresholds on a full service.** `pace_normal_wps`/`pace_fast_wps` (2.0/2.8)
  come from ~2 minutes of one preacher.
- **Run on a Raspberry Pi with Telegram control.** Same quality and latency (all heavy work is in
  the cloud); needs `rumps` made macOS-only in `pyproject.toml`, a Telegram bot in place of the menu
  bar (start/stop/status, alerts on 🐢/⚠️), a systemd service, a check of the Scarlett under ALSA,
  wired Ethernet, and preferably a dedicated Pi for the church.
- Code-switching — an STT model with native on-the-fly language detection instead of a fixed
  `source_language`.
