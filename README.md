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

## Booth Raspberry Pi, controlled from Telegram

The permanent setup: a Raspberry Pi stays in the church with the Scarlett on USB and the network on
a LAN cable, and `church-translator-bot` (a systemd user service) replaces the menu bar. Buttons in
the Telegram chat start, stop and restart translation, and the bot reports what the menu bar icon
used to show. Unlike the menu bar it also recovers on its own: a dead recognition link, a vanished
sound card, or a lag that won't clear triggers the same "⏹ then ▶️" an operator would do, within the
limits of the `bot:` config section. It also sends a message when the mixer feed goes silent, stops
a session someone forgot to stop (AssemblyAI bills the open connection), and deletes recordings
older than `logging.keep_days`.

- Setup, church-day routine, messages and troubleshooting (in Russian): [PI_SETUP.md](PI_SETUP.md).
- Pi config template: `config.pi.example.yaml`. Push code from the Mac with `deploy/deploy.sh`.
- `session.py` builds and runs a session for all three front ends (`run`, the menu bar and the bot),
  so a fix to how a session starts or stops lands in every one of them.
- On Raspberry Pi OS with a desktop, PipeWire claims the Scarlett. `deploy/wireplumber-scarlett.conf`
  disables just that card in PipeWire so ALSA can open it directly.

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

### Applied after the 2026-09-14 live service

The service ran 1:15:54 in "combat mode" (real audience, real risk) and surfaced problems a
~2-minute bench test never could: delay well past `max_backlog_s` (10s) rather than the ~3s
expected, choppy delivery even with continuous-context TTS, and enough dropped phrases (ru 19,
uk 38 of 1403) that the congregation had no way to react when the preacher asked them to. Fixed
from that report, without another live test to re-measure against yet — watch the next service:

- **Shorter clauses on unbroken speech.** This was flagged after 2026-09-11 and deliberately left
  at the old values; 2026-09-14 was the full-service confirmation it needed. `stt.partial_max_words`
  25 -> 15 and `stt.partial_gap_ms` 250 -> 200 now ship as the default — a preacher who does not
  pause was hitting the word ceiling on most turns, at ~11s of speech before translation could even
  start; this should roughly halve that floor. Trade-off: less surrounding context per clause for
  translation.
- **Choppy delivery from the speed control fighting continuous context.** Root cause: Cartesia
  fixes synthesis speed per WebSocket context, so *any* speed change — even the smallest single
  step — ends the current context and starts a new one, losing the intonation continuous context
  exists to carry (see `providers/cartesia_tts.py`). `choose_speed()` recomputes a target every
  segment from how much audio is already queued, and under a full service's sustained near-100%
  channel load that queue depth wobbles by a few seconds turn to turn — crossing the old 0.1
  rounding grid on a large fraction of segments and forcing a reconnect almost every clause. Added
  `SPEED_DEADBAND` (`pipeline.py`): small drift now holds the previous speed exactly; only a
  sustained change moves it. A synthetic replay of a fluctuating queue cut context resets from 35
  to 2 across 200 segments.
- **Log the text, not just its length.** `usage.csv` recorded character counts only, so a dropped
  or odd translation could be read only in the terminal of that one session — gone by the time a
  summary like the 2026-09-14 one arrives with just aggregate counts and no way to tell which
  phrases they were. STT and MT rows now also log the actual recognized/source/translated text
  (capped at 200 chars/row). Next time, the report can be answered from the CSV instead of guesses.

### Applied 2026-09-15 (looking for more delay to cut)

Follow-up pass at the same report, once the fixes above were in but before another live service
existed to re-measure against. This one is **prompt text only, unverified against real sermon
audio** — no ANTHROPIC_API_KEY in the environment this was written in to A/B it, unlike the
2026-09-11 measurement behind the original 91%->77% number. Check it against the next service's
`usage.csv` (now that MT rows log both `in=` and `out=` text) before trusting the ratio moved.

- **Push harder on translation length.** The channel-budget framing in `claude_mt.py`'s system
  prompt was one adjective ("as few words as carry the full meaning") competing with several other
  instructions for the model's attention. Rewrote it as its own explicit paragraph: the channel
  runs close to fully booked for the whole service, so a longer rendering is *heard later*, not
  sooner — same guardrail as before (names/numbers/scripture/commands must still survive), just
  said as the reason rather than left implicit. This is the same lever as the 2026-09-11
  91%->77% cut, pushed further; whether it moves the ratio again needs the next service to confirm.

Deliberately **not** touched this pass, because every option left costs something real without a
live service to weigh it against:
- `stt.partial_min_words`/`partial_gap_ms` further down — cuts more of the pre-translation floor,
  at a further cost to per-clause context (already moved once on 2026-09-14 evidence).
- `audio.max_playback_rate` (catch-up) — see "Still open" below; needs ears on the actual voice in
  use, not a guess from a config comment about a different test.
- Pipelining MT and TTS across the sentences one turn gets split into (`_LanguageStage._handle_segment`
  currently runs each one fully serially) — real latency on paper, but `stt.partial_max_words` 15
  now keeps most turns under `MAX_SEGMENT_CHARS` as a single segment already, so there may be little
  left to pipeline. Worth measuring from the new text log before spending the refactor.
- Streaming the Claude response and pushing partial text into Cartesia's context as it arrives,
  instead of waiting for the full translation — likely the single biggest remaining latency cut
  (shaves the MT round-trip off time-to-first-audio, not just off total channel load), and the
  biggest change: it turns `MTProvider.translate()` from one call into a stream `TTSProvider`
  would need to consume incrementally, touching `base.py`, `claude_mt.py`, `cartesia_tts.py` and
  `pipeline.py` together. Too large to land blind in one pass with no live test to catch a bad
  interaction (a mid-word Cartesia push, a cut-off stream) before a service does.

### Still open

- **The uk channel drops roughly twice the phrases ru does** (38 vs 19 on 2026-09-14, out of a
  shared segment count). Worth checking with the new text logging above: whether Ukrainian MT
  output runs measurably longer per source clause than Russian's, which would mean it needs its
  own, more compressed prompt or a higher `speed_max`.
- **No feedback from the room when the preacher asked for one.** Partly the same delay this report
  addresses, but also structural: the channel ran near 100% busy (see `TTSConfig` and
  `config.py`'s `max_backlog_s` comments), so translated audio is *by design* usually seconds
  behind even with nothing going wrong — `audio.max_playback_rate` (catch-up speed-up) exists for
  this and ships off, because it shifts pitch audibly on a cloned voice. Worth a supervised test at
  a small `max_playback_rate` (1.05-1.1) specifically to see whether that trade is worth it once the
  fixes above have had a service to prove out.
- **Noise in the microphone.** The input is a "copy of Main L/R" from the mixer — the whole mix
  (music, room noise, everyone's mic), not an isolated preacher feed; `config.church.example.yaml`
  already flags this as a setup limitation. No amount of code fixes STT accuracy against a feed
  that was never just speech. Fix is on the mixer side: a dedicated AUX bus carrying only the
  preacher's mic, sent to the interface's input instead of the full mix.
- **A better clone.** Today's clone is 8 seconds of English. 20-30 seconds of clean, expressive
  speech should sound better; a Russian-language sample (if the pastor speaks Russian) would give
  native Russian intonation instead of English prosody carried over.
- **Wired network for the booth.** The T-Mobile line has bandwidth to spare but 0.6-1.1s of latency
  under load; Ethernet (or a separate line) removes the one failure that produced a 90s lag.
- **Calibrate the pace thresholds on a full service.** `pace_normal_wps`/`pace_fast_wps` (2.0/2.8)
  still come from ~2 minutes of one preacher, not the full 2026-09-14 service.
- Code-switching — an STT model with native on-the-fly language detection instead of a fixed
  `source_language`.
