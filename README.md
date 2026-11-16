# audio-sales-coach

Local, consent-first coaching for salespeople. You give it a call recording or a voice message you are allowed to
analyse. It listens to the **original audio** (not just the words), measures what can actually be heard, and drafts
**specific, replayable coaching moments**:

* pauses and response gaps, speaking-pace changes, pitch and level changes (each relative to the same speaker),
* interruptions and overlapping speech, monologues and talk share,
* laughter and audible breathing from a real audio-event detector (YAMNet), never from a language model,
* client questions (from optional local transcription, or rising pitch at the end of a turn),
* each with a timestamp, a link that plays that moment of the original recording, the speaker (or "unknown"), the
  exact words when transcribed, the detector/measurement and its score or uncertainty, the interaction worth
  reviewing, and a constructive alternative question or response.

It does **not** replace the salesperson, score a client's emotions, say anyone was nervous, lying, comfortable or
ready to buy, or suggest pressure tactics. A language guard enforces this on every note.

Everything runs on CPU on the machine (tested on an Intel MacBook Pro, 16 GB RAM, no GPU): no paid API, no cloud
upload, no Hugging Face login.

## Install

Requirements: macOS or Linux, Python 3.10-3.12, `ffmpeg` (`brew install ffmpeg`), `curl`.

```bash
git clone https://github.com/r6kffrrwbd-boop/audio-sales-coach.git
cd audio-sales-coach
./install.sh
```

`install.sh` is idempotent. It creates an isolated venv in `~/.local/share/audio-sales-coach/venv` with pinned
dependencies (`requirements/base.lock`), downloads and SHA-256-verifies ~44 MB of models, installs the Claude Code
skill at `~/.claude/skills/audio-sales-coach`, links `~/.local/bin/audio-sales-coach`, and runs the offline self-test.

Optional, explicit extras (sizes and limits in [docs/models.md](docs/models.md)):

| Flag | Adds |
|---|---|
| `--with-transcription` | faster-whisper (CPU int8) for words; ~140 MB of wheels + ~480 MB `small` model on first use, public, no login |
| `--with-ced` | CED-mini audio tagger (48 MB) as a second event detector |
| `--with-tensorflow` | separate TF 2.16.2 venv (~1.3 GB) only to cross-check YAMNet against Google's TensorFlow code |

### Manual install

```bash
python3.12 -m venv ~/.local/share/audio-sales-coach/venv
~/.local/share/audio-sales-coach/venv/bin/pip install -r requirements/base.lock
~/.local/share/audio-sales-coach/venv/bin/pip install --no-deps .
~/.local/share/audio-sales-coach/venv/bin/audio-sales-coach models fetch
~/.local/share/audio-sales-coach/venv/bin/audio-sales-coach selftest
# optional skill: copy skill/SKILL.md to ~/.claude/skills/audio-sales-coach/SKILL.md
```

### Removal

```bash
./uninstall.sh                # removes venvs, models, skill and command; keeps voice profiles and feedback
./uninstall.sh --purge-data   # also deletes voice profiles, feedback and config
```

Manual removal: delete `~/.local/share/audio-sales-coach`, `~/.claude/skills/audio-sales-coach`,
`~/.local/bin/audio-sales-coach` and `~/.config/audio-sales-coach`. Delete report folders you created yourself.

## Use

```bash
# a two-person call (stereo exports with one side per channel are used as-is; mono is diarized)
audio-sales-coach analyze ~/calls/call.m4a --output ~/coach-reports/call-1 --consent all-parties-consented

# a WhatsApp voice note (one speaker; a second person is never invented)
audio-sales-coach analyze ~/Downloads/note.ogg --output ~/coach-reports/note-1 --mode voice-message

# replay a note's moment from the ORIGINAL recording
audio-sales-coach play ~/coach-reports/call-1 --note note-2

# who is the salesperson? (asked when roles are uncertain; excerpts are listed in the report)
audio-sales-coach roles ~/coach-reports/call-1 --salesperson S1

# feedback (stored separately; used for evaluation, it does not retrain anything)
audio-sales-coach feedback add ~/coach-reports/call-1 --item note-2 --mark incorrect --reason "that was agreement"
audio-sales-coach feedback add ~/coach-reports/call-1 --speaker S2=salesperson --reason "labels swapped"
audio-sales-coach feedback summary
```

Open `report.html` for a click-to-play timeline with the salesperson, client and combined lanes on one timebase,
`report.md` for reading, `report.json` for tools ([format](docs/json-output.md)).

### Salesperson voice ID (optional, local)

```bash
audio-sales-coach voice enroll my-alias --sample me-talking.m4a --consent "I agree to store my voice profile for coaching"
audio-sales-coach voice list | calibrate my-alias | re-enroll my-alias --sample new.m4a --consent "..." | delete my-alias
audio-sales-coach roles ~/coach-reports/call-1 --salesperson S1 --remember my-alias --consent "..."   # learn from a correction
```

When a profile matches above the accept threshold, that speaker is labelled the salesperson and the other
"client in this call". Between the thresholds, or with no match, the report says `unknown` and asks you to assign
roles from short excerpts. Profiles (speaker embeddings) live in `~/.local/share/audio-sales-coach/voice-profiles`
with 0600 permissions, outside Git. Client voice profiles are never stored.

### evidence-council (optional)

If [`evidence-council`](https://github.com/r6kffrrwbd-boop/evidence-council) is installed, the draft notes and only
minimal timestamped observations are sent to it through its JSON CLI (`evidence-council review task.json --output
result.json`). It checks that each note's cue, speaker role and attribution reliability match the observations,
that citations and timestamps exist, and whether any client question has no note. By default no AI provider
receives anything (`providers: ["none"]`); `--council-providers ollama` adds a local model. If the council is not
installed, analysis runs normally and the report says council review was skipped. The exact task sent is saved as
`REPORT/council/task.json`. Details: [docs/privacy.md](docs/privacy.md).

## Example output

From a real run on this machine on a synthetic 101 s call (two macOS TTS voices on separate channels, a licensed
ESC-50 breathing clip and an ESC-50 laughing clip inserted on the client channel). See
[examples/sample-report.md](examples/sample-report.md) and [examples/sample-report.json](examples/sample-report.json).

SAMPLE_PLACEHOLDER

## How it works

| Step | Method |
|---|---|
| Decode | ffmpeg → 16 kHz float32 in memory (WAV, MP3, M4A/AAC, OGG/Opus, OGG/Vorbis, FLAC); no temp files |
| Speech | Silero VAD (ONNX via sherpa-onnx) |
| Speakers | stereo with one side per channel → per-channel turns and true overlaps; mixed mono → CAM++ speaker embeddings on 1.5 s windows, agglomerative clustering, per-frame vote with confidence, "possible overlap" at changes without a pause; single speaker → one speaker. No source separation: all evidence refers to the original recording |
| Measurements | RMS level, autocorrelation pitch (70-400 Hz), syllable-nucleus pace estimate, pauses, response gaps - all relative to the same speaker |
| Events | YAMNet (Google's released weights) run by a NumPy port that matches the TensorFlow model to 1.5e-5 |
| Words (optional) | faster-whisper with word timestamps; per-segment confidence marks unreliable spans |
| Notes | deterministic rules + language guard; optional evidence-council review |

## What was tested, and what was not

See [docs/evaluation.md](docs/evaluation.md) for all numbers.

EVAL_PLACEHOLDER

## Privacy and legal

Only analyse recordings you are authorized to analyse; tell participants about recording and coaching use as your
law requires. Adapters exist for **official** sources only: WhatsApp Business Platform webhooks for your own
business number, and call-recording exports with a consent sidecar. No WhatsApp Web scraping, no silent call
access. Retention: `audio-sales-coach purge`. Read [docs/privacy.md](docs/privacy.md).

## License

MIT for this code. Models and data have their own licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
