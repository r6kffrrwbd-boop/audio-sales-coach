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

Command (run on this Intel MacBook, CPU only, 40 s for a 101 s call including Whisper-small transcription):

```bash
audio-sales-coach analyze demo_call.wav --output report --mode call --transcribe on --salesperson S1 --consent all-parties-consented
```

```json
{
  "layout": "separate_channels",
  "speakers": [{"id": "S1", "role": "salesperson", "talk_share": 0.803},
               {"id": "S2", "role": "client_in_this_call", "talk_share": 0.197}],
  "notes": 9, "observations": 10, "events": 2, "council": "reviewed", "roles": "confident", "warnings": []
}
```

Timestamped observations from that run: 00:00:18.6 client question (price) · 00:00:26.1 salesperson started
speaking during the client's turn, 0.4 s overlap · 00:00:26.1 salesperson pace +55 % · 00:00:31.4 3.2 s silence
before the client replied · 00:00:34.6 client question (cancellation terms) · 00:00:38.5 51 s salesperson
monologue · 00:00:46.6 and 00:01:30.7 audible breathing on the client channel (YAMNet 0.90 / 0.69, marked uncertain;
the first is an inserted ESC-50 *laughing* clip that YAMNet scored as breathing - a real detector miss).

One note as rendered in `report.md`:

### note-3 - 00:00:26.1 - the salesperson (S1)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 24.1 -t 4.4 /path/to/demo_call.wav`
- **Cue:** The salesperson (S1) started speaking while the client in this call (S2) was speaking (0.4 s of overlap); the client in this call (S2) stopped 0.4 s later.
- **Detector / measurement:** per-channel VAD - simultaneous voice activity on the two call channels; score 0.42
- **Uncertainty:** none noted
- **Words:** the client in this call (S2): "Okay, and how does the pricing work? Our budget is quite limited this year, so I need to understand the monthly cost, and whether we can start small?" | the salesperson (S1): "So the annual plan is the best value. It is $1200 a year and that includes everything, onboarding, reporting, support, all of it, and you lock in the price" | the client in this call (S2): "What are the cancellation terms if it does not work out for us?"
- **Interaction to review:** The salesperson (S1) started speaking while the client in this call (S2) was talking about price/budget. Their next turn was a question about contract or cancellation terms.
- **Try instead:** Let them finish, summarise what you heard in one sentence, then ask: "Before I answer - what budget range were you planning for this?"
- **Other explanations:** The client in this call (S2) may have been finishing anyway; Line delay can make turn-taking look like overlap
- **Council:** supported
- Evidence: obs-3

The same note in `report.json`:

```json
{
  "id": "note-3",
  "timestamp": "00:00:26.1",
  "play": {
    "recording": "original",
    "uri": "file:///path/to/demo_call.wav#t=24.1,28.6",
    "from_s": 24.1,
    "to_s": 28.6,
    "ffplay": "ffplay -nodisp -autoexit -loglevel error -ss 24.1 -t 4.4 /path/to/demo_call.wav",
    "command": "audio-sales-coach play /path/to/report --note note-3"
  },
  "speaker": {
    "id": "S1",
    "role": "salesperson",
    "label": "the salesperson (S1)",
    "attribution_confidence": 0.95,
    "attribution_reliable": true
  },
  "words": "the client in this call (S2): \"Okay, and how does the pricing work? Our budget is quite limited this year, so I need to understand the monthly cost, and whether we can start small?\" | the salesperson (S1): \"So the annual ...",
  "cue": {
    "type": "interruption",
    "description": "The salesperson (S1) started speaking while the client in this call (S2) was speaking (0.4 s of overlap); the client in this call (S2) stopped 0.4 s later.",
    "detector": "per-channel VAD",
    "method": "simultaneous voice activity on the two call channels",
    "measurement": {
      "overlap_s": 0.42,
      "interrupter": "S1",
      "interrupted": "S2",
      "earlier_speaker_continued_s": 0.42
    },
    "score": 0.42,
    "uncertainty": "none noted"
  },
  "interaction": "The salesperson (S1) started speaking while the client in this call (S2) was talking about price/budget. Their next turn was a question about contract or cancellation terms.",
  "suggestion": "Let them finish, summarise what you heard in one sentence, then ask: \"Before I answer - what budget range were you planning for this?\"",
  "alternative_explanations": [
    "The client in this call (S2) may have been finishing anyway",
    "Line delay can make turn-taking look like overlap"
  ],
  "council": {
    "status": "supported",
    "claims_checked": 3
  }
}
```


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

**Verified on this machine (Intel i7-6700HQ MacBook Pro, 16 GB, macOS 14.8, CPU only):**

* `./install.sh --with-transcription` from scratch: venv, pinned deps, model downloads with SHA-256 checks, skill,
  command, self-test 16/16 passed. Test suite: 51 passed (unit tests plus end-to-end runs on synthetic calls, and a
  live evidence-council run that catches a tampered speaker attribution).
* YAMNet: NumPy port equals Google's TensorFlow 2.16.2 + tf-keras 2.16.0 output (max diff 1.5e-5). Official code
  fails under Keras 3; TF 2.16.2 is the last Intel-macOS TensorFlow and needs numpy < 2.
* Laughter (held-out ESC-50): precision 0.70, recall 0.88, AUC 0.97; inserted between speech: recall 0.63 (0.54
  after Opus 16 kbit/s), 0 false events on speech. Breathing is weak: precision 0.35 on isolated clips.
* Synthetic calls (TTS voices): speaker attribution 96-99.7 % for stereo, mono, noisy, Opus and similar voices;
  voice notes never split into two people; interruptions, response gaps and pace changes found; 2-4 false
  "possible interruptions" on noisy mono; client questions need transcription (audio-only found 0 of 4).
* Voice ID on TTS voices: enrolled voice 0.92, other voices 0.25-0.29, roles set automatically.
* evidence-council: absent → "skipped"; present with no providers → claims checked; with local `ollama:qwen3.5:4b`
  → ran in 222 s, its free-text critique hallucinated issues, which the evidence checks did not accept.
* Transcription (Whisper small, int8): English accurate; Arabic/English code-switching garbled (English words inside
  Arabic sentences), while Whisper still rated those segments confident.

**Not verified:** accuracy on real authorized sales calls (none were available), real Lebanese dialect speakers,
voice-ID thresholds on real people, the WhatsApp Business adapter against a live account (mocked HTTP only), and
whether salespeople find the suggestions useful. Feedback is stored for evaluation; it does not retrain anything.

## Privacy and legal

Only analyse recordings you are authorized to analyse; tell participants about recording and coaching use as your
law requires. Adapters exist for **official** sources only: WhatsApp Business Platform webhooks for your own
business number, and call-recording exports with a consent sidecar. No WhatsApp Web scraping, no silent call
access. Retention: `audio-sales-coach purge`. Read [docs/privacy.md](docs/privacy.md).

## License

MIT for this code. Models and data have their own licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
