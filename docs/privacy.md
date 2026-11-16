# Privacy, consent and retention

audio-sales-coach is built to coach **your own team** on **recordings you are allowed to analyse**. It runs locally.

## Before you analyse a recording

Laws differ by country and state. Typical requirements (this is not legal advice):

* **Notice / consent to record.** Many places require that every participant knows the call is recorded
  (some require everyone's consent - "all-party consent"). Lebanon's Law 81/2018 on electronic transactions and
  personal data, the EU/UK GDPR, and US all-party-consent states (e.g. California, Florida) are examples where
  notice and a lawful basis matter.
* **Purpose.** Tell participants recordings may be used for **quality and staff coaching**. A recording made only
  "for record keeping" may not cover analysis.
* **Voice profiles are biometric data.** Enrol a salesperson only with their explicit consent; the tool requires
  a consent statement and records it in the profile. Under GDPR this is special-category data.
* **Clients.** No client voice profile is ever stored. A client's speaker embedding exists only in memory during
  one analysis.

Record what participants were told with `--consent all-parties-consented|notified|unknown`; the value is written
into the report, and `unknown` adds a warning.

## What stays where

| Data | Location | Notes |
|---|---|---|
| Original recording | wherever you keep it | never copied, moved or uploaded; reports link to it by path |
| Converted audio | memory only | decoded via ffmpeg stdout; no temporary file. `privacy.keep_converted_audio: true` writes a 16 kHz WAV into the report folder |
| Reports | the `--output` folder you choose | created 0700, files 0600; contain words if transcription ran |
| Voice profiles | `~/.local/share/audio-sales-coach/voice-profiles/` | 0700/0600; outside any git repo; `voice delete NAME` overwrites then removes |
| Feedback | `~/.local/share/audio-sales-coach/feedback/<report_id>.jsonl` | separate from recordings and reports |
| Models | `~/.local/share/audio-sales-coach/models/` | public model files only |

The repository `.gitignore` blocks audio, reports, profiles, feedback, tokens and model files.

## Network use

* `install.sh` downloads Python packages (PyPI) and model files (Google Cloud Storage for YAMNet, GitHub releases
  for Silero VAD and CAM++). Each model is verified by size and SHA-256. No account or login.
* With `--with-transcription`, the first transcription downloads a public faster-whisper model from Hugging Face
  (no login; telemetry disabled via `HF_HUB_DISABLE_TELEMETRY=1`).
* `analyze` makes **no network requests**, except what an optional council provider you configure does.

## evidence-council (optional)

If `evidence-council` is installed, draft notes are sent to it **as a local program** through its JSON interface.
The exact task is saved as `REPORT/council/task.json` so you can audit it. It contains:

* timestamps, neutral role-labelled observation statements ("the client in this call (S2) asked a question"),
  keyed facts (cue type, role, attribution reliability, detector score), the draft notes and their claims;
* a `recording` item with **no path, URI or content**.

It never contains audio, file paths, transcripts or quotes (unless you set `council.include_quotes: true`),
names, phone numbers, e-mail addresses, CRM data or voice embeddings. A guard blocks the request if any text looks
like a path, an e-mail address or a long digit sequence; configured `privacy.redact_terms` are replaced by `[name]`.

The default provider list is `["none"]`: the council only checks claims against the observations; no AI model
receives anything. `--council-providers ollama:qwen3.5:4b` (an explicitly named, locally installed model) adds a
**local** Ollama model. Ollama model names ending in `-cloud` (e.g. `gpt-oss:120b-cloud`) run on Ollama's servers,
not on your machine - name a local model explicitly instead of a bare `ollama`. evidence-council's own `private`
privacy level keeps cloud providers excluded unless the operator allowlists one in its config.

## Retention

Defaults in `~/.config/audio-sales-coach/config.json`:

```json
{"retention": {"reports_days": 30, "feedback_days": 365}}
```

`audio-sales-coach purge --reports-root ~/coach-reports` lists what is older (dry run); add `--yes` to delete.
Schedule it with cron/launchd if you need automatic deletion. Recordings themselves are governed by your own system.

## Ingestion adapters

* **WhatsApp Business Platform** (`ingest whatsapp-webhook`): parses webhook payloads that *your* endpoint received
  for *your* business number and downloads voice messages with *your* access token (`WHATSAPP_ACCESS_TOKEN`, read
  from the environment, never stored). Tell customers that voice messages may be analysed for staff coaching.
  Tested with mocked HTTP only; not run against a live account. There is no WhatsApp Web scraping and no personal
  WhatsApp support.
* **Call-recording exports** (`adapters/recording_folder.py`): picks up exported files only when a
  `<name>.consent.json` sidecar states `notified` or `all-parties-consented`. Nothing listens to live calls.
