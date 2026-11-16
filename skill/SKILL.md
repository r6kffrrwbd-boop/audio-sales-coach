---
name: audio-sales-coach
description: Analyse an authorized sales call recording or voice message the user supplies (WAV, MP3, M4A, WhatsApp OGG/Opus) locally on CPU and return replayable, evidence-linked coaching notes for the salesperson - interruptions, response gaps, monologues, talk share, pace/pitch/level changes, detector-scored laughter and audible breathing, client questions. Use when the user asks to coach, review or analyse a sales call or voice note, enroll or manage a salesperson voice profile, assign speaker roles, or give feedback on coaching notes. Never use it to judge a client's emotions, honesty or buying intent, or on recordings without consent.
---

# audio-sales-coach

A local CLI. All normal use goes through the `audio-sales-coach` command and its JSON report.
If it is not on PATH, use `~/.claude/skills/audio-sales-coach/bin/audio-sales-coach`.

## Ground rules for Claude

1. **Only recordings the user explicitly supplies and is authorized to analyse.** Ask what participants were told and
   pass it as `--consent all-parties-consented|notified|unknown`. Never fetch calls from anywhere, never scrape WhatsApp Web.
2. **Report what the report says.** Quote notes from `report.json` / `report.md`. Do not add interpretations of
   feelings, honesty or buying intent - not even as "probably". Relay each note's `uncertainty`, `alternative_explanations`
   and `council.status`.
3. **Roles:** if `role_assignment.status` is `needs_user`, show the excerpt commands and ask the user who the salesperson
   is; then run `audio-sales-coach roles REPORT --salesperson S1|S2`. Only add `--remember PROFILE` when the salesperson
   themself consents, and pass their own consent statement.
4. **Privacy:** keep reports local. Do not paste transcripts, client names or numbers into other tools. The council
   (if installed) gets only the minimal task saved in `REPORT/council/task.json`; by default it uses no AI provider.
   Only change `--council-providers` when the user asks, and never to a cloud provider for client data without their
   explicit, informed request.
5. **Honesty about limits:** say when cues are unreliable (`reliability`, `warnings`), when no transcript exists, and that
   real-world accuracy on actual sales calls has not been measured.
6. Feedback is stored for evaluation; it does not retrain anything. Do not claim otherwise.

## Commands

| Situation | Command |
|---|---|
| Analyse a call | `audio-sales-coach analyze CALL.m4a --output ~/coach-reports/NAME --consent all-parties-consented` |
| Voice message (one speaker) | `audio-sales-coach analyze NOTE.ogg --output DIR --mode voice-message` |
| Salesperson already known | add `--salesperson S1` (stereo exports: S1 = left/first channel) |
| Words too (if installed with `--with-transcription`) | `--transcribe on` (default `auto`) |
| Replay a note from the original recording | `audio-sales-coach play DIR --note note-3` (or `--print-only`) |
| Assign roles | `audio-sales-coach roles DIR --salesperson S2 [--remember alias --consent "..."]` |
| Feedback | `audio-sales-coach feedback add DIR --item note-3 --mark useful|incorrect|uncertain --reason "..."` |
| Speaker label wrong | `audio-sales-coach feedback add DIR --speaker S1=client --reason "..."` then `roles` |
| Voice profiles | `audio-sales-coach voice enroll|re-enroll|list|delete|calibrate NAME [--sample F --consent "..."]` |
| Health check | `audio-sales-coach models status`, `audio-sales-coach selftest` |
| Retention | `audio-sales-coach purge --reports-root ~/coach-reports [--yes]` |

Exit codes: 0 ok, 1 error, 2 usage/input error, 3 missing ffmpeg/models.

## Reading the report

`report.json` → `coaching_notes[]` each has `timestamp`, `play.uri` (original file `#t=start,end`), `play.ffplay`,
`speaker` (role, `attribution_confidence`, `attribution_reliable`), `words`/`words_status`, `cue` (type, description,
detector, method, measurement, score, uncertainty), `interaction`, `suggestion`, `alternative_explanations`, `council`.
`views` holds the synchronized salesperson / client / combined views (one `speaker` view for voice messages).
`report.html` is a click-to-play timeline. See `references/` for privacy, models, evaluation and the JSON format.
