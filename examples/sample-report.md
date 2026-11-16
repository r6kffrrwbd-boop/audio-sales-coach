# Coaching report rpt-0d2c75642a7d

- Recording: `/path/to/demo_call.wav` (pcm_s16le, 2 ch, 101.4 s)
- Layout: **separate_channels**; speakers: S1 = salesperson (80% of talk), S2 = client_in_this_call (20% of talk)
- Events detector: yamnet; transcription: ok
- Council review: **reviewed** (verdict multiple_supported, providers used: none - local checks only)

## Coaching notes

### note-1 - 00:00:00.0 - the salesperson (S1)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 0.0 -t 103.4 /path/to/demo_call.wav`
- **Cue:** The salesperson (S1) spoke 80% of the speaking time in this recording (busiest 2-minute stretch shown).
- **Detector / measurement:** measurement - sum of turn durations per speaker from voice activity / speaker turns
- **Uncertainty:** none noted
- **Words:** (no words transcribed in this span)
- **Interaction to review:** The salesperson (S1) had 80% of the speaking time; the window shown is the stretch where they spoke most.
- **Try instead:** Plan two or three discovery questions and let the answers run, e.g. "Can you walk me through how you handle this today?"
- **Other explanations:** The call may have been a planned presentation or demo
- **Council:** supported
- Evidence: obs-1

### note-2 - 00:00:18.6 - the client in this call (S2)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 16.6 -t 12.0 /path/to/demo_call.wav`
- **Cue:** The client in this call (S2) asked a question (from the transcript).
- **Detector / measurement:** transcript - question mark or interrogative word (English/Arabic/Arabizi list) in the transcript
- **Uncertainty:** none noted
- **Words:** the client in this call (S2): "Okay, and how does the pricing work? Our budget is quite limited this year, so I need to understand the monthly cost, and whether we can start small?" | the salesperson (S1): "So the annual plan is the best value. It is $1200 a year and that includes everything, onboarding, reporting, support, all of it, and you lock in the price"
- **Interaction to review:** The client in this call (S2) asked a question about price/budget. The salesperson (S1) started replying 0.4 s before the question ended (overlap) and spoke for 5 s.
- **Try instead:** Answer it directly first, then confirm: "Does that answer your question?" For price questions, give the number, then pause.
- **Other explanations:** -
- **Council:** supported
- Evidence: obs-2

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

### note-4 - 00:00:26.1 - the salesperson (S1) (uncertain)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 24.1 -t 9.2 /path/to/demo_call.wav`
- **Cue:** The salesperson (S1) spoke faster than usual in this turn: about 6.9 vs a median of 4.4 estimated syllables/s.
- **Detector / measurement:** intensity-peak syllable estimate - estimated syllable rate: peaks of the smoothed 300-2500 Hz intensity contour with >= 1 dB prominence, >= 70 ms apart, inside speech (after de Jong & Wempe 2009); language-independent proxy, not a word count; on TTS test speech it correlates 0.83 with true syllable rate and under-counts very fast speech; score 0.55
- **Uncertainty:** syllable estimates are approximate (+/- ~15%) and depend on audio quality
- **Words:** the salesperson (S1): "So the annual plan is the best value. It is $1200 a year and that includes everything, onboarding, reporting, support, all of it, and you lock in the price"
- **Interaction to review:** The salesperson (S1)'s estimated speaking rate rose by 55% compared with their usual rate in this recording while discussing price/budget.
- **Try instead:** Slow down on key numbers and terms and pause after each one so the other person can respond or ask.
- **Other explanations:** Reading a list or numbers; Summarising familiar material
- **Council:** supported
- Evidence: obs-4

### note-5 - 00:00:31.4 - the client in this call (S2)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 29.4 -t 7.2 /path/to/demo_call.wav`
- **Cue:** 3.2 s of silence after the salesperson (S1) finished before the client in this call (S2) spoke.
- **Detector / measurement:** VAD - time from the end of one speaker's turn to the start of the other speaker's next turn; score 3.2
- **Uncertainty:** none noted
- **Words:** the salesperson (S1): "So the annual plan is the best value. It is $1200 a year and that includes everything, onboarding, reporting, support, all of it, and you lock in the price" | the client in this call (S2): "What are the cancellation terms if it does not work out for us?"
- **Interaction to review:** After the salesperson (S1) finished about price/budget, there were 3.2 s of silence before the client in this call (S2) replied.
- **Try instead:** Silence can be thinking time - let it run a little. If you add something, make it a short open question rather than more information, e.g. "What questions do you have about that?"
- **Other explanations:** Thinking or reading; Checking something off-call; A connection delay; Muted microphone
- **Council:** supported
- Evidence: obs-5

### note-6 - 00:00:34.6 - the client in this call (S2)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 32.6 -t 7.1 /path/to/demo_call.wav`
- **Cue:** The client in this call (S2) asked a question (from the transcript).
- **Detector / measurement:** transcript - question mark or interrogative word (English/Arabic/Arabizi list) in the transcript
- **Uncertainty:** none noted
- **Words:** the client in this call (S2): "What are the cancellation terms if it does not work out for us?" | the salesperson (S1): "Good question. You can cancel with 30 days notice and honestly, nobody has ever wanted to leave the reporting dashboard board. So let me explain how the contract works in detail. The agreement runs for 12 months from the start date, billing is monthly or annual, whichever you prefer, onboarding takes about 2 weeks, and we assign you a success manager. After the first quarter we review usage together and adjust the setup, reporting is delivered every Monday morning, support is available by chat and email during business hours, and urgent issues are handled the same day. We also run a training session for new staff every month and all of that is included in the plan If you add more users later, the price per seat stays the same and you can move between plans at renewal Most customers also connect their calendar and email which takes about 10 minutes Of"
- **Interaction to review:** The client in this call (S2) asked a question about contract or cancellation terms. The reply from the salesperson (S1) started 0.8 s later and lasted 51 s. Check whether the question was answered early in that reply.
- **Try instead:** Answer it directly first, then confirm: "Does that answer your question?"
- **Other explanations:** -
- **Council:** supported
- Evidence: obs-6

### note-7 - 00:00:38.5 - the salesperson (S1)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 36.5 -t 34.0 /path/to/demo_call.wav`
- **Cue:** The salesperson (S1) spoke for 51 s without the other person taking a turn.
- **Detector / measurement:** measurement - turn duration from speaker turns
- **Uncertainty:** none noted
- **Words:** the salesperson (S1): "Good question. You can cancel with 30 days notice and honestly, nobody has ever wanted to leave the reporting dashboard board. So let me explain how the contract works in detail. The agreement runs for 12 months from the start date, billing is monthly or annual, whichever you prefer, onboarding takes about 2 weeks, and we assign you a success manager. After the first quarter we review usage together and adjust the setup, reporting is delivered every Monday morning, support is available by chat and email during business hours, and urgent issues are handled the same day. We also run a training session for new staff every month and all of that is included in the plan If you add more users later, the price per seat stays the same and you can move between plans at renewal Most customers also connect their calendar and email which takes about 10 minutes Of"
- **Interaction to review:** The salesperson (S1) spoke for 51 s without a turn from the other person about price/budget and contract or cancellation terms.
- **Try instead:** Break long explanations with a check-in about every 30-45 s, e.g. "Does this match what you need so far?" or "Which part is most relevant for you?"
- **Other explanations:** The other person asked for a full explanation; A requested demo
- **Council:** supported
- Evidence: obs-7

### note-8 - 00:00:46.6 - the client in this call (S2) (uncertain)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 44.6 -t 5.0 /path/to/demo_call.wav`
- **Cue:** Audible breathing/sigh detected from the client in this call (S2) (score 0.90).
- **Detector / measurement:** yamnet - yamnet audio-event classifier on the original audio, 0.96 s windows, 0.48 s hop; score 0.9
- **Uncertainty:** breathing is a weak detector class; replay to confirm
- **Words:** (no words transcribed in this span)
- **Interaction to review:** An audible breath or sigh was detected from the client in this call (S2). It is a sound, not evidence of a feeling.
- **Try instead:** Leave room for them to speak; a neutral check-in works either way: "How does that sound so far?"
- **Other explanations:** Microphone close to the mouth; Movement or walking; Detector false alarm on noise
- **Council:** supported
- Evidence: obs-9

### note-9 - 00:01:30.7 - the client in this call (S2) (uncertain)

- **Replay (original recording):** `ffplay -nodisp -autoexit -loglevel error -ss 88.7 -t 5.0 /path/to/demo_call.wav`
- **Cue:** Audible breathing/sigh detected from the client in this call (S2) (score 0.69).
- **Detector / measurement:** yamnet - yamnet audio-event classifier on the original audio, 0.96 s windows, 0.48 s hop; score 0.695
- **Uncertainty:** breathing is a weak detector class; replay to confirm
- **Words:** the client in this call (S2): "I will need to check with my manager before we decide anything."
- **Interaction to review:** An audible breath or sigh was detected from the client in this call (S2) before their turn about who decides. It is a sound, not evidence of a feeling.
- **Try instead:** Leave room for them to speak; a neutral check-in works either way: "How does that sound so far?"
- **Other explanations:** Microphone close to the mouth; Movement or walking; Detector false alarm on noise
- **Council:** supported
- Evidence: obs-10

## All observations

| id | time | type | statement | uncertain |
|---|---|---|---|---|
| obs-1 | 00:00:00.0 | talk_share | The salesperson (S1) spoke 80% of the speaking time in this recording (busiest 2-minute stretch shown). | no |
| obs-2 | 00:00:18.6 | question | The client in this call (S2) asked a question (from the transcript). | no |
| obs-3 | 00:00:26.1 | interruption | The salesperson (S1) started speaking while the client in this call (S2) was speaking (0.4 s of overlap); the client in this call (S2) stopped 0.4 s later. | no |
| obs-4 | 00:00:26.1 | pace_change | The salesperson (S1) spoke faster than usual in this turn: about 6.9 vs a median of 4.4 estimated syllables/s. | yes |
| obs-5 | 00:00:31.4 | response_gap | 3.2 s of silence after the salesperson (S1) finished before the client in this call (S2) spoke. | no |
| obs-6 | 00:00:34.6 | question | The client in this call (S2) asked a question (from the transcript). | no |
| obs-7 | 00:00:38.5 | monologue | The salesperson (S1) spoke for 51 s without the other person taking a turn. | no |
| obs-8 | 00:00:45.6 | long_pause | The salesperson (S1) paused 3.1 s mid-turn. | no |
| obs-9 | 00:00:46.6 | breathing | Audible breathing/sigh detected from the client in this call (S2) (score 0.90). | yes |
| obs-10 | 00:01:30.7 | breathing | Audible breathing/sigh detected from the client in this call (S2) (score 0.69). | yes |

## Limitations

- Real-world accuracy on authorized sales calls has not been measured; evaluation so far uses synthetic speech and licensed sound-event clips (docs/evaluation.md).
- Cues are observations, not emotions: laughter, breathing, pauses, pitch and pace changes have many causes.
- On mixed mono recordings, speaker attribution comes from voice clustering and can be wrong, especially for similar voices, short turns and overlapping speech.
- Breathing detection is weak; treat breathing cues as prompts to replay, not as evidence.
- Mixed Lebanese Arabic/English transcription (optional) is approximate; acoustic cues do not depend on it.

_Every note describes audible, measurable cues. None states how anyone felt, whether they were truthful, or whether they will buy._
