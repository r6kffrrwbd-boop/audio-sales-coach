# Evaluation (what was measured, on what, and what is still unverified)

**Real-world performance on authorized sales calls is unverified.** No real customer recordings were available or
used. Every number below comes from synthetic speech (macOS text-to-speech voices generated locally) and licensed
sound-event clips (ESC-50, CC BY-NC 3.0). TTS voices are cleaner and more consistent than people on phones;
expect real calls to be harder (accents, crosstalk, handset noise, Lebanese dialect).

Machine: Intel Core i7-6700HQ MacBook Pro, 16 GB RAM, macOS 14.8, CPU only. Reproduce with
`audio-sales-coach eval --workdir ./asc-eval` (downloads the ESC-50 clips it needs into the local cache).
Raw results: `docs/results/evaluation-summary.json`.

## 1. Audio-event detector (YAMNet) on licensed clips

ESC-50 classes *laughing* and *breathing* against 8 confusable classes (coughing, sneezing, snoring, clapping,
crying baby, keyboard typing, drinking/sipping, door knock). Thresholds were chosen on **fold 1** (max F1 over a
fixed grid) and tested on **held-out fold 2** (8 positives / 72 negatives per class). TTS speech clips were added as
negatives.

| Cue | Threshold | Fold-2 precision | Fold-2 recall | Fold-2 AUC | False alarms on TTS speech | Confused with |
|---|---|---|---|---|---|---|
| Laughter | 0.25 | 0.70 (7/10) | 0.88 (7/8) | 0.97 | 0/4 | crying baby |
| Breathing | 0.60 | 0.35 (6/17) | 0.75 (6/8) | 0.83 | 0/4 | snoring, sneezing, drinking |

**In context:** every laughing and breathing clip from folds 3-5 (24 each, never used for thresholds) inserted
between two TTS speech turns:

| Cue | Recall (WAV) | Recall (Opus 16 kbit/s, WhatsApp-like) | False laughter/breathing events on the surrounding speech |
|---|---|---|---|
| Laughter | 0.63 (15/24) | 0.54 (13/24) | 0 |
| Breathing | 0.63 (15/24) | 0.58 (14/24) | 0 |

Conclusion: laughter detection is usable as a "replay this" pointer; breathing is weak (low precision on isolated
clips) and every breathing note is marked uncertain. Many ESC-50 "laughing" clips (e.g. 5-263775-A-26, used in the
demo call) are scored by YAMNet as breathing/whimpering, not laughter - such misses are reported, not hidden.

YAMNet NumPy port vs Google's TensorFlow implementation: max score difference 1.5e-5, 100 % top-1 agreement
(`docs/models.md`).

## 2. Synthetic calls with exact ground truth

Five scenarios rendered by `synth.py`. Two-person calls are written both as stereo (one speaker per channel) and as
a mono mix. Variants add pink noise at SNR 10 dB and 5 dB and 16 kbit/s Opus compression. Transcription was off
(audio-only) in this table.

| Scenario / variant | Layout found | Speakers | Frame attribution accuracy | Uncertain share | Interruptions found | False interruptions | Response gaps found | Laughter / breathing found (false) |
|---|---|---|---|---|---|---|---|---|
| demo call, stereo | separate_channels | 2 | 0.985 | 0.00 | 1/1 | 0 | 1/1 | 0/1, 1/1 (1) |
| demo call, mono | mixed_mono | 2 | 0.990 | 0.00 | 1/1 (as *possible*) | 0 | 1/1 | 0/1, 1/1 (0) |
| demo call, mono, SNR 10 dB | mixed_mono | 2 | 0.997 | 0.00 | 1/1 | **4** | 1/1 | 0/1, 0/1 (0) |
| demo call, mono, SNR 5 dB | mixed_mono | 2 | 0.994 | 0.00 | 1/1 | **2** | **0/1** | 0/1, 0/1 (0) |
| demo call, Opus 16 kbit/s | mixed_mono | 2 | 0.962 | 0.00 | 1/1 | 0 | 1/1 | 0/1, 1/1 (0) |
| similar male voices, stereo | separate_channels | 2 | 0.969 | 0.00 | 1/1 | 0 | 1/1 | 0/1, 1/1 (0) |
| similar male voices, mono | mixed_mono | 2 | 0.983 | 0.00 | 1/1 | 0 | 1/1 | 0/1, 1/1 (0) |
| Arabic/English code-switch, stereo | separate_channels | 2 | 0.981 | 0.00 | - | 0 | 1/1 | - |
| Arabic/English code-switch, mono | mixed_mono | 2 | 0.981 | 0.00 | - | 0 | 1/1 | - |
| voice note (18.6 s), WAV / Opus / auto mode | single_speaker | **1** | - | - | - | - | - | - |
| short voice note (4.7 s) | single_speaker | **1** | - | - | - | - | - | - |

Findings:

* **Speaker mistakes:** 1-4 % of speech frames mislabelled on these TTS voices, mostly at turn boundaries. The
  "uncertain" flag never fired here because TTS voices are very distinct even when "similar"; on real voices expect
  more errors and more `unknown`. This is the main unverified risk.
* **Voice messages:** never split into two people (5/5).
* **Noise:** at SNR ≤ 10 dB the mono pipeline produced 2-4 false *possible interruptions* (noise bridges pauses, so
  speaker changes appear to have no gap) and lost the response gap at 5 dB. These cues are already marked uncertain
  on mixed recordings; the report flags low speech-to-background levels.
* **Compression:** Opus 16 kbit/s lowered attribution to 96 % and kept the other cues.
* **Client questions:** audio-only found 0 of 4 (the TTS questions do not end with rising pitch). Questions need
  the optional transcription.

## 3. Transcript-only baseline

Upper-bound baseline: an *oracle* transcript with perfect words, perfect speaker labels and perfect timestamps,
which real ASR on mono calls does not give you.

| Ground-truth coachable moment (all scenarios) | Oracle transcript | Audio pipeline (no transcript) |
|---|---|---|
| Client questions (4) | 4 | 0 (rising-pitch cue missed them) |
| Interruptions (2) | 2, only via perfect overlapping timestamps | 2 (stereo: measured overlap; mono: "possible") |
| Response gaps ≥ 2.5 s (3) | 3, only via perfect timestamps | 3 (the SNR 5 dB variant lost its gap) |
| Laughter (2 inserted) | 0 | 0 in calls; 63 % recall in the 24-clip insertion test |
| Breathing (2 inserted) | 0 | 2; 63 % recall in the insertion test |
| Pace changes (1 fast turn per two-person call) | 0 | found in all 7 call variants (+32 % to +67 % estimated syllable rate, incl. noise and Opus); no false pace changes |
| Pitch / level changes | 0 | measured per turn |

Text alone cannot see laughter, breathing, pace, pitch or level, and only sees interruptions and silences if its
timestamps are perfect. The audio pipeline misses questions without words. Both views are combined when
transcription is installed.

## 4b. Optional transcription (faster-whisper `small`, int8, CPU)

| Recording | Result |
|---|---|
| Demo call, stereo, per-channel transcription | English words accurate (language p = 0.99 / 0.995); both client questions found with exact words; 30 s of transcription time for 101 s of audio (~40 s total analysis) |
| Arabic/English code-switched call, mono | language detected `ar` (p = 0.90). English sentences correct. Arabic mostly recognisable but code-switched English inside Arabic was garbled ("قديش الـ price تبع الـ annual plan" became nonsense, "cancel" became "كنسل", one number became "كيلو متر"), and Whisper still marked every segment above the confidence thresholds - so the "reliable" flag does **not** catch code-switching errors. A word at a turn boundary leaked into the other speaker's turn. After adding mid-sentence Arabic interrogatives (قديش, شو, ...), both client questions were found. |

## 4. Real model-inference runs (timestamped output)

See `examples/sample-report.md` for the full output of a real run on this machine, and the README for an excerpt.
Timings on this CPU: decode 0.5 s, speakers 6-9 s, measurements 1 s, YAMNet 3-6 s per channel per 100 s of audio,
about 11-13 s total for a 100 s call without transcription.

## 5. evidence-council integration (both modes run on this machine)

| Mode | Result |
|---|---|
| Council absent (`evidence-council` not on PATH) | analysis completes; `council.status = "skipped"`, reason "evidence-council is not installed; council review was skipped" |
| Council present, `providers: ["none"]` (default) | `evidence-council review` exit 0; every note's keyed claims (cue, speaker role, attribution reliability) checked against observation facts; verdict `multiple_supported` (all notes consistent); nothing sent to any model |
| Tampered note (test) | a note claiming the interruption came from the client is **contradicted** by the observation fact and flagged |
| Council present, `providers: ["ollama:qwen3.5:4b"]` (local) | 222 s on this CPU, 1,628 input / 611 output tokens, $0; all 7 notes still supported by the evidence checks. The model's free-text critique **hallucinated** problems not present in the notes (it claimed notes mentioned nervousness, readiness to buy and lying). Such comments are shown in the report as *unverified model output* and never change a note's status. |

## 6. Not verified

* Accuracy on real, authorized sales calls (none available).
* Lebanese Arabic speech from real speakers (the Arabic test uses an MSA TTS voice reading code-switched text).
* Voice-ID thresholds on real salespeople (defaults come from TTS voices; use `voice calibrate`).
* The WhatsApp Business Platform adapter against a live account (mocked HTTP only).
* Usefulness of the coaching suggestions to salespeople - collect `feedback` and read `feedback summary`.
* Feedback does not retrain any model; no training is implemented.
