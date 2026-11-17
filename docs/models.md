# Models, runtimes and the YAMNet / TensorFlow evaluation

All defaults run on CPU, need no paid API, no cloud upload and no account or Hugging Face login.
Verified on: MacBook Pro, Intel Core i7-6700HQ (4 cores, AVX2), 16 GB RAM, macOS 14.8 (Darwin 23.6), Python 3.12.13,
ffmpeg 8.1 (Homebrew).
Also checked on Debian 13.7 (x86_64, in a container), Python 3.13.5, ffmpeg 7.1.5: the same pinned libraries install,
the test suite passes, and a synthetic two-person call gives the same notes at the same times as on the Mac
(audio-only; transcription was not tried there).

## Default components (installed by `./install.sh`)

| Component | Model / library | Size | License | Source |
|---|---|---|---|---|
| Audio-event detector | **YAMNet** (MobileNet-v1, AudioSet 521 classes), original `yamnet.h5` weights run by our NumPy port `yamnet_np.py` | 15.3 MB | Apache-2.0 (Google) | storage.googleapis.com/audioset/yamnet.h5 |
| YAMNet class map | `yamnet_class_map.csv` (bundled in the package) | 14 KB | Apache-2.0 | tensorflow/models |
| Voice activity | Silero VAD (ONNX) via sherpa-onnx | 0.6 MB | MIT | k2-fsa/sherpa-onnx release mirror |
| Speaker embeddings | 3D-Speaker CAM++ `zh_en 16k common advanced` (ONNX) via sherpa-onnx | 28.3 MB | Apache-2.0 | k2-fsa/sherpa-onnx release mirror |
| Runtime | sherpa-onnx 2.0.1 (bundles onnxruntime), numpy 2.5.5, scipy 1.18.1, h5py 3.16.0 | ~60 MB of wheels | Apache-2.0 / BSD | PyPI |
| Decoding | ffmpeg / ffprobe (system) | - | LGPL/GPL (not bundled) | Homebrew |

Every model file is pinned by byte size and SHA-256 in `src/audio_sales_coach/models.py` and verified on install and by
`audio-sales-coach selftest`. Downloads use `curl` (system certificates, resume, stall detection) because the
Homebrew Python's `urllib` had no CA bundle on this Mac.

## Optional components (explicit flags)

| Flag | What | Size | Access | Limitations |
|---|---|---|---|---|
| `--with-transcription` | faster-whisper 1.2.1 (CTranslate2, int8 CPU); model `small` by default (`transcription.model`) | ~140 MB wheels + ~480 MB model on first use (`base` ~145 MB, `medium` ~1.5 GB) | public Hugging Face repo `Systran/faster-whisper-*`, **no login** | slow on this CPU (roughly real-time or slower for `small`); Lebanese Arabic and Arabic/English code-switching are often mis-transcribed or translated; one language per 30 s window |
| `--with-ced` | CED-mini audio tagger (AudioSet 527 classes) via sherpa-onnx, for cross-checking YAMNet (`detector: "ced"`) | 47.7 MB | Apache-2.0, GitHub release, no login | sliding 1 s windows, slower than YAMNet |
| `--with-tensorflow` | separate venv with TensorFlow 2.16.2 + tf-keras 2.16.0, only for `scripts/compare_yamnet_tf.py` | ~260 MB wheel, ~1.3 GB installed | PyPI | see below |
| (not provided) | pyannote segmentation / gated diarization models | - | Hugging Face gated (accept terms + login) | not used; mono diarization uses non-gated CAM++ embeddings + clustering |

## YAMNet on this machine: TensorFlow/Keras compatibility (tested)

| Question | Result on this Intel Mac |
|---|---|
| Latest TensorFlow with Intel-macOS wheels | **2.16.2** (cp312 `macosx_10_15_x86_64`, 260 MB). Later versions publish no x86_64 macOS wheels. |
| Installs with numpy 2? | No. TF 2.16.2 pins numpy < 2 (installed 1.26.4), which conflicts with the rest of a modern stack. |
| Default Keras with TF 2.16.2 | Keras 3 (pip resolved keras 3.15.1). |
| Official YAMNet code (`tensorflow/models/research/audioset/yamnet`) under Keras 3 | **Fails**: `ValueError: A KerasTensor cannot be used as input to a TensorFlow function` while building the frames model (the code applies `tf.signal` ops to Keras symbolic tensors). |
| Official code with `tf-keras==2.16.0` (Keras 2) | **Works**; loads `yamnet.h5` in 3.8 s. |
| Download | the 260 MB wheel stalled twice with SSL errors on this network before a resumable download completed. |
| `ai-edge-litert` / `tflite-runtime` for the TFLite YAMNet | no wheels for Intel macOS. |

Decision: the default install runs the **same YAMNet weights** through a NumPy re-implementation of the official
front end (periodic-Hann STFT 25/10 ms, 64-band HTK mel 125-7500 Hz, log(mel + 0.001), 0.96 s patches with 0.48 s hop,
official padding) and network (14 conv/depthwise-separable layers, batch-norm folded, 'same' padding semantics).

Equivalence, measured with `scripts/compare_yamnet_tf.py` against TF 2.16.2 + tf-keras 2.16.0
(`docs/results/yamnet-tf-vs-numpy.json`):

| Audio | Patches TF / NumPy | Max abs score diff | Top-1 class agreement |
|---|---|---|---|
| 101 s synthetic two-person call | 211 / 211 | 1.5e-5 | 100 % |
| ESC-50 laughing clip 1-1791-A-26 | 10 / 10 | 6.4e-6 | 100 % |
| ESC-50 laughing clip 1-30039-A-26 | 10 / 10 | 8.7e-6 | 100 % |
| ESC-50 breathing clip 1-18631-A-23 | 10 / 10 | 1.5e-5 | 100 % |

Speed on this CPU: NumPy YAMNet ≈ 3.2 s per 100 s of audio (TF: ≈ 1.1 s). A 1-hour call takes about 2 minutes for
event detection per channel.

## Is YAMNet reliable enough here?

Measured on ESC-50 (CC BY-NC 3.0) clips, thresholds chosen on fold 1 and tested on held-out fold 2
(`docs/evaluation.md`):

* **Laughter**: usable. AUC 0.97, precision 0.70, recall 0.88 at threshold 0.25; no false alarms on TTS speech;
  confused with crying baby. Some ESC-50 "laughing" clips are scored as breathing/whimpering instead.
* **Audible breathing**: weak. AUC 0.83, precision 0.35, recall 0.75 at 0.60; confused with snoring, sneezing,
  drinking. Breathing cues are therefore always marked uncertain and phrased as "replay to check".

YAMNet stays the default detector; CED-mini is available as an optional cross-check (`--with-ced`,
`detector: "ced"`). Event detection is never replaced by a language model.

## Speaker-embedding thresholds

Defaults: cluster cut 0.55 cosine distance; voice-ID accept ≥ 0.60, reject < 0.40 cosine similarity (uncertain in
between). These are provisional values from synthetic TTS voices, not calibrated on real salespeople. Run
`audio-sales-coach voice calibrate NAME` after ≥ 3 enrolment samples and ≥ 3 confidently matched calls to derive
per-profile thresholds from your own data. CAM++ was trained on Chinese and English; Arabic speakers are not
specifically covered.

## Attributions

See `THIRD_PARTY_NOTICES.md`.
