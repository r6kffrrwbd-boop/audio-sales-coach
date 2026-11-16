# Third-party notices

audio-sales-coach's own code is MIT-licensed (`LICENSE`). It bundles one third-party file and downloads others at
install or evaluation time. No model weights or audio are committed to this repository.

## Bundled

* `src/audio_sales_coach/data/yamnet_class_map.csv` - from tensorflow/models,
  `research/audioset/yamnet/yamnet_class_map.csv`. Copyright Google LLC, Apache License 2.0.

## Re-implemented (not copied)

* `src/audio_sales_coach/yamnet_np.py` re-expresses in NumPy the YAMNet feature extraction and network definition
  described by tensorflow/models `research/audioset/yamnet/{params.py,features.py,yamnet.py}`
  (Copyright Google LLC, Apache License 2.0) and the mel matrix of `tf.signal.linear_to_mel_weight_matrix`
  (TensorFlow, Apache License 2.0). `scripts/compare_yamnet_tf.py` downloads the original files at run time only.

## Downloaded at install time

| File | Project | License |
|---|---|---|
| `yamnet.h5` | YAMNet, Google (tensorflow/models) | Apache-2.0 |
| `silero_vad.onnx` | Silero VAD (snakers4/silero-vad), mirrored by k2-fsa/sherpa-onnx | MIT |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 3D-Speaker CAM++ (modelscope/3D-Speaker), mirrored by k2-fsa/sherpa-onnx | Apache-2.0 |
| optional `sherpa-onnx-ced-mini-audio-tagging-2024-04-19` | CED (RicherMans/CED), mirrored by k2-fsa/sherpa-onnx | Apache-2.0 |
| optional faster-whisper models | OpenAI Whisper weights converted by Systran | MIT |

Python packages (installed from PyPI, not bundled): sherpa-onnx (Apache-2.0), numpy (BSD-3), scipy (BSD-3),
h5py (BSD-3); optional faster-whisper (MIT), ctranslate2 (MIT), TensorFlow (Apache-2.0).
System tool: ffmpeg (LGPL/GPL), invoked as a separate program.

## Evaluation data (downloaded to a local cache only, never committed)

* **ESC-50** - K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
  https://github.com/karolpiczak/ESC-50. Clips are from freesound.org, licensed **CC BY-NC 3.0** by their authors
  (per-clip attribution in the dataset's `meta/esc50.csv`). Used only for non-commercial evaluation.
* Synthetic speech is generated locally with the macOS `say` voices at test time and is not redistributed.
