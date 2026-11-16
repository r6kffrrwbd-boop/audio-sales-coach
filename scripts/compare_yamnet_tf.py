"""Cross-check the NumPy YAMNet runner against Google's official TensorFlow/Keras YAMNet.

Needs a separate environment with TensorFlow (./install.sh --with-tensorflow creates one) and the official
model code from tensorflow/models (Apache-2.0), which this script downloads to a temp folder:
  research/audioset/yamnet/{yamnet.py,params.py,features.py}

usage: python scripts/compare_yamnet_tf.py WEIGHTS_H5 AUDIO.wav [AUDIO2.wav ...] [--keras3]
  --keras3   patch the official code to use Keras 3 (tf.keras) instead of tf_keras, to test compatibility
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keras3 = "--keras3" in sys.argv
    weights, audio = args[0], args[1:]
    td = Path(tempfile.mkdtemp())
    for f in ("yamnet.py", "params.py", "features.py"):
        subprocess.run(["curl", "-fsSL", "-o", str(td / f), RAW + f], check=True)
    if keras3:
        p = td / "yamnet.py"
        p.write_text(p.read_text().replace("from tf_keras import Model, layers", "from tensorflow.keras import Model, layers"))
    sys.path.insert(0, str(td))
    import numpy as np
    import tensorflow as tf
    import params as yp
    import yamnet as ym

    from audio_sales_coach.audio_io import decode
    from audio_sales_coach.events import class_map_path
    from audio_sales_coach.yamnet_np import YamnetNumpy

    out = {"tensorflow": tf.__version__, "numpy": np.__version__, "keras_mode": "keras3 (tf.keras)" if keras3 else "tf_keras"}
    try:
        import keras
        out["keras"] = keras.__version__
    except ImportError:
        pass
    t0 = time.time()
    model = ym.yamnet_frames_model(yp.Params())
    model.load_weights(weights)
    out["tf_load_s"] = round(time.time() - t0, 2)
    npm = YamnetNumpy(Path(weights), class_map_path())
    out["files"] = []
    for a in audio:
        x = decode(a).mono
        t0 = time.time()
        scores_tf, _, _ = model(x)
        scores_tf = scores_tf.numpy()
        t_tf = time.time() - t0
        t0 = time.time()
        _, scores_np = npm.scores(x)
        t_np = time.time() - t0
        n = min(len(scores_tf), len(scores_np))
        d = np.abs(scores_tf[:n] - scores_np[:n])
        top_tf, top_np = scores_tf[:n].argmax(1), scores_np[:n].argmax(1)
        out["files"].append({"file": Path(a).name, "patches_tf": int(len(scores_tf)), "patches_np": int(len(scores_np)),
                             "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                             "top1_agreement": float((top_tf == top_np).mean()), "tf_s": round(t_tf, 2), "np_s": round(t_np, 2)})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
