"""YAMNet inference in NumPy.

YAMNet (Google, Apache-2.0) is a MobileNet-v1 audio-event classifier trained on AudioSet (521 classes).
This module runs the *original released Keras weights* (`yamnet.h5`) with the same log-mel front end and
layer definitions as `tensorflow/models/research/audioset/yamnet` (params.py, features.py, yamnet.py),
re-expressed in NumPy so the default install does not need TensorFlow (the last TensorFlow with Intel-macOS
wheels is 2.16.2; see docs/models.md for the compatibility evaluation).

Equivalence with the TensorFlow implementation is checked by `scripts/compare_yamnet_tf.py` when the
optional TensorFlow runtime is installed (`./install.sh --with-tensorflow`).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

# params.py
SAMPLE_RATE = 16000.0
STFT_WINDOW_S = 0.025
STFT_HOP_S = 0.010
MEL_BANDS = 64
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7500.0
LOG_OFFSET = 0.001
PATCH_WINDOW_S = 0.96
PATCH_HOP_S = 0.48
BN_EPS = 1e-4

LAYER_DEFS = [  # (kind, stride, filters) - all kernels 3x3
    ("conv", 2, 32), ("sep", 1, 64), ("sep", 2, 128), ("sep", 1, 128), ("sep", 2, 256), ("sep", 1, 256),
    ("sep", 2, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512), ("sep", 1, 512),
    ("sep", 2, 1024), ("sep", 1, 1024),
]


def _hz_to_mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_matrix(num_spec_bins: int = 257) -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix (HTK mel, DC bin zeroed)."""
    nyq = SAMPLE_RATE / 2.0
    lin = np.linspace(0.0, nyq, num_spec_bins)[1:]
    spec_mel = _hz_to_mel(lin)[:, None]
    edges = np.linspace(_hz_to_mel(MEL_MIN_HZ), _hz_to_mel(MEL_MAX_HZ), MEL_BANDS + 2)
    lo, ce, hi = edges[:-2][None, :], edges[1:-1][None, :], edges[2:][None, :]
    w = np.maximum(0.0, np.minimum((spec_mel - lo) / (ce - lo), (hi - spec_mel) / (hi - ce)))
    return np.pad(w, [[1, 0], [0, 0]]).astype(np.float32)


def pad_waveform(x: np.ndarray) -> np.ndarray:
    sr = SAMPLE_RATE
    min_n = int((PATCH_WINDOW_S + STFT_WINDOW_S - STFT_HOP_S) * sr)
    n = x.size
    pad = max(0, min_n - n)
    n = max(n, min_n)
    hop = int(PATCH_HOP_S * sr)
    hops_after = math.ceil((n - min_n) / hop)
    pad += hop * hops_after - (n - min_n)
    return np.pad(x.astype(np.float32), (0, pad))


def log_mel_patches(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_mel [frames, 64], patches [P, 96, 64])."""
    win = int(round(SAMPLE_RATE * STFT_WINDOW_S))  # 400
    hop = int(round(SAMPLE_RATE * STFT_HOP_S))  # 160
    nfft = 2 ** int(math.ceil(math.log2(win)))  # 512
    xp = pad_waveform(x)
    n_frames = 1 + (xp.size - win) // hop
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float32)  # periodic Hann
    mel = mel_matrix(nfft // 2 + 1)
    out = np.empty((n_frames, MEL_BANDS), dtype=np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(xp, win)[::hop][:n_frames]
    for i in range(0, n_frames, 8192):
        mag = np.abs(np.fft.rfft(fr[i:i + 8192] * window, nfft, axis=1)).astype(np.float32)
        out[i:i + 8192] = np.log(mag @ mel + LOG_OFFSET)
    pw = int(round((SAMPLE_RATE / hop) * PATCH_WINDOW_S))  # 96
    ph = int(round((SAMPLE_RATE / hop) * PATCH_HOP_S))  # 48
    n_p = 1 + (n_frames - pw) // ph if n_frames >= pw else 0
    patches = np.lib.stride_tricks.sliding_window_view(out, (pw, MEL_BANDS))[::ph, 0][:n_p]
    return out, np.ascontiguousarray(patches)


def _same_pad(x: np.ndarray, stride: int, k: int = 3) -> np.ndarray:
    _, h, w, _ = x.shape
    pads = []
    for n in (h, w):
        total = max((math.ceil(n / stride) - 1) * stride + k - n, 0)
        pads.append((total // 2, total - total // 2))
    return np.pad(x, [(0, 0), pads[0], pads[1], (0, 0)])


def _out(n: int, stride: int) -> int:
    return math.ceil(n / stride)


class YamnetNumpy:
    name = "yamnet"

    def __init__(self, weights_h5: Path, class_map_csv: Path):
        import h5py  # small dependency; only needed to read the released weights

        arrays: dict[str, np.ndarray] = {}
        with h5py.File(weights_h5, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    arrays[name] = np.asarray(obj[()])
            f.visititems(visit)
        self.layers = []
        for i, (kind, stride, filters) in enumerate(LAYER_DEFS, start=1):
            if kind == "conv":
                k = self._get(arrays, f"layer{i}/conv/kernel")
                scale, shift = self._bn(arrays, f"layer{i}/conv/bn")
                self.layers.append(("conv", stride, (k * scale).reshape(-1, k.shape[-1]).astype(np.float32),
                                    shift.astype(np.float32)))
            else:
                dk = self._get(arrays, f"layer{i}/depthwise_conv/depthwise_kernel")[..., 0]  # (3,3,C)
                ds, dsh = self._bn(arrays, f"layer{i}/depthwise_conv/bn")
                pk = self._get(arrays, f"layer{i}/pointwise_conv/kernel")[0, 0]  # (Cin, Cout)
                ps, psh = self._bn(arrays, f"layer{i}/pointwise_conv/bn")
                self.layers.append(("sep", stride, (dk * ds).astype(np.float32), dsh.astype(np.float32),
                                    (pk * ps).astype(np.float32), psh.astype(np.float32)))
        dense_k = [n for n in arrays if n.endswith("kernel:0") and arrays[n].shape == (1024, 521)]
        dense_b = [n for n in arrays if n.endswith("bias:0") and arrays[n].shape == (521,)]
        if len(dense_k) != 1 or len(dense_b) != 1:
            raise ValueError("unexpected yamnet.h5 layout (classifier layer not found)")
        self.dense_w = arrays[dense_k[0]].astype(np.float32)
        self.dense_b = arrays[dense_b[0]].astype(np.float32)
        with open(class_map_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.class_names = [r["display_name"] for r in rows]
        if len(self.class_names) != 521:
            raise ValueError("class map must have 521 rows")

    @staticmethod
    def _get(arrays: dict, stem: str) -> np.ndarray:
        hits = [n for n in arrays if n in (stem, stem + ":0") or n.endswith("/" + stem + ":0")]
        if not hits:
            raise KeyError(f"weight {stem} not found in yamnet.h5")
        return np.asarray(arrays[sorted(hits, key=len)[0]], dtype=np.float64)

    def _bn(self, arrays: dict, stem: str) -> tuple[np.ndarray, np.ndarray]:
        beta = self._get(arrays, f"{stem}/beta")
        mean = self._get(arrays, f"{stem}/moving_mean")
        var = self._get(arrays, f"{stem}/moving_variance")
        scale = 1.0 / np.sqrt(var + BN_EPS)  # batchnorm_scale=False: no gamma
        return scale, beta - mean * scale

    def _forward(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = patches[..., None].astype(np.float32)  # (B, 96, 64, 1)
        for layer in self.layers:
            stride = layer[1]
            b, h, w, c = x.shape
            oh, ow = _out(h, stride), _out(w, stride)
            xp = _same_pad(x, stride)
            if layer[0] == "conv":
                _, _, kmat, shift = layer
                cols = np.concatenate([xp[:, dy:dy + stride * oh:stride, dx:dx + stride * ow:stride, :]
                                       for dy in range(3) for dx in range(3)], axis=-1)
                x = np.maximum(cols.reshape(-1, cols.shape[-1]) @ kmat + shift, 0).reshape(b, oh, ow, -1)
            else:
                _, _, dk, dsh, pk, psh = layer
                acc = np.zeros((b, oh, ow, c), dtype=np.float32)
                for dy in range(3):
                    for dx in range(3):
                        acc += xp[:, dy:dy + stride * oh:stride, dx:dx + stride * ow:stride, :] * dk[dy, dx]
                acc = np.maximum(acc + dsh, 0)
                x = np.maximum(acc.reshape(-1, c) @ pk + psh, 0).reshape(b, oh, ow, -1)
        emb = x.mean(axis=(1, 2))
        logits = emb @ self.dense_w + self.dense_b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))), emb

    def scores(self, x: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Return (patch start times in s, scores [P, 521]) for 16 kHz mono float audio."""
        _, patches = log_mel_patches(x)
        out = np.zeros((patches.shape[0], 521), dtype=np.float32)
        for i in range(0, patches.shape[0], batch):
            out[i:i + batch] = self._forward(patches[i:i + batch])[0]
        return np.arange(patches.shape[0]) * PATCH_HOP_S, out
