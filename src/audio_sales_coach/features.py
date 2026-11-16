"""Waveform measurements with recorded methods.

All times are seconds from the start of the original recording. Every measurement here is computed
directly from samples (no model), and each function documents its method so reports can cite it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

HOP_S = 0.010

METHODS = {
    "energy": "RMS level per 25 ms frame, 10 ms hop, in dBFS (20*log10 rms)",
    "pitch": ("normalised autocorrelation pitch tracker, 40 ms frames, 10 ms hop, 70-400 Hz search, "
              "voiced when peak correlation >= 0.5, 5-frame median smoothing; octave errors are possible "
              "on compressed or noisy audio"),
    "pace": ("estimated syllable rate: peaks of the smoothed 300-2500 Hz intensity contour with >= 1 dB prominence, "
             ">= 70 ms apart, inside speech (after de Jong & Wempe 2009); language-independent proxy, not a word count; "
             "on TTS test speech it correlates 0.83 with true syllable rate and under-counts very fast speech"),
    "pauses": "silence between speech segments of the same speaker, from the voice-activity detector",
    "response_gap": "time from the end of one speaker's turn to the start of the other speaker's next turn",
    "pitch_change": "median F0 of a turn in semitones relative to the same speaker's median F0 in this recording",
    "energy_change": "median dB level of a turn relative to the same speaker's median level in this recording",
}


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    return sliding_window_view(x, win)[::hop]


def rms_db(x: np.ndarray, sr: int, win_s: float = 0.025, hop_s: float = HOP_S) -> np.ndarray:
    win, hop = int(sr * win_s), int(sr * hop_s)
    fr = frames(x, win, hop)
    # chunked to bound memory on long calls
    out = np.empty(fr.shape[0], dtype=np.float32)
    for i in range(0, fr.shape[0], 20000):
        c = fr[i:i + 20000].astype(np.float64)
        out[i:i + 20000] = 10 * np.log10(np.mean(c * c, axis=1) + 1e-10)
    return out


def pitch_track(x: np.ndarray, sr: int, active: np.ndarray | None = None, hop_s: float = HOP_S,
                fmin: float = 70.0, fmax: float = 400.0, voicing: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz with NaN when unvoiced, peak correlation) per 10 ms frame.

    `active` (bool per frame) restricts computation to frames worth analysing (speech)."""
    win, hop = int(sr * 0.040), int(sr * hop_s)
    fr = frames(x, win, hop)
    n = fr.shape[0]
    f0 = np.full(n, np.nan, dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    lag_min, lag_max = int(sr / fmax), min(int(sr / fmin), win - 2)
    idx = np.arange(n) if active is None else np.flatnonzero(active[:n])
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    taper = win / np.maximum(win - np.arange(lag_max + 2), 1)
    for s in range(0, idx.size, 4096):
        sel = idx[s:s + 4096]
        c = fr[sel].astype(np.float32)
        c = c - c.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(c, nfft, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, : lag_max + 2]
        r0 = ac[:, :1]
        ok = r0[:, 0] > 1e-7
        nac = np.where(ok[:, None], ac / np.maximum(r0, 1e-12), 0.0) * taper[None, :]
        band = nac[:, lag_min:lag_max + 1]
        best = band.max(axis=1)
        # prefer the shortest lag close to the best peak (guards against sub-harmonic errors)
        cand = band >= (0.88 * best[:, None])
        lag = np.argmax(cand, axis=1) + lag_min
        # parabolic refinement
        li = np.clip(lag, 1, lag_max)
        y0, y1, y2 = nac[np.arange(len(li)), li - 1], nac[np.arange(len(li)), li], nac[np.arange(len(li)), li + 1]
        denom = y0 - 2 * y1 + y2
        delta = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / denom, 0.0)
        period = li + np.clip(delta, -0.5, 0.5)
        voiced = ok & (best >= voicing)
        f0[sel[voiced]] = sr / period[voiced]
        conf[sel] = np.where(ok, best, 0.0)
    if np.isfinite(f0).sum() > 5:
        filled = np.where(np.isfinite(f0), f0, 0.0)
        med = median_filter(filled, size=5)
        f0 = np.where(np.isfinite(f0) & (med > 0), med, np.nan).astype(np.float32)
    return f0, conf


def energy_vad(db: np.ndarray, min_speech_frames: int = 20, hang_frames: int = 15) -> np.ndarray:
    """Fallback voice-activity mask from frame energy (used only when the neural VAD is unavailable)."""
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    floor, top = np.percentile(db, 10), np.percentile(db, 97)
    thr = floor + max(6.0, 0.35 * (top - floor))
    m = db > thr
    # hangover: bridge short gaps
    m = np.convolve(m.astype(float), np.ones(hang_frames), mode="same") > 0
    return _drop_short(m, min_speech_frames)


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def band_db(x: np.ndarray, sr: int, lo: float = 300.0, hi: float = 2500.0, hop_s: float = HOP_S) -> np.ndarray:
    """dB contour of the 300-2500 Hz band (vowel energy), 20 ms frames."""
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64)) if x.size > 64 else x.astype(np.float64)
    fr = frames(y, int(0.02 * sr), int(hop_s * sr))
    out = np.empty(fr.shape[0], dtype=np.float32)
    for i in range(0, fr.shape[0], 20000):
        c = fr[i:i + 20000]
        out[i:i + 20000] = 10 * np.log10(np.mean(c * c, axis=1) + 1e-10)
    return out


def syllable_nuclei(bdb: np.ndarray, active: np.ndarray, hop_s: float = HOP_S) -> np.ndarray:
    """Times (s) of estimated syllable nuclei (peaks of the vowel-band intensity) within active frames."""
    n = min(bdb.size, active.size)
    bdb, active = bdb[:n], active[:n]
    if n < 5 or not active.any():
        return np.zeros(0)
    k = np.hanning(5)
    sm = np.convolve(bdb, k / k.sum(), mode="same")
    floor = np.median(sm[active]) - 12.0
    peaks, _ = find_peaks(sm, prominence=1.0, distance=int(0.07 / hop_s))
    keep = [p for p in peaks if active[p] and sm[p] > floor]
    return np.asarray(keep, dtype=float) * hop_s


@dataclass
class ChannelFeatures:
    db: np.ndarray
    f0: np.ndarray
    f0_conf: np.ndarray
    nuclei_t: np.ndarray
    hop_s: float = HOP_S

    def idx(self, t: float) -> int:
        return int(round(t / self.hop_s))

    def segment_stats(self, start: float, end: float) -> dict:
        a, b = self.idx(start), max(self.idx(end), self.idx(start) + 1)
        f0 = self.f0[a:b]
        f0v = f0[np.isfinite(f0)]
        db = self.db[a:b]
        dur = max(end - start, 1e-6)
        nuc = int(((self.nuclei_t >= start) & (self.nuclei_t < end)).sum())
        return {
            "duration_s": round(dur, 3),
            "median_f0_hz": round(float(np.median(f0v)), 1) if f0v.size >= 5 else None,
            "f0_range_semitones": round(float(12 * np.log2(np.percentile(f0v, 90) / np.percentile(f0v, 10))), 2)
            if f0v.size >= 10 else None,
            "voiced_fraction": round(float(f0v.size / max(f0.size, 1)), 3),
            "median_db": round(float(np.median(db)), 1) if db.size else None,
            "syllable_rate_per_s": round(nuc / dur, 2) if dur >= 1.0 else None,
            "syllable_count": nuc,
        }

    def final_pitch_slope(self, start: float, end: float, tail_s: float = 0.5) -> float | None:
        """Semitones per second over the last voiced `tail_s` of a turn (rising ending cue)."""
        a, b = self.idx(max(start, end - tail_s - 0.3)), self.idx(end)
        f0 = self.f0[a:b]
        t = np.arange(f0.size) * self.hop_s
        ok = np.isfinite(f0)
        if ok.sum() < 8:
            return None
        st = 12 * np.log2(f0[ok] / np.median(f0[ok]))
        tt = t[ok][-int(tail_s / self.hop_s):]
        st = st[-int(tail_s / self.hop_s):]
        if tt.size < 6 or np.ptp(tt) < 0.15:
            return None
        return float(np.polyfit(tt, st, 1)[0])


def channel_features(x: np.ndarray, sr: int, speech_mask: np.ndarray | None = None) -> ChannelFeatures:
    db = rms_db(x, sr)
    active = speech_mask[: db.size] if speech_mask is not None else energy_vad(db)
    if active.size < db.size:
        active = np.pad(active, (0, db.size - active.size))
    f0, conf = pitch_track(x, sr, active=active)
    n = min(db.size, f0.size)
    db, f0, conf, active = db[:n], f0[:n], conf[:n], active[:n]
    return ChannelFeatures(db=db, f0=f0, f0_conf=conf, nuclei_t=syllable_nuclei(band_db(x, sr), active))
