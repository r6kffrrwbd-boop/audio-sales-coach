"""Speech segments and speaker turns.

* Separate call channels (stereo where each side is on its own channel): turns come from per-channel voice
  activity; overlaps are real simultaneous speech on both channels; attribution is strong.
* Mixed mono: speaker embeddings (CAM++) on short windows, clustered; every segment carries an attribution
  confidence and an `uncertain` flag; quick speaker changes without a pause are marked as possible overlap.
  No source separation is used, so the original recording remains the only evidence.
* Single-speaker voice messages: one speaker; a second person is never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from . import features, models

FRAME = features.HOP_S


@dataclass
class Segment:
    start: float
    end: float
    speaker: str  # "S1", "S2", or "unknown"
    confidence: float  # attribution confidence 0..1
    channel: int | None = None
    overlap: bool = False  # simultaneous speech with the other speaker (separate channels only)
    possible_overlap: bool = False  # mixed mono: rapid change without a pause; cannot be confirmed
    uncertain: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerResult:
    layout: str  # separate_channels | mixed_mono | single_speaker
    segments: list[Segment]
    speakers: list[str]
    centroids: dict[str, np.ndarray] = field(default_factory=dict)  # in memory only; never written
    overlaps: list[tuple[float, float]] = field(default_factory=list)
    method: str = ""
    notes: list[str] = field(default_factory=list)
    channel_of: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- voice activity
class Vad:
    def __init__(self, threshold: float = 0.5, min_speech: float = 0.25, min_silence: float = 0.25):
        self.method = "energy"
        self.vad_cfg = None
        if models.is_installed("silero_vad"):
            import sherpa_onnx
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(models.path_for("silero_vad"))
            cfg.silero_vad.threshold = threshold
            cfg.silero_vad.min_speech_duration = min_speech
            cfg.silero_vad.min_silence_duration = min_silence
            cfg.silero_vad.max_speech_duration = 30
            cfg.sample_rate = 16000
            cfg.num_threads = 1
            self.vad_cfg = cfg
            self.method = "silero-vad (onnx, via sherpa-onnx)"

    def segments(self, x: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]:
        if self.vad_cfg is None:
            db = features.rms_db(x, sr)
            return [(s * FRAME, e * FRAME) for s, e in features.runs(features.energy_vad(db))]
        import sherpa_onnx
        vad = sherpa_onnx.VoiceActivityDetector(self.vad_cfg, buffer_size_in_seconds=60)
        ws = self.vad_cfg.silero_vad.window_size
        out = []

        def drain():
            while not vad.empty():
                seg = vad.front
                out.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
                vad.pop()
        for i in range(0, x.size, ws):
            vad.accept_waveform(x[i:i + ws])
            drain()
        vad.flush()
        drain()
        return merge_close(sorted(out), gap=0.15)


def merge_close(segs, gap: float):
    out = []
    for s, e in segs:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def mask_from(segs, n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s, e in segs:
        m[int(s / FRAME):int(np.ceil(e / FRAME))] = True
    return m


# ---------------------------------------------------------------- embeddings
class Embedder:
    def __init__(self):
        if not models.is_installed("campplus"):
            raise RuntimeError("speaker-embedding model not installed (run: audio-sales-coach models fetch)")
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(models.path_for("campplus")), num_threads=2)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.name = "3D-Speaker CAM++ (zh/en common, 16 kHz) via sherpa-onnx"

    def embed(self, x: np.ndarray, sr: int = 16000) -> np.ndarray:
        st = self.ex.create_stream()
        st.accept_waveform(sample_rate=sr, waveform=np.ascontiguousarray(x, dtype=np.float32))
        st.input_finished()
        if not self.ex.is_ready(st):
            raise ValueError("audio too short for a speaker embedding")
        v = np.asarray(self.ex.compute(st), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def window_spans(segs, win: float = 1.5, hop: float = 0.75, min_len: float = 0.6):
    spans = []
    for s, e in segs:
        if e - s < min_len:
            continue
        if e - s <= win:
            spans.append((s, e))
            continue
        t = s
        while t + win <= e + 1e-6:
            spans.append((t, t + win))
            t += hop
        if spans[-1][1] < e - 0.3:
            spans.append((max(s, e - win), e))
    return spans


# ---------------------------------------------------------------- layout detection
def detect_layout(channels: np.ndarray, sr: int) -> tuple[str, dict]:
    """'separate_channels' when each side of a call is (mostly) on its own channel."""
    if channels.shape[0] < 2:
        return "mono", {}
    a, b = channels[0], channels[1]
    if np.allclose(a, b, atol=1e-4):
        return "mono", {"reason": "both channels identical (dual mono)"}
    da, db_ = features.rms_db(a, sr), features.rms_db(b, sr)
    act = (da > np.percentile(da, 10) + 10) | (db_ > np.percentile(db_, 10) + 10)
    if act.sum() < 50:
        return "mono", {"reason": "too little activity to judge channels"}
    diff = (da - db_)[act]
    dominant = float(np.mean(np.abs(diff) > 10.0))
    corr = float(np.corrcoef(a[: sr * 60], b[: sr * 60])[0, 1]) if a.size > sr else 1.0
    info = {"frames_with_one_channel_dominant": round(dominant, 3), "waveform_correlation_first_60s": round(corr, 3),
            "share_ch0_louder": round(float(np.mean(diff > 10)), 3), "share_ch1_louder": round(float(np.mean(diff < -10)), 3)}
    if dominant >= 0.6 and info["share_ch0_louder"] > 0.05 and info["share_ch1_louder"] > 0.05:
        return "separate_channels", info
    return "mono", info


# ---------------------------------------------------------------- separate channels
def turns_from_channels(channels: np.ndarray, sr: int, vad: Vad) -> SpeakerResult:
    segs_by_ch = [vad.segments(channels[c], sr) for c in range(2)]
    dbs = [features.rms_db(channels[c], sr) for c in range(2)]
    n = min(d.size for d in dbs)
    out: list[Segment] = []
    notes = []
    dropped = 0
    for c in (0, 1):
        o = 1 - c
        for s, e in segs_by_ch[c]:
            a, b = int(s / FRAME), min(int(e / FRAME), n)
            if b <= a:
                continue
            # crosstalk / bleed: this channel is much quieter than the other over the whole segment
            if np.median(dbs[c][a:b] - dbs[o][a:b]) < -12:
                dropped += 1
                continue
            out.append(Segment(s, e, f"S{c + 1}", 0.95, channel=c))
    if dropped:
        notes.append(f"{dropped} segment(s) ignored as channel bleed (>12 dB quieter than the other channel)")
    out.sort(key=lambda x: x.start)
    overlaps = []
    s1 = [x for x in out if x.speaker == "S1"]
    s2 = [x for x in out if x.speaker == "S2"]
    for x in s1:
        for y in s2:
            lo, hi = max(x.start, y.start), min(x.end, y.end)
            if hi - lo >= 0.1:
                overlaps.append((lo, hi))
                x.overlap = y.overlap = True
    speakers = [sp for sp in ("S1", "S2") if any(x.speaker == sp for x in out)]
    return SpeakerResult("separate_channels" if len(speakers) == 2 else "single_speaker", out, speakers,
                         overlaps=sorted(overlaps), method="per-channel voice activity (" + vad.method + ")",
                         notes=notes, channel_of={"S1": 0, "S2": 1})


# ---------------------------------------------------------------- mixed mono diarization
def diarize_mono(x: np.ndarray, sr: int, vad: Vad, embedder: Embedder | None, mode: str = "auto",
                 cluster_threshold: float = 0.55) -> SpeakerResult:
    """mode: 'call' (expect 2), 'voice-message' (exactly 1), 'auto' (1 or 2 from the audio)."""
    segs = vad.segments(x, sr)
    if not segs:
        return SpeakerResult("single_speaker", [], [], method=vad.method, notes=["no speech detected"])
    if mode == "voice-message" or embedder is None:
        notes = [] if embedder else ["speaker-embedding model missing: treated as one speaker"]
        return SpeakerResult("single_speaker", [Segment(s, e, "S1", 0.9) for s, e in segs], ["S1"],
                             method=vad.method, notes=notes)
    spans = window_spans(segs)
    if len(spans) < 3:
        return SpeakerResult("single_speaker", [Segment(s, e, "S1", 0.6, uncertain=True) for s, e in segs], ["S1"],
                             method=vad.method, notes=["too little speech to separate speakers"])
    emb = np.stack([embedder.embed(x[int(s * sr):int(e * sr)], sr) for s, e in spans])
    Z = linkage(emb, method="average", metric="cosine")
    labels = fcluster(Z, t=cluster_threshold, criterion="distance")
    durs = {lab: sum(e - s for (s, e), l2 in zip(spans, labels) if l2 == lab) for lab in set(labels)}
    total = sum(durs.values())
    main = [lab for lab, d in sorted(durs.items(), key=lambda kv: -kv[1]) if d / total >= 0.08]
    notes = []
    if mode == "call" and len(main) < 2:
        two = fcluster(Z, t=2, criterion="maxclust")
        d2 = {lab: sum(e - s for (s, e), l2 in zip(spans, two) if l2 == lab) for lab in set(two)}
        cents = [emb[two == lab].mean(0) for lab in sorted(d2)]
        sim = float(np.dot(cents[0], cents[1]) / (np.linalg.norm(cents[0]) * np.linalg.norm(cents[1])))
        if len(d2) == 2 and min(d2.values()) / total >= 0.05:
            labels, main = two, sorted(d2, key=lambda k: -d2[k])
            notes.append(f"voices are similar (centroid cosine {sim:.2f}); speaker split is uncertain")
    if len(main) > 2:
        notes.append(f"{len(main)} voice clusters found; the two largest are kept, other windows are marked uncertain")
        main = main[:2]
    if len(main) <= 1:
        return SpeakerResult("single_speaker", [Segment(s, e, "S1", 0.85) for s, e in segs], ["S1"],
                             method=f"{vad.method}; {embedder.name}; one voice cluster", notes=notes)
    cents = {f"S{i + 1}": emb[labels == lab].mean(0) for i, lab in enumerate(main)}
    for k in cents:
        cents[k] /= np.linalg.norm(cents[k]) + 1e-9
    names = list(cents)
    C = np.stack([cents[k] for k in names])
    sims = emb @ C.T  # (W, 2)
    # frame-level vote
    n = int(np.ceil(x.size / sr / FRAME)) + 1
    votes = np.zeros((n, 2))
    for (s, e), sv in zip(spans, sims):
        margin = float(sv.max() - sv.min())
        w = np.zeros(2)
        w[int(sv.argmax())] = margin
        votes[int(s / FRAME):int(e / FRAME)] += w
    speech = mask_from(segs, n)
    out: list[Segment] = []
    for s, e in segs:
        a, b = int(s / FRAME), int(np.ceil(e / FRAME))
        v = votes[a:b]
        lab = np.where(v.sum(1) > 0, v.argmax(1), -1)
        # fill frames with no window (very short segments) from the nearest labelled frame
        if (lab < 0).all():
            emb_s = embedder.embed(x[int(s * sr):int(e * sr)], sr) if e - s >= 0.3 else None
            if emb_s is None:
                out.append(Segment(s, e, "unknown", 0.0, uncertain=True))
                continue
            sv = C @ emb_s
            out.append(Segment(s, e, names[int(sv.argmax())], float(np.clip((sv.max() - sv.min()) / 0.3, 0, 1)) * 0.8,
                               uncertain=bool(sv.max() - sv.min() < 0.1)))
            continue
        for i in range(lab.size):
            if lab[i] < 0:
                lab[i] = lab[i - 1] if i > 0 and lab[i - 1] >= 0 else lab[lab >= 0][0]
        conf_fr = np.abs(v[:, 0] - v[:, 1]) / (v.sum(1) + 1e-9)
        for rs, re_ in _label_runs(lab):
            seg_s, seg_e = s + rs * FRAME, min(e, s + re_ * FRAME)
            if seg_e - seg_s < 0.05:
                continue
            margin_conf = float(np.clip(np.mean(np.max(v[rs:re_], 1) / (np.max(v[rs:re_], 1).max() + 1e-9)), 0, 1))
            conf = float(np.clip(0.5 * np.mean(conf_fr[rs:re_]) + 0.5 * margin_conf, 0, 1)) * 0.85  # mono cap
            out.append(Segment(seg_s, seg_e, names[int(lab[rs])], round(conf, 3), uncertain=conf < 0.45))
    out = _absorb_short(out)
    for a, b in zip(out, out[1:]):
        if a.speaker != b.speaker and b.start - a.end < 0.15:
            a.possible_overlap = b.possible_overlap = True
    _ = speech
    return SpeakerResult("mixed_mono", out, names, centroids=cents,
                         method=f"{vad.method}; {embedder.name}; 1.5 s windows, 0.75 s hop, average-link cosine "
                                f"clustering (cut {cluster_threshold}); frame vote by similarity margin",
                         notes=notes)


def _label_runs(lab: np.ndarray):
    out, s = [], 0
    for i in range(1, lab.size + 1):
        if i == lab.size or lab[i] != lab[s]:
            out.append((s, i))
            s = i
    return out


def _absorb_short(segs: list[Segment], min_len: float = 0.3) -> list[Segment]:
    out: list[Segment] = []
    for sg in segs:
        if out and sg.duration < min_len and out[-1].end >= sg.start - 0.05:
            out[-1].end = sg.end
            out[-1].uncertain = True
            continue
        if out and out[-1].speaker == sg.speaker and sg.start - out[-1].end < 0.05:
            out[-1].end = sg.end
            out[-1].confidence = round(min(out[-1].confidence, sg.confidence), 3)
            out[-1].uncertain = out[-1].uncertain or sg.uncertain
            continue
        out.append(sg)
    return out


def speaker_embedding_for(x: np.ndarray, sr: int, segs: list[Segment], speaker: str, embedder: Embedder,
                          max_s: float = 60.0) -> np.ndarray | None:
    """Mean embedding over a speaker's confident segments (used for voice ID); kept in memory only."""
    spans, tot = [], 0.0
    for s in sorted((s for s in segs if s.speaker == speaker and not s.uncertain and not s.overlap),
                    key=lambda s: -s.duration):
        spans.append((s.start, s.end))
        tot += s.duration
        if tot >= max_s:
            break
    vecs = []
    for s, e in window_spans(spans, win=3.0, hop=3.0, min_len=1.0):
        try:
            vecs.append(embedder.embed(x[int(s * sr):int(e * sr)], sr))
        except ValueError:
            pass
    if not vecs:
        return None
    v = np.mean(vecs, 0)
    return v / (np.linalg.norm(v) + 1e-9)
