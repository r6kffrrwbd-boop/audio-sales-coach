"""Audio-event detection (laughter, audible breathing, and context classes).

Events come only from an audio classifier run on the waveform - never from a language model or from
transcript words. Default: YAMNet (NumPy runner over the released weights). Optional: CED-mini via
sherpa-onnx as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from . import models

# AudioSet display names grouped into the cues the coach reports.
GROUPS: dict[str, list[str]] = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "breathing": ["Breathing", "Gasp", "Pant", "Sigh", "Wheeze", "Snort"],
    "speech": ["Speech", "Conversation", "Narration, monologue", "Male speech, man speaking",
               "Female speech, woman speaking", "Child speech, kid speaking"],
    "cough_throat": ["Cough", "Throat clearing", "Sneeze", "Sniff"],
    "music": ["Music"],
    "noise": ["Noise", "Static", "Mains hum", "Hubbub, speech noise, speech babble", "Inside, small room",
              "Vehicle", "Traffic noise, roadway noise", "Wind"],
}
REPORTED = ("laughter", "breathing", "cough_throat")

# Thresholds on the group score (max of member-class sigmoid scores), chosen on ESC-50 fold 1 and checked on
# fold 2 (docs/evaluation.md). Breathing is a weak class in AudioSet; its notes are always marked uncertain.
DEFAULT_THRESHOLDS = {"yamnet": {"laughter": 0.25, "breathing": 0.60, "cough_throat": 0.35},
                      "ced": {"laughter": 0.30, "breathing": 0.20, "cough_throat": 0.35}}


@dataclass
class EventTrack:
    detector: str
    model_file: str
    hop_s: float
    window_s: float
    times: np.ndarray  # window start times
    group_scores: dict[str, np.ndarray]
    top_class: list[str] = field(default_factory=list)

    def score_between(self, group: str, start: float, end: float) -> float:
        s = self.group_scores[group]
        sel = (self.times + self.window_s > start) & (self.times < end)
        return float(s[sel].max()) if sel.any() else 0.0


@dataclass
class Event:
    type: str
    start: float
    end: float
    score: float
    classes: list[str]
    detector: str
    channel: int | None = None


class DetectorUnavailable(Exception):
    pass


def class_map_path() -> Path:
    return Path(str(resources.files("audio_sales_coach") / "data" / "yamnet_class_map.csv"))


class YamnetDetector:
    name = "yamnet"
    hop_s, window_s = 0.48, 0.96

    def __init__(self):
        if not models.is_installed("yamnet"):
            raise DetectorUnavailable("YAMNet weights not installed (run: audio-sales-coach models fetch)")
        from .yamnet_np import YamnetNumpy
        self.model = YamnetNumpy(models.path_for("yamnet"), class_map_path())
        names = self.model.class_names
        self.idx = {g: [names.index(c) for c in cs if c in names] for g, cs in GROUPS.items()}
        self.names = names
        self.model_file = models.REGISTRY["yamnet"].filename

    def track(self, x: np.ndarray, sr: int) -> EventTrack:
        assert sr == 16000
        t, sc = self.model.scores(x)
        groups = {g: sc[:, ix].max(axis=1) if ix else np.zeros(len(t)) for g, ix in self.idx.items()}
        top = [self.names[i] for i in sc.argmax(axis=1)]
        return EventTrack(self.name, self.model_file, self.hop_s, self.window_s, t, groups, top)


class CedDetector:
    """CED-mini through sherpa-onnx, run on 1 s windows with 0.5 s hop."""
    name = "ced"
    hop_s, window_s = 0.5, 1.0

    def __init__(self):
        if not models.is_installed("ced_mini"):
            raise DetectorUnavailable("CED-mini not installed (optional: audio-sales-coach models fetch --with ced_mini)")
        import sherpa_onnx
        d = models.path_for("ced_mini")
        cfg = sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(ced=str(d / "model.int8.onnx"), num_threads=2),
            labels=str(d / "class_labels_indices.csv"), top_k=527)
        self.tagger = sherpa_onnx.AudioTagging(cfg)
        self.model_file = d.name

    def track(self, x: np.ndarray, sr: int) -> EventTrack:
        hop, win = int(self.hop_s * sr), int(self.window_s * sr)
        starts = list(range(0, max(1, x.size - win + 1), hop))
        groups = {g: np.zeros(len(starts)) for g in GROUPS}
        top = []
        for i, s in enumerate(starts):
            st = self.tagger.create_stream()
            st.accept_waveform(sr, x[s:s + win])
            res = self.tagger.compute(st, top_k=527)
            by = {e.name: e.prob for e in res}
            top.append(res[0].name if res else "")
            for g, cs in GROUPS.items():
                groups[g][i] = max((by.get(c, 0.0) for c in cs), default=0.0)
        return EventTrack(self.name, self.model_file, self.hop_s, self.window_s,
                          np.asarray(starts) / sr, groups, top)


def make_detector(name: str = "auto"):
    order = {"auto": ["yamnet", "ced"], "yamnet": ["yamnet"], "ced": ["ced"]}[name]
    errors = []
    for n in order:
        try:
            return YamnetDetector() if n == "yamnet" else CedDetector()
        except DetectorUnavailable as e:
            errors.append(str(e))
    raise DetectorUnavailable("; ".join(errors))


def extract_events(track: EventTrack, thresholds: dict[str, float], channel: int | None = None,
                   groups=REPORTED) -> list[Event]:
    out = []
    for g in groups:
        thr = thresholds[g]
        s = track.group_scores[g]
        above = s >= thr
        i = 0
        while i < len(s):
            if not above[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(s) and above[j + 1]:
                j += 1
            seg = slice(i, j + 1)
            classes = sorted(set(track.top_class[i:j + 1])) if track.top_class else []
            out.append(Event(type=g, start=float(track.times[i]), end=float(track.times[j] + track.window_s),
                             score=round(float(s[seg].max()), 3), classes=classes, detector=track.detector,
                             channel=channel))
            i = j + 1
    return sorted(out, key=lambda e: e.start)
