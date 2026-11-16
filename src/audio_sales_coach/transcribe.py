"""Optional local transcription with faster-whisper (CTranslate2, CPU int8).

Installed only with `./install.sh --with-transcription`. Models download once from the public
Hugging Face repositories (no account or login) into the model cache. Nothing is uploaded.

Mixed Lebanese Arabic / English: Whisper detects one language per 30 s window and is known to
translate or drop code-switched words, misspell Arabizi and confuse dialect with MSA. Each segment
carries Whisper's own confidence signals and is marked unreliable below the configured thresholds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .config import models_dir


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float


@dataclass
class TSegment:
    start: float
    end: float
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float
    reliable: bool
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    model: str
    language: str
    language_prob: float
    segments: list[TSegment]
    notes: list[str]

    def words_between(self, start: float, end: float) -> str:
        ws = [w.text for s in self.segments for w in s.words if w.end > start and w.start < end]
        if ws:
            return "".join(ws).strip()
        return " ".join(s.text.strip() for s in self.segments if s.end > start and s.start < end).strip()

    def reliable_between(self, start: float, end: float) -> bool | None:
        segs = [s for s in self.segments if s.end > start and s.start < end]
        return all(s.reliable for s in segs) if segs else None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe(x: np.ndarray, sr: int, cfg: dict) -> Transcript:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain resumable HTTPS; the Xet client stalled on a slow link
    from faster_whisper import WhisperModel

    from .models import whisper_dir
    size = cfg.get("model", "small")
    local = whisper_dir(size)
    source = str(local) if (local / "model.bin").exists() else size  # prefer the curl-fetched local copy
    cache = models_dir() / "whisper"
    model = WhisperModel(source, device="cpu", compute_type=cfg.get("compute_type", "int8"),
                         download_root=str(cache), cpu_threads=max(1, (os.cpu_count() or 4) // 2))
    lang = cfg.get("language")
    segs, info = model.transcribe(x.astype(np.float32), language=lang, word_timestamps=True, vad_filter=True,
                                  beam_size=5, condition_on_previous_text=False, multilingual=lang is None)
    out = []
    min_lp, max_ns = cfg.get("min_avg_logprob", -1.0), cfg.get("max_no_speech_prob", 0.6)
    for s in segs:
        words = [Word(w.start, w.end, w.word, round(w.probability, 3)) for w in (s.words or [])]
        out.append(TSegment(s.start, s.end, s.text.strip(), getattr(s, "language", None) or info.language,
                            round(s.avg_logprob, 3), round(s.no_speech_prob, 3),
                            s.avg_logprob >= min_lp and s.no_speech_prob <= max_ns, words))
    notes = []
    if info.language in ("ar", "en") and info.language_probability < 0.8:
        notes.append(f"language detection uncertain ({info.language} p={info.language_probability:.2f}); "
                     "mixed Arabic/English speech is likely mis-transcribed in places")
    if info.language not in ("ar", "en"):
        notes.append(f"detected language '{info.language}' is outside the tested set (Arabic, English)")
    bad = sum(1 for s in out if not s.reliable)
    if out and bad / len(out) > 0.3:
        notes.append(f"{bad} of {len(out)} transcript segments are below the confidence thresholds")
    return Transcript(f"faster-whisper {size} ({cfg.get('compute_type', 'int8')}, CPU)", info.language,
                      round(info.language_probability, 3), out, notes)
