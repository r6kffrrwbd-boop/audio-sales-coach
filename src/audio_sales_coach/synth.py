"""Synthetic test recordings with exact ground truth.

* `tone_speech()` makes speech-like signals from pure maths (harmonic source, syllable-rate envelope, formant
  filtering) with known pitch, syllable rate and pauses. These are bundled-safe (no third-party content).
* `render_scenario()` builds two-person calls and voice notes from text-to-speech (macOS `say`, generated
  locally at test time, never committed) and optionally inserts licensed ESC-50 laughter/breathing clips
  (CC BY-NC 3.0, downloaded to the local cache only; see docs/evaluation.md for attribution).
  Without macOS `say` (Linux) each line is a `stand_in_voice()`: the same maths as `tone_speech()`, so a whole
  call can still be rendered and analysed; it has no words, so transcript checks need real voices.

Ground truth (turns, overlaps, inserted events, questions) is written next to each rendering.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from . import audio_io
from .config import data_dir
from .models import http_download

SR = 16000


def tone_speech(f0: float, dur: float, syll_rate: float = 4.0, sr: int = SR, seed: int = 0, glide_st: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    f = f0 * 2 ** ((glide_st * t / max(dur, 1e-6)) / 12) * (1 + 0.03 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    src = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 25))
    env = 0.5 * (1 - np.cos(2 * np.pi * syll_rate * t))  # one peak per syllable
    env = env ** 1.5
    x = src * env
    for fc, bw in ((700, 130), (1220, 70), (2600, 160)):  # vowel-like formants
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.3
    return (x + rng.normal(0, 0.002, x.size)).astype(np.float32)


def have_tts() -> bool:
    return shutil.which("say") is not None


def tts(text: str, voice: str, rate: int = 180) -> np.ndarray:
    """Render text with macOS `say` and decode to 16 kHz mono."""
    if not have_tts():
        raise RuntimeError("macOS `say` is not available; TTS fixtures need macOS")
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "u.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True, timeout=120)
        x = audio_io.decode(aiff).mono
    # trim leading/trailing silence
    a = np.abs(x)
    idx = np.flatnonzero(a > 0.01)
    if idx.size:
        x = x[max(0, idx[0] - 160): idx[-1] + 160]
    return x


def stand_in_voice(text: str, voice: str, rate: int = 180) -> np.ndarray:
    """A text-free stand-in for one spoken line, for machines without macOS `say` (Linux).

    Harmonics with a syllable rhythm, as long as the line would take to read at `rate` words a minute, at a
    steady pitch that depends on the voice name. It carries no words, so nothing transcript-based can be
    checked with it; voice activity, turns, overlaps, gaps and pitch behave as they do for speech."""
    words = max(1, len(text.split()))
    f0 = 105.0 + (zlib.crc32(voice.encode()) % 12) * 10.0
    return tone_speech(f0, max(0.6, words * 60.0 / rate), syll_rate=rate / 45.0, seed=zlib.crc32(text.encode()) % 1000)


def speak_line(text: str, voice: str, rate: int = 180) -> np.ndarray:
    """One line of a scenario: macOS `say` when it is there, the stand-in voice otherwise."""
    return tts(text, voice, rate) if have_tts() else stand_in_voice(text, voice, rate)


def voice_source() -> str:
    return "macOS say" if have_tts() else "stand-in tones (no text-to-speech on this machine)"


# ---------------------------------------------------------------- ESC-50 (licensed, optional)
ESC50_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"


def esc50_dir() -> Path:
    return data_dir() / "eval-cache" / "esc50"


def esc50_meta() -> list[dict]:
    d = esc50_dir()
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "esc50.csv"
    if not meta.exists():
        http_download(f"{ESC50_BASE}/meta/esc50.csv", meta)
    with open(meta, newline="") as f:
        return list(csv.DictReader(f))


def esc50_clip(filename: str) -> Path:
    p = esc50_dir() / filename
    if not p.exists():
        tmp = p.with_suffix(".part")
        http_download(f"{ESC50_BASE}/audio/{filename}", tmp)
        tmp.rename(p)
    return p


def esc50_files(category: str, folds=("1", "2"), limit: int | None = None) -> list[dict]:
    rows = [r for r in esc50_meta() if r["category"] == category and r["fold"] in folds]
    return rows[:limit] if limit else rows


def load_event_clip(filename: str, max_s: float = 2.5) -> np.ndarray:
    x = audio_io.decode(esc50_clip(filename)).mono
    a = np.abs(x)
    # keep the loudest max_s window
    win = int(max_s * SR)
    if x.size > win:
        e = np.convolve(a, np.ones(SR // 10), mode="same")
        c = int(np.argmax(e))
        s = int(np.clip(c - win // 2, 0, x.size - win))
        x = x[s:s + win]
    fade = np.linspace(0, 1, 400)
    x[:400] *= fade
    x[-400:] *= fade[::-1]
    return x / (np.max(np.abs(x)) + 1e-9) * 0.25


# ---------------------------------------------------------------- scenarios
@dataclass
class Line:
    speaker: str  # "sales" | "client"
    text: str
    gap: float = 0.6  # silence before this line (negative = starts before previous ends: overlap)
    rate: int = 180
    cut_previous_after: float | None = None  # previous speaker stops this long after our onset (yields)
    event: str | None = None  # insert an ESC-50 event clip instead of speech: "laughing" | "breathing"
    question: bool = False
    topic: str | None = None


@dataclass
class Scenario:
    name: str
    voices: dict
    lines: list[Line]
    single_speaker: bool = False
    notes: list[str] = field(default_factory=list)


def demo_call() -> Scenario:
    return Scenario("demo_call", {"sales": "Daniel", "client": "Samantha"}, [
        Line("sales", "Hi, thanks for taking the time today. I wanted to walk you through our plan and see if it fits your team.", gap=0.5),
        Line("client", "Sure. We are looking at options for next quarter.", gap=0.7),
        Line("sales", "Great. Our standard package includes onboarding, weekly reporting and the mobile app, and most teams start on the annual plan.", gap=0.6),
        Line("client", "Okay. And how does the pricing work? Our budget is quite limited this year, so I need to understand the monthly cost, and whether we can start smaller and then",
             gap=0.8, topic="price"),
        Line("sales", "Right, so the annual plan is the best value, it is twelve hundred dollars a year and that includes everything, onboarding, reporting, support, all of it, and you lock in the price.",
             gap=-1.2, rate=255, cut_previous_after=0.6, topic="price"),
        Line("client", "What are the cancellation terms if it does not work out for us?", gap=3.4, question=True, topic="contract_terms"),
        Line("sales", "Good question. You can cancel with thirty days notice. And honestly, nobody has ever wanted to leave the reporting dashboard.", gap=0.9),
        Line("client", "", gap=0.4, event="laughing"),
        Line("sales", "So let me explain how the contract works in detail. The agreement runs for twelve months from the start date. "
                      "Billing is monthly or annual, whichever you prefer. Onboarding takes about two weeks, and we assign you a success manager. "
                      "After the first quarter we review usage together and adjust the setup. Reporting is delivered every Monday morning. "
                      "Support is available by chat and email during business hours, and urgent issues are handled the same day. "
                      "We also run a training session for new staff every month, and all of that is included in the plan. "
                      "If you add more users later, the price per seat stays the same, and you can move between plans at renewal. "
                      "Most customers also connect their calendar and email, which takes about ten minutes.", gap=0.8,
             topic="contract_terms"),
        Line("client", "", gap=0.5, event="breathing"),
        Line("client", "I will need to check with my manager before we decide anything.", gap=0.3, topic="decision_process"),
        Line("sales", "Of course. When would be a good time to follow up?", gap=0.7, question=True),
        Line("client", "Next week works.", gap=1.0),
    ])


def similar_voices_call() -> Scenario:
    s = demo_call()
    return Scenario("similar_voices_call", {"sales": "Daniel", "client": "Rocko (English (UK))"}, s.lines,
                    notes=["two male UK English TTS voices: tests speaker-mistake rate"])


def arabic_mixed_call() -> Scenario:
    return Scenario("arabic_mixed_call", {"sales": "Daniel", "client": "Majed"}, [
        Line("sales", "Hello, thanks for your time. Shall we go through the plan?", gap=0.5),
        Line("client", "أهلا، أكيد. بس بدي أعرف قديش الـ price تبع الـ annual plan؟", gap=0.7, question=True, topic="price"),
        Line("sales", "The annual plan is twelve hundred dollars, including onboarding and support.", gap=0.8, topic="price"),
        Line("client", "طيب، وإذا بدنا نعمل cancel بعد كم شهر، شو الشروط؟", gap=2.8, question=True, topic="contract_terms"),
        Line("sales", "You can cancel with thirty days notice at any time after the first three months.", gap=0.7),
        Line("client", "تمام، رح إحكي مع المدير وبرجعلك.", gap=0.9, topic="decision_process"),
    ], notes=["client lines are Arabic/English code-switched text read by an MSA Arabic TTS voice (not real Lebanese dialect)"])


def voice_note() -> Scenario:
    return Scenario("voice_note", {"sales": "Daniel"}, [
        Line("sales", "Hi, it is Daniel from the sales team. Just following up on our call about the reporting plan.", gap=0.3),
        Line("sales", "I have attached the pricing summary. The annual plan is twelve hundred dollars and it includes onboarding.", gap=2.6, topic="price"),
        Line("sales", "Let me know if Tuesday or Wednesday works for a short call.", gap=0.8, question=True),
    ], single_speaker=True)


def short_voice_note() -> Scenario:
    return Scenario("short_voice_note", {"sales": "Samantha"}, [
        Line("sales", "Hi, quick one: does Thursday at ten still work for you?", gap=0.2, question=True)], single_speaker=True)


SCENARIOS = {f.__name__: f for f in (demo_call, similar_voices_call, arabic_mixed_call, voice_note, short_voice_note)}


def render_scenario(sc: Scenario, out_dir: Path, use_esc50: bool = True, seed: int = 7) -> dict:
    """Render to out_dir: stereo.wav (sales left, client right), mono.wav, truth.json. Returns truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chans = {"sales": [], "client": []}
    t = 0.5
    truth = {"scenario": sc.name, "voices": sc.voices, "single_speaker": sc.single_speaker, "turns": [], "events": [],
             "interruptions": [], "response_gaps": [], "notes": list(sc.notes), "sr": SR,
             "voice_source": voice_source()}
    prev_end = {"sales": 0.0, "client": 0.0}
    last = None
    esc_used = []
    for i, ln in enumerate(sc.lines):
        start = (last["end"] if last else 0.5) + ln.gap
        start = max(0.0, start)
        if ln.event:
            if not use_esc50:
                truth["notes"].append(f"skipped {ln.event} event (ESC-50 not used)")
                continue
            rows = esc50_files(ln.event, folds=("3", "4", "5"))  # never the folds used to set thresholds
            row = rows[int(rng.integers(len(rows)))]
            x = load_event_clip(row["filename"], max_s=2.0 if ln.event == "laughing" else 1.5)
            esc_used.append({"file": row["filename"], "category": row["category"], "fold": row["fold"]})
            chans[ln.speaker].append((start, x))
            truth["events"].append({"type": {"laughing": "laughter", "breathing": "breathing"}[ln.event],
                                    "speaker": ln.speaker, "start": round(start, 2), "end": round(start + x.size / SR, 2),
                                    "source": f"ESC-50 {row['filename']}"})
            last = {"speaker": ln.speaker, "end": start + x.size / SR}
            continue
        voice = sc.voices[ln.speaker]
        x = speak_line(ln.text, voice, ln.rate)
        if ln.cut_previous_after is not None and last is not None:
            # previous speaker yields: cut their audio cut_previous_after seconds after our onset
            ps = last["speaker"]
            st0, px = chans[ps][-1]
            keep = int((start + ln.cut_previous_after - st0) * SR)
            if 0 < keep < px.size:
                fade = np.linspace(1, 0, min(1600, keep))
                px = px[:keep].copy()
                px[-fade.size:] *= fade
                chans[ps][-1] = (st0, px)
                truth["turns"][-1]["end"] = round(st0 + keep / SR, 2)
                truth["turns"][-1]["text_truncated"] = True
            truth["interruptions"].append({"by": ln.speaker, "of": ps, "at": round(start, 2),
                                           "overlap_s": round(st0 + keep / SR - start, 2) if keep > 0 else 0.0})
        elif last is not None and ln.speaker != last["speaker"] and ln.gap >= 2.0:
            truth["response_gaps"].append({"after": last["speaker"], "by": ln.speaker, "start": round(last["end"], 2),
                                           "gap_s": round(start - last["end"], 2)})
        chans[ln.speaker].append((start, x))
        end = start + x.size / SR
        truth["turns"].append({"speaker": ln.speaker, "start": round(start, 2), "end": round(end, 2), "text": ln.text,
                               "rate": ln.rate, "question": ln.question, "topic": ln.topic})
        prev_end[ln.speaker] = end
        last = {"speaker": ln.speaker, "end": truth["turns"][-1]["end"]}
    total = max([s + x.size / SR for c in chans.values() for s, x in c] + [1.0]) + 0.8
    n = int(total * SR)
    out = np.zeros((2, n), dtype=np.float32)
    for ci, sp in enumerate(("sales", "client")):
        for s, x in chans[sp]:
            a = int(s * SR)
            out[ci, a:a + x.size] += x[: n - a]
    out += rng.normal(0, 0.0015, out.shape).astype(np.float32)  # faint line noise
    truth["duration_s"] = round(total, 2)
    truth["esc50_clips"] = esc_used
    if sc.single_speaker:
        audio_io.write_wav(out_dir / "mono.wav", out[0])
    else:
        audio_io.write_wav(out_dir / "stereo.wav", out)
        audio_io.write_wav(out_dir / "mono.wav", out.sum(0) * 0.7)
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    return truth


def add_noise(src: Path, dst: Path, snr_db: float, seed: int = 3) -> Path:
    d = audio_io.decode(src)
    x = d.mono
    rng = np.random.default_rng(seed)
    # pink-ish noise
    w = rng.normal(0, 1, x.size)
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    noise = lfilter(b, a, w)
    act = np.abs(x) > 0.01
    p_sig = np.mean(x[act] ** 2) if act.any() else np.mean(x ** 2)
    noise *= np.sqrt(p_sig / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    audio_io.write_wav(dst, np.clip(x + noise, -1, 1))
    return dst
