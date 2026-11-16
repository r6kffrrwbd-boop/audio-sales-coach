"""Evaluation harness.

1. Detector check on licensed clips: ESC-50 (CC BY-NC 3.0) folds 1 (threshold choice) and 2 (held-out test),
   target classes laughing / breathing against confusable sound classes, plus TTS speech as negatives.
2. Scenario check on synthetic calls with exact ground truth: speaker attribution (separate channels vs mixed mono,
   different vs similar voices), unknown/uncertain rate, interruption and response-gap recall, laughter/breathing
   recall and false detections, under noise, Opus 16 kbit/s compression and short voice notes.
3. Transcript-only baseline: what an (oracle, perfect) transcript with speaker labels and timestamps could flag,
   versus the audio pipeline, on the same ground-truth coachable moments.

All numbers come from synthetic or licensed audio. Real-world performance on authorized sales calls is unverified.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import audio_io, events, synth

TARGETS = {"laughter": "laughing", "breathing": "breathing"}
NEGATIVES = ["coughing", "sneezing", "snoring", "clapping", "crying_baby", "keyboard_typing", "drinking_sipping",
             "door_wood_knock"]


def _clip_scores(det, rows) -> list[dict]:
    out = []
    for r in rows:
        x = audio_io.decode(synth.esc50_clip(r["filename"])).mono
        tr = det.track(x, 16000)
        out.append({"file": r["filename"], "category": r["category"], "fold": r["fold"],
                    **{g: float(tr.group_scores[g].max()) for g in ("laughter", "breathing", "speech")}})
    return out


def _prf(scores, labels, thr):
    pred = scores >= thr
    tp, fp, fn = int((pred & labels).sum()), int((pred & ~labels).sum()), int((~pred & labels).sum())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"threshold": round(float(thr), 3), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 3) if p is not None else None, "recall": round(r, 3) if r is not None else None}


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not pos.size or not neg.size:
        return None
    return round(float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())), 3)


def detector_eval(det, speech_negatives: list[np.ndarray] | None = None) -> dict:
    cats = list(TARGETS.values()) + NEGATIVES
    rows = [r for c in cats for r in synth.esc50_files(c, folds=("1", "2"))]
    sc = _clip_scores(det, rows)
    speech_sc = []
    for x in speech_negatives or []:
        tr = det.track(x, 16000)
        speech_sc.append({g: float(tr.group_scores[g].max()) for g in ("laughter", "breathing")})
    res = {"detector": det.name, "clips": len(sc), "speech_negatives": len(speech_sc), "classes": {}}
    for g, cat in TARGETS.items():
        f1 = [s for s in sc if s["fold"] == "1"]
        f2 = [s for s in sc if s["fold"] == "2"]
        y1 = np.array([s["category"] == cat for s in f1])
        s1 = np.array([s[g] for s in f1])
        # threshold: maximise F1 on fold 1 (precision weighted equally), from a fixed grid
        grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
        best = max(grid, key=lambda t: _f1(_prf(s1, y1, t)))
        y2 = np.array([s["category"] == cat for s in f2])
        s2 = np.array([s[g] for s in f2])
        test = _prf(s2, y2, best)
        sp_fp = sum(1 for s in speech_sc if s[g] >= best)
        res["classes"][g] = {
            "chosen_threshold_fold1": best, "fold1": _prf(s1, y1, best), "fold2_heldout": test,
            "auc_fold2": _auc(s2[y2], s2[~y2]),
            "false_alarms_on_tts_speech": f"{sp_fp}/{len(speech_sc)}",
            "confused_with": sorted({s["category"] for s in f2 if s["category"] != cat and s[g] >= best}),
        }
    res["per_clip"] = sc
    return res


def _f1(m):
    p, r = m["precision"] or 0, m["recall"] or 0
    return 2 * p * r / (p + r) if p + r else 0


# ---------------------------------------------------------------- scenario evaluation
def _frame_truth(truth, n, hop=0.01):
    lab = np.full(n, -1)
    over = np.zeros(n, bool)
    for t in truth["turns"]:
        a, b = int(t["start"] / hop), int(t["end"] / hop)
        v = 0 if t["speaker"] == "sales" else 1
        over[a:b] |= (lab[a:b] >= 0) & (lab[a:b] != v)
        lab[a:b] = np.where(lab[a:b] < 0, v, lab[a:b])
    return lab, over


def speaker_metrics(report, truth, sales_id_hint=None) -> dict:
    hop = 0.01
    n = int(truth["duration_s"] / hop) + 1
    lab, over = _frame_truth(truth, n)
    pred = np.full(n, -1)
    unc = np.zeros(n, bool)
    for s in report["segments"]:
        a, b = int(s["start"] / hop), int(s["end"] / hop)
        pred[a:b] = {"S1": 1, "S2": 2}.get(s["speaker"], 0)
        unc[a:b] |= s["uncertain"]
    speech = (lab >= 0) & ~over
    if not speech.any():
        return {}
    # best mapping of S1/S2 to sales/client (diarization labels are arbitrary)
    best = None
    for m in ({1: 0, 2: 1}, {1: 1, 2: 0}):
        mapped = np.vectorize(lambda v: m.get(v, -2))(pred)
        acc = float(np.mean(mapped[speech] == lab[speech]))
        if best is None or acc > best[0]:
            best = (acc, m)
    mapped = np.vectorize(lambda v: best[1].get(v, -2))(pred)
    detected = speech & (pred > 0)
    confident = detected & ~unc
    return {"speech_frames_s": round(speech.sum() * hop, 1),
            "attribution_accuracy": round(best[0], 3),
            "accuracy_on_confident_frames": round(float(np.mean(mapped[confident] == lab[confident])), 3) if confident.any() else None,
            "missed_speech_share": round(float(np.mean(pred[speech] <= 0)), 3),
            "unknown_or_uncertain_share": round(float(np.mean((pred[speech] == 0) | unc[speech])), 3),
            "speakers_found": len(report["speakers"]),
            "role_map": {f"S{k}": ("sales" if v == 0 else "client") for k, v in best[1].items()}}


def event_metrics(report, truth, tol=1.0) -> dict:
    out = {}
    for typ in ("laughter", "breathing"):
        gt = [e for e in truth["events"] if e["type"] == typ]
        det = [e for e in report["events"] if e["type"] == typ]
        hit = [g for g in gt if any(d["start"] - tol <= g["end"] and d["end"] + tol >= g["start"] for d in det)]
        fa = [d for d in det if not any(d["start"] - tol <= g["end"] and d["end"] + tol >= g["start"] for g in gt)]
        out[typ] = {"truth": len(gt), "detected": len(hit), "false_detections": len(fa),
                    "false_detection_times": [d["start"] for d in fa]}
    return out


def moment_metrics(report, truth, tol=1.5) -> dict:
    obs = report["observations"]

    def near(types, t):
        return any(o["type"] in types and abs(o["start"] - t) <= tol for o in obs)
    inter = truth["interruptions"]
    gaps = truth["response_gaps"]
    qs = [t for t in truth["turns"] if t["question"] and t["speaker"] == "client"]
    return {
        "interruptions": {"truth": len(inter), "found": sum(near(("interruption", "possible_interruption"), i["at"]) for i in inter)},
        "response_gaps": {"truth": len(gaps), "found": sum(near(("response_gap",), g["start"]) for g in gaps)},
        "client_questions": {"truth": len(qs), "found": sum(near(("question", "rising_ending"), q["start"]) or
                                                            near(("question", "rising_ending"), q["end"] - 1) for q in qs)},
        "false_interruptions": sum(1 for o in obs if o["type"] in ("interruption", "possible_interruption")
                                   and not any(abs(o["start"] - i["at"]) <= tol for i in inter)),
    }


def transcript_only_baseline(truth) -> dict:
    """Oracle transcript (perfect words, speaker labels and start/end times) - what text alone could flag."""
    turns = truth["turns"]
    found = {"client_questions": sum(1 for t in turns if t["question"] and t["speaker"] == "client"),
             # a transcript has no overlap information unless timestamps overlap; oracle times do, typical ASR does not
             "interruptions_from_overlapping_timestamps": sum(
                 1 for a, b in zip(turns, turns[1:]) if a["speaker"] != b["speaker"] and b["start"] < a["end"]),
             "response_gaps_from_timestamps": sum(1 for a, b in zip(turns, turns[1:])
                                                  if a["speaker"] != b["speaker"] and b["start"] - a["end"] >= 2.5),
             "laughter": 0, "breathing": 0, "pace_changes": 0, "pitch_or_level_changes": 0}
    return {"assumption": "oracle transcript with perfect words, speaker labels and timestamps (upper bound for "
                          "transcript-only coaching); laughter, breathing, pace, pitch and level are not in text",
            "found": found,
            "truth": {"client_questions": sum(1 for t in turns if t["question"] and t["speaker"] == "client"),
                      "interruptions": len(truth["interruptions"]), "response_gaps": len(truth["response_gaps"]),
                      "laughter": sum(1 for e in truth["events"] if e["type"] == "laughter"),
                      "breathing": sum(1 for e in truth["events"] if e["type"] == "breathing"),
                      "pace_changes": sum(1 for t in turns if t.get("rate", 180) >= 240)}}


def run_scenarios(work: Path, cfg: dict, analyze, use_esc50: bool = True, log=print) -> dict:
    """Render each scenario, build variants, analyse, and score against ground truth."""
    results = {}
    for name in ("demo_call", "similar_voices_call", "arabic_mixed_call", "voice_note", "short_voice_note"):
        sc = synth.SCENARIOS[name]()
        d = work / name
        t0 = time.time()
        truth = synth.render_scenario(sc, d, use_esc50=use_esc50)
        variants = {}
        if not truth["single_speaker"]:
            variants["stereo_separate_channels"] = (d / "stereo.wav", "call")
        variants["mono_mix"] = (d / "mono.wav", "call" if not truth["single_speaker"] else "voice-message")
        if name == "demo_call":
            variants["mono_noisy_snr10"] = (synth.add_noise(d / "mono.wav", d / "noisy10.wav", 10), "call")
            variants["mono_noisy_snr5"] = (synth.add_noise(d / "mono.wav", d / "noisy5.wav", 5), "call")
            variants["opus_16k_whatsapp_like"] = (audio_io.encode(d / "mono.wav", d / "mono.ogg", "opus", "16k"), "call")
        if name == "voice_note":
            variants["opus_16k_whatsapp_like"] = (audio_io.encode(d / "mono.wav", d / "note.ogg", "opus", "16k"), "voice-message")
            variants["auto_mode_mono"] = (d / "mono.wav", "auto")
        res = {"truth_summary": {k: len(truth[k]) for k in ("turns", "events", "interruptions", "response_gaps")},
               "duration_s": truth["duration_s"], "notes": truth["notes"], "variants": {}}
        for vname, (path, mode) in variants.items():
            rep = analyze(str(path), str(d / f"report-{vname}"), cfg, mode=mode, transcription="off",
                          council_mode="off", log=lambda *a: None)
            res["variants"][vname] = {
                "layout": rep["layout"]["result"], "speakers_found": len(rep["speakers"]),
                "speaker": speaker_metrics(rep, truth) if not truth["single_speaker"] else
                {"speakers_found": len(rep["speakers"]), "invented_second_speaker": len(rep["speakers"]) > 1},
                "events": event_metrics(rep, truth),
                "moments": moment_metrics(rep, truth),
                "notes": len(rep["coaching_notes"]), "seconds": rep["timings_s"].get("total"),
            }
            log(f"  {name}/{vname}: done in {rep['timings_s'].get('total')} s")
        res["transcript_only_baseline"] = transcript_only_baseline(truth)
        res["render_s"] = round(time.time() - t0, 1)
        results[name] = res
    return results


def write(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def insertion_eval(det, thresholds: dict, work: Path, folds=("3", "4", "5"), opus: bool = True) -> dict:
    """Insert every ESC-50 laughing / breathing clip from folds not used for threshold choice between two TTS speech
    turns (speech | 0.3 s | clip | 0.3 s | speech), run the detector, and count hits inside the clip window and
    false events elsewhere. Repeated after 16 kbit/s Opus (WhatsApp-like) compression."""
    work.mkdir(parents=True, exist_ok=True)
    a = synth.tts("Thanks, so as I was saying, the plan includes onboarding and weekly reports.", "Daniel")
    b = synth.tts("Okay, and what about the monthly cost for a small team?", "Samantha")
    gap = np.zeros(int(0.3 * 16000), np.float32)
    out = {}
    for cat, typ in (("laughing", "laughter"), ("breathing", "breathing")):
        rows = synth.esc50_files(cat, folds=folds)
        res = {"clips": len(rows), "plain": {"hit": 0, "false_events": 0}, "opus16k": {"hit": 0, "false_events": 0}}
        for r in rows:
            clip = synth.load_event_clip(r["filename"], max_s=2.0)
            x = np.concatenate([a, gap, clip, gap, b])
            s0 = (a.size + gap.size) / 16000
            s1 = s0 + clip.size / 16000
            variants = {"plain": x}
            if opus:
                audio_io.write_wav(work / "x.wav", x)
                variants["opus16k"] = audio_io.decode(audio_io.encode(work / "x.wav", work / "x.ogg", "opus", "16k")).mono
            for vn, xx in variants.items():
                evs = [e for e in events.extract_events(det.track(xx, 16000), thresholds) if e.type in ("laughter", "breathing")]
                hit = any(e.type == typ and e.start <= s1 + 0.25 and e.end >= s0 - 0.25 for e in evs)
                res[vn]["hit"] += int(hit)
                res[vn]["false_events"] += sum(1 for e in evs if not (e.start <= s1 + 0.25 and e.end >= s0 - 0.25))
        for vn in ("plain", "opus16k"):
            res[vn]["recall"] = round(res[vn]["hit"] / max(1, len(rows)), 3)
        out[typ] = res
    out["note"] = ("false_events = laughter/breathing events outside the inserted clip (i.e. on the surrounding TTS "
                   "speech); a wrong-type event inside the clip window is not counted as a hit")
    return out
