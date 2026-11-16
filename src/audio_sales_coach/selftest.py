"""Offline self-test: models, decoding of every supported format, detector inference, a full two-person analysis
and a voice-message analysis on synthetic audio. Nothing leaves the machine (the council, if installed, is asked
with providers ["none"])."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import audio_io, coaching, events, models, synth
from .config import load_config


def _ok(name, cond, detail=""):
    return {"check": name, "ok": bool(cond), "detail": detail}


def _short_call() -> synth.Scenario:
    L = synth.Line
    return synth.Scenario("selftest_call", {"sales": "Daniel", "client": "Samantha"}, [
        L("sales", "Hello, thanks for your time. I would like to understand how your team handles reporting today.", gap=0.4),
        L("client", "We mostly use spreadsheets, and the budget for new tools is quite tight, so I need to know the cost and", gap=0.6),
        L("sales", "Right, the annual plan is twelve hundred dollars and it includes onboarding and support for everyone.", gap=-1.0,
          rate=250, cut_previous_after=0.9),
        L("client", "What happens if we want to cancel?", gap=3.0, question=True),
        L("sales", "You can cancel with thirty days notice.", gap=0.7),
    ])


def run(verbose: bool = True) -> int:
    say = print if verbose else (lambda *a, **k: None)
    res = []
    t_all = time.time()
    res.append(_ok("ffmpeg/ffprobe on PATH", shutil.which("ffmpeg") and shutil.which("ffprobe")))
    for k, m in models.REGISTRY.items():
        if m.default:
            ok, why = models.verify(k)
            res.append(_ok(f"model {k} verified (size + SHA-256)", ok, why))
    cfg = load_config()
    cfg["council"]["providers"] = ["none"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. pure-maths signal: formats round trip
        x = np.concatenate([synth.tone_speech(130, 2.0, seed=1), np.zeros(8000, np.float32), synth.tone_speech(210, 2.0, seed=2)])
        audio_io.write_wav(td / "t.wav", x)
        for codec, ext, br in (("opus", "ogg", "16k"), ("mp3", "mp3", "64k"), ("aac", "m4a", "64k")):
            try:
                p = audio_io.encode(td / "t.wav", td / f"t.{ext}", codec, br)
                d = audio_io.decode(p)
                res.append(_ok(f"decode {ext} ({codec})", abs(d.duration - x.size / 16000) < 0.15, f"{d.duration:.2f} s"))
            except audio_io.AudioError as e:
                res.append(_ok(f"decode {ext} ({codec})", False, str(e)))
        # 2. detector inference on the original waveform
        try:
            det = events.make_detector(cfg.get("detector", "auto"))
            tr = det.track(x, 16000)
            res.append(_ok(f"{det.name} inference", tr.times.size > 0 and np.isfinite(tr.group_scores["speech"]).all(),
                           f"{tr.times.size} windows"))
        except events.DetectorUnavailable as e:
            res.append(_ok("event detector", False, str(e)))
        # 3. full analysis of a synthetic two-person call and a voice note
        if synth.have_tts():
            from .pipeline import analyze
            truth = synth.render_scenario(_short_call(), td / "call", use_esc50=False)
            rep = analyze(str(td / "call" / "stereo.wav"), str(td / "rep-call"), cfg, mode="call", transcription="off",
                          log=lambda *a: None)
            types = {o["type"] for o in rep["observations"]}
            res.append(_ok("two-party call: separate channels found", rep["layout"]["result"] == "separate_channels"))
            res.append(_ok("two-party call: interruption observed", "interruption" in types, ", ".join(sorted(types))))
            res.append(_ok("two-party call: response gap observed", "response_gap" in types))
            notes = rep["coaching_notes"]
            need = ("timestamp", "play", "speaker", "cue", "interaction", "suggestion", "words_status")
            res.append(_ok("notes have every required field", notes and all(all(k in n for k in need) for n in notes),
                           f"{len(notes)} notes"))
            res.append(_ok("notes reference the ORIGINAL recording",
                           all(n["play"]["uri"].startswith("file://") and "stereo.wav#t=" in n["play"]["uri"] for n in notes)))
            bad = [coaching.guard_text(n["interaction"] + n["suggestion"] + n["cue"]["description"]) for n in notes]
            res.append(_ok("no emotion/manipulation wording", not any(bad), str(bad)))
            res.append(_ok("council review ran or was skipped cleanly",
                           rep["council"]["status"] in ("reviewed", "skipped"),
                           f"{rep['council']['status']}: {rep['council'].get('reason') or rep['council'].get('verdict_status')}"))
            vn = synth.render_scenario(synth.short_voice_note(), td / "vn", use_esc50=False)
            audio_io.encode(td / "vn" / "mono.wav", td / "vn" / "note.ogg", "opus", "16k")
            rep2 = analyze(str(td / "vn" / "note.ogg"), str(td / "rep-vn"), cfg, mode="auto", transcription="off",
                           council_mode="off", log=lambda *a: None)
            res.append(_ok("voice note (Opus): one speaker, no invented second person",
                           len(rep2["speakers"]) == 1 and "client" not in rep2["views"], rep2["layout"]["result"]))
            _ = truth, vn
        else:
            res.append(_ok("TTS-based analysis (macOS `say`)", True, "skipped: not on macOS"))
    failed = [r for r in res if not r["ok"]]
    for r in res:
        say(f"  [{'ok' if r['ok'] else 'FAIL'}] {r['check']}" + (f" - {r['detail']}" if r["detail"] else ""))
    say(json.dumps({"selftest": "pass" if not failed else "fail", "checks": len(res), "failed": len(failed),
                    "seconds": round(time.time() - t_all, 1)}))
    return 0 if not failed else 1
