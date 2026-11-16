"""Command-line interface: `audio-sales-coach <command> ...`"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

EPILOG = """examples:
  audio-sales-coach analyze ~/calls/call.m4a --output ~/coach-reports/call-1 --consent all-parties-consented
  audio-sales-coach analyze note.ogg --output ~/coach-reports/note-1 --mode voice-message
  audio-sales-coach play ~/coach-reports/call-1 --note note-2
  audio-sales-coach roles ~/coach-reports/call-1 --salesperson S1 --remember my-alias --consent "I agree ..."
  audio-sales-coach feedback ~/coach-reports/call-1 --item note-2 --mark incorrect --reason "that was agreement"
  audio-sales-coach voice enroll my-alias --sample my-voice.m4a --consent "I agree to store my voice profile"
  audio-sales-coach models status | audio-sales-coach selftest
exit codes: 0 ok, 1 error, 2 usage/input error, 3 missing models or ffmpeg
"""


def _cfg(args):
    from .config import load_config
    cfg = load_config(Path(args.config) if getattr(args, "config", None) else None)
    return cfg


def cmd_analyze(args) -> int:
    from . import pipeline
    from .audio_io import AudioError
    cfg = _cfg(args)
    if args.council_providers:
        cfg["council"]["providers"] = [p.strip() for p in args.council_providers.split(",") if p.strip()]
    user_roles = None
    if args.salesperson:
        user_roles = {s: ("salesperson" if s == args.salesperson else "client_in_this_call") for s in ("S1", "S2")}
    log = (lambda *a: None) if args.quiet else (lambda m: print(f"[audio-sales-coach] {m}", file=sys.stderr))
    try:
        rep = pipeline.analyze(args.recording, args.output, cfg, mode=args.mode, transcription=args.transcribe,
                               council_mode=args.council, user_roles=user_roles, consent=args.consent, log=log)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(args.output).expanduser().resolve()
    summary = {"report": str(out / "report.json"), "markdown": str(out / "report.md"), "html": str(out / "report.html"),
               "layout": rep["layout"]["result"], "speakers": [{k: s[k] for k in ("id", "role", "talk_share")} for s in rep["speakers"]],
               "notes": len(rep["coaching_notes"]), "observations": len(rep["observations"]),
               "events": len(rep["events"]), "council": rep["council"].get("status"),
               "roles": rep["role_assignment"]["status"], "warnings": rep["warnings"]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_play(args) -> int:
    from .pipeline import load_report
    rep = load_report(args.report)
    if args.note:
        n = next((n for n in rep["coaching_notes"] if n["id"] == args.note), None)
        if not n:
            print(f"error: {args.note} not in report", file=sys.stderr)
            return 2
        a, b = n["play"]["from_s"], n["play"]["to_s"]
    elif args.at is not None:
        a, b = args.at, args.at + (args.duration or 8.0)
    else:
        print("error: give --note ID or --at SECONDS", file=sys.stderr)
        return 2
    path = rep["recording"]["path"]
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", path]
    if args.print_only or not shutil.which("ffplay"):
        print(" ".join(cmd))
        return 0
    print(f"playing {Path(path).name} {a:.1f}-{b:.1f} s (original recording)", file=sys.stderr)
    return subprocess.call(cmd)


def cmd_roles(args) -> int:
    from . import feedback, pipeline, speakers, voiceid
    from .audio_io import decode, sha256_file
    cfg = _cfg(args)
    rep = pipeline.load_report(args.report)
    ids = [s["id"] for s in rep["speakers"]]
    if args.salesperson not in ids:
        print(f"error: --salesperson must be one of {ids}", file=sys.stderr)
        return 2
    roles = {s: ("salesperson" if s == args.salesperson else "client_in_this_call") for s in ids}
    feedback.add(rep, None, None, args.reason, speaker_correction={"roles": roles})
    rep = pipeline.rebuild_with_roles(args.report, roles, cfg, council_mode=args.council)
    msg = {"roles": roles, "notes": len(rep["coaching_notes"]), "council": rep["council"].get("status")}
    if args.remember:
        if not args.consent:
            print("error: --remember needs --consent \"<the salesperson's own consent statement>\"", file=sys.stderr)
            return 2
        path = rep["recording"]["path"]
        if not Path(path).exists() or sha256_file(Path(path)) != rep["recording"]["sha256"]:
            print("error: original recording missing or changed; cannot compute the voice profile", file=sys.stderr)
            return 2
        dec = decode(path)
        emb = speakers.Embedder()
        seg = [speakers.Segment(s["start"], s["end"], s["speaker"], s["confidence"], overlap=s["overlap"],
                                uncertain=s["uncertain"]) for s in rep["segments"]]
        x = dec.channels[[s for s in rep["speakers"] if s["id"] == args.salesperson][0].get("channel", 1) - 1] \
            if rep["layout"]["result"] == "separate_channels" else dec.mono
        v = speakers.speaker_embedding_for(x, dec.sr, seg, args.salesperson, emb)
        if v is None:
            print("error: not enough clear speech from that speaker to build a profile", file=sys.stderr)
            return 1
        voiceid.enroll(args.remember, v, emb.name, args.consent, source="role correction in call",
                       report_id=rep["report_id"])
        msg["profile_updated"] = args.remember
    print(json.dumps(msg, indent=2))
    return 0


def cmd_feedback(args) -> int:
    from . import feedback, pipeline
    if args.action == "summary":
        print(json.dumps(feedback.summary(), indent=2))
        return 0
    if not args.report:
        print("error: give a report directory", file=sys.stderr)
        return 2
    rep = pipeline.load_report(args.report)
    if args.action == "list":
        print(json.dumps(feedback.entries(rep["report_id"]), indent=2, ensure_ascii=False))
        return 0
    corr = None
    if args.speaker:
        k, _, v = args.speaker.partition("=")
        if v not in ("salesperson", "client", "client_in_this_call", "unknown", "other"):
            print("error: --speaker S1=salesperson|client|unknown|other", file=sys.stderr)
            return 2
        corr = {k: "client_in_this_call" if v == "client" else v}
    try:
        e = feedback.add(rep, args.item, args.mark, args.reason, corr)
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    print(json.dumps({"saved": e, "note": "stored locally for evaluation; no model is retrained"}, indent=2, ensure_ascii=False))
    return 0


def cmd_voice(args) -> int:
    from . import speakers, voiceid
    from .audio_io import decode
    cfg = _cfg(args)
    try:
        if args.action == "list":
            print(json.dumps(voiceid.list_profiles(), indent=2))
        elif args.action == "delete":
            ok = voiceid.delete(args.name)
            print(json.dumps({"deleted": ok, "name": args.name}))
            return 0 if ok else 1
        elif args.action in ("enroll", "re-enroll"):
            if not args.sample:
                print("error: --sample FILE is required", file=sys.stderr)
                return 2
            dec = decode(args.sample)
            vad = speakers.Vad()
            segs = [speakers.Segment(s, e, "S1", 1.0) for s, e in vad.segments(dec.mono, dec.sr)]
            speech = sum(s.duration for s in segs)
            if speech < 8:
                print(f"error: only {speech:.1f} s of speech; enrol with at least 8 s (30-60 s recommended)", file=sys.stderr)
                return 2
            emb = speakers.Embedder()
            v = speakers.speaker_embedding_for(dec.mono, dec.sr, segs, "S1", emb, max_s=120)
            d = voiceid.enroll(args.name, v, emb.name, args.consent or "", source="enrolment sample",
                               replace=args.action == "re-enroll" or args.replace)
            print(json.dumps({"enrolled": args.name, "samples": len(d["embeddings"]), "speech_s": round(speech, 1),
                              "stored_in": str(voiceid.profile_path(args.name).parent)}, indent=2))
        elif args.action == "calibrate":
            print(json.dumps(voiceid.calibrate(args.name, cfg["voice_id"]), indent=2))
        elif args.action == "identify":
            dec = decode(args.sample)
            vad = speakers.Vad()
            segs = [speakers.Segment(s, e, "S1", 1.0) for s, e in vad.segments(dec.mono, dec.sr)]
            emb = speakers.Embedder()
            v = speakers.speaker_embedding_for(dec.mono, dec.sr, segs, "S1", emb)
            m, summ = voiceid.identify({"S1": v}, cfg["voice_id"])
            print(json.dumps(summ, indent=2))
    except voiceid.ProfileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_models(args) -> int:
    from . import models
    if args.action == "status":
        print(json.dumps(models.status(), indent=2))
        return 0
    if args.whisper:
        try:
            print(f"ok whisper-{args.whisper}: {models.fetch_whisper(args.whisper, progress=lambda m: print(m, file=sys.stderr))}",
                  file=sys.stderr)
        except Exception as e:
            print(f"error fetching whisper-{args.whisper}: {e}", file=sys.stderr)
            return 3
        if args.only_whisper:
            return 0
    keys = [k for k, m in models.REGISTRY.items() if m.default] + (args.with_ or [])
    for k in keys:
        if k not in models.REGISTRY:
            print(f"error: unknown model {k}; known: {', '.join(models.REGISTRY)}", file=sys.stderr)
            return 2
        try:
            p = models.fetch(k, progress=lambda m: print(m, file=sys.stderr))
            print(f"ok {k}: {p}", file=sys.stderr)
        except Exception as e:
            print(f"error fetching {k}: {e}", file=sys.stderr)
            return 3
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run
    return run(verbose=not args.quiet)


def cmd_eval(args) -> int:
    from . import evaluation, events, pipeline, synth
    from .config import load_config
    cfg = load_config()
    work = Path(args.workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    out = {"warning": "synthetic/licensed audio only; real-world performance on authorized calls is unverified"}
    if not args.skip_detector:
        det = events.make_detector(cfg.get("detector", "auto"))
        sp = [synth.tts(t, v) for t, v in [("Thanks for taking the call, let me walk you through the plan.", "Daniel"),
                                             ("How much does the annual plan cost?", "Samantha"),
                                             ("We need to check with our finance team first.", "Karen"),
                                             ("That sounds good, we can start next week.", "Rishi")]]
        out["detector"] = {k: v for k, v in evaluation.detector_eval(det, sp).items() if k != "per_clip"}
        thr = dict(events.DEFAULT_THRESHOLDS[det.name])
        out["insertion"] = evaluation.insertion_eval(det, thr, work / "insertion")
    out["scenarios"] = evaluation.run_scenarios(work, cfg, pipeline.analyze, use_esc50=not args.no_esc50,
                                                log=lambda m: print(m, file=sys.stderr))
    evaluation.write(out, work / "evaluation.json")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_purge(args) -> int:
    from .config import load_config
    from .retention import purge
    cfg = load_config()
    r = purge(Path(args.reports_root).expanduser() if args.reports_root else None,
              args.reports_days or cfg["retention"]["reports_days"], args.feedback_days or cfg["retention"]["feedback_days"],
              dry_run=not args.yes)
    print(json.dumps(r, indent=2))
    return 0


def cmd_ingest(args) -> int:
    from .adapters import whatsapp_cloud
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    items = whatsapp_cloud.parse_webhook(payload)
    if not args.download_dir:
        print(json.dumps([i.__dict__ for i in items], indent=2))
        return 0
    paths = [str(whatsapp_cloud.download(i, Path(args.download_dir))) for i in items]
    print(json.dumps({"downloaded": paths}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-sales-coach", description="Local coaching notes from authorized sales "
                                "recordings: measured audio cues, replayable moments, no emotion guessing.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"audio-sales-coach {__version__}")
    p.add_argument("--config", help="config JSON (default ~/.config/audio-sales-coach/config.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse one recording you supply")
    a.add_argument("recording")
    a.add_argument("--output", "-o", required=True, help="report directory (created, private)")
    a.add_argument("--mode", choices=["auto", "call", "voice-message"], default="auto")
    a.add_argument("--transcribe", choices=["auto", "on", "off"], default="auto")
    a.add_argument("--council", choices=["auto", "on", "off"], default="auto")
    a.add_argument("--council-providers", help="comma list for evidence-council (default: none = local checks only)")
    a.add_argument("--salesperson", choices=["S1", "S2"], help="you already know which speaker is the salesperson")
    a.add_argument("--consent", choices=["all-parties-consented", "notified", "unknown"], default="unknown",
                   help="what participants were told about recording/coaching use (recorded in the report)")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    pl = sub.add_parser("play", help="play a moment of the ORIGINAL recording")
    pl.add_argument("report")
    pl.add_argument("--note")
    pl.add_argument("--at", type=float)
    pl.add_argument("--duration", type=float)
    pl.add_argument("--print-only", action="store_true", help="print the ffplay command instead of playing")
    pl.set_defaults(fn=cmd_play)

    r = sub.add_parser("roles", help="assign who is the salesperson; re-render notes")
    r.add_argument("report")
    r.add_argument("--salesperson", required=True, choices=["S1", "S2"])
    r.add_argument("--remember", metavar="PROFILE", help="add this speaker's voice to a salesperson profile")
    r.add_argument("--consent", help="salesperson's consent statement (needed with --remember)")
    r.add_argument("--reason", help="why the automatic label was wrong (stored as feedback)")
    r.add_argument("--council", choices=["auto", "on", "off"], default="auto")
    r.set_defaults(fn=cmd_roles)

    f = sub.add_parser("feedback", help="mark notes/observations useful|incorrect|uncertain; correct speakers")
    f.add_argument("action", nargs="?", choices=["add", "list", "summary"], default="add")
    f.add_argument("report", nargs="?")
    f.add_argument("--item", help="note-N or obs-N")
    f.add_argument("--mark", choices=["useful", "incorrect", "uncertain"])
    f.add_argument("--reason")
    f.add_argument("--speaker", help="speaker label correction, e.g. S2=salesperson")
    f.set_defaults(fn=cmd_feedback)

    v = sub.add_parser("voice", help="salesperson voice profiles (local, consented, deletable)")
    v.add_argument("action", choices=["enroll", "re-enroll", "list", "delete", "calibrate", "identify"])
    v.add_argument("name", nargs="?")
    v.add_argument("--sample", help="audio of the salesperson alone (30-60 s recommended)")
    v.add_argument("--consent", help="the salesperson's own consent statement")
    v.add_argument("--replace", action="store_true")
    v.set_defaults(fn=cmd_voice)

    m = sub.add_parser("models", help="model status / download (verified by size and SHA-256)")
    m.add_argument("action", choices=["status", "fetch"])
    m.add_argument("--with", dest="with_", action="append", help="also fetch an optional model (e.g. ced_mini)")
    m.add_argument("--whisper", choices=["tiny", "base", "small", "medium"], help="also fetch a faster-whisper model")
    m.add_argument("--only-whisper", action="store_true", help="fetch only the whisper model")
    m.set_defaults(fn=cmd_models)

    s = sub.add_parser("selftest", help="offline end-to-end check on synthetic audio")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_selftest)

    e = sub.add_parser("eval", help="evaluation on synthetic calls and licensed ESC-50 clips")
    e.add_argument("--workdir", default="./asc-eval")
    e.add_argument("--skip-detector", action="store_true")
    e.add_argument("--no-esc50", action="store_true", help="do not download/insert ESC-50 clips")
    e.set_defaults(fn=cmd_eval)

    pu = sub.add_parser("purge", help="apply retention: delete old reports/feedback (dry run unless --yes)")
    pu.add_argument("--reports-root")
    pu.add_argument("--reports-days", type=int)
    pu.add_argument("--feedback-days", type=int)
    pu.add_argument("--yes", action="store_true")
    pu.set_defaults(fn=cmd_purge)

    i = sub.add_parser("ingest", help="adapters for authorized sources (WhatsApp Business Platform webhooks)")
    i.add_argument("source", choices=["whatsapp-webhook"])
    i.add_argument("payload", help="webhook JSON your own endpoint received")
    i.add_argument("--download-dir", help="download the voice notes (needs WHATSAPP_ACCESS_TOKEN)")
    i.set_defaults(fn=cmd_ingest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
