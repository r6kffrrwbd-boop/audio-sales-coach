"""End-to-end analysis of one recording."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import REPORT_SCHEMA_VERSION, __version__, audio_io, coaching, council, events, features, models, observations
from . import speakers as spk_mod
from . import transcribe as tx_mod
from . import voiceid
from .config import private_mkdir

CONSENT_STATES = ("all-parties-consented", "notified", "unknown")


class _MultiTranscript:
    """Per-channel transcripts for separate-channel calls (words are attributed by channel)."""

    def __init__(self, by_speaker: dict):
        self.by = by_speaker

    def for_speaker(self, s):
        return self.by.get(s)


def _snr(db: np.ndarray, mask: np.ndarray) -> float | None:
    n = min(db.size, mask.size)
    db, mask = db[:n], mask[:n]
    if mask.sum() < 50 or (~mask).sum() < 50:
        return None
    return float(np.median(db[mask]) - np.percentile(db[~mask], 50))


def analyze(path: str, output_dir: str, cfg: dict, mode: str = "auto", transcription: str = "auto",
            council_mode: str = "auto", user_roles: dict | None = None, consent: str = "unknown",
            log=print) -> dict:
    t_start = time.time()
    out = private_mkdir(Path(output_dir).expanduser().resolve())
    timings = {}

    def tick(name, t0):
        timings[name] = round(time.time() - t0, 2)

    # 1. decode
    t0 = time.time()
    dec = audio_io.decode(path)
    tick("decode", t0)
    sr = dec.sr
    info = dec.info
    log(f"decoded {Path(info.path).name}: {info.codec}, {info.channels} ch, {dec.duration:.1f} s")
    rid = "rpt-" + hashlib.sha256(f"{info.sha256}:{time.time()}".encode()).hexdigest()[:12]
    warnings = list(dec.notes)
    if consent not in CONSENT_STATES:
        raise ValueError(f"consent must be one of {CONSENT_STATES}")
    if consent == "unknown":
        warnings.append("consent/notice for this recording was not confirmed (--consent); only analyse calls where "
                        "every participant was told about recording and coaching use, as your local law requires")

    # 2. layout and speaker turns
    t0 = time.time()
    vad = spk_mod.Vad(cfg["vad"]["threshold"], cfg["vad"]["min_speech_s"], cfg["vad"]["min_silence_s"])
    layout, layout_info = spk_mod.detect_layout(dec.channels, sr)
    embedder = None
    try:
        embedder = spk_mod.Embedder()
    except RuntimeError as e:
        warnings.append(str(e))
    if layout == "separate_channels" and mode != "voice-message":
        sres = spk_mod.turns_from_channels(dec.channels, sr, vad)
        audio_for = {"S1": dec.channels[0], "S2": dec.channels[1]}
    else:
        x = dec.mono
        sres = spk_mod.diarize_mono(x, sr, vad, embedder, mode=mode,
                                    cluster_threshold=cfg["speakers"]["cluster_threshold"])
        audio_for = {s: x for s in sres.speakers}
    warnings += sres.notes
    tick("speakers", t0)
    log(f"speaker layout: {sres.layout}; speakers: {', '.join(sres.speakers) or 'none'}; {len(sres.segments)} segments")

    # 3. measurements
    t0 = time.time()
    feats: dict[str, features.ChannelFeatures] = {}
    snr = {}
    if sres.layout == "separate_channels":
        for s, ch in sres.channel_of.items():
            segs = [(g.start, g.end) for g in sres.segments if g.speaker == s]
            n = int(dec.channels.shape[1] / sr / features.HOP_S) + 2
            m = spk_mod.mask_from(segs, n)
            feats[s] = features.channel_features(dec.channels[ch], sr, m)
            snr[s] = _snr(feats[s].db, m)
    else:
        x = dec.mono
        n = int(x.size / sr / features.HOP_S) + 2
        m = spk_mod.mask_from([(g.start, g.end) for g in sres.segments], n)
        f = features.channel_features(x, sr, m)
        for s in sres.speakers or ["S1"]:
            feats[s] = f
        snr["mix"] = _snr(f.db, m)
    tick("measurements", t0)

    # 4. audio events (detector on the original audio, never a language model)
    t0 = time.time()
    evs: list[events.Event] = []
    det_info = {"status": "unavailable"}
    try:
        det = events.make_detector(cfg.get("detector", "auto"))
        thr = dict(events.DEFAULT_THRESHOLDS[det.name])
        thr.update({k: v for k, v in (cfg.get("event_thresholds") or {}).items() if v is not None})
        tracks = []
        if sres.layout == "separate_channels":
            for s, ch in sres.channel_of.items():
                tr = det.track(dec.channels[ch], sr)
                tracks.append(tr)
                evs += events.extract_events(tr, thr, channel=ch)
            evs = _dedupe_bleed(evs)
        else:
            tr = det.track(dec.mono, sr)
            tracks.append(tr)
            evs = events.extract_events(tr, thr)
        det_info = {"status": "ok", "detector": det.name, "model_file": det.model_file, "thresholds": thr,
                    "window_s": det.window_s, "hop_s": det.hop_s,
                    "speech_windows_share": round(float(np.mean(np.concatenate([t.group_scores["speech"] for t in tracks]) > 0.5)), 3)}
    except events.DetectorUnavailable as e:
        warnings.append(f"audio-event detector unavailable: {e}; laughter/breathing cues were not analysed")
        det_info = {"status": "unavailable", "reason": str(e)}
    tick("events", t0)
    log(f"audio events: {len(evs)} ({', '.join(sorted({e.type for e in evs})) or 'none'})")

    # 5. optional transcription
    t0 = time.time()
    transcript = None
    tx_info = {"status": "off"}
    want_tx = transcription == "on" or (transcription == "auto" and cfg["transcription"]["enabled"] in ("auto", True)
                                          and tx_mod.available())
    if transcription == "on" and not tx_mod.available():
        warnings.append("transcription requested but faster-whisper is not installed (./install.sh --with-transcription)")
        want_tx = False
    if want_tx:
        try:
            if sres.layout == "separate_channels":
                by = {s: tx_mod.transcribe(dec.channels[ch], sr, cfg["transcription"]) for s, ch in sres.channel_of.items()}
                transcript = _MultiTranscript(by)
                first = next(iter(by.values()))
                tx_info = {"status": "ok", "model": first.model, "per_channel": True,
                           "languages": {s: [t.language, t.language_prob] for s, t in by.items()},
                           "notes": sum((t.notes for t in by.values()), [])}
            else:
                transcript = tx_mod.transcribe(dec.mono, sr, cfg["transcription"])
                tx_info = {"status": "ok", "model": transcript.model, "language": transcript.language,
                           "language_prob": transcript.language_prob, "notes": transcript.notes,
                           "unreliable_segments": sum(1 for s in transcript.segments if not s.reliable),
                           "segments": len(transcript.segments)}
            warnings += tx_info.get("notes", [])
        except Exception as e:  # transcription is optional; audio analysis continues
            warnings.append(f"transcription failed ({type(e).__name__}: {e}); continuing audio-only")
            tx_info = {"status": "failed", "reason": str(e)[:300]}
            transcript = None
    elif transcription != "off":
        tx_info = {"status": "not installed (audio-only analysis)"}
    tick("transcription", t0)

    # 6. turns
    turns = observations.build_turns(sres, feats, transcript)

    # 7. roles: user > channel map > voice profile > unknown
    t0 = time.time()
    match_summary = None
    if embedder is not None and voiceid.list_profiles() and sres.speakers:
        embs = {s: spk_mod.speaker_embedding_for(audio_for[s], sr, sres.segments, s, embedder) for s in sres.speakers}
        _, match_summary = voiceid.identify(embs, cfg["voice_id"])
        if match_summary.get("status") == "confident":
            matched = [r["profile"] for r in match_summary["scores"].values() if r["decision"] == "match"][0]
            others = [r["similarity"] for r in match_summary["scores"].values() if r["decision"] != "match"]
            voiceid.record_scores(matched, impostor=others, report_id=rid)
    roles, role_source = coaching.roles_from(sres.speakers, sres.layout, match_summary,
                                             cfg["speakers"].get("channel_map") or {}, user_roles)
    tick("voice_id", t0)

    # 8. observations and notes
    obs = observations.derive(sres, turns, evs, feats, cfg, dec.duration)
    notes = coaching.build_notes(obs, turns, roles, info.path, str(out), transcript is not None, sres.layout)

    # 9. role assignment request
    needs_roles = len(sres.speakers) == 2 and "salesperson" not in roles.values()
    excerpts = _excerpts(sres, info.path, str(out)) if (needs_roles or role_source == "voice_profile") else []

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": {"name": "audio-sales-coach", "version": __version__, "python": platform.python_version()},
        "report_id": rid,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recording": {"path": info.path, "sha256": info.sha256, "container": info.container, "codec": info.codec,
                      "channels": info.channels, "sample_rate": info.sample_rate, "duration_s": round(dec.duration, 2),
                      "bit_rate": info.bit_rate, "consent": consent,
                      "converted_audio": "decoded in memory only; no converted copy written"},
        "mode": mode,
        "layout": {"result": sres.layout, "detail": layout_info},
        "methods": {
            **features.METHODS,
            "voice_activity": vad.method,
            "speaker_turns": sres.method,
            "events": det_info,
            "transcription": tx_info,
            "source_separation": "not used; all evidence refers to the original recording",
        },
        "reliability": _reliability(sres, snr, info, tx_info, det_info),
        "speakers": _speakers_block(sres, turns, roles, role_source, match_summary),
        "role_assignment": {"status": "needs_user" if needs_roles else ("confident" if role_source != "none" else "not_needed"),
                            "source": role_source, "voice_id": match_summary,
                            "how_to_assign": (f"audio-sales-coach roles {out} --salesperson S1|S2 [--remember PROFILE]"
                                              if needs_roles else None),
                            "excerpts": excerpts},
        "turns": [_turn_dict(t) for t in turns],
        "segments": [_seg_dict(g) for g in sres.segments],
        "overlaps": [[round(a, 2), round(b, 2)] for a, b in sres.overlaps],
        "events": [{"id": f"ev-{i}", "type": e.type, "start": round(e.start, 2), "end": round(e.end, 2), "score": e.score,
                    "classes": e.classes, "detector": e.detector,
                    "channel": None if e.channel is None else e.channel + 1} for i, e in enumerate(evs, 1)],
        "observations": [o.to_dict() for o in obs],
        "coaching_notes": [],
        "views": {},
        "council": {},
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "timings_s": timings,
    }

    # 10. council (optional, independent challenge of the drafts)
    t0 = time.time()
    if council_mode == "off" or (council_mode == "auto" and cfg["council"]["enabled"] is False):
        cres = {"status": "skipped", "reason": "council review turned off"}
    elif not notes:
        cres = {"status": "skipped", "reason": "no draft notes to review"}
    else:
        task = council.build_task(report, notes, obs, roles, cfg, {t.id: t for t in turns})
        cres = council.review(task, out / "council", cfg)
    council.apply_to_notes(notes, cres)
    tick("council", t0)
    log(f"council: {cres['status']}" + (f" ({cres.get('reason')})" if cres.get("reason") else
                                        f" (verdict {cres.get('verdict_status')})"))
    report["coaching_notes"] = [n.to_dict() for n in notes]
    report["council"] = cres
    report["views"] = build_views(report, roles)
    timings["total"] = round(time.time() - t_start, 2)

    write_report(report, out)
    if cfg["privacy"].get("keep_converted_audio"):
        audio_io.write_wav(out / "converted-16k.wav", dec.channels, sr)
        report["recording"]["converted_audio"] = str(out / "converted-16k.wav")
        write_report(report, out)
    del dec
    return report


def _renumber(turns):
    for i, t in enumerate(turns, 1):
        t.id = f"turn-{i}"
    return turns


def _dedupe_bleed(evs: list[events.Event]) -> list[events.Event]:
    """Same event type at the same time on both channels: keep the louder-scoring channel unless both are strong."""
    keep = []
    for e in evs:
        twins = [o for o in evs if o is not e and o.type == e.type and o.channel != e.channel
                 and min(o.end, e.end) - max(o.start, e.start) > 0.3]
        if twins and any(o.score > e.score + 0.2 for o in twins):
            continue
        keep.append(e)
    return keep


def _excerpts(sres, rec_path: str, rdir: str) -> list[dict]:
    out = []
    for s in sres.speakers:
        cands = sorted([g for g in sres.segments if g.speaker == s and 2.0 <= g.duration and not g.overlap
                        and not g.uncertain], key=lambda g: -g.confidence)[:2]
        for g in cands:
            a, b = g.start, min(g.end, g.start + 5.0)
            ref = coaching.play_ref(rec_path, a + 2.0, b - 2.0, None, "")
            out.append({"speaker": s, "start": round(a, 2), "end": round(b, 2), "uri": ref["uri"], "ffplay": ref["ffplay"],
                        "command": f"audio-sales-coach play {rdir} --at {a:.1f} --duration {b - a:.1f}"})
    return out


def _turn_dict(t) -> dict:
    return {"id": t.id, "speaker": t.speaker, "start": round(t.start, 2), "end": round(t.end, 2),
            "segment_count": t.segment_count, "attribution_confidence": t.attribution_confidence, "uncertain": t.uncertain, "overlap": t.overlap,
            "possible_overlap": t.possible_overlap, "stats": t.stats, "pauses": t.pauses, "words": t.words,
            "words_reliable": t.words_reliable, "topics": t.topics, "question_text": t.question_text,
            "rising_end_st_per_s": t.rising_end_st_per_s}


def _seg_dict(g) -> dict:
    return {"start": round(g.start, 2), "end": round(g.end, 2), "speaker": g.speaker, "confidence": round(g.confidence, 3),
            "channel": None if g.channel is None else g.channel + 1, "overlap": g.overlap,
            "possible_overlap": g.possible_overlap, "uncertain": g.uncertain}


def _speakers_block(sres, turns, roles, source, match):
    total = sum(t.end - t.start for t in turns) or 1.0
    out = []
    for s in sres.speakers:
        tt = [t for t in turns if t.speaker == s]
        talk = sum(t.end - t.start for t in tt)
        row = {"id": s, "role": roles.get(s, "unknown"), "role_source": source if roles.get(s) not in ("unknown",) else "none",
               "turns": len(tt), "talk_s": round(talk, 1), "talk_share": round(talk / total, 3),
               "mean_attribution_confidence": round(float(np.mean([t.attribution_confidence for t in tt])), 3) if tt else None}
        if sres.channel_of:
            row["channel"] = sres.channel_of[s] + 1
        if match and match.get("scores", {}).get(s):
            row["voice_id"] = match["scores"][s]
        out.append(row)
    return out


def _reliability(sres, snr, info, tx_info, det_info) -> dict:
    segs = sres.segments
    unc = sum(g.duration for g in segs if g.uncertain) / max(sum(g.duration for g in segs), 1e-6)
    snr_vals = [v for v in snr.values() if v is not None]
    noisy = bool(snr_vals and min(snr_vals) < 12)
    compressed = info.codec in ("opus", "mp3", "aac", "vorbis") and (info.bit_rate or 64000) < 32000
    att = {"separate_channels": "high (one speaker per channel)",
           "mixed_mono": "moderate to low (speaker clustering on a mixed recording; overlaps cannot be separated)",
           "single_speaker": "not applicable (one speaker)"}[sres.layout]
    return {
        "acoustic": {"speech_to_background_db": {k: (round(v, 1) if v is not None else None) for k, v in snr.items()},
                     "noisy": noisy, "low_bitrate_compression": compressed,
                     "statement": ("cues are less reliable: " + ", ".join(x for x, f in
                                   [("background noise", noisy), ("low-bit-rate compression", compressed)] if f))
                     if (noisy or compressed) else "adequate for level, pause and turn measurements"},
        "speaker_attribution": {"level": att, "uncertain_speech_share": round(unc, 3)},
        "transcription": {"status": tx_info.get("status"),
                          "statement": ("words are approximate; Lebanese Arabic and Arabic/English code-switching are "
                                        "often mis-transcribed" if tx_info.get("status") == "ok" else
                                        "no transcript; notes are based on audio cues only")},
        "events": {"status": det_info.get("status"),
                   "statement": "laughter detection is moderately reliable; breathing detection is weak (see docs/evaluation.md)"},
    }


def build_views(report: dict, roles: dict) -> dict:
    """Three synchronized views (salesperson, client, combined) on one timebase; one view for a single speaker."""
    def view_for(spk_ids):
        ids = set(spk_ids)
        return {
            "speaker_ids": sorted(ids),
            "turn_ids": [t["id"] for t in report["turns"] if t["speaker"] in ids],
            "event_ids": [e["id"] for e in report["events"]],
            "observation_ids": [o["id"] for o in report["observations"] if ids & set(o["speakers"])
                                or o["measurement"].get("attributed_to") in ids],
            "note_ids": [n["id"] for n in report["coaching_notes"] if n["speaker"]["id"] in ids
                         or any(oid in {o["id"] for o in report["observations"] if ids & set(o["speakers"])}
                                for oid in n["observation_ids"])],
            "talk_s": round(sum(t["end"] - t["start"] for t in report["turns"] if t["speaker"] in ids), 1),
        }
    spks = [s["id"] for s in report["speakers"]]
    v = {"timebase": "seconds from the start of the original recording (shared by every view)"}
    if report["layout"]["result"] == "single_speaker" or len(spks) <= 1:
        v["speaker"] = view_for(spks)
        v["note"] = "single-speaker recording: one view; no second person is inferred"
        return v
    sp = [s for s in spks if roles.get(s) == "salesperson"]
    cl = [s for s in spks if roles.get(s) == "client_in_this_call"]
    if sp and cl:
        v["salesperson"] = view_for(sp)
        v["client"] = view_for(cl)
    else:
        for s in spks:
            v[f"speaker_{s}"] = view_for([s])
        v["note"] = "roles not assigned yet: views are per speaker; run `audio-sales-coach roles` to label them"
    v["combined"] = {**view_for(spks), "overlaps": report["overlaps"],
                     "event_ids": [e["id"] for e in report["events"]]}
    return v


LIMITATIONS = [
    "Real-world accuracy on authorized sales calls has not been measured; evaluation so far uses synthetic speech and "
    "licensed sound-event clips (docs/evaluation.md).",
    "Cues are observations, not emotions: laughter, breathing, pauses, pitch and pace changes have many causes.",
    "On mixed mono recordings, speaker attribution comes from voice clustering and can be wrong, especially for similar "
    "voices, short turns and overlapping speech.",
    "Breathing detection is weak; treat breathing cues as prompts to replay, not as evidence.",
    "Mixed Lebanese Arabic/English transcription (optional) is approximate; acoustic cues do not depend on it.",
]


def write_report(report: dict, out: Path) -> None:
    from .report_render import render_html, render_markdown
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "report.html").write_text(render_html(report), encoding="utf-8")
    for f in ("report.json", "report.md", "report.html"):
        try:
            os.chmod(out / f, 0o600)
        except OSError:
            pass


def load_report(report_dir: str) -> dict:
    p = Path(report_dir).expanduser() / "report.json"
    if not p.exists():
        raise FileNotFoundError(f"no report.json in {report_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def models_ready() -> dict:
    return {k: models.is_installed(k) for k in models.REGISTRY}


def rebuild_with_roles(report_dir: str, roles: dict, cfg: dict, council_mode: str = "auto") -> dict:
    """Re-render notes, views and (optionally) council review for new roles - from report data, no audio needed."""
    out = Path(report_dir).expanduser().resolve()
    report = load_report(str(out))
    turns = [observations.Turn(**t) for t in report["turns"]]
    obs = [observations.Observation(**o) for o in report["observations"]]
    layout = report["layout"]["result"]
    notes = coaching.build_notes(obs, turns, roles, report["recording"]["path"], str(out),
                                 report["methods"]["transcription"].get("status") == "ok", layout)
    for s in report["speakers"]:
        s["role"] = roles.get(s["id"], s["role"])
        s["role_source"] = "user"
    report["role_assignment"].update({"status": "confident", "source": "user", "how_to_assign": None})
    if council_mode == "off" or not notes:
        cres = {"status": "skipped", "reason": "council review turned off" if notes else "no draft notes to review"}
    else:
        task = council.build_task(report, notes, obs, roles, cfg, {t.id: t for t in turns})
        cres = council.review(task, out / "council", cfg)
    council.apply_to_notes(notes, cres)
    report["coaching_notes"] = [n.to_dict() for n in notes]
    report["council"] = cres
    report["views"] = build_views(report, roles)
    write_report(report, out)
    return report
