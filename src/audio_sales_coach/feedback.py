"""Salesperson feedback on observations, notes and speaker labels.

Stored separately from recordings and reports, in `<data dir>/feedback/<report_id>.jsonl` (0600). Feedback is
used for evaluation (how often each cue type is judged useful / incorrect / uncertain) and to remember speaker-role
corrections. It does NOT retrain any model: no training is implemented in this project.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config import feedback_dir, private_mkdir

MARKS = ("useful", "incorrect", "uncertain")


def _file(report_id: str) -> Path:
    if not report_id.startswith("rpt-") or "/" in report_id:
        raise ValueError("bad report id")
    return private_mkdir(feedback_dir()) / f"{report_id}.jsonl"


def add(report: dict, item: str | None, mark: str | None, reason: str | None = None,
        speaker_correction: dict | None = None) -> dict:
    kind = None
    target = None
    if item:
        if item.startswith("note-"):
            kind = "note"
            target = next((n for n in report["coaching_notes"] if n["id"] == item), None)
        elif item.startswith("obs-"):
            kind = "observation"
            target = next((o for o in report["observations"] if o["id"] == item), None)
        if target is None:
            raise ValueError(f"{item} is not in report {report['report_id']}")
        if mark not in MARKS:
            raise ValueError(f"--mark must be one of {MARKS}")
    elif not speaker_correction:
        raise ValueError("give --item with --mark, or --speaker S1=role")
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "report_id": report["report_id"],
             "item": item, "kind": kind, "mark": mark, "reason": (reason or "")[:2000],
             "cue_type": (target["cue"]["type"] if kind == "note" else target["type"]) if target else None,
             "detector": (target["cue"]["detector"] if kind == "note" else target.get("detector")) if target else None,
             "speaker_correction": speaker_correction}
    p = _file(report["report_id"])
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def entries(report_id: str | None = None) -> list[dict]:
    d = feedback_dir()
    if not d.exists():
        return []
    files = [d / f"{report_id}.jsonl"] if report_id else sorted(d.glob("*.jsonl"))
    out = []
    for p in files:
        if p.exists():
            out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def summary() -> dict:
    es = entries()
    by = defaultdict(Counter)
    for e in es:
        if e.get("mark"):
            by[e.get("cue_type") or "unknown"][e["mark"]] += 1
    rows = {}
    for cue, c in by.items():
        judged = c["useful"] + c["incorrect"]
        rows[cue] = {**{m: c[m] for m in MARKS}, "useful_rate_among_judged": round(c["useful"] / judged, 3) if judged else None}
    corrections = [e for e in es if e.get("speaker_correction")]
    return {"entries": len(es), "reports": len({e["report_id"] for e in es}), "by_cue_type": rows,
            "speaker_corrections": len(corrections),
            "note": "Feedback is stored for evaluation and role memory only; it does not retrain any model."}
