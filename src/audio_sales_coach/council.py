"""Optional independent review of draft notes by `evidence-council` (a separate program).

Uses only the council's documented executable JSON interface:
  evidence-council review task.json --output result.json   (exit codes: `evidence-council codes`)
  evidence-council explain result.json --json

What is sent (and saved next to the report as council/task.json for audit):
  * one `recording` evidence item with no URI and no content (an anchor for citations)
  * one `observation` per relevant cue: timestamp, neutral role-labelled statement, keyed facts
  * every client question observation (so missed questions are detectable), even without a note
  * each draft note as a candidate whose claims (cue, speaker role, attribution reliability) cite observations
What is never sent: audio, file paths, transcripts or quotes (unless council.include_quotes is on),
names, phone numbers, e-mail addresses, CRM data, voice embeddings.
Default providers: ["none"] (local fact/citation checks only; nothing reaches any model).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .coaching import Note, hms, render
from .observations import Observation

EXIT_MEANING = {0: "ok", 1: "council internal error", 2: "invalid input", 3: "not enough usable providers",
                4: "blocked by privacy policy", 5: "recursive council call refused", 7: "missing prerequisite stage",
                8: "background job still running"}

_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PATH = re.compile(r"(/Users/|/home/|[A-Za-z]:\\|file://)")


def find_command(cfg: dict) -> str | None:
    cmd = cfg["council"].get("command") or "evidence-council"
    return shutil.which(cmd)


def scrub(text: str, extra_terms: list[str]) -> str:
    t = _EMAIL.sub("[email]", text)
    t = _PHONE.sub("[number]", t)
    for term in extra_terms:
        if term and len(term) >= 2:
            t = re.sub(re.escape(term), "[name]", t, flags=re.IGNORECASE)
    return t


def payload_problems(task: dict) -> list[str]:
    """Defence in depth: refuse to send anything that looks like a path, e-mail or phone number."""
    s = json.dumps(task, ensure_ascii=False)
    texts = [e.get("content", "") + " " + e.get("title", "") for e in task.get("evidence", [])]
    texts += [c.get("answer", "") + " " + " ".join(cl.get("statement", "") for cl in c.get("claims", []))
              for c in task.get("candidates", [])]
    text = " ".join(texts)
    probs = []
    if _PATH.search(s):
        probs.append("file path")
    if _EMAIL.search(s):
        probs.append("e-mail address")
    if any(len(re.sub(r"\D", "", m.group(0))) >= 7 for m in _PHONE.finditer(text)):
        probs.append("long digit sequence (possible phone number)")
    if any(e.get("kind") in ("audio", "recording") and (e.get("uri") or e.get("content")) for e in task.get("evidence", [])):
        probs.append("recording evidence carries a URI or content")
    return probs


def build_task(report: dict, notes: list[Note], observations: list[Observation], roles: dict, cfg: dict,
               turns_by_id: dict | None = None) -> dict:
    terms = cfg["privacy"].get("redact_terms") or []
    rid = report["report_id"]
    by_id = {o.id: o for o in observations}
    used = {oid for n in notes for oid in n.observation_ids if oid in by_id}
    questions = [o for o in observations if o.type in ("question", "rising_ending")
                 and o.speakers and roles.get(o.speakers[0]) == "client_in_this_call"]
    used |= {o.id for o in questions}
    evidence = [{"id": "rec", "kind": "recording", "title": "call recording (not forwarded)", "source_ref": rid}]
    for oid in sorted(used, key=lambda i: by_id[i].start):
        o = by_id[oid]
        primary = o.speakers[0] if o.speakers else o.measurement.get("attributed_to", "unknown")
        role = roles.get(primary, "unknown") if primary not in ("unknown", "both", None) else "unknown"
        facts = [{"key": f"{oid}.cue", "value": o.type},
                 {"key": f"{oid}.speaker_role", "value": role},
                 {"key": f"{oid}.attribution_reliable", "value": bool(o.attribution_reliable)},
                 {"key": f"{oid}.uncertain", "value": bool(o.uncertain)}]
        if o.score is not None:
            facts.append({"key": f"{oid}.score", "value": float(o.score)})
        content = render(o.statement, roles)
        if o.uncertainty:
            content += f" Uncertainty: {o.uncertainty}."
        if o.alternatives:
            content += " Other explanations: " + "; ".join(render(a, roles) for a in o.alternatives) + "."
        if cfg["council"].get("include_quotes") and turns_by_id:
            words = " / ".join(turns_by_id[t].words for t in o.turn_ids if t in turns_by_id and turns_by_id[t].words)
            if words:
                content += f" Words: \"{words[:300]}\""
        evidence.append({"id": oid, "kind": "observation", "timestamp": hms(o.start),
                         "source_ref": f"{rid}#t={o.start:.1f}", "content": scrub(content, terms), "facts": facts})
    covered = {oid for n in notes for oid in n.observation_ids}
    missed = [o for o in questions if o.id not in covered]
    evidence.append({"id": "coverage", "kind": "measurement", "title": "note coverage computed by audio-sales-coach",
                     "content": f"{len(questions)} client question cue(s); {len(missed)} without a coaching note.",
                     "facts": [{"key": "client_questions_total", "value": len(questions)},
                               {"key": "client_questions_without_note", "value": len(missed)}]})
    candidates = []
    for n in notes:
        claims = []
        for oid in n.observation_ids:
            if oid not in by_id:
                continue
            o = by_id[oid]
            claims.append({"statement": f"The cue is {o.type}", "key": f"{oid}.cue", "value": o.type,
                           "evidence_ids": [oid], "timestamp": hms(o.start)})
            if oid == n.observation_ids[0]:
                claims.append({"statement": f"The cue belongs to the {n.speaker['role'].replace('_', ' ')}",
                               "key": f"{oid}.speaker_role", "value": n.speaker["role"], "evidence_ids": [oid]})
                claims.append({"statement": "Speaker attribution is reliable" if n.speaker["attribution_reliable"]
                               else "Speaker attribution is not reliable",
                               "key": f"{oid}.attribution_reliable", "value": bool(n.speaker["attribution_reliable"]),
                               "evidence_ids": [oid]})
        if not claims:
            continue
        candidates.append({"id": n.id, "label": f"draft note at {n.timestamp}",
                           "answer": scrub(f"{n.interaction} Suggestion: {n.suggestion}", terms), "claims": claims,
                           "assumptions": ["The note describes audible cues only, not feelings or intentions."]})
    candidates.append({"id": "coverage-claim", "label": "coverage of client questions",
                       "answer": "Every client question cue has a coaching note.",
                       "claims": [{"statement": "No client question was left without a note",
                                   "key": "client_questions_without_note", "value": 0, "evidence_ids": ["coverage"]}]})
    task = {
        "schema_version": "1.0",
        "task": {"id": f"asc-{rid}"[:200], "title": "Review draft sales-coaching notes",
                 "prompt": ("Independently challenge these draft coaching notes against the timestamped observations. "
                            "Flag unsupported claims about emotions or intentions, wrong speaker attribution, missed "
                            "client questions, and alternative explanations for each cue. Do not infer feelings."),
                 "domain": "coaching", "answer_format": "text", "tags": ["sales-coaching", "audio-sales-coach"]},
        "evidence": evidence,
        "constraints": ["Never state as fact that a person was nervous, lying, comfortable or ready to buy.",
                        "No pressure or manipulation tactics."],
        "candidates": candidates[:50],
        "checks": [{"id": "facts", "type": "evidence_facts", "decisive": True,
                    "description": "Each note's cue, speaker role and attribution must match the observation facts."},
                   {"id": "cites", "type": "citations", "require_citations": True, "decisive": True,
                    "description": "Every claim must cite an existing observation with a matching timestamp."}],
        "privacy": {"level": cfg["council"].get("privacy_level", "private"), "redact": True},
        "providers": {"requested": list(cfg["council"].get("providers") or ["none"])},
        "options": {"use_lessons": False, "create_lesson": False},
    }
    return task


def review(task: dict, out_dir: Path, cfg: dict) -> dict:
    cmd = find_command(cfg)
    if not cmd:
        return {"status": "skipped", "reason": "evidence-council is not installed; council review was skipped"}
    probs = payload_problems(task)
    if probs:
        return {"status": "skipped", "reason": "payload blocked by audio-sales-coach privacy check: " + ", ".join(probs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    tpath, rpath = out_dir / "task.json", out_dir / "result.json"
    tpath.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        r = subprocess.run([cmd, "review", str(tpath), "--output", str(rpath)], capture_output=True, text=True,
                           timeout=cfg["council"].get("timeout_s", 900))
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "council review timed out", "task_file": str(tpath)}
    base = {"command": "evidence-council review", "exit_code": r.returncode,
            "exit_meaning": EXIT_MEANING.get(r.returncode, "unknown"), "task_file": str(tpath),
            "providers_requested": task["providers"]["requested"]}
    if r.returncode != 0 or not rpath.exists():
        return {**base, "status": "error" if r.returncode not in (3, 4, 5) else "not_reviewed",
                "reason": (r.stderr or r.stdout).strip()[-500:]}
    result = json.loads(rpath.read_text(encoding="utf-8"))
    explain = None
    try:
        e = subprocess.run([cmd, "explain", str(rpath), "--json"], capture_output=True, text=True, timeout=120)
        if e.returncode == 0:
            explain = json.loads(e.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    per_note: dict[str, dict] = {}
    for c in result.get("claims", []):
        pid = str(c.get("proposal_id", "")).removeprefix("cand:")
        d = per_note.setdefault(pid, {"supported": 0, "contradicted": 0, "citation_problem": 0, "unverified": 0, "issues": []})
        st = c.get("status", "unverified")
        d[st] = d.get(st, 0) + 1
        if st not in ("supported",):
            d["issues"].append(f"{c.get('statement')}: {st}")
    rejected = {}
    for rj in result.get("verdict", {}).get("rejected", []):
        for h in rj.get("held_by", []):
            rejected[h] = rj.get("reasons", [])
    comments = []
    for p in result.get("proposals", []):
        if p.get("source") == "provider":
            comments.append({"provider": p.get("origin"), "text": str(p.get("answer", ""))[:1500],
                             "uncertainties": [str(u)[:300] for u in (p.get("uncertainties") or [])][:5],
                             "status": "unverified model output - not checked against the observations; "
                                       "small local models can invent problems that are not in the notes"})
    return {**base, "status": "reviewed", "result_file": str(rpath), "provider_comments": comments,
            "warnings": result.get("warnings", []),
            "verdict_status": result.get("verdict", {}).get("status"),
            "providers_used": result.get("providers", {}).get("used", []),
            "headline": (explain or {}).get("headline"), "per_note": per_note, "rejected": rejected,
            "uncertainty": result.get("uncertainty")}


def apply_to_notes(notes: list[Note], council: dict) -> None:
    pn = council.get("per_note", {}) if council.get("status") == "reviewed" else {}
    for n in notes:
        d = pn.get(n.id)
        if council.get("status") != "reviewed":
            n.council = {"status": council.get("status", "skipped")}
        elif not d:
            n.council = {"status": "not_checked"}
        elif d.get("contradicted") or d.get("citation_problem"):
            n.council = {"status": "flagged", "issues": d["issues"]}
        elif d.get("supported"):
            n.council = {"status": "supported", "claims_checked": d["supported"]}
        else:
            n.council = {"status": "unverified"}
