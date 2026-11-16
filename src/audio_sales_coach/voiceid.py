"""Optional salesperson voice profiles (local, consented, deletable).

A profile holds speaker embeddings (biometric data) for one salesperson who consented. Profiles live in
`<data dir>/voice-profiles/` with 0700/0600 permissions, are never written into reports or the repository,
and can be listed, deleted or re-enrolled at any time. Client voice profiles are never stored: a client's
speaker embedding exists only in memory during one analysis.

Matching: cosine similarity between a call speaker's mean embedding and the profile's mean embedding.
  similarity >= accept_threshold  -> that speaker is labelled the salesperson
  similarity <  reject_threshold  -> not this salesperson
  in between, or both speakers match -> "unknown"; the user is asked to assign roles from short excerpts.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import private_mkdir, profiles_dir

SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ProfileError(Exception):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def profile_path(name: str) -> Path:
    if not SLUG.match(name):
        raise ProfileError("profile name: letters, digits, . _ - only (use an alias, not a full name)")
    return profiles_dir() / f"{name}.json"


def _write(path: Path, data: dict) -> None:
    private_mkdir(path.parent)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def load(name: str) -> dict:
    p = profile_path(name)
    if not p.exists():
        raise ProfileError(f"no voice profile named '{name}'")
    return json.loads(p.read_text(encoding="utf-8"))


def list_profiles() -> list[dict]:
    d = profiles_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"name": data["name"], "samples": len(data.get("embeddings", [])), "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"), "model": data.get("model"),
                    "thresholds": data.get("thresholds"), "consent": data.get("consent", {}).get("statement")})
    return out


def enroll(name: str, embedding: np.ndarray, model: str, consent: str, source: str, replace: bool = False,
           report_id: str | None = None) -> dict:
    if not consent or len(consent.strip()) < 10:
        raise ProfileError("a consent statement from the salesperson is required (--consent \"...\")")
    p = profile_path(name)
    if p.exists() and not replace:
        data = load(name)
        if data.get("model") != model:
            raise ProfileError(f"profile '{name}' was made with a different model; re-enroll with --replace")
    else:
        data = {"name": name, "created_at": _now(), "model": model, "embeddings": [], "history": [],
                "impostor_scores": [], "thresholds": None}
    data["consent"] = {"statement": consent.strip(), "recorded_at": _now(), "source": source}
    data["embeddings"].append([round(float(v), 6) for v in embedding])
    data["embeddings"] = data["embeddings"][-20:]
    data["updated_at"] = _now()
    data["history"].append({"at": _now(), "event": "re-enrolled" if replace else f"sample added ({source})",
                            "report_id": report_id})
    _write(p, data)
    return data


def delete(name: str) -> bool:
    p = profile_path(name)
    if not p.exists():
        return False
    # overwrite before unlinking (best effort on APFS/SSD; the file is also excluded from backups by location)
    try:
        size = p.stat().st_size
        with open(p, "r+b") as f:
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    p.unlink()
    return True


def mean_embedding(data: dict) -> np.ndarray:
    m = np.mean(np.asarray(data["embeddings"], dtype=np.float32), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def record_scores(name: str, impostor: list[float] | None = None, report_id: str | None = None) -> None:
    """Keep similarity *scores* of non-matching call speakers (numbers only, no client embeddings) for calibration."""
    data = load(name)
    if impostor:
        data["impostor_scores"] = (data.get("impostor_scores", []) + [round(float(s), 4) for s in impostor])[-200:]
    data["updated_at"] = _now()
    if report_id:
        data["history"].append({"at": _now(), "event": "scored in call", "report_id": report_id})
    _write(profile_path(name), data)


def calibrate(name: str, defaults: dict) -> dict:
    data = load(name)
    embs = np.asarray(data["embeddings"], dtype=np.float32)
    genuine = []
    for i in range(len(embs)):
        rest = np.delete(embs, i, axis=0)
        if len(rest):
            m = rest.mean(0)
            m /= np.linalg.norm(m) + 1e-9
            genuine.append(float(embs[i] @ m))
    imp = data.get("impostor_scores", [])
    result = {"genuine_scores": [round(g, 3) for g in genuine], "impostor_scores_n": len(imp)}
    if len(genuine) >= 3 and len(imp) >= 3:
        g10, i90 = float(np.percentile(genuine, 10)), float(np.percentile(imp, 90))
        accept = float(np.clip((g10 + i90) / 2, 0.35, 0.85))
        reject = float(np.clip(min(i90 + 0.02, accept - 0.05), 0.2, accept))
        data["thresholds"] = {"accept": round(accept, 3), "reject": round(reject, 3), "calibrated_at": _now(),
                              "basis": f"{len(genuine)} genuine (leave-one-out) / {len(imp)} impostor scores"}
        _write(profile_path(name), data)
        result["thresholds"] = data["thresholds"]
        result["status"] = "calibrated"
    else:
        result["status"] = "not enough data"
        result["thresholds"] = {"accept": defaults["accept_threshold"], "reject": defaults["reject_threshold"],
                                "basis": "defaults (need >= 3 enrolment samples and >= 3 scored calls)"}
    return result


@dataclass
class Match:
    speaker: str
    profile: str | None
    similarity: float | None
    decision: str  # match | uncertain | no_match


def identify(speaker_embs: dict[str, np.ndarray | None], defaults: dict) -> tuple[dict[str, Match], dict]:
    """Return per-speaker matches and a summary with the decision."""
    profiles = []
    for meta in list_profiles():
        data = load(meta["name"])
        th = data.get("thresholds") or {}
        profiles.append((meta["name"], mean_embedding(data), th.get("accept", defaults["accept_threshold"]),
                         th.get("reject", defaults["reject_threshold"])))
    out: dict[str, Match] = {}
    if not profiles:
        return {s: Match(s, None, None, "no_match") for s in speaker_embs}, {"status": "no_profiles"}
    table = {}
    for spk, e in speaker_embs.items():
        if e is None:
            out[spk] = Match(spk, None, None, "uncertain")
            continue
        best = max(((n, float(e @ m), a, r) for n, m, a, r in profiles), key=lambda t: t[1])
        n, sim, a, r = best
        dec = "match" if sim >= a else ("no_match" if sim < r else "uncertain")
        out[spk] = Match(spk, n, round(sim, 3), dec)
        table[spk] = {"profile": n, "similarity": round(sim, 3), "accept": a, "reject": r, "decision": dec}
    matched = [m for m in out.values() if m.decision == "match"]
    if len(matched) == 1 and all(m.decision == "no_match" for m in out.values() if m is not matched[0]):
        status = "confident"
    elif len(matched) > 1:
        status = "ambiguous_multiple_matches"
    elif any(m.decision == "uncertain" for m in out.values()) or matched:
        status = "uncertain"
    else:
        status = "no_enrolled_salesperson_detected"
    return out, {"status": status, "scores": table}
