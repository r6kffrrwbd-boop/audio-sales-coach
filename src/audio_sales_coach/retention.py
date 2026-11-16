"""Configurable retention: delete old reports and feedback.

Recordings themselves are never copied by this tool, so their retention is up to the operator's own system.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from .config import feedback_dir


def purge(reports_root: Path | None, reports_days: int, feedback_days: int, dry_run: bool = True) -> dict:
    now = time.time()
    removed = {"reports": [], "feedback": []}
    if reports_root and reports_root.exists():
        for rj in reports_root.glob("*/report.json"):
            if now - rj.stat().st_mtime > reports_days * 86400:
                removed["reports"].append(str(rj.parent))
                if not dry_run:
                    shutil.rmtree(rj.parent)
    fd = feedback_dir()
    if fd.exists():
        for f in fd.glob("*.jsonl"):
            if now - f.stat().st_mtime > feedback_days * 86400:
                removed["feedback"].append(str(f))
                if not dry_run:
                    f.unlink()
    removed["dry_run"] = dry_run
    return removed
