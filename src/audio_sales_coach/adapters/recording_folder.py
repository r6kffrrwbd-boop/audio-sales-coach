"""Call-recording export adapter (a folder the operator's own call system exports into).

Many business phone systems (PBX, contact-centre or CRM dialers) can export recordings of calls they recorded with
a played or written notice. Point this adapter at that export folder. A recording is picked up only when a sidecar
`<name>.consent.json` exists next to it, e.g.:

    {"notice": "recording announcement played", "consent": "all-parties-consented",
     "channel_map": {"0": "salesperson", "1": "client"}}

Nothing is captured from live calls; this only reads files your system already exported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}
CONSENT_VALUES = {"all-parties-consented", "notified"}


@dataclass
class PendingRecording:
    path: Path
    consent: str
    channel_map: dict


def scan(folder: Path) -> tuple[list[PendingRecording], list[str]]:
    ready, skipped = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        side = p.with_name(p.stem + ".consent.json")
        if not side.exists():
            skipped.append(f"{p.name}: no consent sidecar")
            continue
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(f"{p.name}: unreadable consent sidecar")
            continue
        if meta.get("consent") not in CONSENT_VALUES:
            skipped.append(f"{p.name}: consent must be one of {sorted(CONSENT_VALUES)}")
            continue
        ready.append(PendingRecording(p, meta["consent"], meta.get("channel_map") or {}))
    return ready, skipped
