"""WhatsApp Business Platform (Cloud API) voice-message adapter.

For businesses that receive customer voice messages on their own WhatsApp Business number through Meta's official
Cloud API webhooks. The operator runs their own webhook endpoint, saves the JSON payload it receives, and passes it
here. This module:
  * parses `messages[].type == "audio"` items (voice notes have `audio.voice == true`) from the payload;
  * downloads the media with the operator's own access token: GET /{media-id} -> short-lived URL -> GET URL
    with `Authorization: Bearer <token>` (token from WHATSAPP_ACCESS_TOKEN, never stored or logged).
It never scrapes WhatsApp Web, never uses personal WhatsApp accounts, and never contacts anyone.

Status: implemented against Meta's documented Cloud API shapes and tested with mocked HTTP only. It has not been run
against a live WhatsApp Business account.
Before use: tell customers (e.g. in your WhatsApp business profile / first reply and privacy notice) that voice
messages may be analysed for staff coaching, follow Meta's Business Messaging policies and local law.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GRAPH = "https://graph.facebook.com"
API_VERSION = os.environ.get("WHATSAPP_GRAPH_VERSION", "v21.0")
_ID = re.compile(r"^[0-9]{1,40}$")


@dataclass
class VoiceItem:
    message_id: str
    media_id: str
    mime_type: str
    is_voice_note: bool
    timestamp: str | None
    sha256: str | None
    # the sender's phone number is deliberately not copied out of the payload


def parse_webhook(payload: dict) -> list[VoiceItem]:
    items = []
    for entry in payload.get("entry", []):
        for ch in entry.get("changes", []):
            val = ch.get("value", {})
            for m in val.get("messages", []) or []:
                if m.get("type") != "audio" or "audio" not in m:
                    continue
                a = m["audio"]
                if not _ID.match(str(a.get("id", ""))):
                    continue
                items.append(VoiceItem(message_id=str(m.get("id", "")), media_id=str(a["id"]),
                                       mime_type=a.get("mime_type", ""), is_voice_note=bool(a.get("voice", False)),
                                       timestamp=m.get("timestamp"), sha256=a.get("sha256")))
    return items


def _get(url: str, token: str, opener=urllib.request.urlopen):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "User-Agent": "audio-sales-coach-adapter"})
    return opener(req, timeout=60)


def download(item: VoiceItem, dest_dir: Path, token: str | None = None, opener=urllib.request.urlopen) -> Path:
    token = token or os.environ.get("WHATSAPP_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("set WHATSAPP_ACCESS_TOKEN (a token for your own WhatsApp Business app)")
    if not _ID.match(item.media_id):
        raise ValueError("invalid media id")
    with _get(f"{GRAPH}/{API_VERSION}/{item.media_id}", token, opener) as r:
        meta = json.loads(r.read().decode())
    url = meta.get("url", "")
    if not url.startswith("https://"):
        raise ValueError("media URL must be https")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(dest_dir, 0o700)
    except OSError:
        pass
    ext = {"audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/aac": ".aac", "audio/amr": ".amr"}
    out = dest_dir / f"wa-{item.media_id}{ext.get(item.mime_type.split(';')[0].strip(), '.bin')}"
    with _get(url, token, opener) as r:
        data = r.read()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return out
