import io
import json

import pytest

from audio_sales_coach import feedback, retention
from audio_sales_coach.adapters import recording_folder, whatsapp_cloud

REPORT = {"report_id": "rpt-000000000001",
          "coaching_notes": [{"id": "note-1", "cue": {"type": "interruption", "detector": "per-channel VAD"}}],
          "observations": [{"id": "obs-1", "type": "laughter", "detector": "yamnet"}]}


def test_feedback_is_stored_separately_and_summarised(private_home):
    feedback.add(REPORT, "note-1", "useful", "good catch")
    feedback.add(REPORT, "obs-1", "incorrect", "that was a cough")
    feedback.add(REPORT, None, None, "wrong side", speaker_correction={"S1": "client_in_this_call"})
    s = feedback.summary()
    assert s["by_cue_type"]["interruption"]["useful"] == 1
    assert s["by_cue_type"]["laughter"]["incorrect"] == 1
    assert s["speaker_corrections"] == 1
    assert "does not retrain" in s["note"]
    files = list((private_home / "data" / "feedback").glob("*.jsonl"))
    assert len(files) == 1
    with pytest.raises(ValueError):
        feedback.add(REPORT, "note-9", "useful")
    with pytest.raises(ValueError):
        feedback.add(REPORT, "note-1", "great")


def test_retention_dry_run_then_delete(private_home, tmp_path):
    feedback.add(REPORT, "note-1", "useful")
    r = retention.purge(None, 30, 0, dry_run=True)
    assert r["feedback"] and r["dry_run"]
    retention.purge(None, 30, -1, dry_run=False)
    assert not list((private_home / "data" / "feedback").glob("*.jsonl"))


PAYLOAD = {"object": "whatsapp_business_account", "entry": [{"id": "1", "changes": [{"field": "messages", "value": {
    "messaging_product": "whatsapp", "contacts": [{"wa_id": "9613000000", "profile": {"name": "Client"}}],
    "messages": [
        {"from": "9613000000", "id": "wamid.A", "timestamp": "1731760000", "type": "audio",
         "audio": {"id": "123456789", "mime_type": "audio/ogg; codecs=opus", "voice": True, "sha256": "abc"}},
        {"from": "9613000000", "id": "wamid.B", "timestamp": "1731760001", "type": "text", "text": {"body": "hi"}}]}}]}]}


def test_whatsapp_parse_keeps_no_phone_numbers():
    items = whatsapp_cloud.parse_webhook(PAYLOAD)
    assert len(items) == 1 and items[0].is_voice_note and items[0].media_id == "123456789"
    assert "9613000000" not in json.dumps(items[0].__dict__)


def test_whatsapp_download_with_mocked_graph_api(tmp_path):
    calls = []

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=60):
        calls.append((req.full_url, req.get_header("Authorization")))
        if "graph.facebook.com" in req.full_url:
            return R(json.dumps({"url": "https://lookaside.example/media/1"}).encode())
        return R(b"OggS fake")

    p = whatsapp_cloud.download(whatsapp_cloud.parse_webhook(PAYLOAD)[0], tmp_path / "dl", token="TEST", opener=opener)
    assert p.name == "wa-123456789.ogg" and p.read_bytes() == b"OggS fake"
    assert all(h == "Bearer TEST" for _, h in calls) and len(calls) == 2
    with pytest.raises(RuntimeError):
        whatsapp_cloud.download(whatsapp_cloud.parse_webhook(PAYLOAD)[0], tmp_path, token=None, opener=opener) \
            if not __import__("os").environ.get("WHATSAPP_ACCESS_TOKEN") else (_ for _ in ()).throw(RuntimeError())


def test_recording_folder_requires_consent_sidecar(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"x")
    (tmp_path / "b.m4a").write_bytes(b"x")
    (tmp_path / "b.consent.json").write_text(json.dumps({"consent": "notified", "channel_map": {"0": "salesperson"}}))
    (tmp_path / "c.wav").write_bytes(b"x")
    (tmp_path / "c.consent.json").write_text(json.dumps({"consent": "no"}))
    ready, skipped = recording_folder.scan(tmp_path)
    assert [r.path.name for r in ready] == ["b.m4a"]
    assert len(skipped) == 2
