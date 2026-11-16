import json
import os
import shutil
import stat

import pytest

from audio_sales_coach import coaching, council
from audio_sales_coach.config import DEFAULTS
from helpers import call_fixture

ROLES = {"S1": "salesperson", "S2": "client_in_this_call"}


def setup():
    turns, obs = call_fixture()
    notes = coaching.build_notes(obs, turns, ROLES, "/Users/someone/calls/call.m4a", "/tmp/r", True, "separate_channels")
    report = {"report_id": "rpt-abc123def456"}
    cfg = json.loads(json.dumps(DEFAULTS))
    return report, notes, obs, turns, cfg


def test_task_is_minimal_and_private():
    report, notes, obs, turns, cfg = setup()
    task = council.build_task(report, notes, obs, ROLES, cfg, {t.id: t for t in turns})
    s = json.dumps(task)
    assert "/Users/" not in s and "file://" not in s and "call.m4a" not in s
    assert "twelve hundred" not in s and "cancellation terms?" not in s  # no transcript words by default
    rec = [e for e in task["evidence"] if e["kind"] == "recording"][0]
    assert "uri" not in rec and "content" not in rec
    assert task["providers"]["requested"] == ["none"]
    assert task["privacy"]["level"] == "private"
    assert council.payload_problems(task) == []
    ids = {e["id"] for e in task["evidence"]}
    for c in task["candidates"]:
        for cl in c["claims"]:
            assert set(cl["evidence_ids"]) <= ids


def test_quotes_only_when_enabled():
    report, notes, obs, turns, cfg = setup()
    cfg["council"]["include_quotes"] = True
    task = council.build_task(report, notes, obs, ROLES, cfg, {t.id: t for t in turns})
    assert "twelve hundred" in json.dumps(task)


def test_privacy_guard_blocks_phone_email_path():
    base = {"evidence": [{"id": "o", "kind": "observation", "content": "call me on +961 3 123 456"}], "candidates": []}
    assert "long digit sequence (possible phone number)" in council.payload_problems(base)
    base["evidence"][0]["content"] = "mail x@y.com"
    assert "e-mail address" in council.payload_problems(base)
    base["evidence"][0]["content"] = "see /Users/a/b.wav"
    assert "file path" in council.payload_problems(base)
    assert council.scrub("ring +961 3 123 456 or a@b.co, ask Rami", ["Rami"]) == "ring [number] or [email], ask [name]"


def test_absent_council_is_skipped(monkeypatch):
    report, notes, obs, turns, cfg = setup()
    cfg["council"]["command"] = "definitely-not-installed-xyz"
    r = council.review(council.build_task(report, notes, obs, ROLES, cfg), __import__("pathlib").Path("/tmp/x"), cfg)
    assert r["status"] == "skipped" and "not installed" in r["reason"]


def test_council_exit_code_is_reported(tmp_path):
    fake = tmp_path / "fake-council"
    fake.write_text("#!/bin/sh\necho 'not enough usable providers' >&2\nexit 3\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    report, notes, obs, turns, cfg = setup()
    cfg["council"]["command"] = str(fake)
    r = council.review(council.build_task(report, notes, obs, ROLES, cfg), tmp_path / "c", cfg)
    assert r["status"] == "not_reviewed" and r["exit_code"] == 3


@pytest.mark.skipif(not shutil.which("evidence-council"), reason="evidence-council not installed")
def test_real_council_reviews_and_flags_a_wrong_attribution(tmp_path):
    report, notes, obs, turns, cfg = setup()
    task = council.build_task(report, notes, obs, ROLES, cfg)
    # tamper with one note: claim the interruption came from the client
    for c in task["candidates"]:
        for cl in c["claims"]:
            if cl["key"] == "obs-1.speaker_role":
                cl["value"] = "client_in_this_call"
    r = council.review(task, tmp_path / "council", cfg)
    assert r["status"] == "reviewed", r
    assert r["providers_used"] == []
    flagged = [k for k, v in r["per_note"].items() if v.get("contradicted")]
    assert flagged, r["per_note"]
    assert (tmp_path / "council" / "task.json").exists()
