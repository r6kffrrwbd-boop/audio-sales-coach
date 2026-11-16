import json

from audio_sales_coach import cli


def test_help_lists_commands(capsys):
    try:
        cli.main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    for c in ("analyze", "play", "roles", "feedback", "voice", "models", "selftest", "eval", "purge", "ingest"):
        assert c in out


def test_play_print_only(tmp_path, capsys):
    rep = {"recording": {"path": "/x/call.m4a"}, "coaching_notes": [{"id": "note-1", "play": {"from_s": 3.0, "to_s": 9.5}}]}
    (tmp_path / "report.json").write_text(json.dumps(rep))
    assert cli.main(["play", str(tmp_path), "--note", "note-1", "--print-only"]) == 0
    out = capsys.readouterr().out
    assert "-ss 3.00" in out and "-t 6.50" in out and "/x/call.m4a" in out


def test_missing_file_is_input_error(tmp_path, private_home):
    assert cli.main(["analyze", str(tmp_path / "nope.m4a"), "--output", str(tmp_path / "o")]) == 2
