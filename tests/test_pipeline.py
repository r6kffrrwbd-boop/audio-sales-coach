"""End-to-end on synthetic TTS calls (needs ./install.sh models and macOS `say`)."""
import json

import pytest

from audio_sales_coach import audio_io, evaluation, synth
from audio_sales_coach.config import load_config
from audio_sales_coach.pipeline import analyze, rebuild_with_roles
from conftest import needs_models, needs_tts

pytestmark = [needs_models, needs_tts]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    d = tmp_path_factory.mktemp("call")
    from audio_sales_coach.selftest import _short_call
    truth = synth.render_scenario(_short_call(), d, use_esc50=False)
    return d, truth


def cfg_off():
    c = load_config()
    c["council"]["providers"] = ["none"]
    return c


def test_separate_channels_call(rendered, tmp_path, private_home):
    d, truth = rendered
    rep = analyze(str(d / "stereo.wav"), str(tmp_path / "r"), cfg_off(), mode="call", transcription="off",
                  council_mode="off", log=lambda *a: None)
    assert rep["layout"]["result"] == "separate_channels"
    m = evaluation.speaker_metrics(rep, truth)
    assert m["attribution_accuracy"] > 0.95
    mm = evaluation.moment_metrics(rep, truth)
    assert mm["interruptions"]["found"] == 1 and mm["response_gaps"]["found"] == 1
    assert rep["role_assignment"]["status"] == "needs_user" and rep["role_assignment"]["excerpts"]
    assert {"speaker_S1", "speaker_S2", "combined"} <= set(rep["views"])
    # role correction re-renders notes without audio
    rep2 = rebuild_with_roles(str(tmp_path / "r"), {"S1": "salesperson", "S2": "client_in_this_call"}, cfg_off(), "off")
    assert {"salesperson", "client", "combined"} <= set(rep2["views"])
    assert any("salesperson (S1)" in n["interaction"] for n in rep2["coaching_notes"])
    saved = json.loads((tmp_path / "r" / "report.json").read_text())
    assert saved["role_assignment"]["source"] == "user"


def test_mixed_mono_call_marks_uncertainty(rendered, tmp_path, private_home):
    d, truth = rendered
    rep = analyze(str(d / "mono.wav"), str(tmp_path / "m"), cfg_off(), mode="call", transcription="off",
                  council_mode="off", log=lambda *a: None)
    assert rep["layout"]["result"] == "mixed_mono"
    assert len(rep["speakers"]) == 2
    assert evaluation.speaker_metrics(rep, truth)["attribution_accuracy"] > 0.85
    assert all(not o["attribution_reliable"] for o in rep["observations"] if o["type"] == "possible_interruption")
    assert "mixed recording" in rep["reliability"]["speaker_attribution"]["level"]


def test_voice_message_never_invents_second_person(tmp_path, private_home):
    truth = synth.render_scenario(synth.voice_note(), tmp_path / "vn", use_esc50=False)
    ogg = audio_io.encode(tmp_path / "vn" / "mono.wav", tmp_path / "vn" / "n.ogg", "opus", "16k")
    rep = analyze(str(ogg), str(tmp_path / "r"), cfg_off(), mode="auto", transcription="off", council_mode="off",
                  log=lambda *a: None)
    assert len(rep["speakers"]) == 1 and rep["speakers"][0]["role"] in ("speaker", "salesperson")
    assert "client" not in rep["views"] and "speaker" in rep["views"]
    assert truth["single_speaker"]
