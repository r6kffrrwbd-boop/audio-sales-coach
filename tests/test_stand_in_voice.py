"""The whole call path with the stand-in voice, so it is covered on machines without macOS `say` (Linux)."""
import numpy as np

from audio_sales_coach import synth
from audio_sales_coach.config import load_config
from audio_sales_coach.pipeline import analyze
from audio_sales_coach.selftest import _short_call
from conftest import needs_ffmpeg, needs_models


def test_stand_in_voice_is_repeatable_and_differs_by_voice():
    line = "Hello there, thanks for your time."
    a, b, c = (synth.stand_in_voice(line, v) for v in ("Daniel", "Daniel", "Samantha"))
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert abs(a.size / synth.SR - 6 * 60 / 180) < 0.01  # six words at 180 words a minute
    assert a.size == c.size and not np.array_equal(a, c)


@needs_models
@needs_ffmpeg
def test_call_without_tts_finds_the_interruption_and_the_gap(tmp_path, private_home, monkeypatch):
    monkeypatch.setattr(synth, "have_tts", lambda: False)
    truth = synth.render_scenario(_short_call(), tmp_path / "call", use_esc50=False)
    assert truth["voice_source"].startswith("stand-in")
    cfg = load_config()
    cfg["council"]["providers"] = ["none"]
    rep = analyze(str(tmp_path / "call" / "stereo.wav"), str(tmp_path / "r"), cfg, mode="call", transcription="off",
                  council_mode="off", log=lambda *a: None)
    assert rep["layout"]["result"] == "separate_channels" and len(rep["speakers"]) == 2
    first = {}
    for o in rep["observations"]:
        first.setdefault(o["type"], o)
    assert abs(first["interruption"]["start"] - truth["interruptions"][0]["at"]) < 0.5
    assert abs(first["response_gap"]["start"] - truth["response_gaps"][0]["start"]) < 0.5


@needs_models
@needs_ffmpeg
def test_voice_note_without_tts_stays_one_speaker(tmp_path, private_home, monkeypatch):
    monkeypatch.setattr(synth, "have_tts", lambda: False)
    synth.render_scenario(synth.short_voice_note(), tmp_path / "vn", use_esc50=False)
    cfg = load_config()
    rep = analyze(str(tmp_path / "vn" / "mono.wav"), str(tmp_path / "r"), cfg, mode="auto", transcription="off",
                  council_mode="off", log=lambda *a: None)
    assert len(rep["speakers"]) == 1 and "client" not in rep["views"]
