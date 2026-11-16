import os
import stat

import numpy as np
import pytest

from audio_sales_coach import voiceid
from audio_sales_coach.config import DEFAULTS

TH = DEFAULTS["voice_id"]


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_enroll_requires_consent(private_home):
    with pytest.raises(voiceid.ProfileError):
        voiceid.enroll("rep-a", unit(np.ones(8)), "m", "", "test")


def test_profile_is_private_and_deletable(private_home):
    voiceid.enroll("rep-a", unit(np.arange(1, 9)), "m", "I agree to store my voice profile", "test")
    p = voiceid.profile_path("rep-a")
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(p.parent).st_mode) == 0o700
    assert voiceid.list_profiles()[0]["name"] == "rep-a"
    assert voiceid.delete("rep-a") and not p.exists()


def test_identify_confident_uncertain_and_unknown(private_home):
    rng = np.random.default_rng(0)
    rep = unit(rng.normal(size=192))
    voiceid.enroll("rep-a", rep, "m", "I agree to store my voice profile", "test")
    other = unit(rng.normal(size=192))
    near = unit(rep + 0.03 * rng.normal(size=192))  # cosine ~0.9
    _, s = voiceid.identify({"S1": near, "S2": other}, TH)
    assert s["status"] == "confident" and s["scores"]["S1"]["decision"] == "match"
    mid = unit(0.45 * rep + np.sqrt(1 - 0.45 ** 2) * unit(other - (other @ rep) * rep))
    _, s = voiceid.identify({"S1": mid, "S2": other}, TH)
    assert s["status"] in ("uncertain",) and s["scores"]["S1"]["decision"] == "uncertain"
    _, s = voiceid.identify({"S1": other, "S2": unit(rng.normal(size=192))}, TH)
    assert s["status"] == "no_enrolled_salesperson_detected"


def test_no_profiles(private_home):
    _, s = voiceid.identify({"S1": unit(np.ones(4))}, TH)
    assert s["status"] == "no_profiles"


def test_calibration_needs_data_then_sets_thresholds(private_home):
    rng = np.random.default_rng(1)
    base = unit(rng.normal(size=64))
    for i in range(4):
        voiceid.enroll("rep-b", unit(base + 0.3 * rng.normal(size=64)), "m", "I agree to store my voice profile", "test")
    assert voiceid.calibrate("rep-b", TH)["status"] == "not enough data"
    voiceid.record_scores("rep-b", impostor=[0.05, 0.1, 0.12, 0.2])
    r = voiceid.calibrate("rep-b", TH)
    assert r["status"] == "calibrated"
    assert r["thresholds"]["reject"] < r["thresholds"]["accept"]
