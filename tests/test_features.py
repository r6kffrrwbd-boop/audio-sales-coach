import numpy as np

from audio_sales_coach import features, synth


def test_pitch_tracks_known_f0():
    x = synth.tone_speech(140.0, 3.0, syll_rate=4.0)
    f = features.channel_features(x, 16000)
    med = np.nanmedian(f.f0)
    assert abs(med - 140.0) / 140.0 < 0.04


def test_pitch_glide_is_measured_in_semitones():
    lo = features.channel_features(synth.tone_speech(120.0, 3.0), 16000)
    hi = features.channel_features(synth.tone_speech(120.0 * 2 ** (4 / 12), 3.0), 16000)
    st = 12 * np.log2(np.nanmedian(hi.f0) / np.nanmedian(lo.f0))
    assert 3.5 < st < 4.5


def test_syllable_rate_tracks_modulation_rate():
    slow = features.channel_features(synth.tone_speech(130, 6.0, syll_rate=3.0), 16000).segment_stats(0.5, 5.5)
    fast = features.channel_features(synth.tone_speech(130, 6.0, syll_rate=6.0), 16000).segment_stats(0.5, 5.5)
    assert 2.2 <= slow["syllable_rate_per_s"] <= 3.8
    assert fast["syllable_rate_per_s"] > slow["syllable_rate_per_s"] * 1.5


def test_runs_and_energy_vad_find_pause():
    sr = 16000
    x = np.concatenate([synth.tone_speech(150, 1.5), np.zeros(sr, np.float32), synth.tone_speech(150, 1.5)])
    m = features.energy_vad(features.rms_db(x, sr))
    r = features.runs(m)
    assert len(r) == 2
    gap = (r[1][0] - r[0][1]) * features.HOP_S
    assert 0.7 < gap < 1.3
