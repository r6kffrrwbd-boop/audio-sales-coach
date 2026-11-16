import numpy as np

from audio_sales_coach import yamnet_np


def test_mel_matrix_matches_tf_conventions():
    m = yamnet_np.mel_matrix(257)
    assert m.shape == (257, 64)
    assert np.all(m[0] == 0)  # DC bin zeroed like tf.signal.linear_to_mel_weight_matrix
    assert np.all(m >= 0) and m.max() <= 1.0


def test_patch_count_follows_official_padding():
    for seconds, expected in ((0.5, 1), (0.975, 1), (1.0, 2), (10.0, 20)):
        _, p = yamnet_np.log_mel_patches(np.zeros(int(16000 * seconds), np.float32))
        assert p.shape[1:] == (96, 64)
        assert p.shape[0] == expected, (seconds, p.shape)


def test_same_padding():
    x = np.zeros((1, 96, 64, 1), np.float32)
    assert yamnet_np._same_pad(x, 2).shape == (1, 97, 65, 1)
    assert yamnet_np._same_pad(x, 1).shape == (1, 98, 66, 1)
