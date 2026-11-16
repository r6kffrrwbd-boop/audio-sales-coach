import os
import tempfile

import numpy as np
import pytest

from audio_sales_coach import audio_io, synth
from conftest import needs_ffmpeg


@needs_ffmpeg
@pytest.mark.parametrize("codec,ext,br", [("pcm", "wav", None), ("opus", "ogg", "16k"), ("mp3", "mp3", "64k"),
                                          ("aac", "m4a", "64k")])
def test_decode_supported_formats(tmp_path, codec, ext, br):
    x = synth.tone_speech(150, 2.0)
    audio_io.write_wav(tmp_path / "s.wav", x)
    p = audio_io.encode(tmp_path / "s.wav", tmp_path / f"o.{ext}", codec, br)
    d = audio_io.decode(p)
    assert d.sr == 16000 and abs(d.duration - 2.0) < 0.15
    assert len(d.info.sha256) == 64


@needs_ffmpeg
def test_stereo_channels_are_kept(tmp_path):
    a, b = synth.tone_speech(120, 1.0), np.zeros(16000, np.float32)
    audio_io.write_wav(tmp_path / "st.wav", np.stack([a, b]))
    d = audio_io.decode(tmp_path / "st.wav")
    assert d.channels.shape[0] == 2 and np.abs(d.channels[1]).max() < 1e-3


@needs_ffmpeg
def test_decoding_writes_no_temporary_files(tmp_path):
    audio_io.write_wav(tmp_path / "s.wav", synth.tone_speech(150, 1.0))
    before = set(os.listdir(tempfile.gettempdir()))
    audio_io.decode(tmp_path / "s.wav")
    assert set(os.listdir(tempfile.gettempdir())) - before == set()


@needs_ffmpeg
def test_rejects_non_audio(tmp_path):
    (tmp_path / "x.m4a").write_text("not audio")
    with pytest.raises(audio_io.AudioError):
        audio_io.decode(tmp_path / "x.m4a")
