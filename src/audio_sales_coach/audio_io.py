"""Probe and decode local recordings with ffmpeg.

Decoding streams 32-bit float PCM from ffmpeg's stdout straight into memory, so by default no
converted copy of the recording is ever written to disk. If `privacy.keep_converted_audio` is on,
a 16 kHz WAV copy is written next to the report and listed in it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
SUPPORTED_CODECS = {
    "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_u8", "pcm_s16be", "pcm_mulaw", "pcm_alaw",
    "mp3", "aac", "opus", "vorbis", "flac", "alac",
}
SUPPORTED_HINT = "WAV, MP3, M4A (AAC/ALAC), OGG/Opus (WhatsApp voice notes), OGG/Vorbis, FLAC"
MAX_DURATION_S = 3 * 3600


class AudioError(Exception):
    """Unreadable, unsupported or missing input."""


@dataclass
class ProbeInfo:
    path: str
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration_s: float
    bit_rate: int | None
    size_bytes: int
    sha256: str


@dataclass
class Decoded:
    info: ProbeInfo
    sr: int
    channels: np.ndarray  # shape (C, N), float32 in [-1, 1]; C is 1 or 2
    notes: list[str] = field(default_factory=list)

    @property
    def mono(self) -> np.ndarray:
        if self.channels.shape[0] == 1:
            return self.channels[0]
        return self.channels.mean(axis=0).astype(np.float32)

    @property
    def duration(self) -> float:
        return self.channels.shape[1] / self.sr


def require_ffmpeg() -> tuple[str, str]:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff or not fp:
        raise AudioError("ffmpeg/ffprobe not found. Install with: brew install ffmpeg")
    return ff, fp


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: str | Path) -> ProbeInfo:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise AudioError(f"file not found: {p}")
    _, ffprobe = require_ffmpeg()
    r = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(p)],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise AudioError(f"ffprobe could not read {p.name}: {r.stderr.strip()[:300]}")
    meta = json.loads(r.stdout or "{}")
    audio = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise AudioError(f"{p.name} has no audio stream")
    s = audio[0]
    codec = s.get("codec_name", "unknown")
    if codec not in SUPPORTED_CODECS:
        raise AudioError(f"{p.name}: codec '{codec}' is not supported (supported: {SUPPORTED_HINT})")
    fmt = meta.get("format", {})
    dur = float(s.get("duration") or fmt.get("duration") or 0.0)
    br = s.get("bit_rate") or fmt.get("bit_rate")
    return ProbeInfo(
        path=str(p),
        container=fmt.get("format_name", "unknown"),
        codec=codec,
        sample_rate=int(s.get("sample_rate") or 0),
        channels=int(s.get("channels") or 1),
        duration_s=dur,
        bit_rate=int(br) if br else None,
        size_bytes=p.stat().st_size,
        sha256=sha256_file(p),
    )


def decode(path: str | Path, sr: int = SR, keep_channels: bool = True) -> Decoded:
    """Decode to float32 at `sr`. Stereo is kept as two channels (for per-channel speaker turns);
    more than two channels are down-mixed to mono."""
    info = probe(path)
    if info.duration_s and info.duration_s > MAX_DURATION_S:
        raise AudioError(f"recording is {info.duration_s/3600:.1f} h; the limit is {MAX_DURATION_S/3600:.0f} h")
    ffmpeg, _ = require_ffmpeg()
    n_ch = 2 if (keep_channels and info.channels == 2) else 1
    notes = []
    if info.channels > 2:
        notes.append(f"{info.channels}-channel input was down-mixed to mono")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", info.path, "-vn", "-map", "0:a:0",
           "-ac", str(n_ch), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=max(120, int(info.duration_s * 2) + 60))
    if r.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode {Path(info.path).name}: {r.stderr.decode(errors='replace')[:300]}")
    pcm = np.frombuffer(r.stdout, dtype="<f4")
    if pcm.size < sr * 0.2 * n_ch:
        raise AudioError(f"{Path(info.path).name}: less than 0.2 s of audio decoded")
    pcm = pcm[: (pcm.size // n_ch) * n_ch].reshape(-1, n_ch).T.copy()
    np.clip(pcm, -1.0, 1.0, out=pcm)
    if info.codec in ("opus", "mp3", "aac", "vorbis") and info.bit_rate and info.bit_rate < 32000:
        notes.append(f"low-bit-rate {info.codec} ({info.bit_rate // 1000} kbit/s): pitch, breathing and "
                     "laughter cues are less reliable")
    return Decoded(info=info, sr=sr, channels=pcm.astype(np.float32), notes=notes)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    """Write mono or (C, N) float audio as 16-bit PCM WAV."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    data = (np.clip(a.T, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def encode(src_wav: Path, dst: Path, codec: str, bitrate: str | None = None) -> Path:
    """Encode a WAV to another format (used to build compressed test fixtures)."""
    ffmpeg, _ = require_ffmpeg()
    args = {"opus": ["-c:a", "libopus", "-application", "voip"], "mp3": ["-c:a", "libmp3lame"],
            "aac": ["-c:a", "aac"], "pcm": ["-c:a", "pcm_s16le"]}[codec]
    if bitrate:
        args += ["-b:a", bitrate]
    r = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(src_wav), *args, str(dst)],
                       capture_output=True, timeout=300)
    if r.returncode != 0:
        raise AudioError(r.stderr.decode(errors="replace")[:300])
    return dst
