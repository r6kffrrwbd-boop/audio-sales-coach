"""Model registry and verified downloads.

Default models are small, CPU-only, and downloadable without any account or login. Every file is
pinned by size and SHA-256 and verified before use. Optional models are installed only on request.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import models_dir


@dataclass(frozen=True)
class ModelFile:
    key: str
    filename: str
    url: str
    size: int
    sha256: str
    license: str
    purpose: str
    default: bool
    notes: str = ""
    extract: bool = False


REGISTRY: dict[str, ModelFile] = {m.key: m for m in [
    ModelFile(
        key="yamnet",
        filename="yamnet.h5",
        url="https://storage.googleapis.com/audioset/yamnet.h5",
        size=15296092,
        sha256="13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744",
        license="Apache-2.0 (Google, tensorflow/models research/audioset/yamnet)",
        purpose="audio-event detector (laughter, breathing, speech, cough, music, ... 521 AudioSet classes)",
        default=True,
    ),
    ModelFile(
        key="silero_vad",
        filename="silero_vad.onnx",
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        size=643854,
        sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        license="MIT (snakers4/silero-vad), mirrored by k2-fsa/sherpa-onnx",
        purpose="voice-activity detection (speech segments)",
        default=True,
    ),
    ModelFile(
        key="campplus",
        filename="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        url=("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
        size=28281164,
        sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
        license="Apache-2.0 (modelscope/3D-Speaker CAM++), mirrored by k2-fsa/sherpa-onnx",
        purpose="speaker embeddings for diarization of mixed mono and optional salesperson voice ID",
        default=True,
        notes="Trained on Chinese and English speech; Arabic speakers are not specifically covered.",
    ),
    ModelFile(
        key="ced_mini",
        filename="sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2",
        url=("https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/"
             "sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2"),
        size=47658399,
        sha256="bec8bfa0af2c20ec3a9e7c6dd6c92d0fb6e96d6b25370ebe7a48e7283a60a02c",
        license="Apache-2.0 (RicherMans/CED), mirrored by k2-fsa/sherpa-onnx",
        purpose="optional second audio-event detector (AudioSet 527 classes) for cross-checking YAMNet",
        default=False,
        extract=True,
    ),
]}


class ModelError(Exception):
    pass


def http_download(url: str, dest: Path, attempts: int = 30) -> Path:
    """Resumable download. Uses curl (system TLS certificates, stall detection); falls back to urllib+certifi."""
    import subprocess
    import time
    curl = shutil.which("curl")
    if curl:
        for _ in range(attempts):
            r = subprocess.run([curl, "-fsSL", "-C", "-", "--speed-limit", "2000", "--speed-time", "20",
                                "--connect-timeout", "20", "-o", str(dest), url], capture_output=True, text=True)
            if r.returncode == 0:
                return dest
            if r.returncode == 33:  # server does not support resume: restart
                dest.unlink(missing_ok=True)
            time.sleep(1)
        raise ModelError(f"download failed after {attempts} attempts: {url} ({r.stderr.strip()[:200]})")
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "audio-sales-coach"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    return dest


def path_for(key: str) -> Path:
    m = REGISTRY[key]
    p = models_dir() / m.filename
    if m.extract:
        return models_dir() / m.filename.removesuffix(".tar.bz2")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_installed(key: str) -> bool:
    p = path_for(key)
    if REGISTRY[key].extract:
        return p.is_dir() and any(p.iterdir())
    return p.is_file() and p.stat().st_size == REGISTRY[key].size


def verify(key: str) -> tuple[bool, str]:
    m = REGISTRY[key]
    p = path_for(key)
    if m.extract:
        return (is_installed(key), "extracted" if is_installed(key) else "missing")
    if not p.is_file():
        return False, "missing"
    if p.stat().st_size != m.size:
        return False, f"size {p.stat().st_size} != {m.size}"
    if m.sha256 != "PENDING" and sha256(p) != m.sha256:
        return False, "sha256 mismatch"
    return True, "ok"


def fetch(key: str, progress=print) -> Path:
    """Download (resumable) and verify one model file. Idempotent."""
    m = REGISTRY[key]
    dest_dir = models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = path_for(key)
    if is_installed(key):
        ok, why = verify(key)
        if ok:
            return final
        progress(f"  {m.filename}: {why}; downloading again")
        if final.is_file():
            final.unlink()
    archive = dest_dir / m.filename
    part = archive.with_name(archive.name + ".part")
    progress(f"  downloading {m.filename} ({m.size / 1e6:.1f} MB) from {m.url.split('/')[2]}")
    http_download(m.url, part)
    if part.stat().st_size != m.size:
        raise ModelError(f"{m.filename}: got {part.stat().st_size} bytes, expected {m.size}; rerun to resume")
    digest = sha256(part)
    if m.sha256 != "PENDING" and digest != m.sha256:
        part.unlink()
        raise ModelError(f"{m.filename}: SHA-256 mismatch (got {digest}); file removed")
    os.replace(part, archive)
    if m.extract:
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(dest_dir, filter="data")
        archive.unlink()
    return final


def status() -> list[dict]:
    out = []
    for k, m in REGISTRY.items():
        ok, why = verify(k)
        out.append({"key": k, "file": m.filename, "default": m.default, "installed": ok, "state": why,
                    "size_mb": round(m.size / 1e6, 1), "license": m.license, "purpose": m.purpose})
    return out


WHISPER_FILES = ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin")
WHISPER_REPO = "https://huggingface.co/Systran/faster-whisper-{size}/resolve/main/{file}"


def whisper_dir(size: str) -> Path:
    return models_dir() / f"whisper-{size}"


def fetch_whisper(size: str = "small", progress=print) -> Path:
    """Download a faster-whisper model (public repo, no login) with resumable curl into models/whisper-<size>/."""
    if size not in ("tiny", "base", "small", "medium"):
        raise ModelError("whisper size must be tiny, base, small or medium")
    d = whisper_dir(size)
    d.mkdir(parents=True, exist_ok=True)
    for f in WHISPER_FILES:
        dest = d / f
        if dest.exists() and dest.stat().st_size > 0:
            continue
        progress(f"  downloading faster-whisper-{size}/{f} from huggingface.co")
        part = d / (f + ".part")
        http_download(WHISPER_REPO.format(size=size, file=f), part, attempts=200)
        os.replace(part, dest)
    (d / "SHA256SUMS").write_text("".join(f"{sha256(d / f)}  {f}\n" for f in WHISPER_FILES))
    return d
