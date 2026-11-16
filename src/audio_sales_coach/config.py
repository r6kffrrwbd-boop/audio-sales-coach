"""Locations and settings.

Everything personal (voice profiles, feedback, model cache, config) lives under the data directory,
never inside the repository. Override with environment variables:

  ASC_DATA_DIR   default ~/.local/share/audio-sales-coach
  ASC_CONFIG     default ~/.config/audio-sales-coach/config.json
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "detector": "auto",  # auto | yamnet | ced  (auto = yamnet when its runtime works, else ced)
    "event_thresholds": {  # calibrated on ESC-50 fold 1, see docs/evaluation.md
        "laughter": None,  # None = use the detector's calibrated default
        "breathing": None,
    },
    "vad": {"threshold": 0.5, "min_speech_s": 0.25, "min_silence_s": 0.25},
    "analysis": {
        "pause_min_s": 0.6,
        "long_response_gap_s": 2.5,
        "monologue_s": 45.0,
        "talk_share_flag": 0.70,
        "interruption_min_overlap_s": 0.3,
        "pace_change_ratio": 0.30,
        "pitch_change_semitones": 3.0,
        "energy_change_db": 6.0,
    },
    "speakers": {
        "embedding_model": "campplus",
        "cluster_threshold": 0.55,  # cosine distance cut for agglomerative clustering
        "channel_map": {},  # e.g. {"0": "salesperson", "1": "client"} for known dual-channel exports
    },
    "voice_id": {
        # Cosine similarity between an enrolled profile and a speaker in a call.
        # >= accept -> labelled as that salesperson; < reject -> not them; between -> ask the user.
        "accept_threshold": 0.60,
        "reject_threshold": 0.40,
    },
    "transcription": {
        "enabled": "auto",  # auto = when faster-whisper is installed
        "model": "small",
        "language": None,  # None = detect; "ar" / "en" to force
        "compute_type": "int8",
        "min_avg_logprob": -1.0,  # below this a segment is marked unreliable
        "max_no_speech_prob": 0.6,
    },
    "council": {
        "enabled": "auto",  # auto = when the evidence-council command is installed
        "command": "evidence-council",
        "providers": ["none"],  # ["none"] = local fact/citation checks only; ["ollama"] adds a local model
        "include_quotes": False,  # never send transcript words unless turned on
        "timeout_s": 900,
    },
    "privacy": {
        "keep_converted_audio": False,
        "redact_terms": [],  # extra names/companies to scrub from anything sent to the council
        "store_client_profiles": False,  # client voice profiles are never stored; kept for clarity
    },
    "retention": {"reports_days": 30, "feedback_days": 365},
}


def data_dir() -> Path:
    return Path(os.environ.get("ASC_DATA_DIR") or Path.home() / ".local/share/audio-sales-coach").expanduser()


def config_path() -> Path:
    return Path(os.environ.get("ASC_CONFIG") or Path.home() / ".config/audio-sales-coach/config.json").expanduser()


def models_dir() -> Path:
    return data_dir() / "models"


def profiles_dir() -> Path:
    return data_dir() / "voice-profiles"


def feedback_dir() -> Path:
    return data_dir() / "feedback"


def private_mkdir(path: Path) -> Path:
    """Create a directory readable only by the current user."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or config_path()
    if p.exists():
        with open(p, encoding="utf-8") as f:
            user = json.load(f)
        if not isinstance(user, dict):
            raise ValueError(f"{p}: config must be a JSON object")
        return _merge(DEFAULTS, user)
    return copy.deepcopy(DEFAULTS)
