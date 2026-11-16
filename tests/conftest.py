import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_sales_coach import models  # noqa: E402


def models_ready():
    return all(models.is_installed(k) for k, m in models.REGISTRY.items() if m.default)


needs_models = pytest.mark.skipif(not models_ready(), reason="default models not installed (./install.sh)")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
needs_tts = pytest.mark.skipif(not shutil.which("say"), reason="macOS `say` not available")


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    """Profiles/feedback go to a temp dir; models stay where they are installed."""
    real_models = models.models_dir()
    monkeypatch.setenv("ASC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ASC_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setattr(models, "models_dir", lambda: real_models)
    return tmp_path
