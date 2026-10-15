import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/: tests run without installing

from ber.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Default config rooted in a temp dir (artifacts/output land there)."""
    (tmp_path / "dataset").mkdir()
    return load_config(overrides=[f"paths.root={tmp_path}"], run_id="test")
