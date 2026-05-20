from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cli import daily_workflow  # noqa: E402


def test_config_no_longer_exposes_old_p1_engine_switch() -> None:
    config = daily_workflow.load_config()
    assert "p1_engine" not in (config.get("fundamental_screening") or {})


def test_old_phase1_engine_files_are_removed() -> None:
    assert not (SCRIPTS / "fundamental_legacy.py").exists()
    assert not (SCRIPTS / "fundamental_regime.py").exists()
    assert not (SCRIPTS / "compare_p1_engines.py").exists()
