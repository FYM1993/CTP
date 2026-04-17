from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_cases import BACKTEST_CASES, get_case  # noqa: E402


def test_registry_includes_baseline_cases():
    assert "lh0_long" in BACKTEST_CASES
    assert "ps0_short" in BACKTEST_CASES


def test_get_case_returns_registered_case():
    case = get_case("lh0_long")

    assert case.case_id == "lh0_long"
    assert case.symbol == "LH0"
    assert case.direction == "long"
    assert case.note == "baseline long case"
