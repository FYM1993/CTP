from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_cases import BUILTIN_CASES, get_case  # noqa: E402


def test_builtin_cases_include_baselines():
    assert "lh0_long" in BUILTIN_CASES
    assert "ps0_short" in BUILTIN_CASES


def test_builtin_cases_have_contract_values():
    lh0 = BUILTIN_CASES["lh0_long"]
    ps0 = BUILTIN_CASES["ps0_short"]

    assert lh0.symbol == "LH0"
    assert lh0.direction == "long"
    assert ps0.symbol == "PS0"
    assert ps0.direction == "short"
    assert lh0.start_dt == date(2025, 1, 1)
    assert lh0.end_dt == date(2025, 12, 31)
    assert ps0.start_dt == date(2025, 1, 1)
    assert ps0.end_dt == date(2025, 12, 31)
    assert isinstance(lh0.start_dt, date)
    assert isinstance(lh0.end_dt, date)
    assert isinstance(ps0.start_dt, date)
    assert isinstance(ps0.end_dt, date)


def test_get_case_returns_registered_case():
    case = get_case("lh0_long")

    assert case is BUILTIN_CASES["lh0_long"]
