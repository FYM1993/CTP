from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.cases import BUILTIN_CASES, get_case  # noqa: E402


def test_builtin_cases_include_baselines():
    assert "lh0_reversal_long" in BUILTIN_CASES
    assert "lh0_trend_short" in BUILTIN_CASES
    assert "ps0_reversal_short" in BUILTIN_CASES
    assert "ps0_trend_long" in BUILTIN_CASES


def test_builtin_cases_include_event_replay_cases() -> None:
    assert "lh0_reversal_long_jul2025" in BUILTIN_CASES
    assert "ps0_reversal_long_jul2025" in BUILTIN_CASES
    assert "jm0_reversal_long_jul2025" in BUILTIN_CASES
    assert "ao0_reversal_long_jul2024" in BUILTIN_CASES
    assert "lh0_mixed_long_jul2025" in BUILTIN_CASES
    assert "ps0_mixed_long_jul2025" in BUILTIN_CASES
    assert "jm0_mixed_long_jul2025" in BUILTIN_CASES
    assert "ao0_mixed_long_jul2024" in BUILTIN_CASES


def test_builtin_cases_have_contract_values():
    lh0_reversal = BUILTIN_CASES["lh0_reversal_long"]
    lh0_trend = BUILTIN_CASES["lh0_trend_short"]
    ps0_reversal = BUILTIN_CASES["ps0_reversal_short"]
    ps0_trend = BUILTIN_CASES["ps0_trend_long"]

    assert lh0_reversal.symbol == "LH0"
    assert lh0_reversal.direction == "long"
    assert lh0_trend.symbol == "LH0"
    assert lh0_trend.direction == "short"
    assert ps0_reversal.symbol == "PS0"
    assert ps0_reversal.direction == "short"
    assert ps0_trend.symbol == "PS0"
    assert ps0_trend.direction == "long"
    assert lh0_reversal.start_dt == date(2025, 1, 1)
    assert lh0_reversal.end_dt == date(2025, 12, 31)
    assert ps0_reversal.start_dt == date(2025, 1, 1)
    assert ps0_reversal.end_dt == date(2025, 12, 31)
    assert isinstance(lh0_reversal.start_dt, date)
    assert isinstance(lh0_reversal.end_dt, date)
    assert isinstance(ps0_reversal.start_dt, date)
    assert isinstance(ps0_reversal.end_dt, date)


def test_event_replay_cases_have_expected_windows() -> None:
    hog = BUILTIN_CASES["lh0_reversal_long_jul2025"]
    polysilicon = BUILTIN_CASES["ps0_reversal_long_jul2025"]
    coking_coal = BUILTIN_CASES["jm0_reversal_long_jul2025"]
    alumina = BUILTIN_CASES["ao0_reversal_long_jul2024"]
    hog_mixed = BUILTIN_CASES["lh0_mixed_long_jul2025"]
    polysilicon_mixed = BUILTIN_CASES["ps0_mixed_long_jul2025"]
    coking_coal_mixed = BUILTIN_CASES["jm0_mixed_long_jul2025"]
    alumina_mixed = BUILTIN_CASES["ao0_mixed_long_jul2024"]

    assert hog.symbol == "LH0"
    assert hog.direction == "long"
    assert hog.start_dt == date(2025, 7, 1)
    assert hog.end_dt == date(2025, 7, 31)

    assert polysilicon.symbol == "PS0"
    assert polysilicon.direction == "long"
    assert polysilicon.start_dt == date(2025, 6, 25)
    assert polysilicon.end_dt == date(2025, 7, 25)

    assert coking_coal.symbol == "JM0"
    assert coking_coal.direction == "long"
    assert coking_coal.start_dt == date(2025, 6, 4)
    assert coking_coal.end_dt == date(2025, 7, 24)

    assert alumina.symbol == "AO0"
    assert alumina.direction == "long"
    assert alumina.start_dt == date(2024, 7, 8)
    assert alumina.end_dt == date(2024, 7, 19)

    assert hog_mixed.symbol == "LH0"
    assert hog_mixed.direction == "long"
    assert hog_mixed.strategy_family == ""
    assert hog_mixed.start_dt == hog.start_dt
    assert hog_mixed.end_dt == hog.end_dt

    assert polysilicon_mixed.symbol == "PS0"
    assert polysilicon_mixed.direction == "long"
    assert polysilicon_mixed.strategy_family == ""
    assert polysilicon_mixed.start_dt == polysilicon.start_dt
    assert polysilicon_mixed.end_dt == polysilicon.end_dt

    assert coking_coal_mixed.symbol == "JM0"
    assert coking_coal_mixed.direction == "long"
    assert coking_coal_mixed.strategy_family == ""
    assert coking_coal_mixed.start_dt == coking_coal.start_dt
    assert coking_coal_mixed.end_dt == coking_coal.end_dt

    assert alumina_mixed.symbol == "AO0"
    assert alumina_mixed.direction == "long"
    assert alumina_mixed.strategy_family == ""
    assert alumina_mixed.start_dt == alumina.start_dt
    assert alumina_mixed.end_dt == alumina.end_dt


def test_get_case_returns_alias_case_with_requested_case_id():
    case = get_case("lh0_long")

    assert case.case_id == "lh0_long"
    assert case.symbol == BUILTIN_CASES["lh0_reversal_long"].symbol
    assert case.direction == BUILTIN_CASES["lh0_reversal_long"].direction


def test_get_case_returns_ps0_trend_long_alias() -> None:
    case = get_case("ps0_long")

    assert case.case_id == "ps0_long"
    assert case.symbol == BUILTIN_CASES["ps0_trend_long"].symbol
    assert case.direction == "long"
