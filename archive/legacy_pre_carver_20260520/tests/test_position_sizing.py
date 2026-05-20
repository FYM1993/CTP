from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared.position_sizing import calculate_position_sizing  # noqa: E402


def test_calculate_position_sizing_takes_minimum_of_score_margin_and_stop_risk() -> None:
    result = calculate_position_sizing(
        score_margin_budget=200_000.0,
        portfolio_margin_budget=300_000.0,
        risk_budget=15_000.0,
        margin_per_lot=50_000.0,
        risk_per_lot=7_000.0,
    )

    assert result.score_lots == 4
    assert result.portfolio_lots == 6
    assert result.risk_lots == 2
    assert result.suggested_lots == 2
    assert result.sizing_limited_by == "stop_risk"
    assert result.sizing_zero_lot_reason == ""
    assert result.suggested_margin == pytest.approx(100_000.0)
    assert result.suggested_stop_risk == pytest.approx(14_000.0)


def test_calculate_position_sizing_explains_zero_lot_due_to_stop_risk() -> None:
    result = calculate_position_sizing(
        score_margin_budget=300_000.0,
        portfolio_margin_budget=300_000.0,
        risk_budget=15_000.0,
        margin_per_lot=100_000.0,
        risk_per_lot=20_000.0,
    )

    assert result.suggested_lots == 0
    assert result.sizing_limited_by == "stop_risk"
    assert result.sizing_zero_lot_reason == "single_trade_stop_risk_below_one_lot"


def test_calculate_position_sizing_accepts_precomputed_account_caps() -> None:
    result = calculate_position_sizing(
        score_lots=3,
        portfolio_lots=1,
        risk_lots=4,
        margin_per_lot=60_000.0,
        risk_per_lot=5_000.0,
        risk_budget=15_000.0,
    )

    assert result.suggested_lots == 1
    assert result.sizing_limited_by == "portfolio_margin"
    assert result.suggested_margin == pytest.approx(60_000.0)
    assert result.suggested_stop_risk == pytest.approx(5_000.0)
