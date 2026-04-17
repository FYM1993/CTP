from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_models import BacktestCase, BacktestResult, TradePlan, TradeRecord  # noqa: E402


def test_backtest_case_preserves_fields_and_is_frozen():
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="baseline long case",
        direction="long",
        start_dt="2024-01-01",
        end_dt="2024-01-31",
        note="test note",
    )

    assert case.case_id == "lh0_long"
    assert case.symbol == "LH0"
    assert case.name == "baseline long case"
    assert case.direction == "long"
    assert case.start_dt == "2024-01-01"
    assert case.end_dt == "2024-01-31"
    assert case.note == "test note"

    try:
        case.symbol = "PS0"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("BacktestCase should be frozen")


def test_trade_plan_preserves_fixed_fields():
    plan = TradePlan(
        entry_ref="entry-1",
        stop=97.5,
        tp1=102.5,
        tp2=105.0,
        phase2_score=88.0,
        signal_type="reversal",
    )

    assert plan.entry_ref == "entry-1"
    assert plan.stop == 97.5
    assert plan.tp1 == 102.5
    assert plan.tp2 == 105.0
    assert plan.phase2_score == 88.0
    assert plan.signal_type == "reversal"


def test_trade_record_preserves_exit_fields():
    record = TradeRecord(
        exit_reason="tp1",
        tp1_hit=True,
        days_held=4,
    )

    assert record.exit_reason == "tp1"
    assert record.tp1_hit is True
    assert record.days_held == 4


def test_backtest_result_exists():
    result = BacktestResult()
    assert result == BacktestResult()
