from __future__ import annotations

import sys
from datetime import date
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_models import BacktestCase, BacktestResult, TradePlan, TradeRecord  # noqa: E402


def test_backtest_case_stores_date_objects():
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        note="baseline long case",
    )

    assert case.case_id == "lh0_long"
    assert case.symbol == "LH0"
    assert case.name == "生猪"
    assert case.direction == "long"
    assert case.start_dt == date(2025, 1, 1)
    assert case.end_dt == date(2025, 12, 31)
    assert case.note == "baseline long case"
    assert isinstance(case.start_dt, date)
    assert isinstance(case.end_dt, date)

    try:
        case.symbol = "PS0"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("BacktestCase should be frozen")


def test_trade_plan_stores_contract_fields():
    plan = TradePlan(
        trade_id="trade-1",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-03",
        entry_ref=123.4,
        stop=120.0,
        tp1=126.0,
        tp2=130.0,
        phase2_score=88.5,
        signal_type="reversal",
        meta={"source": "baseline"},
    )

    assert plan.trade_id == "trade-1"
    assert plan.symbol == "LH0"
    assert plan.direction == "long"
    assert plan.plan_date == "2025-01-03"
    assert plan.entry_ref == 123.4
    assert plan.stop == 120.0
    assert plan.tp1 == 126.0
    assert plan.tp2 == 130.0
    assert plan.phase2_score == 88.5
    assert plan.signal_type == "reversal"
    assert plan.meta == {"source": "baseline"}

    try:
        plan.entry_ref = 100.0  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("TradePlan should be frozen")


def test_trade_record_stores_exit_and_hold_fields():
    record = TradeRecord(
        trade_id="trade-1",
        symbol="LH0",
        direction="long",
        entry_time="2025-01-03T09:00:00",
        entry_price=123.4,
        exit_time="2025-01-06T14:30:00",
        exit_price=128.1,
        exit_reason="tp1",
        bars_held=6,
        days_held=3,
        tp1_hit=True,
        pnl_ratio=0.038,
        meta={"notes": "executed"},
    )

    assert record.trade_id == "trade-1"
    assert record.symbol == "LH0"
    assert record.direction == "long"
    assert record.entry_time == "2025-01-03T09:00:00"
    assert record.entry_price == 123.4
    assert record.exit_time == "2025-01-06T14:30:00"
    assert record.exit_price == 128.1
    assert record.exit_reason == "tp1"
    assert record.bars_held == 6
    assert record.days_held == 3
    assert record.tp1_hit is True
    assert record.pnl_ratio == 0.038
    assert record.meta == {"notes": "executed"}

    try:
        record.exit_reason = "stop"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("TradeRecord should be frozen")


def test_backtest_result_stores_case_trades_and_summary():
    record = TradeRecord(
        trade_id="trade-1",
        symbol="LH0",
        direction="long",
        entry_time="2025-01-03T09:00:00",
        entry_price=123.4,
        exit_time="2025-01-06T14:30:00",
        exit_price=128.1,
        exit_reason="tp1",
        bars_held=6,
        days_held=3,
        tp1_hit=True,
        pnl_ratio=0.038,
    )
    result = BacktestResult(
        case_id="lh0_long",
        trades=[record],
        summary={"trade_count": 1, "win_rate": 1.0},
    )

    assert result.case_id == "lh0_long"
    assert result.trades == [record]
    assert result.summary == {"trade_count": 1, "win_rate": 1.0}

    try:
        result.summary = {"trade_count": 2}  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("BacktestResult should be frozen")
