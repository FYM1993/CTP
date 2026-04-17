from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_models import BacktestCase, TradePlan  # noqa: E402
from backtest_phase23 import (  # noqa: E402
    aggregate_partial_5m_bars,
    make_trade_plan_from_phase2,
    run_case_from_frames,
)


def _make_case(direction: str) -> BacktestCase:
    return BacktestCase(
        case_id=f"case_{direction}",
        symbol="LH0" if direction == "long" else "PS0",
        name="生猪" if direction == "long" else "豆粕",
        direction=direction,
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
    )


def test_aggregate_partial_5m_bars_keeps_incomplete_current_bar():
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 09:31:00",
                    "2025-01-06 09:32:00",
                    "2025-01-06 09:34:00",
                    "2025-01-06 09:35:00",
                    "2025-01-06 09:36:00",
                ]
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 104.5, 105.5],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    bars = aggregate_partial_5m_bars(minute_df)

    assert list(bars["datetime"]) == [
        pd.Timestamp("2025-01-06 09:30:00"),
        pd.Timestamp("2025-01-06 09:35:00"),
    ]
    assert float(bars.iloc[0]["open"]) == 100.0
    assert float(bars.iloc[0]["high"]) == 103.0
    assert float(bars.iloc[0]["low"]) == 99.0
    assert float(bars.iloc[0]["close"]) == 102.5
    assert float(bars.iloc[0]["volume"]) == 60.0
    assert float(bars.iloc[1]["open"]) == 103.0
    assert float(bars.iloc[1]["high"]) == 106.0
    assert float(bars.iloc[1]["low"]) == 102.0
    assert float(bars.iloc[1]["close"]) == 105.5
    assert float(bars.iloc[1]["volume"]) == 90.0


def test_make_trade_plan_from_phase2_maps_actionable_raw_plan(monkeypatch):
    raw_plan = {
        "symbol": "LH0",
        "direction": "long",
        "score": 36.5,
        "actionable": True,
        "entry": 103.0,
        "stop": 99.0,
        "tp1": 106.0,
        "tp2": 109.0,
        "reversal_status": {
            "signal_date": "2025-01-06",
            "signal_type": "spring",
        },
    }

    monkeypatch.setattr(
        "backtest_phase23.pre_market.build_trade_plan_from_daily_df",
        lambda **_kwargs: raw_plan,
    )

    plan = make_trade_plan_from_phase2(
        case=_make_case("long"),
        daily_df=pd.DataFrame({"date": pd.to_datetime(["2025-01-03"])}),
        pre_market_cfg={},
    )

    assert plan == TradePlan(
        trade_id="case_long_2025-01-06",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=103.0,
        stop=99.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=36.5,
        signal_type="spring",
    )


def test_run_case_from_frames_opens_long_and_exits_on_tp2(monkeypatch):
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 09:31:00",
                    "2025-01-06 09:32:00",
                    "2025-01-06 09:33:00",
                    "2025-01-06 09:34:00",
                ]
            ),
            "open": [100.0, 101.0, 103.0, 107.0],
            "high": [101.0, 103.0, 108.0, 111.0],
            "low": [99.0, 100.0, 102.0, 106.0],
            "close": [100.0, 103.0, 107.0, 110.0],
            "volume": [10, 10, 10, 10],
        }
    )
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05"]),
            "open": [95.0, 96.0, 97.0],
            "high": [101.0, 102.0, 103.0],
            "low": [94.0, 95.0, 96.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
            "oi": [5000, 5001, 5002],
        }
    )
    plan = TradePlan(
        trade_id="trade-long",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=102.0,
        stop=98.0,
        tp1=106.0,
        tp2=109.0,
        phase2_score=30.0,
        signal_type="spring",
    )

    monkeypatch.setattr(
        "backtest_phase23.generate_signals",
        lambda df, direction, cfg: [{"type": "开多"}] if len(df) >= 1 else [],
    )

    result = run_case_from_frames(
        case=_make_case("long"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.case_id == "case_long"
    assert result.summary == {"num_trades": 1}
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trade_id == "trade-long"
    assert trade.direction == "long"
    assert trade.entry_time == "2025-01-06 09:31:00"
    assert trade.entry_price == 100.0
    assert trade.exit_time == "2025-01-06 09:34:00"
    assert trade.exit_price == 110.0
    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.pnl_ratio == 0.10


def test_run_case_from_frames_opens_short_and_exits_on_tp2(monkeypatch):
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 09:31:00",
                    "2025-01-06 09:32:00",
                    "2025-01-06 09:33:00",
                    "2025-01-06 09:34:00",
                ]
            ),
            "open": [200.0, 198.0, 194.0, 190.0],
            "high": [201.0, 199.0, 195.0, 191.0],
            "low": [199.0, 197.0, 193.0, 188.0],
            "close": [200.0, 194.0, 190.0, 188.0],
            "volume": [10, 10, 10, 10],
        }
    )
    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05"]),
            "open": [205.0, 204.0, 203.0],
            "high": [206.0, 205.0, 204.0],
            "low": [199.0, 198.0, 197.0],
            "close": [200.0, 199.0, 198.0],
            "volume": [1000, 1000, 1000],
            "oi": [7000, 7001, 7002],
        }
    )
    plan = TradePlan(
        trade_id="trade-short",
        symbol="PS0",
        direction="short",
        plan_date="2025-01-06",
        entry_ref=199.0,
        stop=205.0,
        tp1=192.0,
        tp2=189.0,
        phase2_score=-31.0,
        signal_type="upthrust",
    )

    monkeypatch.setattr(
        "backtest_phase23.generate_signals",
        lambda df, direction, cfg: [{"type": "开空"}] if len(df) >= 1 else [],
    )

    result = run_case_from_frames(
        case=_make_case("short"),
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert result.case_id == "case_short"
    assert result.summary == {"num_trades": 1}
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trade_id == "trade-short"
    assert trade.direction == "short"
    assert trade.entry_time == "2025-01-06 09:31:00"
    assert trade.entry_price == 200.0
    assert trade.exit_time == "2025-01-06 09:34:00"
    assert trade.exit_price == 188.0
    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.pnl_ratio == 0.06
