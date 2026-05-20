from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.experiment_cache import (  # noqa: E402
    build_phase2_experiment_cache,
    build_phase23_experiment_cache,
    compare_phase2_cache_to_fresh_build,
    read_phase2_experiment_cache,
    read_phase23_experiment_cache,
    write_phase2_experiment_cache,
    write_phase23_experiment_cache,
)
from backtest.models import BacktestCase, BacktestResult, TradePlan, TradeRecord  # noqa: E402


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
            "oi": [500, 520, 540],
        }
    )


def _minute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00", "2025-01-03 09:00:00"]),
            "open": [101.0, 102.0],
            "high": [102.0, 103.0],
            "low": [100.0, 101.0],
            "close": [101.0, 102.0],
            "volume": [10, 12],
        }
    )


def _plan_factory(*, case, daily_df, pre_market_cfg):
    plan_date = str(pd.to_datetime(daily_df["date"]).max().date())
    entry = float(daily_df.iloc[-1]["close"])
    return TradePlan(
        trade_id=f"{case.case_id}_{plan_date}",
        symbol=case.symbol,
        direction=case.direction,
        plan_date=plan_date,
        entry_ref=entry,
        stop=entry - 2.0,
        tp1=entry + 3.0,
        tp2=entry + 6.0,
        phase2_score=66.0,
        signal_type="TrendPullback",
        meta={"entry_family": "trend", "strategy_family": "trend_following"},
    )


def test_phase2_experiment_cache_roundtrips_and_matches_fresh_build(tmp_path: Path) -> None:
    case = BacktestCase(
        case_id="au0_trend_long_2025",
        symbol="AU0",
        name="黄金",
        direction="long",
        start_dt=date(2025, 1, 2),
        end_dt=date(2025, 1, 3),
        strategy_family="trend_following",
    )
    pre_market_cfg = {"min_history_bars": 1}

    cache = build_phase2_experiment_cache(
        case=case,
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        pre_market_cfg=pre_market_cfg,
        plan_factory=_plan_factory,
    )
    output = tmp_path / "phase2_cache.json"
    write_phase2_experiment_cache(cache, output)
    loaded = read_phase2_experiment_cache(output)

    assert [entry.trade_date for entry in loaded.entries] == ["2025-01-02", "2025-01-03"]
    assert [entry.plan.trade_id for entry in loaded.entries if entry.plan is not None] == [
        "au0_trend_long_2025_2025-01-01",
        "au0_trend_long_2025_2025-01-02",
    ]
    assert loaded.entries[0].plan is not None
    assert loaded.entries[0].plan.entry_ref == 100.0
    assert loaded.entries[0].debug_snapshot["entry_family"] == "trend"

    mismatches = compare_phase2_cache_to_fresh_build(
        loaded,
        case=case,
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        pre_market_cfg=pre_market_cfg,
        plan_factory=_plan_factory,
    )

    assert mismatches == []


def test_phase23_experiment_cache_roundtrips_backtest_result(tmp_path: Path) -> None:
    case = BacktestCase(
        case_id="au0_trend_long_2025",
        symbol="AU0",
        name="黄金",
        direction="long",
        start_dt=date(2025, 1, 2),
        end_dt=date(2025, 1, 3),
        strategy_family="trend_following",
    )
    result = BacktestResult(
        case_id=case.case_id,
        trades=[
            TradeRecord(
                trade_id="au0_trend_long_2025_2025-01-02",
                symbol="AU0",
                direction="long",
                entry_time="2025-01-02 09:00:00",
                entry_price=100.0,
                exit_time="2025-01-03 09:00:00",
                exit_price=110.0,
                exit_reason="tp2",
                bars_held=20,
                days_held=1,
                tp1_hit=True,
                pnl_ratio=0.10,
                meta={"phase2_score": 66.0, "entry_family": "trend"},
            )
        ],
        summary={"num_trades": 1},
        diagnostics={"phase2_actionable_days": 1, "phase3_entry_signal_hits": 1},
    )

    cache = build_phase23_experiment_cache(
        case=case,
        result=result,
        pre_market_cfg={"trend_min_first_target_rr": 1.0},
        signal_cfg={"entry": "test"},
    )
    output = tmp_path / "phase23_cache.json"
    write_phase23_experiment_cache(cache, output)
    loaded = read_phase23_experiment_cache(output)

    assert loaded.case_id == "au0_trend_long_2025"
    assert loaded.result.case_id == "au0_trend_long_2025"
    assert loaded.result.trades[0].trade_id == "au0_trend_long_2025_2025-01-02"
    assert loaded.result.trades[0].meta["phase2_score"] == 66.0
    assert loaded.result.diagnostics["phase3_entry_signal_hits"] == 1
