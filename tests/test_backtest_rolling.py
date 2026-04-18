from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.models import BacktestCase, BacktestResult, TradeRecord  # noqa: E402
from backtest.rolling import (  # noqa: E402
    RollingBacktestResult,
    RollingWindow,
    RollingWindowResult,
    build_walk_forward_windows,
    find_effective_start_date,
    run_walk_forward_from_frames,
)


def _case() -> BacktestCase:
    return BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
    )


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=15, freq="D"),
            "open": [100.0] * 15,
            "high": [101.0] * 15,
            "low": [99.0] * 15,
            "close": [100.5] * 15,
            "volume": [1000.0] * 15,
            "oi": [5000.0] * 15,
        }
    )


def _minute_frame() -> pd.DataFrame:
    datetimes = pd.date_range("2025-01-04 09:31:00", periods=10, freq="D")
    return pd.DataFrame(
        {
            "datetime": datetimes,
            "open": [100.0] * len(datetimes),
            "high": [101.0] * len(datetimes),
            "low": [99.0] * len(datetimes),
            "close": [100.5] * len(datetimes),
            "volume": [10.0] * len(datetimes),
        }
    )


def test_find_effective_start_date_uses_visible_daily_history() -> None:
    out = find_effective_start_date(
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        requested_start=date(2025, 1, 1),
        requested_end=date(2025, 1, 31),
        min_history_bars=3,
    )

    assert out == date(2025, 1, 4)


def test_build_walk_forward_windows_returns_empty_when_history_never_sufficient() -> None:
    windows = build_walk_forward_windows(
        case=_case(),
        daily_df=_daily_frame().head(2),
        minute_df=_minute_frame().head(2),
        min_history_bars=10,
        train_days=5,
        validation_days=5,
        step_days=5,
    )

    assert windows == []


def test_build_walk_forward_windows_uses_effective_start_and_step() -> None:
    windows = build_walk_forward_windows(
        case=_case(),
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        min_history_bars=3,
        train_days=10,
        validation_days=5,
        step_days=5,
    )

    assert windows == [
        RollingWindow(
            index=1,
            train_start=date(2024, 12, 25),
            train_end=date(2025, 1, 3),
            validation_start=date(2025, 1, 4),
            validation_end=date(2025, 1, 8),
        ),
        RollingWindow(
            index=2,
            train_start=date(2024, 12, 30),
            train_end=date(2025, 1, 8),
            validation_start=date(2025, 1, 9),
            validation_end=date(2025, 1, 13),
        ),
    ]


def test_run_walk_forward_from_frames_aggregates_validation_windows() -> None:
    trade = TradeRecord(
        trade_id="t1",
        symbol="LH0",
        direction="long",
        entry_time="2025-01-04 09:31:00",
        entry_price=100.0,
        exit_time="2025-01-04 10:00:00",
        exit_price=101.0,
        exit_reason="tp1",
        bars_held=2,
        days_held=0,
        tp1_hit=True,
        pnl_ratio=0.01,
    )

    def fake_run_window(*, case, daily_df, minute_df, **kwargs):
        if case.start_dt == date(2025, 1, 4):
            return BacktestResult(
                case_id=case.case_id,
                trades=[trade],
                summary={"num_trades": 1},
                diagnostics={"phase2_actionable_days": 2},
            )
        return BacktestResult(
            case_id=case.case_id,
            trades=[],
            summary={"num_trades": 0},
            diagnostics={"phase2_actionable_days": 0},
        )

    result = run_walk_forward_from_frames(
        case=_case(),
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        pre_market_cfg={"min_history_bars": 3},
        signal_cfg={},
        train_days=10,
        validation_days=5,
        step_days=5,
        run_window=fake_run_window,
    )

    assert result.effective_start == date(2025, 1, 4)
    assert result.aggregate_summary == {
        "num_windows": 2,
        "num_windows_with_trades": 1,
        "num_trades": 1,
        "wins": 1,
        "win_rate": 1.0,
        "avg_pnl": 0.01,
        "total_pnl": 0.01,
        "tp1_hits": 1,
        "tp2_hits": 0,
        "stop_hits": 0,
    }
    assert result.aggregate_diagnostics == {
        "effective_start": "2025-01-04",
        "windows_skipped_no_history": 0,
        "total_phase2_actionable_days": 2,
        "total_phase2_actionable_reversal_days": 0,
        "total_phase2_actionable_trend_days": 0,
        "total_phase2_non_actionable_days": 0,
        "total_phase2_history_insufficient_days": 0,
        "total_phase2_reject_no_signal_days": 0,
        "total_phase2_reject_score_gate_days": 0,
        "total_phase2_reject_rr_gate_days": 0,
        "total_phase2_reject_duplicate_signal_days": 0,
        "total_phase3_signal_eval_bars": 0,
        "total_phase3_entry_signal_hits": 0,
        "total_trades_opened": 0,
        "total_trades_opened_reversal": 0,
        "total_trades_opened_trend": 0,
    }
    assert [window.window.index for window in result.windows] == [1, 2]
    assert result.windows[0] == RollingWindowResult(
        window=RollingWindow(
            index=1,
            train_start=date(2024, 12, 25),
            train_end=date(2025, 1, 3),
            validation_start=date(2025, 1, 4),
            validation_end=date(2025, 1, 8),
        ),
        result=BacktestResult(
            case_id="lh0_long_wf01_2025-01-04_2025-01-08",
            trades=[trade],
            summary={"num_trades": 1},
            diagnostics={"phase2_actionable_days": 2},
        ),
    )
    assert result.windows[1].result.diagnostics == {"phase2_actionable_days": 0}


def test_run_walk_forward_from_frames_marks_no_history_when_no_windows() -> None:
    result = run_walk_forward_from_frames(
        case=_case(),
        daily_df=_daily_frame().head(2),
        minute_df=_minute_frame().head(2),
        pre_market_cfg={"min_history_bars": 10},
        signal_cfg={},
        train_days=10,
        validation_days=5,
        step_days=5,
    )

    assert result == RollingBacktestResult(
        base_case_id="lh0_long",
        effective_start=None,
        windows=[],
        aggregate_summary={
            "num_windows": 0,
            "num_windows_with_trades": 0,
            "num_trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "stop_hits": 0,
        },
        aggregate_diagnostics={
            "effective_start": "NONE",
            "windows_skipped_no_history": 1,
            "total_phase2_actionable_days": 0,
            "total_phase2_actionable_reversal_days": 0,
            "total_phase2_actionable_trend_days": 0,
            "total_phase2_non_actionable_days": 0,
            "total_phase2_history_insufficient_days": 0,
            "total_phase2_reject_no_signal_days": 0,
            "total_phase2_reject_score_gate_days": 0,
            "total_phase2_reject_rr_gate_days": 0,
            "total_phase2_reject_duplicate_signal_days": 0,
            "total_phase3_signal_eval_bars": 0,
            "total_phase3_entry_signal_hits": 0,
            "total_trades_opened": 0,
            "total_trades_opened_reversal": 0,
            "total_trades_opened_trend": 0,
        },
    )


def test_run_walk_forward_from_frames_aggregates_window_diagnostics() -> None:
    def fake_run_window(*, case, daily_df, minute_df, **kwargs):
        if case.start_dt == date(2025, 1, 4):
            diagnostics = {
                "phase2_actionable_days": 2,
                "phase2_actionable_reversal_days": 0,
                "phase2_actionable_trend_days": 2,
                "phase2_non_actionable_days": 1,
                "phase2_history_insufficient_days": 0,
                "phase2_reject_no_signal_days": 0,
                "phase2_reject_score_gate_days": 1,
                "phase2_reject_rr_gate_days": 0,
                "phase2_reject_duplicate_signal_days": 0,
                "phase3_signal_eval_bars": 6,
                "phase3_entry_signal_hits": 1,
                "trades_opened": 1,
                "trades_opened_reversal": 0,
                "trades_opened_trend": 1,
            }
        else:
            diagnostics = {
                "phase2_actionable_days": 0,
                "phase2_actionable_reversal_days": 0,
                "phase2_actionable_trend_days": 0,
                "phase2_non_actionable_days": 3,
                "phase2_history_insufficient_days": 2,
                "phase2_reject_no_signal_days": 2,
                "phase2_reject_score_gate_days": 0,
                "phase2_reject_rr_gate_days": 1,
                "phase2_reject_duplicate_signal_days": 0,
                "phase3_signal_eval_bars": 4,
                "phase3_entry_signal_hits": 0,
                "trades_opened": 0,
                "trades_opened_reversal": 0,
                "trades_opened_trend": 0,
            }
        return BacktestResult(
            case_id=case.case_id,
            trades=[],
            summary={"num_trades": 0},
            diagnostics=diagnostics,
        )

    result = run_walk_forward_from_frames(
        case=_case(),
        daily_df=_daily_frame(),
        minute_df=_minute_frame(),
        pre_market_cfg={"min_history_bars": 3},
        signal_cfg={},
        train_days=10,
        validation_days=5,
        step_days=5,
        run_window=fake_run_window,
    )

    assert result.aggregate_diagnostics == {
        "effective_start": "2025-01-04",
        "windows_skipped_no_history": 0,
        "total_phase2_actionable_days": 2,
        "total_phase2_actionable_reversal_days": 0,
        "total_phase2_actionable_trend_days": 2,
        "total_phase2_non_actionable_days": 4,
        "total_phase2_history_insufficient_days": 2,
        "total_phase2_reject_no_signal_days": 2,
        "total_phase2_reject_score_gate_days": 1,
        "total_phase2_reject_rr_gate_days": 1,
        "total_phase2_reject_duplicate_signal_days": 0,
        "total_phase3_signal_eval_bars": 10,
        "total_phase3_entry_signal_hits": 1,
        "total_trades_opened": 1,
        "total_trades_opened_reversal": 0,
        "total_trades_opened_trend": 1,
    }
