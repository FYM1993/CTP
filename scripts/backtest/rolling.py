from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

from backtest.metrics import summarize_trades
from backtest.models import BacktestCase, BacktestResult
from backtest.phase23 import make_trade_plan_from_phase2, run_case_from_frames


@dataclass(frozen=True, slots=True)
class RollingWindow:
    index: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


@dataclass(frozen=True, slots=True)
class RollingWindowResult:
    window: RollingWindow
    result: BacktestResult


@dataclass(frozen=True, slots=True)
class RollingBacktestResult:
    base_case_id: str
    effective_start: date | None
    windows: list[RollingWindowResult]
    aggregate_summary: dict[str, float | int]
    aggregate_diagnostics: dict[str, float | int]


WINDOW_DIAGNOSTIC_KEYS = (
    "phase2_actionable_days",
    "phase2_actionable_reversal_days",
    "phase2_actionable_trend_days",
    "phase2_non_actionable_days",
    "phase2_history_insufficient_days",
    "phase2_reject_no_signal_days",
    "phase2_reject_score_gate_days",
    "phase2_reject_rr_gate_days",
    "phase2_reject_duplicate_signal_days",
    "phase3_signal_eval_bars",
    "phase3_entry_signal_hits",
    "trades_opened",
    "trades_opened_reversal",
    "trades_opened_trend",
)


def find_effective_start_date(
    *,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    requested_start: date,
    requested_end: date,
    min_history_bars: int,
) -> date | None:
    if daily_df.empty or minute_df.empty:
        return None

    daily_dates = pd.to_datetime(daily_df["date"]).dt.date
    minute_dates = (
        pd.to_datetime(minute_df["datetime"])
        .dt.date
    )
    trade_days = sorted({d for d in minute_dates.tolist() if requested_start <= d <= requested_end})

    for trade_day in trade_days:
        visible_rows = sum(1 for d in daily_dates.tolist() if d < trade_day)
        if visible_rows >= min_history_bars:
            return trade_day
    return None


def build_walk_forward_windows(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    min_history_bars: int,
    train_days: int,
    validation_days: int,
    step_days: int,
) -> list[RollingWindow]:
    effective_start = find_effective_start_date(
        daily_df=daily_df,
        minute_df=minute_df,
        requested_start=case.start_dt,
        requested_end=case.end_dt,
        min_history_bars=min_history_bars,
    )
    if effective_start is None or minute_df.empty:
        return []

    last_trade_day = pd.to_datetime(minute_df["datetime"]).dt.date.max()
    effective_end = min(case.end_dt, last_trade_day)
    windows: list[RollingWindow] = []
    current = effective_start
    index = 1

    while current <= effective_end:
        validation_end = min(current + timedelta(days=validation_days - 1), effective_end)
        train_end = current - timedelta(days=1)
        train_start = current - timedelta(days=train_days)
        windows.append(
            RollingWindow(
                index=index,
                train_start=train_start,
                train_end=train_end,
                validation_start=current,
                validation_end=validation_end,
            )
        )
        current = current + timedelta(days=step_days)
        index += 1

    return windows


def _empty_aggregate_summary() -> dict[str, float | int]:
    return {
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
    }


def _empty_aggregate_diagnostics(*, effective_start: date | None, windows_skipped_no_history: int) -> dict[str, float | int]:
    diagnostics: dict[str, float | int] = {
        "effective_start": effective_start.isoformat() if effective_start else "NONE",
        "windows_skipped_no_history": windows_skipped_no_history,
    }
    for key in WINDOW_DIAGNOSTIC_KEYS:
        diagnostics[f"total_{key}"] = 0
    return diagnostics


def run_walk_forward_from_frames(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    pre_market_cfg: dict,
    signal_cfg: dict,
    train_days: int,
    validation_days: int,
    step_days: int,
    plan_factory=make_trade_plan_from_phase2,
    run_window=run_case_from_frames,
) -> RollingBacktestResult:
    min_history_bars = max(int((pre_market_cfg or {}).get("min_history_bars", 60)), 1)
    effective_start = find_effective_start_date(
        daily_df=daily_df,
        minute_df=minute_df,
        requested_start=case.start_dt,
        requested_end=case.end_dt,
        min_history_bars=min_history_bars,
    )
    windows = build_walk_forward_windows(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        min_history_bars=min_history_bars,
        train_days=train_days,
        validation_days=validation_days,
        step_days=step_days,
    )
    if not windows:
        return RollingBacktestResult(
            base_case_id=case.case_id,
            effective_start=None,
            windows=[],
            aggregate_summary=_empty_aggregate_summary(),
            aggregate_diagnostics=_empty_aggregate_diagnostics(
                effective_start=None,
                windows_skipped_no_history=1,
            ),
        )

    window_results: list[RollingWindowResult] = []
    aggregate_trades = []
    windows_with_trades = 0

    for window in windows:
        window_case = replace(
            case,
            case_id=f"{case.case_id}_wf{window.index:02d}_{window.validation_start.isoformat()}_{window.validation_end.isoformat()}",
            start_dt=window.validation_start,
            end_dt=window.validation_end,
        )
        window_daily = daily_df.loc[pd.to_datetime(daily_df["date"]).dt.date <= window.validation_end].copy()
        window_minute = minute_df.loc[
            pd.to_datetime(minute_df["datetime"]).dt.date.between(window.validation_start, window.validation_end)
        ].copy()
        result = run_window(
            case=window_case,
            daily_df=window_daily,
            minute_df=window_minute,
            plan_factory=plan_factory,
            pre_market_cfg=pre_market_cfg,
            signal_cfg=signal_cfg,
        )
        window_results.append(RollingWindowResult(window=window, result=result))
        aggregate_trades.extend(result.trades)
        if result.trades:
            windows_with_trades += 1

    aggregate_summary = _empty_aggregate_summary()
    aggregate_summary.update(summarize_trades(aggregate_trades))
    aggregate_summary["num_windows"] = len(window_results)
    aggregate_summary["num_windows_with_trades"] = windows_with_trades
    aggregate_diagnostics = _empty_aggregate_diagnostics(
        effective_start=effective_start,
        windows_skipped_no_history=0,
    )
    for window_result in window_results:
        diagnostics = window_result.result.diagnostics or {}
        for key in WINDOW_DIAGNOSTIC_KEYS:
            aggregate_diagnostics[f"total_{key}"] += int(diagnostics.get(key, 0))

    return RollingBacktestResult(
        base_case_id=case.case_id,
        effective_start=effective_start,
        windows=window_results,
        aggregate_summary=aggregate_summary,
        aggregate_diagnostics=aggregate_diagnostics,
    )
