from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

from backtest.cases import get_case
from backtest.models import BacktestCase
from backtest.metrics import summarize_trades
from backtest.rolling import run_walk_forward_from_frames
from backtest.phase23 import (
    load_case_frames_with_tqbacktest,
    make_trade_plan_from_phase2,
    run_case_from_frames,
)
from data_cache import _load_config as load_config


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal Phase 2/3 backtest case")
    parser.add_argument("--case", required=True, help="Built-in backtest case id, e.g. lh0_long")
    parser.add_argument("--start", type=_parse_date, help="Override case start date, e.g. 2025-01-01")
    parser.add_argument("--end", type=_parse_date, help="Override case end date, e.g. 2025-03-31")
    parser.add_argument("--rolling", action="store_true", help="Run walk-forward validation instead of a single continuous case")
    parser.add_argument("--train-days", type=int, default=180, help="Calendar days in each walk-forward train segment")
    parser.add_argument("--validation-days", type=int, default=60, help="Calendar days in each validation segment")
    parser.add_argument("--step-days", type=int, default=60, help="Calendar days to advance each walk-forward step")
    return parser


def _print_summary(case_id: str, summary: dict[str, float | int]) -> None:
    print(f"case={case_id}")
    for key, value in summary.items():
        print(f"{key}={value}")


def _print_diagnostics(diagnostics: dict[str, float | int]) -> None:
    for key, value in diagnostics.items():
        print(f"diag_{key}={value}")


def _print_rolling_windows(result) -> None:
    keys = (
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
    for window_result in result.windows:
        prefix = f"wf{window_result.window.index:02d}"
        print(f"diag_{prefix}_validation_start={window_result.window.validation_start.isoformat()}")
        print(f"diag_{prefix}_validation_end={window_result.window.validation_end.isoformat()}")
        print(f"diag_{prefix}_num_trades={len(window_result.result.trades)}")
        diagnostics = window_result.result.diagnostics or {}
        for key in keys:
            print(f"diag_{prefix}_{key}={diagnostics.get(key, 0)}")


def _apply_date_overrides(case: BacktestCase, *, start_dt: date | None, end_dt: date | None) -> BacktestCase:
    effective_start = start_dt or case.start_dt
    effective_end = end_dt or case.end_dt
    if effective_start > effective_end:
        raise ValueError("start date must be on or before end date")
    if effective_start == case.start_dt and effective_end == case.end_dt:
        return case
    return replace(
        case,
        case_id=f"{case.case_id}_{effective_start.isoformat()}_{effective_end.isoformat()}",
        start_dt=effective_start,
        end_dt=effective_end,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        case = _apply_date_overrides(
            get_case(args.case),
            start_dt=args.start,
            end_dt=args.end,
        )
    except ValueError as exc:
        parser.error(str(exc))
    config = load_config()

    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
    if args.rolling:
        result = run_walk_forward_from_frames(
            case=case,
            daily_df=daily_df,
            minute_df=minute_df,
            pre_market_cfg=config.get("pre_market") or {},
            signal_cfg=config.get("intraday") or {},
            train_days=args.train_days,
            validation_days=args.validation_days,
            step_days=args.step_days,
        )
        _print_summary(case.case_id, result.aggregate_summary)
        _print_diagnostics(result.aggregate_diagnostics)
        _print_rolling_windows(result)
        return 0

    result = run_case_from_frames(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=make_trade_plan_from_phase2,
        pre_market_cfg=config.get("pre_market") or {},
        signal_cfg=config.get("intraday") or {},
    )
    summary = summarize_trades(result.trades)
    _print_summary(case.case_id, summary)
    _print_diagnostics(result.diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
