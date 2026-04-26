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
from strategy_reversal.backtest import make_reversal_trade_plan
from strategy_trend.backtest import make_trend_trade_plan


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
    parser.add_argument("--debug-trades", action="store_true", help="Print detailed trade breakdown for self-debugging")
    parser.add_argument("--debug-phase2", action="store_true", help="Print per-day Phase 2 rejection/actionable details for self-debugging")
    parser.add_argument("--window-index", type=int, help="When used with --rolling debug output, only print the selected validation window")
    parser.add_argument(
        "--fundamental-mode",
        choices=("strict", "proxy"),
        help="Backtest reversal fundamentals as strict confirmed evidence or allow proxy replay",
    )
    return parser


def _print_summary(case_id: str, summary: dict[str, float | int]) -> None:
    print(f"case={case_id}")
    for key, value in summary.items():
        print(f"{key}={value}")


def _print_case_metadata(case: BacktestCase) -> None:
    print(f"diag_strategy_family={case.strategy_family}")


def _print_diagnostics(diagnostics: dict[str, float | int]) -> None:
    for key, value in diagnostics.items():
        print(f"diag_{key}={value}")


def _print_rolling_windows(result) -> None:
    keys = (
        "phase2_actionable_days",
        "phase2_actionable_reversal_days",
        "phase2_actionable_trend_days",
        "phase2_actionable_strategy_reversal_days",
        "phase2_actionable_strategy_trend_days",
        "phase2_non_actionable_days",
        "phase2_history_insufficient_days",
        "phase2_reject_no_signal_days",
        "phase2_reject_score_gate_days",
        "phase2_reject_rr_gate_days",
        "phase2_reject_duplicate_signal_days",
        "phase2_reject_missing_fundamental_days",
        "phase3_signal_eval_bars",
        "phase3_entry_signal_hits",
        "trades_opened",
        "trades_opened_reversal",
        "trades_opened_trend",
        "trades_opened_strategy_reversal",
        "trades_opened_strategy_trend",
        "backtest_fundamental_blocked_days",
        "backtest_fundamental_proxy_days",
    )
    for window_result in result.windows:
        prefix = f"wf{window_result.window.index:02d}"
        print(f"diag_{prefix}_validation_start={window_result.window.validation_start.isoformat()}")
        print(f"diag_{prefix}_validation_end={window_result.window.validation_end.isoformat()}")
        print(f"diag_{prefix}_num_trades={len(window_result.result.trades)}")
        diagnostics = window_result.result.diagnostics or {}
        for key in keys:
            print(f"diag_{prefix}_{key}={diagnostics.get(key, 0)}")


def _print_trade_debug(*, trades, prefix: str = "") -> None:
    for idx, trade in enumerate(trades, start=1):
        label = f"debug_{prefix}trade_{idx}" if prefix else f"debug_trade_{idx}"
        print(f"{label}_trade_id={trade.trade_id}")
        print(f"{label}_symbol={trade.symbol}")
        print(f"{label}_direction={trade.direction}")
        print(f"{label}_plan_date={trade.meta.get('plan_date', '')}")
        print(f"{label}_strategy_family={trade.meta.get('strategy_family', '')}")
        print(f"{label}_entry_family={trade.meta.get('entry_family', '')}")
        print(f"{label}_entry_signal_type={trade.meta.get('entry_signal_type', '')}")
        print(f"{label}_entry_signal_detail={trade.meta.get('entry_signal_detail', '')}")
        print(f"{label}_phase2_score={trade.meta.get('phase2_score', '')}")
        print(f"{label}_management_profile={trade.meta.get('management_profile', '')}")
        print(f"{label}_management_tp1_exit_fraction={trade.meta.get('management_tp1_exit_fraction', '')}")
        print(
            f"{label}_management_move_stop_to_entry_after_tp1="
            f"{trade.meta.get('management_move_stop_to_entry_after_tp1', '')}"
        )
        print(f"{label}_entry_time={trade.entry_time}")
        print(f"{label}_entry_price={trade.entry_price}")
        print(f"{label}_exit_time={trade.exit_time}")
        print(f"{label}_exit_price={trade.exit_price}")
        print(f"{label}_exit_reason={trade.exit_reason}")
        print(f"{label}_bars_held={trade.bars_held}")
        print(f"{label}_days_held={trade.days_held}")
        print(f"{label}_tp1_hit={trade.tp1_hit}")
        print(f"{label}_pnl_ratio={trade.pnl_ratio}")


def _print_phase2_debug(*, phase2_days: list[dict[str, object]], prefix: str = "") -> None:
    for idx, day in enumerate(phase2_days, start=1):
        label = f"debug_{prefix}phase2_day_{idx}" if prefix else f"debug_phase2_day_{idx}"
        for key, value in day.items():
            print(f"{label}_{key}={value}")


def _selected_window_results(result, window_index: int | None):
    if window_index is None:
        return result.windows
    return [window_result for window_result in result.windows if window_result.window.index == window_index]


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


def _plan_factory_for_case(case: BacktestCase):
    if case.strategy_family == "reversal_fundamental":
        return make_reversal_trade_plan
    if case.strategy_family == "trend_following":
        return make_trend_trade_plan
    return make_trade_plan_from_phase2


def _resolved_fundamental_mode(config: dict, cli_mode: str | None) -> str:
    if cli_mode in {"strict", "proxy"}:
        return cli_mode
    pre_market_cfg = config.get("pre_market") or {}
    backtest_cfg = pre_market_cfg.get("backtest") or {}
    if not isinstance(backtest_cfg, dict):
        backtest_cfg = {}
    configured = (
        pre_market_cfg.get("backtest_fundamental_mode")
        or pre_market_cfg.get("fundamental_data_mode")
        or backtest_cfg.get("fundamental_mode")
        or backtest_cfg.get("fundamental_data_mode")
    )
    if str(configured).strip().lower() == "proxy":
        return "proxy"
    return "strict"


def _config_with_fundamental_mode(config: dict, mode: str) -> dict:
    out = dict(config)
    pre_market_cfg = dict(out.get("pre_market") or {})
    pre_market_cfg["backtest_fundamental_mode"] = mode
    out["pre_market"] = pre_market_cfg
    return out


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
    loaded_config = load_config()
    config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )

    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
    plan_factory = _plan_factory_for_case(case)
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
            plan_factory=plan_factory,
            capture_debug=bool(args.debug_phase2),
        )
        _print_summary(case.case_id, result.aggregate_summary)
        _print_case_metadata(case)
        _print_diagnostics(result.aggregate_diagnostics)
        _print_rolling_windows(result)
        if args.debug_trades:
            for window_result in _selected_window_results(result, args.window_index):
                prefix = f"wf{window_result.window.index:02d}_"
                _print_trade_debug(trades=window_result.result.trades, prefix=prefix)
        if args.debug_phase2:
            for window_result in _selected_window_results(result, args.window_index):
                prefix = f"wf{window_result.window.index:02d}_"
                _print_phase2_debug(
                    phase2_days=list(window_result.result.debug.get("phase2_days") or []),
                    prefix=prefix,
                )
        return 0

    result = run_case_from_frames(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=plan_factory,
        pre_market_cfg=config.get("pre_market") or {},
        signal_cfg=config.get("intraday") or {},
        capture_debug=bool(args.debug_phase2),
    )
    summary = summarize_trades(result.trades)
    _print_summary(case.case_id, summary)
    _print_case_metadata(case)
    _print_diagnostics(result.diagnostics)
    if args.debug_trades:
        _print_trade_debug(trades=result.trades)
    if args.debug_phase2:
        _print_phase2_debug(phase2_days=list(result.debug.get("phase2_days") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
