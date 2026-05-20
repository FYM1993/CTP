from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backtest.cases import get_case
from backtest.models import TradeRecord
from backtest.phase23 import load_case_frames_with_tqbacktest, run_case_from_frames
from backtest.metrics import summarize_trades
from run_backtest import (
    _apply_date_overrides,
    _config_with_fundamental_mode,
    _plan_factory_for_case,
    _resolved_fundamental_mode,
)
from shared.config_loader import load_yaml_config


def load_config() -> dict:
    return load_yaml_config(__file__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trend backtest parameter combinations")
    parser.add_argument("--case", required=True)
    parser.add_argument("--start", type=lambda value: __import__("datetime").date.fromisoformat(value))
    parser.add_argument("--end", type=lambda value: __import__("datetime").date.fromisoformat(value))
    parser.add_argument("--trend-min-stop-atr-multiples", default="legacy,1.0,1.5,2.0")
    parser.add_argument("--trend-max-entry-adverse-r-values", default="legacy,0.5,1.0")
    parser.add_argument("--trend-min-first-target-rr-values", default="legacy")
    parser.add_argument("--trend-second-target-rr-relax-threshold-values", default="legacy")
    parser.add_argument("--trend-relaxed-first-target-rr-values", default="legacy")
    parser.add_argument("--trend-entry-confirmation-bars-values", default="legacy")
    parser.add_argument("--trend-initial-stop-grace-bars-values", default="legacy")
    parser.add_argument("--early-stop-days", type=float, default=3.0)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--output", type=Path)
    return parser


def _parse_grid(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for token in raw.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in {"legacy", "none", "off"}:
            values.append(None)
        else:
            values.append(float(value))
    return values or [None]


def _config_for_combo(
    base_config: dict,
    *,
    min_stop_atr: float | None,
    max_entry_adverse_r: float | None,
    min_first_target_rr: float | None,
    entry_confirmation_bars: float | None,
    initial_stop_grace_bars: float | None,
    second_target_rr_relax_threshold: float | None = None,
    relaxed_first_target_rr: float | None = None,
) -> dict:
    config = dict(base_config)
    pre_market_cfg = dict(config.get("pre_market") or {})
    if min_stop_atr is not None:
        pre_market_cfg["trend_min_stop_atr_multiple"] = float(min_stop_atr)
    else:
        pre_market_cfg.pop("trend_min_stop_atr_multiple", None)
    if max_entry_adverse_r is not None:
        pre_market_cfg["trend_max_entry_adverse_deviation_r"] = float(max_entry_adverse_r)
    else:
        pre_market_cfg.pop("trend_max_entry_adverse_deviation_r", None)
    if min_first_target_rr is not None:
        pre_market_cfg["trend_min_first_target_rr"] = float(min_first_target_rr)
    else:
        pre_market_cfg.pop("trend_min_first_target_rr", None)
    if second_target_rr_relax_threshold is not None:
        pre_market_cfg["trend_second_target_rr_relax_threshold"] = float(second_target_rr_relax_threshold)
    else:
        pre_market_cfg.pop("trend_second_target_rr_relax_threshold", None)
    if relaxed_first_target_rr is not None:
        pre_market_cfg["trend_relaxed_first_target_rr"] = float(relaxed_first_target_rr)
    else:
        pre_market_cfg.pop("trend_relaxed_first_target_rr", None)
    if entry_confirmation_bars is not None:
        pre_market_cfg["trend_entry_confirmation_bars"] = int(entry_confirmation_bars)
    else:
        pre_market_cfg.pop("trend_entry_confirmation_bars", None)
    if initial_stop_grace_bars is not None:
        pre_market_cfg["trend_initial_stop_grace_bars"] = int(initial_stop_grace_bars)
    else:
        pre_market_cfg.pop("trend_initial_stop_grace_bars", None)
    config["pre_market"] = pre_market_cfg
    return config


def _label(value: float | None) -> str:
    return "legacy" if value is None else str(value)


def _hold_days(trade: TradeRecord) -> float:
    try:
        entry = datetime.fromisoformat(str(trade.entry_time))
        exit_ = datetime.fromisoformat(str(trade.exit_time))
        return max((exit_ - entry).total_seconds() / 86400.0, 0.0)
    except ValueError:
        return float(trade.days_held)


def _early_stop_metrics(trades: list[TradeRecord], *, early_stop_days: float) -> dict[str, Any]:
    stops = [trade for trade in trades if trade.exit_reason == "stop"]
    early_stops = [trade for trade in stops if _hold_days(trade) < float(early_stop_days)]
    early_total = float(sum(float(trade.pnl_ratio) for trade in early_stops))
    return {
        "stop_trades": len(stops),
        "early_stop_days": float(early_stop_days),
        "early_stop_trades": len(early_stops),
        "early_stop_total_pnl": early_total,
        "early_stop_avg_pnl": early_total / len(early_stops) if early_stops else 0.0,
    }


def _write_rows(rows: list[dict[str, object]], output: Path | None) -> None:
    fields = [
        "case",
        "trend_min_stop_atr_multiple",
        "trend_max_entry_adverse_deviation_r",
        "trend_min_first_target_rr",
        "trend_second_target_rr_relax_threshold",
        "trend_relaxed_first_target_rr",
        "trend_entry_confirmation_bars",
        "trend_initial_stop_grace_bars",
        "num_trades",
        "wins",
        "losses",
        "total_pnl",
        "avg_pnl",
        "stop_trades",
        "early_stop_days",
        "early_stop_trades",
        "early_stop_total_pnl",
        "early_stop_avg_pnl",
        "phase3_entry_adverse_deviation_rejects",
        "phase3_entry_confirmation_wait_bars",
        "phase3_initial_stop_grace_skips",
        "trades_opened",
        "phase2_actionable_days",
    ]
    fh = output.open("w", newline="") if output else sys.stdout
    try:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            fh.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    case = _apply_date_overrides(get_case(args.case), start_dt=args.start, end_dt=args.end)
    loaded_config = load_config()
    base_config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=base_config)
    plan_factory = _plan_factory_for_case(case)

    rows: list[dict[str, object]] = []
    for min_stop_atr in _parse_grid(args.trend_min_stop_atr_multiples):
        for max_entry_adverse_r in _parse_grid(args.trend_max_entry_adverse_r_values):
            for min_first_target_rr in _parse_grid(args.trend_min_first_target_rr_values):
                for second_target_rr_relax_threshold in _parse_grid(args.trend_second_target_rr_relax_threshold_values):
                    for relaxed_first_target_rr in _parse_grid(args.trend_relaxed_first_target_rr_values):
                        for entry_confirmation_bars in _parse_grid(args.trend_entry_confirmation_bars_values):
                            for initial_stop_grace_bars in _parse_grid(args.trend_initial_stop_grace_bars_values):
                                config = _config_for_combo(
                                    base_config,
                                    min_stop_atr=min_stop_atr,
                                    max_entry_adverse_r=max_entry_adverse_r,
                                    min_first_target_rr=min_first_target_rr,
                                    second_target_rr_relax_threshold=second_target_rr_relax_threshold,
                                    relaxed_first_target_rr=relaxed_first_target_rr,
                                    entry_confirmation_bars=entry_confirmation_bars,
                                    initial_stop_grace_bars=initial_stop_grace_bars,
                                )
                                result = run_case_from_frames(
                                    case=case,
                                    daily_df=daily_df,
                                    minute_df=minute_df,
                                    plan_factory=plan_factory,
                                    pre_market_cfg=config.get("pre_market") or {},
                                    signal_cfg=config.get("intraday") or {},
                                )
                                summary = summarize_trades(result.trades)
                                early_stop = _early_stop_metrics(result.trades, early_stop_days=args.early_stop_days)
                                diagnostics = result.diagnostics or {}
                                rows.append(
                                    {
                                        "case": case.case_id,
                                        "trend_min_stop_atr_multiple": _label(min_stop_atr),
                                        "trend_max_entry_adverse_deviation_r": _label(max_entry_adverse_r),
                                        "trend_min_first_target_rr": _label(min_first_target_rr),
                                        "trend_second_target_rr_relax_threshold": _label(second_target_rr_relax_threshold),
                                        "trend_relaxed_first_target_rr": _label(relaxed_first_target_rr),
                                        "trend_entry_confirmation_bars": _label(entry_confirmation_bars),
                                        "trend_initial_stop_grace_bars": _label(initial_stop_grace_bars),
                                        "num_trades": summary.get("num_trades", 0),
                                        "wins": summary.get("wins", 0),
                                        "losses": summary.get("losses", 0),
                                        "total_pnl": summary.get("total_pnl", 0.0),
                                        "avg_pnl": summary.get("avg_pnl", 0.0),
                                        "stop_trades": early_stop["stop_trades"],
                                        "early_stop_days": early_stop["early_stop_days"],
                                        "early_stop_trades": early_stop["early_stop_trades"],
                                        "early_stop_total_pnl": early_stop["early_stop_total_pnl"],
                                        "early_stop_avg_pnl": early_stop["early_stop_avg_pnl"],
                                        "phase3_entry_adverse_deviation_rejects": diagnostics.get("phase3_entry_adverse_deviation_rejects", 0),
                                        "phase3_entry_confirmation_wait_bars": diagnostics.get("phase3_entry_confirmation_wait_bars", 0),
                                        "phase3_initial_stop_grace_skips": diagnostics.get("phase3_initial_stop_grace_skips", 0),
                                        "trades_opened": diagnostics.get("trades_opened", 0),
                                        "phase2_actionable_days": diagnostics.get("phase2_actionable_days", 0),
                                    }
                                )
    _write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
