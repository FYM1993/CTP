from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.account_runner import (
    AccountBacktestConfig,
    AccountBacktestResult,
    AccountCandidate,
    candidate_from_trade,
    run_account_backtest,
    write_account_outputs,
)
from backtest.cases import get_case_group
from backtest.experiment_cache import (
    build_phase23_experiment_cache,
    phase23_cache_matches_config,
    phase23_experiment_cache_path,
    read_phase23_experiment_cache,
    write_phase23_experiment_cache,
)
from backtest.models import BacktestCase
from backtest.phase23 import load_case_frames_with_tqbacktest, run_case_from_frames
from compare_trend_backtest_params import _config_for_combo, _label, _parse_grid
from run_account_backtest import (
    _candidate_with_execution_quality,
    _candidate_in_year,
    _case_for_year,
    _config_with_fundamental_mode,
    _contract_spec,
    _float_spec,
    _price_lookup_from_minute_bars,
    _resolved_fundamental_mode,
    _risk_cap,
    _risk_label,
    load_config,
)
from run_backtest import _plan_factory_for_case


@dataclass(frozen=True, slots=True)
class LoadedTrendCase:
    case: BacktestCase
    run_case: BacktestCase
    daily_df: pd.DataFrame
    minute_df: pd.DataFrame
    plan_factory: Any


SUMMARY_FIELDS = [
    "combo_id",
    "case_group",
    "year",
    "risk_cap",
    "max_portfolio_margin_pct",
    "ordinary_risk_per_trade_pct",
    "strong_risk_per_trade_pct",
    "strong_first_target_rr",
    "strong_second_target_rr",
    "execution_policy",
    "split_initial_fraction",
    "split_second_entry_trigger",
    "trend_min_stop_atr_multiple",
    "trend_max_entry_adverse_deviation_r",
    "trend_min_first_target_rr",
    "trend_second_target_rr_relax_threshold",
    "trend_relaxed_first_target_rr",
    "trend_entry_confirmation_bars",
    "trend_initial_stop_grace_bars",
    "initial_equity",
    "final_equity",
    "net_profit",
    "return_pct",
    "max_drawdown_realized_pct",
    "max_drawdown_start",
    "max_drawdown_end",
    "accepted_trades",
    "closed_trades",
    "candidate_trades",
    "skipped_trades",
    "replacements",
    "wins",
    "losses",
    "win_rate",
    "total_gross_pnl",
    "total_fees",
    "split_target_lots",
    "split_first_entry_lots",
    "split_planned_second_entry_lots",
    "split_second_entry_filled_lots",
    "split_second_entry_unfilled_lots",
    "execution_profile_counts",
    "max_margin_pct_observed",
    "max_initial_stop_risk_pct_equity",
    "avg_initial_stop_risk_pct_equity",
    "stop_trades",
    "stop_net_pnl",
    "early_stop_days",
    "early_stop_trades",
    "early_stop_net_pnl",
    "early_stop_avg_net_pnl",
    "phase3_entry_adverse_deviation_rejects",
    "phase3_entry_confirmation_wait_bars",
    "phase3_initial_stop_grace_skips",
    "trades_opened",
    "phase2_actionable_days",
    "phase23_cache_hits",
    "phase23_cache_misses",
    "skipped_reasons",
]

BY_SYMBOL_FIELDS = [
    "combo_id",
    "case_group",
    "year",
    "risk_cap",
    "max_portfolio_margin_pct",
    "ordinary_risk_per_trade_pct",
    "strong_risk_per_trade_pct",
    "strong_first_target_rr",
    "strong_second_target_rr",
    "execution_policy",
    "split_initial_fraction",
    "split_second_entry_trigger",
    "trend_min_stop_atr_multiple",
    "trend_max_entry_adverse_deviation_r",
    "trend_min_first_target_rr",
    "trend_second_target_rr_relax_threshold",
    "trend_relaxed_first_target_rr",
    "trend_entry_confirmation_bars",
    "trend_initial_stop_grace_bars",
    "symbol",
    "name",
    "trades",
    "wins",
    "win_rate",
    "net_pnl",
    "return_contribution_pct",
    "fees",
]

DIAGNOSTIC_FIELDS = [
    "phase3_entry_adverse_deviation_rejects",
    "phase3_entry_confirmation_wait_bars",
    "phase3_initial_stop_grace_skips",
    "trades_opened",
    "phase2_actionable_days",
    "phase23_cache_hits",
    "phase23_cache_misses",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fast account-level trend parameter sweep")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--case-limit", type=int, help="Experiment scaling: only run the first N cases in the group")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--initial-equity", type=float, default=1_000_000.0)
    parser.add_argument("--max-portfolio-margin-pct", type=float, default=0.30)
    parser.add_argument("--max-portfolio-margin-pct-values")
    parser.add_argument("--risk-cap", type=_risk_cap, default=0.015)
    parser.add_argument("--ordinary-risk-pct-values", default="legacy")
    parser.add_argument("--strong-risk-pct-values", default="legacy")
    parser.add_argument("--strong-first-target-rr-values", default="legacy")
    parser.add_argument("--strong-second-target-rr-values", default="legacy")
    parser.add_argument("--execution-policy-values", default="fixed")
    parser.add_argument("--split-initial-fraction-values", default="legacy")
    parser.add_argument("--split-second-entry-trigger-values", default="none")
    parser.add_argument("--commission-multiplier", type=float, default=1.01)
    parser.add_argument("--min-replacement-hold-hours", type=float, default=0.0)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--trend-min-stop-atr-multiples", default="legacy")
    parser.add_argument("--trend-max-entry-adverse-r-values", default="legacy")
    parser.add_argument("--trend-min-first-target-rr-values", default="legacy")
    parser.add_argument("--trend-second-target-rr-relax-threshold-values", default="legacy")
    parser.add_argument("--trend-relaxed-first-target-rr-values", default="legacy")
    parser.add_argument("--trend-entry-confirmation-bars-values", default="legacy")
    parser.add_argument("--trend-initial-stop-grace-bars-values", default="legacy")
    parser.add_argument("--early-stop-days", type=float, default=3.0)
    parser.add_argument("--phase23-cache-dir", type=Path)
    parser.add_argument("--write-missing-phase23-cache", action="store_true")
    parser.add_argument("--require-phase23-cache", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--write-detail-outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def _default_output_prefix(*, year: int, risk_cap: float | None) -> Path:
    return Path("data/reports/backtest") / (
        f"trend_account_{year}_phase2score_margin30_risk_{_risk_label(risk_cap)}_trend_param_sweep"
    )


def _parse_string_grid(raw: str) -> list[str | None]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        return [None]
    return [None if value.lower() == "legacy" else value for value in values]


def _combo_rows(args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    portfolio_margin_values = (
        _parse_grid(args.max_portfolio_margin_pct_values)
        if args.max_portfolio_margin_pct_values
        else [float(args.max_portfolio_margin_pct)]
    )
    for (
        max_portfolio_margin_pct,
        ordinary_risk_pct,
        strong_risk_pct,
        strong_first_target_rr,
        strong_second_target_rr,
        execution_policy,
        split_initial_fraction,
        split_second_entry_trigger,
        min_stop_atr,
        max_entry_adverse_r,
        min_first_target_rr,
        second_target_rr_relax_threshold,
        relaxed_first_target_rr,
        entry_confirmation_bars,
        initial_stop_grace_bars,
    ) in product(
        portfolio_margin_values,
        _parse_grid(args.ordinary_risk_pct_values),
        _parse_grid(args.strong_risk_pct_values),
        _parse_grid(args.strong_first_target_rr_values),
        _parse_grid(args.strong_second_target_rr_values),
        _parse_string_grid(args.execution_policy_values),
        _parse_grid(args.split_initial_fraction_values),
        _parse_string_grid(args.split_second_entry_trigger_values),
        _parse_grid(args.trend_min_stop_atr_multiples),
        _parse_grid(args.trend_max_entry_adverse_r_values),
        _parse_grid(args.trend_min_first_target_rr_values),
        _parse_grid(args.trend_second_target_rr_relax_threshold_values),
        _parse_grid(args.trend_relaxed_first_target_rr_values),
        _parse_grid(args.trend_entry_confirmation_bars_values),
        _parse_grid(args.trend_initial_stop_grace_bars_values),
    ):
        rows.append(
            {
                "max_portfolio_margin_pct": max_portfolio_margin_pct,
                "ordinary_risk_pct": ordinary_risk_pct,
                "strong_risk_pct": strong_risk_pct,
                "strong_first_target_rr": strong_first_target_rr,
                "strong_second_target_rr": strong_second_target_rr,
                "execution_policy": str(execution_policy or "fixed"),
                "split_initial_fraction": split_initial_fraction,
                "split_second_entry_trigger": split_second_entry_trigger,
                "min_stop_atr": min_stop_atr,
                "max_entry_adverse_r": max_entry_adverse_r,
                "min_first_target_rr": min_first_target_rr,
                "second_target_rr_relax_threshold": second_target_rr_relax_threshold,
                "relaxed_first_target_rr": relaxed_first_target_rr,
                "entry_confirmation_bars": entry_confirmation_bars,
                "initial_stop_grace_bars": initial_stop_grace_bars,
            }
        )
    return rows


def _combo_labels(combo: dict[str, Any]) -> dict[str, str]:
    def bar_label(value: float | None) -> str:
        return "legacy" if value is None else str(int(value))

    return {
        "max_portfolio_margin_pct": _label(combo["max_portfolio_margin_pct"]),
        "ordinary_risk_per_trade_pct": _label(combo["ordinary_risk_pct"]),
        "strong_risk_per_trade_pct": _label(combo["strong_risk_pct"]),
        "strong_first_target_rr": _label(combo["strong_first_target_rr"]),
        "strong_second_target_rr": _label(combo["strong_second_target_rr"]),
        "execution_policy": str(combo["execution_policy"] or "fixed"),
        "split_initial_fraction": _label(combo["split_initial_fraction"]),
        "split_second_entry_trigger": "legacy"
        if combo["split_second_entry_trigger"] is None
        else str(combo["split_second_entry_trigger"]),
        "trend_min_stop_atr_multiple": _label(combo["min_stop_atr"]),
        "trend_max_entry_adverse_deviation_r": _label(combo["max_entry_adverse_r"]),
        "trend_min_first_target_rr": _label(combo["min_first_target_rr"]),
        "trend_second_target_rr_relax_threshold": _label(combo["second_target_rr_relax_threshold"]),
        "trend_relaxed_first_target_rr": _label(combo["relaxed_first_target_rr"]),
        "trend_entry_confirmation_bars": bar_label(combo["entry_confirmation_bars"]),
        "trend_initial_stop_grace_bars": bar_label(combo["initial_stop_grace_bars"]),
    }


def load_trend_case_frames_once(
    *,
    case_group: str,
    year: int,
    config: dict[str, Any],
    case_limit: int | None = None,
    progress: bool = False,
) -> tuple[list[LoadedTrendCase], dict[str, pd.DataFrame]]:
    loaded: list[LoadedTrendCase] = []
    minute_bars_by_symbol: dict[str, pd.DataFrame] = {}
    cases = get_case_group(case_group)
    if case_limit is not None:
        cases = cases[: max(int(case_limit), 0)]
    for case in cases:
        started = time.perf_counter()
        if progress:
            print(f"case {case.case_id}: load frames", flush=True)
        daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
        run_case = _case_for_year(case, year)
        loaded.append(
            LoadedTrendCase(
                case=case,
                run_case=run_case,
                daily_df=daily_df,
                minute_df=minute_df,
                plan_factory=_plan_factory_for_case(run_case),
            )
        )
        minute_bars_by_symbol.setdefault(case.symbol, minute_df.copy())
        if progress:
            elapsed = time.perf_counter() - started
            print(
                f"case {case.case_id}: loaded daily={len(daily_df)} minute={len(minute_df)} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
    return loaded, minute_bars_by_symbol


def _collect_combo_candidates(
    *,
    loaded_cases: list[LoadedTrendCase],
    year: int,
    config: dict[str, Any],
    phase23_cache_dir: Path | None = None,
    write_missing_phase23_cache: bool = False,
    require_phase23_cache: bool = False,
) -> tuple[list[AccountCandidate], Counter[str]]:
    candidates: list[AccountCandidate] = []
    diagnostics: Counter[str] = Counter()
    pre_market_cfg = config.get("pre_market") or {}
    signal_cfg = config.get("intraday") or {}
    for loaded in loaded_cases:
        result = None
        cache_path: Path | None = None
        if phase23_cache_dir is not None:
            cache_path = phase23_experiment_cache_path(
                phase23_cache_dir,
                case=loaded.run_case,
                pre_market_cfg=pre_market_cfg,
                signal_cfg=signal_cfg,
            )
            if cache_path.exists():
                cache = read_phase23_experiment_cache(cache_path)
                if phase23_cache_matches_config(
                    cache,
                    case=loaded.run_case,
                    pre_market_cfg=pre_market_cfg,
                    signal_cfg=signal_cfg,
                ):
                    result = cache.result
                    diagnostics["phase23_cache_hits"] += 1
                elif require_phase23_cache:
                    raise ValueError(f"Phase23 cache config mismatch: {cache_path}")
            if result is None:
                diagnostics["phase23_cache_misses"] += 1
                if require_phase23_cache:
                    raise FileNotFoundError(f"Phase23 cache missing: {cache_path}")
        if result is None:
            result = run_case_from_frames(
                case=loaded.run_case,
                daily_df=loaded.daily_df,
                minute_df=loaded.minute_df,
                plan_factory=loaded.plan_factory,
                pre_market_cfg=pre_market_cfg,
                signal_cfg=signal_cfg,
            )
            if cache_path is not None and write_missing_phase23_cache:
                write_phase23_experiment_cache(
                    build_phase23_experiment_cache(
                        case=loaded.run_case,
                        result=result,
                        pre_market_cfg=pre_market_cfg,
                        signal_cfg=signal_cfg,
                    ),
                    cache_path,
                )
        for key in DIAGNOSTIC_FIELDS:
            diagnostics[key] += int((result.diagnostics or {}).get(key, 0) or 0)
        spec = _contract_spec(config, loaded.case.symbol)
        close_today_commission_per_lot = (
            _float_spec(spec, "close_today_commission_per_lot")
            if spec.get("close_today_commission_per_lot") is not None
            else None
        )
        for trade in result.trades:
            candidate = candidate_from_trade(
                trade,
                name=loaded.case.name,
                commission_per_lot=_float_spec(spec, "commission_per_lot"),
                commission_rate=_float_spec(spec, "commission_rate"),
                close_today_commission_per_lot=close_today_commission_per_lot,
                margin_rate=_float_spec(spec, "margin_rate") if spec.get("margin_rate") is not None else None,
            )
            candidate = _candidate_with_execution_quality(candidate, loaded.daily_df)
            if _candidate_in_year(candidate, year):
                candidates.append(candidate)
    return candidates, diagnostics


def _hold_days(trade: dict[str, Any]) -> float:
    try:
        entry = datetime.fromisoformat(str(trade["entry_time"]))
        exit_ = datetime.fromisoformat(str(trade["actual_exit_time"]))
        return max((exit_ - entry).total_seconds() / 86400.0, 0.0)
    except (KeyError, ValueError):
        return 0.0


def _account_early_stop_metrics(result: AccountBacktestResult, *, early_stop_days: float) -> dict[str, Any]:
    stops = [trade for trade in result.trades if str(trade.get("actual_exit_reason")) == "stop"]
    early_stops = [trade for trade in stops if _hold_days(trade) < float(early_stop_days)]
    stop_net_pnl = float(sum(float(trade.get("net_pnl") or 0.0) for trade in stops))
    early_net_pnl = float(sum(float(trade.get("net_pnl") or 0.0) for trade in early_stops))
    return {
        "stop_trades": len(stops),
        "stop_net_pnl": stop_net_pnl,
        "early_stop_days": float(early_stop_days),
        "early_stop_trades": len(early_stops),
        "early_stop_net_pnl": early_net_pnl,
        "early_stop_avg_net_pnl": early_net_pnl / len(early_stops) if early_stops else 0.0,
    }


def _account_scope(case_group: str, year: int) -> str:
    return f"{case_group} continuous {year}, long+short candidates, one active trade per symbol"


def _combo_or_default(value: float | None, default: float) -> float:
    return float(default if value is None else value)


def _summary_row(
    *,
    combo_id: int,
    case_group: str,
    year: int,
    risk_cap: float | None,
    combo_labels: dict[str, str],
    result: AccountBacktestResult,
    diagnostics: Counter[str],
    early_stop: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "combo_id": int(combo_id),
        "case_group": case_group,
        "year": int(year),
        "risk_cap": "none" if risk_cap is None else float(risk_cap),
        **combo_labels,
    }
    for field in SUMMARY_FIELDS:
        if field in row or field in combo_labels:
            continue
        if field in early_stop:
            row[field] = early_stop[field]
        elif field in DIAGNOSTIC_FIELDS:
            row[field] = int(diagnostics[field])
        elif field == "skipped_reasons":
            row[field] = result.summary.get(field, {})
        else:
            row[field] = result.summary.get(field, "")
    return row


def _by_symbol_rows(
    *,
    combo_id: int,
    case_group: str,
    year: int,
    risk_cap: float | None,
    combo_labels: dict[str, str],
    result: AccountBacktestResult,
) -> list[dict[str, Any]]:
    prefix = {
        "combo_id": int(combo_id),
        "case_group": case_group,
        "year": int(year),
        "risk_cap": "none" if risk_cap is None else float(risk_cap),
        **combo_labels,
    }
    return [{**prefix, **row} for row in result.by_symbol]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def write_sweep_outputs(
    *,
    output_prefix: Path,
    summary_rows: list[dict[str, Any]],
    by_symbol_rows: list[dict[str, Any]],
    detail_results: list[tuple[int, AccountBacktestResult]],
) -> list[Path]:
    summary_csv = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    summary_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    by_symbol_csv = output_prefix.with_name(f"{output_prefix.name}_by_symbol.csv")
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    _write_json(summary_json, summary_rows)
    _write_csv(by_symbol_csv, by_symbol_rows, BY_SYMBOL_FIELDS)
    written = [summary_csv, summary_json, by_symbol_csv]
    for combo_id, result in detail_results:
        detail_prefix = output_prefix.with_name(f"{output_prefix.name}_combo_{combo_id:03d}")
        written.extend(write_account_outputs(result, detail_prefix))
    return written


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    loaded_config = load_config()
    base_config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    combos = _combo_rows(args)
    if not args.quiet:
        print(
            f"loading frames once: case_group={args.case_group} year={args.year} "
            f"combos={len(combos)} case_limit={args.case_limit or 'all'}",
            flush=True,
        )
    loaded_cases, minute_bars_by_symbol = load_trend_case_frames_once(
        case_group=args.case_group,
        year=args.year,
        config=base_config,
        case_limit=args.case_limit,
        progress=not args.quiet,
    )
    price_lookup = _price_lookup_from_minute_bars(minute_bars_by_symbol)
    summary_rows: list[dict[str, Any]] = []
    by_symbol_rows: list[dict[str, Any]] = []
    detail_results: list[tuple[int, AccountBacktestResult]] = []
    for combo_id, combo in enumerate(combos, start=1):
        config = _config_for_combo(
            base_config,
            min_stop_atr=combo["min_stop_atr"],
            max_entry_adverse_r=combo["max_entry_adverse_r"],
            min_first_target_rr=combo["min_first_target_rr"],
            second_target_rr_relax_threshold=combo["second_target_rr_relax_threshold"],
            relaxed_first_target_rr=combo["relaxed_first_target_rr"],
            entry_confirmation_bars=combo["entry_confirmation_bars"],
            initial_stop_grace_bars=combo["initial_stop_grace_bars"],
        )
        labels = _combo_labels(combo)
        if not args.quiet:
            print(f"combo {combo_id}/{len(combos)}: {labels}", flush=True)
        candidates, diagnostics = _collect_combo_candidates(
            loaded_cases=loaded_cases,
            year=args.year,
            config=config,
            phase23_cache_dir=args.phase23_cache_dir,
            write_missing_phase23_cache=args.write_missing_phase23_cache,
            require_phase23_cache=args.require_phase23_cache,
        )
        result = run_account_backtest(
            candidates,
            AccountBacktestConfig(
                initial_equity=args.initial_equity,
                max_portfolio_margin_pct=_combo_or_default(combo["max_portfolio_margin_pct"], args.max_portfolio_margin_pct),
                risk_per_trade_pct=args.risk_cap,
                ordinary_risk_per_trade_pct=combo["ordinary_risk_pct"],
                strong_risk_per_trade_pct=combo["strong_risk_pct"],
                strong_first_target_rr=_combo_or_default(combo["strong_first_target_rr"], 2.0),
                strong_second_target_rr=_combo_or_default(combo["strong_second_target_rr"], 3.0),
                execution_policy=str(combo["execution_policy"] or "fixed"),
                split_initial_fraction=_combo_or_default(combo["split_initial_fraction"], 1.0),
                split_second_entry_trigger=str(combo["split_second_entry_trigger"] or "none"),
                commission_multiplier=args.commission_multiplier,
                min_replacement_hold_hours=args.min_replacement_hold_hours,
                scope=_account_scope(args.case_group, args.year),
            ),
            price_lookup=price_lookup,
        )
        early_stop = _account_early_stop_metrics(result, early_stop_days=args.early_stop_days)
        summary_rows.append(
            _summary_row(
                combo_id=combo_id,
                case_group=args.case_group,
                year=args.year,
                risk_cap=args.risk_cap,
                combo_labels=labels,
                result=result,
                diagnostics=diagnostics,
                early_stop=early_stop,
            )
        )
        by_symbol_rows.extend(
            _by_symbol_rows(
                combo_id=combo_id,
                case_group=args.case_group,
                year=args.year,
                risk_cap=args.risk_cap,
                combo_labels=labels,
                result=result,
            )
        )
        if args.write_detail_outputs:
            detail_results.append((combo_id, result))
        if not args.quiet:
            print(
                f"combo {combo_id}: candidates={result.summary['candidate_trades']} "
                f"accepted={result.summary['accepted_trades']} "
                f"return={float(result.summary['return_pct']):.2%} "
                f"early_stop={early_stop['early_stop_trades']}",
                flush=True,
            )

    output_prefix = args.output_prefix or _default_output_prefix(year=args.year, risk_cap=args.risk_cap)
    written = write_sweep_outputs(
        output_prefix=output_prefix,
        summary_rows=summary_rows,
        by_symbol_rows=by_symbol_rows,
        detail_results=detail_results,
    )
    for path in written:
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
