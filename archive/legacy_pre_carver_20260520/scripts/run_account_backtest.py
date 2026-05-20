from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
import math
from pathlib import Path
import time
from typing import Any

import pandas as pd

from backtest.account_runner import (
    AccountBacktestConfig,
    AccountCandidate,
    candidate_from_trade,
    run_account_backtest,
    write_account_outputs,
)
from backtest.cases import get_case_group
from backtest.execution_quality import medium_term_quality_from_daily, visible_daily_before_entry
from backtest.phase23 import load_case_frames_with_tqbacktest, run_case_from_frames
from market.contract_specs import builtin_contract_spec
from run_backtest import (
    _config_with_fundamental_mode,
    _config_with_trend_experiment_overrides,
    _plan_factory_for_case,
    _resolved_fundamental_mode,
)
from shared.config_loader import load_yaml_config


def load_config() -> dict:
    return load_yaml_config(__file__)


def _risk_cap(value: str) -> float | None:
    if value.strip().lower() in {"none", "off"}:
        return None
    return float(value)


def _risk_label(value: float | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run account-level futures backtest from case-group candidates")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--initial-equity", type=float, default=1_000_000.0)
    parser.add_argument("--max-portfolio-margin-pct", type=float, default=0.30)
    parser.add_argument("--risk-cap", type=_risk_cap, default=0.015)
    parser.add_argument("--ordinary-risk-pct", type=float)
    parser.add_argument("--strong-risk-pct", type=float)
    parser.add_argument("--strong-first-target-rr", type=float, default=2.0)
    parser.add_argument("--strong-second-target-rr", type=float, default=3.0)
    parser.add_argument("--execution-policy", choices=("fixed", "conditional"), default="fixed")
    parser.add_argument("--position-budget-source", choices=("phase2_score", "phase1_story"), default="phase2_score")
    parser.add_argument("--split-initial-fraction", type=float, default=1.0)
    parser.add_argument("--split-second-entry-trigger", choices=("none", "tp1"), default="none")
    parser.add_argument("--commission-multiplier", type=float, default=1.01)
    parser.add_argument("--min-replacement-hold-hours", type=float, default=0.0)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--trend-min-stop-atr-multiple", type=float)
    parser.add_argument("--trend-max-entry-adverse-deviation-r", type=float)
    parser.add_argument("--trend-min-first-target-rr", type=float)
    parser.add_argument("--trend-entry-confirmation-bars", type=int)
    parser.add_argument("--trend-initial-stop-grace-bars", type=int)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress timing output")
    return parser


def _contract_spec(config: dict[str, Any], symbol: str) -> dict[str, Any]:
    spec = dict(builtin_contract_spec(symbol))
    pre_cfg = config.get("pre_market") or {}
    specs = pre_cfg.get("contract_specs") or {}
    if isinstance(specs, dict):
        configured = specs.get(symbol)
        if isinstance(configured, dict):
            spec.update(configured)
    position = (config.get("positions") or {}).get(symbol)
    if isinstance(position, dict):
        for key in (
            "multiplier",
            "margin_rate",
            "commission",
            "commission_per_lot",
            "commission_rate",
            "close_today_commission_per_lot",
            "close_today_commission",
        ):
            if position.get(key) is not None:
                spec[key] = position[key]
    if "commission_per_lot" not in spec and spec.get("commission") is not None:
        spec["commission_per_lot"] = spec["commission"]
    if "close_today_commission_per_lot" not in spec and spec.get("close_today_commission") is not None:
        spec["close_today_commission_per_lot"] = spec["close_today_commission"]
    return spec


def _float_spec(spec: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        if spec.get(key) is None:
            return default
        return float(spec[key])
    except (TypeError, ValueError):
        return default


def _candidate_in_year(candidate: AccountCandidate, year: int) -> bool:
    return datetime.fromisoformat(candidate.entry_time).year == int(year)


def _case_for_year(case, year: int):
    return replace(
        case,
        case_id=f"{case.case_id}_{year}",
        start_dt=datetime(int(year), 1, 1).date(),
        end_dt=datetime(int(year), 12, 31).date(),
    )


def _candidate_with_execution_quality(candidate: AccountCandidate, daily_df: pd.DataFrame) -> AccountCandidate:
    visible_daily = visible_daily_before_entry(daily_df, candidate.entry_time)
    quality = medium_term_quality_from_daily(
        visible_daily,
        direction=candidate.direction,
        trend_phase=candidate.trend_phase,
    )
    try:
        quality_score = float(quality.get("medium_term_quality_score"))
    except (TypeError, ValueError):
        return candidate
    if not math.isfinite(quality_score):
        return candidate
    return replace(
        candidate,
        medium_term_quality_score=quality_score,
        medium_term_entry_location_score=float(quality.get("medium_term_entry_location_score") or 0.0),
    )


def collect_account_candidates(
    *,
    case_group: str,
    year: int,
    config: dict,
    progress: bool = False,
) -> tuple[list[AccountCandidate], dict[str, pd.DataFrame]]:
    candidates: list[AccountCandidate] = []
    minute_bars_by_symbol: dict[str, pd.DataFrame] = {}
    for case in get_case_group(case_group):
        started = time.perf_counter()
        case_candidates = 0
        if progress:
            print(f"case {case.case_id}: start", flush=True)
        daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
        if progress:
            elapsed = time.perf_counter() - started
            print(
                f"case {case.case_id}: frames_loaded daily={len(daily_df)} "
                f"minute={len(minute_df)} elapsed={elapsed:.2f}s",
                flush=True,
            )
        minute_bars_by_symbol.setdefault(case.symbol, minute_df.copy())
        run_case = _case_for_year(case, year)
        result = run_case_from_frames(
            case=run_case,
            daily_df=daily_df,
            minute_df=minute_df,
            plan_factory=_plan_factory_for_case(run_case),
            pre_market_cfg=config.get("pre_market") or {},
            signal_cfg=config.get("intraday") or {},
        )
        spec = _contract_spec(config, case.symbol)
        close_today_commission_per_lot = (
            _float_spec(spec, "close_today_commission_per_lot")
            if spec.get("close_today_commission_per_lot") is not None
            else None
        )
        for trade in result.trades:
            candidate = candidate_from_trade(
                trade,
                name=case.name,
                commission_per_lot=_float_spec(spec, "commission_per_lot"),
                commission_rate=_float_spec(spec, "commission_rate"),
                close_today_commission_per_lot=close_today_commission_per_lot,
                margin_rate=_float_spec(spec, "margin_rate") if spec.get("margin_rate") is not None else None,
            )
            candidate = _candidate_with_execution_quality(candidate, daily_df)
            if _candidate_in_year(candidate, year):
                candidates.append(candidate)
                case_candidates += 1
        if progress:
            elapsed = time.perf_counter() - started
            print(
                f"case {case.case_id}: candidates={case_candidates} "
                f"trades={len(result.trades)} elapsed={elapsed:.2f}s",
                flush=True,
            )
    return candidates, minute_bars_by_symbol


def _price_lookup_from_minute_bars(
    minute_bars_by_symbol: dict[str, pd.DataFrame],
):
    normalized: dict[str, pd.DataFrame] = {}
    for symbol, df in minute_bars_by_symbol.items():
        if df.empty or "datetime" not in df or "close" not in df:
            continue
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"])
        normalized[symbol] = out.sort_values("datetime", kind="stable")

    def lookup(symbol: str, at: datetime) -> float | None:
        df = normalized.get(symbol)
        if df is None or df.empty:
            return None
        rows = df.loc[df["datetime"] <= pd.Timestamp(at)]
        if rows.empty:
            return None
        return float(rows.iloc[-1]["close"])

    return lookup


def _default_output_prefix(*, year: int, risk_cap: float | None) -> Path:
    return Path("data/reports/backtest") / f"trend_account_{year}_phase2score_margin30_risk_{_risk_label(risk_cap)}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    loaded_config = load_config()
    config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    config = _config_with_trend_experiment_overrides(config, args)
    if not args.quiet:
        print(f"collecting candidates: case_group={args.case_group} year={args.year}", flush=True)
    started = time.perf_counter()
    collected = collect_account_candidates(
        case_group=args.case_group,
        year=args.year,
        config=config,
        progress=not args.quiet,
    )
    if isinstance(collected, tuple):
        candidates, minute_bars_by_symbol = collected
    else:
        candidates = collected
        minute_bars_by_symbol = {}
    if not args.quiet:
        elapsed = time.perf_counter() - started
        print(f"collected candidates: {len(candidates)} elapsed={elapsed:.2f}s", flush=True)
    output_prefix = args.output_prefix or _default_output_prefix(year=args.year, risk_cap=args.risk_cap)
    if not args.quiet:
        print("running account constraints", flush=True)
    result = run_account_backtest(
        candidates,
        AccountBacktestConfig(
            initial_equity=args.initial_equity,
            max_portfolio_margin_pct=args.max_portfolio_margin_pct,
            position_budget_source=args.position_budget_source,
            risk_per_trade_pct=args.risk_cap,
            ordinary_risk_per_trade_pct=args.ordinary_risk_pct,
            strong_risk_per_trade_pct=args.strong_risk_pct,
            strong_first_target_rr=args.strong_first_target_rr,
            strong_second_target_rr=args.strong_second_target_rr,
            execution_policy=args.execution_policy,
            split_initial_fraction=args.split_initial_fraction,
            split_second_entry_trigger=args.split_second_entry_trigger,
            commission_multiplier=args.commission_multiplier,
            min_replacement_hold_hours=args.min_replacement_hold_hours,
            scope=f"{args.case_group} continuous {args.year}, long+short candidates, one active trade per symbol",
        ),
        price_lookup=_price_lookup_from_minute_bars(minute_bars_by_symbol),
    )
    if not args.quiet:
        print(
            f"account result: accepted={result.summary['accepted_trades']} "
            f"skipped={result.summary['skipped_trades']} "
            f"return={float(result.summary['return_pct']):.2%}",
            flush=True,
        )
    for path in write_account_outputs(result, output_prefix):
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
