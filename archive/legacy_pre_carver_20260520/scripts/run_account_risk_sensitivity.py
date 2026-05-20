from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from backtest.account_runner import AccountBacktestConfig, AccountBacktestResult, run_account_backtest, write_account_outputs
from run_account_backtest import (
    _config_with_fundamental_mode,
    _config_with_trend_experiment_overrides,
    _price_lookup_from_minute_bars,
    _resolved_fundamental_mode,
    _risk_cap,
    _risk_label,
    collect_account_candidates,
    load_config,
)


DEFAULT_RISK_CAPS = (0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, None)

SUMMARY_FIELDS = [
    "risk_cap",
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
    "max_margin_pct_observed",
    "max_initial_stop_risk_pct_equity",
    "avg_initial_stop_risk_pct_equity",
    "skipped_reasons",
]

BY_SYMBOL_FIELDS = [
    "risk_cap",
    "symbol",
    "name",
    "trades",
    "wins",
    "win_rate",
    "net_pnl",
    "return_contribution_pct",
    "fees",
]


def _risk_output_value(value: float | None) -> float | str:
    return "none" if value is None else float(value)


def _parse_risk_caps(value: str) -> list[float | None]:
    caps: list[float | None] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        caps.append(_risk_cap(stripped))
    if not caps:
        raise argparse.ArgumentTypeError("at least one risk cap is required")
    return caps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run account-level risk-cap sensitivity backtests")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--initial-equity", type=float, default=1_000_000.0)
    parser.add_argument("--max-portfolio-margin-pct", type=float, default=0.30)
    parser.add_argument("--risk-caps", type=_parse_risk_caps, default=list(DEFAULT_RISK_CAPS))
    parser.add_argument("--commission-multiplier", type=float, default=1.01)
    parser.add_argument("--min-replacement-hold-hours", type=float, default=0.0)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--trend-min-stop-atr-multiple", type=float)
    parser.add_argument("--trend-max-entry-adverse-deviation-r", type=float)
    parser.add_argument("--trend-min-first-target-rr", type=float)
    parser.add_argument("--trend-entry-confirmation-bars", type=int)
    parser.add_argument("--trend-initial-stop-grace-bars", type=int)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--skip-detail-outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def _default_output_prefix(*, year: int) -> Path:
    return Path("data/reports/backtest") / f"trend_account_{year}_phase2score_margin30"


def _summary_row(result: AccountBacktestResult, risk_cap: float | None) -> dict[str, Any]:
    row = {field: result.summary.get(field, "") for field in SUMMARY_FIELDS}
    row["risk_cap"] = _risk_output_value(risk_cap)
    return row


def _by_symbol_rows(result: AccountBacktestResult, risk_cap: float | None) -> list[dict[str, Any]]:
    cap = _risk_output_value(risk_cap)
    return [{"risk_cap": cap, **row} for row in result.by_symbol]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


def write_risk_sensitivity_outputs(
    *,
    output_prefix: Path,
    results: list[tuple[float | None, AccountBacktestResult]],
    write_detail_outputs: bool,
) -> list[Path]:
    summary_rows: list[dict[str, Any]] = []
    by_symbol_rows: list[dict[str, Any]] = []
    written: list[Path] = []
    for risk_cap, result in results:
        summary_rows.append(_summary_row(result, risk_cap))
        by_symbol_rows.extend(_by_symbol_rows(result, risk_cap))
        if write_detail_outputs:
            detail_prefix = output_prefix.with_name(f"{output_prefix.name}_risk_{_risk_label(risk_cap)}")
            written.extend(write_account_outputs(result, detail_prefix))

    csv_path = output_prefix.with_name(f"{output_prefix.name}_risk_sensitivity.csv")
    json_path = output_prefix.with_name(f"{output_prefix.name}_risk_sensitivity.json")
    by_symbol_path = output_prefix.with_name(f"{output_prefix.name}_risk_sensitivity_by_symbol.csv")
    _write_csv(csv_path, summary_rows, SUMMARY_FIELDS)
    _write_json(json_path, summary_rows)
    _write_csv(by_symbol_path, by_symbol_rows, BY_SYMBOL_FIELDS)
    return [csv_path, json_path, by_symbol_path, *written]


def _account_scope(case_group: str, year: int) -> str:
    return f"{case_group} continuous {year}, long+short candidates, one active trade per symbol"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    loaded_config = load_config()
    config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    config = _config_with_trend_experiment_overrides(config, args)
    if not args.quiet:
        print(
            f"collecting candidates once: case_group={args.case_group} year={args.year} "
            f"risk_caps={','.join(_risk_label(cap) for cap in args.risk_caps)}",
            flush=True,
        )
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
    price_lookup = _price_lookup_from_minute_bars(minute_bars_by_symbol)
    results: list[tuple[float | None, AccountBacktestResult]] = []
    for risk_cap in args.risk_caps:
        result = run_account_backtest(
            candidates,
            AccountBacktestConfig(
                initial_equity=args.initial_equity,
                max_portfolio_margin_pct=args.max_portfolio_margin_pct,
                risk_per_trade_pct=risk_cap,
                commission_multiplier=args.commission_multiplier,
                min_replacement_hold_hours=args.min_replacement_hold_hours,
                scope=_account_scope(args.case_group, args.year),
            ),
            price_lookup=price_lookup,
        )
        results.append((risk_cap, result))
        if not args.quiet:
            print(
                f"risk={_risk_label(risk_cap)} accepted={result.summary['accepted_trades']} "
                f"skipped={result.summary['skipped_trades']} "
                f"return={float(result.summary['return_pct']):.2%}",
                flush=True,
            )

    output_prefix = args.output_prefix or _default_output_prefix(year=args.year)
    written = write_risk_sensitivity_outputs(
        output_prefix=output_prefix,
        results=results,
        write_detail_outputs=not args.skip_detail_outputs,
    )
    for path in written:
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
