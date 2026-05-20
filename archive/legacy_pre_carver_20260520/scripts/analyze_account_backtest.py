from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.account_diagnostics import (
    analyze_account_report,
    load_risk_sensitivity,
    load_trade_rows,
    render_markdown,
)


DEFAULT_TRADES = Path("data/reports/backtest/trend_account_2025_phase2score_margin30_risk_0p015_trades.csv")
DEFAULT_RISK_SENSITIVITY = Path("data/reports/backtest/trend_account_2025_phase2score_margin30_risk_sensitivity.json")
DEFAULT_DAILY_CACHE_DIR = Path("data/cache/backtest")


def _risk_cap(value: str) -> float | str:
    if value == "none":
        return value
    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze account-level backtest trade diagnostics")
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--risk-sensitivity", type=Path, default=DEFAULT_RISK_SENSITIVITY)
    parser.add_argument("--risk-cap", type=_risk_cap, default=0.015)
    parser.add_argument("--slippage-bps-per-side", type=float, default=0.0)
    parser.add_argument("--daily-cache-dir", type=Path, default=DEFAULT_DAILY_CACHE_DIR)
    parser.add_argument("--trend-lookahead-days", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _daily_cache_candidates(cache_dir: Path, symbol: str) -> list[Path]:
    files = list(cache_dir.glob(f"{symbol.lower()}_*_daily.parquet"))
    return sorted(
        files,
        key=lambda path: (
            "20220101_20251231" not in path.name,
            "v3" not in path.name,
            path.name,
        ),
    )


def _load_daily_bars(cache_dir: Path, trades: list[dict[str, object]]) -> dict[str, pd.DataFrame]:
    if not cache_dir.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted({str(row.get("symbol") or "") for row in trades if row.get("symbol")}):
        candidates = _daily_cache_candidates(cache_dir, symbol)
        if not candidates:
            continue
        try:
            out[symbol] = pd.read_parquet(candidates[0])
        except Exception:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trades = load_trade_rows(args.trades)
    risk_summary = None
    if args.risk_sensitivity and args.risk_sensitivity.exists():
        risk_summary = load_risk_sensitivity(args.risk_sensitivity, risk_cap=args.risk_cap)

    analysis = analyze_account_report(
        trades,
        risk_summary=risk_summary,
        slippage_bps_per_side=args.slippage_bps_per_side,
        daily_bars_by_symbol=_load_daily_bars(args.daily_cache_dir, trades),
        trend_lookahead_days=args.trend_lookahead_days,
    )
    markdown = render_markdown(analysis)

    if args.output_json:
        _write_text(args.output_json, json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    if args.output_md:
        _write_text(args.output_md, markdown)
    if not args.output_json and not args.output_md:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
