from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.data_audit import audit_source_directory, write_data_source_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit systematic futures data source coverage and schema")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--backtest-cache-dir", default=str(ROOT / "data" / "cache" / "backtest"))
    parser.add_argument("--tq-main-dir", default=str(ROOT / "data" / "systematic" / "tq_main_daily"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "reports" / "systematic_trading"))
    parser.add_argument("--historical-min-start", default="2022-01-03")
    parser.add_argument("--historical-min-end", default="2025-12-30")
    parser.add_argument("--historical-min-rows", type=int, default=900)
    parser.add_argument("--current-min-start", default="2022-01-03")
    parser.add_argument("--current-min-end", default="2026-05-19")
    parser.add_argument("--current-min-rows", type=int, default=260)
    parser.add_argument("--min-tradeable-symbols", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = [
        audit_source_directory(
            source_name="systematic_tq_main_daily",
            directory=Path(args.tq_main_dir),
            pattern="*_daily.parquet",
            min_start=pd.Timestamp(args.historical_min_start),
            min_end=pd.Timestamp(args.current_min_end),
            min_rows=args.historical_min_rows,
        ),
        audit_source_directory(
            source_name="live_daily_cache",
            directory=Path(args.cache_dir),
            pattern="daily_*_*.parquet",
            min_start=pd.Timestamp(args.current_min_start),
            min_end=pd.Timestamp(args.current_min_end),
            min_rows=args.current_min_rows,
        ),
        audit_source_directory(
            source_name="historical_backtest_cache",
            directory=Path(args.backtest_cache_dir),
            pattern="*_daily.parquet",
            min_start=pd.Timestamp(args.historical_min_start),
            min_end=pd.Timestamp(args.historical_min_end),
            min_rows=args.historical_min_rows,
        ),
    ]
    for audit in audits:
        audit.files.to_csv(output_dir / f"{audit.source_name}_audit.csv", index=False)
    report_path = output_dir / "data_source_audit.md"
    write_data_source_report(report_path, audits, min_tradeable_symbols=args.min_tradeable_symbols)
    print(f"Wrote data source audit to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
