from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.backtest import run_backtest  # noqa: E402
from systematic_trading.contracts import InstrumentConfig, SystemConfig  # noqa: E402
from systematic_trading.data import audit_market_data, load_market_data_dir  # noqa: E402
from systematic_trading.forecasts import build_price_forecasts, combine_forecasts, default_forecast_specs  # noqa: E402
from systematic_trading.reports import write_markdown_report  # noqa: E402
from systematic_trading.risk import calculate_daily_cash_volatility  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run systematic futures portfolio backtest")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "systematic" / "tq_main_daily"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "reports" / "systematic_trading"))
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--max-margin", type=float, default=0.30)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    return parser


def default_instruments(symbols: list[str]) -> dict[str, InstrumentConfig]:
    return {
        symbol: InstrumentConfig(
            symbol=symbol,
            name=symbol,
            price_multiplier=1.0,
            margin_rate=0.12,
            commission_rate=0.00005 * 1.01,
            slippage_rate=0.00002,
        )
        for symbol in symbols
    }


def prepare_market(frames: dict[str, pd.DataFrame], config: SystemConfig) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    specs = default_forecast_specs(config.forecast_cap)
    for symbol, frame in frames.items():
        working = frame.copy()
        if len(working) < config.min_history_days:
            continue
        forecasts = build_price_forecasts(working["close"], config.forecast_cap)
        working["forecast"] = combine_forecasts(forecasts, specs, config.forecast_cap)
        working["daily_cash_volatility"] = calculate_daily_cash_volatility(
            working["close"],
            price_multiplier=1.0,
            span=config.volatility_span,
        )
        prepared[symbol] = working.dropna(subset=["forecast", "daily_cash_volatility"]).reset_index(drop=True)
    return prepared


def _filter_dates(
    market: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    start_date = pd.Timestamp(start) if start else None
    end_date = pd.Timestamp(end) if end else None
    for symbol, frame in market.items():
        working = frame
        if start_date is not None:
            working = working.loc[working["date"] >= start_date]
        if end_date is not None:
            working = working.loc[working["date"] <= end_date]
        working = working.reset_index(drop=True)
        if not working.empty:
            filtered[symbol] = working
    return filtered


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SystemConfig(
        initial_capital=args.initial_capital,
        target_annual_volatility=args.target_vol,
        max_margin_to_equity=args.max_margin,
    )
    frames = load_market_data_dir(Path(args.data_dir), pattern="*_daily.parquet")
    audit = audit_market_data(frames, min_history_days=config.min_history_days)
    tradeable_symbols = audit.loc[audit["is_tradeable"], "symbol"].astype(str).tolist()
    filtered = {symbol: frames[symbol] for symbol in tradeable_symbols}
    market = prepare_market(filtered, config)
    market = _filter_dates(market, start=args.start, end=args.end)
    instruments = default_instruments(sorted(market))

    result = run_backtest(market=market, instruments=instruments, system_config=config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(output_dir / "systematic_futures_equity.csv", index=False)
    result.trades.to_csv(output_dir / "systematic_futures_trades.csv", index=False)
    result.positions.to_csv(output_dir / "systematic_futures_positions.csv", index=False)
    audit.to_csv(output_dir / "systematic_futures_data_audit.csv", index=False)
    write_markdown_report(
        path=output_dir / "systematic_futures_report.md",
        equity=result.equity,
        trades=result.trades,
        positions=result.positions,
        initial_capital=config.initial_capital,
    )
    print(f"Wrote systematic futures report to {output_dir / 'systematic_futures_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
