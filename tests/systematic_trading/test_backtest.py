from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.backtest import run_backtest  # noqa: E402
from systematic_trading.contracts import InstrumentConfig, SystemConfig  # noqa: E402


def test_run_backtest_produces_equity_curve_and_trades() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series([100.0 + i * 0.5 for i in range(len(dates))])
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AG0",
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
            "open_interest": 5000,
            "forecast": 20.0,
            "daily_cash_volatility": 20.0,
        }
    )
    instrument = InstrumentConfig(
        symbol="AG0",
        name="白银",
        price_multiplier=10.0,
        margin_rate=0.10,
        commission_rate=0.0,
        slippage_rate=0.0,
    )
    config = SystemConfig(initial_capital=100_000.0, max_margin_to_equity=0.30)

    result = run_backtest(
        market={"AG0": frame},
        instruments={"AG0": instrument},
        system_config=config,
    )

    assert not result.equity.empty
    assert result.equity["equity"].iloc[-1] > result.equity["equity"].iloc[0]
    assert not result.trades.empty
