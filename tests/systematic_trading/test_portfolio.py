from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.contracts import InstrumentConfig, SystemConfig  # noqa: E402
from systematic_trading.portfolio import (  # noqa: E402
    build_target_positions,
    equal_instrument_weights,
    instrument_diversification_multiplier,
)


def _instrument(symbol: str) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=symbol,
        name=symbol,
        price_multiplier=10.0,
        margin_rate=0.10,
        commission_rate=0.00005,
        slippage_rate=0.00002,
    )


def test_equal_instrument_weights_sum_to_one() -> None:
    weights = equal_instrument_weights(["AG0", "AU0", "CU0"])

    assert weights == {"AG0": 1 / 3, "AU0": 1 / 3, "CU0": 1 / 3}
    assert sum(weights.values()) == 1.0


def test_diversification_multiplier_increases_with_more_instruments() -> None:
    assert instrument_diversification_multiplier(1) == 1.0
    assert instrument_diversification_multiplier(8) > 1.0
    assert instrument_diversification_multiplier(100) <= 2.5


def test_build_target_positions_respects_margin_cap() -> None:
    date = pd.Timestamp("2024-01-03")
    market = {
        "AG0": pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02"), date],
                "symbol": ["AG0", "AG0"],
                "open": [100.0, 100.0],
                "high": [100.0, 100.0],
                "low": [100.0, 100.0],
                "close": [100.0, 100.0],
                "volume": [1000, 1000],
                "open_interest": [5000, 5000],
                "forecast": [20.0, 20.0],
                "daily_cash_volatility": [10.0, 10.0],
            }
        )
    }
    config = SystemConfig(initial_capital=100_000.0, max_margin_to_equity=0.10)

    targets = build_target_positions(
        date=date,
        market=market,
        instruments={"AG0": _instrument("AG0")},
        current_lots={"AG0": 0},
        system_config=config,
        equity=100_000.0,
    )

    assert len(targets) == 1
    assert targets[0].margin <= 10_000.0


def test_build_target_positions_keeps_small_rebalance_unchanged() -> None:
    date = pd.Timestamp("2024-01-03")
    market = {
        "AG0": pd.DataFrame(
            {
                "date": [date],
                "symbol": ["AG0"],
                "open": [100.0],
                "high": [100.0],
                "low": [100.0],
                "close": [100.0],
                "volume": [1000],
                "open_interest": [5000],
                "forecast": [0.1],
                "daily_cash_volatility": [10_000.0],
            }
        )
    }
    config = SystemConfig(initial_capital=100_000.0, rebalance_buffer_lots=2)

    targets = build_target_positions(
        date=date,
        market=market,
        instruments={"AG0": _instrument("AG0")},
        current_lots={"AG0": 1},
        system_config=config,
        equity=100_000.0,
    )

    assert targets[0].trade_lots == 0
    assert targets[0].target_lots == 1
