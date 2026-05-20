from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.contracts import InstrumentConfig, SystemConfig  # noqa: E402
from systematic_trading.risk import calculate_daily_cash_volatility, forecast_to_lots, margin_for_lots  # noqa: E402


def test_daily_cash_volatility_scales_with_contract_multiplier() -> None:
    close = pd.Series([100.0, 101.0, 99.0, 102.0, 100.0] * 20)

    small = calculate_daily_cash_volatility(close, price_multiplier=10.0, span=16)
    large = calculate_daily_cash_volatility(close, price_multiplier=20.0, span=16)

    assert small.iloc[-1] > 0
    assert large.iloc[-1] == pytest.approx(small.iloc[-1] * 2.0)


def test_forecast_to_lots_respects_forecast_sign_and_cap() -> None:
    config = SystemConfig(initial_capital=1_000_000.0, target_annual_volatility=0.20)

    long_lots = forecast_to_lots(
        forecast=10.0,
        daily_cash_volatility=200.0,
        instrument_weight=0.25,
        diversification_multiplier=1.0,
        system_config=config,
    )
    short_lots = forecast_to_lots(
        forecast=-10.0,
        daily_cash_volatility=200.0,
        instrument_weight=0.25,
        diversification_multiplier=1.0,
        system_config=config,
    )

    assert long_lots > 0
    assert short_lots == -long_lots


def test_forecast_to_lots_gives_fewer_lots_to_higher_volatility() -> None:
    config = SystemConfig(initial_capital=1_000_000.0, target_annual_volatility=0.20)

    low_vol_lots = forecast_to_lots(
        forecast=20.0,
        daily_cash_volatility=100.0,
        instrument_weight=0.25,
        diversification_multiplier=1.0,
        system_config=config,
    )
    high_vol_lots = forecast_to_lots(
        forecast=20.0,
        daily_cash_volatility=400.0,
        instrument_weight=0.25,
        diversification_multiplier=1.0,
        system_config=config,
    )

    assert 0 < high_vol_lots < low_vol_lots


def test_margin_for_lots_uses_abs_lots() -> None:
    instrument = InstrumentConfig(
        symbol="AG0",
        name="白银",
        price_multiplier=15.0,
        margin_rate=0.12,
        commission_rate=0.00005,
        slippage_rate=0.00002,
    )

    margin = margin_for_lots(lots=-2, price=5000.0, instrument=instrument)

    assert margin == pytest.approx(18_000.0)
