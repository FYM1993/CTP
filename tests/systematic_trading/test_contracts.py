from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.contracts import ForecastSpec, InstrumentConfig, SystemConfig  # noqa: E402


def test_system_config_defaults_are_conservative() -> None:
    config = SystemConfig(initial_capital=1_000_000.0)

    assert config.initial_capital == 1_000_000.0
    assert config.target_annual_volatility == 0.20
    assert config.forecast_cap == 20.0
    assert config.rebalance_buffer_lots == 1
    assert config.max_margin_to_equity == 0.30
    assert config.annual_trading_days == 252


def test_instrument_config_describes_tradeable_contract() -> None:
    config = InstrumentConfig(
        symbol="AG0",
        name="白银",
        price_multiplier=15.0,
        margin_rate=0.12,
        commission_rate=0.00005,
        slippage_rate=0.00002,
    )

    assert config.symbol == "AG0"
    assert config.name == "白银"
    assert config.price_multiplier == 15.0
    assert config.margin_rate == 0.12
    assert config.round_lot == 1


def test_forecast_spec_is_weighted_and_capped() -> None:
    spec = ForecastSpec(name="ewmac_16_64", weight=0.5, cap=20.0)

    assert spec.name == "ewmac_16_64"
    assert spec.weight == 0.5
    assert spec.cap == 20.0
