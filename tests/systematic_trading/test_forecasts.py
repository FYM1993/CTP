from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.contracts import ForecastSpec  # noqa: E402
from systematic_trading.forecasts import breakout_forecast, combine_forecasts, ewmac_forecast  # noqa: E402


def test_ewmac_forecast_is_positive_for_persistent_uptrend() -> None:
    close = pd.Series([100.0 + i for i in range(160)])

    forecast = ewmac_forecast(close, fast_span=16, slow_span=64, cap=20.0)

    assert forecast.iloc[-1] > 0
    assert forecast.abs().max() <= 20.0


def test_breakout_forecast_is_negative_near_range_low() -> None:
    close = pd.Series([100.0] * 100 + [80.0])

    forecast = breakout_forecast(close, lookback=100, cap=20.0)

    assert forecast.iloc[-1] < -15.0
    assert forecast.abs().max() <= 20.0


def test_combine_forecasts_uses_weights_and_cap() -> None:
    forecasts = {
        "fast": pd.Series([30.0, 30.0]),
        "slow": pd.Series([10.0, 10.0]),
    }
    specs = [
        ForecastSpec(name="fast", weight=0.25, cap=20.0),
        ForecastSpec(name="slow", weight=0.75, cap=20.0),
    ]

    combined = combine_forecasts(forecasts, specs, cap=20.0)

    assert combined.iloc[-1] == 12.5
