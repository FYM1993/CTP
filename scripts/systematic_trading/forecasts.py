from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_trading.contracts import ForecastSpec


def cap_forecast(series: pd.Series, cap: float) -> pd.Series:
    return series.clip(lower=-abs(cap), upper=abs(cap)).fillna(0.0)


def ewmac_forecast(close: pd.Series, fast_span: int, slow_span: int, cap: float) -> pd.Series:
    fast = close.ewm(span=fast_span, min_periods=fast_span).mean()
    slow = close.ewm(span=slow_span, min_periods=slow_span).mean()
    daily_returns = close.pct_change()
    daily_vol = daily_returns.ewm(span=32, min_periods=20).std().replace(0.0, np.nan)
    raw = ((fast - slow) / close) / daily_vol
    scaled = raw * 4.0
    return cap_forecast(scaled, cap)


def breakout_forecast(close: pd.Series, lookback: int, cap: float) -> pd.Series:
    rolling_high = close.rolling(lookback, min_periods=lookback).max()
    rolling_low = close.rolling(lookback, min_periods=lookback).min()
    width = (rolling_high - rolling_low).replace(0.0, np.nan)
    location = ((close - rolling_low) / width - 0.5) * 2.0
    return cap_forecast(location * cap, cap)


def build_price_forecasts(close: pd.Series, cap: float) -> dict[str, pd.Series]:
    return {
        "ewmac_16_64": ewmac_forecast(close, fast_span=16, slow_span=64, cap=cap),
        "ewmac_32_128": ewmac_forecast(close, fast_span=32, slow_span=128, cap=cap),
        "breakout_80": breakout_forecast(close, lookback=80, cap=cap),
        "breakout_160": breakout_forecast(close, lookback=160, cap=cap),
    }


def default_forecast_specs(cap: float) -> list[ForecastSpec]:
    return [
        ForecastSpec(name="ewmac_16_64", weight=0.30, cap=cap),
        ForecastSpec(name="ewmac_32_128", weight=0.30, cap=cap),
        ForecastSpec(name="breakout_80", weight=0.20, cap=cap),
        ForecastSpec(name="breakout_160", weight=0.20, cap=cap),
    ]


def combine_forecasts(
    forecasts: dict[str, pd.Series],
    specs: list[ForecastSpec],
    cap: float,
) -> pd.Series:
    if not specs:
        raise ValueError("At least one forecast spec is required")
    total_weight = sum(spec.weight for spec in specs)
    if total_weight <= 0:
        raise ValueError("Forecast weights must sum to a positive value")
    combined = None
    for spec in specs:
        if spec.name not in forecasts:
            raise KeyError(f"Missing forecast series: {spec.name}")
        capped = cap_forecast(forecasts[spec.name], spec.cap)
        weighted = capped * (spec.weight / total_weight)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)
    if combined is None:
        return pd.Series(dtype=float)
    return cap_forecast(combined.replace([np.inf, -np.inf], np.nan), cap)
