from __future__ import annotations

import math

import numpy as np
import pandas as pd

from systematic_trading.contracts import InstrumentConfig, SystemConfig


def calculate_daily_cash_volatility(
    close: pd.Series,
    price_multiplier: float,
    span: int,
) -> pd.Series:
    daily_price_change = close.diff()
    daily_price_volatility = daily_price_change.ewm(span=span, min_periods=max(5, span // 2)).std()
    return (daily_price_volatility.abs() * price_multiplier).replace([np.inf, -np.inf], np.nan)


def daily_cash_risk_budget(
    *,
    equity: float,
    target_annual_volatility: float,
    annual_trading_days: int,
    instrument_weight: float,
    diversification_multiplier: float,
) -> float:
    account_daily_risk = equity * target_annual_volatility / math.sqrt(annual_trading_days)
    return account_daily_risk * instrument_weight * diversification_multiplier


def forecast_to_lots(
    *,
    forecast: float,
    daily_cash_volatility: float,
    instrument_weight: float,
    diversification_multiplier: float,
    system_config: SystemConfig,
    equity: float | None = None,
) -> int:
    if daily_cash_volatility <= 0 or not math.isfinite(daily_cash_volatility):
        return 0
    account_equity = system_config.initial_capital if equity is None else equity
    risk_budget = daily_cash_risk_budget(
        equity=account_equity,
        target_annual_volatility=system_config.target_annual_volatility,
        annual_trading_days=system_config.annual_trading_days,
        instrument_weight=instrument_weight,
        diversification_multiplier=diversification_multiplier,
    )
    forecast_fraction = max(min(forecast / system_config.forecast_cap, 1.0), -1.0)
    raw_lots = forecast_fraction * risk_budget / daily_cash_volatility
    if raw_lots > 0:
        return int(math.floor(raw_lots))
    return int(math.ceil(raw_lots))


def margin_for_lots(lots: int, price: float, instrument: InstrumentConfig) -> float:
    return abs(lots) * price * instrument.price_multiplier * instrument.margin_rate


def trade_cost(lots_changed: int, price: float, instrument: InstrumentConfig) -> float:
    turnover = abs(lots_changed) * price * instrument.price_multiplier
    return turnover * (instrument.commission_rate + instrument.slippage_rate)
