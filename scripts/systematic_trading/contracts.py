from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    initial_capital: float
    target_annual_volatility: float = 0.20
    forecast_cap: float = 20.0
    max_margin_to_equity: float = 0.30
    max_abs_forecast: float = 20.0
    rebalance_buffer_lots: int = 1
    annual_trading_days: int = 252
    min_history_days: int = 260
    volatility_span: int = 32
    report_top_n: int = 20


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    name: str
    price_multiplier: float
    margin_rate: float
    commission_rate: float
    slippage_rate: float
    round_lot: int = 1


@dataclass(frozen=True)
class ForecastSpec:
    name: str
    weight: float
    cap: float


@dataclass(frozen=True)
class TargetPosition:
    date: str
    symbol: str
    name: str
    forecast: float
    target_lots: int
    current_lots: int
    trade_lots: int
    price: float
    margin: float
    daily_cash_volatility: float


@dataclass(frozen=True)
class TradeFill:
    date: str
    symbol: str
    name: str
    trade_lots: int
    price: float
    turnover: float
    cost: float


@dataclass(frozen=True)
class EquityRow:
    date: str
    equity: float
    pnl: float
    cost: float
    margin: float
    gross_exposure: float
    net_exposure: float
    num_positions: int


@dataclass(frozen=True)
class BacktestSummary:
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    total_cost: float
    turnover: float
    avg_margin_to_equity: float
    max_margin_to_equity: float
