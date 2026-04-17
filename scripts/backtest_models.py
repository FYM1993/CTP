from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class BacktestCase:
    case_id: str
    symbol: str
    name: str
    direction: str
    start_dt: date
    end_dt: date
    note: str = ""


@dataclass(frozen=True, slots=True)
class TradePlan:
    trade_id: str
    symbol: str
    direction: str
    plan_date: str
    entry_ref: float
    stop: float
    tp1: float
    tp2: float
    phase2_score: float
    signal_type: str
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    bars_held: int
    days_held: int
    tp1_hit: bool
    pnl_ratio: float
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    case_id: str
    trades: list[TradeRecord]
    summary: dict[str, float | int]
