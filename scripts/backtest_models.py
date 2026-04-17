from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestCase:
    case_id: str
    symbol: str
    name: str
    direction: str
    start_dt: str
    end_dt: str
    note: str


@dataclass(frozen=True, slots=True)
class TradePlan:
    entry_ref: str
    stop: float
    tp1: float
    tp2: float
    phase2_score: float
    signal_type: str


@dataclass(frozen=True, slots=True)
class TradeRecord:
    exit_reason: str
    tp1_hit: bool
    days_held: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    pass
