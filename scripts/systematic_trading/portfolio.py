from __future__ import annotations

import math

import pandas as pd

from systematic_trading.contracts import InstrumentConfig, SystemConfig, TargetPosition
from systematic_trading.risk import forecast_to_lots, margin_for_lots


def equal_instrument_weights(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def instrument_diversification_multiplier(num_instruments: int) -> float:
    if num_instruments <= 1:
        return 1.0
    return min(1.0 + 0.15 * math.log(num_instruments), 2.5)


def _row_for_date(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    matches = frame.loc[frame["date"] == date]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _cap_lots_by_margin(
    *,
    lots: int,
    price: float,
    instrument: InstrumentConfig,
    remaining_margin: float,
) -> int:
    if lots == 0:
        return 0
    per_lot_margin = margin_for_lots(1, price, instrument)
    if per_lot_margin <= 0:
        return 0
    max_abs_lots = int(math.floor(max(remaining_margin, 0.0) / per_lot_margin))
    capped_abs = min(abs(lots), max_abs_lots)
    return capped_abs if lots > 0 else -capped_abs


def build_target_positions(
    *,
    date: pd.Timestamp,
    market: dict[str, pd.DataFrame],
    instruments: dict[str, InstrumentConfig],
    current_lots: dict[str, int],
    system_config: SystemConfig,
    equity: float,
) -> list[TargetPosition]:
    tradeable_symbols = sorted(symbol for symbol in market if symbol in instruments)
    weights = equal_instrument_weights(tradeable_symbols)
    diversification_multiplier = instrument_diversification_multiplier(len(tradeable_symbols))
    margin_limit = equity * system_config.max_margin_to_equity
    used_margin = 0.0
    targets: list[TargetPosition] = []

    for symbol in tradeable_symbols:
        row = _row_for_date(market[symbol], date)
        if row is None:
            continue
        instrument = instruments[symbol]
        raw_lots = forecast_to_lots(
            forecast=float(row["forecast"]),
            daily_cash_volatility=float(row["daily_cash_volatility"]),
            instrument_weight=weights[symbol],
            diversification_multiplier=diversification_multiplier,
            system_config=system_config,
            equity=equity,
        )
        capped_lots = _cap_lots_by_margin(
            lots=raw_lots,
            price=float(row["close"]),
            instrument=instrument,
            remaining_margin=margin_limit - used_margin,
        )
        margin = margin_for_lots(capped_lots, float(row["close"]), instrument)
        used_margin += margin
        current = int(current_lots.get(symbol, 0))
        trade_lots = capped_lots - current
        if abs(trade_lots) < system_config.rebalance_buffer_lots:
            trade_lots = 0
            capped_lots = current
            margin = margin_for_lots(capped_lots, float(row["close"]), instrument)
        targets.append(
            TargetPosition(
                date=date.date().isoformat(),
                symbol=symbol,
                name=instrument.name,
                forecast=float(row["forecast"]),
                target_lots=int(capped_lots),
                current_lots=current,
                trade_lots=int(trade_lots),
                price=float(row["close"]),
                margin=float(margin),
                daily_cash_volatility=float(row["daily_cash_volatility"]),
            )
        )
    return targets
