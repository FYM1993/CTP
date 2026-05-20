from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from systematic_trading.contracts import InstrumentConfig, SystemConfig
from systematic_trading.portfolio import build_target_positions
from systematic_trading.risk import margin_for_lots, trade_cost


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


def _all_dates(market: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates = sorted(set().union(*[set(frame["date"]) for frame in market.values()]))
    return [pd.Timestamp(date) for date in dates]


def _price_on(frame: pd.DataFrame, date: pd.Timestamp) -> float | None:
    rows = frame.loc[frame["date"] == date]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["close"])


def _mark_to_market_pnl(
    *,
    market: dict[str, pd.DataFrame],
    instruments: dict[str, InstrumentConfig],
    lots: dict[str, int],
    previous_prices: dict[str, float],
    date: pd.Timestamp,
) -> tuple[float, dict[str, float]]:
    pnl = 0.0
    latest_prices = dict(previous_prices)
    for symbol, position in lots.items():
        if symbol not in market or symbol not in instruments:
            continue
        price = _price_on(market[symbol], date)
        if price is None:
            continue
        previous = previous_prices.get(symbol, price)
        pnl += position * (price - previous) * instruments[symbol].price_multiplier
        latest_prices[symbol] = price
    return pnl, latest_prices


def _portfolio_snapshot(
    *,
    market: dict[str, pd.DataFrame],
    instruments: dict[str, InstrumentConfig],
    current_lots: dict[str, int],
    date: pd.Timestamp,
) -> tuple[float, float, float, int]:
    total_margin = 0.0
    gross_exposure = 0.0
    net_exposure = 0.0
    num_positions = 0
    for symbol, position in current_lots.items():
        if symbol not in market or symbol not in instruments:
            continue
        price = _price_on(market[symbol], date)
        if price is None:
            continue
        notional = position * price * instruments[symbol].price_multiplier
        gross_exposure += abs(notional)
        net_exposure += notional
        total_margin += margin_for_lots(position, price, instruments[symbol])
        if position != 0:
            num_positions += 1
    return total_margin, gross_exposure, net_exposure, num_positions


def run_backtest(
    *,
    market: dict[str, pd.DataFrame],
    instruments: dict[str, InstrumentConfig],
    system_config: SystemConfig,
) -> BacktestResult:
    dates = _all_dates(market)
    equity = float(system_config.initial_capital)
    current_lots: dict[str, int] = {symbol: 0 for symbol in market}
    previous_prices: dict[str, float] = {}
    equity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for date in dates:
        daily_pnl, previous_prices = _mark_to_market_pnl(
            market=market,
            instruments=instruments,
            lots=current_lots,
            previous_prices=previous_prices,
            date=date,
        )
        equity += daily_pnl

        targets = build_target_positions(
            date=date,
            market=market,
            instruments=instruments,
            current_lots=current_lots,
            system_config=system_config,
            equity=equity,
        )

        daily_cost = 0.0
        for target in targets:
            if target.trade_lots != 0:
                instrument = instruments[target.symbol]
                cost = trade_cost(target.trade_lots, target.price, instrument)
                daily_cost += cost
                trade_rows.append(
                    {
                        "date": target.date,
                        "symbol": target.symbol,
                        "name": target.name,
                        "trade_lots": target.trade_lots,
                        "price": target.price,
                        "turnover": abs(target.trade_lots) * target.price * instrument.price_multiplier,
                        "cost": cost,
                    }
                )
                current_lots[target.symbol] = target.target_lots
                previous_prices[target.symbol] = target.price
            position_rows.append(asdict(target))

        equity -= daily_cost
        total_margin, gross_exposure, net_exposure, num_positions = _portfolio_snapshot(
            market=market,
            instruments=instruments,
            current_lots=current_lots,
            date=date,
        )
        equity_rows.append(
            {
                "date": date.date().isoformat(),
                "equity": equity,
                "pnl": daily_pnl,
                "cost": daily_cost,
                "margin": total_margin,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "num_positions": num_positions,
            }
        )

    return BacktestResult(
        equity=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
        positions=pd.DataFrame(position_rows),
    )
