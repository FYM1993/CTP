from __future__ import annotations

from collections.abc import Callable

import pandas as pd

import pre_market
from backtest_models import BacktestCase, BacktestResult, TradePlan, TradeRecord
from intraday import generate_signals


def aggregate_partial_5m_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return minute_df.copy()

    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime", kind="stable")
    df["bucket"] = df["datetime"].dt.floor("5min")

    grouped = (
        df.groupby("bucket", sort=True, as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"bucket": "datetime"})
    )
    return grouped


def make_trade_plan_from_phase2(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
) -> TradePlan | None:
    raw = pre_market.build_trade_plan_from_daily_df(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    if raw is None or not raw.get("actionable"):
        return None

    reversal_status = raw.get("reversal_status") or {}
    signal_date = str(reversal_status.get("signal_date") or "")
    return TradePlan(
        trade_id=f"{case.case_id}_{signal_date}",
        symbol=case.symbol,
        direction=case.direction,
        plan_date=signal_date,
        entry_ref=float(raw["entry"]),
        stop=float(raw["stop"]),
        tp1=float(raw["tp1"]),
        tp2=float(raw["tp2"]),
        phase2_score=float(raw["score"]),
        signal_type=str(reversal_status.get("signal_type") or ""),
    )


def _safe_generate_signals(df: pd.DataFrame, direction: str, cfg: dict) -> list[dict]:
    if len(df) < 2:
        return []
    return generate_signals(df, direction, cfg)


def _calc_pnl_ratio(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _build_trade_record(
    *,
    case: BacktestCase,
    plan: TradePlan,
    entry_time: pd.Timestamp,
    entry_price: float,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    bars_held: int,
    tp1_hit: bool,
) -> TradeRecord:
    entry_ts = pd.Timestamp(entry_time)
    exit_ts = pd.Timestamp(exit_time)
    return TradeRecord(
        trade_id=plan.trade_id,
        symbol=case.symbol,
        direction=case.direction,
        entry_time=str(entry_ts),
        entry_price=entry_price,
        exit_time=str(exit_ts),
        exit_price=exit_price,
        exit_reason=exit_reason,
        bars_held=bars_held,
        days_held=(exit_ts.date() - entry_ts.date()).days,
        tp1_hit=tp1_hit,
        pnl_ratio=_calc_pnl_ratio(case.direction, entry_price, exit_price),
        meta={
            "plan_date": plan.plan_date,
            "signal_type": plan.signal_type,
            "phase2_score": plan.phase2_score,
        },
    )


def _resolve_exit_on_bar(
    *, direction: str, bar: pd.Series, plan: TradePlan
) -> tuple[str, float] | None:
    high = float(bar["high"])
    low = float(bar["low"])

    if direction == "long":
        if low <= plan.stop:
            return "stop", plan.stop
        if high >= plan.tp2:
            return "tp2", plan.tp2
        return None

    if high >= plan.stop:
        return "stop", plan.stop
    if low <= plan.tp2:
        return "tp2", plan.tp2
    return None


def run_case_from_frames(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    plan_factory: Callable[..., TradePlan | None] = make_trade_plan_from_phase2,
    pre_market_cfg: dict,
    signal_cfg: dict,
) -> BacktestResult:
    trades: list[TradeRecord] = []

    minute_data = minute_df.copy()
    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0})

    minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])
    minute_data["trade_date"] = minute_data["datetime"].dt.date
    minute_data = minute_data.sort_values("datetime", kind="stable")
    minute_data = minute_data.loc[
        minute_data["trade_date"].between(case.start_dt, case.end_dt)
    ].copy()

    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0})

    visible_daily = daily_df.copy()
    visible_daily["date"] = pd.to_datetime(visible_daily["date"])
    visible_daily = visible_daily.sort_values("date", kind="stable")
    visible_daily["trade_date"] = visible_daily["date"].dt.date

    active_plan: TradePlan | None = None
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    bars_held = 0
    tp1_hit = False
    consumed_trade_ids: set[str] = set()

    for trade_date, day_minutes in minute_data.groupby("trade_date", sort=True):
        day_minutes = day_minutes.sort_values("datetime", kind="stable")
        day_visible_daily = visible_daily.loc[visible_daily["trade_date"] < trade_date].copy()

        if active_plan is None:
            day_plan = plan_factory(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
            )
            if day_plan is not None and day_plan.trade_id in consumed_trade_ids:
                day_plan = None
        else:
            refreshed_plan = plan_factory(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
            )
            if refreshed_plan is None:
                first_bar = day_minutes.iloc[0]
                trades.append(
                    _build_trade_record(
                        case=case,
                        plan=active_plan,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=first_bar["datetime"],
                        exit_price=float(first_bar["open"]),
                        exit_reason="phase2_invalidated",
                        bars_held=bars_held,
                        tp1_hit=tp1_hit,
                    )
                )
                active_plan = None
                entry_time = None
                entry_price = None
                bars_held = 0
                tp1_hit = False
                continue
            day_plan = None

        for minute_idx in range(len(day_minutes)):
            current_minute = day_minutes.iloc[minute_idx]

            if active_plan is None:
                if day_plan is None:
                    break
                partial_5m = aggregate_partial_5m_bars(day_minutes.iloc[: minute_idx + 1])
                signals = _safe_generate_signals(partial_5m, case.direction, signal_cfg)
                open_signal = "开多" if case.direction == "long" else "开空"
                if any(signal.get("type") == open_signal for signal in signals):
                    consumed_trade_ids.add(day_plan.trade_id)
                    active_plan = day_plan
                    entry_time = pd.Timestamp(current_minute["datetime"])
                    entry_price = float(current_minute["close"])
                    bars_held = 0
                    tp1_hit = False
                continue

            bars_held += 1
            last_price = float(current_minute["close"])

            if case.direction == "long":
                tp1_hit = tp1_hit or last_price >= active_plan.tp1
            else:
                tp1_hit = tp1_hit or last_price <= active_plan.tp1
            exit = _resolve_exit_on_bar(
                direction=case.direction,
                bar=current_minute,
                plan=active_plan,
            )

            if exit is None:
                continue
            exit_reason, exit_price = exit

            trades.append(
                _build_trade_record(
                    case=case,
                    plan=active_plan,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=current_minute["datetime"],
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    tp1_hit=tp1_hit,
                )
            )
            active_plan = None
            entry_time = None
            entry_price = None
            bars_held = 0
            tp1_hit = False
            break

    if active_plan is not None:
        final_bar = minute_data.iloc[-1]
        trades.append(
            _build_trade_record(
                case=case,
                plan=active_plan,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=final_bar["datetime"],
                exit_price=float(final_bar["close"]),
                exit_reason="end_of_data",
                bars_held=bars_held,
                tp1_hit=tp1_hit,
            )
        )

    return BacktestResult(
        case_id=case.case_id,
        trades=trades,
        summary={"num_trades": len(trades)},
    )
