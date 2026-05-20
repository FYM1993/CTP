from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mainline_radar import _finite_float, _snapshot_context, build_daily_snapshot
from market.contract_specs import builtin_contract_spec


@dataclass(frozen=True)
class MainlineEventParams:
    lookback_days: int = 5
    top_n: int = 1
    min_active_days: int = 2
    sprouting_exposure: float = 0.50
    confirmed_exposure: float = 1.50
    markup_exposure: float = 3.00
    crowded_exposure: float = 0.50
    min_sprouting_score: float = 55.0
    min_confirmed_score: float = 62.0
    min_markup_score: float = 72.0
    switch_score_advantage: float = 18.0
    allow_rotation: bool = False
    allow_markup_entry_from_cash: bool = False
    cooldown_days: int = 5
    max_trial_days: int = 3
    commission_multiplier: float = 1.01


@dataclass(frozen=True)
class EventState:
    state: str
    target_exposure: float
    can_open: bool
    exit_reason: str = ""


@dataclass(frozen=True)
class EventCandidate:
    date: str
    board: str
    direction: str
    symbol: str
    state: str
    target_exposure: float
    weekly_score: float
    latest_mainline_score: float
    active_days: int
    weekly_status: str


@dataclass
class OpenPosition:
    board: str
    direction: str
    symbol: str
    entry_date: str
    entry_price: float
    exposure: float
    entry_exposure: float
    state: str
    entry_state: str
    weekly_score: float
    max_exposure: float
    max_equity: float = 1.0
    crowded_days: int = 0
    bars_held: int = 0


@dataclass(frozen=True)
class EventBacktestResult:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    daily_rows: list[dict[str, Any]]


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _weekly_rows_for_date(
    state_rows: list[dict[str, Any]],
    daily_snapshots: dict[str, list[dict[str, Any]]],
    ordered_dates: list[str],
    date_value: str,
    params: MainlineEventParams,
) -> list[dict[str, Any]]:
    dates = [date for date in ordered_dates if date <= date_value][-int(params.lookback_days) :]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for date in dates:
        for row in daily_snapshots[date]:
            grouped[(str(row.get("board") or ""), str(row.get("direction") or ""))].append(row)

    weekly_rows: list[dict[str, Any]] = []
    for (board, direction), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: str(row.get("date") or ""))
        first = ordered[0]
        latest = ordered[-1]
        active = [
            row
            for row in ordered
            if row.get("lifecycle_stage") != "no_mainline" and not bool(row.get("is_isolated_move"))
        ]
        active_days = len(active)
        avg_score = sum(_finite_float(row.get("mainline_score"), 0.0) for row in ordered) / max(len(ordered), 1)
        avg_crowding = sum(_finite_float(row.get("crowding_warning"), 0.0) for row in ordered) / max(len(ordered), 1)
        score_change = _finite_float(latest.get("mainline_score"), 0.0) - _finite_float(first.get("mainline_score"), 0.0)
        breadth_change = _finite_float(latest.get("leadership_breadth"), 0.0) - _finite_float(first.get("leadership_breadth"), 0.0)
        retrace_change = _finite_float(first.get("retrace_atr"), 0.0) - _finite_float(latest.get("retrace_atr"), 0.0)
        crowding_change = _finite_float(latest.get("crowding_warning"), 0.0) - _finite_float(first.get("crowding_warning"), 0.0)
        latest_stage = str(latest.get("lifecycle_stage") or "")
        late_risk = latest_stage == "crowded" or (
            _finite_float(latest.get("mainline_score"), 0.0) >= 75.0 and crowding_change >= 45.0
        )
        improving = active_days >= 2 and (score_change >= 8.0 or breadth_change >= 0.25 or retrace_change >= 1.0)
        if late_risk:
            status = "late_cycle_risk"
        elif improving:
            status = "improving_consensus"
        elif score_change <= -8.0 or breadth_change <= -0.25:
            status = "weakening"
        elif active_days <= 1 and latest_stage != "no_mainline":
            status = "one_day_spike"
        elif latest_stage == "confirming":
            status = "confirming"
        elif active_days >= max(3, min(int(params.lookback_days), len(dates)) - 1):
            status = "persistent_mainline"
        else:
            status = "watching"
        weekly_score = (
            25.0 * active_days
            + avg_score
            + 10.0 * _finite_float(latest.get("leadership_breadth"), 0.0)
            + 0.05 * _finite_float(latest.get("capital_confirmation"), 0.0)
            - 0.08 * avg_crowding
        )
        weekly_rows.append(
            {
                "date": date_value,
                "board": board,
                "direction": direction,
                "active_days": active_days,
                "weekly_status": status,
                "weekly_score": round(weekly_score, 4),
                "latest_mainline_score": latest.get("mainline_score"),
                "latest_stage": latest_stage,
                "crowding_change": round(crowding_change, 4),
                "representative_symbol": str(latest.get("representative_symbol") or "").upper(),
            }
        )
    return sorted(
        weekly_rows,
        key=lambda row: (_event_rank(classify_event_state(row, params).state), _finite_float(row.get("weekly_score"), 0.0)),
        reverse=True,
    )


def classify_event_state(row: dict[str, Any], params: MainlineEventParams) -> EventState:
    status = str(row.get("weekly_status") or "")
    latest_stage = str(row.get("latest_stage") or "")
    active_days = int(_finite_float(row.get("active_days"), 0.0))
    score = _finite_float(row.get("latest_mainline_score"), 0.0)
    if status == "late_cycle_risk" or latest_stage == "crowded":
        return EventState("crowded", float(params.crowded_exposure), False, "mainline_crowded")
    if status == "weakening":
        return EventState("ended", 0.0, False, "mainline_weakening")
    if active_days < int(params.min_active_days):
        return EventState("none", 0.0, False, "mainline_not_active")
    if status == "persistent_mainline" and score >= float(params.min_markup_score):
        return EventState("markup", float(params.markup_exposure), True)
    if status in {"confirming", "improving_consensus", "persistent_mainline"} and score >= float(params.min_confirmed_score):
        return EventState("confirmed", float(params.confirmed_exposure), True)
    if status in {"confirming", "improving_consensus"} and score >= float(params.min_sprouting_score):
        return EventState("sprouting", float(params.sprouting_exposure), True)
    return EventState("none", 0.0, False, "mainline_score_low")


def _event_rank(state: str) -> int:
    return {"markup": 4, "confirmed": 3, "sprouting": 2, "crowded": 1}.get(str(state), 0)


def _build_event_candidates(
    state_rows: list[dict[str, Any]],
    daily_snapshots: dict[str, list[dict[str, Any]]],
    ordered_dates: list[str],
    date_value: str,
    params: MainlineEventParams,
) -> list[EventCandidate]:
    out: list[EventCandidate] = []
    for row in _weekly_rows_for_date(state_rows, daily_snapshots, ordered_dates, date_value, params):
        event = classify_event_state(row, params)
        if event.state in {"none", "ended"}:
            continue
        symbol = str(row.get("representative_symbol") or "").upper()
        if not symbol:
            continue
        out.append(
            EventCandidate(
                date=_date_str(date_value),
                board=str(row.get("board") or ""),
                direction=str(row.get("direction") or ""),
                symbol=symbol,
                state=event.state,
                target_exposure=float(event.target_exposure),
                weekly_score=_finite_float(row.get("weekly_score"), 0.0),
                latest_mainline_score=_finite_float(row.get("latest_mainline_score"), 0.0),
                active_days=int(_finite_float(row.get("active_days"), 0.0)),
                weekly_status=str(row.get("weekly_status") or ""),
            )
        )
    return sorted(out, key=lambda item: (_event_rank(item.state), item.weekly_score), reverse=True)


def _close_index(state_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in state_rows:
        symbol = str(row.get("symbol") or "").upper()
        close = _finite_float(row.get("close"), math.nan)
        if symbol and math.isfinite(close) and close > 0:
            out[symbol][_date_str(row.get("date"))] = close
    return dict(out)


def _fee_rate(symbol: str, close: float, params: MainlineEventParams) -> tuple[float, bool]:
    spec = builtin_contract_spec(symbol)
    if not spec:
        return 0.0, True
    commission_rate = _finite_float(spec.get("commission_rate"), 0.0)
    if commission_rate > 0:
        return commission_rate * float(params.commission_multiplier), False
    per_lot = _finite_float(spec.get("commission_per_lot"), 0.0)
    multiplier = _finite_float(spec.get("multiplier"), 0.0)
    if per_lot > 0 and multiplier > 0 and close > 0:
        return per_lot / (close * multiplier) * float(params.commission_multiplier), False
    return 0.0, True


def _same_event(position: OpenPosition, candidate: EventCandidate) -> bool:
    return position.board == candidate.board and position.direction == candidate.direction


def _directional_return(direction: str, first: float, second: float) -> float:
    if first <= 0 or second <= 0:
        return 0.0
    raw = second / first - 1.0
    return -raw if str(direction) == "short" else raw


def run_event_strategy(state_rows: list[dict[str, Any]], params: MainlineEventParams | None = None) -> EventBacktestResult:
    resolved = params or MainlineEventParams()
    rows = list(state_rows)
    dates = sorted({_date_str(row.get("date")) for row in rows if str(row.get("date") or "")})
    context = _snapshot_context(rows)
    daily_snapshots = {date: build_daily_snapshot(rows, date, context=context) for date in dates}
    closes = _close_index(rows)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    position: OpenPosition | None = None
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    equity_curve = [{"date": dates[0] if dates else "", "equity": equity, "event": "start", "active_positions": 0}]
    missing_fee_symbols: set[str] = set()
    entries = 0
    exits = 0
    exposure_changes = 0
    max_positions = 0
    cooldown_remaining = 0

    for index, date in enumerate(dates[:-1]):
        next_date = dates[index + 1]
        candidates = _build_event_candidates(rows, daily_snapshots, dates, date, resolved)
        tradable = [item for item in candidates if item.state != "crowded" and item.target_exposure > 0]
        cash_tradable = [
            item
            for item in tradable
            if bool(resolved.allow_markup_entry_from_cash) or item.state != "markup"
        ]
        best_tradable = tradable[0] if tradable else None
        desired = cash_tradable[0] if cash_tradable else None
        candidate_by_event = {(item.board, item.direction): item for item in candidates}

        turnover = 0.0
        action = "hold_cash"
        exit_reason = ""
        if position is not None:
            position.bars_held += 1
            current = candidate_by_event.get((position.board, position.direction))
            should_exit = current is None or current.state in {"none", "ended"}
            if current is not None and current.state == "crowded":
                position.crowded_days += 1
                if position.exposure != current.target_exposure:
                    turnover += abs(position.exposure - current.target_exposure)
                    position.exposure = current.target_exposure
                    position.max_exposure = max(position.max_exposure, position.exposure)
                    exposure_changes += 1
                action = "reduce_crowded"
                should_exit = position.crowded_days >= 3
                exit_reason = "mainline_crowded" if should_exit else ""
            elif current is not None:
                position.crowded_days = 0
                if position.exposure != current.target_exposure:
                    turnover += abs(position.exposure - current.target_exposure)
                    position.exposure = current.target_exposure
                    position.max_exposure = max(position.max_exposure, position.exposure)
                    exposure_changes += 1
                position.state = current.state
                position.weekly_score = current.weekly_score
                if (
                    position.entry_state == "sprouting"
                    and current.state == "sprouting"
                    and position.bars_held >= int(resolved.max_trial_days)
                ):
                    should_exit = True
                    exit_reason = "trial_not_confirmed"
            if bool(resolved.allow_rotation) and best_tradable is not None and not _same_event(position, best_tradable):
                if current is None or best_tradable.weekly_score >= position.weekly_score + float(resolved.switch_score_advantage):
                    should_exit = True
                    exit_reason = "rotated_to_stronger_mainline"
            if should_exit:
                close = closes.get(position.symbol, {}).get(date, position.entry_price)
                turnover += position.exposure
                trades.append(
                    {
                        "entry_date": position.entry_date,
                        "exit_date": date,
                        "entry_symbol": position.symbol,
                        "exit_symbol": position.symbol,
                        "board": position.board,
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "exit_price": close,
                        "exit_reason": exit_reason or "mainline_lost",
                        "entry_state": position.entry_state,
                        "exit_state": position.state,
                        "entry_exposure": position.entry_exposure,
                        "exit_exposure": position.exposure,
                        "max_exposure": position.max_exposure,
                    }
                )
                exits += 1
                position = None
                action = exit_reason or "mainline_lost"
                cooldown_remaining = max(int(resolved.cooldown_days), 0)

        if position is None and cooldown_remaining <= 0 and desired is not None:
            close = closes.get(desired.symbol, {}).get(date)
            if close is not None:
                position = OpenPosition(
                    board=desired.board,
                    direction=desired.direction,
                    symbol=desired.symbol,
                    entry_date=date,
                    entry_price=close,
                    exposure=desired.target_exposure,
                    entry_exposure=desired.target_exposure,
                    state=desired.state,
                    entry_state=desired.state,
                    weekly_score=desired.weekly_score,
                    max_exposure=desired.target_exposure,
                )
                turnover += desired.target_exposure
                entries += 1
                action = f"entry_{desired.state}"
        elif position is None and cooldown_remaining > 0:
            action = "cooldown"

        daily_ret = 0.0
        fee = 0.0
        active_positions = 0
        exposure = 0.0
        held_symbol = ""
        held_board = ""
        held_direction = ""
        held_state = ""
        if position is not None:
            first = closes.get(position.symbol, {}).get(date)
            second = closes.get(position.symbol, {}).get(next_date)
            if first is not None and second is not None:
                daily_ret = float(position.exposure) * _directional_return(position.direction, first, second)
            exposure = float(position.exposure)
            held_symbol = position.symbol
            held_board = position.board
            held_direction = position.direction
            held_state = position.state
            active_positions = 1
            max_positions = max(max_positions, active_positions)
            position.max_equity = max(position.max_equity, equity)
        if turnover > 0:
            fee_symbol = held_symbol or (trades[-1]["exit_symbol"] if trades else "")
            ref_close = closes.get(fee_symbol, {}).get(date, 0.0)
            cost, missing = _fee_rate(fee_symbol, ref_close, resolved) if fee_symbol else (0.0, False)
            fee = turnover * cost
            if missing and fee_symbol:
                missing_fee_symbols.add(fee_symbol)
        net_ret = daily_ret - fee
        equity *= max(0.0, 1.0 + net_ret)
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        daily_rows.append(
            {
                "date": date,
                "next_date": next_date,
                "action": action,
                "symbol": held_symbol,
                "board": held_board,
                "direction": held_direction,
                "state": held_state,
                "exposure": exposure,
                "daily_return": daily_ret,
                "fee_return": fee,
                "net_return": net_ret,
                "equity": equity,
                "drawdown": drawdown,
                "active_positions": active_positions,
            }
        )
        equity_curve.append({"date": next_date, "equity": equity, "event": action, "active_positions": active_positions})
        if position is None and cooldown_remaining > 0:
            cooldown_remaining -= 1

    if position is not None:
        last_date = dates[-1]
        close = closes.get(position.symbol, {}).get(last_date, position.entry_price)
        trades.append(
            {
                "entry_date": position.entry_date,
                "exit_date": last_date,
                "entry_symbol": position.symbol,
                "exit_symbol": position.symbol,
                "board": position.board,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": close,
                "exit_reason": "end_of_data",
                "entry_state": position.entry_state,
                "exit_state": position.state,
                "entry_exposure": position.entry_exposure,
                "exit_exposure": position.exposure,
                "max_exposure": position.max_exposure,
            }
        )
        exits += 1

    years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1e-9) if len(dates) >= 2 else 1.0
    summary = {
        "start_date": dates[0] if dates else "",
        "end_date": dates[-1] if dates else "",
        "total_return": equity - 1.0,
        "annualized_return": equity ** (1.0 / years) - 1.0 if equity > 0 else -1.0,
        "max_drawdown": abs(max_drawdown),
        "trades": len(trades),
        "entries": entries,
        "exits": exits,
        "exposure_changes": exposure_changes,
        "active_day_ratio": sum(1 for row in daily_rows if int(row["active_positions"]) > 0) / max(len(daily_rows), 1),
        "avg_exposure": sum(float(row["exposure"]) for row in daily_rows) / max(len(daily_rows), 1),
        "max_exposure": max((float(row["exposure"]) for row in daily_rows), default=0.0),
        "max_simultaneous_positions": max_positions,
        "missing_fee_symbols": sorted(missing_fee_symbols),
        "final_equity": equity,
    }
    return EventBacktestResult(summary=summary, trades=trades, equity_curve=equity_curve, daily_rows=daily_rows)
