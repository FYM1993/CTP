from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_split_execution_factors import _finite_float, _read_market_frame
from backtest.account_runner import AccountBacktestConfig, AccountCandidate, _fees_for_leg
from shared.position_sizing import positive_int_floor


DEFAULT_TRADES_CSV = Path("data/reports/backtest/trend_state_transitions_2022_2025_trades.csv")
DEFAULT_STATES_CSV = Path("data/reports/backtest/trend_state_transitions_2022_2025_states.csv")
DEFAULT_MARKET_CACHE_DIR = Path("data/cache/backtest")
DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/single_campaign_trader_2022_2025")


@dataclass(frozen=True)
class CampaignParams:
    name: str = "single_campaign"
    initial_equity: float = 1_000_000.0
    trial_margin_pct: float = 0.05
    medium_margin_pct: float = 0.15
    high_margin_pct: float = 0.30
    late_high_margin_pct: float = 0.10
    max_margin_pct: float = 0.30
    risk_per_trade_pct: float = 0.015
    trial_confirm_days: int = 5
    late_high_days: int = 10
    commission_multiplier: float = 1.01
    allow_confirm_start: bool = False
    max_campaigns_per_year: int = 0
    max_campaigns_per_transition_event: int = 0
    min_sniper_score: float = 0.0
    min_mainline_score: float = 0.0


@dataclass(frozen=True)
class CampaignBacktestResult:
    summary: dict[str, Any]
    campaigns: list[dict[str, Any]]
    events: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


@dataclass(frozen=True)
class CampaignOutputPaths:
    policy_csv: Path
    campaign_csv: Path
    event_csv: Path
    equity_csv: Path
    summary_json: Path
    report_md: Path


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def _fmt_dt(value: datetime) -> str:
    return value.isoformat(sep=" ")


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _state_event_time(date_value: Any) -> datetime:
    return datetime.fromisoformat(f"{_date_str(date_value)} 15:00:00")


def _candidate_from_row(row: dict[str, Any]) -> AccountCandidate:
    entry_price = _finite_float(row.get("entry_price"), 0.0)
    exit_price = _finite_float(row.get("exit_price") or row.get("planned_exit_price"), entry_price)
    multiplier = _finite_float(row.get("multiplier"), 0.0)
    notional = _finite_float(row.get("notional_per_lot"), 0.0)
    if notional <= 0:
        notional = entry_price * max(multiplier, 0.0)
    return AccountCandidate(
        symbol=str(row.get("symbol") or "").upper(),
        name=str(row.get("name") or row.get("symbol") or ""),
        direction=str(row.get("direction") or "long"),
        entry_time=str(row.get("entry_time") or ""),
        planned_exit_time=str(row.get("exit_time") or row.get("planned_exit_time") or ""),
        entry_price=float(entry_price),
        planned_exit_price=float(exit_price),
        planned_exit_reason=str(row.get("exit_reason") or row.get("planned_exit_reason") or ""),
        phase2_score=_finite_float(row.get("phase2_score"), 0.0),
        risk_per_lot=_finite_float(row.get("risk_per_lot"), 0.0),
        margin_per_lot=_finite_float(row.get("margin_per_lot"), 0.0),
        notional_per_lot=float(notional),
        multiplier=float(multiplier),
        pnl_ratio_price=_finite_float(row.get("pnl_ratio"), 0.0),
        tp1_hit=str(row.get("tp1_hit")).strip().lower() in {"1", "true", "yes", "y"},
        phase1_story_budget_margin_pct=_finite_float(row.get("trend_budget_margin_pct"), 0.0),
        phase1_story_budget_source="trend_state_campaign",
        commission_per_lot=_finite_float(row.get("commission_per_lot"), 0.0),
        commission_rate=_finite_float(row.get("commission_rate"), 0.0),
        close_today_commission_per_lot=(
            _finite_float(row.get("close_today_commission_per_lot"), 0.0)
            if str(row.get("close_today_commission_per_lot") or "").strip().lower() not in {"", "nan", "none"}
            else None
        ),
        trade_id=str(row.get("trade_id") or ""),
        entry_rr=_finite_float(row.get("entry_rr"), 0.0),
        entry_admission_rr=_finite_float(row.get("entry_admission_rr"), 0.0),
        medium_term_quality_score=_finite_float(row.get("trend_opportunity_quality_score"), 0.0),
        medium_term_entry_location_score=_finite_float(row.get("trend_structure_health_score"), 0.0),
        entry_adverse_deviation_r=_finite_float(row.get("entry_adverse_deviation_r"), 0.0),
        tp1_price=_finite_float(row.get("tp1_exit_price") or row.get("initial_tp1_price"), 0.0),
        tp1_time=str(row.get("tp1_exit_time") or ""),
    )


def _clip_score(value: Any, default: float = 0.0) -> float:
    number = _finite_float(value, default)
    if not math.isfinite(number):
        return float(default)
    return float(max(0.0, min(100.0, number)))


def sniper_entry_score(row: dict[str, Any]) -> float:
    phase2_component = min(abs(_finite_float(row.get("phase2_score"), 0.0)), 80.0) / 80.0 * 100.0
    admission_component = min(max(_finite_float(row.get("entry_admission_rr"), 0.0), 0.0), 6.0) / 6.0 * 100.0
    score = (
        0.25 * _clip_score(row.get("trend_opportunity_quality_score"))
        + 0.20 * _clip_score(row.get("state_score"))
        + 0.20 * _clip_score(row.get("trend_structure_health_score"))
        + 0.15 * _clip_score(row.get("trend_remaining_space_score"))
        + 0.10 * _clip_score(row.get("trend_participation_score"))
        + 0.05 * phase2_component
        + 0.05 * admission_component
    )
    return float(score)


def _initial_target_pct(row: dict[str, Any], params: CampaignParams) -> tuple[float, str]:
    stage = str(row.get("transition_stage") or "")
    if stage == "trial_low_to_medium":
        return float(params.trial_margin_pct), "trial_start"
    if bool(params.allow_confirm_start) and stage == "confirm_medium_to_high":
        return float(params.high_margin_pct), "confirm_start"
    return 0.0, "not_primary_campaign_start"


def _target_lots(candidate: AccountCandidate, equity: float, target_margin_pct: float, params: CampaignParams) -> int:
    capped_pct = max(0.0, min(float(target_margin_pct), float(params.max_margin_pct)))
    if candidate.margin_per_lot <= 0:
        return 0
    margin_lots = positive_int_floor(float(equity) * capped_pct / float(candidate.margin_per_lot))
    risk_lots = (
        positive_int_floor(float(equity) * float(params.risk_per_trade_pct) / float(candidate.risk_per_lot))
        if candidate.risk_per_lot > 0 and params.risk_per_trade_pct > 0
        else 1_000_000_000
    )
    return int(min(margin_lots, risk_lots))


def _gross_for_leg(candidate: AccountCandidate, leg: dict[str, Any], exit_price: float) -> float:
    lots = int(leg.get("lots") or 0)
    entry_price = _finite_float(leg.get("entry_price"), candidate.entry_price)
    if candidate.direction == "short":
        return float(entry_price - float(exit_price)) * float(candidate.multiplier) * lots
    return float(float(exit_price) - entry_price) * float(candidate.multiplier) * lots


def _current_lots(legs: list[dict[str, Any]]) -> int:
    return sum(int(leg.get("lots") or 0) for leg in legs)


def _mark_unrealized(candidate: AccountCandidate, legs: list[dict[str, Any]], price: float) -> float:
    return float(sum(_gross_for_leg(candidate, leg, float(price)) for leg in legs))


def _marked_equity(candidate: AccountCandidate, legs: list[dict[str, Any]], cash_equity: float, price: float) -> float:
    return float(cash_equity + _mark_unrealized(candidate, legs, float(price)))


def _margin_cap_lots(candidate: AccountCandidate, equity: float, params: CampaignParams) -> int:
    if candidate.margin_per_lot <= 0 or params.max_margin_pct <= 0:
        return 0
    return int(positive_int_floor(max(float(equity), 0.0) * float(params.max_margin_pct) / float(candidate.margin_per_lot)))


def _close_lots(
    *,
    candidate: AccountCandidate,
    legs: list[dict[str, Any]],
    lots_to_close: int,
    exit_time: str,
    exit_price: float,
    fee_config: AccountBacktestConfig,
) -> tuple[int, float, float]:
    remaining = int(lots_to_close)
    closed = 0
    gross = 0.0
    fees = 0.0
    while remaining > 0 and legs:
        leg = legs[-1]
        leg_lots = int(leg.get("lots") or 0)
        take = min(remaining, leg_lots)
        close_leg = dict(leg)
        close_leg["lots"] = take
        gross += _gross_for_leg(candidate, close_leg, float(exit_price))
        fees += _fees_for_leg(
            candidate,
            take,
            fee_config,
            entry_time=str(leg.get("entry_time") or candidate.entry_time),
            entry_price=_finite_float(leg.get("entry_price"), candidate.entry_price),
            exit_time=exit_time,
            exit_price=float(exit_price),
        )
        leg["lots"] = leg_lots - take
        if int(leg["lots"]) <= 0:
            legs.pop()
        remaining -= take
        closed += take
    return int(closed), float(gross), float(fees)


def _resize_position(
    *,
    candidate: AccountCandidate,
    legs: list[dict[str, Any]],
    target_lots: int,
    action: str,
    time: str,
    price: float,
    cash_equity: float,
    fee_config: AccountBacktestConfig,
    events: list[dict[str, Any]],
) -> tuple[float, int]:
    current = _current_lots(legs)
    target = max(int(target_lots), 0)
    realized_gross = 0.0
    realized_fees = 0.0
    changed_lots = target - current
    if changed_lots > 0:
        legs.append({"lots": changed_lots, "entry_price": float(price), "entry_time": str(time)})
    elif changed_lots < 0:
        _, realized_gross, realized_fees = _close_lots(
            candidate=candidate,
            legs=legs,
            lots_to_close=abs(changed_lots),
            exit_time=str(time),
            exit_price=float(price),
            fee_config=fee_config,
        )
        cash_equity += realized_gross - realized_fees
    events.append(
        {
            "time": str(time),
            "symbol": candidate.symbol,
            "direction": candidate.direction,
            "trade_id": candidate.trade_id,
            "action": action,
            "previous_lots": int(current),
            "target_lots": int(target),
            "changed_lots": int(changed_lots),
            "price": float(price),
            "cash_equity": float(cash_equity),
            "realized_gross_pnl": float(realized_gross),
            "realized_fees": float(realized_fees),
        }
    )
    return float(cash_equity), int(target)


def _enforce_margin_cap(
    *,
    candidate: AccountCandidate,
    legs: list[dict[str, Any]],
    time: str,
    price: float,
    cash_equity: float,
    params: CampaignParams,
    fee_config: AccountBacktestConfig,
    events: list[dict[str, Any]],
) -> float:
    current = _current_lots(legs)
    if current <= 0:
        return float(cash_equity)
    allowed = _margin_cap_lots(
        candidate,
        _marked_equity(candidate, legs, cash_equity, float(price)),
        params,
    )
    if allowed >= current:
        return float(cash_equity)
    cash_equity, _ = _resize_position(
        candidate=candidate,
        legs=legs,
        target_lots=allowed,
        action="reduce_margin_cap",
        time=str(time),
        price=float(price),
        cash_equity=cash_equity,
        fee_config=fee_config,
        events=events,
    )
    return float(cash_equity)


def _max_drawdown(equity_curve: list[dict[str, Any]]) -> tuple[float, str, str]:
    peak = -math.inf
    peak_time = ""
    max_dd = 0.0
    max_start = ""
    max_end = ""
    for row in equity_curve:
        equity = _finite_float(row.get("equity"), 0.0)
        if equity > peak:
            peak = equity
            peak_time = str(row.get("time") or "")
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                max_start = peak_time
                max_end = str(row.get("time") or "")
    return float(max_dd), max_start, max_end


def _rows_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("symbol") or "").upper(), str(row.get("direction") or ""))
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: str(item.get("date") or ""))
    return grouped


def _state_close(row: dict[str, Any], fallback: float) -> float:
    close = _finite_float(row.get("close"), math.nan)
    return float(close) if math.isfinite(close) and close > 0 else float(fallback)


def _append_mark(
    *,
    equity_curve: list[dict[str, Any]],
    time: str,
    cash_equity: float,
    candidate: AccountCandidate,
    legs: list[dict[str, Any]],
    price: float,
    event: str,
) -> None:
    lots = _current_lots(legs)
    marked = float(cash_equity + _mark_unrealized(candidate, legs, float(price)))
    equity_curve.append(
        {
            "time": str(time),
            "equity": marked,
            "cash_equity": float(cash_equity),
            "active_margin": float(lots * candidate.margin_per_lot),
            "active_lots": int(lots),
            "event": event,
        }
    )


def _simulate_campaign(
    *,
    source_row: dict[str, Any],
    states: list[dict[str, Any]],
    params: CampaignParams,
    starting_equity: float,
) -> tuple[float, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], str]:
    candidate = _candidate_from_row(source_row)
    initial_pct, start_reason = _initial_target_pct(source_row, params)
    if initial_pct <= 0:
        return float(starting_equity), None, [], [], start_reason
    initial_lots = _target_lots(candidate, starting_equity, initial_pct, params)
    if initial_lots <= 0:
        return float(starting_equity), None, [], [], "target_below_one_lot"

    fee_config = AccountBacktestConfig(commission_multiplier=float(params.commission_multiplier))
    entry_dt = _parse_dt(candidate.entry_time)
    planned_exit_dt = _parse_dt(candidate.planned_exit_time)
    cash_equity = float(starting_equity)
    legs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    campaign_start_equity = float(starting_equity)
    max_lots = 0
    max_margin_pct = 0.0
    raised_to_medium = start_reason == "confirm_start"
    reached_high = start_reason == "confirm_start"
    high_since: datetime | None = entry_dt if reached_high else None
    late_reduced = False
    exit_reason = str(candidate.planned_exit_reason)
    exit_time = candidate.planned_exit_time
    exit_price = float(candidate.planned_exit_price)

    cash_equity, _ = _resize_position(
        candidate=candidate,
        legs=legs,
        target_lots=initial_lots,
        action="open",
        time=candidate.entry_time,
        price=float(candidate.entry_price),
        cash_equity=cash_equity,
        fee_config=fee_config,
        events=events,
    )
    _append_mark(
        equity_curve=equity_curve,
        time=candidate.entry_time,
        cash_equity=cash_equity,
        candidate=candidate,
        legs=legs,
        price=float(candidate.entry_price),
        event="open",
    )
    max_lots = max(max_lots, _current_lots(legs))
    max_margin_pct = max(max_margin_pct, _current_lots(legs) * candidate.margin_per_lot / max(campaign_start_equity, 1.0))

    for state in states:
        state_time = _state_event_time(state.get("date"))
        if state_time <= entry_dt:
            continue
        if state_time >= planned_exit_dt:
            break
        bucket = str(state.get("bucket") or "")
        price = _state_close(state, candidate.entry_price)
        days_since_entry = (state_time.date() - entry_dt.date()).days
        if bucket == "low":
            exit_reason = "trial_failed_state_low" if not raised_to_medium else "trend_state_invalidated_low"
            exit_time = _fmt_dt(state_time)
            exit_price = float(price)
            break

        if not raised_to_medium and days_since_entry >= int(params.trial_confirm_days):
            if bucket in {"medium", "high", "crowded"}:
                if bucket == "crowded":
                    target_pct = float(_current_lots(legs) * candidate.margin_per_lot / max(cash_equity, 1.0))
                    action = "crowded_no_add"
                else:
                    target_pct = float(params.high_margin_pct if bucket == "high" else params.medium_margin_pct)
                    action = "raise_to_high" if bucket == "high" else "raise_to_medium"
                target = _target_lots(candidate, cash_equity, target_pct, params)
                cash_equity, _ = _resize_position(
                    candidate=candidate,
                    legs=legs,
                    target_lots=target,
                    action=action,
                    time=_fmt_dt(state_time),
                    price=float(price),
                    cash_equity=cash_equity,
                    fee_config=fee_config,
                    events=events,
                )
                raised_to_medium = True
                reached_high = bucket in {"high", "crowded"}
                high_since = state_time if bucket in {"high", "crowded"} else high_since
                late_reduced = bool(bucket == "crowded")
            else:
                exit_reason = "trial_timeout_no_confirmation"
                exit_time = _fmt_dt(state_time)
                exit_price = float(price)
                break

        if bucket == "crowded" and raised_to_medium and not late_reduced:
            target = _target_lots(candidate, cash_equity, float(params.late_high_margin_pct), params)
            cash_equity, _ = _resize_position(
                candidate=candidate,
                legs=legs,
                target_lots=target,
                action="reduce_crowded",
                time=_fmt_dt(state_time),
                price=float(price),
                cash_equity=cash_equity,
                fee_config=fee_config,
                events=events,
            )
            late_reduced = True

        if bucket == "high":
            if high_since is None:
                high_since = state_time
            if not reached_high and raised_to_medium:
                target = _target_lots(candidate, cash_equity, float(params.high_margin_pct), params)
                cash_equity, _ = _resize_position(
                    candidate=candidate,
                    legs=legs,
                    target_lots=target,
                    action="raise_to_high",
                    time=_fmt_dt(state_time),
                    price=float(price),
                    cash_equity=cash_equity,
                    fee_config=fee_config,
                    events=events,
                )
                reached_high = True
            if (
                reached_high
                and not late_reduced
                and high_since is not None
                and (state_time.date() - high_since.date()).days >= int(params.late_high_days)
            ):
                target = _target_lots(candidate, cash_equity, float(params.late_high_margin_pct), params)
                cash_equity, _ = _resize_position(
                    candidate=candidate,
                    legs=legs,
                    target_lots=target,
                    action="reduce_late_high",
                    time=_fmt_dt(state_time),
                    price=float(price),
                    cash_equity=cash_equity,
                    fee_config=fee_config,
                    events=events,
                )
                late_reduced = True

        cash_equity = _enforce_margin_cap(
            candidate=candidate,
            legs=legs,
            time=_fmt_dt(state_time),
            price=float(price),
            cash_equity=cash_equity,
            params=params,
            fee_config=fee_config,
            events=events,
        )
        max_lots = max(max_lots, _current_lots(legs))
        max_margin_pct = max(max_margin_pct, _current_lots(legs) * candidate.margin_per_lot / max(campaign_start_equity, 1.0))
        _append_mark(
            equity_curve=equity_curve,
            time=_fmt_dt(state_time),
            cash_equity=cash_equity,
            candidate=candidate,
            legs=legs,
            price=float(price),
            event=f"mark_{bucket or 'missing'}",
        )

    if _current_lots(legs) > 0:
        closed_lots, gross, fees = _close_lots(
            candidate=candidate,
            legs=legs,
            lots_to_close=_current_lots(legs),
            exit_time=exit_time,
            exit_price=float(exit_price),
            fee_config=fee_config,
        )
        cash_equity += gross - fees
        events.append(
            {
                "time": exit_time,
                "symbol": candidate.symbol,
                "direction": candidate.direction,
                "trade_id": candidate.trade_id,
                "action": "exit",
                "previous_lots": int(closed_lots),
                "target_lots": 0,
                "changed_lots": -int(closed_lots),
                "price": float(exit_price),
                "cash_equity": float(cash_equity),
                "realized_gross_pnl": float(gross),
                "realized_fees": float(fees),
            }
        )
    _append_mark(
        equity_curve=equity_curve,
        time=exit_time,
        cash_equity=cash_equity,
        candidate=candidate,
        legs=legs,
        price=float(exit_price),
        event=f"exit_{exit_reason}",
    )
    net_pnl = float(cash_equity - starting_equity)
    campaign = {
        "trade_id": candidate.trade_id,
        "symbol": candidate.symbol,
        "direction": candidate.direction,
        "entry_time": candidate.entry_time,
        "entry_price": float(candidate.entry_price),
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "start_reason": start_reason,
        "planned_exit_reason": candidate.planned_exit_reason,
        "planned_exit_time": candidate.planned_exit_time,
        "planned_exit_price": float(candidate.planned_exit_price),
        "starting_equity": float(starting_equity),
        "ending_equity": float(cash_equity),
        "net_pnl": net_pnl,
        "return_pct": float(net_pnl / starting_equity) if starting_equity > 0 else 0.0,
        "max_lots": int(max_lots),
        "final_lots": int(_current_lots(legs)),
        "max_margin_pct": float(max_margin_pct),
        "raised_to_medium": bool(raised_to_medium),
        "reached_high": bool(reached_high),
        "late_reduced": bool(late_reduced),
        "source_transition_stage": str(source_row.get("transition_stage") or ""),
        "source_entry_path_bucket": str(source_row.get("entry_path_bucket") or ""),
        "source_entry_rr_bucket": str(source_row.get("entry_rr_bucket") or ""),
        "source_medium_quality_bucket": str(source_row.get("medium_quality_bucket") or ""),
        "sniper_entry_score": float(sniper_entry_score(source_row)),
    }
    return float(cash_equity), campaign, events, equity_curve, ""


def run_single_campaign_backtest(
    trade_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    params: CampaignParams,
) -> CampaignBacktestResult:
    equity = float(params.initial_equity)
    busy_until: datetime | None = None
    campaigns: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    campaigns_by_year: Counter[int] = Counter()
    campaigns_by_transition_event: Counter[str] = Counter()
    equity_curve: list[dict[str, Any]] = [
        {
            "time": "2022-01-01 00:00:00",
            "equity": float(equity),
            "cash_equity": float(equity),
            "active_margin": 0.0,
            "active_lots": 0,
            "event": "start",
        }
    ]
    states_by_key = _rows_by_key(state_rows)
    sorted_trades = sorted(
        trade_rows,
        key=lambda row: (
            str(row.get("entry_time") or ""),
            str(row.get("symbol") or ""),
            str(row.get("direction") or ""),
        ),
    )
    for row in sorted_trades:
        entry_time = _parse_dt(str(row.get("entry_time") or ""))
        if busy_until is not None and entry_time < busy_until:
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": "single_campaign_busy",
                    "busy_until": _fmt_dt(busy_until),
                }
            )
            continue
        entry_year = int(entry_time.year)
        transition_event_id = str(row.get("transition_event_id") or "")
        if (
            int(params.max_campaigns_per_transition_event) > 0
            and transition_event_id
            and campaigns_by_transition_event[transition_event_id] >= int(params.max_campaigns_per_transition_event)
        ):
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": "transition_event_quota_full",
                    "transition_event_id": transition_event_id,
                    "busy_until": "",
                }
            )
            continue
        if int(params.max_campaigns_per_year) > 0 and campaigns_by_year[entry_year] >= int(params.max_campaigns_per_year):
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": "yearly_campaign_quota_full",
                    "busy_until": "",
                    "sniper_entry_score": float(sniper_entry_score(row)),
                }
            )
            continue
        entry_score = sniper_entry_score(row)
        if entry_score < float(params.min_sniper_score):
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": "sniper_score_below_threshold",
                    "busy_until": "",
                    "sniper_entry_score": float(entry_score),
                }
            )
            continue
        mainline_score = _finite_float(row.get("mainline_score"), _finite_float(row.get("state_score"), 0.0))
        if mainline_score < float(params.min_mainline_score):
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": "mainline_score_below_threshold",
                    "busy_until": "",
                    "mainline_score": float(mainline_score),
                    "min_mainline_score": float(params.min_mainline_score),
                }
            )
            continue
        key = (str(row.get("symbol") or "").upper(), str(row.get("direction") or ""))
        equity_after, campaign, campaign_events, campaign_curve, skip_reason = _simulate_campaign(
            source_row=row,
            states=states_by_key.get(key, []),
            params=params,
            starting_equity=equity,
        )
        if campaign is None:
            skipped.append(
                {
                    "trade_id": str(row.get("trade_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time": str(row.get("entry_time") or ""),
                    "skip_reason": skip_reason,
                    "busy_until": "",
                }
            )
            continue
        equity = float(equity_after)
        campaigns.append(campaign)
        campaigns_by_year[entry_year] += 1
        if transition_event_id:
            campaigns_by_transition_event[transition_event_id] += 1
        events.extend(campaign_events)
        equity_curve.extend(campaign_curve)
        busy_until = _parse_dt(str(campaign["exit_time"]))
    max_dd, dd_start, dd_end = _max_drawdown(equity_curve)
    wins = sum(1 for row in campaigns if float(row.get("net_pnl") or 0.0) > 0)
    planned_tp2 = sum(1 for row in campaigns if str(row.get("planned_exit_reason") or "") == "tp2")
    actual_tp2 = sum(1 for row in campaigns if str(row.get("exit_reason") or "") == "tp2")
    early_exits = sum(1 for row in campaigns if str(row.get("exit_reason") or "") not in {"tp2", "stop", "daily_reverse_confirmed"})
    action_counts = Counter(str(row.get("action") or "") for row in events)
    avg_margin = (
        sum(_finite_float(row.get("active_margin"), 0.0) / max(_finite_float(row.get("equity"), 1.0), 1.0) for row in equity_curve)
        / len(equity_curve)
        if equity_curve
        else 0.0
    )
    summary = {
        "policy": params.name,
        "initial_equity": float(params.initial_equity),
        "ending_equity": float(equity),
        "net_profit": float(equity - params.initial_equity),
        "return_pct": float((equity - params.initial_equity) / params.initial_equity) if params.initial_equity > 0 else 0.0,
        "max_drawdown_pct": float(max_dd),
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "campaigns": int(len(campaigns)),
        "wins": int(wins),
        "win_rate": float(wins / len(campaigns)) if campaigns else 0.0,
        "skipped": int(len(skipped)),
        "skipped_busy": sum(1 for row in skipped if row.get("skip_reason") == "single_campaign_busy"),
        "skipped_year_quota": sum(1 for row in skipped if row.get("skip_reason") == "yearly_campaign_quota_full"),
        "skipped_transition_event_quota": sum(1 for row in skipped if row.get("skip_reason") == "transition_event_quota_full"),
        "skipped_sniper_score": sum(1 for row in skipped if row.get("skip_reason") == "sniper_score_below_threshold"),
        "skipped_mainline_score": sum(1 for row in skipped if row.get("skip_reason") == "mainline_score_below_threshold"),
        "early_campaign_exits": int(early_exits),
        "tp2_campaigns": int(actual_tp2),
        "tp2_capture_rate": float(actual_tp2 / planned_tp2) if planned_tp2 else 0.0,
        "raise_to_medium_events": int(action_counts.get("raise_to_medium", 0)),
        "raise_to_medium_add_events": sum(
            1 for row in events if row.get("action") == "raise_to_medium" and int(row.get("changed_lots") or 0) > 0
        ),
        "raise_to_high_events": int(action_counts.get("raise_to_high", 0)),
        "raise_to_high_add_events": sum(
            1 for row in events if row.get("action") == "raise_to_high" and int(row.get("changed_lots") or 0) > 0
        ),
        "reduce_late_high_events": int(action_counts.get("reduce_late_high", 0)),
        "reduce_late_high_cut_events": sum(
            1 for row in events if row.get("action") == "reduce_late_high" and int(row.get("changed_lots") or 0) < 0
        ),
        "reduce_crowded_events": int(action_counts.get("reduce_crowded", 0)),
        "reduce_crowded_cut_events": sum(
            1 for row in events if row.get("action") == "reduce_crowded" and int(row.get("changed_lots") or 0) < 0
        ),
        "crowded_no_add_events": int(action_counts.get("crowded_no_add", 0)),
        "reduce_margin_cap_events": int(action_counts.get("reduce_margin_cap", 0)),
        "avg_margin_pct_observed": float(avg_margin),
        "max_margin_pct_observed": max(
            (_finite_float(row.get("active_margin"), 0.0) / max(_finite_float(row.get("equity"), 1.0), 1.0) for row in equity_curve),
            default=0.0,
        ),
        "annualized_return_pct": float((equity / params.initial_equity) ** 0.25 - 1.0) if params.initial_equity > 0 and equity > 0 else 0.0,
        **asdict(params),
    }
    return CampaignBacktestResult(summary=summary, campaigns=campaigns, events=events, equity_curve=equity_curve, skipped=skipped)


class MarketPriceLookup:
    def __init__(self, market_cache_dir: Path) -> None:
        self.market_cache_dir = Path(market_cache_dir)
        self.frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.price_cache: dict[str, dict[str, float]] = {}

    def close_for(self, symbol: str, date_value: Any) -> float:
        key = str(symbol).upper()
        if key not in self.price_cache:
            daily_df = _read_market_frame(key, "daily", self.market_cache_dir, self.frame_cache)
            prices: dict[str, float] = {}
            if not daily_df.empty and {"date", "close"}.issubset(daily_df.columns):
                data = daily_df.copy()
                data["date"] = pd.to_datetime(data["date"], errors="coerce")
                data = data.dropna(subset=["date"])
                for row in data.to_dict("records"):
                    close = _finite_float(row.get("close"), math.nan)
                    if math.isfinite(close) and close > 0:
                        prices[_date_str(row.get("date"))] = float(close)
            self.price_cache[key] = prices
        return float(self.price_cache.get(key, {}).get(_date_str(date_value), math.nan))


def enrich_state_closes(state_rows: list[dict[str, Any]], market_cache_dir: Path = DEFAULT_MARKET_CACHE_DIR) -> list[dict[str, Any]]:
    lookup = MarketPriceLookup(market_cache_dir)
    out: list[dict[str, Any]] = []
    for row in state_rows:
        item = dict(row)
        close = _finite_float(item.get("close"), math.nan)
        if not math.isfinite(close) or close <= 0:
            close = lookup.close_for(str(item.get("symbol") or ""), item.get("date"))
        if math.isfinite(close) and close > 0:
            item["close"] = float(close)
        out.append(item)
    return out


def iter_baseline_campaign_params() -> Iterable[CampaignParams]:
    for allow_confirm_start in (False, True):
        mode = "trial_first" if not allow_confirm_start else "trial_or_confirm"
        for trial_pct in (0.03, 0.05, 0.08):
            for medium_pct in (0.15, 0.20):
                for late_pct in (0.10, 0.15):
                    for confirm_days in (3, 5, 10):
                        for late_days in (10, 20):
                            name = (
                                f"{mode}_trial{int(trial_pct*100)}_medium{int(medium_pct*100)}_"
                                f"high30_late{int(late_pct*100)}_confirm{confirm_days}_lateDays{late_days}"
                            )
                            yield CampaignParams(
                                name=name,
                                trial_margin_pct=trial_pct,
                                medium_margin_pct=medium_pct,
                                high_margin_pct=0.30,
                                late_high_margin_pct=late_pct,
                                trial_confirm_days=confirm_days,
                                late_high_days=late_days,
                                allow_confirm_start=allow_confirm_start,
                            )


def iter_aggressive_campaign_params() -> Iterable[CampaignParams]:
    for allow_confirm_start in (False, True):
        start_mode = "trial_first" if not allow_confirm_start else "trial_or_confirm"
        for max_per_year in (1, 2):
            for min_score in (0.0, 55.0, 60.0, 65.0, 70.0):
                for trial_pct in (0.05, 0.08, 0.12):
                    for medium_pct in (0.30, 0.50):
                        for high_pct in (0.50, 0.80, 1.00):
                            for late_pct in (0.20, 0.30, 0.50):
                                if late_pct > high_pct:
                                    continue
                                for risk_pct in (0.03, 0.05, 0.08, 0.12, 0.20, 0.30):
                                    for confirm_days in (3, 5):
                                        for late_days in (10, 20):
                                            name = (
                                                f"{start_mode}_sniper_y{max_per_year}_score{int(min_score)}_"
                                                f"trial{int(trial_pct*100)}_medium{int(medium_pct*100)}_"
                                                f"high{int(high_pct*100)}_late{int(late_pct*100)}_"
                                                f"risk{int(risk_pct*100)}_confirm{confirm_days}_lateDays{late_days}"
                                            )
                                            yield CampaignParams(
                                                name=name,
                                                trial_margin_pct=trial_pct,
                                                medium_margin_pct=medium_pct,
                                                high_margin_pct=high_pct,
                                                late_high_margin_pct=late_pct,
                                                max_margin_pct=1.0,
                                                risk_per_trade_pct=risk_pct,
                                                trial_confirm_days=confirm_days,
                                                late_high_days=late_days,
                                                allow_confirm_start=allow_confirm_start,
                                                max_campaigns_per_year=max_per_year,
                                                min_sniper_score=min_score,
                                            )


def iter_selective_campaign_params() -> Iterable[CampaignParams]:
    for min_mainline_score in (55.0, 60.0, 65.0, 70.0, 75.0, 80.0):
        for min_score in (65.0, 70.0, 75.0):
            for trial_pct in (0.03, 0.05, 0.08):
                for medium_pct in (0.30, 0.50):
                    for high_pct in (0.50, 0.80, 1.00):
                        for late_pct in (0.20, 0.30):
                            if late_pct > high_pct:
                                continue
                            for risk_pct in (0.08, 0.12, 0.20, 0.30):
                                for confirm_days in (3, 5):
                                    name = (
                                        f"selective_noquota_mainline{int(min_mainline_score)}_score{int(min_score)}_"
                                        f"trial{int(trial_pct*100)}_medium{int(medium_pct*100)}_"
                                        f"high{int(high_pct*100)}_late{int(late_pct*100)}_"
                                        f"risk{int(risk_pct*100)}_confirm{confirm_days}"
                                    )
                                    yield CampaignParams(
                                        name=name,
                                        trial_margin_pct=trial_pct,
                                        medium_margin_pct=medium_pct,
                                        high_margin_pct=high_pct,
                                        late_high_margin_pct=late_pct,
                                        max_margin_pct=1.0,
                                        risk_per_trade_pct=risk_pct,
                                        trial_confirm_days=confirm_days,
                                        late_high_days=10,
                                        allow_confirm_start=True,
                                        max_campaigns_per_year=0,
                                        max_campaigns_per_transition_event=1,
                                        min_sniper_score=min_score,
                                        min_mainline_score=min_mainline_score,
                                    )


def iter_selective_constrained_campaign_params() -> Iterable[CampaignParams]:
    for max_margin_pct in (0.30, 0.50):
        medium_options = (0.15, 0.20, 0.30) if max_margin_pct <= 0.30 else (0.20, 0.30, 0.50)
        high_options = (0.30,) if max_margin_pct <= 0.30 else (0.30, 0.50)
        late_options = (0.05, 0.10, 0.20) if max_margin_pct <= 0.30 else (0.10, 0.20, 0.30)
        for min_mainline_score in (55.0, 60.0, 65.0, 70.0, 75.0, 80.0):
            for min_score in (70.0, 75.0, 80.0):
                for trial_pct in (0.03, 0.05, 0.08):
                    if trial_pct > max_margin_pct:
                        continue
                    for medium_pct in medium_options:
                        if medium_pct > max_margin_pct:
                            continue
                        for high_pct in high_options:
                            if high_pct > max_margin_pct or medium_pct > high_pct:
                                continue
                            for late_pct in late_options:
                                if late_pct > high_pct:
                                    continue
                                for risk_pct in (0.05, 0.08, 0.12, 0.20):
                                    for confirm_days in (3, 5):
                                        name = (
                                            f"selective_cap{int(max_margin_pct*100)}_mainline{int(min_mainline_score)}_"
                                            f"score{int(min_score)}_trial{int(trial_pct*100)}_"
                                            f"medium{int(medium_pct*100)}_high{int(high_pct*100)}_"
                                            f"late{int(late_pct*100)}_risk{int(risk_pct*100)}_"
                                            f"confirm{confirm_days}"
                                        )
                                        yield CampaignParams(
                                            name=name,
                                            trial_margin_pct=trial_pct,
                                            medium_margin_pct=medium_pct,
                                            high_margin_pct=high_pct,
                                            late_high_margin_pct=late_pct,
                                            max_margin_pct=max_margin_pct,
                                            risk_per_trade_pct=risk_pct,
                                            trial_confirm_days=confirm_days,
                                            late_high_days=10,
                                            allow_confirm_start=True,
                                            max_campaigns_per_year=0,
                                            max_campaigns_per_transition_event=1,
                                            min_sniper_score=min_score,
                                            min_mainline_score=min_mainline_score,
                                        )


def iter_campaign_params(profile: str = "baseline") -> Iterable[CampaignParams]:
    resolved = str(profile or "").strip().lower()
    if resolved == "aggressive":
        yield from iter_aggressive_campaign_params()
        return
    if resolved == "selective":
        yield from iter_selective_campaign_params()
        return
    if resolved == "selective_constrained":
        yield from iter_selective_constrained_campaign_params()
        return
    yield from iter_baseline_campaign_params()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _pct(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _policy_rank(row: dict[str, Any]) -> tuple[float, float]:
    return (_finite_float(row.get("return_pct"), 0.0), -_finite_float(row.get("max_drawdown_pct"), 0.0))


def render_markdown(policy_rows: list[dict[str, Any]], selected: CampaignBacktestResult) -> str:
    top_return = sorted(policy_rows, key=_policy_rank, reverse=True)[:12]
    controlled = [
        row
        for row in sorted(policy_rows, key=_policy_rank, reverse=True)
        if _finite_float(row.get("max_drawdown_pct"), 1.0) <= 0.25
    ][:12]
    lines = [
        "# 单主线趋势交易员回测 2022-2025",
        "",
        "口径：同一时间只允许一个主线机会在场；确认前试错，状态维持/升级后加仓，high 持续后降仓；只做回测分析，不改实盘默认规则。",
        "",
        "## 收益最高参数",
        "",
        "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | 提前退出 | TP2捕获 | 平均占用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_return:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['campaigns']} | {_pct(row['win_rate'])} | {row['early_campaign_exits']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 回撤不超过 25% 的较优参数",
            "",
            "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | 提前退出 | TP2捕获 | 平均占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in controlled:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['campaigns']} | {_pct(row['win_rate'])} | {row['early_campaign_exits']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 选中参数的年份表现",
            "",
            "| 年份 | 交易数 | 收益贡献 | 胜率 | 提前退出 | TP2数 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_year: dict[int, list[dict[str, Any]]] = {}
    for campaign in selected.campaigns:
        by_year.setdefault(pd.Timestamp(campaign["entry_time"]).year, []).append(campaign)
    for year in sorted(by_year):
        rows = by_year[year]
        wins = sum(1 for row in rows if _finite_float(row.get("net_pnl"), 0.0) > 0)
        pnl = sum(_finite_float(row.get("net_pnl"), 0.0) for row in rows)
        early = sum(1 for row in rows if str(row.get("exit_reason") or "") not in {"tp2", "stop", "daily_reverse_confirmed"})
        tp2 = sum(1 for row in rows if str(row.get("exit_reason") or "") == "tp2")
        lines.append(f"| {year} | {len(rows)} | {_pct(pnl / selected.summary['initial_equity'])} | {_pct(wins / len(rows))} | {early} | {tp2} |")
    lines.extend(
        [
            "",
            "## 选中参数的执行诊断",
            "",
            f"- medium 升仓信号 {selected.summary.get('raise_to_medium_events', 0)} 次，其中真正增加手数 {selected.summary.get('raise_to_medium_add_events', 0)} 次。",
            f"- high 升仓信号 {selected.summary.get('raise_to_high_events', 0)} 次，其中真正增加手数 {selected.summary.get('raise_to_high_add_events', 0)} 次。",
            f"- 后排 high 降仓信号 {selected.summary.get('reduce_late_high_events', 0)} 次，其中真正减少手数 {selected.summary.get('reduce_late_high_cut_events', 0)} 次。",
            (
                f"- 当前最大实际保证金占用 {_pct(selected.summary.get('max_margin_pct_observed'))}，"
                f"本参数允许最高目标保证金 {_pct(selected.summary.get('max_margin_pct'))}；"
                "实际手数仍会同时受单笔止损风险预算约束。"
            ),
            "",
            "## 使用边界",
            "",
            "- 试错仓的基本面/舆论强弱无法从历史缓存真实还原，所以用参数网格模拟。",
            "- 这是单主线 campaign 口径，不是多品种组合；被跳过的重叠信号不会再进入账户。",
            "- 加仓和降仓按每日收盘价近似，入场仍使用原 Phase2 触发价。",
            "",
        ]
    )
    return "\n".join(lines)


def run_parameter_sweep(
    trade_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    *,
    profile: str = "baseline",
) -> tuple[list[dict[str, Any]], CampaignBacktestResult]:
    policy_rows: list[dict[str, Any]] = []
    results: list[CampaignBacktestResult] = []
    for params in iter_campaign_params(profile):
        result = run_single_campaign_backtest(trade_rows, state_rows, params)
        policy_rows.append(dict(result.summary))
        results.append(result)
    selected = max(results, key=lambda item: _policy_rank(item.summary)) if results else run_single_campaign_backtest([], [], CampaignParams())
    return policy_rows, selected


def write_outputs(
    *,
    output_prefix: Path,
    policy_rows: list[dict[str, Any]],
    selected: CampaignBacktestResult,
) -> CampaignOutputPaths:
    prefix = Path(output_prefix)
    paths = CampaignOutputPaths(
        policy_csv=prefix.with_name(f"{prefix.name}_policy.csv"),
        campaign_csv=prefix.with_name(f"{prefix.name}_campaigns.csv"),
        event_csv=prefix.with_name(f"{prefix.name}_events.csv"),
        equity_csv=prefix.with_name(f"{prefix.name}_equity.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.policy_csv, policy_rows)
    _write_csv(paths.campaign_csv, selected.campaigns)
    _write_csv(paths.event_csv, selected.events)
    _write_csv(paths.equity_csv, selected.equity_curve)
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(
        json.dumps(
            {
                "selected_summary": selected.summary,
                "policy_rows": policy_rows,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    paths.report_md.write_text(render_markdown(policy_rows, selected), encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest a single-mainline trend campaign trader")
    parser.add_argument("--trades-csv", type=Path, default=DEFAULT_TRADES_CSV)
    parser.add_argument("--states-csv", type=Path, default=DEFAULT_STATES_CSV)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--profile", choices=("baseline", "aggressive", "selective", "selective_constrained"), default="baseline")
    args = parser.parse_args()

    trade_rows = pd.read_csv(args.trades_csv).to_dict("records")
    raw_states = pd.read_csv(args.states_csv).to_dict("records")
    state_rows = enrich_state_closes(raw_states, args.market_cache_dir)
    policy_rows, selected = run_parameter_sweep(trade_rows, state_rows, profile=args.profile)
    paths = write_outputs(output_prefix=args.output_prefix, policy_rows=policy_rows, selected=selected)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
