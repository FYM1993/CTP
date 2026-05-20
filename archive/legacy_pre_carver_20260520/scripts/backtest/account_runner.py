from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backtest.models import TradeRecord
from shared.position_sizing import positive_int_floor


@dataclass(frozen=True, slots=True)
class AccountBacktestConfig:
    initial_equity: float = 1_000_000.0
    max_portfolio_margin_pct: float = 0.30
    position_budget_source: str = "phase2_score"
    reserve_planned_second_entry_margin: bool = False
    min_phase1_story_budget_margin_pct: float = 0.0
    min_phase2_abs_score: float = 0.0
    risk_per_trade_pct: float | None = 0.015
    ordinary_risk_per_trade_pct: float | None = None
    strong_risk_per_trade_pct: float | None = None
    strong_first_target_rr: float = 2.0
    strong_second_target_rr: float = 3.0
    execution_policy: str = "fixed"
    split_initial_fraction: float = 1.0
    split_second_entry_trigger: str = "none"
    conditional_direct_quality_min: float = 60.0
    conditional_direct_entry_rr_min: float = 1.5
    conditional_direct_admission_rr_min: float = 3.0
    conditional_direct_entry_adverse_r_max: float = 0.25
    conditional_scale_quality_min: float = 50.0
    conditional_scale_entry_rr_min: float = 1.0
    conditional_scale_admission_rr_min: float = 2.8
    conditional_scale_entry_adverse_r_max: float = 0.75
    conditional_scale_initial_fraction: float = 0.70
    conditional_trial_quality_min: float = 45.0
    conditional_trial_admission_rr_min: float = 2.5
    conditional_trial_entry_adverse_r_max: float = 1.0
    conditional_trial_initial_fraction: float = 0.30
    commission_multiplier: float = 1.01
    allow_replacement: bool = True
    min_replacement_hold_hours: float = 0.0
    scope: str = "long_trend_core continuous 2025, long+short candidates, one active trade per symbol"


@dataclass(frozen=True, slots=True)
class AccountCandidate:
    symbol: str
    name: str
    direction: str
    entry_time: str
    planned_exit_time: str
    entry_price: float
    planned_exit_price: float
    planned_exit_reason: str
    phase2_score: float
    risk_per_lot: float
    margin_per_lot: float
    notional_per_lot: float
    multiplier: float
    pnl_ratio_price: float
    tp1_hit: bool
    phase1_story_budget_margin_pct: float = 0.0
    phase1_story_budget_source: str = ""
    commission_per_lot: float = 0.0
    commission_rate: float = 0.0
    close_today_commission_per_lot: float | None = None
    trade_id: str = ""
    entry_rr: float = 0.0
    entry_admission_rr: float = 0.0
    medium_term_quality_score: float = 0.0
    medium_term_entry_location_score: float = 0.0
    entry_adverse_deviation_r: float = 0.0
    trend_phase: str = ""
    tp1_price: float = 0.0
    tp1_time: str = ""


@dataclass(frozen=True, slots=True)
class AccountBacktestResult:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    by_symbol: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ExecutionSelection:
    profile: str
    initial_fraction: float
    second_entry_trigger: str
    skip_reason: str = ""


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def _fmt_dt(value: datetime) -> str:
    return value.isoformat(sep=" ")


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clamp_fraction(value: object, default: float = 1.0) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(fraction):
        return default
    return min(max(fraction, 0.0), 1.0)


def _entry_adverse_deviation_r_from_meta(meta: dict[str, Any], direction: str) -> float:
    explicit = _as_float(meta.get("entry_adverse_deviation_r"), math.nan)
    if math.isfinite(explicit):
        return max(explicit, 0.0)
    planned_entry = _as_float(meta.get("planned_entry_ref"), 0.0)
    planned_stop = _as_float(meta.get("planned_stop") or meta.get("initial_stop_price"), 0.0)
    deviation = _as_float(meta.get("entry_trigger_deviation"), 0.0)
    risk_distance = abs(planned_entry - planned_stop)
    if risk_distance <= 0:
        return 0.0
    adverse = deviation if str(direction) == "long" else -deviation
    return float(max(adverse, 0.0) / risk_distance)


def target_margin_pct_from_phase2_score(score: float) -> float:
    raw = (abs(float(score)) - 25.0) / 35.0 * 0.30
    return float(max(0.03, min(0.30, raw)))


def _position_budget_source(config: AccountBacktestConfig) -> str:
    source = str(config.position_budget_source or "phase2_score").strip().lower()
    if source in {"phase1_story", "story", "phase1"}:
        return "phase1_story"
    return "phase2_score"


def _bounded_story_budget_margin_pct(value: object, default: float = 0.03) -> float:
    pct = _as_float(value, default)
    if not math.isfinite(pct) or pct <= 0:
        pct = default
    return float(max(0.0, min(0.30, pct)))


def _story_budget_margin_pct_from_meta(meta: dict[str, Any]) -> float:
    return _bounded_story_budget_margin_pct(
        meta.get("phase1_story_budget_margin_pct")
        or meta.get("trend_story_budget_margin_pct")
        or meta.get("story_budget_margin_pct"),
        default=0.03,
    )


def _target_margin_pct(candidate: AccountCandidate, config: AccountBacktestConfig) -> float:
    if _position_budget_source(config) == "phase1_story":
        return _bounded_story_budget_margin_pct(candidate.phase1_story_budget_margin_pct, default=0.03)
    return target_margin_pct_from_phase2_score(candidate.phase2_score)


def candidate_from_trade(
    trade: TradeRecord,
    *,
    name: str = "",
    commission_per_lot: float = 0.0,
    commission_rate: float = 0.0,
    close_today_commission_per_lot: float | None = None,
    margin_rate: float | None = None,
) -> AccountCandidate:
    meta = trade.meta or {}
    multiplier = _as_float(meta.get("contract_multiplier"), 0.0)
    notional_per_lot = float(trade.entry_price) * max(multiplier, 0.0)
    if notional_per_lot <= 0:
        notional_per_lot = _as_float(meta.get("notional_per_lot"), 0.0)
    margin_per_lot = _as_float(meta.get("execution_margin_per_lot"), _as_float(meta.get("margin_per_lot"), 0.0))
    if margin_rate is not None and notional_per_lot > 0:
        margin_per_lot = float(notional_per_lot) * max(float(margin_rate), 0.0)
    return AccountCandidate(
        symbol=trade.symbol,
        name=name or str(meta.get("name") or trade.symbol),
        direction=trade.direction,
        entry_time=trade.entry_time,
        planned_exit_time=trade.exit_time,
        entry_price=float(trade.entry_price),
        planned_exit_price=float(trade.exit_price),
        planned_exit_reason=trade.exit_reason,
        phase2_score=_as_float(meta.get("phase2_score"), 0.0),
        risk_per_lot=_as_float(meta.get("execution_risk_per_lot"), _as_float(meta.get("risk_per_lot"), 0.0)),
        margin_per_lot=margin_per_lot,
        notional_per_lot=notional_per_lot,
        multiplier=multiplier,
        pnl_ratio_price=float(trade.pnl_ratio),
        tp1_hit=bool(trade.tp1_hit),
        phase1_story_budget_margin_pct=_story_budget_margin_pct_from_meta(meta),
        phase1_story_budget_source=str(meta.get("phase1_story_budget_source") or ""),
        commission_per_lot=_as_float(meta.get("commission_per_lot"), commission_per_lot),
        commission_rate=_as_float(meta.get("commission_rate"), commission_rate),
        close_today_commission_per_lot=(
            _as_float(meta.get("close_today_commission_per_lot"), 0.0)
            if meta.get("close_today_commission_per_lot") is not None
            else close_today_commission_per_lot
        ),
        trade_id=trade.trade_id,
        entry_rr=_as_float(meta.get("entry_rr"), _as_float(meta.get("execution_rr"), _as_float(meta.get("rr"), 0.0))),
        entry_admission_rr=_as_float(
            meta.get("entry_admission_rr"),
            _as_float(meta.get("execution_admission_rr"), _as_float(meta.get("admission_rr"), 0.0)),
        ),
        medium_term_quality_score=_as_float(meta.get("medium_term_quality_score"), 0.0),
        medium_term_entry_location_score=_as_float(meta.get("medium_term_entry_location_score"), 0.0),
        entry_adverse_deviation_r=_entry_adverse_deviation_r_from_meta(meta, trade.direction),
        trend_phase=str(meta.get("trend_phase") or ""),
        tp1_price=_as_float(meta.get("tp1_exit_price"), _as_float(meta.get("initial_tp1_price"), 0.0)),
        tp1_time=str(meta.get("tp1_exit_time") or ""),
    )


def _planned_gross_pnl(candidate: AccountCandidate, lots: int) -> float:
    return float(candidate.pnl_ratio_price) * float(candidate.notional_per_lot) * int(lots)


def _mark_gross_pnl(candidate: AccountCandidate, lots: int, exit_price: float) -> float:
    if candidate.direction == "long":
        return (float(exit_price) - float(candidate.entry_price)) * float(candidate.multiplier) * int(lots)
    return (float(candidate.entry_price) - float(exit_price)) * float(candidate.multiplier) * int(lots)


def _same_trade_date(first: str, second: str) -> bool:
    try:
        return _parse_dt(first).date() == _parse_dt(second).date()
    except ValueError:
        return False


def _fee_for_side(
    candidate: AccountCandidate,
    *,
    price: float,
    is_close: bool,
    is_close_today: bool,
) -> float:
    if candidate.commission_rate > 0:
        return float(price) * float(candidate.multiplier) * float(candidate.commission_rate)
    if is_close and is_close_today and candidate.close_today_commission_per_lot is not None:
        return float(candidate.close_today_commission_per_lot)
    return float(candidate.commission_per_lot)


def _fees(
    candidate: AccountCandidate,
    lots: int,
    config: AccountBacktestConfig,
    *,
    exit_time: str,
    exit_price: float,
) -> float:
    is_close_today = _same_trade_date(candidate.entry_time, exit_time)
    entry_fee = _fee_for_side(
        candidate,
        price=float(candidate.entry_price),
        is_close=False,
        is_close_today=False,
    )
    exit_fee = _fee_for_side(
        candidate,
        price=float(exit_price),
        is_close=True,
        is_close_today=is_close_today,
    )
    return (entry_fee + exit_fee) * float(config.commission_multiplier) * int(lots)


def _fees_for_leg(
    candidate: AccountCandidate,
    lots: int,
    config: AccountBacktestConfig,
    *,
    entry_time: str,
    entry_price: float,
    exit_time: str,
    exit_price: float,
) -> float:
    if int(lots) <= 0:
        return 0.0
    is_close_today = _same_trade_date(entry_time, exit_time)
    entry_fee = _fee_for_side(
        candidate,
        price=float(entry_price),
        is_close=False,
        is_close_today=False,
    )
    exit_fee = _fee_for_side(
        candidate,
        price=float(exit_price),
        is_close=True,
        is_close_today=is_close_today,
    )
    return (entry_fee + exit_fee) * float(config.commission_multiplier) * int(lots)


def _row_gross_pnl(row: dict[str, Any], exit_price: float, *, use_planned_first_leg: bool = False) -> float:
    candidate: AccountCandidate = row["_candidate"]
    first_lots = int(row.get("first_entry_lots") or row.get("lots") or 0)
    second_lots = int(row.get("second_entry_lots") or 0)
    if use_planned_first_leg:
        gross = _planned_gross_pnl(candidate, first_lots)
    else:
        gross = _mark_gross_pnl(candidate, first_lots, float(exit_price))
    if second_lots > 0:
        second_price = _as_float(row.get("second_entry_price"), float(candidate.entry_price))
        if candidate.direction == "long":
            gross += (float(exit_price) - second_price) * float(candidate.multiplier) * second_lots
        else:
            gross += (second_price - float(exit_price)) * float(candidate.multiplier) * second_lots
    return float(gross)


def _row_fees(row: dict[str, Any], *, exit_time: str, exit_price: float, config: AccountBacktestConfig) -> float:
    candidate: AccountCandidate = row["_candidate"]
    first_lots = int(row.get("first_entry_lots") or row.get("lots") or 0)
    second_lots = int(row.get("second_entry_lots") or 0)
    fees = _fees_for_leg(
        candidate,
        first_lots,
        config,
        entry_time=str(row.get("entry_time") or candidate.entry_time),
        entry_price=float(row.get("entry_price") or candidate.entry_price),
        exit_time=exit_time,
        exit_price=float(exit_price),
    )
    if second_lots > 0:
        fees += _fees_for_leg(
            candidate,
            second_lots,
            config,
            entry_time=str(row.get("second_entry_time") or candidate.entry_time),
            entry_price=float(row.get("second_entry_price") or candidate.entry_price),
            exit_time=exit_time,
            exit_price=float(exit_price),
        )
    return float(fees)


def _max_drawdown(equity_curve: list[dict[str, Any]]) -> tuple[float, str, str]:
    peak = -math.inf
    peak_time = ""
    max_dd = 0.0
    max_start = ""
    max_end = ""
    for row in equity_curve:
        equity = float(row["equity"])
        if equity > peak:
            peak = equity
            peak_time = str(row["time"])
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                max_start = peak_time
                max_end = str(row["time"])
    return float(max_dd), max_start, max_end


def _active_margin(active: list[dict[str, Any]]) -> float:
    return sum(float(row["margin_used"]) for row in active)


def _remaining_second_entry_margin(row: dict[str, Any]) -> float:
    remaining = max(
        int(row.get("planned_second_entry_lots") or 0) - int(row.get("second_entry_lots") or 0),
        0,
    )
    return float(remaining) * _as_float(row.get("margin_per_lot"), 0.0)


def _portfolio_accounted_margin(active: list[dict[str, Any]], config: AccountBacktestConfig) -> float:
    actual = _active_margin(active)
    if not bool(config.reserve_planned_second_entry_margin):
        return float(actual)
    return float(actual + sum(_remaining_second_entry_margin(row) for row in active))


def _portfolio_accounted_margin_for_second_entry(
    active: list[dict[str, Any]],
    row: dict[str, Any],
    config: AccountBacktestConfig,
) -> float:
    accounted = _portfolio_accounted_margin(active, config)
    if not bool(config.reserve_planned_second_entry_margin):
        return float(accounted)
    return float(max(accounted - _remaining_second_entry_margin(row), 0.0))


def _hold_hours(row: dict[str, Any], now: datetime) -> float:
    return max((now - _parse_dt(str(row["entry_time"]))).total_seconds() / 3600.0, 0.0)


def _risk_tier(candidate: AccountCandidate, config: AccountBacktestConfig) -> str:
    if config.ordinary_risk_per_trade_pct is None and config.strong_risk_per_trade_pct is None:
        return "default"
    if (
        float(candidate.entry_rr) >= float(config.strong_first_target_rr)
        and float(candidate.entry_admission_rr) >= float(config.strong_second_target_rr)
    ):
        return "strong"
    return "ordinary"


def _effective_risk_per_trade_pct(candidate: AccountCandidate, config: AccountBacktestConfig) -> float | None:
    tier = _risk_tier(candidate, config)
    if tier == "strong" and config.strong_risk_per_trade_pct is not None:
        return float(config.strong_risk_per_trade_pct)
    if tier == "ordinary" and config.ordinary_risk_per_trade_pct is not None:
        return float(config.ordinary_risk_per_trade_pct)
    return config.risk_per_trade_pct


def _risk_lots(candidate: AccountCandidate, equity: float, config: AccountBacktestConfig) -> int:
    risk_pct = _effective_risk_per_trade_pct(candidate, config)
    if risk_pct is None:
        return 1_000_000_000
    risk_budget = max(float(equity) * float(risk_pct), 0.0)
    if risk_budget <= 0 or candidate.risk_per_lot <= 0:
        return 0
    return positive_int_floor(risk_budget / float(candidate.risk_per_lot))


def _conditional_execution_selection(candidate: AccountCandidate, config: AccountBacktestConfig) -> ExecutionSelection:
    quality = float(candidate.medium_term_quality_score)
    entry_rr = float(candidate.entry_rr)
    admission_rr = float(candidate.entry_admission_rr)
    adverse_r = float(candidate.entry_adverse_deviation_r)
    if (
        quality >= float(config.conditional_direct_quality_min)
        and entry_rr >= float(config.conditional_direct_entry_rr_min)
        and admission_rr >= float(config.conditional_direct_admission_rr_min)
        and adverse_r <= float(config.conditional_direct_entry_adverse_r_max)
    ):
        return ExecutionSelection("direct_full", 1.0, "none")
    if (
        quality >= float(config.conditional_scale_quality_min)
        and entry_rr >= float(config.conditional_scale_entry_rr_min)
        and admission_rr >= float(config.conditional_scale_admission_rr_min)
        and adverse_r <= float(config.conditional_scale_entry_adverse_r_max)
    ):
        return ExecutionSelection(
            "confirm_scale_in_70",
            _clamp_fraction(config.conditional_scale_initial_fraction, 0.70),
            "tp1",
        )
    if (
        quality >= float(config.conditional_trial_quality_min)
        and admission_rr >= float(config.conditional_trial_admission_rr_min)
        and adverse_r <= float(config.conditional_trial_entry_adverse_r_max)
    ):
        return ExecutionSelection(
            "trial_scale_in_30",
            _clamp_fraction(config.conditional_trial_initial_fraction, 0.30),
            "tp1",
        )
    return ExecutionSelection(
        "skip_weak_quality",
        0.0,
        "none",
        skip_reason="conditional_execution_quality_below_trade_threshold",
    )


def _execution_selection(candidate: AccountCandidate, config: AccountBacktestConfig) -> ExecutionSelection:
    policy = str(config.execution_policy or "fixed").strip().lower()
    if policy == "conditional":
        return _conditional_execution_selection(candidate, config)
    trigger = str(config.split_second_entry_trigger or "none").strip().lower() or "none"
    fraction = _clamp_fraction(config.split_initial_fraction, 1.0)
    profile = "fixed_full" if fraction >= 1.0 or trigger in {"", "none", "off", "false"} else "fixed_split"
    return ExecutionSelection(profile, fraction, trigger)


def _admission_skip_reason(candidate: AccountCandidate, config: AccountBacktestConfig) -> str:
    min_story_budget = max(float(config.min_phase1_story_budget_margin_pct), 0.0)
    if min_story_budget > 0 and _bounded_story_budget_margin_pct(candidate.phase1_story_budget_margin_pct) < min_story_budget:
        return "phase1_story_below_trade_threshold"
    min_phase2_abs_score = max(float(config.min_phase2_abs_score), 0.0)
    if min_phase2_abs_score > 0 and abs(float(candidate.phase2_score)) < min_phase2_abs_score:
        return "phase2_trigger_below_trade_threshold"
    return ""


def _build_open_row(
    candidate: AccountCandidate,
    *,
    target_lots: int,
    first_entry_lots: int,
    planned_second_entry_lots: int,
    split_initial_fraction: float,
    split_second_entry_trigger: str,
    execution_profile: str,
    equity: float,
    target_margin_pct: float,
    position_budget_source: str,
    margin_lots_cap: int,
    risk_lots_cap: int,
    risk_tier: str,
    effective_risk_per_trade_pct: float | None,
) -> dict[str, Any]:
    first_lots = int(first_entry_lots)
    margin_used = first_lots * float(candidate.margin_per_lot)
    notional = first_lots * float(candidate.notional_per_lot)
    initial_stop_risk = first_lots * float(candidate.risk_per_lot)
    planned_second_time = candidate.tp1_time if split_second_entry_trigger == "tp1" else ""
    planned_second_price = candidate.tp1_price if split_second_entry_trigger == "tp1" else 0.0
    return {
        "symbol": candidate.symbol,
        "name": candidate.name,
        "direction": candidate.direction,
        "entry_time": candidate.entry_time,
        "planned_exit_time": candidate.planned_exit_time,
        "entry_price": float(candidate.entry_price),
        "planned_exit_price": float(candidate.planned_exit_price),
        "planned_exit_reason": candidate.planned_exit_reason,
        "phase2_score": float(candidate.phase2_score),
        "phase2_abs_score": abs(float(candidate.phase2_score)),
        "position_budget_source": position_budget_source,
        "phase1_story_budget_margin_pct": float(candidate.phase1_story_budget_margin_pct),
        "phase1_story_budget_source": str(candidate.phase1_story_budget_source),
        "target_margin_pct": float(target_margin_pct),
        "entry_rr": float(candidate.entry_rr),
        "entry_admission_rr": float(candidate.entry_admission_rr),
        "medium_term_quality_score": float(candidate.medium_term_quality_score),
        "medium_term_entry_location_score": float(candidate.medium_term_entry_location_score),
        "entry_adverse_deviation_r": float(candidate.entry_adverse_deviation_r),
        "execution_profile": execution_profile,
        "risk_tier": risk_tier,
        "risk_budget_pct": float(effective_risk_per_trade_pct or 0.0),
        "target_lots": int(target_lots),
        "lots": first_lots,
        "first_entry_lots": first_lots,
        "planned_second_entry_lots": int(planned_second_entry_lots),
        "second_entry_lots": 0,
        "unfilled_second_entry_lots": int(planned_second_entry_lots),
        "split_initial_fraction": float(split_initial_fraction),
        "split_second_entry_trigger": split_second_entry_trigger,
        "planned_second_entry_time": planned_second_time,
        "planned_second_entry_price": float(planned_second_price or 0.0),
        "second_entry_time": "",
        "second_entry_price": 0.0,
        "first_entry_margin": float(margin_used),
        "planned_second_entry_margin": int(planned_second_entry_lots) * float(candidate.margin_per_lot),
        "second_entry_margin": 0.0,
        "second_entry_skip_reason": "",
        "risk_per_lot": float(candidate.risk_per_lot),
        "margin_per_lot": float(candidate.margin_per_lot),
        "notional_per_lot": float(candidate.notional_per_lot),
        "margin_lots_cap": int(margin_lots_cap),
        "risk_lots_cap": int(risk_lots_cap),
        "entry_equity": float(equity),
        "margin_used": float(margin_used),
        "notional": float(notional),
        "initial_stop_risk": float(initial_stop_risk),
        "initial_stop_risk_pct_equity": float(initial_stop_risk / equity) if equity > 0 else 0.0,
        "tp1_hit": bool(candidate.tp1_hit),
        "pnl_ratio_price": float(candidate.pnl_ratio_price),
        "_candidate": candidate,
    }


def _close_row(
    row: dict[str, Any],
    *,
    exit_time: str,
    exit_price: float,
    exit_reason: str,
    gross_pnl: float,
    fees: float,
) -> dict[str, Any]:
    out = dict(row)
    out.pop("_candidate", None)
    out["unfilled_second_entry_lots"] = max(
        int(out.get("planned_second_entry_lots") or 0) - int(out.get("second_entry_lots") or 0),
        0,
    )
    out.update(
        {
            "actual_exit_time": exit_time,
            "actual_exit_price": float(exit_price),
            "actual_exit_reason": exit_reason,
            "gross_pnl": float(gross_pnl),
            "fees": float(fees),
            "net_pnl": float(gross_pnl - fees),
        }
    )
    return out


def _score_lots(candidate: AccountCandidate, equity: float, config: AccountBacktestConfig) -> int:
    if candidate.margin_per_lot <= 0:
        return 0
    return positive_int_floor(equity * _target_margin_pct(candidate, config) / candidate.margin_per_lot)


def _portfolio_lots(candidate: AccountCandidate, equity: float, active_margin: float, config: AccountBacktestConfig) -> int:
    if candidate.margin_per_lot <= 0:
        return 0
    free_margin = max(equity * float(config.max_portfolio_margin_pct) - active_margin, 0.0)
    return positive_int_floor(free_margin / candidate.margin_per_lot)


def _candidate_lots(
    candidate: AccountCandidate,
    *,
    equity: float,
    active_margin: float,
    config: AccountBacktestConfig,
) -> tuple[int, int, int, int]:
    score_lots = _score_lots(candidate, equity, config)
    portfolio_lots = _portfolio_lots(candidate, equity, active_margin, config)
    risk_lots = _risk_lots(candidate, equity, config)
    lots = min(score_lots, portfolio_lots, risk_lots)
    return int(lots), int(score_lots), int(portfolio_lots), int(risk_lots)


def _split_entry_lots_for(total_lots: int, *, initial_fraction: float, second_entry_trigger: str) -> tuple[int, int, float, str]:
    total = int(total_lots)
    fraction = _clamp_fraction(initial_fraction, 1.0)
    trigger = str(second_entry_trigger or "none").strip().lower()
    if total <= 0:
        return 0, 0, fraction, trigger
    if fraction >= 1.0 or trigger in {"", "none", "off", "false"}:
        return total, 0, fraction, trigger or "none"
    first = max(1, positive_int_floor(total * fraction))
    first = min(first, total)
    return first, max(total - first, 0), fraction, trigger


def _split_entry_lots(total_lots: int, config: AccountBacktestConfig) -> tuple[int, int, float, str]:
    return _split_entry_lots_for(
        total_lots,
        initial_fraction=config.split_initial_fraction,
        second_entry_trigger=config.split_second_entry_trigger,
    )


def _tp1_add_ready(row: dict[str, Any], now: datetime) -> bool:
    if str(row.get("split_second_entry_trigger") or "") != "tp1":
        return False
    if str(row.get("second_entry_skip_reason") or ""):
        return False
    if int(row.get("planned_second_entry_lots") or 0) <= int(row.get("second_entry_lots") or 0):
        return False
    if not _as_bool(row.get("tp1_hit")):
        return False
    tp1_time = str(row.get("planned_second_entry_time") or "")
    if not tp1_time:
        return False
    return _parse_dt(tp1_time) <= now


def _add_due_second_legs(
    *,
    now: datetime,
    active: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    config: AccountBacktestConfig,
    equity: float,
) -> None:
    for row in sorted(active, key=lambda item: str(item.get("planned_second_entry_time") or "")):
        if not _tp1_add_ready(row, now):
            continue
        candidate: AccountCandidate = row["_candidate"]
        remaining = int(row.get("planned_second_entry_lots") or 0) - int(row.get("second_entry_lots") or 0)
        if remaining <= 0:
            continue
        available = _portfolio_lots(
            candidate,
            equity,
            _portfolio_accounted_margin_for_second_entry(active, row, config),
            config,
        )
        add_lots = min(remaining, available)
        if add_lots <= 0:
            row["second_entry_skip_reason"] = "portfolio_margin_full_at_second_entry"
            continue
        row["second_entry_lots"] = int(row.get("second_entry_lots") or 0) + int(add_lots)
        row["lots"] = int(row.get("lots") or 0) + int(add_lots)
        row["margin_used"] = float(row.get("margin_used") or 0.0) + int(add_lots) * float(candidate.margin_per_lot)
        row["notional"] = float(row.get("notional") or 0.0) + int(add_lots) * float(candidate.notional_per_lot)
        row["initial_stop_risk"] = float(row.get("initial_stop_risk") or 0.0) + int(add_lots) * float(candidate.risk_per_lot)
        entry_equity = float(row.get("entry_equity") or equity)
        row["initial_stop_risk_pct_equity"] = (
            float(row["initial_stop_risk"]) / entry_equity if entry_equity > 0 else 0.0
        )
        row["second_entry_time"] = str(row.get("planned_second_entry_time") or "")
        row["second_entry_price"] = float(row.get("planned_second_entry_price") or candidate.tp1_price or candidate.entry_price)
        row["second_entry_margin"] = float(row.get("second_entry_lots") or 0) * float(candidate.margin_per_lot)
        row["unfilled_second_entry_lots"] = max(
            int(row.get("planned_second_entry_lots") or 0) - int(row.get("second_entry_lots") or 0),
            0,
        )
        if int(row["unfilled_second_entry_lots"]) > 0:
            row["second_entry_skip_reason"] = "portfolio_margin_partial_at_second_entry"
        equity_curve.append(
            {
                "time": str(row["second_entry_time"]),
                "equity": float(equity),
                "active_margin": _portfolio_accounted_margin(active, config),
                "event": f"second_entry {row['symbol']}",
            }
        )


def _skip_reason(score_lots: int, portfolio_lots: int, risk_lots: int, config: AccountBacktestConfig) -> str:
    if risk_lots < 1:
        return "single_trade_stop_risk_below_one_lot"
    if portfolio_lots < 1:
        return "portfolio_margin_full_lower_score"
    if score_lots < 1:
        if _position_budget_source(config) == "phase1_story":
            return "phase1_story_budget_below_one_lot"
        return "phase2_score_budget_below_one_lot"
    return "portfolio_margin_still_full_after_replacement"


def _active_symbols(active: list[dict[str, Any]]) -> str:
    return ";".join(sorted(str(row["symbol"]) for row in active))


def _replacement_priority_for_candidate(candidate: AccountCandidate, config: AccountBacktestConfig) -> float:
    if _position_budget_source(config) == "phase1_story":
        return _target_margin_pct(candidate, config)
    return abs(float(candidate.phase2_score))


def _replacement_priority_for_row(row: dict[str, Any], config: AccountBacktestConfig) -> float:
    if _position_budget_source(config) == "phase1_story":
        return _as_float(row.get("target_margin_pct"), 0.0)
    return abs(_as_float(row.get("phase2_score"), 0.0))


def _replacement_exit_reason(config: AccountBacktestConfig) -> str:
    if _position_budget_source(config) == "phase1_story":
        return "replaced_by_higher_phase1_story_budget"
    return "replaced_by_higher_phase2_score"


def _decision_base_row(
    *,
    sequence: int,
    candidate: AccountCandidate,
    now: datetime,
    equity_before: float,
    active_margin_before: float,
    active_positions_before: int,
    active_symbols_before: str,
    config: AccountBacktestConfig,
) -> dict[str, Any]:
    risk_tier = _risk_tier(candidate, config)
    effective_risk_pct = _effective_risk_per_trade_pct(candidate, config)
    budget_source = _position_budget_source(config)
    target_margin_pct = _target_margin_pct(candidate, config)
    return {
        "sequence": int(sequence),
        "candidate_trade_id": candidate.trade_id,
        "symbol": candidate.symbol,
        "name": candidate.name,
        "direction": candidate.direction,
        "entry_time": candidate.entry_time,
        "planned_exit_time": candidate.planned_exit_time,
        "phase2_score": float(candidate.phase2_score),
        "phase2_abs_score": abs(float(candidate.phase2_score)),
        "position_budget_source": budget_source,
        "phase1_story_budget_margin_pct": float(candidate.phase1_story_budget_margin_pct),
        "phase1_story_budget_source": str(candidate.phase1_story_budget_source),
        "target_margin_pct": float(target_margin_pct),
        "equity_before": float(equity_before),
        "active_margin_before": float(active_margin_before),
        "active_margin_pct_before": float(active_margin_before / equity_before) if equity_before > 0 else 0.0,
        "active_positions_before": int(active_positions_before),
        "active_symbols_before": active_symbols_before,
        "candidate_margin_per_lot": float(candidate.margin_per_lot),
        "candidate_risk_per_lot": float(candidate.risk_per_lot),
        "candidate_notional_per_lot": float(candidate.notional_per_lot),
        "entry_rr": float(candidate.entry_rr),
        "entry_admission_rr": float(candidate.entry_admission_rr),
        "medium_term_quality_score": float(candidate.medium_term_quality_score),
        "medium_term_entry_location_score": float(candidate.medium_term_entry_location_score),
        "entry_adverse_deviation_r": float(candidate.entry_adverse_deviation_r),
        "execution_policy": str(config.execution_policy or "fixed").strip().lower() or "fixed",
        "risk_tier": risk_tier,
        "effective_risk_per_trade_pct": float(effective_risk_pct or 0.0),
        "_now": now,
    }


def _finalize_decision_row(
    row: dict[str, Any],
    *,
    decision: str,
    lots: int,
    target_lots: int | None = None,
    first_entry_lots: int | None = None,
    planned_second_entry_lots: int = 0,
    split_initial_fraction: float = 1.0,
    split_second_entry_trigger: str = "none",
    execution_profile: str = "",
    score_lots: int,
    portfolio_lots: int,
    risk_lots: int,
    active_margin_after: float,
    equity_after: float,
    skip_reason: str = "",
    replacement_attempted: bool = False,
    replacement_count: int = 0,
    replaced_symbols: list[str] | None = None,
    replacement_block_reason: str = "",
    replacement_block_symbol: str = "",
) -> dict[str, Any]:
    out = dict(row)
    out.pop("_now", None)
    out.update(
        {
            "decision": decision,
            "accepted_lots": int(lots) if decision in {"accepted", "replaced_then_accepted"} else 0,
            "target_lots": int(target_lots if target_lots is not None else lots),
            "first_entry_lots": int(first_entry_lots if first_entry_lots is not None else lots),
            "planned_second_entry_lots": int(planned_second_entry_lots),
            "split_initial_fraction": float(split_initial_fraction),
            "split_second_entry_trigger": split_second_entry_trigger,
            "execution_profile": execution_profile,
            "score_lots": int(score_lots),
            "portfolio_lots": int(portfolio_lots),
            "risk_lots": int(risk_lots),
            "skip_reason": skip_reason,
            "replacement_attempted": bool(replacement_attempted),
            "replacement_count": int(replacement_count),
            "replaced_symbols": ";".join(replaced_symbols or []),
            "replacement_block_reason": replacement_block_reason,
            "replacement_block_symbol": replacement_block_symbol,
            "equity_after": float(equity_after),
            "active_margin_after": float(active_margin_after),
            "active_margin_pct_after": float(active_margin_after / equity_after) if equity_after > 0 else 0.0,
        }
    )
    return out


def _close_due_positions(
    *,
    now: datetime,
    active: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    config: AccountBacktestConfig,
    equity: float,
) -> float:
    due = [row for row in active if _parse_dt(str(row["planned_exit_time"])) <= now]
    for row in sorted(due, key=lambda item: _parse_dt(str(item["planned_exit_time"]))):
        gross = _row_gross_pnl(row, float(row["planned_exit_price"]), use_planned_first_leg=True)
        fees = _row_fees(
            row,
            exit_time=str(row["planned_exit_time"]),
            exit_price=float(row["planned_exit_price"]),
            config=config,
        )
        closed_row = _close_row(
            row,
            exit_time=str(row["planned_exit_time"]),
            exit_price=float(row["planned_exit_price"]),
            exit_reason=str(row["planned_exit_reason"]),
            gross_pnl=gross,
            fees=fees,
        )
        closed.append(closed_row)
        active.remove(row)
        equity += float(closed_row["net_pnl"])
        equity_curve.append(
            {
                "time": str(row["planned_exit_time"]),
                "equity": float(equity),
                "active_margin": _portfolio_accounted_margin(active, config),
                "event": f"exit {row['symbol']}",
            }
        )
    return float(equity)


def _replacement_price(
    row: dict[str, Any],
    replace_time: datetime,
    price_lookup: Callable[[str, datetime], float | None] | None,
) -> float:
    if price_lookup is not None:
        price = price_lookup(str(row["symbol"]), replace_time)
        if price is not None and price > 0:
            return float(price)
    return float(row["entry_price"])


def _replace_lower_score_position(
    *,
    candidate: AccountCandidate,
    now: datetime,
    active: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    config: AccountBacktestConfig,
    equity: float,
    price_lookup: Callable[[str, datetime], float | None] | None,
) -> tuple[float, dict[str, Any] | None, str, str]:
    candidate_priority = _replacement_priority_for_candidate(candidate, config)
    lower_score = [
        row
        for row in active
        if str(row["symbol"]) != candidate.symbol
        and _replacement_priority_for_row(row, config) < candidate_priority
    ]
    if not lower_score:
        return float(equity), None, "no_lower_score_position", ""
    row = min(lower_score, key=lambda item: _replacement_priority_for_row(item, config))
    if _hold_hours(row, now) < max(float(config.min_replacement_hold_hours), 0.0):
        return float(equity), None, "min_replacement_hold_hours", str(row["symbol"])
    exit_price = _replacement_price(row, now, price_lookup)
    gross = _row_gross_pnl(row, exit_price)
    fees = _row_fees(
        row,
        exit_time=_fmt_dt(now),
        exit_price=exit_price,
        config=config,
    )
    closed_row = _close_row(
        row,
        exit_time=_fmt_dt(now),
        exit_price=exit_price,
        exit_reason=_replacement_exit_reason(config),
        gross_pnl=gross,
        fees=fees,
    )
    closed.append(closed_row)
    active.remove(row)
    equity += float(closed_row["net_pnl"])
    equity_curve.append(
        {
            "time": _fmt_dt(now),
            "equity": float(equity),
            "active_margin": _portfolio_accounted_margin(active, config),
            "event": f"{_replacement_exit_reason(config)} {row['symbol']}",
        }
    )
    return float(equity), closed_row, "", ""


def _by_symbol(trades: list[dict[str, Any]], initial_equity: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(trade["symbol"])].append(trade)
    rows: list[dict[str, Any]] = []
    for symbol, symbol_trades in sorted(grouped.items()):
        net_pnl = sum(float(row["net_pnl"]) for row in symbol_trades)
        wins = sum(1 for row in symbol_trades if float(row["net_pnl"]) > 0)
        rows.append(
            {
                "symbol": symbol,
                "name": str(symbol_trades[0]["name"]),
                "trades": len(symbol_trades),
                "wins": wins,
                "win_rate": wins / len(symbol_trades) if symbol_trades else 0.0,
                "net_pnl": net_pnl,
                "return_contribution_pct": net_pnl / initial_equity if initial_equity > 0 else 0.0,
                "fees": sum(float(row["fees"]) for row in symbol_trades),
            }
        )
    return rows


def run_account_backtest(
    candidates: list[AccountCandidate],
    config: AccountBacktestConfig,
    *,
    price_lookup: Callable[[str, datetime], float | None] | None = None,
) -> AccountBacktestResult:
    equity = float(config.initial_equity)
    active: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    skipped_reasons: Counter[str] = Counter()
    replacements = 0
    equity_curve = [
        {
            "time": "2025-01-01 00:00:00",
            "equity": float(equity),
            "active_margin": 0.0,
            "event": "start",
        }
    ]

    sorted_candidates = sorted(candidates, key=lambda item: (_parse_dt(item.entry_time), item.symbol, item.direction))
    for sequence, candidate in enumerate(sorted_candidates, start=1):
        now = _parse_dt(candidate.entry_time)
        equity = _close_due_positions(
            now=now,
            active=active,
            closed=closed,
            equity_curve=equity_curve,
            config=config,
            equity=equity,
        )
        _add_due_second_legs(
            now=now,
            active=active,
            equity_curve=equity_curve,
            config=config,
            equity=equity,
        )
        equity_before = float(equity)
        active_margin_before = _portfolio_accounted_margin(active, config)
        decision_base = _decision_base_row(
            sequence=sequence,
            candidate=candidate,
            now=now,
            equity_before=equity_before,
            active_margin_before=active_margin_before,
            active_positions_before=len(active),
            active_symbols_before=_active_symbols(active),
            config=config,
        )
        admission_skip_reason = _admission_skip_reason(candidate, config)
        if admission_skip_reason:
            skipped_reasons[admission_skip_reason] += 1
            decisions.append(
                _finalize_decision_row(
                    decision_base,
                    decision="skipped",
                    lots=0,
                    score_lots=0,
                    portfolio_lots=0,
                    risk_lots=0,
                    active_margin_after=_portfolio_accounted_margin(active, config),
                    equity_after=equity,
                    skip_reason=admission_skip_reason,
                )
            )
            continue
        if any(str(row["symbol"]) == candidate.symbol for row in active):
            skipped_reasons["same_symbol_overlap"] += 1
            decisions.append(
                _finalize_decision_row(
                    decision_base,
                    decision="skipped",
                    lots=0,
                    score_lots=0,
                    portfolio_lots=0,
                    risk_lots=0,
                    active_margin_after=_portfolio_accounted_margin(active, config),
                    equity_after=equity,
                    skip_reason="same_symbol_overlap",
                )
            )
            continue

        target_lots, score_lots, portfolio_lots, risk_lots = _candidate_lots(
            candidate,
            equity=equity,
            active_margin=_portfolio_accounted_margin(active, config),
            config=config,
        )
        replacement_attempted = False
        replaced_symbols: list[str] = []
        replacement_block_reason = ""
        replacement_block_symbol = ""
        while target_lots < 1 and portfolio_lots < min(score_lots, risk_lots) and config.allow_replacement:
            replacement_attempted = True
            equity, replaced_row, block_reason, block_symbol = _replace_lower_score_position(
                candidate=candidate,
                now=now,
                active=active,
                closed=closed,
                equity_curve=equity_curve,
                config=config,
                equity=equity,
                price_lookup=price_lookup,
            )
            if replaced_row is None:
                replacement_block_reason = block_reason
                replacement_block_symbol = block_symbol
                break
            replacements += 1
            replaced_symbols.append(str(replaced_row["symbol"]))
            target_lots, score_lots, portfolio_lots, risk_lots = _candidate_lots(
                candidate,
                equity=equity,
                active_margin=_portfolio_accounted_margin(active, config),
                config=config,
            )

        if target_lots < 1:
            reason = _skip_reason(score_lots, portfolio_lots, risk_lots, config)
            skipped_reasons[reason] += 1
            decisions.append(
                _finalize_decision_row(
                    decision_base,
                    decision="skipped",
                    lots=0,
                    score_lots=score_lots,
                    portfolio_lots=portfolio_lots,
                    risk_lots=risk_lots,
                    active_margin_after=_portfolio_accounted_margin(active, config),
                    equity_after=equity,
                    skip_reason=reason,
                    replacement_attempted=replacement_attempted,
                    replacement_count=len(replaced_symbols),
                    replaced_symbols=replaced_symbols,
                    replacement_block_reason=replacement_block_reason,
                    replacement_block_symbol=replacement_block_symbol,
                )
            )
            continue

        execution = _execution_selection(candidate, config)
        if execution.skip_reason:
            skipped_reasons[execution.skip_reason] += 1
            decisions.append(
                _finalize_decision_row(
                    decision_base,
                    decision="skipped",
                    lots=0,
                    target_lots=target_lots,
                    first_entry_lots=0,
                    planned_second_entry_lots=0,
                    split_initial_fraction=0.0,
                    split_second_entry_trigger="none",
                    execution_profile=execution.profile,
                    score_lots=score_lots,
                    portfolio_lots=portfolio_lots,
                    risk_lots=risk_lots,
                    active_margin_after=_portfolio_accounted_margin(active, config),
                    equity_after=equity,
                    skip_reason=execution.skip_reason,
                    replacement_attempted=replacement_attempted,
                    replacement_count=len(replaced_symbols),
                    replaced_symbols=replaced_symbols,
                    replacement_block_reason=replacement_block_reason,
                    replacement_block_symbol=replacement_block_symbol,
                )
            )
            continue

        first_lots, planned_second_lots, split_fraction, split_trigger = _split_entry_lots_for(
            target_lots,
            initial_fraction=execution.initial_fraction,
            second_entry_trigger=execution.second_entry_trigger,
        )
        open_row = _build_open_row(
            candidate,
            target_lots=target_lots,
            first_entry_lots=first_lots,
            planned_second_entry_lots=planned_second_lots,
            split_initial_fraction=split_fraction,
            split_second_entry_trigger=split_trigger,
            execution_profile=execution.profile,
            equity=equity,
            target_margin_pct=_target_margin_pct(candidate, config),
            position_budget_source=_position_budget_source(config),
            margin_lots_cap=min(score_lots, portfolio_lots),
            risk_lots_cap=risk_lots,
            risk_tier=_risk_tier(candidate, config),
            effective_risk_per_trade_pct=_effective_risk_per_trade_pct(candidate, config),
        )
        active.append(open_row)
        decisions.append(
            _finalize_decision_row(
                decision_base,
                decision="replaced_then_accepted" if replaced_symbols else "accepted",
                lots=first_lots,
                target_lots=target_lots,
                first_entry_lots=first_lots,
                planned_second_entry_lots=planned_second_lots,
                split_initial_fraction=split_fraction,
                split_second_entry_trigger=split_trigger,
                execution_profile=execution.profile,
                score_lots=score_lots,
                portfolio_lots=portfolio_lots,
                risk_lots=risk_lots,
                active_margin_after=_portfolio_accounted_margin(active, config),
                equity_after=equity,
                replacement_attempted=replacement_attempted,
                replacement_count=len(replaced_symbols),
                replaced_symbols=replaced_symbols,
                replacement_block_reason=replacement_block_reason,
                replacement_block_symbol=replacement_block_symbol,
            )
        )
        equity_curve.append(
            {
                "time": candidate.entry_time,
                "equity": float(equity),
                "active_margin": _portfolio_accounted_margin(active, config),
                "event": f"entry {candidate.symbol}",
            }
        )

    for row in sorted(list(active), key=lambda item: _parse_dt(str(item["planned_exit_time"]))):
        planned_exit_time = _parse_dt(str(row["planned_exit_time"]))
        _add_due_second_legs(
            now=planned_exit_time,
            active=active,
            equity_curve=equity_curve,
            config=config,
            equity=equity,
        )
        equity = _close_due_positions(
            now=planned_exit_time,
            active=active,
            closed=closed,
            equity_curve=equity_curve,
            config=config,
            equity=equity,
        )

    wins = sum(1 for row in closed if float(row["net_pnl"]) > 0)
    losses = sum(1 for row in closed if float(row["net_pnl"]) < 0)
    max_dd, dd_start, dd_end = _max_drawdown(equity_curve)
    max_margin_pct = max(
        (float(row["active_margin"]) / float(row["equity"]) for row in equity_curve if float(row["equity"]) > 0),
        default=0.0,
    )
    risk_pcts = [float(row["initial_stop_risk_pct_equity"]) for row in closed]
    split_target_lots = sum(int(row.get("target_lots") or row.get("lots") or 0) for row in closed)
    split_first_lots = sum(int(row.get("first_entry_lots") or row.get("lots") or 0) for row in closed)
    split_planned_second_lots = sum(int(row.get("planned_second_entry_lots") or 0) for row in closed)
    split_filled_second_lots = sum(int(row.get("second_entry_lots") or 0) for row in closed)
    split_unfilled_second_lots = sum(int(row.get("unfilled_second_entry_lots") or 0) for row in closed)
    execution_profile_counts = Counter(str(row.get("execution_profile") or "") for row in closed)
    execution_profile_counts.pop("", None)
    budget_source = _position_budget_source(config)
    if budget_source == "phase1_story":
        sizing_note = (
            "Phase1 trend story budget maps to target margin; portfolio max margin cap; "
            "single trade stop risk cap; higher story budget may replace lower story budget"
        )
        target_margin_formula = "clamp(phase1_story_budget_margin_pct, 0%, 30%) with 3% fallback"
    else:
        sizing_note = (
            "Phase2 final absolute score maps to target margin; portfolio max margin cap; "
            "single trade stop risk cap; higher score may replace lower score"
        )
        target_margin_formula = "max(3%, min(30%, (abs(phase2_score)-25)/35*30%))"
    summary = {
        "scope": config.scope,
        "sizing": sizing_note,
        "position_budget_source": budget_source,
        "target_margin_formula": target_margin_formula,
        "initial_equity": float(config.initial_equity),
        "final_equity": float(equity),
        "net_profit": float(equity - config.initial_equity),
        "return_pct": float((equity - config.initial_equity) / config.initial_equity) if config.initial_equity > 0 else 0.0,
        "max_drawdown_realized_pct": float(max_dd),
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "accepted_trades": len(closed),
        "closed_trades": len(closed),
        "candidate_trades": len(candidates),
        "skipped_trades": sum(skipped_reasons.values()),
        "replacements": int(replacements),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(closed) if closed else 0.0,
        "total_gross_pnl": float(sum(float(row["gross_pnl"]) for row in closed)),
        "total_fees": float(sum(float(row["fees"]) for row in closed)),
        "commission_multiplier": float(config.commission_multiplier),
        "min_replacement_hold_hours": float(config.min_replacement_hold_hours),
        "max_portfolio_margin_pct": float(config.max_portfolio_margin_pct),
        "reserve_planned_second_entry_margin": bool(config.reserve_planned_second_entry_margin),
        "min_phase1_story_budget_margin_pct": float(config.min_phase1_story_budget_margin_pct),
        "min_phase2_abs_score": float(config.min_phase2_abs_score),
        "max_trade_stop_risk_pct": config.risk_per_trade_pct,
        "ordinary_trade_stop_risk_pct": config.ordinary_risk_per_trade_pct,
        "strong_trade_stop_risk_pct": config.strong_risk_per_trade_pct,
        "strong_first_target_rr": float(config.strong_first_target_rr),
        "strong_second_target_rr": float(config.strong_second_target_rr),
        "execution_policy": str(config.execution_policy or "fixed").strip().lower() or "fixed",
        "execution_profile_counts": dict(execution_profile_counts),
        "split_initial_fraction": float(_clamp_fraction(config.split_initial_fraction, 1.0)),
        "split_second_entry_trigger": str(config.split_second_entry_trigger or "none").strip().lower() or "none",
        "split_target_lots": int(split_target_lots),
        "split_first_entry_lots": int(split_first_lots),
        "split_planned_second_entry_lots": int(split_planned_second_lots),
        "split_second_entry_filled_lots": int(split_filled_second_lots),
        "split_second_entry_unfilled_lots": int(split_unfilled_second_lots),
        "max_margin_pct_observed": float(max_margin_pct),
        "max_initial_stop_risk_pct_equity": max(risk_pcts) if risk_pcts else 0.0,
        "avg_initial_stop_risk_pct_equity": sum(risk_pcts) / len(risk_pcts) if risk_pcts else 0.0,
        "skipped_reasons": dict(skipped_reasons),
    }
    return AccountBacktestResult(
        summary=summary,
        trades=sorted(closed, key=lambda item: (str(item["entry_time"]), str(item["symbol"]))),
        equity_curve=equity_curve,
        by_symbol=_by_symbol(closed, float(config.initial_equity)),
        decisions=decisions,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


TRADE_FIELDS = [
    "symbol",
    "name",
    "direction",
    "entry_time",
    "planned_exit_time",
    "entry_price",
    "planned_exit_price",
    "planned_exit_reason",
    "phase2_score",
    "phase2_abs_score",
    "position_budget_source",
    "phase1_story_budget_margin_pct",
    "phase1_story_budget_source",
    "target_margin_pct",
    "entry_rr",
    "entry_admission_rr",
    "medium_term_quality_score",
    "medium_term_entry_location_score",
    "entry_adverse_deviation_r",
    "execution_profile",
    "risk_tier",
    "risk_budget_pct",
    "target_lots",
    "lots",
    "first_entry_lots",
    "planned_second_entry_lots",
    "second_entry_lots",
    "unfilled_second_entry_lots",
    "split_initial_fraction",
    "split_second_entry_trigger",
    "planned_second_entry_time",
    "planned_second_entry_price",
    "second_entry_time",
    "second_entry_price",
    "first_entry_margin",
    "planned_second_entry_margin",
    "second_entry_margin",
    "second_entry_skip_reason",
    "risk_per_lot",
    "margin_per_lot",
    "notional_per_lot",
    "margin_lots_cap",
    "risk_lots_cap",
    "entry_equity",
    "margin_used",
    "notional",
    "initial_stop_risk",
    "initial_stop_risk_pct_equity",
    "tp1_hit",
    "pnl_ratio_price",
    "actual_exit_time",
    "actual_exit_price",
    "actual_exit_reason",
    "gross_pnl",
    "fees",
    "net_pnl",
]


DECISION_FIELDS = [
    "sequence",
    "candidate_trade_id",
    "symbol",
    "name",
    "direction",
    "entry_time",
    "planned_exit_time",
    "phase2_score",
    "phase2_abs_score",
    "position_budget_source",
    "phase1_story_budget_margin_pct",
    "phase1_story_budget_source",
    "target_margin_pct",
    "entry_rr",
    "entry_admission_rr",
    "medium_term_quality_score",
    "medium_term_entry_location_score",
    "entry_adverse_deviation_r",
    "execution_policy",
    "risk_tier",
    "effective_risk_per_trade_pct",
    "equity_before",
    "active_margin_before",
    "active_margin_pct_before",
    "active_positions_before",
    "active_symbols_before",
    "candidate_margin_per_lot",
    "candidate_risk_per_lot",
    "candidate_notional_per_lot",
    "score_lots",
    "portfolio_lots",
    "risk_lots",
    "target_lots",
    "first_entry_lots",
    "planned_second_entry_lots",
    "split_initial_fraction",
    "split_second_entry_trigger",
    "execution_profile",
    "decision",
    "accepted_lots",
    "skip_reason",
    "replacement_attempted",
    "replacement_count",
    "replaced_symbols",
    "replacement_block_reason",
    "replacement_block_symbol",
    "equity_after",
    "active_margin_after",
    "active_margin_pct_after",
]


def write_account_outputs(result: AccountBacktestResult, output_prefix: str | Path) -> list[Path]:
    prefix = Path(output_prefix)
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    trades_path = prefix.with_name(f"{prefix.name}_trades.csv")
    equity_path = prefix.with_name(f"{prefix.name}_equity.csv")
    by_symbol_path = prefix.with_name(f"{prefix.name}_by_symbol.csv")
    decisions_path = prefix.with_name(f"{prefix.name}_decisions.csv")
    _write_json(summary_path, result.summary)
    _write_csv(trades_path, result.trades, TRADE_FIELDS)
    _write_csv(equity_path, result.equity_curve, ["time", "equity", "active_margin", "event"])
    _write_csv(
        by_symbol_path,
        result.by_symbol,
        ["symbol", "name", "trades", "wins", "win_rate", "net_pnl", "return_contribution_pct", "fees"],
    )
    _write_csv(decisions_path, result.decisions, DECISION_FIELDS)
    return [summary_path, trades_path, equity_path, by_symbol_path, decisions_path]
