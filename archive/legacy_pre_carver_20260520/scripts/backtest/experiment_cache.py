from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backtest.models import BacktestCase, BacktestResult, TradePlan, TradeRecord
from backtest.phase23 import (
    _empty_phase2_rejection_counts,
    _evaluate_phase2_plan,
    _phase2_debug_snapshot_from_plan,
    _phase2_min_history_bars,
)


PHASE2_EXPERIMENT_CACHE_VERSION = "phase2-experiment-v1"
PHASE23_EXPERIMENT_CACHE_VERSION = "phase23-experiment-v1"

PLAN_COMPARE_FIELDS = (
    "trade_id",
    "symbol",
    "direction",
    "plan_date",
    "entry_ref",
    "stop",
    "tp1",
    "tp2",
    "phase2_score",
    "signal_type",
)

PLAN_META_COMPARE_FIELDS = (
    "entry_family",
    "strategy_family",
    "entry_signal_type",
    "entry_signal_detail",
)


@dataclass(frozen=True, slots=True)
class Phase2CacheEntry:
    trade_date: str
    visible_daily_rows: int
    history_insufficient: bool
    plan: TradePlan | None
    rejection_counts: dict[str, int]
    debug_snapshot: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Phase2ExperimentCache:
    version: str
    case_id: str
    symbol: str
    direction: str
    start_date: str
    end_date: str
    pre_market_fingerprint: str
    entries: list[Phase2CacheEntry]


@dataclass(frozen=True, slots=True)
class Phase23ExperimentCache:
    version: str
    case_id: str
    symbol: str
    direction: str
    start_date: str
    end_date: str
    pre_market_fingerprint: str
    signal_fingerprint: str
    result: BacktestResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _fingerprint_config(config: dict[str, Any]) -> str:
    safe = _json_safe(config)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint_digest(*configs: dict[str, Any]) -> str:
    raw = "|".join(_fingerprint_config(config) for config in configs)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _trade_plan_to_dict(plan: TradePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "trade_id": plan.trade_id,
        "symbol": plan.symbol,
        "direction": plan.direction,
        "plan_date": plan.plan_date,
        "entry_ref": float(plan.entry_ref),
        "stop": float(plan.stop),
        "tp1": float(plan.tp1),
        "tp2": float(plan.tp2),
        "phase2_score": float(plan.phase2_score),
        "signal_type": plan.signal_type,
        "meta": _json_safe(plan.meta),
    }


def _trade_plan_from_dict(payload: dict[str, Any] | None) -> TradePlan | None:
    if payload is None:
        return None
    return TradePlan(
        trade_id=str(payload["trade_id"]),
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        plan_date=str(payload["plan_date"]),
        entry_ref=float(payload["entry_ref"]),
        stop=float(payload["stop"]),
        tp1=float(payload["tp1"]),
        tp2=float(payload["tp2"]),
        phase2_score=float(payload["phase2_score"]),
        signal_type=str(payload.get("signal_type") or ""),
        meta=dict(payload.get("meta") or {}),
    )


def _entry_to_dict(entry: Phase2CacheEntry) -> dict[str, Any]:
    return {
        "trade_date": entry.trade_date,
        "visible_daily_rows": int(entry.visible_daily_rows),
        "history_insufficient": bool(entry.history_insufficient),
        "plan": _trade_plan_to_dict(entry.plan),
        "rejection_counts": {str(key): int(value) for key, value in entry.rejection_counts.items()},
        "debug_snapshot": _json_safe(entry.debug_snapshot),
    }


def _entry_from_dict(payload: dict[str, Any]) -> Phase2CacheEntry:
    return Phase2CacheEntry(
        trade_date=str(payload["trade_date"]),
        visible_daily_rows=int(payload.get("visible_daily_rows") or 0),
        history_insufficient=bool(payload.get("history_insufficient")),
        plan=_trade_plan_from_dict(payload.get("plan")),
        rejection_counts={str(key): int(value) for key, value in (payload.get("rejection_counts") or {}).items()},
        debug_snapshot=dict(payload.get("debug_snapshot") or {}),
    )


def _cache_to_dict(cache: Phase2ExperimentCache) -> dict[str, Any]:
    return {
        "version": cache.version,
        "case_id": cache.case_id,
        "symbol": cache.symbol,
        "direction": cache.direction,
        "start_date": cache.start_date,
        "end_date": cache.end_date,
        "pre_market_fingerprint": cache.pre_market_fingerprint,
        "entries": [_entry_to_dict(entry) for entry in cache.entries],
    }


def _cache_from_dict(payload: dict[str, Any]) -> Phase2ExperimentCache:
    return Phase2ExperimentCache(
        version=str(payload["version"]),
        case_id=str(payload["case_id"]),
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]),
        pre_market_fingerprint=str(payload.get("pre_market_fingerprint") or ""),
        entries=[_entry_from_dict(entry) for entry in payload.get("entries") or []],
    )


def _trade_record_to_dict(trade: TradeRecord) -> dict[str, Any]:
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "direction": trade.direction,
        "entry_time": trade.entry_time,
        "entry_price": float(trade.entry_price),
        "exit_time": trade.exit_time,
        "exit_price": float(trade.exit_price),
        "exit_reason": trade.exit_reason,
        "bars_held": int(trade.bars_held),
        "days_held": int(trade.days_held),
        "tp1_hit": bool(trade.tp1_hit),
        "pnl_ratio": float(trade.pnl_ratio),
        "meta": _json_safe(trade.meta),
    }


def _trade_record_from_dict(payload: dict[str, Any]) -> TradeRecord:
    return TradeRecord(
        trade_id=str(payload["trade_id"]),
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        entry_time=str(payload["entry_time"]),
        entry_price=float(payload["entry_price"]),
        exit_time=str(payload["exit_time"]),
        exit_price=float(payload["exit_price"]),
        exit_reason=str(payload["exit_reason"]),
        bars_held=int(payload["bars_held"]),
        days_held=int(payload["days_held"]),
        tp1_hit=bool(payload["tp1_hit"]),
        pnl_ratio=float(payload["pnl_ratio"]),
        meta=dict(payload.get("meta") or {}),
    )


def _backtest_result_to_dict(result: BacktestResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "trades": [_trade_record_to_dict(trade) for trade in result.trades],
        "summary": _json_safe(result.summary),
        "diagnostics": _json_safe(result.diagnostics),
        "debug": _json_safe(result.debug),
    }


def _backtest_result_from_dict(payload: dict[str, Any]) -> BacktestResult:
    return BacktestResult(
        case_id=str(payload["case_id"]),
        trades=[_trade_record_from_dict(trade) for trade in payload.get("trades") or []],
        summary=dict(payload.get("summary") or {}),
        diagnostics=dict(payload.get("diagnostics") or {}),
        debug=dict(payload.get("debug") or {}),
    )


def _phase23_cache_to_dict(cache: Phase23ExperimentCache) -> dict[str, Any]:
    return {
        "version": cache.version,
        "case_id": cache.case_id,
        "symbol": cache.symbol,
        "direction": cache.direction,
        "start_date": cache.start_date,
        "end_date": cache.end_date,
        "pre_market_fingerprint": cache.pre_market_fingerprint,
        "signal_fingerprint": cache.signal_fingerprint,
        "result": _backtest_result_to_dict(cache.result),
    }


def _phase23_cache_from_dict(payload: dict[str, Any]) -> Phase23ExperimentCache:
    return Phase23ExperimentCache(
        version=str(payload["version"]),
        case_id=str(payload["case_id"]),
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]),
        pre_market_fingerprint=str(payload.get("pre_market_fingerprint") or ""),
        signal_fingerprint=str(payload.get("signal_fingerprint") or ""),
        result=_backtest_result_from_dict(payload["result"]),
    )


def _visible_daily_by_trade_date(daily_df: pd.DataFrame) -> pd.DataFrame:
    visible_daily = daily_df.copy()
    if visible_daily.empty:
        return pd.DataFrame(columns=["date", "trade_date"])
    visible_daily["date"] = pd.to_datetime(visible_daily["date"])
    visible_daily = visible_daily.sort_values("date", kind="stable")
    visible_daily["trade_date"] = visible_daily["date"].dt.date
    return visible_daily


def _trade_dates_from_minute_frame(case: BacktestCase, minute_df: pd.DataFrame) -> list[date]:
    if minute_df.empty:
        return []
    minute_data = minute_df.copy()
    minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])
    minute_data["trade_date"] = minute_data["datetime"].dt.date
    minute_data = minute_data.loc[minute_data["trade_date"].between(case.start_dt, case.end_dt)]
    return sorted(minute_data["trade_date"].dropna().unique().tolist())


def build_phase2_experiment_cache(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    pre_market_cfg: dict[str, Any],
    plan_factory: Callable[..., TradePlan | None],
) -> Phase2ExperimentCache:
    visible_daily = _visible_daily_by_trade_date(daily_df)
    min_history_bars = _phase2_min_history_bars(pre_market_cfg)
    entries: list[Phase2CacheEntry] = []
    for trade_date in _trade_dates_from_minute_frame(case, minute_df):
        day_visible_daily = visible_daily.loc[visible_daily["trade_date"] < trade_date].copy()
        history_insufficient = len(day_visible_daily) < min_history_bars
        if history_insufficient:
            plan = None
            rejection_counts = _empty_phase2_rejection_counts()
            debug_snapshot = _phase2_debug_snapshot_from_plan(None)
        else:
            plan, rejection_counts, debug_snapshot = _evaluate_phase2_plan(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
                plan_factory=plan_factory,
                capture_debug=False,
            )
        entries.append(
            Phase2CacheEntry(
                trade_date=trade_date.isoformat(),
                visible_daily_rows=int(len(day_visible_daily)),
                history_insufficient=bool(history_insufficient),
                plan=plan,
                rejection_counts=dict(rejection_counts),
                debug_snapshot=dict(debug_snapshot),
            )
        )
    return Phase2ExperimentCache(
        version=PHASE2_EXPERIMENT_CACHE_VERSION,
        case_id=case.case_id,
        symbol=case.symbol,
        direction=case.direction,
        start_date=case.start_dt.isoformat(),
        end_date=case.end_dt.isoformat(),
        pre_market_fingerprint=_fingerprint_config(pre_market_cfg),
        entries=entries,
    )


def write_phase2_experiment_cache(cache: Phase2ExperimentCache, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_cache_to_dict(cache), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_phase2_experiment_cache(path: Path) -> Phase2ExperimentCache:
    return _cache_from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_phase23_experiment_cache(
    *,
    case: BacktestCase,
    result: BacktestResult,
    pre_market_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
) -> Phase23ExperimentCache:
    return Phase23ExperimentCache(
        version=PHASE23_EXPERIMENT_CACHE_VERSION,
        case_id=case.case_id,
        symbol=case.symbol,
        direction=case.direction,
        start_date=case.start_dt.isoformat(),
        end_date=case.end_dt.isoformat(),
        pre_market_fingerprint=_fingerprint_config(pre_market_cfg),
        signal_fingerprint=_fingerprint_config(signal_cfg),
        result=result,
    )


def phase23_experiment_cache_path(
    cache_dir: Path,
    *,
    case: BacktestCase,
    pre_market_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
) -> Path:
    digest = _fingerprint_digest(pre_market_cfg, signal_cfg)
    return Path(cache_dir) / f"{case.case_id}_{digest}_phase23_cache.json"


def write_phase23_experiment_cache(cache: Phase23ExperimentCache, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_phase23_cache_to_dict(cache), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_phase23_experiment_cache(path: Path) -> Phase23ExperimentCache:
    return _phase23_cache_from_dict(json.loads(path.read_text(encoding="utf-8")))


def phase23_cache_matches_config(
    cache: Phase23ExperimentCache,
    *,
    case: BacktestCase,
    pre_market_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
) -> bool:
    return (
        cache.version == PHASE23_EXPERIMENT_CACHE_VERSION
        and cache.case_id == case.case_id
        and cache.symbol == case.symbol
        and cache.direction == case.direction
        and cache.start_date == case.start_dt.isoformat()
        and cache.end_date == case.end_dt.isoformat()
        and cache.pre_market_fingerprint == _fingerprint_config(pre_market_cfg)
        and cache.signal_fingerprint == _fingerprint_config(signal_cfg)
    )


def _plan_compare_payload(plan: TradePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    payload = {field: getattr(plan, field) for field in PLAN_COMPARE_FIELDS}
    payload.update({f"meta.{field}": plan.meta.get(field, "") for field in PLAN_META_COMPARE_FIELDS})
    return _json_safe(payload)


def compare_phase2_cache_to_fresh_build(
    cache: Phase2ExperimentCache,
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    pre_market_cfg: dict[str, Any],
    plan_factory: Callable[..., TradePlan | None],
) -> list[dict[str, Any]]:
    fresh = build_phase2_experiment_cache(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        pre_market_cfg=pre_market_cfg,
        plan_factory=plan_factory,
    )
    mismatches: list[dict[str, Any]] = []
    cached_by_date = {entry.trade_date: entry for entry in cache.entries}
    fresh_by_date = {entry.trade_date: entry for entry in fresh.entries}
    for trade_date in sorted(set(cached_by_date) | set(fresh_by_date)):
        cached = cached_by_date.get(trade_date)
        rebuilt = fresh_by_date.get(trade_date)
        if cached is None or rebuilt is None:
            mismatches.append({"trade_date": trade_date, "field": "entry_presence"})
            continue
        if cached.visible_daily_rows != rebuilt.visible_daily_rows:
            mismatches.append(
                {
                    "trade_date": trade_date,
                    "field": "visible_daily_rows",
                    "cached": cached.visible_daily_rows,
                    "fresh": rebuilt.visible_daily_rows,
                }
            )
        if cached.history_insufficient != rebuilt.history_insufficient:
            mismatches.append(
                {
                    "trade_date": trade_date,
                    "field": "history_insufficient",
                    "cached": cached.history_insufficient,
                    "fresh": rebuilt.history_insufficient,
                }
            )
        cached_plan = _plan_compare_payload(cached.plan)
        fresh_plan = _plan_compare_payload(rebuilt.plan)
        if cached_plan != fresh_plan:
            mismatches.append(
                {
                    "trade_date": trade_date,
                    "field": "plan",
                    "cached": cached_plan,
                    "fresh": fresh_plan,
                }
            )
    return mismatches


__all__ = [
    "PHASE2_EXPERIMENT_CACHE_VERSION",
    "PHASE23_EXPERIMENT_CACHE_VERSION",
    "Phase2CacheEntry",
    "Phase2ExperimentCache",
    "Phase23ExperimentCache",
    "build_phase2_experiment_cache",
    "build_phase23_experiment_cache",
    "compare_phase2_cache_to_fresh_build",
    "phase23_cache_matches_config",
    "phase23_experiment_cache_path",
    "read_phase2_experiment_cache",
    "read_phase23_experiment_cache",
    "write_phase2_experiment_cache",
    "write_phase23_experiment_cache",
]
