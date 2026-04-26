from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from phase2 import pre_market
from backtest.models import BacktestCase, BacktestResult, TradePlan, TradeRecord
from backtest.trade_management import (
    TradeManagementProfile,
    realized_pnl_ratio,
    resolve_trade_management_profile,
    stop_after_first_target,
    trade_management_meta,
    trend_daily_review_keeps_position,
)
from data_cache import CACHE_DIR, _exchange_for_symbol
from phase3.intraday import generate_signals, is_entry_confirmation_signal
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND
from market.tq import (
    klines_to_daily_frame,
    resolve_tq_continuous_symbol,
    symbol_to_tq_main,
)


PHASE2_REJECTION_KEYS = (
    "phase2_reject_no_signal_days",
    "phase2_reject_score_gate_days",
    "phase2_reject_rr_gate_days",
    "phase2_reject_duplicate_signal_days",
    "phase2_reject_missing_fundamental_days",
)

PHASE2_ENTRY_FAMILY_KEYS = (
    "phase2_actionable_reversal_days",
    "phase2_actionable_trend_days",
    "trades_opened_reversal",
    "trades_opened_trend",
)

PHASE2_STRATEGY_FAMILY_KEYS = (
    "phase2_actionable_strategy_reversal_days",
    "phase2_actionable_strategy_trend_days",
    "trades_opened_strategy_reversal",
    "trades_opened_strategy_trend",
)

BACKTEST_ANCHOR_OFFSETS = (1, 2, 3, 5, 7, 10, 14)
BACKTEST_MINUTE_CHUNK_DAYS = 7
BACKTEST_MINUTE_DATA_LENGTH = 10_000
BACKTEST_CACHE_VERSION = "v3"
BACKTEST_CACHE_READ_VERSIONS = ("v3", "v2", "v1")
BACKTEST_CACHE_DIR = CACHE_DIR / "backtest"
BACKTEST_FUNDAMENTAL_MODE_STRICT = "strict"
BACKTEST_FUNDAMENTAL_MODE_PROXY = "proxy"


def _create_backtest_api(*, start_dt, end_dt, account: str, password: str):
    from tqsdk import TqApi, TqAuth, TqBacktest, TqSim

    return TqApi(
        TqSim(),
        backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt),
        auth=TqAuth(account, password),
    )


def resolve_case_tq_symbol(case: BacktestCase, api=None) -> str:
    symbol = case.symbol.strip()
    exchange = _exchange_for_symbol(symbol)
    if not exchange:
        raise ValueError(f"未知品种代码（未在内置列表）: {symbol}")

    try:
        resolved = resolve_tq_continuous_symbol(
            symbol,
            exchange,
            api=api,
            wait_timeout=2.0,
        )
        if resolved:
            return resolved
    except Exception:
        pass
    return symbol_to_tq_main(symbol, exchange)


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "oi"])


def _empty_minute_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def _daily_warmup_bars(config: dict) -> int:
    pre_market_cfg = config.get("pre_market") or {}
    return max(int(pre_market_cfg.get("min_history_bars", 60)), 1)


def _backtest_start_dt(case: BacktestCase, config: dict):
    warmup_bars = _daily_warmup_bars(config)
    return case.start_dt - timedelta(days=warmup_bars * 2)


def _sanitize_cache_fragment(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return out.strip("_") or "unknown"


def _backtest_cache_key(*, case: BacktestCase, warmup_bars: int, version: str = BACKTEST_CACHE_VERSION) -> str:
    symbol = _sanitize_cache_fragment(case.symbol)
    start = case.start_dt.strftime("%Y%m%d")
    end = case.end_dt.strftime("%Y%m%d")
    return f"{symbol}_{start}_{end}_w{warmup_bars}_{version}"


def _backtest_cache_paths_for_version(
    *,
    case: BacktestCase,
    warmup_bars: int,
    version: str,
) -> tuple[Path, Path]:
    key = _backtest_cache_key(case=case, warmup_bars=warmup_bars, version=version)
    return (
        BACKTEST_CACHE_DIR / f"{key}_daily.parquet",
        BACKTEST_CACHE_DIR / f"{key}_minute.parquet",
    )


def _backtest_cache_paths(*, case: BacktestCase, warmup_bars: int) -> tuple[Path, Path]:
    return _backtest_cache_paths_for_version(
        case=case,
        warmup_bars=warmup_bars,
        version=BACKTEST_CACHE_VERSION,
    )


def _klines_to_minute_frame(klines: pd.DataFrame | None) -> pd.DataFrame:
    if klines is None or len(klines) < 1:
        return _empty_minute_frame()

    df = klines.copy()
    required = ("datetime", "open", "high", "low", "close", "volume")
    for column in required:
        if column not in df.columns:
            return _empty_minute_frame()

    df["datetime"] = pd.to_datetime(df["datetime"])
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    out = df[list(required)].dropna(subset=["close"])
    if out.empty:
        return _empty_minute_frame()
    return out.sort_values("datetime", kind="stable").reset_index(drop=True)


def _normalize_daily_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or len(frame) < 1:
        return _empty_daily_frame()

    df = frame.copy()
    required = ("date", "open", "high", "low", "close", "volume", "oi")
    for column in required:
        if column not in df.columns:
            return _empty_daily_frame()

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "oi"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    out = df[list(required)].dropna(subset=["date", "close"])
    if out.empty:
        return _empty_daily_frame()
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def _open_backtest_anchor_api(*, target_dt: date, account: str, password: str):
    last_exc: Exception | None = None
    for offset_days in BACKTEST_ANCHOR_OFFSETS:
        anchor_dt = target_dt + timedelta(days=offset_days)
        try:
            api = _create_backtest_api(
                start_dt=anchor_dt,
                end_dt=anchor_dt,
                account=account,
                password=password,
            )
            return api, anchor_dt
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("无法创建 TqBacktest 锚点会话")


def _filter_minute_chunk_range(frame: pd.DataFrame, *, start_dt: date, end_dt: date) -> pd.DataFrame:
    if frame.empty:
        return _empty_minute_frame()
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.loc[out["datetime"].dt.date.between(start_dt, end_dt)].copy()
    return out.reset_index(drop=True)


def _load_daily_frame_with_tqbacktest(
    *,
    case: BacktestCase,
    tq_symbol: str,
    warmup_bars: int,
    in_case_daily_bars: int,
    account: str,
    password: str,
) -> pd.DataFrame:
    api = None
    try:
        api, _anchor_dt = _open_backtest_anchor_api(
            target_dt=case.end_dt,
            account=account,
            password=password,
        )
        daily_length = max(int(in_case_daily_bars) + warmup_bars + 10, 30)
        daily_klines = api.get_kline_serial(tq_symbol, 86400, data_length=daily_length)
        return _filter_daily_range(klines_to_daily_frame(daily_klines), case)
    finally:
        try:
            if api is not None:
                api.close()
        except Exception:
            pass


def _load_minute_frame_with_tqbacktest(
    *,
    case: BacktestCase,
    tq_symbol: str,
    account: str,
    password: str,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    chunk_end = case.end_dt
    while chunk_end >= case.start_dt:
        chunk_start = max(case.start_dt, chunk_end - timedelta(days=BACKTEST_MINUTE_CHUNK_DAYS - 1))
        api = None
        try:
            api, _anchor_dt = _open_backtest_anchor_api(
                target_dt=chunk_end,
                account=account,
                password=password,
            )
            minute_klines = api.get_kline_serial(
                tq_symbol,
                60,
                data_length=BACKTEST_MINUTE_DATA_LENGTH,
            )
            minute_df = _filter_minute_chunk_range(
                _klines_to_minute_frame(minute_klines),
                start_dt=chunk_start,
                end_dt=chunk_end,
            )
            if not minute_df.empty:
                chunks.append(minute_df)
        finally:
            try:
                if api is not None:
                    api.close()
            except Exception:
                pass
        chunk_end = chunk_start - timedelta(days=1)

    if not chunks:
        return _empty_minute_frame()

    merged = pd.concat(chunks, ignore_index=True)
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.sort_values("datetime", kind="stable")
    merged = merged.drop_duplicates(subset=["datetime"], keep="last")
    return merged.reset_index(drop=True)


def _filter_daily_range(frame: pd.DataFrame, case: BacktestCase) -> pd.DataFrame:
    if frame.empty:
        return _empty_daily_frame()
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.loc[out["date"].dt.date <= case.end_dt].copy()
    return out.reset_index(drop=True)


def _filter_minute_range(frame: pd.DataFrame, case: BacktestCase) -> pd.DataFrame:
    if frame.empty:
        return _empty_minute_frame()
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.loc[out["datetime"].dt.date.between(case.start_dt, case.end_dt)].copy()
    return out.reset_index(drop=True)


def _in_case_trade_day_count(*, case: BacktestCase, minute_df: pd.DataFrame) -> int:
    if minute_df.empty:
        return max(len(pd.bdate_range(case.start_dt, case.end_dt)), 1)
    trade_dates = pd.to_datetime(minute_df["datetime"]).dt.date.nunique()
    return max(int(trade_dates), 1)


def _load_cached_case_frames(
    *,
    case: BacktestCase,
    warmup_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    for version in BACKTEST_CACHE_READ_VERSIONS:
        daily_path, minute_path = _backtest_cache_paths_for_version(
            case=case,
            warmup_bars=warmup_bars,
            version=version,
        )
        if not daily_path.exists() or not minute_path.exists():
            continue

        try:
            daily_df = _filter_daily_range(_normalize_daily_frame(pd.read_parquet(daily_path)), case)
            minute_df = _filter_minute_range(_klines_to_minute_frame(pd.read_parquet(minute_path)), case)
            return daily_df, minute_df
        except Exception:
            for path in (daily_path, minute_path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            return None
    return None


def _write_cached_case_frames(
    *,
    case: BacktestCase,
    warmup_bars: int,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> None:
    daily_path, minute_path = _backtest_cache_paths(case=case, warmup_bars=warmup_bars)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_daily_frame(daily_df).to_parquet(daily_path, index=False)
    _klines_to_minute_frame(minute_df).to_parquet(minute_path, index=False)


def load_case_frames_with_tqbacktest(
    *,
    case: BacktestCase,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warmup_bars = _daily_warmup_bars(config)
    cached = _load_cached_case_frames(case=case, warmup_bars=warmup_bars)
    if cached is not None:
        return cached

    tq_cfg = config.get("tqsdk") or {}
    account = str(tq_cfg.get("account") or "").strip()
    password = str(tq_cfg.get("password") or "").strip()
    if not account or not password:
        raise ValueError("缺少 tqsdk 账号或密码配置")

    api = None
    try:
        backtest_start_dt = _backtest_start_dt(case, config)
        api = _create_backtest_api(
            start_dt=backtest_start_dt,
            end_dt=case.end_dt,
            account=account,
            password=password,
        )
        tq_symbol = resolve_case_tq_symbol(case, api=api)
    finally:
        try:
            if api is not None:
                api.close()
        except Exception:
            pass

    minute_df = _load_minute_frame_with_tqbacktest(
        case=case,
        tq_symbol=tq_symbol,
        account=account,
        password=password,
    )
    minute_df = _filter_minute_range(minute_df, case)
    daily_df = _load_daily_frame_with_tqbacktest(
        case=case,
        tq_symbol=tq_symbol,
        warmup_bars=warmup_bars,
        in_case_daily_bars=_in_case_trade_day_count(case=case, minute_df=minute_df),
        account=account,
        password=password,
    )
    _write_cached_case_frames(
        case=case,
        warmup_bars=warmup_bars,
        daily_df=daily_df,
        minute_df=minute_df,
    )
    return daily_df, minute_df


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
    return make_trade_plan_from_raw_builder(
        case=case,
        daily_df=daily_df,
        pre_market_cfg=pre_market_cfg,
        raw_builder=pre_market.build_trade_plan_from_daily_df,
        strategy_family=case.strategy_family,
    )


def make_trade_plan_from_raw_builder(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
    raw_builder,
    strategy_family: str = "",
) -> TradePlan | None:
    raw = raw_builder(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    if raw is None or not raw.get("actionable"):
        return None

    reversal_status = raw.get("reversal_status") or {}
    entry_family = str(raw.get("entry_family") or "").strip()
    entry_signal_type = str(raw.get("entry_signal_type") or reversal_status.get("signal_type") or "")
    entry_signal_detail = str(raw.get("entry_signal_detail") or "")
    resolved_strategy_family = str(raw.get("strategy_family") or strategy_family or case.strategy_family or "").strip()
    signal_date = str(reversal_status.get("signal_date") or "")
    if not signal_date and not daily_df.empty:
        signal_date = str(pd.to_datetime(daily_df["date"]).max().date())
    if not signal_date:
        signal_date = case.start_dt.isoformat()
    meta = {
        "strategy_family": resolved_strategy_family,
        "entry_family": entry_family,
        "entry_signal_type": entry_signal_type,
        "entry_signal_detail": entry_signal_detail,
    }
    for key in (
        "rr",
        "admission_rr",
        "risk_pct",
        "contract_multiplier",
        "margin_rate",
        "risk_per_lot",
        "margin_per_lot",
        "suggested_lots",
        "fundamental_snapshot",
        "fundamental_reversal_confirmed",
        "fundamental_coverage_score",
        "fundamental_coverage_status",
        "fundamental_domains_present",
        "fundamental_domains_missing",
        "fundamental_missing_domain_reasons",
        "fundamental_extreme_state_direction",
        "fundamental_extreme_state_reasons",
        "fundamental_marginal_turn_direction",
        "fundamental_marginal_turn_reasons",
        "fundamental_only_proxy_evidence",
        "only_proxy_evidence",
    ):
        if key in raw:
            meta[key] = raw[key]
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
        signal_type=entry_signal_type,
        meta=meta,
    )


def _empty_phase2_rejection_counts() -> dict[str, int]:
    return {key: 0 for key in PHASE2_REJECTION_KEYS}


def _empty_phase2_entry_family_counts() -> dict[str, int]:
    return {key: 0 for key in PHASE2_ENTRY_FAMILY_KEYS}


def _empty_phase2_strategy_family_counts() -> dict[str, int]:
    return {key: 0 for key in PHASE2_STRATEGY_FAMILY_KEYS}


def _score_gate_passed(direction: str, total: float) -> bool:
    return (direction == "long" and total > 20) or (direction == "short" and total < -20)


def _entry_family_for_plan(plan: TradePlan | None) -> str:
    if plan is None:
        return ""
    entry_family = str(plan.meta.get("entry_family") or "").strip()
    if entry_family in {"reversal", "trend"}:
        return entry_family
    return ""


def _strategy_family_for_plan(plan: TradePlan | None, case: BacktestCase | None = None) -> str:
    if plan is not None:
        strategy_family = str(plan.meta.get("strategy_family") or "").strip()
        if strategy_family in {STRATEGY_REVERSAL, STRATEGY_TREND}:
            return strategy_family
    if case is not None and case.strategy_family in {STRATEGY_REVERSAL, STRATEGY_TREND}:
        return case.strategy_family
    return ""


def _backtest_fundamental_mode(pre_market_cfg: dict) -> str:
    backtest_cfg = _dict_config((pre_market_cfg or {}).get("backtest"))
    raw_mode = (
        (pre_market_cfg or {}).get("backtest_fundamental_mode")
        or (pre_market_cfg or {}).get("fundamental_data_mode")
        or backtest_cfg.get("fundamental_mode")
        or backtest_cfg.get("fundamental_data_mode")
        or BACKTEST_FUNDAMENTAL_MODE_STRICT
    )
    mode = str(raw_mode).strip().lower()
    if mode == BACKTEST_FUNDAMENTAL_MODE_PROXY:
        return BACKTEST_FUNDAMENTAL_MODE_PROXY
    return BACKTEST_FUNDAMENTAL_MODE_STRICT


def _mapping_has_key(mapping: dict[str, object], key: str) -> bool:
    if key in mapping:
        return True
    snapshot = mapping.get("fundamental_snapshot")
    if isinstance(snapshot, dict):
        if key in snapshot:
            return True
        if key.startswith("fundamental_") and key.removeprefix("fundamental_") in snapshot:
            return True
    return False


def _mapping_value(mapping: dict[str, object], key: str, default: object = None) -> object:
    if key in mapping:
        return mapping.get(key)
    snapshot = mapping.get("fundamental_snapshot")
    if isinstance(snapshot, dict):
        if key in snapshot:
            return snapshot.get(key)
        if key.startswith("fundamental_"):
            return snapshot.get(key.removeprefix("fundamental_"), default)
    return default


def _backtest_fundamental_assessment(mapping: dict[str, object] | None) -> dict[str, object]:
    source = mapping or {}
    tracked_keys = (
        "fundamental_snapshot",
        "fundamental_reversal_confirmed",
        "fundamental_coverage_score",
        "fundamental_coverage_status",
        "fundamental_domains_present",
        "fundamental_domains_missing",
        "fundamental_missing_domain_reasons",
        "fundamental_only_proxy_evidence",
        "only_proxy_evidence",
    )
    has_fundamental_fields = any(_mapping_has_key(source, key) for key in tracked_keys)
    confirmed = bool(_mapping_value(source, "fundamental_reversal_confirmed", False))
    only_proxy = bool(
        _mapping_value(source, "fundamental_only_proxy_evidence", False)
        or _mapping_value(source, "only_proxy_evidence", False)
    )
    if confirmed and not only_proxy:
        status = "confirmed"
    elif only_proxy:
        status = "proxy"
    elif has_fundamental_fields:
        status = "unconfirmed"
    else:
        status = "proxy_missing"
    return {
        "confirmed": status == "confirmed",
        "proxy_used": status != "confirmed",
        "status": status,
        "coverage_score": _mapping_value(source, "fundamental_coverage_score", 0.0),
        "coverage_status": _mapping_value(source, "fundamental_coverage_status", ""),
    }


def _annotate_backtest_fundamental_plan(
    plan: TradePlan,
    *,
    mode: str,
    assessment: dict[str, object],
) -> TradePlan:
    meta = dict(plan.meta)
    meta.update(
        {
            "backtest_fundamental_mode": mode,
            "backtest_fundamental_status": str(assessment.get("status") or ""),
            "backtest_fundamental_confirmed": bool(assessment.get("confirmed")),
            "backtest_fundamental_proxy_used": bool(assessment.get("proxy_used")),
        }
    )
    return replace(plan, meta=meta)


def _with_backtest_fundamental_debug(
    snapshot: dict[str, object],
    *,
    mode: str,
    assessment: dict[str, object],
) -> dict[str, object]:
    out = dict(snapshot)
    out.update(
        {
            "backtest_fundamental_mode": mode,
            "backtest_fundamental_status": str(assessment.get("status") or ""),
            "backtest_fundamental_confirmed": bool(assessment.get("confirmed")),
            "backtest_fundamental_proxy_used": bool(assessment.get("proxy_used")),
        }
    )
    return out


def _apply_backtest_fundamental_mode(
    *,
    case: BacktestCase,
    plan: TradePlan | None,
    pre_market_cfg: dict,
    source: dict[str, object] | None = None,
    debug_snapshot: dict[str, object] | None = None,
) -> tuple[TradePlan | None, dict[str, int], dict[str, object]]:
    rejection_counts = _empty_phase2_rejection_counts()
    snapshot = debug_snapshot or _phase2_debug_snapshot_from_plan(plan)
    if plan is None:
        return None, rejection_counts, snapshot
    if _strategy_family_for_plan(plan, case) != STRATEGY_REVERSAL:
        return plan, rejection_counts, snapshot

    mode = _backtest_fundamental_mode(pre_market_cfg)
    assessment = _backtest_fundamental_assessment(source or plan.meta)
    annotated_plan = _annotate_backtest_fundamental_plan(
        plan,
        mode=mode,
        assessment=assessment,
    )
    snapshot = _with_backtest_fundamental_debug(
        snapshot,
        mode=mode,
        assessment=assessment,
    )

    if mode == BACKTEST_FUNDAMENTAL_MODE_STRICT and not bool(assessment.get("confirmed")):
        rejection_counts["phase2_reject_missing_fundamental_days"] += 1
        rejection_counts["backtest_fundamental_blocked_days"] = 1
        return None, rejection_counts, snapshot

    return annotated_plan, rejection_counts, snapshot


def _phase2_debug_snapshot_from_raw(raw: dict) -> dict[str, object]:
    reversal = raw.get("reversal_status") or {}
    trend = raw.get("trend_status") or {}
    entry_signal_type = str(raw.get("entry_signal_type") or reversal.get("signal_type") or "")
    reversal_signal_days_ago = reversal.get("signal_days_ago")
    reversal_signal_fresh = bool(raw.get("reversal_signal_fresh"))
    if not reversal_signal_fresh and reversal.get("has_signal") and reversal_signal_days_ago is not None:
        reversal_signal_fresh = int(reversal_signal_days_ago) <= pre_market.REVERSAL_FRESH_SIGNAL_MAX_DAYS_AGO
    direction = str(raw.get("direction") or "")
    score = float(raw.get("score") or 0.0)
    rr = float(raw.get("rr") or 0.0)
    return {
        "strategy_family": str(raw.get("strategy_family") or ""),
        "score": score,
        "rr": rr,
        "phase2_score_gate_passed": bool(raw.get("phase2_score_gate_passed"))
        if "phase2_score_gate_passed" in raw
        else _score_gate_passed(direction, score),
        "phase2_rr_gate_passed": bool(raw.get("phase2_rr_gate_passed"))
        if "phase2_rr_gate_passed" in raw
        else rr >= 1.0,
        "entry_family": str(raw.get("entry_family") or "").strip(),
        "entry_signal_type": entry_signal_type,
        "entry_signal_detail": str(raw.get("entry_signal_detail") or ""),
        "reversal_has_signal": bool(reversal.get("has_signal")),
        "reversal_signal_type": str(reversal.get("signal_type") or ""),
        "reversal_signal_days_ago": reversal_signal_days_ago,
        "reversal_signal_strength": float(reversal.get("signal_strength") or 0.0),
        "reversal_signal_fresh": reversal_signal_fresh,
        "reversal_current_stage": str(reversal.get("current_stage") or ""),
        "reversal_next_expected": str(reversal.get("next_expected") or ""),
        "trend_has_signal": bool(trend.get("has_signal")),
        "trend_signal_type": str(trend.get("signal_type") or ""),
        "trend_phase": str(trend.get("phase") or ""),
        "trend_phase_ok": bool(trend.get("phase_ok")) if "phase_ok" in trend else False,
        "trend_slope_ok": bool(trend.get("slope_ok")) if "slope_ok" in trend else False,
        "trend_indicator_ok": bool(trend.get("trend_indicator_ok")) if "trend_indicator_ok" in trend else False,
    }


def _rr_from_plan(plan: TradePlan) -> float:
    if plan.direction == "long":
        return (plan.tp1 - plan.entry_ref) / (plan.entry_ref - plan.stop + 1e-12)
    return (plan.entry_ref - plan.tp1) / (plan.stop - plan.entry_ref + 1e-12)


def _phase2_debug_snapshot_from_plan(plan: TradePlan | None) -> dict[str, object]:
    if plan is None:
        return {
            "strategy_family": "",
            "score": 0.0,
            "rr": 0.0,
            "phase2_score_gate_passed": False,
            "phase2_rr_gate_passed": False,
            "entry_family": "",
            "entry_signal_type": "",
            "entry_signal_detail": "",
            "reversal_has_signal": False,
            "reversal_signal_type": "",
            "reversal_signal_days_ago": None,
            "reversal_signal_strength": 0.0,
            "reversal_signal_fresh": False,
            "reversal_current_stage": "",
            "reversal_next_expected": "",
            "trend_has_signal": False,
            "trend_signal_type": "",
            "trend_phase": "",
            "trend_phase_ok": False,
            "trend_slope_ok": False,
            "trend_indicator_ok": False,
        }
    entry_signal_type = str(plan.meta.get("entry_signal_type") or plan.signal_type or "")
    return {
        "strategy_family": _strategy_family_for_plan(plan),
        "score": float(plan.phase2_score),
        "rr": float(_rr_from_plan(plan)),
        "phase2_score_gate_passed": True,
        "phase2_rr_gate_passed": True,
        "entry_family": _entry_family_for_plan(plan),
        "entry_signal_type": entry_signal_type,
        "entry_signal_detail": str(plan.meta.get("entry_signal_detail") or ""),
        "reversal_has_signal": _entry_family_for_plan(plan) == "reversal",
        "reversal_signal_type": entry_signal_type if _entry_family_for_plan(plan) == "reversal" else "",
        "reversal_signal_days_ago": None,
        "reversal_signal_strength": 0.0,
        "reversal_signal_fresh": False,
        "reversal_current_stage": "",
        "reversal_next_expected": "",
        "trend_has_signal": _entry_family_for_plan(plan) == "trend",
        "trend_signal_type": entry_signal_type if _entry_family_for_plan(plan) == "trend" else "",
        "trend_phase": "",
        "trend_phase_ok": False,
        "trend_slope_ok": False,
        "trend_indicator_ok": False,
    }


def _phase2_state_from_rejections(
    *,
    history_insufficient: bool,
    duplicate_rejected: bool,
    actionable: bool,
    rejection_counts: dict[str, int],
) -> str:
    if history_insufficient:
        return "history_insufficient"
    if duplicate_rejected:
        return "duplicate_signal"
    if actionable:
        return "actionable"

    states: list[str] = []
    if rejection_counts.get("phase2_reject_no_signal_days"):
        states.append("no_signal")
    if rejection_counts.get("phase2_reject_score_gate_days"):
        states.append("score_gate")
    if rejection_counts.get("phase2_reject_rr_gate_days"):
        states.append("rr_gate")
    if rejection_counts.get("phase2_reject_missing_fundamental_days"):
        states.append("missing_fundamental")
    if not states:
        return "no_plan"
    return "+".join(states)


def _build_phase2_debug_day(
    *,
    trade_date,
    visible_daily_rows: int,
    history_insufficient: bool,
    duplicate_rejected: bool,
    plan: TradePlan | None,
    rejection_counts: dict[str, int],
    snapshot: dict[str, object],
) -> dict[str, object]:
    return {
        "trade_date": str(trade_date),
        "visible_daily_rows": int(visible_daily_rows),
        "phase2_state": _phase2_state_from_rejections(
            history_insufficient=history_insufficient,
            duplicate_rejected=duplicate_rejected,
            actionable=plan is not None,
            rejection_counts=rejection_counts,
        ),
        "strategy_family": str(snapshot.get("strategy_family") or ""),
        "score": float(snapshot.get("score") or 0.0),
        "rr": float(snapshot.get("rr") or 0.0),
        "phase2_score_gate_passed": bool(snapshot.get("phase2_score_gate_passed")),
        "phase2_rr_gate_passed": bool(snapshot.get("phase2_rr_gate_passed")),
        "entry_family": str(snapshot.get("entry_family") or ""),
        "entry_signal_type": str(snapshot.get("entry_signal_type") or ""),
        "entry_signal_detail": str(snapshot.get("entry_signal_detail") or ""),
        "reversal_has_signal": bool(snapshot.get("reversal_has_signal")),
        "reversal_signal_type": str(snapshot.get("reversal_signal_type") or ""),
        "reversal_signal_days_ago": snapshot.get("reversal_signal_days_ago"),
        "reversal_signal_strength": float(snapshot.get("reversal_signal_strength") or 0.0),
        "reversal_signal_fresh": bool(snapshot.get("reversal_signal_fresh")),
        "reversal_current_stage": str(snapshot.get("reversal_current_stage") or ""),
        "reversal_next_expected": str(snapshot.get("reversal_next_expected") or ""),
        "trend_has_signal": bool(snapshot.get("trend_has_signal")),
        "trend_signal_type": str(snapshot.get("trend_signal_type") or ""),
        "trend_phase": str(snapshot.get("trend_phase") or ""),
        "trend_phase_ok": bool(snapshot.get("trend_phase_ok")),
        "trend_slope_ok": bool(snapshot.get("trend_slope_ok")),
        "trend_indicator_ok": bool(snapshot.get("trend_indicator_ok")),
        "backtest_fundamental_mode": str(snapshot.get("backtest_fundamental_mode") or ""),
        "backtest_fundamental_status": str(snapshot.get("backtest_fundamental_status") or ""),
        "backtest_fundamental_confirmed": bool(snapshot.get("backtest_fundamental_confirmed")),
        "backtest_fundamental_proxy_used": bool(snapshot.get("backtest_fundamental_proxy_used")),
    }


def _strategy_watch_raw(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
) -> dict | None:
    if case.strategy_family == STRATEGY_REVERSAL:
        return pre_market.build_reversal_trade_plan_from_daily_df(
            symbol=case.symbol,
            name=case.name,
            direction=case.direction,
            df=daily_df,
            cfg=pre_market_cfg,
            allow_watch_plan=True,
        )
    if case.strategy_family == STRATEGY_TREND:
        return pre_market.build_trend_trade_plan_from_daily_df(
            symbol=case.symbol,
            name=case.name,
            direction=case.direction,
            df=daily_df,
            cfg=pre_market_cfg,
            allow_watch_plan=True,
        )
    return None


def _phase2_outcome_from_raw(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
    raw: dict | None,
    plan_factory,
) -> tuple[TradePlan | None, dict[str, int], dict[str, object]]:
    if raw is None:
        return None, _empty_phase2_rejection_counts(), _phase2_debug_snapshot_from_plan(None)

    debug_snapshot = _phase2_debug_snapshot_from_raw(raw)
    if raw.get("actionable"):
        plan = plan_factory(
            case=case,
            daily_df=daily_df,
            pre_market_cfg=pre_market_cfg,
        )
        return _apply_backtest_fundamental_mode(
            case=case,
            plan=plan,
            pre_market_cfg=pre_market_cfg,
            source=raw,
            debug_snapshot=debug_snapshot,
        )

    rejection_counts = _empty_phase2_rejection_counts()
    reversal = raw.get("reversal_status") or {}
    trend = raw.get("trend_status") or {}
    if not reversal.get("has_signal") and not trend.get("has_signal"):
        rejection_counts["phase2_reject_no_signal_days"] += 1
        return None, rejection_counts, debug_snapshot

    total = float(raw.get("score") or 0.0)
    rr = float(raw.get("rr") or 0.0)
    score_gate_passed = bool(raw.get("phase2_score_gate_passed")) if "phase2_score_gate_passed" in raw else _score_gate_passed(case.direction, total)
    rr_gate_passed = bool(raw.get("phase2_rr_gate_passed")) if "phase2_rr_gate_passed" in raw else rr >= 1.0
    if not score_gate_passed:
        rejection_counts["phase2_reject_score_gate_days"] += 1
    if not rr_gate_passed:
        rejection_counts["phase2_reject_rr_gate_days"] += 1
    return None, rejection_counts, debug_snapshot


def _evaluate_phase2_plan(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
    plan_factory,
    capture_debug: bool = False,
) -> tuple[TradePlan | None, dict[str, int], dict[str, object]]:
    if plan_factory is not make_trade_plan_from_phase2:
        if capture_debug:
            raw = _strategy_watch_raw(
                case=case,
                daily_df=daily_df,
                pre_market_cfg=pre_market_cfg,
            )
            if raw is not None:
                return _phase2_outcome_from_raw(
                    case=case,
                    daily_df=daily_df,
                    pre_market_cfg=pre_market_cfg,
                    raw=raw,
                    plan_factory=plan_factory,
                )
        plan = plan_factory(case=case, daily_df=daily_df, pre_market_cfg=pre_market_cfg)
        return _apply_backtest_fundamental_mode(
            case=case,
            plan=plan,
            pre_market_cfg=pre_market_cfg,
            debug_snapshot=_phase2_debug_snapshot_from_plan(plan),
        )

    raw = pre_market.build_trade_plan_from_daily_df(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    return _phase2_outcome_from_raw(
        case=case,
        daily_df=daily_df,
        pre_market_cfg=pre_market_cfg,
        raw=raw,
        plan_factory=make_trade_plan_from_phase2,
    )


def _active_plan_still_valid(
    *,
    case: BacktestCase,
    active_plan: TradePlan,
    daily_df: pd.DataFrame,
    pre_market_cfg: dict,
    plan_factory,
) -> bool:
    strategy_family = _strategy_family_for_plan(active_plan, case)
    entry_family = _entry_family_for_plan(active_plan)

    if strategy_family == STRATEGY_TREND or entry_family == "trend":
        hold_status = pre_market.assess_active_trend_hold_from_daily_df(
            symbol=case.symbol,
            name=case.name,
            direction=case.direction,
            df=daily_df,
            cfg=pre_market_cfg,
        )
        profile = resolve_trade_management_profile(
            plan=active_plan,
            case=case,
            pre_market_cfg=pre_market_cfg,
        )
        return trend_daily_review_keeps_position(
            direction=case.direction,
            hold_status=hold_status,
            profile=profile,
        )

    if strategy_family == STRATEGY_REVERSAL or entry_family == "reversal":
        hold_status = pre_market.assess_active_reversal_hold_from_daily_df(
            symbol=case.symbol,
            name=case.name,
            direction=case.direction,
            df=daily_df,
            cfg=pre_market_cfg,
        )
        return bool(hold_status and hold_status.get("hold_valid"))

    if plan_factory is not make_trade_plan_from_phase2:
        refreshed_plan = plan_factory(case=case, daily_df=daily_df, pre_market_cfg=pre_market_cfg)
        return refreshed_plan is not None

    refreshed_plan = make_trade_plan_from_phase2(
        case=case,
        daily_df=daily_df,
        pre_market_cfg=pre_market_cfg,
    )
    return refreshed_plan is not None


def _safe_generate_signals(df: pd.DataFrame, direction: str, cfg: dict) -> list[dict]:
    if len(df) < 2:
        return []
    return generate_signals(df, direction, cfg)


def _calc_pnl_ratio(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _reward_risk(direction: str, *, entry: float, target: float, stop: float) -> float:
    if direction == "long":
        return (target - entry) / (entry - stop + 1e-12)
    return (entry - target) / (stop - entry + 1e-12)


def _execution_plan_distances(plan: TradePlan) -> tuple[float, float, float]:
    if plan.direction == "long":
        return (
            plan.entry_ref - plan.stop,
            plan.tp1 - plan.entry_ref,
            plan.tp2 - plan.entry_ref,
        )
    return (
        plan.stop - plan.entry_ref,
        plan.entry_ref - plan.tp1,
        plan.entry_ref - plan.tp2,
    )


def _float_meta(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _rebase_plan_to_entry(plan: TradePlan, entry_price: float) -> TradePlan:
    risk_distance, tp1_distance, tp2_distance = _execution_plan_distances(plan)
    if entry_price <= 0 or risk_distance <= 0 or tp1_distance <= 0 or tp2_distance <= 0:
        return plan

    if plan.direction == "long":
        stop = entry_price - risk_distance
        tp1 = entry_price + tp1_distance
        tp2 = entry_price + tp2_distance
    else:
        stop = entry_price + risk_distance
        tp1 = entry_price - tp1_distance
        tp2 = entry_price - tp2_distance

    meta = dict(plan.meta)
    multiplier = _float_meta(meta, "contract_multiplier", 1.0)
    margin_rate = _float_meta(meta, "margin_rate", 0.0)
    meta.update(
        {
            "planned_entry_ref": float(plan.entry_ref),
            "planned_stop": float(plan.stop),
            "planned_tp1": float(plan.tp1),
            "planned_tp2": float(plan.tp2),
            "execution_entry": float(entry_price),
            "execution_stop": float(stop),
            "execution_tp1": float(tp1),
            "execution_tp2": float(tp2),
            "execution_rr": float(_reward_risk(plan.direction, entry=entry_price, target=tp1, stop=stop)),
            "execution_admission_rr": float(_reward_risk(plan.direction, entry=entry_price, target=tp2, stop=stop)),
            "execution_risk_per_lot": float(risk_distance * max(multiplier, 0.0)),
            "execution_margin_per_lot": float(entry_price * max(multiplier, 0.0) * max(margin_rate, 0.0)),
            "execution_rebased": bool(abs(entry_price - plan.entry_ref) > 1e-12),
        }
    )
    return TradePlan(
        trade_id=plan.trade_id,
        symbol=plan.symbol,
        direction=plan.direction,
        plan_date=plan.plan_date,
        entry_ref=float(entry_price),
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        phase2_score=plan.phase2_score,
        signal_type=plan.signal_type,
        meta=meta,
    )


def _dict_config(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _tp1_touched(direction: str, bar: pd.Series, plan: TradePlan) -> bool:
    if direction == "long":
        return float(bar["high"]) >= plan.tp1
    return float(bar["low"]) <= plan.tp1


def _stop_touched(direction: str, bar: pd.Series, stop: float) -> bool:
    if direction == "long":
        return float(bar["low"]) <= stop
    return float(bar["high"]) >= stop


def _tp2_touched(direction: str, bar: pd.Series, plan: TradePlan) -> bool:
    if direction == "long":
        return float(bar["high"]) >= plan.tp2
    return float(bar["low"]) <= plan.tp2


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
    pnl_ratio_override: float | None = None,
    extra_meta: dict[str, object] | None = None,
) -> TradeRecord:
    entry_ts = pd.Timestamp(entry_time)
    exit_ts = pd.Timestamp(exit_time)
    meta = dict(plan.meta)
    meta.update({
        "plan_date": plan.plan_date,
        "strategy_family": _strategy_family_for_plan(plan, case),
        "entry_family": _entry_family_for_plan(plan),
        "entry_signal_type": str(plan.meta.get("entry_signal_type") or plan.signal_type or ""),
        "entry_signal_detail": str(plan.meta.get("entry_signal_detail") or ""),
        "signal_type": plan.signal_type,
        "phase2_score": plan.phase2_score,
    })
    if extra_meta:
        meta.update(extra_meta)
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
        pnl_ratio=pnl_ratio_override
        if pnl_ratio_override is not None
        else _calc_pnl_ratio(case.direction, entry_price, exit_price),
        meta=meta,
    )


def _resolve_exit_on_bar(
    *, direction: str, bar: pd.Series, plan: TradePlan, stop: float | None = None
) -> tuple[str, float] | None:
    stop_price = plan.stop if stop is None else stop
    if _stop_touched(direction, bar, stop_price):
        return "stop", stop_price
    if _tp2_touched(direction, bar, plan):
        return "tp2", plan.tp2
    return None


def _phase2_min_history_bars(pre_market_cfg: dict) -> int:
    return max(int((pre_market_cfg or {}).get("min_history_bars", 60)), 1)


def run_case_from_frames(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    plan_factory: Callable[..., TradePlan | None] = make_trade_plan_from_phase2,
    pre_market_cfg: dict,
    signal_cfg: dict,
    capture_debug: bool = False,
) -> BacktestResult:
    daily_dates = pd.to_datetime(daily_df["date"]) if not daily_df.empty else pd.Series(dtype="datetime64[ns]")
    trades: list[TradeRecord] = []
    diagnostics: dict[str, int] = {
        "daily_rows_loaded": int(len(daily_df)),
        "first_trade_day_visible_daily_rows": 0,
        "trade_days_total": 0,
        "phase2_history_insufficient_days": 0,
        "phase2_non_actionable_days": 0,
        "phase2_actionable_days": 0,
        "phase3_signal_eval_bars": 0,
        "phase3_entry_signal_hits": 0,
        "trades_opened": 0,
        "backtest_fundamental_blocked_days": 0,
        "backtest_fundamental_proxy_days": 0,
    }
    diagnostics.update(_empty_phase2_rejection_counts())
    diagnostics.update(_empty_phase2_entry_family_counts())
    diagnostics.update(_empty_phase2_strategy_family_counts())
    phase2_debug_days: list[dict[str, object]] = []
    if not daily_df.empty:
        diagnostics["daily_first_date"] = str(daily_dates.min().date())
        diagnostics["daily_last_date"] = str(daily_dates.max().date())
    else:
        diagnostics["daily_first_date"] = "NONE"
        diagnostics["daily_last_date"] = "NONE"

    minute_data = minute_df.copy()
    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0}, diagnostics=diagnostics)

    minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])
    minute_data["trade_date"] = minute_data["datetime"].dt.date
    minute_data = minute_data.sort_values("datetime", kind="stable")
    minute_data = minute_data.loc[
        minute_data["trade_date"].between(case.start_dt, case.end_dt)
    ].copy()

    if minute_data.empty:
        return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": 0}, diagnostics=diagnostics)

    visible_daily = daily_df.copy()
    visible_daily["date"] = pd.to_datetime(visible_daily["date"])
    visible_daily = visible_daily.sort_values("date", kind="stable")
    visible_daily["trade_date"] = visible_daily["date"].dt.date

    active_plan: TradePlan | None = None
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    bars_held = 0
    tp1_hit = False
    tp1_exit_price: float | None = None
    active_stop: float | None = None
    active_management_profile: TradeManagementProfile | None = None
    consumed_trade_ids: set[str] = set()
    min_history_bars = _phase2_min_history_bars(pre_market_cfg)

    for trade_date, day_minutes in minute_data.groupby("trade_date", sort=True):
        diagnostics["trade_days_total"] += 1
        day_minutes = day_minutes.sort_values("datetime", kind="stable")
        day_visible_daily = visible_daily.loc[visible_daily["trade_date"] < trade_date].copy()
        if diagnostics["trade_days_total"] == 1:
            diagnostics["first_trade_day"] = str(trade_date)
            diagnostics["first_trade_day_visible_daily_rows"] = int(len(day_visible_daily))
            if not day_visible_daily.empty:
                diagnostics["first_trade_day_visible_first_date"] = str(day_visible_daily["date"].min().date())
                diagnostics["first_trade_day_visible_last_date"] = str(day_visible_daily["date"].max().date())
            else:
                diagnostics["first_trade_day_visible_first_date"] = "NONE"
                diagnostics["first_trade_day_visible_last_date"] = "NONE"

        if active_plan is None:
            history_insufficient = len(day_visible_daily) < min_history_bars
            day_plan, rejection_counts, debug_snapshot = _evaluate_phase2_plan(
                case=case,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
                plan_factory=plan_factory,
                capture_debug=capture_debug,
            )
            duplicate_rejected = False
            if day_plan is not None and day_plan.trade_id in consumed_trade_ids:
                diagnostics["phase2_reject_duplicate_signal_days"] += 1
                day_plan = None
                duplicate_rejected = True
            if history_insufficient:
                diagnostics["phase2_history_insufficient_days"] += 1
            elif day_plan is None:
                diagnostics["phase2_non_actionable_days"] += 1
                for key, value in rejection_counts.items():
                    diagnostics[key] += value
            else:
                diagnostics["phase2_actionable_days"] += 1
                if bool(day_plan.meta.get("backtest_fundamental_proxy_used")):
                    diagnostics["backtest_fundamental_proxy_days"] += 1
                entry_family = _entry_family_for_plan(day_plan)
                strategy_family = _strategy_family_for_plan(day_plan, case)
                if entry_family == "trend":
                    diagnostics["phase2_actionable_trend_days"] += 1
                elif entry_family == "reversal":
                    diagnostics["phase2_actionable_reversal_days"] += 1
                if strategy_family == STRATEGY_TREND:
                    diagnostics["phase2_actionable_strategy_trend_days"] += 1
                elif strategy_family == STRATEGY_REVERSAL:
                    diagnostics["phase2_actionable_strategy_reversal_days"] += 1
            if capture_debug:
                phase2_debug_days.append(
                    _build_phase2_debug_day(
                        trade_date=trade_date,
                        visible_daily_rows=len(day_visible_daily),
                        history_insufficient=history_insufficient,
                        duplicate_rejected=duplicate_rejected,
                        plan=day_plan,
                        rejection_counts=rejection_counts,
                        snapshot=debug_snapshot,
                    )
                )
        else:
            if not _active_plan_still_valid(
                case=case,
                active_plan=active_plan,
                daily_df=day_visible_daily,
                pre_market_cfg=pre_market_cfg,
                plan_factory=plan_factory,
            ):
                first_bar = day_minutes.iloc[0]
                management_profile = active_management_profile or resolve_trade_management_profile(
                    plan=active_plan,
                    case=case,
                    pre_market_cfg=pre_market_cfg,
                )
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
                        pnl_ratio_override=realized_pnl_ratio(
                            direction=case.direction,
                            entry_price=float(entry_price),
                            exit_price=float(first_bar["open"]),
                            tp1_hit=tp1_hit,
                            tp1_exit_price=tp1_exit_price,
                            tp1_exit_fraction=management_profile.tp1_exit_fraction,
                        ),
                        extra_meta=trade_management_meta(
                            profile=management_profile,
                            tp1_hit=tp1_hit,
                            tp1_exit_price=tp1_exit_price,
                            protective_stop=active_stop or active_plan.stop,
                        ),
                    )
                )
                active_plan = None
                entry_time = None
                entry_price = None
                bars_held = 0
                tp1_hit = False
                tp1_exit_price = None
                active_stop = None
                active_management_profile = None
                continue
            day_plan = None

        for minute_idx in range(len(day_minutes)):
            current_minute = day_minutes.iloc[minute_idx]

            if active_plan is None:
                if day_plan is None:
                    break
                diagnostics["phase3_signal_eval_bars"] += 1
                partial_5m = aggregate_partial_5m_bars(day_minutes.iloc[: minute_idx + 1])
                signals = _safe_generate_signals(partial_5m, case.direction, signal_cfg)
                entry_family = _entry_family_for_plan(day_plan)
                if any(
                    is_entry_confirmation_signal(signal, case.direction, entry_family)
                    for signal in signals
                ):
                    diagnostics["phase3_entry_signal_hits"] += 1
                    consumed_trade_ids.add(day_plan.trade_id)
                    entry_time = pd.Timestamp(current_minute["datetime"])
                    entry_price = float(current_minute["close"])
                    active_plan = _rebase_plan_to_entry(day_plan, entry_price)
                    active_management_profile = resolve_trade_management_profile(
                        plan=active_plan,
                        case=case,
                        pre_market_cfg=pre_market_cfg,
                    )
                    bars_held = 0
                    tp1_hit = False
                    tp1_exit_price = None
                    active_stop = active_plan.stop
                    diagnostics["trades_opened"] += 1
                    entry_family = _entry_family_for_plan(day_plan)
                    strategy_family = _strategy_family_for_plan(day_plan, case)
                    if entry_family == "trend":
                        diagnostics["trades_opened_trend"] += 1
                    elif entry_family == "reversal":
                        diagnostics["trades_opened_reversal"] += 1
                    if strategy_family == STRATEGY_TREND:
                        diagnostics["trades_opened_strategy_trend"] += 1
                    elif strategy_family == STRATEGY_REVERSAL:
                        diagnostics["trades_opened_strategy_reversal"] += 1
                continue

            bars_held += 1
            stop_price = active_stop if active_stop is not None else active_plan.stop
            management_profile = active_management_profile or resolve_trade_management_profile(
                plan=active_plan,
                case=case,
                pre_market_cfg=pre_market_cfg,
            )
            if _stop_touched(case.direction, current_minute, stop_price):
                exit = ("stop", stop_price)
            else:
                exit = None
                if not tp1_hit and _tp1_touched(case.direction, current_minute, active_plan):
                    tp1_hit = True
                    tp1_exit_price = active_plan.tp1
                    active_stop = stop_after_first_target(
                        direction=case.direction,
                        entry_price=float(entry_price),
                        initial_stop=active_plan.stop,
                        current_stop=active_stop if active_stop is not None else active_plan.stop,
                        profile=management_profile,
                    )
                    if _stop_touched(case.direction, current_minute, active_stop):
                        exit = ("stop", active_stop)
                if exit is None and _tp2_touched(case.direction, current_minute, active_plan):
                    exit = ("tp2", active_plan.tp2)

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
                    pnl_ratio_override=realized_pnl_ratio(
                        direction=case.direction,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        tp1_hit=tp1_hit,
                        tp1_exit_price=tp1_exit_price,
                        tp1_exit_fraction=management_profile.tp1_exit_fraction,
                    ),
                    extra_meta=trade_management_meta(
                        profile=management_profile,
                        tp1_hit=tp1_hit,
                        tp1_exit_price=tp1_exit_price,
                        protective_stop=active_stop or active_plan.stop,
                    ),
                )
            )
            active_plan = None
            entry_time = None
            entry_price = None
            bars_held = 0
            tp1_hit = False
            tp1_exit_price = None
            active_stop = None
            active_management_profile = None
            break

    if active_plan is not None:
        final_bar = minute_data.iloc[-1]
        management_profile = active_management_profile or resolve_trade_management_profile(
            plan=active_plan,
            case=case,
            pre_market_cfg=pre_market_cfg,
        )
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
                pnl_ratio_override=realized_pnl_ratio(
                    direction=case.direction,
                    entry_price=float(entry_price),
                    exit_price=float(final_bar["close"]),
                    tp1_hit=tp1_hit,
                    tp1_exit_price=tp1_exit_price,
                    tp1_exit_fraction=management_profile.tp1_exit_fraction,
                ),
                extra_meta=trade_management_meta(
                    profile=management_profile,
                    tp1_hit=tp1_hit,
                    tp1_exit_price=tp1_exit_price,
                    protective_stop=active_stop or active_plan.stop,
                ),
            )
        )

    return BacktestResult(
        case_id=case.case_id,
        trades=trades,
        summary={"num_trades": len(trades)},
        diagnostics=diagnostics,
        debug={"phase2_days": phase2_debug_days} if capture_debug else {},
    )
