from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from backtest.cases import get_case_group
from backtest.execution_quality import medium_term_quality_from_daily as _shared_medium_term_quality_from_daily
from backtest.models import BacktestCase
from backtest.phase23 import (
    _build_phase2_debug_day,
    _evaluate_phase2_plan,
    _phase2_min_history_bars,
    load_case_frames_with_tqbacktest,
)
from run_account_backtest import (
    _config_with_fundamental_mode,
    _resolved_fundamental_mode,
    load_config,
)
from run_backtest import _plan_factory_for_case


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
PHASE2_SCORE_VALIDATION_CACHE_VERSION = "phase2-score-validation-v5"

SHORT_TERM_PRICE_VOLUME_KEYS = ("MACD", "动量", "量价关系", "VSA信号", "持仓信号")
TREND_STRUCTURE_KEYS = ("均线排列",)
TREND_PHASE_KEYS = ("Wyckoff阶段",)

COMPONENT_SPECS = (
    {
        "field": "short_term_price_volume_score",
        "meaning": "短期量价强度",
        "role": "入场优先级",
    },
    {
        "field": "trend_structure_score",
        "meaning": "趋势结构强度",
        "role": "持仓资格过滤",
    },
    {
        "field": "trend_phase_score",
        "meaning": "趋势阶段强度",
        "role": "持仓资格过滤",
    },
    {
        "field": "admission_reward_risk",
        "meaning": "计划收益风险比",
        "role": "仓位上限参考",
    },
    {
        "field": "medium_term_quality_score",
        "meaning": "中期趋势机会质量",
        "role": "持仓与仓位参考候选",
    },
    {
        "field": "medium_term_momentum_score",
        "meaning": "中期时间序列动量",
        "role": "趋势方向候选",
    },
    {
        "field": "medium_term_efficiency_score",
        "meaning": "中期趋势效率",
        "role": "趋势质量候选",
    },
    {
        "field": "medium_term_trend_state_score",
        "meaning": "ADX/DMI 趋势状态",
        "role": "趋势市场过滤候选",
    },
    {
        "field": "medium_term_freshness_score",
        "meaning": "Aroon 趋势新鲜度",
        "role": "趋势老化过滤候选",
    },
    {
        "field": "medium_term_regime_score",
        "meaning": "Hurst 趋势状态",
        "role": "轻权重 regime 候选",
    },
)

DETAIL_BASE_FIELDS = [
    "case_id",
    "symbol",
    "name",
    "direction",
    "trade_date",
    "phase2_state",
    "score",
    "abs_score",
    "score_bucket",
    "short_term_price_volume_score",
    "trend_structure_score",
    "trend_phase_score",
    "trend_structure_filter_passed",
    "risk_reward_ratio",
    "admission_reward_risk",
    "stop_risk_pct",
    "risk_reward_gate_passed",
    "position_risk_gate_passed",
    "medium_term_quality_score",
    "medium_term_slope_score",
    "medium_term_persistence_score",
    "medium_term_structure_score",
    "medium_term_phase_alignment_score",
    "medium_term_entry_location_score",
    "medium_term_momentum_score",
    "medium_term_efficiency_score",
    "medium_term_trend_state_score",
    "medium_term_freshness_score",
    "medium_term_regime_score",
    "medium_term_quality_status",
    "rr",
    "phase2_score_gate_passed",
    "phase2_rr_gate_passed",
    "entry_family",
    "entry_signal_type",
    "trend_phase",
    "trend_phase_ok",
    "trend_slope_ok",
    "trend_indicator_ok",
    "entry_price",
    "available_forward_days",
]

BUCKET_SUMMARY_FIELDS = [
    "pool",
    "horizon_days",
    "score_bucket",
    "rows",
    "avg_abs_score",
    "actionable_rows",
    "mean_forward_return",
    "median_forward_return",
    "win_rate",
    "mean_max_favorable_return",
    "mean_max_adverse_return",
]

MONOTONICITY_FIELDS = [
    "pool",
    "horizon_days",
    "rows",
    "buckets",
    "spearman_abs_score_forward_return",
    "low_bucket_mean_forward_return",
    "high_bucket_mean_forward_return",
    "high_minus_low_forward_return",
    "adjacent_up_steps",
    "adjacent_pairs",
    "monotonic_bucket_pass",
]

COMPONENT_SUMMARY_FIELDS = [
    "pool",
    "component",
    "component_meaning",
    "component_role",
    "horizon_days",
    "rows",
    "spearman_component_forward_return",
    "low_component_mean",
    "middle_component_mean",
    "high_component_mean",
    "low_bucket_mean_forward_return",
    "middle_bucket_mean_forward_return",
    "high_bucket_mean_forward_return",
    "high_minus_low_forward_return",
    "adjacent_up_steps",
    "adjacent_pairs",
    "monotonic_bucket_pass",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase2 score vs future directional returns")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--fundamental-mode", choices=("strict", "proxy"))
    parser.add_argument("--horizons", default=",".join(str(item) for item in DEFAULT_HORIZONS))
    parser.add_argument("--phase2-score-cache-dir", type=Path)
    parser.add_argument("--write-missing-phase2-score-cache", action="store_true")
    parser.add_argument("--require-phase2-score-cache", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _case_for_year(case: BacktestCase, year: int) -> BacktestCase:
    return replace(
        case,
        case_id=f"{case.case_id}_{year}",
        start_dt=pd.Timestamp(year=int(year), month=1, day=1).date(),
        end_dt=pd.Timestamp(year=int(year), month=12, day=31).date(),
    )


def _parse_horizons(raw: str) -> tuple[int, ...]:
    horizons: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon < 1:
            raise ValueError("horizons must be positive integers")
        horizons.append(horizon)
    return tuple(sorted(set(horizons))) or DEFAULT_HORIZONS


def _default_output_prefix(*, case_group: str, year: int) -> Path:
    return Path("data/reports/backtest") / f"phase2_score_forward_return_{case_group}_{year}"


def _config_fingerprint(config: dict[str, Any]) -> str:
    safe = _json_safe(config)
    raw = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _phase2_score_cache_path(cache_dir: Path, *, case: BacktestCase, pre_market_cfg: dict[str, Any]) -> Path:
    digest = _config_fingerprint(pre_market_cfg)
    return Path(cache_dir) / f"{case.case_id}_{digest}_phase2_score_validation_cache.json"


def _phase2_score_cache_matches(
    payload: dict[str, Any],
    *,
    case: BacktestCase,
    pre_market_cfg: dict[str, Any],
) -> bool:
    return (
        payload.get("version") == PHASE2_SCORE_VALIDATION_CACHE_VERSION
        and payload.get("case_id") == case.case_id
        and payload.get("symbol") == case.symbol
        and payload.get("direction") == case.direction
        and payload.get("start_date") == case.start_dt.isoformat()
        and payload.get("end_date") == case.end_dt.isoformat()
        and payload.get("pre_market_fingerprint") == _config_fingerprint(pre_market_cfg)
    )


def _read_phase2_score_cache(
    path: Path,
    *,
    case: BacktestCase,
    pre_market_cfg: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _phase2_score_cache_matches(payload, case=case, pre_market_cfg=pre_market_cfg):
        return None
    return [dict(row) for row in payload.get("rows") or []]


def _write_phase2_score_cache(
    path: Path,
    *,
    case: BacktestCase,
    pre_market_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PHASE2_SCORE_VALIDATION_CACHE_VERSION,
        "case_id": case.case_id,
        "symbol": case.symbol,
        "direction": case.direction,
        "start_date": case.start_dt.isoformat(),
        "end_date": case.end_dt.isoformat(),
        "pre_market_fingerprint": _config_fingerprint(pre_market_cfg),
        "rows": rows,
    }
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _score_bucket(abs_score: float) -> str:
    value = float(abs_score)
    if value < 20.0:
        return "00_lt20"
    if value < 30.0:
        return "20_30"
    if value < 40.0:
        return "30_40"
    if value < 50.0:
        return "40_50"
    return "50_plus"


def _aligned_component_score(scores: dict[str, Any], keys: tuple[str, ...], direction: str) -> float:
    total = 0.0
    for key in keys:
        value = _float_or_nan(scores.get(key, 0.0))
        if not math.isfinite(value):
            continue
        if direction == "short":
            total += max(-value, 0.0)
        else:
            total += max(value, 0.0)
    return float(total)


def _component_fields_from_snapshot(snapshot: dict[str, Any], *, direction: str) -> dict[str, Any]:
    raw_scores = snapshot.get("scores")
    scores = dict(raw_scores) if isinstance(raw_scores, dict) else {}
    rr = _float_or_nan(snapshot.get("rr"))
    admission_rr = _float_or_nan(snapshot.get("admission_rr"))
    if not math.isfinite(admission_rr):
        admission_rr = rr
    trend_structure_filter_passed = bool(
        snapshot.get("trend_phase_ok")
        and snapshot.get("trend_slope_ok")
        and snapshot.get("trend_indicator_ok")
    )
    return {
        "short_term_price_volume_score": _aligned_component_score(
            scores,
            SHORT_TERM_PRICE_VOLUME_KEYS,
            direction,
        ),
        "trend_structure_score": _aligned_component_score(scores, TREND_STRUCTURE_KEYS, direction),
        "trend_phase_score": _aligned_component_score(scores, TREND_PHASE_KEYS, direction),
        "trend_structure_filter_passed": trend_structure_filter_passed,
        "risk_reward_ratio": rr,
        "admission_reward_risk": admission_rr,
        "stop_risk_pct": _float_or_nan(snapshot.get("risk_pct")),
        "risk_reward_gate_passed": bool(snapshot.get("phase2_rr_gate_passed")),
        "position_risk_gate_passed": bool(snapshot.get("phase2_risk_gate_passed")),
        "medium_term_quality_score": math.nan,
        "medium_term_slope_score": math.nan,
        "medium_term_persistence_score": math.nan,
        "medium_term_structure_score": math.nan,
        "medium_term_phase_alignment_score": math.nan,
        "medium_term_entry_location_score": math.nan,
        "medium_term_momentum_score": math.nan,
        "medium_term_efficiency_score": math.nan,
        "medium_term_trend_state_score": math.nan,
        "medium_term_freshness_score": math.nan,
        "medium_term_regime_score": math.nan,
        "medium_term_quality_status": "not_available",
    }


def _directional_return(direction: str, *, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0 or exit_price <= 0:
        return math.nan
    if direction == "short":
        return float(entry_price - exit_price) / float(entry_price)
    return float(exit_price - entry_price) / float(entry_price)


def _directional_favorable(direction: str, *, entry_price: float, high: float, low: float) -> float:
    if entry_price <= 0:
        return math.nan
    if direction == "short":
        return float(entry_price - low) / float(entry_price)
    return float(high - entry_price) / float(entry_price)


def _directional_adverse(direction: str, *, entry_price: float, high: float, low: float) -> float:
    if entry_price <= 0:
        return math.nan
    if direction == "short":
        return float(entry_price - high) / float(entry_price)
    return float(low - entry_price) / float(entry_price)


def _daily_bars_from_minutes(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close"])
    data = minute_df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["trade_date"] = data["datetime"].dt.date
    data = data.sort_values("datetime", kind="stable")
    grouped = (
        data.groupby("trade_date", sort=True, as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .sort_values("trade_date", kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def _forward_metrics_by_date(
    minute_df: pd.DataFrame,
    *,
    direction: str,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, float | int]]:
    daily = _daily_bars_from_minutes(minute_df)
    if daily.empty:
        return {}
    out: dict[str, dict[str, float | int]] = {}
    for idx, row in daily.iterrows():
        trade_date = str(row["trade_date"])
        entry_price = float(row["open"])
        metrics: dict[str, float | int] = {
            "entry_price": entry_price,
            "available_forward_days": int(len(daily) - idx - 1),
        }
        for horizon in horizons:
            end_idx = idx + horizon
            if end_idx >= len(daily):
                metrics[f"forward_{horizon}d_return"] = math.nan
                metrics[f"forward_{horizon}d_max_favorable_return"] = math.nan
                metrics[f"forward_{horizon}d_max_adverse_return"] = math.nan
                continue
            window = daily.iloc[idx : end_idx + 1]
            end_close = float(daily.iloc[end_idx]["close"])
            metrics[f"forward_{horizon}d_return"] = _directional_return(
                direction,
                entry_price=entry_price,
                exit_price=end_close,
            )
            metrics[f"forward_{horizon}d_max_favorable_return"] = _directional_favorable(
                direction,
                entry_price=entry_price,
                high=float(window["high"].max()),
                low=float(window["low"].min()),
            )
            metrics[f"forward_{horizon}d_max_adverse_return"] = _directional_adverse(
                direction,
                entry_price=entry_price,
                high=float(window["high"].max()),
                low=float(window["low"].min()),
            )
        out[trade_date] = metrics
    return out


def _float_or_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        out = float(value)
        return out
    except (TypeError, ValueError):
        return math.nan


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = [_float_or_nan(row.get(field)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _directional_price_return(close: pd.Series, *, direction: str, window: int) -> float:
    if len(close) < 2:
        return 0.0
    resolved_window = min(max(int(window), 1), len(close) - 1)
    anchor = float(close.iloc[-resolved_window - 1])
    last = float(close.iloc[-1])
    if anchor <= 0 or last <= 0:
        return 0.0
    if direction == "short":
        return float(anchor - last) / anchor
    return float(last - anchor) / anchor


def _series_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(max(variance, 0.0)))


def _pct_change_values(close: pd.Series) -> list[float]:
    values = [float(value) for value in close.dropna().tolist()]
    out: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous > 0 and current > 0:
            out.append((current - previous) / previous)
    return out


def _average_true_range_pct(daily_df: pd.DataFrame, *, window: int = 20) -> float:
    if daily_df.empty or "close" not in daily_df:
        return 0.01
    tail = daily_df.tail(max(int(window), 1)).copy()
    close = pd.to_numeric(tail["close"], errors="coerce")
    last = float(close.iloc[-1]) if len(close) else 0.0
    if last <= 0:
        return 0.01
    if {"high", "low"}.issubset(tail.columns):
        high = pd.to_numeric(tail["high"], errors="coerce")
        low = pd.to_numeric(tail["low"], errors="coerce")
        true_range = (high - low).abs().dropna()
    else:
        true_range = close.diff().abs().dropna()
    if true_range.empty:
        return 0.01
    return float(max(float(true_range.mean()) / last, 0.0025))


def _time_series_momentum_score(close: pd.Series, *, direction: str) -> float:
    scores: list[float] = []
    weights: list[float] = []
    for window, weight in ((20, 0.30), (60, 0.40), (120, 0.30)):
        if len(close) < window + 1:
            continue
        directional_return = _directional_price_return(close, direction=direction, window=window)
        returns = _pct_change_values(close.tail(window + 1))
        vol = _series_std(returns) * math.sqrt(window)
        if vol <= 1e-8:
            raw = 1.0 if directional_return > 0 else 0.0
        else:
            raw = _clip01(0.5 + directional_return / (2.0 * vol))
        scores.append(raw)
        weights.append(weight)
    if not scores:
        return 0.0
    weight_sum = sum(weights)
    return float(sum(score * weight for score, weight in zip(scores, weights)) / weight_sum)


def _kaufman_efficiency_score(close: pd.Series, *, direction: str) -> float:
    scores: list[float] = []
    weights: list[float] = []
    for window, weight in ((20, 0.30), (60, 0.40), (120, 0.30)):
        if len(close) < window + 1:
            continue
        window_close = close.tail(window + 1).astype(float)
        first = float(window_close.iloc[0])
        last = float(window_close.iloc[-1])
        if direction == "short":
            net_move = first - last
        else:
            net_move = last - first
        path = float(window_close.diff().abs().sum())
        if path <= 0 or net_move <= 0:
            score = 0.0
        else:
            score = _clip01(net_move / path)
        scores.append(score)
        weights.append(weight)
    if not scores:
        return 0.0
    weight_sum = sum(weights)
    return float(sum(score * weight for score, weight in zip(scores, weights)) / weight_sum)


def _adx_dmi_score(daily_df: pd.DataFrame, *, direction: str, window: int = 14) -> float:
    if len(daily_df) < window + 2 or not {"high", "low", "close"}.issubset(daily_df.columns):
        return 0.0
    data = daily_df.tail(max(window * 3, window + 2)).copy()
    high = pd.to_numeric(data["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(data["low"], errors="coerce").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce").reset_index(drop=True)
    tr_values: list[float] = []
    plus_dm_values: list[float] = []
    minus_dm_values: list[float] = []
    for idx in range(1, len(data)):
        high_diff = float(high.iloc[idx] - high.iloc[idx - 1])
        low_diff = float(low.iloc[idx - 1] - low.iloc[idx])
        plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0.0
        true_range = max(
            float(high.iloc[idx] - low.iloc[idx]),
            abs(float(high.iloc[idx] - close.iloc[idx - 1])),
            abs(float(low.iloc[idx] - close.iloc[idx - 1])),
        )
        tr_values.append(true_range)
        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)
    if len(tr_values) < window:
        return 0.0
    dx_values: list[float] = []
    for end in range(window, len(tr_values) + 1):
        tr_sum = sum(tr_values[end - window : end])
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * sum(plus_dm_values[end - window : end]) / tr_sum
        minus_di = 100.0 * sum(minus_dm_values[end - window : end]) / tr_sum
        denom = plus_di + minus_di
        if denom <= 0:
            continue
        dx_values.append(100.0 * abs(plus_di - minus_di) / denom)
    if not dx_values:
        return 0.0
    last_end = len(tr_values)
    tr_sum = sum(tr_values[last_end - window : last_end])
    if tr_sum <= 0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm_values[last_end - window : last_end]) / tr_sum
    minus_di = 100.0 * sum(minus_dm_values[last_end - window : last_end]) / tr_sum
    direction_ok = plus_di > minus_di if direction == "long" else minus_di > plus_di
    if not direction_ok:
        return 0.0
    adx = sum(dx_values[-window:]) / min(len(dx_values), window)
    return _clip01(adx / 35.0)


def _aroon_freshness_score(daily_df: pd.DataFrame, *, direction: str) -> float:
    if not {"high", "low"}.issubset(daily_df.columns):
        return 0.0
    scores: list[float] = []
    weights: list[float] = []
    for window, weight in ((25, 0.45), (60, 0.55)):
        if len(daily_df) < window:
            continue
        tail = daily_df.tail(window)
        high_values = pd.to_numeric(tail["high"], errors="coerce").tolist()
        low_values = pd.to_numeric(tail["low"], errors="coerce").tolist()
        if not high_values or not low_values:
            continue
        high_idx = max(range(len(high_values)), key=lambda idx: high_values[idx])
        low_idx = min(range(len(low_values)), key=lambda idx: low_values[idx])
        periods_since_high = len(high_values) - 1 - high_idx
        periods_since_low = len(low_values) - 1 - low_idx
        aroon_up = 100.0 * (window - periods_since_high) / window
        aroon_down = 100.0 * (window - periods_since_low) / window
        spread = aroon_down - aroon_up if direction == "short" else aroon_up - aroon_down
        scores.append(_clip01((spread + 100.0) / 200.0))
        weights.append(weight)
    if not scores:
        return 0.0
    weight_sum = sum(weights)
    return float(sum(score * weight for score, weight in zip(scores, weights)) / weight_sum)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_avg = sum(xs) / len(xs)
    y_avg = sum(ys) / len(ys)
    denom = sum((x - x_avg) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return float(sum((x - x_avg) * (y - y_avg) for x, y in zip(xs, ys)) / denom)


def _hurst_regime_score(close: pd.Series) -> float:
    clean = [float(value) for value in close.dropna().tolist()]
    if len(clean) < 40:
        return 0.5
    lengths = [length for length in (20, 40, 80, 120) if len(clean) >= length]
    xs: list[float] = []
    ys: list[float] = []
    for length in lengths:
        segment = clean[-length:]
        avg = sum(segment) / len(segment)
        cumulative: list[float] = []
        running = 0.0
        for value in segment:
            running += value - avg
            cumulative.append(running)
        range_value = max(cumulative) - min(cumulative)
        scale = _series_std(segment)
        if range_value <= 0 or scale <= 0:
            continue
        xs.append(math.log(float(length)))
        ys.append(math.log(range_value / scale))
    if len(xs) < 2:
        return 0.5
    hurst = _linear_slope(xs, ys)
    return _clip01((hurst - 0.45) / 0.30)


def _trend_persistence(close: pd.Series, *, direction: str, window: int = 60) -> float:
    if len(close) < 3:
        return 0.0
    resolved_window = min(max(int(window), 2), len(close) - 1)
    window_close = close.tail(resolved_window + 1).astype(float)
    anchor = float(window_close.iloc[0])
    last = float(window_close.iloc[-1])
    if direction == "short":
        net_move = anchor - last
    else:
        net_move = last - anchor
    path = float(window_close.diff().abs().sum())
    if path <= 0:
        return 0.0
    return _clip01(net_move / path)


def _moving_average_structure(close: pd.Series, *, direction: str) -> float:
    if len(close) < 2:
        return 0.0
    checks: list[bool] = []
    last = float(close.iloc[-1])
    windows = [20, 60, 120]
    ma_values: dict[int, float] = {}
    ma_slopes: dict[int, float] = {}
    for window in windows:
        if len(close) < window:
            continue
        ma = close.rolling(window).mean().dropna()
        if ma.empty:
            continue
        ma_values[window] = float(ma.iloc[-1])
        if len(ma) >= 5:
            ma_slopes[window] = float(ma.iloc[-1] - ma.iloc[-5])
    if 20 in ma_values:
        checks.append(last < ma_values[20] if direction == "short" else last > ma_values[20])
    if 20 in ma_values and 60 in ma_values:
        checks.append(ma_values[20] < ma_values[60] if direction == "short" else ma_values[20] > ma_values[60])
    if 60 in ma_values and 120 in ma_values:
        checks.append(ma_values[60] < ma_values[120] if direction == "short" else ma_values[60] > ma_values[120])
    for slope in ma_slopes.values():
        checks.append(slope < 0 if direction == "short" else slope > 0)
    if not checks:
        return 0.0
    return float(sum(1 for item in checks if item) / len(checks))


def _entry_location_score(close: pd.Series, *, direction: str, atr_pct: float, window: int = 20) -> float:
    if len(close) < window:
        return 0.5
    last = float(close.iloc[-1])
    ma = float(close.rolling(window).mean().iloc[-1])
    atr_abs = max(last * max(float(atr_pct), 0.0025), 1e-12)
    if direction == "short":
        extension_atr = (ma - last) / atr_abs
    else:
        extension_atr = (last - ma) / atr_abs
    if extension_atr < -0.5:
        return 0.0
    if extension_atr <= 2.0:
        return 1.0
    if extension_atr <= 5.0:
        return _clip01(1.0 - (extension_atr - 2.0) / 3.0)
    return 0.0


def _trend_phase_alignment(*, direction: str, trend_phase: str) -> float:
    phase = str(trend_phase or "").strip().lower()
    if direction == "short":
        if phase == "markdown":
            return 1.0
        if phase == "distribution":
            return 0.5
        return 0.0
    if phase == "markup":
        return 1.0
    if phase == "accumulation":
        return 0.5
    return 0.0


def _medium_term_quality_from_daily(
    daily_df: pd.DataFrame,
    *,
    direction: str,
    trend_phase: str,
) -> dict[str, Any]:
    return _shared_medium_term_quality_from_daily(
        daily_df,
        direction=direction,
        trend_phase=trend_phase,
    )


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return math.nan
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_dev = [value - left_mean for value in left]
    right_dev = [value - right_mean for value in right]
    denom_left = math.sqrt(sum(value * value for value in left_dev))
    denom_right = math.sqrt(sum(value * value for value in right_dev))
    if denom_left <= 0 or denom_right <= 0:
        return math.nan
    return float(sum(a * b for a, b in zip(left_dev, right_dev)) / (denom_left * denom_right))


def _spearman(left: list[float], right: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(left, right) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return math.nan
    ranked_left = _rankdata([pair[0] for pair in pairs])
    ranked_right = _rankdata([pair[1] for pair in pairs])
    return _pearson(ranked_left, ranked_right)


def _pool_rows(rows: list[dict[str, Any]], pool: str) -> list[dict[str, Any]]:
    if pool == "signal":
        return [
            row
            for row in rows
            if abs(float(row.get("score") or 0.0)) > 0.0
            or bool(row.get("trend_has_signal"))
            or bool(row.get("reversal_has_signal"))
        ]
    if pool == "score_gate":
        return [row for row in rows if bool(row.get("phase2_score_gate_passed"))]
    if pool == "actionable":
        return [row for row in rows if row.get("phase2_state") == "actionable"]
    raise ValueError(f"unknown pool: {pool}")


def summarize_buckets(
    rows: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    pools: tuple[str, ...] = ("signal", "score_gate", "actionable"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bucket_rows: list[dict[str, Any]] = []
    monotonic_rows: list[dict[str, Any]] = []
    for pool in pools:
        pool_rows = _pool_rows(rows, pool)
        for horizon in horizons:
            return_field = f"forward_{horizon}d_return"
            favorable_field = f"forward_{horizon}d_max_favorable_return"
            adverse_field = f"forward_{horizon}d_max_adverse_return"
            eligible = [row for row in pool_rows if math.isfinite(_float_or_nan(row.get(return_field)))]
            by_bucket: dict[str, list[dict[str, Any]]] = {}
            for row in eligible:
                by_bucket.setdefault(str(row["score_bucket"]), []).append(row)
            bucket_means: list[tuple[str, float]] = []
            for bucket in sorted(by_bucket):
                group = by_bucket[bucket]
                returns = _finite_values(group, return_field)
                favorable = _finite_values(group, favorable_field)
                adverse = _finite_values(group, adverse_field)
                abs_scores = _finite_values(group, "abs_score")
                actionable = sum(1 for row in group if row.get("phase2_state") == "actionable")
                mean_return = _mean(returns)
                bucket_means.append((bucket, mean_return))
                bucket_rows.append(
                    {
                        "pool": pool,
                        "horizon_days": horizon,
                        "score_bucket": bucket,
                        "rows": len(group),
                        "avg_abs_score": _mean(abs_scores),
                        "actionable_rows": actionable,
                        "mean_forward_return": mean_return,
                        "median_forward_return": float(median(returns)) if returns else math.nan,
                        "win_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else math.nan,
                        "mean_max_favorable_return": _mean(favorable),
                        "mean_max_adverse_return": _mean(adverse),
                    }
                )
            score_values = _finite_values(eligible, "abs_score")
            return_values = _finite_values(eligible, return_field)
            adjacent_pairs = max(len(bucket_means) - 1, 0)
            adjacent_up_steps = 0
            for (_, previous), (_, current) in zip(bucket_means, bucket_means[1:]):
                if math.isfinite(previous) and math.isfinite(current) and current >= previous:
                    adjacent_up_steps += 1
            monotonic_rows.append(
                {
                    "pool": pool,
                    "horizon_days": horizon,
                    "rows": len(eligible),
                    "buckets": ",".join(bucket for bucket, _ in bucket_means),
                    "spearman_abs_score_forward_return": _spearman(score_values, return_values),
                    "low_bucket_mean_forward_return": bucket_means[0][1] if bucket_means else math.nan,
                    "high_bucket_mean_forward_return": bucket_means[-1][1] if bucket_means else math.nan,
                    "high_minus_low_forward_return": (
                        bucket_means[-1][1] - bucket_means[0][1] if len(bucket_means) >= 2 else math.nan
                    ),
                    "adjacent_up_steps": adjacent_up_steps,
                    "adjacent_pairs": adjacent_pairs,
                    "monotonic_bucket_pass": bool(adjacent_pairs > 0 and adjacent_up_steps == adjacent_pairs),
                }
            )
    return bucket_rows, monotonic_rows


def _component_bucket_summary(rows: list[dict[str, Any]], component_field: str, return_field: str) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if math.isfinite(_float_or_nan(row.get(component_field)))
        and math.isfinite(_float_or_nan(row.get(return_field)))
    ]
    if len(eligible) < 3:
        return {
            "rows": len(eligible),
            "low_component_mean": math.nan,
            "middle_component_mean": math.nan,
            "high_component_mean": math.nan,
            "low_bucket_mean_forward_return": math.nan,
            "middle_bucket_mean_forward_return": math.nan,
            "high_bucket_mean_forward_return": math.nan,
            "high_minus_low_forward_return": math.nan,
            "adjacent_up_steps": 0,
            "adjacent_pairs": 0,
            "monotonic_bucket_pass": False,
        }
    ordered = sorted(eligible, key=lambda row: _float_or_nan(row.get(component_field)))
    split = max(len(ordered) // 3, 1)
    low = ordered[:split]
    high = ordered[-split:]
    middle = ordered[split:-split] or ordered[split : split + split]

    def component_mean(group: list[dict[str, Any]]) -> float:
        return _mean(_finite_values(group, component_field))

    def return_mean(group: list[dict[str, Any]]) -> float:
        return _mean(_finite_values(group, return_field))

    low_return = return_mean(low)
    middle_return = return_mean(middle)
    high_return = return_mean(high)
    bucket_returns = [low_return, middle_return, high_return]
    adjacent_up_steps = 0
    adjacent_pairs = 2
    for previous, current in zip(bucket_returns, bucket_returns[1:]):
        if math.isfinite(previous) and math.isfinite(current) and current >= previous:
            adjacent_up_steps += 1
    return {
        "rows": len(eligible),
        "low_component_mean": component_mean(low),
        "middle_component_mean": component_mean(middle),
        "high_component_mean": component_mean(high),
        "low_bucket_mean_forward_return": low_return,
        "middle_bucket_mean_forward_return": middle_return,
        "high_bucket_mean_forward_return": high_return,
        "high_minus_low_forward_return": high_return - low_return
        if math.isfinite(high_return) and math.isfinite(low_return)
        else math.nan,
        "adjacent_up_steps": adjacent_up_steps,
        "adjacent_pairs": adjacent_pairs,
        "monotonic_bucket_pass": adjacent_up_steps == adjacent_pairs,
    }


def summarize_components(
    rows: list[dict[str, Any]],
    *,
    horizons: tuple[int, ...],
    pools: tuple[str, ...] = ("signal", "score_gate", "actionable"),
) -> list[dict[str, Any]]:
    component_rows: list[dict[str, Any]] = []
    for pool in pools:
        pool_rows = _pool_rows(rows, pool)
        for horizon in horizons:
            return_field = f"forward_{horizon}d_return"
            for spec in COMPONENT_SPECS:
                component_field = str(spec["field"])
                eligible = [
                    row
                    for row in pool_rows
                    if math.isfinite(_float_or_nan(row.get(component_field)))
                    and math.isfinite(_float_or_nan(row.get(return_field)))
                ]
                component_values = _finite_values(eligible, component_field)
                return_values = _finite_values(eligible, return_field)
                bucket_summary = _component_bucket_summary(eligible, component_field, return_field)
                component_rows.append(
                    {
                        "pool": pool,
                        "component": component_field,
                        "component_meaning": spec["meaning"],
                        "component_role": spec["role"],
                        "horizon_days": horizon,
                        "spearman_component_forward_return": _spearman(component_values, return_values),
                        **bucket_summary,
                    }
                )
    return component_rows


def _collect_case_phase2_rows(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    pre_market_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if minute_df.empty:
        return []
    run_case = case
    plan_factory = _plan_factory_for_case(run_case)
    minute_data = minute_df.copy()
    minute_data["datetime"] = pd.to_datetime(minute_data["datetime"])
    minute_data["trade_date"] = minute_data["datetime"].dt.date
    minute_data = minute_data.loc[minute_data["trade_date"].between(run_case.start_dt, run_case.end_dt)].copy()
    if minute_data.empty:
        return []

    visible_daily = daily_df.copy()
    visible_daily["date"] = pd.to_datetime(visible_daily["date"])
    visible_daily = visible_daily.sort_values("date", kind="stable")
    visible_daily["trade_date"] = visible_daily["date"].dt.date
    min_history_bars = _phase2_min_history_bars(pre_market_cfg)
    rows: list[dict[str, Any]] = []
    for trade_date in sorted(minute_data["trade_date"].dropna().unique()):
        day_visible_daily = visible_daily.loc[visible_daily["trade_date"] < trade_date].copy()
        history_insufficient = len(day_visible_daily) < min_history_bars
        day_plan, rejection_counts, debug_snapshot = _evaluate_phase2_plan(
            case=run_case,
            daily_df=day_visible_daily,
            pre_market_cfg=pre_market_cfg,
            plan_factory=plan_factory,
            capture_debug=True,
        )
        debug_day = _build_phase2_debug_day(
            trade_date=trade_date,
            visible_daily_rows=len(day_visible_daily),
            history_insufficient=history_insufficient,
            duplicate_rejected=False,
            plan=day_plan,
            rejection_counts=rejection_counts,
            snapshot=debug_snapshot,
        )
        score = float(debug_day.get("score") or 0.0)
        row: dict[str, Any] = {
            "case_id": run_case.case_id,
            "symbol": run_case.symbol,
            "name": run_case.name,
            "direction": run_case.direction,
            "trade_date": str(trade_date),
            "phase2_state": str(debug_day.get("phase2_state") or ""),
            "score": score,
            "abs_score": abs(score),
            "score_bucket": _score_bucket(abs(score)),
            "rr": float(debug_day.get("rr") or 0.0),
            "phase2_score_gate_passed": bool(debug_day.get("phase2_score_gate_passed")),
            "phase2_rr_gate_passed": bool(debug_day.get("phase2_rr_gate_passed")),
            "entry_family": str(debug_day.get("entry_family") or ""),
            "entry_signal_type": str(debug_day.get("entry_signal_type") or ""),
            "entry_signal_detail": str(debug_day.get("entry_signal_detail") or ""),
            "reversal_has_signal": bool(debug_day.get("reversal_has_signal")),
            "trend_has_signal": bool(debug_day.get("trend_has_signal")),
            "trend_phase": str(debug_day.get("trend_phase") or ""),
            "trend_phase_ok": bool(debug_day.get("trend_phase_ok")),
            "trend_slope_ok": bool(debug_day.get("trend_slope_ok")),
            "trend_indicator_ok": bool(debug_day.get("trend_indicator_ok")),
            "backtest_fundamental_status": str(debug_day.get("backtest_fundamental_status") or ""),
        }
        row.update(_component_fields_from_snapshot(debug_snapshot, direction=run_case.direction))
        row.update(
            _medium_term_quality_from_daily(
                day_visible_daily,
                direction=run_case.direction,
                trend_phase=str(debug_day.get("trend_phase") or ""),
            )
        )
        rows.append(row)
    return rows


def _attach_forward_metrics(
    rows: list[dict[str, Any]],
    *,
    minute_df: pd.DataFrame,
    direction: str,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    forward_by_date = _forward_metrics_by_date(minute_df, direction=direction, horizons=horizons)
    out: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged.update(forward_by_date.get(str(row.get("trade_date")), {}))
        out.append(merged)
    return out


def collect_validation_rows(
    *,
    case_group: str,
    year: int,
    config: dict[str, Any],
    horizons: tuple[int, ...],
    case_limit: int | None = None,
    phase2_score_cache_dir: Path | None = None,
    write_missing_phase2_score_cache: bool = False,
    require_phase2_score_cache: bool = False,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cases = get_case_group(case_group)
    if case_limit is not None:
        cases = cases[: max(int(case_limit), 0)]
    rows: list[dict[str, Any]] = []
    cache_diagnostics = {
        "phase2_score_cache_hits": 0,
        "phase2_score_cache_misses": 0,
    }
    pre_market_cfg = config.get("pre_market") or {}
    for case in cases:
        run_case = _case_for_year(case, year)
        if progress:
            print(f"case {run_case.case_id}: start", flush=True)
        daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
        base_rows: list[dict[str, Any]] | None = None
        cache_path: Path | None = None
        if phase2_score_cache_dir is not None:
            cache_path = _phase2_score_cache_path(
                phase2_score_cache_dir,
                case=run_case,
                pre_market_cfg=pre_market_cfg,
            )
            base_rows = _read_phase2_score_cache(
                cache_path,
                case=run_case,
                pre_market_cfg=pre_market_cfg,
            )
            if base_rows is not None:
                cache_diagnostics["phase2_score_cache_hits"] += 1
            elif require_phase2_score_cache:
                raise FileNotFoundError(f"missing Phase2 score validation cache: {cache_path}")
        if base_rows is None:
            cache_diagnostics["phase2_score_cache_misses"] += 1
            base_rows = _collect_case_phase2_rows(
                case=run_case,
                daily_df=daily_df,
                minute_df=minute_df,
                pre_market_cfg=pre_market_cfg,
            )
            if cache_path is not None and write_missing_phase2_score_cache:
                _write_phase2_score_cache(
                    cache_path,
                    case=run_case,
                    pre_market_cfg=pre_market_cfg,
                    rows=base_rows,
                )
        case_rows = _attach_forward_metrics(
            base_rows,
            minute_df=minute_df,
            direction=run_case.direction,
            horizons=horizons,
        )
        rows.extend(case_rows)
        if progress:
            signal_rows = len(_pool_rows(case_rows, "signal"))
            actionable_rows = len(_pool_rows(case_rows, "actionable"))
            print(
                f"case {run_case.case_id}: rows={len(case_rows)} "
                f"signals={signal_rows} actionable={actionable_rows}",
                flush=True,
            )
    return rows, cache_diagnostics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _format_pct(value: Any) -> str:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


def _write_markdown_report(
    path: Path,
    *,
    case_group: str,
    year: int,
    horizons: tuple[int, ...],
    detail_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    monotonic_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Phase2 Score Forward Return Validation ({case_group} {year})",
        "",
        "This report validates whether stronger Phase2 scores line up with stronger later directional returns.",
        "It is a backtest-only audit and does not change live or default strategy behavior.",
        "",
        "## Scope",
        "",
        f"- Cases: {case_group}",
        f"- Year: {year}",
        f"- Horizons: {', '.join(str(item) for item in horizons)} trading days",
        f"- Daily rows evaluated: {len(detail_rows)}",
        f"- Signal rows: {len(_pool_rows(detail_rows, 'signal'))}",
        f"- Score-gate rows: {len(_pool_rows(detail_rows, 'score_gate'))}",
        f"- Actionable rows: {len(_pool_rows(detail_rows, 'actionable'))}",
        "",
        "## Monotonicity",
        "",
        "| Pool | Horizon | Rows | Spearman | High - Low | Adjacent Steps | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in monotonic_rows:
        lines.append(
            "| {pool} | {horizon} | {rows} | {spearman} | {spread} | {steps}/{pairs} | {passed} |".format(
                pool=row["pool"],
                horizon=row["horizon_days"],
                rows=row["rows"],
                spearman=(
                    "n/a"
                    if row["spearman_abs_score_forward_return"] is None
                    else f"{row['spearman_abs_score_forward_return']:.3f}"
                ),
                spread=_format_pct(row["high_minus_low_forward_return"]),
                steps=row["adjacent_up_steps"],
                pairs=row["adjacent_pairs"],
                passed="yes" if row["monotonic_bucket_pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Bucket Means",
            "",
            "| Pool | Horizon | Bucket | Rows | Avg Score | Mean Return | Win Rate | MFE | MAE |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bucket_rows:
        lines.append(
            "| {pool} | {horizon} | {bucket} | {rows} | {score:.2f} | {mean} | {win} | {mfe} | {mae} |".format(
                pool=row["pool"],
                horizon=row["horizon_days"],
                bucket=row["score_bucket"],
                rows=row["rows"],
                score=float(row["avg_abs_score"]) if math.isfinite(float(row["avg_abs_score"])) else math.nan,
                mean=_format_pct(row["mean_forward_return"]),
                win=_format_pct(row["win_rate"]),
                mfe=_format_pct(row["mean_max_favorable_return"]),
                mae=_format_pct(row["mean_max_adverse_return"]),
            )
        )
    lines.extend(
        [
            "",
            "## Component Attribution",
            "",
            "The current Phase2 trend score is decomposed into short-term price-volume strength, trend structure, trend phase, risk/reward, and a candidate medium-term opportunity quality score.",
            "",
            "| Pool | Component | Horizon | Rows | Spearman | High - Low | Adjacent Steps | Pass |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in component_rows:
        lines.append(
            "| {pool} | {component} | {horizon} | {rows} | {spearman} | {spread} | {steps}/{pairs} | {passed} |".format(
                pool=row["pool"],
                component=row["component_meaning"],
                horizon=row["horizon_days"],
                rows=row["rows"],
                spearman=(
                    "n/a"
                    if row["spearman_component_forward_return"] is None
                    else f"{row['spearman_component_forward_return']:.3f}"
                ),
                spread=_format_pct(row["high_minus_low_forward_return"]),
                steps=row["adjacent_up_steps"],
                pairs=row["adjacent_pairs"],
                passed="yes" if row["monotonic_bucket_pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Medium-Term Quality Gap",
            "",
            "This report now adds a candidate v3 medium-term opportunity quality score based only on daily history visible before the trade date. It combines time-series momentum, Kaufman efficiency, ADX/DMI trend state, Aroon freshness, a light Hurst regime check, and an overextension penalty. It is for backtest validation only and is not used by default trading or live sizing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_prefix: Path,
    detail_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    monotonic_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    cache_diagnostics: dict[str, int],
    case_group: str,
    year: int,
    horizons: tuple[int, ...],
) -> list[Path]:
    detail_fields = list(DETAIL_BASE_FIELDS)
    for horizon in horizons:
        detail_fields.extend(
            [
                f"forward_{horizon}d_return",
                f"forward_{horizon}d_max_favorable_return",
                f"forward_{horizon}d_max_adverse_return",
            ]
        )
    detail_path = output_prefix.with_name(f"{output_prefix.name}_details.csv")
    bucket_path = output_prefix.with_name(f"{output_prefix.name}_buckets.csv")
    monotonic_path = output_prefix.with_name(f"{output_prefix.name}_monotonicity.csv")
    component_path = output_prefix.with_name(f"{output_prefix.name}_components.csv")
    json_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    md_path = output_prefix.with_name(f"{output_prefix.name}.md")
    _write_csv(detail_path, detail_rows, detail_fields)
    _write_csv(bucket_path, bucket_rows, BUCKET_SUMMARY_FIELDS)
    _write_csv(monotonic_path, monotonic_rows, MONOTONICITY_FIELDS)
    _write_csv(component_path, component_rows, COMPONENT_SUMMARY_FIELDS)
    _write_json(
        json_path,
        {
            "case_group": case_group,
            "year": year,
            "horizons": list(horizons),
            "daily_rows": len(detail_rows),
            "signal_rows": len(_pool_rows(detail_rows, "signal")),
            "score_gate_rows": len(_pool_rows(detail_rows, "score_gate")),
            "actionable_rows": len(_pool_rows(detail_rows, "actionable")),
            **cache_diagnostics,
            "medium_term_quality_status": "candidate_v3",
            "bucket_summary": bucket_rows,
            "monotonicity": monotonic_rows,
            "component_summary": component_rows,
        },
    )
    _write_markdown_report(
        md_path,
        case_group=case_group,
        year=year,
        horizons=horizons,
        detail_rows=detail_rows,
        bucket_rows=bucket_rows,
        monotonic_rows=monotonic_rows,
        component_rows=component_rows,
    )
    return [detail_path, bucket_path, monotonic_path, component_path, json_path, md_path]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    horizons = _parse_horizons(args.horizons)
    loaded_config = load_config()
    config = _config_with_fundamental_mode(
        loaded_config,
        _resolved_fundamental_mode(loaded_config, args.fundamental_mode),
    )
    detail_rows, cache_diagnostics = collect_validation_rows(
        case_group=args.case_group,
        year=args.year,
        config=config,
        horizons=horizons,
        case_limit=args.case_limit,
        phase2_score_cache_dir=args.phase2_score_cache_dir,
        write_missing_phase2_score_cache=args.write_missing_phase2_score_cache,
        require_phase2_score_cache=args.require_phase2_score_cache,
        progress=not args.quiet,
    )
    bucket_rows, monotonic_rows = summarize_buckets(detail_rows, horizons=horizons)
    component_rows = summarize_components(detail_rows, horizons=horizons)
    output_prefix = args.output_prefix or _default_output_prefix(case_group=args.case_group, year=args.year)
    for path in write_outputs(
        output_prefix=output_prefix,
        detail_rows=detail_rows,
        bucket_rows=bucket_rows,
        monotonic_rows=monotonic_rows,
        component_rows=component_rows,
        cache_diagnostics=cache_diagnostics,
        case_group=args.case_group,
        year=args.year,
        horizons=horizons,
    ):
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
