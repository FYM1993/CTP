from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    if frame is None or frame.empty or field not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna().astype(float)


def _clean_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    data = daily_df.copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date", kind="stable")
    return data.reset_index(drop=True)


def _average_true_range_abs(daily_df: pd.DataFrame, *, window: int = 20) -> float:
    if daily_df.empty or "close" not in daily_df:
        return 0.0
    tail = daily_df.tail(max(int(window), 1)).copy()
    close = _numeric(tail, "close")
    if close.empty:
        return 0.0
    if {"high", "low"}.issubset(tail.columns):
        high = pd.to_numeric(tail["high"], errors="coerce").reset_index(drop=True)
        low = pd.to_numeric(tail["low"], errors="coerce").reset_index(drop=True)
        prev_close = pd.to_numeric(tail["close"], errors="coerce").shift(1).reset_index(drop=True)
        ranges = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        value = float(ranges.dropna().mean()) if not ranges.dropna().empty else 0.0
    else:
        diff = close.diff().abs().dropna()
        value = float(diff.mean()) if not diff.empty else 0.0
    if value <= 0 and not close.empty:
        value = max(float(close.iloc[-1]) * 0.005, 1e-9)
    return float(value)


def _directional_return(close: pd.Series, *, direction: str, window: int) -> float:
    if len(close) < 2:
        return 0.0
    resolved = min(max(int(window), 1), len(close) - 1)
    first = float(close.iloc[-resolved - 1])
    last = float(close.iloc[-1])
    if first <= 0:
        return 0.0
    return float((first - last) / first) if direction == "short" else float((last - first) / first)


def _path_efficiency(close: pd.Series, *, direction: str, window: int) -> float:
    if len(close) < 3:
        return 0.0
    resolved = min(max(int(window), 2), len(close) - 1)
    segment = close.tail(resolved + 1).astype(float)
    first = float(segment.iloc[0])
    last = float(segment.iloc[-1])
    net = first - last if direction == "short" else last - first
    path = float(segment.diff().abs().sum())
    if path <= 0 or net <= 0:
        return 0.0
    return _clip01(net / path)


def _ma_value(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return math.nan
    return float(close.rolling(window).mean().iloc[-1])


def _ma_slope(close: pd.Series, window: int, lookback: int = 5) -> float:
    if len(close) < window + lookback:
        return 0.0
    ma = close.rolling(window).mean().dropna()
    if len(ma) <= lookback:
        return 0.0
    return float(ma.iloc[-1] - ma.iloc[-lookback - 1])


def _ma_alignment_score(close: pd.Series, *, direction: str) -> float:
    if close.empty:
        return 0.0
    last = float(close.iloc[-1])
    ma20 = _ma_value(close, 20)
    ma60 = _ma_value(close, 60)
    ma120 = _ma_value(close, 120)
    checks: list[bool] = []
    if math.isfinite(ma20):
        checks.append(last < ma20 if direction == "short" else last > ma20)
    if math.isfinite(ma20) and math.isfinite(ma60):
        checks.append(ma20 < ma60 if direction == "short" else ma20 > ma60)
    if math.isfinite(ma60) and math.isfinite(ma120):
        checks.append(ma60 < ma120 if direction == "short" else ma60 > ma120)
    for window in (20, 60):
        slope = _ma_slope(close, window)
        checks.append(slope < 0 if direction == "short" else slope > 0)
    if not checks:
        return 0.0
    return float(sum(1 for item in checks if item) / len(checks))


def _direction_stability_score(close: pd.Series, *, direction: str, atr_abs: float) -> float:
    if close.empty:
        return 0.0
    last = max(float(close.iloc[-1]), 1e-9)
    atr_pct = max(float(atr_abs) / last, 0.0025)
    window_scores: list[float] = []
    weights: list[float] = []
    for window, weight in ((20, 0.25), (60, 0.45), (120, 0.30)):
        if len(close) < window + 1:
            continue
        directional = _directional_return(close, direction=direction, window=window)
        raw = _clip01(0.50 + directional / max(atr_pct * math.sqrt(window) * 3.0, 0.03))
        window_scores.append(raw)
        weights.append(weight)
    if not window_scores:
        return 0.0
    weighted = sum(score * weight for score, weight in zip(window_scores, weights)) / sum(weights)
    persistence = _path_efficiency(close, direction=direction, window=min(60, len(close) - 1))
    ma_alignment = _ma_alignment_score(close, direction=direction)
    return float(100.0 * (0.45 * weighted + 0.25 * persistence + 0.30 * ma_alignment))


def _extension_atr(close: pd.Series, *, direction: str, atr_abs: float) -> float:
    ma20 = _ma_value(close, 20)
    if not math.isfinite(ma20) or close.empty:
        return 0.0
    last = float(close.iloc[-1])
    atr = max(float(atr_abs), 1e-9)
    return float((ma20 - last) / atr) if direction == "short" else float((last - ma20) / atr)


def _extension_score(extension_atr: float) -> float:
    if extension_atr < -1.0:
        return 0.0
    if extension_atr <= 2.5:
        return 1.0
    if extension_atr <= 5.0:
        return _clip01(1.0 - (extension_atr - 2.5) / 2.5)
    return 0.0


def _structure_break_score(daily_df: pd.DataFrame, close: pd.Series, *, direction: str, atr_abs: float) -> float:
    if close.empty:
        return 0.0
    last = float(close.iloc[-1])
    ma20 = _ma_value(close, 20)
    ma60 = _ma_value(close, 60)
    atr = max(float(atr_abs), 1e-9)
    checks: list[bool] = []
    if math.isfinite(ma20):
        checks.append(last >= ma20 - 0.50 * atr if direction == "long" else last <= ma20 + 0.50 * atr)
    if math.isfinite(ma60):
        checks.append(last >= ma60 - 0.75 * atr if direction == "long" else last <= ma60 + 0.75 * atr)
    if {"high", "low"}.issubset(daily_df.columns) and len(daily_df) >= 40:
        tail = daily_df.tail(40)
        recent = daily_df.tail(10)
        if direction == "short":
            prior_high = float(pd.to_numeric(tail.iloc[:-10]["high"], errors="coerce").max())
            recent_high = float(pd.to_numeric(recent["high"], errors="coerce").max())
            checks.append(recent_high <= prior_high + 1.0 * atr)
        else:
            prior_low = float(pd.to_numeric(tail.iloc[:-10]["low"], errors="coerce").min())
            recent_low = float(pd.to_numeric(recent["low"], errors="coerce").min())
            checks.append(recent_low >= prior_low - 1.0 * atr)
    if not checks:
        return 0.5
    return float(sum(1 for item in checks if item) / len(checks))


def _pullback_order_score(daily_df: pd.DataFrame, close: pd.Series, *, direction: str, atr_abs: float) -> float:
    if len(close) < 20:
        return 0.5
    atr = max(float(atr_abs), 1e-9)
    tail = daily_df.tail(20)
    last = float(close.iloc[-1])
    if direction == "short":
        recent_extreme = float(pd.to_numeric(tail["low"], errors="coerce").min()) if "low" in tail else float(close.tail(20).min())
        retrace_atr = max(last - recent_extreme, 0.0) / atr
    else:
        recent_extreme = float(pd.to_numeric(tail["high"], errors="coerce").max()) if "high" in tail else float(close.tail(20).max())
        retrace_atr = max(recent_extreme - last, 0.0) / atr
    if retrace_atr <= 3.0:
        return 1.0
    if retrace_atr <= 7.0:
        return _clip01(1.0 - (retrace_atr - 3.0) / 4.0)
    return 0.0


def _structure_health_score(daily_df: pd.DataFrame, close: pd.Series, *, direction: str, atr_abs: float) -> tuple[float, str]:
    extension = _extension_atr(close, direction=direction, atr_abs=atr_abs)
    extension_component = _extension_score(extension)
    break_component = _structure_break_score(daily_df, close, direction=direction, atr_abs=atr_abs)
    pullback_component = _pullback_order_score(daily_df, close, direction=direction, atr_abs=atr_abs)
    ma_component = _ma_alignment_score(close, direction=direction)
    score = 100.0 * (
        0.35 * break_component
        + 0.30 * extension_component
        + 0.20 * pullback_component
        + 0.15 * ma_component
    )
    if break_component <= 0.35:
        return float(min(score, 35.0)), "damaged"
    if extension > 5.0:
        return float(min(score, 60.0)), "overextended"
    if score >= 65.0:
        return float(score), "healthy_continuation"
    return float(score), "unclear"


def _remaining_space_score(daily_df: pd.DataFrame, close: pd.Series, *, direction: str, extension_atr: float) -> float:
    if close.empty:
        return 0.0
    if len(close) < 60 or not {"high", "low"}.issubset(daily_df.columns):
        return 50.0
    tail = daily_df.tail(min(120, len(daily_df)))
    high = float(pd.to_numeric(tail["high"], errors="coerce").max())
    low = float(pd.to_numeric(tail["low"], errors="coerce").min())
    last = float(close.iloc[-1])
    width = max(high - low, 1e-9)
    favorable_pos = (high - last) / width if direction == "short" else (last - low) / width
    if extension_atr <= 2.5:
        base = 0.90
    elif extension_atr <= 4.0:
        base = 0.65
    elif extension_atr <= 6.0:
        base = 0.35
    else:
        base = 0.15
    if favorable_pos < 0.25:
        base *= 0.70
    return float(100.0 * _clip01(base))


def _volatility_fit_score(daily_df: pd.DataFrame, close: pd.Series, *, direction: str) -> float:
    if len(close) < 40:
        return 50.0
    atr20 = _average_true_range_abs(daily_df, window=20)
    atr60 = _average_true_range_abs(daily_df, window=60)
    ratio = atr20 / max(atr60, 1e-9)
    if 0.70 <= ratio <= 1.45:
        ratio_score = 1.0
    elif ratio < 0.70:
        ratio_score = _clip01((ratio - 0.30) / 0.40)
    else:
        ratio_score = _clip01(1.0 - (ratio - 1.45) / 1.25)
    efficiency = _path_efficiency(close, direction=direction, window=min(60, len(close) - 1))
    return float(100.0 * (0.55 * ratio_score + 0.45 * efficiency))


def _relative_activity_score(values: pd.Series) -> float | None:
    if len(values) < 60:
        return None
    short = float(values.tail(20).mean())
    long = float(values.tail(60).mean())
    if long <= 0:
        return None
    ratio = short / long
    if 0.85 <= ratio <= 1.75:
        return 1.0
    if ratio < 0.85:
        return _clip01((ratio - 0.45) / 0.40)
    return _clip01(1.0 - (ratio - 1.75) / 1.25)


def _participation_score(daily_df: pd.DataFrame) -> float:
    components: list[float] = []
    volume = _numeric(daily_df, "volume")
    volume_score = _relative_activity_score(volume)
    if volume_score is not None:
        components.append(volume_score)
    oi_field = "oi" if "oi" in daily_df else "open_interest"
    oi = _numeric(daily_df, oi_field)
    if len(oi) >= 60:
        short = float(oi.tail(20).mean())
        long = float(oi.tail(60).mean())
        if long > 0:
            ratio = short / long
            if ratio >= 1.02:
                components.append(1.0)
            elif ratio >= 0.98:
                components.append(0.75)
            else:
                components.append(_clip01((ratio - 0.90) / 0.08) * 0.75)
    if not components:
        return 50.0
    return float(100.0 * sum(components) / len(components))


def trend_opportunity_bucket(score: float) -> str:
    number = float(score)
    if not math.isfinite(number):
        return "missing"
    if number >= 65.0:
        return "high"
    if number >= 45.0:
        return "medium"
    return "low"


def trend_budget_margin_pct(score: float) -> float:
    number = float(score)
    if not math.isfinite(number):
        return 0.03
    if number >= 65.0:
        return 0.30
    if number >= 45.0:
        return 0.15
    if number >= 40.0:
        return 0.08
    return 0.03


def trend_opportunity_quality_from_daily(daily_df: pd.DataFrame, *, direction: str) -> dict[str, Any]:
    data = _clean_daily(daily_df)
    close = _numeric(data, "close")
    if len(close) < 60:
        return {
            "trend_opportunity_quality_score": math.nan,
            "trend_opportunity_quality_status": "insufficient_history",
            "trend_opportunity_bucket": "missing",
            "trend_budget_margin_pct": 0.03,
            "trend_structure_bucket": "missing",
        }

    resolved_direction = "short" if str(direction).lower() == "short" else "long"
    atr_abs = _average_true_range_abs(data, window=20)
    extension = _extension_atr(close, direction=resolved_direction, atr_abs=atr_abs)
    direction_score = _direction_stability_score(close, direction=resolved_direction, atr_abs=atr_abs)
    structure_score, structure_bucket = _structure_health_score(
        data,
        close,
        direction=resolved_direction,
        atr_abs=atr_abs,
    )
    remaining_score = _remaining_space_score(data, close, direction=resolved_direction, extension_atr=extension)
    volatility_score = _volatility_fit_score(data, close, direction=resolved_direction)
    participation = _participation_score(data)

    raw = (
        0.25 * direction_score
        + 0.30 * structure_score
        + 0.20 * remaining_score
        + 0.15 * volatility_score
        + 0.10 * participation
    )
    if direction_score < 35.0:
        raw = min(raw, 45.0)
    if structure_score < 35.0:
        raw = min(raw, 42.0)
    if structure_bucket == "overextended":
        raw = min(raw, 62.0)
    score = float(max(0.0, min(100.0, raw)))
    return {
        "trend_opportunity_quality_score": score,
        "trend_direction_stability_score": float(direction_score),
        "trend_structure_health_score": float(structure_score),
        "trend_remaining_space_score": float(remaining_score),
        "trend_volatility_fit_score": float(volatility_score),
        "trend_participation_score": float(participation),
        "trend_extension_atr": float(extension),
        "trend_opportunity_bucket": trend_opportunity_bucket(score),
        "trend_structure_bucket": structure_bucket,
        "trend_budget_margin_pct": trend_budget_margin_pct(score),
        "trend_opportunity_quality_status": "candidate_v1",
    }
