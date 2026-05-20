from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def visible_daily_before_entry(daily_df: pd.DataFrame, entry_time: str) -> pd.DataFrame:
    if daily_df is None or daily_df.empty or "date" not in daily_df:
        return pd.DataFrame(columns=[] if daily_df is None else list(daily_df.columns))
    out = daily_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date", kind="stable")
    trade_date = pd.Timestamp(entry_time).date()
    out["trade_date"] = out["date"].dt.date
    return out.loc[out["trade_date"] < trade_date].drop(columns=["trade_date"]).copy()


def _directional_price_return(close: pd.Series, *, direction: str, window: int) -> float:
    if len(close) < 2:
        return 0.0
    resolved_window = min(max(int(window), 1), len(close) - 1)
    anchor = float(close.iloc[-resolved_window - 1])
    last = float(close.iloc[-1])
    if anchor <= 0 or last <= 0:
        return 0.0
    return float(anchor - last) / anchor if direction == "short" else float(last - anchor) / anchor


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
        raw = 1.0 if vol <= 1e-8 and directional_return > 0 else _clip01(0.5 + directional_return / (2.0 * vol)) if vol > 1e-8 else 0.0
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
        net_move = first - last if direction == "short" else last - first
        path = float(window_close.diff().abs().sum())
        scores.append(0.0 if path <= 0 or net_move <= 0 else _clip01(net_move / path))
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
        plus_dm_values.append(high_diff if high_diff > low_diff and high_diff > 0 else 0.0)
        minus_dm_values.append(low_diff if low_diff > high_diff and low_diff > 0 else 0.0)
        tr_values.append(
            max(
                float(high.iloc[idx] - low.iloc[idx]),
                abs(float(high.iloc[idx] - close.iloc[idx - 1])),
                abs(float(low.iloc[idx] - close.iloc[idx - 1])),
            )
        )
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
        if denom > 0:
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
        aroon_up = 100.0 * (window - (len(high_values) - 1 - high_idx)) / window
        aroon_down = 100.0 * (window - (len(low_values) - 1 - low_idx)) / window
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
    return _clip01((_linear_slope(xs, ys) - 0.45) / 0.30)


def _trend_persistence(close: pd.Series, *, direction: str, window: int = 60) -> float:
    if len(close) < 3:
        return 0.0
    resolved_window = min(max(int(window), 2), len(close) - 1)
    window_close = close.tail(resolved_window + 1).astype(float)
    anchor = float(window_close.iloc[0])
    last = float(window_close.iloc[-1])
    net_move = anchor - last if direction == "short" else last - anchor
    path = float(window_close.diff().abs().sum())
    if path <= 0:
        return 0.0
    return _clip01(net_move / path)


def _moving_average_structure(close: pd.Series, *, direction: str) -> float:
    if len(close) < 2:
        return 0.0
    checks: list[bool] = []
    last = float(close.iloc[-1])
    ma_values: dict[int, float] = {}
    ma_slopes: dict[int, float] = {}
    for window in [20, 60, 120]:
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
    extension_atr = (ma - last) / atr_abs if direction == "short" else (last - ma) / atr_abs
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


def medium_term_quality_from_daily(
    daily_df: pd.DataFrame,
    *,
    direction: str,
    trend_phase: str,
) -> dict[str, Any]:
    if daily_df is None or daily_df.empty or "close" not in daily_df:
        return {"medium_term_quality_score": math.nan, "medium_term_quality_status": "insufficient_history"}
    close = pd.to_numeric(daily_df["close"], errors="coerce").dropna()
    if len(close) < 2:
        return {"medium_term_quality_score": math.nan, "medium_term_quality_status": "insufficient_history"}

    atr_pct = _average_true_range_pct(daily_df)
    momentum = _time_series_momentum_score(close, direction=direction)
    efficiency = _kaufman_efficiency_score(close, direction=direction)
    trend_state = _adx_dmi_score(daily_df, direction=direction)
    freshness = _aroon_freshness_score(daily_df, direction=direction)
    regime = _hurst_regime_score(close)
    persistence = _trend_persistence(close, direction=direction, window=60)
    structure = _moving_average_structure(close, direction=direction)
    phase = _trend_phase_alignment(direction=direction, trend_phase=trend_phase)
    entry_location = _entry_location_score(close, direction=direction, atr_pct=atr_pct)
    score = 100.0 * (
        0.25 * momentum
        + 0.20 * efficiency
        + 0.20 * trend_state
        + 0.15 * freshness
        + 0.10 * entry_location
        + 0.05 * regime
        + 0.05 * phase
    )
    return {
        "medium_term_quality_score": float(max(0.0, min(100.0, score))),
        "medium_term_slope_score": float(momentum * 100.0),
        "medium_term_persistence_score": float(persistence * 100.0),
        "medium_term_structure_score": float(structure * 100.0),
        "medium_term_phase_alignment_score": float(phase * 100.0),
        "medium_term_entry_location_score": float(entry_location * 100.0),
        "medium_term_momentum_score": float(momentum * 100.0),
        "medium_term_efficiency_score": float(efficiency * 100.0),
        "medium_term_trend_state_score": float(trend_state * 100.0),
        "medium_term_freshness_score": float(freshness * 100.0),
        "medium_term_regime_score": float(regime * 100.0),
        "medium_term_quality_status": "candidate_v3",
    }
