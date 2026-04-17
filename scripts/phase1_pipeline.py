from __future__ import annotations

from typing import Any

import pandas as pd

from data_cache import (
    get_hog_fundamentals,
    get_inventory,
    get_seasonality,
    get_warehouse_receipt,
)
from phase1_scoring import build_labels, build_state_labels, calc_attention_raw


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _safe_percentile(last: float, low: float, high: float) -> float:
    return (last - low) / (high - low + 1e-12) * 100.0


def _attention_threshold(raw_threshold: float) -> float:
    return _clip_score(55.0 + raw_threshold)


def _symbol_thresholds(
    *,
    symbols: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
) -> dict[str, float]:
    fs_cfg = config.get("fundamental_screening") or {}
    default_raw = float(fs_cfg.get("default_threshold", threshold))
    resolved = {info["symbol"]: _attention_threshold(default_raw) for info in symbols}

    for cat_cfg in (fs_cfg.get("categories") or {}).values():
        if not isinstance(cat_cfg, dict):
            continue
        if cat_cfg.get("threshold") is None:
            continue
        cat_threshold = _attention_threshold(float(cat_cfg["threshold"]))
        for symbol in cat_cfg.get("symbols") or []:
            resolved[str(symbol)] = cat_threshold
    return resolved


def _price_stats(df: pd.DataFrame) -> dict[str, float]:
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    last = float(close.iloc[-1])

    recent = close.tail(min(len(close), 300))
    range_pct = _safe_percentile(last, float(recent.min()), float(recent.max()))
    full_pct = _safe_percentile(last, float(close.min()), float(close.max()))

    tail = close.tail(min(len(close), 120))
    low_band = float(tail.quantile(0.35))
    high_band = float(tail.quantile(0.65))

    low_days = 0
    high_days = 0
    for value in reversed(tail.tolist()):
        if value <= low_band:
            low_days += 1
        else:
            break
    for value in reversed(tail.tolist()):
        if value >= high_band:
            high_days += 1
        else:
            break

    return {
        "price": last,
        "range_pct": range_pct,
        "price_percentile_300d": range_pct,
        "price_percentile_full": full_pct,
        "low_persistence_days": low_days,
        "high_persistence_days": high_days,
    }


def _inventory_scores(inv: dict[str, Any] | None) -> tuple[float, float, list[tuple[str, float]]]:
    if not inv:
        return 0.0, 0.0, []

    change = float(inv.get("inv_change_4wk", 0.0))
    percentile = float(inv.get("inv_percentile", 50.0))
    cum_weeks = int(inv.get("inv_cumulating_weeks", 4))

    tightening = 0.0
    oversupply = 0.0
    reasons: list[tuple[str, float]] = []

    if change < 0:
        tightening += _clip_score(abs(change) * 3.0)
        if change <= -10:
            reasons.append((f"库存4周去化{change:+.1f}%", abs(change)))
    elif change > 0:
        oversupply += _clip_score(change * 3.0)
        if change >= 10:
            reasons.append((f"库存4周累积{change:+.1f}%", abs(change)))

    if percentile <= 30:
        tightening += _clip_score((30.0 - percentile) * 2.5)
        if percentile <= 20:
            reasons.append((f"库存处于低位{percentile:.0f}%", 30.0 - percentile))
    elif percentile >= 70:
        oversupply += _clip_score((percentile - 70.0) * 2.5)
        if percentile >= 80:
            reasons.append((f"库存处于高位{percentile:.0f}%", percentile - 70.0))

    if cum_weeks <= 2:
        tightening += 22.0
        reasons.append((f"最近8周仅{cum_weeks}周累库", 22.0))
    elif cum_weeks >= 6:
        oversupply += 22.0
        reasons.append((f"最近8周有{cum_weeks}周累库", 22.0))

    return _clip_score(tightening), _clip_score(oversupply), reasons


def _receipt_scores(receipt: dict[str, Any] | None) -> tuple[float, float, list[tuple[str, float]]]:
    if not receipt:
        return 0.0, 0.0, []

    change = float(receipt.get("receipt_change", 0.0))
    total = abs(float(receipt.get("receipt_total", 0.0)))
    ratio = abs(change) / (total + 1e-12) if total > 0 else 0.0

    tightening = 0.0
    oversupply = 0.0
    reasons: list[tuple[str, float]] = []

    if change < 0:
        tightening = _clip_score(ratio * 220.0)
        if ratio > 0.03:
            reasons.append((f"仓单明显下降{change:+.0f}", ratio * 1000.0))
    elif change > 0:
        oversupply = _clip_score(ratio * 220.0)
        if ratio > 0.03:
            reasons.append((f"仓单明显增加{change:+.0f}", ratio * 1000.0))

    return tightening, oversupply, reasons


def _seasonal_scores(seasonal: dict[str, Any] | None) -> tuple[float, float, list[tuple[str, float]]]:
    if not seasonal:
        return 0.0, 0.0, []

    signal = float(seasonal.get("seasonal_signal", 0.0))
    avg_ret = float(seasonal.get("hist_avg_return", 0.0))
    strength = _clip_score(abs(signal) * 45.0)

    reasons: list[tuple[str, float]] = []
    if strength >= 20:
        direction = "偏强" if signal > 0 else "偏弱"
        reasons.append((f"季节性{direction}(历史{avg_ret:+.1f}%)", strength))

    if signal > 0:
        return strength, 0.0, reasons
    if signal < 0:
        return 0.0, strength, reasons
    return 0.0, 0.0, reasons


def _hog_scores(hog: dict[str, Any] | None) -> tuple[float, float, float | None, list[tuple[str, float]]]:
    if not hog:
        return 0.0, 0.0, None, []

    profit_margin = hog.get("profit_margin")
    price_trend = float(hog.get("price_trend", 0.0)) if hog.get("price_trend") is not None else 0.0

    reversal_bias = 0.0
    trend_bias = 0.0
    reasons: list[tuple[str, float]] = []

    pm_value: float | None = None
    if profit_margin is not None:
        pm_value = float(profit_margin)
        if pm_value < 0:
            reversal_bias = _clip_score(abs(pm_value) * 2.4)
            if pm_value <= -10:
                reasons.append((f"养殖利润深亏{pm_value:.0f}%", abs(pm_value)))
        elif pm_value > 0:
            reversal_bias = _clip_score(pm_value * 1.8)
            if pm_value >= 12:
                reasons.append((f"养殖利润偏高{pm_value:.0f}%", pm_value))

    if price_trend != 0.0:
        trend_bias = _clip_score(abs(price_trend) * 6.0)
        if abs(price_trend) >= 3:
            direction = "走强" if price_trend > 0 else "走弱"
            reasons.append((f"现货短期{direction}{price_trend:+.1f}%", abs(price_trend) * 5.0))

    return reversal_bias, trend_bias, pm_value, reasons


def _build_candidate(
    *,
    info: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    if df is None or len(df) < 60:
        return None

    stats = _price_stats(df)
    inv = get_inventory(info["symbol"])
    receipt = get_warehouse_receipt(info["symbol"])
    seasonal = get_seasonality(df)
    hog = get_hog_fundamentals() if info["symbol"] == "LH0" else None

    inv_up, inv_down, inv_reasons = _inventory_scores(inv)
    receipt_up, receipt_down, receipt_reasons = _receipt_scores(receipt)
    seasonal_up, seasonal_down, seasonal_reasons = _seasonal_scores(seasonal)
    hog_reversal, hog_trend, hog_profit, hog_reasons = _hog_scores(hog)

    avg_pct = (stats["price_percentile_300d"] + stats["price_percentile_full"]) / 2.0
    low_price = _clip_score(max(0.0, 25.0 - avg_pct) * 4.0)
    high_price = _clip_score(max(0.0, avg_pct - 75.0) * 4.0)
    low_persistence = _clip_score(stats["low_persistence_days"] / 25.0 * 100.0)
    high_persistence = _clip_score(stats["high_persistence_days"] / 25.0 * 100.0)

    reversal_up = (
        low_price * 0.32
        + low_persistence * 0.16
        + hog_reversal * 0.36
        + max(inv_down, receipt_down) * 0.16
    )
    if low_price >= 80 and hog_reversal >= 45:
        reversal_up += 15.0
    if hog_reversal <= 0:
        reversal_up *= 0.72
    reversal_up = _clip_score(reversal_up)

    reversal_down = (
        high_price * 0.32
        + high_persistence * 0.16
        + hog_reversal * 0.36
        + max(inv_up, receipt_up) * 0.16
    )
    if high_price >= 80 and hog_reversal >= 45:
        reversal_down += 15.0
    if hog_reversal <= 0:
        reversal_down *= 0.72
    reversal_down = _clip_score(reversal_down)

    trend_up = _clip_score(
        inv_up * 0.52
        + receipt_up * 0.23
        + seasonal_up * 0.15
        + (hog_trend if hog and float(hog.get("price_trend", 0.0)) > 0 else 0.0) * 0.10
    )
    trend_down = _clip_score(
        inv_down * 0.52
        + receipt_down * 0.23
        + seasonal_down * 0.15
        + (hog_trend if hog and float(hog.get("price_trend", 0.0)) < 0 else 0.0) * 0.10
    )

    available_families = 1
    if inv:
        available_families += 1
    if receipt:
        available_families += 1
    if seasonal or hog:
        available_families += 1
    data_coverage = round(available_families / 4.0, 2)
    shrink = 0.6 + 0.4 * data_coverage

    reversal_score = round(max(reversal_up, reversal_down) * shrink, 2)
    trend_score = round(max(trend_up, trend_down) * shrink, 2)

    labels = build_labels(
        reversal_score=reversal_score,
        trend_score=trend_score,
        data_coverage=data_coverage,
    )
    state_labels = build_state_labels(
        reversal_up_score=reversal_up,
        reversal_down_score=reversal_down,
        trend_up_score=trend_up,
        trend_down_score=trend_down,
    )

    dominant_reasons: list[tuple[str, float]] = []
    if reversal_score >= trend_score:
        if reversal_up >= reversal_down:
            dominant_reasons.extend(
                [
                    ("长期价格处于低位", low_price),
                    ("低位失衡持续", low_persistence),
                ]
            )
            dominant_reasons.extend([item for item in hog_reasons if "亏" in item[0] or "利润" in item[0]])
            dominant_reasons.extend([item for item in inv_reasons + receipt_reasons if "高位" in item[0] or "累" in item[0]])
        else:
            dominant_reasons.extend(
                [
                    ("长期价格处于高位", high_price),
                    ("高位失衡持续", high_persistence),
                ]
            )
            dominant_reasons.extend([item for item in hog_reasons if "偏高" in item[0] or "利润" in item[0]])
            dominant_reasons.extend([item for item in inv_reasons + receipt_reasons if "低位" in item[0] or "下降" in item[0]])
    else:
        if trend_up >= trend_down:
            dominant_reasons.extend(inv_reasons + receipt_reasons + seasonal_reasons + hog_reasons)
        else:
            dominant_reasons.extend(inv_reasons + receipt_reasons + seasonal_reasons + hog_reasons)

    dominant_reasons = [item for item in dominant_reasons if item[1] > 0]
    dominant_reasons.sort(key=lambda item: item[1], reverse=True)
    reason_summary = "；".join(reason for reason, _ in dominant_reasons[:3]) or "基本面证据有限"

    dominant_family = "反转机会分" if reversal_score >= trend_score else "趋势机会分"
    entry_pool_reason = f"{dominant_family}达标（反转{reversal_score:.0f} / 趋势{trend_score:.0f}）"

    return {
        "symbol": info["symbol"],
        "name": info["name"],
        "exchange": info["exchange"],
        "price": stats["price"],
        "range_pct": round(stats["range_pct"], 2),
        "score": 0.0,
        "fund_score": 0.0,
        "fund_details": reason_summary,
        "inv_change_4wk": None if not inv else inv.get("inv_change_4wk"),
        "inv_percentile": None if not inv else inv.get("inv_percentile"),
        "receipt_change": None if not receipt else receipt.get("receipt_change"),
        "seasonal_signal": None if not seasonal else seasonal.get("seasonal_signal"),
        "hog_profit": hog_profit,
        "reversal_score": reversal_score,
        "trend_score": trend_score,
        "attention_score": 0.0,
        "labels": labels,
        "state_labels": state_labels,
        "data_coverage": data_coverage,
        "reason_summary": reason_summary,
        "entry_pool_reason": entry_pool_reason,
        "_reversal_up_score": reversal_up,
        "_reversal_down_score": reversal_down,
        "_trend_up_score": trend_up,
        "_trend_down_score": trend_down,
    }


def select_top_candidates(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        dominant_score = max(row["reversal_score"], row["trend_score"])
        entry_threshold = float(row.get("entry_threshold", 55.0))
        if dominant_score >= entry_threshold:
            eligible.append(row)
            continue
        if row.get("data_coverage", 1.0) < 0.5 and dominant_score >= max(entry_threshold, 85.0):
            eligible.append(row)
    eligible.sort(key=lambda row: row["attention_score"], reverse=True)
    return eligible[:top_n]


def run_phase1_pipeline(
    *,
    all_data: dict[str, pd.DataFrame],
    symbols: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = _symbol_thresholds(symbols=symbols, threshold=threshold, config=config)
    rows: list[dict[str, Any]] = []
    for info in symbols:
        df = all_data.get(info["symbol"])
        candidate = _build_candidate(info=info, df=df)
        if candidate is None:
            continue
        candidate["entry_threshold"] = thresholds.get(info["symbol"], _attention_threshold(threshold))
        rows.append(candidate)

    if not rows:
        return []

    reversal_ranked = sorted(rows, key=lambda row: row["reversal_score"], reverse=True)
    trend_ranked = sorted(rows, key=lambda row: row["trend_score"], reverse=True)
    reversal_rank = {id(row): rank + 1 for rank, row in enumerate(reversal_ranked)}
    trend_rank = {id(row): rank + 1 for rank, row in enumerate(trend_ranked)}

    max_attention_raw = 0.0
    for row in rows:
        row["_attention_raw"] = calc_attention_raw(
            reversal_rank=reversal_rank[id(row)],
            trend_rank=trend_rank[id(row)],
            reversal_score=row["reversal_score"],
            trend_score=row["trend_score"],
            data_coverage=row["data_coverage"],
        )
        max_attention_raw = max(max_attention_raw, row["_attention_raw"])

    for row in rows:
        dominant = max(row["reversal_score"], row["trend_score"])
        raw_scale = row["_attention_raw"] / max_attention_raw if max_attention_raw > 0 else 0.0
        row["attention_score"] = round(dominant * (0.6 + 0.4 * raw_scale), 2)
        row["score"] = row["attention_score"]
        row["fund_score"] = row["attention_score"]
        row.pop("_attention_raw", None)
        row.pop("_reversal_up_score", None)
        row.pop("_reversal_down_score", None)
        row.pop("_trend_up_score", None)
        row.pop("_trend_down_score", None)

    fs_cfg = config.get("fundamental_screening") or {}
    top_n = int(fs_cfg.get("top_n") or 40)
    return select_top_candidates(rows, top_n=top_n)
