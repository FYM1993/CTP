from __future__ import annotations

import inspect
from datetime import date
from typing import Any

import pandas as pd

from data_cache import (
    get_hog_fundamentals,
    get_inventory,
    get_oi_structure,
    get_seasonality,
    get_spot_basis,
    get_warehouse_receipt,
)
from market.fundamental_snapshot import build_fundamental_snapshot
from phase1.scoring import build_labels, build_state_labels, calc_attention_raw

DOMAIN_TECHNICAL_TREND = "technical_trend"
DOMAIN_POSITIONING_OI = "positioning_oi"
DOMAIN_INVENTORY_SUPPLY = "inventory_supply"
DOMAIN_WAREHOUSE_RECEIPT = "warehouse_receipt"
DOMAIN_SEASONALITY_PROFIT = "seasonality_profit"


def _snapshot_domain(snapshot: dict[str, Any] | None, domain: str) -> dict[str, Any] | None:
    raw_details = (snapshot or {}).get("raw_details") or {}
    payload = raw_details.get(domain)
    if not isinstance(payload, dict) or not payload:
        return None
    return dict(payload)


def _snapshot_fields(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    return {
        "fundamental_snapshot": snapshot,
        "fundamental_coverage_score": float(snapshot.get("coverage_score") or 0.0),
        "fundamental_coverage_status": str(snapshot.get("coverage_status") or ""),
        "fundamental_domains_present": list(snapshot.get("evidence_domains_present") or []),
        "fundamental_domains_missing": list(snapshot.get("evidence_domains_missing") or []),
        "fundamental_missing_domain_reasons": list(snapshot.get("missing_domain_reasons") or []),
        "fundamental_extreme_state_direction": str(snapshot.get("extreme_state_direction") or ""),
        "fundamental_extreme_state_confirmed": bool(snapshot.get("extreme_state_confirmed")),
        "fundamental_extreme_state_reasons": list(snapshot.get("extreme_state_reasons") or []),
        "fundamental_marginal_turn_direction": str(snapshot.get("marginal_turn_direction") or ""),
        "fundamental_marginal_turn_confirmed": bool(snapshot.get("marginal_turn_confirmed")),
        "fundamental_marginal_turn_reasons": list(snapshot.get("marginal_turn_reasons") or []),
        "fundamental_reversal_confirmed": bool(snapshot.get("fundamental_reversal_confirmed")),
        "fundamental_only_proxy_evidence": bool(snapshot.get("only_proxy_evidence")),
    }


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _safe_percentile(last: float, low: float, high: float) -> float:
    return (last - low) / (high - low + 1e-12) * 100.0


def _attention_threshold(raw_threshold: float) -> float:
    return _clip_score(55.0 + raw_threshold)


def _supports_as_of_date(func: Any) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "as_of_date" in signature.parameters


def _call_factor(func: Any, *args: Any, as_of_date: date | None = None) -> Any:
    if as_of_date is not None and _supports_as_of_date(func):
        return func(*args, as_of_date=as_of_date)
    return func(*args)


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


def _entry_pool_reason(row: dict[str, Any]) -> str:
    reversal_score = float(row.get("reversal_score") or 0.0)
    trend_score = float(row.get("trend_score") or 0.0)
    dominant_family = "反转机会分" if reversal_score >= trend_score else "趋势机会分"
    dominant_score = max(reversal_score, trend_score)
    entry_threshold = float(row.get("entry_threshold") or 0.0)
    if dominant_score >= entry_threshold:
        return f"{dominant_family}达标（反转{reversal_score:.0f} / 趋势{trend_score:.0f}）"
    return (
        f"{dominant_family}未达标（反转{reversal_score:.0f} / 趋势{trend_score:.0f}；"
        f"门槛{entry_threshold:.0f}）"
    )


def _unique_domains(*domains: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for domain in domains:
        if not domain or domain in seen:
            continue
        seen.add(domain)
        ordered.append(domain)
    return ordered


def _score_driver(label: str, value: float | None, *, min_value: float = 1.0) -> str | None:
    if value is None:
        return None
    numeric = float(value)
    if abs(numeric) < min_value:
        return None
    return f"{label}{numeric:.0f}"


def _metric_driver(label: str, value: Any, *, suffix: str = "") -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value or "").strip()
        return f"{label}{text}" if text else None
    if abs(numeric) < 1e-9:
        return None
    return f"{label}{numeric:+.1f}{suffix}"


def _reason_drivers(
    reasons: list[tuple[str, float]],
    *,
    include_keywords: tuple[str, ...],
) -> list[str]:
    drivers: list[str] = []
    for reason, value in reasons:
        if include_keywords and not any(keyword in reason for keyword in include_keywords):
            continue
        if value <= 0:
            continue
        drivers.append(f"{reason}{value:.0f}")
    return drivers


def _compact_drivers(*drivers: str | None) -> list[str]:
    return [driver for driver in drivers if driver]


def _phase1_score_details(
    *,
    reversal_direction: str,
    trend_direction: str,
    reversal_score: float,
    trend_score: float,
    reversal_up: float,
    reversal_down: float,
    trend_up: float,
    trend_down: float,
    low_price: float,
    high_price: float,
    low_persistence: float,
    high_persistence: float,
    structural_down: float,
    structural_up: float,
    hog_reversal: float,
    proxy_scores: dict[str, float],
    proxy_reasons: list[tuple[str, float]],
    oi_structure: dict[str, Any] | None,
    data_coverage: float,
    shrink: float,
) -> dict[str, Any]:
    reversal_up_drivers = _compact_drivers(
        _score_driver("价格低位", low_price),
        _score_driver("低位持续", low_persistence),
        _score_driver("过剩/高库存", structural_down),
        _score_driver("利润压力", hog_reversal),
        _score_driver("历史代理反转向上", proxy_scores.get("reversal_up")),
    )
    reversal_down_drivers = _compact_drivers(
        _score_driver("价格高位", high_price),
        _score_driver("高位持续", high_persistence),
        _score_driver("过剩/高库存", structural_down),
        _score_driver("利润高位", hog_reversal),
        _score_driver("历史代理反转向下", proxy_scores.get("reversal_down")),
    )

    oi_vs_price = str((oi_structure or {}).get("oi_vs_price") or "")
    oi_change = (oi_structure or {}).get("oi_20d_change")
    oi_percentile = (oi_structure or {}).get("oi_percentile")
    trend_up_drivers = _compact_drivers(
        _score_driver("技术趋势上行", trend_up),
        f"价格/OI={oi_vs_price}" if oi_vs_price else None,
        _metric_driver("OI20日", oi_change, suffix="%"),
        _metric_driver("OI分位", oi_percentile, suffix="%"),
    )
    trend_up_drivers.extend(
        _reason_drivers(proxy_reasons, include_keywords=("上行", "多头"))
    )

    trend_down_drivers = _compact_drivers(
        _score_driver("技术趋势下行", trend_down),
        f"价格/OI={oi_vs_price}" if oi_vs_price else None,
        _metric_driver("OI20日", oi_change, suffix="%"),
        _metric_driver("OI分位", oi_percentile, suffix="%"),
    )
    trend_down_drivers.extend(
        _reason_drivers(proxy_reasons, include_keywords=("下行", "空头"))
    )

    return {
        "reversal": {
            "direction": reversal_direction,
            "score": round(reversal_score, 2),
            "up_score": round(reversal_up, 2),
            "down_score": round(reversal_down, 2),
            "drivers": reversal_up_drivers if reversal_direction == "long" else reversal_down_drivers,
        },
        "trend": {
            "direction": trend_direction,
            "score": round(trend_score, 2),
            "up_score": round(trend_up, 2),
            "down_score": round(trend_down, 2),
            "drivers": trend_up_drivers if trend_direction == "long" else trend_down_drivers,
        },
        "coverage": {
            "data_coverage": round(data_coverage, 2),
            "shrink": round(shrink, 2),
        },
    }


def _trend_evidence_domains(
    *,
    direction: str,
    trend_score: float,
    oi_structure: dict[str, Any] | None,
    inventory_score: float,
    receipt_score: float,
    seasonal_score: float,
    hog: dict[str, Any] | None,
) -> list[str]:
    domains: list[str] = []
    if trend_score > 0:
        domains.append(DOMAIN_TECHNICAL_TREND)

    oi_vs_price = str((oi_structure or {}).get("oi_vs_price") or "")
    if (direction == "long" and oi_vs_price == "增仓上涨") or (direction == "short" and oi_vs_price == "增仓下跌"):
        domains.append(DOMAIN_POSITIONING_OI)

    if inventory_score > 0:
        domains.append(DOMAIN_INVENTORY_SUPPLY)
    if receipt_score > 0:
        domains.append(DOMAIN_WAREHOUSE_RECEIPT)

    hog_trend = float((hog or {}).get("price_trend", 0.0))
    if seasonal_score > 0 or (direction == "long" and hog_trend > 0) or (direction == "short" and hog_trend < 0):
        domains.append(DOMAIN_SEASONALITY_PROFIT)
    return _unique_domains(*domains)


def _reversal_evidence_domains(
    *,
    direction: str,
    structural_score: float,
    oi_structure: dict[str, Any] | None,
    hog_reversal: float,
    hog_profit: float | None,
) -> list[str]:
    domains: list[str] = []
    if structural_score > 0:
        domains.append(DOMAIN_INVENTORY_SUPPLY)

    oi_vs_price = str((oi_structure or {}).get("oi_vs_price") or "")
    if (direction == "long" and oi_vs_price == "减仓下跌") or (direction == "short" and oi_vs_price == "减仓上涨"):
        domains.append(DOMAIN_POSITIONING_OI)

    if hog_reversal > 0 or hog_profit is not None:
        domains.append(DOMAIN_SEASONALITY_PROFIT)
    return _unique_domains(*domains)


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


def _soft_low_price_score(percentile: float) -> float:
    extreme_low = max(0.0, 25.0 - percentile) * 3.2
    broad_low = max(0.0, 45.0 - percentile) * 1.6
    return _clip_score(extreme_low + broad_low)


def _soft_high_price_score(percentile: float) -> float:
    extreme_high = max(0.0, percentile - 75.0) * 3.2
    broad_high = max(0.0, percentile - 55.0) * 1.6
    return _clip_score(extreme_high + broad_high)


def _reversal_price_scores(
    *,
    stats: dict[str, float],
    hog: dict[str, Any] | None,
) -> tuple[float, float]:
    low_anchor = min(stats["price_percentile_300d"], stats["price_percentile_full"])
    high_anchor = max(stats["price_percentile_300d"], stats["price_percentile_full"])

    if hog and hog.get("spot_price_percentile") is not None:
        spot_percentile = float(hog["spot_price_percentile"])
        low_anchor = min(low_anchor, spot_percentile)
        high_anchor = max(high_anchor, spot_percentile)

    return _soft_low_price_score(low_anchor), _soft_high_price_score(high_anchor)


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


def _historical_proxy_scores(
    *,
    df: pd.DataFrame,
    stats: dict[str, float],
) -> tuple[dict[str, float], list[tuple[str, float]]]:
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 60:
        return {
            "reversal_up": 0.0,
            "reversal_down": 0.0,
            "trend_up": 0.0,
            "trend_down": 0.0,
        }, []

    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ma120 = float(close.tail(min(len(close), 120)).mean())
    last = float(close.iloc[-1])
    ret20 = (last / (float(close.iloc[-21]) + 1e-12) - 1.0) * 100.0 if len(close) > 21 else 0.0
    ret60 = (last / (float(close.iloc[-61]) + 1e-12) - 1.0) * 100.0 if len(close) > 61 else ret20

    avg_pct = (stats["price_percentile_300d"] + stats["price_percentile_full"]) / 2.0
    low_price = _clip_score(max(0.0, 25.0 - avg_pct) * 4.0)
    high_price = _clip_score(max(0.0, avg_pct - 75.0) * 4.0)
    low_persistence = _clip_score(stats["low_persistence_days"] / 25.0 * 100.0)
    high_persistence = _clip_score(stats["high_persistence_days"] / 25.0 * 100.0)

    trend_alignment_up = 100.0 if last > ma20 > ma60 > ma120 else 0.0
    trend_alignment_down = 100.0 if last < ma20 < ma60 < ma120 else 0.0
    trend_impulse_up = _clip_score(max(ret20, 0.0) * 8.0 + max(ret60, 0.0) * 3.5)
    trend_impulse_down = _clip_score(max(-ret20, 0.0) * 8.0 + max(-ret60, 0.0) * 3.5)

    oi_structure = get_oi_structure(df)
    oi_trend_up = 0.0
    oi_trend_down = 0.0
    oi_reversal_up = 0.0
    oi_reversal_down = 0.0
    proxy_reasons: list[tuple[str, float]] = []

    if oi_structure:
        oi_vs_price = str(oi_structure.get("oi_vs_price") or "")
        oi_change = float(oi_structure.get("oi_20d_change", 0.0))
        oi_pct = float(oi_structure.get("oi_percentile", 50.0))

        if oi_vs_price == "增仓上涨":
            oi_trend_up = _clip_score(55.0 + max(oi_change, 0.0) * 2.5)
            proxy_reasons.append(("历史代理:价格/OI共振上行", oi_trend_up))
        elif oi_vs_price == "增仓下跌":
            oi_trend_down = _clip_score(55.0 + max(oi_change, 0.0) * 2.5)
            proxy_reasons.append(("历史代理:价格/OI共振下行", oi_trend_down))
        elif oi_vs_price == "减仓下跌":
            oi_reversal_up = _clip_score(45.0 + max(50.0 - oi_pct, 0.0) * 0.6 + max(-oi_change, 0.0) * 2.0)
            proxy_reasons.append(("历史代理:低位减仓洗出", oi_reversal_up))
        elif oi_vs_price == "减仓上涨":
            oi_reversal_down = _clip_score(45.0 + max(oi_pct - 50.0, 0.0) * 0.6 + max(-oi_change, 0.0) * 2.0)
            proxy_reasons.append(("历史代理:高位减仓衰竭", oi_reversal_down))

    trend_up = _clip_score(
        high_price * 0.30
        + high_persistence * 0.15
        + trend_alignment_up * 0.25
        + trend_impulse_up * 0.15
        + oi_trend_up * 0.15
    )
    trend_down = _clip_score(
        low_price * 0.30
        + low_persistence * 0.15
        + trend_alignment_down * 0.25
        + trend_impulse_down * 0.15
        + oi_trend_down * 0.15
    )
    if oi_trend_up <= 0:
        trend_up = _clip_score(trend_up * 0.82)
    if oi_trend_down <= 0:
        trend_down = _clip_score(trend_down * 0.82)
    reversal_up = _clip_score(
        low_price * 0.42
        + low_persistence * 0.18
        + trend_impulse_down * 0.15
        + oi_reversal_up * 0.25
    )
    reversal_down = _clip_score(
        high_price * 0.42
        + high_persistence * 0.18
        + trend_impulse_up * 0.15
        + oi_reversal_down * 0.25
    )

    if trend_up >= 60:
        proxy_reasons.append(("历史代理:价格沿均线多头扩散", trend_up))
    if trend_down >= 60:
        proxy_reasons.append(("历史代理:价格沿均线空头扩散", trend_down))
    if reversal_up >= 60:
        proxy_reasons.append(("历史代理:长期低位+抛压出清", reversal_up))
    if reversal_down >= 60:
        proxy_reasons.append(("历史代理:长期高位+追涨衰竭", reversal_down))

    return {
        "reversal_up": reversal_up,
        "reversal_down": reversal_down,
        "trend_up": trend_up,
        "trend_down": trend_down,
    }, proxy_reasons


def _build_candidate(
    *,
    info: dict[str, Any],
    df: pd.DataFrame,
    as_of_date: date | None = None,
) -> dict[str, Any] | None:
    if df is not None and as_of_date is not None and "date" in df.columns:
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        df = frame.loc[frame["date"].dt.date <= as_of_date].reset_index(drop=True)
    if df is None or len(df) < 60:
        return None

    stats = _price_stats(df)
    fundamental_snapshot = build_fundamental_snapshot(
        symbol=info["symbol"],
        name=info["name"],
        exchange=info["exchange"],
        daily_df=df,
        as_of_date=as_of_date,
        inventory_fetcher=get_inventory,
        warehouse_receipt_fetcher=get_warehouse_receipt,
        spot_basis_fetcher=get_spot_basis,
        seasonality_fetcher=get_seasonality,
    )
    inv = _snapshot_domain(fundamental_snapshot, "inventory")
    receipt = _snapshot_domain(fundamental_snapshot, "warehouse_receipt")
    seasonal = _snapshot_domain(fundamental_snapshot, "seasonality")
    hog = _call_factor(get_hog_fundamentals, as_of_date=as_of_date) if info["symbol"] == "LH0" else None
    oi_structure = get_oi_structure(df)

    inv_up, inv_down, inv_reasons = _inventory_scores(inv)
    receipt_up, receipt_down, receipt_reasons = _receipt_scores(receipt)
    seasonal_up, seasonal_down, seasonal_reasons = _seasonal_scores(seasonal)
    hog_reversal, hog_trend, hog_profit, hog_reasons = _hog_scores(hog)
    use_proxy_scores = as_of_date is not None or not any((inv, receipt, seasonal, hog))
    technical_proxy_scores, technical_proxy_reasons = _historical_proxy_scores(df=df, stats=stats)
    if use_proxy_scores:
        proxy_scores = technical_proxy_scores
        proxy_reasons = technical_proxy_reasons
    else:
        proxy_scores = {
            "reversal_up": 0.0,
            "reversal_down": 0.0,
            "trend_up": technical_proxy_scores["trend_up"],
            "trend_down": technical_proxy_scores["trend_down"],
        }
        proxy_reasons = [
            item
            for item in technical_proxy_reasons
            if "价格/OI共振" in item[0] or "价格沿均线" in item[0]
        ]

    low_price, high_price = _reversal_price_scores(stats=stats, hog=hog)
    low_persistence = _clip_score(stats["low_persistence_days"] / 25.0 * 100.0)
    high_persistence = _clip_score(stats["high_persistence_days"] / 25.0 * 100.0)
    structural_down = max(inv_down, receipt_down)
    structural_up = max(inv_up, receipt_up)

    reversal_up = (
        low_price * 0.18
        + low_persistence * 0.28
        + hog_reversal * 0.42
        + structural_down * 0.30
    )
    if low_price >= 50 and hog_reversal >= 45:
        reversal_up += 12.0
    if low_persistence >= 60 and structural_down >= 70:
        reversal_up += 13.0
    if low_persistence >= 60 and hog_reversal >= 50:
        reversal_up += 6.0
    reversal_up = _clip_score(reversal_up)

    reversal_down = (
        high_price * 0.18
        + high_persistence * 0.22
        + hog_reversal * 0.42
        + structural_down * 0.12
    )
    if high_price >= 50 and hog_reversal >= 45:
        reversal_down += 12.0
    if high_persistence >= 60 and hog_reversal >= 50:
        reversal_down += 6.0
    reversal_down = _clip_score(reversal_down)

    trend_up = _clip_score(proxy_scores["trend_up"])
    trend_down = _clip_score(proxy_scores["trend_down"])

    reversal_up = _clip_score(max(reversal_up, proxy_scores["reversal_up"]))
    reversal_down = _clip_score(max(reversal_down, proxy_scores["reversal_down"]))
    trend_up = _clip_score(max(trend_up, proxy_scores["trend_up"]))
    trend_down = _clip_score(max(trend_down, proxy_scores["trend_down"]))

    available_families = 1
    if inv:
        available_families += 1
    if receipt:
        available_families += 1
    if seasonal or hog:
        available_families += 1
    if max(proxy_scores.values()) >= 40.0:
        available_families += 1
    available_families = min(available_families, 4)
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
            dominant_reasons.extend(
                [
                    item
                    for item in inv_reasons + receipt_reasons
                    if "高位" in item[0] or "累" in item[0] or "增加" in item[0]
                ]
            )
    else:
        if trend_up >= trend_down:
            dominant_reasons.append(("历史代理:价格趋势上行", trend_up))
        else:
            dominant_reasons.append(("历史代理:价格趋势下行", trend_down))

    dominant_reasons.extend(proxy_reasons)
    dominant_reasons = [item for item in dominant_reasons if item[1] > 0]
    dominant_reasons.sort(key=lambda item: item[1], reverse=True)
    reason_summary = "；".join(reason for reason, _ in dominant_reasons[:3]) or "基本面证据有限"

    reversal_direction = "long" if reversal_up >= reversal_down else "short"
    trend_direction = "long" if trend_up >= trend_down else "short"
    trend_evidence_domains = _trend_evidence_domains(
        direction=trend_direction,
        trend_score=max(trend_up, trend_down),
        oi_structure=oi_structure,
        inventory_score=inv_up if trend_direction == "long" else inv_down,
        receipt_score=receipt_up if trend_direction == "long" else receipt_down,
        seasonal_score=seasonal_up if trend_direction == "long" else seasonal_down,
        hog=hog,
    )
    reversal_evidence_domains = _reversal_evidence_domains(
        direction=reversal_direction,
        structural_score=structural_down,
        oi_structure=oi_structure,
        hog_reversal=hog_reversal,
        hog_profit=hog_profit,
    )
    score_details = _phase1_score_details(
        reversal_direction=reversal_direction,
        trend_direction=trend_direction,
        reversal_score=reversal_score,
        trend_score=trend_score,
        reversal_up=reversal_up,
        reversal_down=reversal_down,
        trend_up=trend_up,
        trend_down=trend_down,
        low_price=low_price,
        high_price=high_price,
        low_persistence=low_persistence,
        high_persistence=high_persistence,
        structural_down=structural_down,
        structural_up=structural_up,
        hog_reversal=hog_reversal,
        proxy_scores=proxy_scores,
        proxy_reasons=proxy_reasons,
        oi_structure=oi_structure,
        data_coverage=data_coverage,
        shrink=shrink,
    )

    candidate = {
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
        "hog_price_trend": None if not hog else hog.get("price_trend"),
        "oi_vs_price": None if not oi_structure else oi_structure.get("oi_vs_price"),
        "oi_20d_change": None if not oi_structure else oi_structure.get("oi_20d_change"),
        "oi_percentile": None if not oi_structure else oi_structure.get("oi_percentile"),
        "reversal_score": reversal_score,
        "trend_score": trend_score,
        "reversal_direction": reversal_direction,
        "trend_direction": trend_direction,
        "trend_evidence_domains": trend_evidence_domains,
        "reversal_evidence_domains": reversal_evidence_domains,
        "attention_score": 0.0,
        "labels": labels,
        "state_labels": state_labels,
        "data_coverage": data_coverage,
        "reason_summary": reason_summary,
        "entry_pool_reason": "",
        "phase1_score_details": score_details,
        "_reversal_up_score": reversal_up,
        "_reversal_down_score": reversal_down,
        "_trend_up_score": trend_up,
        "_trend_down_score": trend_down,
    }
    candidate.update(_snapshot_fields(fundamental_snapshot))
    return candidate


def build_phase1_candidates(
    *,
    all_data: dict[str, pd.DataFrame],
    symbols: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
    as_of_date: date | None = None,
) -> list[dict[str, Any]]:
    thresholds = _symbol_thresholds(symbols=symbols, threshold=threshold, config=config)
    rows: list[dict[str, Any]] = []
    for info in symbols:
        df = all_data.get(info["symbol"])
        candidate = _build_candidate(info=info, df=df, as_of_date=as_of_date)
        if candidate is None:
            continue
        candidate["entry_threshold"] = thresholds.get(info["symbol"], _attention_threshold(threshold))
        candidate["entry_pool_reason"] = _entry_pool_reason(candidate)
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

    return rows


def is_phase1_candidate_eligible(row: dict[str, Any]) -> bool:
    dominant_score = max(row["reversal_score"], row["trend_score"])
    entry_threshold = float(row.get("entry_threshold", 55.0))
    if dominant_score >= entry_threshold:
        return True
    return row.get("data_coverage", 1.0) < 0.5 and dominant_score >= max(entry_threshold, 85.0)


def select_top_candidates(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        if is_phase1_candidate_eligible(row):
            eligible.append(row)
    eligible.sort(key=lambda row: row["attention_score"], reverse=True)
    return eligible[:top_n]


def run_phase1_pipeline(
    *,
    all_data: dict[str, pd.DataFrame],
    symbols: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
    as_of_date: date | None = None,
) -> list[dict[str, Any]]:
    rows = build_phase1_candidates(
        all_data=all_data,
        symbols=symbols,
        threshold=threshold,
        config=config,
        as_of_date=as_of_date,
    )

    fs_cfg = config.get("fundamental_screening") or {}
    top_n = int(fs_cfg.get("top_n") or 40)
    return select_top_candidates(rows, top_n=top_n)
