"""
Phase 1 基本面评分 — Legacy（原 daily_workflow._score_symbol 逻辑）

供 production 默认路径与 regime 引擎对照使用；行为变更前请先跑 compare_p1_engines / pytest。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_cache import (
    get_inventory,
    get_warehouse_receipt,
    get_seasonality,
    get_hog_fundamentals,
)


def compute_range_pct(df: pd.DataFrame) -> tuple[float, float]:
    """返回 (last_close, range_pct)，range 为近 min(len,300) 日收盘极值区间位。"""
    close = df["close"]
    last = float(close.iloc[-1])
    n = min(len(df), 300)
    recent = df.tail(n)
    high_all = float(recent["close"].max())
    low_all = float(recent["close"].min())
    range_pct = (last - low_all) / (high_all - low_all + 1e-12) * 100
    return last, range_pct


def compute_supply_layers(sym: str, df: pd.DataFrame) -> tuple[float, list[str], dict]:
    """
    库存 + 仓单 + 季节性（与 legacy 完全一致）。

    Returns
    -------
    score, detail_parts, meta
        meta: inv_change_4wk, inv_percentile, receipt_change, seasonal_signal
    """
    inv = get_inventory(sym)
    receipt = get_warehouse_receipt(sym)
    seasonal = get_seasonality(df)

    fund_score = 0.0
    fund_details: list[str] = []

    inv_4wk = None
    inv_pct = None
    if inv:
        inv_4wk = inv["inv_change_4wk"]
        inv_pct = inv.get("inv_percentile")
        cum = inv.get("inv_cumulating_weeks", 4)
        trend = inv.get("inv_trend", "持平")

        if inv_4wk < -10:
            fund_score += 10
        elif inv_4wk < -3:
            fund_score += 5
        elif inv_4wk > 10:
            fund_score -= 10
        elif inv_4wk > 3:
            fund_score -= 5
        if cum <= 2:
            fund_score += 10
        elif cum >= 6:
            fund_score -= 10

        fund_details.append(f"库存{trend}({inv_4wk:+.1f}%)")

    if inv_pct is not None:
        if inv_pct < 15:
            fund_score += 10
            fund_details.append(f"库存低位{inv_pct:.0f}%")
        elif inv_pct < 30:
            fund_score += 5
        elif inv_pct > 85:
            fund_score -= 10
            fund_details.append(f"库存高位{inv_pct:.0f}%")
        elif inv_pct > 70:
            fund_score -= 5

    receipt_change = None
    if receipt:
        rc = receipt["receipt_change"]
        receipt_change = rc
        rt = receipt["receipt_total"]
        if rc < 0 and rt > 0:
            change_ratio = abs(rc) / (rt + 1e-12)
            if change_ratio > 0.05:
                fund_score += 15
                fund_details.append(f"仓单大减{rc:+.0f}")
            elif change_ratio > 0.01:
                fund_score += 8
        elif rc > 0 and rt > 0:
            change_ratio = rc / (rt + 1e-12)
            if change_ratio > 0.05:
                fund_score -= 15
                fund_details.append(f"仓单大增{rc:+.0f}")
            elif change_ratio > 0.01:
                fund_score -= 8

    seasonal_sig = None
    if seasonal:
        sig = seasonal["seasonal_signal"]
        seasonal_sig = sig
        avg_ret = seasonal["hist_avg_return"]
        if abs(sig) > 0.6:
            s = float(np.clip(sig * 10, -10, 10))
            fund_score += s
            direction = "上涨" if sig > 0 else "下跌"
            fund_details.append(f"季节性偏{direction}(历史{avg_ret:+.1f}%)")

    meta = {
        "inv_change_4wk": inv_4wk,
        "inv_percentile": inv_pct,
        "receipt_change": receipt_change,
        "seasonal_signal": seasonal_sig,
    }
    return fund_score, fund_details, meta


def _hog_block_legacy(sym: str, range_pct: float, hog: dict | None) -> tuple[float, list[str], float | None]:
    """原生猪专项 + 缺失补偿；返回 (hog_score, detail_lines, hog_profit)。"""
    fund_score = 0.0
    fund_details: list[str] = []
    hog_profit = None

    if hog:
        if "profit_margin" in hog:
            pm = hog["profit_margin"]
            if pm is not None:
                hog_profit = pm
                if pm < -15:
                    fund_score += 15
                    fund_details.append(f"养殖亏损{pm:.0f}%")
                elif pm < -5:
                    fund_score += 8
                elif pm > 20:
                    fund_score -= 10
                elif pm > 10:
                    fund_score -= 5

        if "price_trend" in hog:
            pt = hog["price_trend"]
            if pt < -3:
                fund_score += 5
            elif pt > 3:
                fund_score -= 5

    hog_margin_missing = sym == "LH0" and (hog is None or hog.get("profit_margin") is None)
    if hog_margin_missing:
        if range_pct < 5:
            fund_score += 10
            fund_details.append("数据缺失但处于历史极低位(补偿+10)")
        elif range_pct < 15:
            fund_score += 5
            fund_details.append("数据缺失但处于历史低位(补偿+5)")

    return fund_score, fund_details, hog_profit


def score_symbol_legacy(
    sym: str,
    name: str,
    exchange: str,
    df: pd.DataFrame,
    threshold: float = 10.0,
) -> dict | None:
    """
    对单品种计算筛选评分（纯基本面）— 与历史 daily_workflow 版本一致。
    """
    try:
        last, range_pct = compute_range_pct(df)
        layers_score, layers_details, meta = compute_supply_layers(sym, df)
        fund_score = layers_score
        fund_details = list(layers_details)

        hog = get_hog_fundamentals() if sym == "LH0" else None
        hog_profit = None
        if sym == "LH0":
            hs, hdet, hog_profit = _hog_block_legacy(sym, range_pct, hog)
            fund_score += hs
            fund_details.extend(hdet)
        else:
            hog = None

        score = fund_score

        return {
            "symbol": sym,
            "name": name,
            "exchange": exchange,
            "price": last,
            "range_pct": range_pct,
            "score": score,
            "fund_score": fund_score,
            "fund_details": ", ".join(fund_details) if fund_details else "",
            "inv_change_4wk": meta["inv_change_4wk"],
            "inv_percentile": meta["inv_percentile"],
            "receipt_change": meta["receipt_change"],
            "seasonal_signal": meta["seasonal_signal"],
            "hog_profit": hog_profit,
            "fund_threshold": threshold,
            "p1_engine": "legacy",
        }
    except Exception:
        return None
