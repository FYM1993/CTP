#!/usr/bin/env python3
"""
全市场品种筛选器
========================================

自动扫描所有期货品种，发现：
1. 价格处于极端位置（年度高/低附近）的品种
2. 库存持续累积或去化的品种
3. 技术面超卖/超买的品种
4. Wyckoff量价异常的品种

使用方法:
    python scripts/screener.py
    python scripts/screener.py --top 10
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from data_cache import (
    get_all_symbols, prefetch_all, get_inventory, get_request_count,
)
from wyckoff import wyckoff_score


def analyze_symbol(sym: str, name: str, df: pd.DataFrame) -> dict | None:
    """分析单个品种的技术面和价格位置（纯本地计算）"""
    try:
        if len(df) < 60:
            return None

        close = df["close"]
        last = float(close.iloc[-1])

        high_250 = float(close.max())
        low_250 = float(close.min())
        range_pct = (last - low_250) / (high_250 - low_250 + 1e-12) * 100

        pct_from_high = (last / high_250 - 1) * 100
        pct_from_low = (last / low_250 - 1) * 100

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_14 = float((100 - 100 / (1 + gain / (loss + 1e-12))).iloc[-1])

        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma60 = float(close.rolling(min(60, len(close))).mean().iloc[-1])
        ma_trend = (ma5 / ma60 - 1) * 100

        ret = close.pct_change().tail(20)
        volatility = float(ret.std() * np.sqrt(252) * 100)

        oi = pd.to_numeric(df.get("oi", pd.Series(dtype=float)), errors="coerce")
        oi_now = float(oi.iloc[-1]) if not oi.isna().all() else 0
        oi_20ago = float(oi.iloc[-21]) if len(oi) > 20 and not pd.isna(oi.iloc[-21]) else oi_now
        oi_change = (oi_now / (oi_20ago + 1e-12) - 1) * 100

        ret_5d = (last / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0
        ret_20d = (last / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0

        return {
            "symbol": sym, "name": name, "price": last,
            "range_pct": range_pct, "pct_from_high": pct_from_high,
            "pct_from_low": pct_from_low, "rsi_14": rsi_14,
            "ma_trend": ma_trend, "volatility": volatility,
            "oi_change_20d": oi_change,
            "ret_5d": ret_5d, "ret_20d": ret_20d,
        }
    except Exception:
        return None


def compute_score(row: dict, wyckoff_data: dict | None = None) -> float:
    score = 0.0

    range_pct = row["range_pct"]
    if range_pct < 10: score -= 30
    elif range_pct < 20: score -= 20
    elif range_pct < 30: score -= 10
    elif range_pct > 90: score += 30
    elif range_pct > 80: score += 20
    elif range_pct > 70: score += 10

    rsi = row["rsi_14"]
    if rsi < 25: score -= 15
    elif rsi < 35: score -= 8
    elif rsi > 75: score += 15
    elif rsi > 65: score += 8

    ma = row["ma_trend"]
    if ma < -5: score -= 10
    elif ma < -2: score -= 5
    elif ma > 5: score += 10
    elif ma > 2: score += 5

    if "inv_change_4wk" in row and row.get("inv_now") is not None:
        inv_4wk = row["inv_change_4wk"]
        cum_weeks = row.get("inv_cumulating_weeks", 4)
        if inv_4wk > 10: score += 10
        elif inv_4wk > 3: score += 5
        elif inv_4wk < -10: score -= 10
        elif inv_4wk < -3: score -= 5
        if cum_weeks >= 6: score += 10
        elif cum_weeks <= 2: score -= 10

    if wyckoff_data:
        score += np.clip(wyckoff_data.get("composite", 0), -25, 25)

    return score


def main():
    parser = argparse.ArgumentParser(description="期货全市场筛选器")
    parser.add_argument("--top", type=int, default=15, help="显示前N个极端品种")
    args = parser.parse_args()

    now = datetime.now()
    print("╔" + "═" * 58 + "╗")
    print("║" + "全市场品种扫描".center(50) + "║")
    print("║" + f"  {now.strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    print("╚" + "═" * 58 + "╝")

    symbols = get_all_symbols()
    print(f"\n⏳ 数据加载...")
    all_data = prefetch_all(symbols)

    print(f"⏳ 分析中...", flush=True)
    results = []
    for info in symbols:
        sym = info["symbol"]
        df = all_data.get(sym)
        if df is None:
            continue

        data = analyze_symbol(sym, info["name"], df)
        if not data:
            continue

        data["exchange"] = info["exchange"]
        inv = get_inventory(sym)
        if inv:
            data.update(inv)

        wk = None
        try:
            if len(df) > 60:
                wk = wyckoff_score(df)
                data["wyckoff_phase"] = wk.get("phase", "")
                data["wyckoff_composite"] = wk.get("composite", 0)
        except Exception:
            pass

        data["score"] = compute_score(data, wk)
        results.append(data)

    if not results:
        print("❌ 没有获取到任何数据")
        return

    df_results = pd.DataFrame(results)
    phase_icons = {"accumulation": "吸筹", "distribution": "派发", "markup": "上涨", "markdown": "下跌"}

    # 超跌
    oversold = df_results.nsmallest(args.top, "score")
    print(f"\n{'='*80}")
    print(f"  🟢 超跌品种 TOP{args.top} — 潜在做多机会")
    print(f"{'='*80}")
    print(f"{'品种':8s} {'代码':6s} {'价格':>8s} {'区间位':>6s} {'RSI':>5s} {'均线':>6s} {'5日':>6s} {'库存4周':>8s} {'Wyckoff':8s} {'得分':>6s}")
    print("─" * 80)
    for _, row in oversold.iterrows():
        inv_str = f"{row.get('inv_change_4wk', 0):+.1f}%" if pd.notna(row.get("inv_now")) else "  N/A"
        wk_phase = phase_icons.get(row.get("wyckoff_phase", ""), "  -")
        print(f"{row['name']:8s} {row['symbol']:6s} {row['price']:>8.0f} "
              f"{row['range_pct']:>5.0f}% {row['rsi_14']:>5.1f} "
              f"{row['ma_trend']:>+5.1f}% {row['ret_5d']:>+5.1f}% "
              f"{inv_str:>8s} {wk_phase:8s} {row['score']:>+5.0f}")

    # 超涨
    overbought = df_results.nlargest(args.top, "score")
    print(f"\n{'='*80}")
    print(f"  🔴 超涨品种 TOP{args.top} — 潜在做空机会")
    print(f"{'='*80}")
    print(f"{'品种':8s} {'代码':6s} {'价格':>8s} {'区间位':>6s} {'RSI':>5s} {'均线':>6s} {'5日':>6s} {'库存4周':>8s} {'Wyckoff':8s} {'得分':>6s}")
    print("─" * 80)
    for _, row in overbought.iterrows():
        inv_str = f"{row.get('inv_change_4wk', 0):+.1f}%" if pd.notna(row.get("inv_now")) else "  N/A"
        wk_phase = phase_icons.get(row.get("wyckoff_phase", ""), "  -")
        print(f"{row['name']:8s} {row['symbol']:6s} {row['price']:>8.0f} "
              f"{row['range_pct']:>5.0f}% {row['rsi_14']:>5.1f} "
              f"{row['ma_trend']:>+5.1f}% {row['ret_5d']:>+5.1f}% "
              f"{inv_str:>8s} {wk_phase:8s} {row['score']:>+5.0f}")

    # 详情
    extremes = pd.concat([oversold.head(3), overbought.head(3)])
    print(f"\n{'='*70}")
    print(f"  📋 极端品种详情")
    print(f"{'='*70}")
    for _, row in extremes.iterrows():
        direction = "做多" if row["score"] < 0 else "做空"
        icon = "🟢" if row["score"] < 0 else "🔴"
        print(f"\n  {icon} {row['name']} ({row['symbol']}) — 建议{direction}")
        print(f"     价格: {row['price']:.0f}")
        print(f"     年度区间位: {row['range_pct']:.0f}%  (距高点{row['pct_from_high']:+.1f}%, 距低点{row['pct_from_low']:+.1f}%)")
        print(f"     RSI(14): {row['rsi_14']:.1f}  均线趋势: {row['ma_trend']:+.1f}%")
        print(f"     5日涨跌: {row['ret_5d']:+.1f}%  20日涨跌: {row['ret_20d']:+.1f}%")
        if pd.notna(row.get("inv_now")):
            print(f"     库存: {row['inv_now']:.0f}  4周变化: {row['inv_change_4wk']:+.1f}%")
        wk_phase = phase_icons.get(row.get("wyckoff_phase", ""), "未知")
        print(f"     Wyckoff: {wk_phase}阶段  量价评分: {row.get('wyckoff_composite', 0):+.0f}")
        print(f"     综合评分: {row['score']:+.0f}")

    api_total = get_request_count()
    print(f"\n{'='*70}")
    print(f"💡 将感兴趣的品种加入 config.yaml 后运行 pre_market.py")
    print(f"📊 本次API调用: {api_total} 次")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
