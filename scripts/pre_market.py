#!/usr/bin/env python3
"""
盘前分析 - 入场时机判断 & 目标价位预估
========================================

每天盘前运行，输出：
1. 技术面综合评分（做多/做空信号强度）
2. 关键支撑阻力位
3. 斐波那契目标价位
4. 入场/止损/止盈建议

使用方法:
    python scripts/pre_market.py
"""

from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import akshare as aks


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """拉取最新主力合约日线数据"""
    start = pd.Timestamp.now() - pd.Timedelta(days=days * 2)
    df = aks.futures_main_sina(symbol=symbol, start_date=start.strftime("%Y%m%d"))
    df = df.rename(columns={
        "日期": "date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close",
        "成交量": "volume", "持仓量": "oi", "动态结算价": "settle",
    })
    for c in ["open", "high", "low", "close", "volume", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True).tail(days)


# ========== 技术指标计算 ==========

def calc_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()

def calc_ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def calc_atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(w).mean()

def calc_rsi(s: pd.Series, w: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def calc_macd(s: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(s, fast)
    ema_slow = calc_ema(s, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    histogram = (dif - dea) * 2
    return dif, dea, histogram

def calc_bollinger(s: pd.Series, w=20, std=2):
    mid = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return mid + std * sd, mid, mid - std * sd

def find_support_resistance(df: pd.DataFrame, lookback: int = 120, n_levels: int = 5):
    """基于价格密集区和历史高低点计算支撑阻力"""
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    current = closes[-1]

    all_prices = np.concatenate([highs, lows])
    
    # 用核密度估计找到价格密集区
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(all_prices, bw_method=0.05)
        price_range = np.linspace(all_prices.min() * 0.95, all_prices.max() * 1.05, 500)
        density = kde(price_range)
        
        # 找到密度峰值作为关键价位
        from scipy.signal import find_peaks
        peaks, props = find_peaks(density, distance=20, prominence=0.0001)
        key_levels = sorted(price_range[peaks])
    except Exception:
        # fallback: 使用简单分位数
        key_levels = list(np.percentile(all_prices, [10, 25, 50, 75, 90]))

    supports = [p for p in key_levels if p < current]
    resistances = [p for p in key_levels if p > current]

    return supports[-n_levels:], resistances[:n_levels]

def calc_fibonacci(high: float, low: float, direction: str, levels: list[float]):
    """计算斐波那契回撤/扩展目标位"""
    diff = high - low
    targets = {}
    if direction == "long":
        for lvl in levels:
            targets[f"回撤{lvl:.1%}"] = high - diff * lvl
        targets["扩展1.0"] = high
        targets["扩展1.272"] = low + diff * 1.272
        targets["扩展1.618"] = low + diff * 1.618
    else:
        for lvl in levels:
            targets[f"回撤{lvl:.1%}"] = low + diff * lvl
        targets["扩展1.0"] = low
        targets["扩展1.272"] = high - diff * 1.272
        targets["扩展1.618"] = high - diff * 1.618
    return targets


# ========== 信号评分 ==========

def score_signals(df: pd.DataFrame, direction: str, cfg: dict) -> dict:
    """综合评分: -100(强空) ~ +100(强多)"""
    close = df["close"]
    last = close.iloc[-1]
    scores = {}

    # 1. 均线排列 (权重25)
    ma_cfg = cfg.get("ma_windows", [5, 10, 20, 60, 120])
    mas = {w: calc_ma(close, w).iloc[-1] for w in ma_cfg}
    short_mas = [mas[w] for w in ma_cfg[:3] if w in mas]
    long_mas = [mas[w] for w in ma_cfg[3:] if w in mas]
    
    if short_mas and long_mas:
        avg_short = np.mean(short_mas)
        avg_long = np.mean(long_mas)
        ma_score = (avg_short / avg_long - 1) * 1000
        scores["均线排列"] = np.clip(ma_score, -25, 25)
    else:
        scores["均线排列"] = 0

    # 2. MACD (权重20)
    mcfg = cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    
    if hist_now > 0 and hist_now > hist_prev:
        scores["MACD"] = 20
    elif hist_now > 0:
        scores["MACD"] = 10
    elif hist_now < 0 and hist_now < hist_prev:
        scores["MACD"] = -20
    elif hist_now < 0:
        scores["MACD"] = -10
    else:
        scores["MACD"] = 0

    # 3. RSI (权重15)
    rsi_w = cfg.get("rsi_window", 14)
    rsi = calc_rsi(close, rsi_w).iloc[-1]
    if rsi < 30:
        scores["RSI"] = 15    # 超卖 → 利多
    elif rsi > 70:
        scores["RSI"] = -15   # 超买 → 利空
    elif rsi < 45:
        scores["RSI"] = 5
    elif rsi > 55:
        scores["RSI"] = -5
    else:
        scores["RSI"] = 0

    # 4. 布林带位置 (权重15)
    bcfg = cfg.get("bollinger", {})
    upper, mid, lower = calc_bollinger(close, bcfg.get("window", 20), bcfg.get("std", 2))
    bb_pos = (last - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-12)
    scores["布林带"] = np.clip((0.5 - bb_pos) * 30, -15, 15)

    # 5. 成交量趋势 (权重10)
    vol = df["volume"]
    vol_ma5 = calc_ma(vol, 5).iloc[-1]
    vol_ma20 = calc_ma(vol, 20).iloc[-1]
    vol_ratio = vol_ma5 / (vol_ma20 + 1e-12)
    if vol_ratio > 1.5:
        scores["量能"] = 10 if direction == "long" and last > close.iloc[-2] else -10
    elif vol_ratio < 0.7:
        scores["量能"] = -5
    else:
        scores["量能"] = 0

    # 6. 价格动量 (权重15)
    ret_5 = last / close.iloc[-6] - 1
    ret_20 = last / close.iloc[-21] - 1
    momentum = (ret_5 * 0.6 + ret_20 * 0.4) * 1000
    scores["动量"] = np.clip(momentum, -15, 15)

    return scores


def analyze_one(symbol: str, name: str, direction: str, cfg: dict):
    """对单个品种执行完整的盘前分析"""
    print(f"\n{'='*60}")
    dir_icon = "🟢 做多" if direction == "long" else "🔴 做空"
    print(f"  {name} ({symbol})  {dir_icon}")
    print(f"  基本面判断: {cfg.get('reason', '')}")
    print(f"{'='*60}")

    # 拉取数据
    df = fetch_data(symbol, days=400)
    if df.empty:
        print("  ❌ 数据获取失败")
        return

    close = df["close"]
    last = close.iloc[-1]
    prev = close.iloc[-2]
    change_pct = (last / prev - 1) * 100

    print(f"\n  📊 最新行情 ({df['date'].iloc[-1].date()})")
    print(f"     收盘价: {last:.0f}  涨跌: {change_pct:+.2f}%")
    print(f"     成交量: {df['volume'].iloc[-1]:,.0f}  持仓量: {df['oi'].iloc[-1]:,.0f}")

    # 均线
    pre_cfg = cfg
    ma_windows = pre_cfg.get("ma_windows", [5, 10, 20, 60, 120])
    print(f"\n  📈 均线系统")
    for w in ma_windows:
        ma_val = calc_ma(close, w).iloc[-1]
        diff_pct = (last / ma_val - 1) * 100
        above = "▲" if last > ma_val else "▼"
        print(f"     MA{w}: {ma_val:.0f}  ({above} {diff_pct:+.2f}%)")

    # MACD
    mcfg = pre_cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    print(f"\n  📉 MACD")
    print(f"     DIF: {dif.iloc[-1]:.2f}  DEA: {dea.iloc[-1]:.2f}  柱: {hist.iloc[-1]:.2f}")
    if hist.iloc[-1] > hist.iloc[-2]:
        print(f"     状态: 红柱放大 ↑")
    elif hist.iloc[-1] > 0:
        print(f"     状态: 红柱缩小 ↓")
    elif hist.iloc[-1] < hist.iloc[-2]:
        print(f"     状态: 绿柱放大 ↓")
    else:
        print(f"     状态: 绿柱缩小 ↑")

    # RSI
    rsi = calc_rsi(close, pre_cfg.get("rsi_window", 14)).iloc[-1]
    rsi_status = "超卖" if rsi < 30 else "超买" if rsi > 70 else "中性"
    print(f"\n  📊 RSI(14): {rsi:.1f}  ({rsi_status})")

    # ATR
    atr = calc_atr(df, pre_cfg.get("atr_window", 14)).iloc[-1]
    atr_pct = atr / last * 100
    print(f"  📊 ATR(14): {atr:.1f}  ({atr_pct:.2f}%)")

    # 布林带
    bcfg = pre_cfg.get("bollinger", {})
    upper, mid, lower = calc_bollinger(close, bcfg.get("window", 20), bcfg.get("std", 2))
    print(f"\n  📊 布林带")
    print(f"     上轨: {upper.iloc[-1]:.0f}  中轨: {mid.iloc[-1]:.0f}  下轨: {lower.iloc[-1]:.0f}")

    # 支撑阻力
    lookback = pre_cfg.get("sr_lookback", 120)
    supports, resistances = find_support_resistance(df, lookback=lookback)
    print(f"\n  🔑 关键价位 (近{lookback}日)")
    if supports:
        print(f"     支撑位: {', '.join([f'{p:.0f}' for p in reversed(supports)])}")
    if resistances:
        print(f"     阻力位: {', '.join([f'{p:.0f}' for p in resistances])}")

    # 斐波那契目标位
    fib_levels = pre_cfg.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
    recent_high = df.tail(120)["high"].max()
    recent_low = df.tail(120)["low"].min()
    fib_targets = calc_fibonacci(recent_high, recent_low, direction, fib_levels)

    print(f"\n  🎯 斐波那契目标位 (近120日 高:{recent_high:.0f} 低:{recent_low:.0f})")
    if direction == "long":
        for k, v in sorted(fib_targets.items(), key=lambda x: x[1]):
            marker = " ← 当前" if abs(v - last) / last < 0.01 else ""
            print(f"     {k}: {v:.0f}{marker}")
    else:
        for k, v in sorted(fib_targets.items(), key=lambda x: -x[1]):
            marker = " ← 当前" if abs(v - last) / last < 0.01 else ""
            print(f"     {k}: {v:.0f}{marker}")

    # 综合评分
    scores = score_signals(df, direction, pre_cfg)
    total = sum(scores.values())

    print(f"\n  🧮 综合评分")
    for k, v in scores.items():
        bar = "█" * int(abs(v)) + "░" * (25 - int(abs(v)))
        sign = "+" if v > 0 else ""
        print(f"     {k:6s}: {sign}{v:5.1f}  {'🟢' if v > 0 else '🔴' if v < 0 else '⚪'} {bar}")
    print(f"     {'─'*40}")
    print(f"     总分:   {total:+.1f} / 100  ", end="")

    if direction == "long":
        if total > 30:
            print("→ ✅ 强多信号，可考虑入场")
        elif total > 10:
            print("→ 🟡 偏多，等待确认")
        elif total > -10:
            print("→ ⚪ 中性，继续观望")
        else:
            print("→ ❌ 偏空，不宜做多")
    else:
        if total < -30:
            print("→ ✅ 强空信号，可考虑入场")
        elif total < -10:
            print("→ 🟡 偏空，等待确认")
        elif total < 10:
            print("→ ⚪ 中性，继续观望")
        else:
            print("→ ❌ 偏多，不宜做空")

    # 交易建议
    print(f"\n  💡 交易建议")
    if direction == "long":
        entry = max(supports[-1] if supports else last * 0.98, lower.iloc[-1])
        stop = entry - 2 * atr
        tp1 = resistances[0] if resistances else last * 1.05
        tp2 = resistances[1] if len(resistances) > 1 else last * 1.10
        print(f"     方向: 做多")
        print(f"     参考入场: {entry:.0f} (支撑位/布林下轨附近)")
        print(f"     止损位:   {stop:.0f} (入场价 - 2*ATR)")
        print(f"     止盈1:    {tp1:.0f} (第一阻力位)")
        print(f"     止盈2:    {tp2:.0f} (第二阻力位)")
        print(f"     盈亏比:   {(tp1 - entry) / (entry - stop + 1e-12):.2f}")
    else:
        entry = min(resistances[0] if resistances else last * 1.02, upper.iloc[-1])
        stop = entry + 2 * atr
        tp1 = supports[-1] if supports else last * 0.95
        tp2 = supports[-2] if len(supports) > 1 else last * 0.90
        print(f"     方向: 做空")
        print(f"     参考入场: {entry:.0f} (阻力位/布林上轨附近)")
        print(f"     止损位:   {stop:.0f} (入场价 + 2*ATR)")
        print(f"     止盈1:    {tp1:.0f} (第一支撑位)")
        print(f"     止盈2:    {tp2:.0f} (第二支撑位)")
        print(f"     盈亏比:   {(entry - tp1) / (stop - entry + 1e-12):.2f}")


def main():
    config = load_config()
    positions = config["positions"]
    pre_cfg = config["pre_market"]

    print("╔" + "═"*58 + "╗")
    print("║" + "盘前分析报告".center(50) + "║")
    print("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    print("╚" + "═"*58 + "╝")

    for symbol, pos_cfg in positions.items():
        merged_cfg = {**pre_cfg, **pos_cfg}
        analyze_one(
            symbol=symbol,
            name=pos_cfg["name"],
            direction=pos_cfg["direction"],
            cfg=merged_cfg,
        )

    print(f"\n{'='*60}")
    print("⚠️  以上为技术面分析，需结合基本面判断综合决策")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
