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

from data_cache import get_daily
from wyckoff import (
    wyckoff_phase, vsa_scan, analyze_oi, oi_divergence,
    analyze_volume_pattern, detect_climax, detect_spring_upthrust,
    detect_sos_sow, relative_volume, close_position,
)


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """通过缓存层获取日线数据"""
    df = get_daily(symbol, days)
    return df if df is not None else pd.DataFrame()


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
    """
    综合评分: -100(强空) ~ +100(强多)

    评分体系 (总计 ±100):
      经典指标 (50分):
        均线排列  15分  — 趋势方向
        MACD     10分  — 趋势动能
        RSI      10分  — 超买超卖
        布林带    10分  — 价格位置
        动量       5分  — 短期惯性
      Wyckoff 量价 (35分):
        阶段判断  15分  — 吸筹/派发/上涨/下跌
        量价关系  10分  — 上涨日量 vs 下跌日量
        VSA信号  10分  — 最近K线的量价行为
      期货持仓 (15分):
        OI信号   15分  — 价格+持仓量四象限
    """
    close = df["close"]
    last = close.iloc[-1]
    scores = {}

    # ===== 经典指标 (50分) =====

    # 1. 均线排列 (15) — 综合价格位置和均线排列
    ma_cfg = cfg.get("ma_windows", [5, 10, 20, 60, 120])
    mas = {w: calc_ma(close, w).iloc[-1] for w in ma_cfg}
    all_ma_vals = [mas[w] for w in ma_cfg if w in mas]
    if all_ma_vals:
        # (a) 价格在均线上方/下方的数量: 全在上方=多头, 全在下方=空头
        above_count = sum(1 for m in all_ma_vals if last > m)
        price_vs_ma = (above_count / len(all_ma_vals) - 0.5) * 2  # -1~+1

        # (b) 均线排列方向: 短均 vs 长均
        short_mas = [mas[w] for w in ma_cfg[:3] if w in mas]
        long_mas = [mas[w] for w in ma_cfg[3:] if w in mas]
        if short_mas and long_mas:
            alignment = np.sign(np.mean(short_mas) / np.mean(long_mas) - 1)
        else:
            alignment = 0

        # 价格位置权重70%、排列方向权重30%
        # 价格跌破所有均线时，即使短均还在长均上方也给负分
        raw = price_vs_ma * 0.7 + alignment * 0.3
        scores["均线排列"] = np.clip(raw * 15, -15, 15)
    else:
        scores["均线排列"] = 0

    # 2. MACD (10)
    mcfg = cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    if hist_now > 0 and hist_now > hist_prev:
        scores["MACD"] = 10
    elif hist_now > 0:
        scores["MACD"] = 5
    elif hist_now < 0 and hist_now < hist_prev:
        scores["MACD"] = -10
    elif hist_now < 0:
        scores["MACD"] = -5
    else:
        scores["MACD"] = 0

    # 3. RSI (10)
    rsi_w = cfg.get("rsi_window", 14)
    rsi = calc_rsi(close, rsi_w).iloc[-1]
    if rsi < 25:
        scores["RSI"] = 10
    elif rsi < 35:
        scores["RSI"] = 5
    elif rsi > 75:
        scores["RSI"] = -10
    elif rsi > 65:
        scores["RSI"] = -5
    else:
        scores["RSI"] = 0

    # 4. 布林带位置 (10)
    bcfg = cfg.get("bollinger", {})
    upper, mid, lower = calc_bollinger(close, bcfg.get("window", 20), bcfg.get("std", 2))
    bb_pos = (last - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-12)
    scores["布林带"] = np.clip((0.5 - bb_pos) * 20, -10, 10)

    # 5. 动量 (5)
    ret_5 = last / close.iloc[-6] - 1
    ret_20 = last / close.iloc[-21] - 1
    momentum = (ret_5 * 0.6 + ret_20 * 0.4) * 500
    scores["动量"] = np.clip(momentum, -5, 5)

    # ===== Wyckoff 量价分析 (35分) =====

    # 6. 阶段判断 (15)
    phase = wyckoff_phase(df, lookback=120)
    if phase.phase == "accumulation":
        scores["Wyckoff阶段"] = 15 * phase.confidence
    elif phase.phase == "markup":
        scores["Wyckoff阶段"] = 8 * phase.confidence
    elif phase.phase == "distribution":
        scores["Wyckoff阶段"] = -15 * phase.confidence
    elif phase.phase == "markdown":
        scores["Wyckoff阶段"] = -8 * phase.confidence
    else:
        scores["Wyckoff阶段"] = 0

    # 7. 量价关系 (10) — 上涨日成交量 vs 下跌日成交量
    vp = analyze_volume_pattern(df, lookback=60)
    ratio = vp["up_down_ratio"]
    if ratio > 1.5:
        scores["量价关系"] = 10
    elif ratio > 1.2:
        scores["量价关系"] = 5
    elif ratio < 0.67:
        scores["量价关系"] = -10
    elif ratio < 0.83:
        scores["量价关系"] = -5
    else:
        scores["量价关系"] = 0

    # 8. VSA 最近K线 (10)
    vsa_bars = vsa_scan(df, window=20)
    recent_vsa = vsa_bars[-5:]
    bullish_strength = sum(b.strength for b in recent_vsa if b.bias == "bullish")
    bearish_strength = sum(b.strength for b in recent_vsa if b.bias == "bearish")
    vsa_net = bullish_strength - bearish_strength
    scores["VSA信号"] = np.clip(vsa_net * 3, -10, 10)

    # ===== 期货持仓分析 (15分) =====

    # 9. OI 信号 (15)
    oi_sig = analyze_oi(df, window=5)
    if oi_sig:
        if oi_sig.pattern == "new_long":
            scores["持仓信号"] = 15
        elif oi_sig.pattern == "short_covering":
            scores["持仓信号"] = 5
        elif oi_sig.pattern == "new_short":
            scores["持仓信号"] = -15
        elif oi_sig.pattern == "long_liquidation":
            scores["持仓信号"] = -5
        else:
            scores["持仓信号"] = 0
    else:
        scores["持仓信号"] = 0

    return scores


def analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict | None:
    """
    对单个品种执行完整的盘前分析

    返回: {"symbol", "name", "direction", "score", "actionable", "entry", "stop", "tp1"} 或 None
    """
    print(f"\n{'='*60}")
    dir_icon = "🟢 做多" if direction == "long" else "🔴 做空"
    print(f"  {name} ({symbol})  {dir_icon}")
    print(f"  基本面判断: {cfg.get('reason', '')}")
    print(f"{'='*60}")

    # 拉取数据
    df = fetch_data(symbol, days=400)
    if df.empty:
        print("  ❌ 数据获取失败")
        return None

    close = df["close"]
    last = close.iloc[-1]
    prev = close.iloc[-2]
    change_pct = (last / prev - 1) * 100

    print(f"\n  📊 最新行情 ({df['date'].iloc[-1].date()})")
    print(f"     收盘价: {last:.0f}  涨跌: {change_pct:+.2f}%")
    print(f"     成交量: {df['volume'].iloc[-1]:,.0f}  持仓量: {df['oi'].iloc[-1]:,.0f}")

    pre_cfg = cfg

    # ==================== 第一部分: 经典技术指标 ====================
    print(f"\n  ┌─ 经典技术指标 ─────────────────────────────┐")

    # 均线
    ma_windows = pre_cfg.get("ma_windows", [5, 10, 20, 60, 120])
    print(f"  │ 均线系统:")
    for w in ma_windows:
        ma_val = calc_ma(close, w).iloc[-1]
        diff_pct = (last / ma_val - 1) * 100
        above = "▲" if last > ma_val else "▼"
        print(f"  │   MA{w}: {ma_val:.0f}  ({above} {diff_pct:+.2f}%)")

    # MACD
    mcfg = pre_cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_status = ("红柱放大 ↑" if hist.iloc[-1] > hist.iloc[-2] else "红柱缩小 ↓") \
        if hist.iloc[-1] > 0 else ("绿柱放大 ↓" if hist.iloc[-1] < hist.iloc[-2] else "绿柱缩小 ↑")
    print(f"  │ MACD: DIF={dif.iloc[-1]:.2f} DEA={dea.iloc[-1]:.2f} 柱={hist.iloc[-1]:.2f} [{hist_status}]")

    # RSI
    rsi = calc_rsi(close, pre_cfg.get("rsi_window", 14)).iloc[-1]
    rsi_status = "超卖🔥" if rsi < 30 else "超买⚠️" if rsi > 70 else "中性"
    print(f"  │ RSI(14): {rsi:.1f}  ({rsi_status})")

    # ATR
    atr = calc_atr(df, pre_cfg.get("atr_window", 14)).iloc[-1]
    atr_pct = atr / last * 100
    print(f"  │ ATR(14): {atr:.1f}  ({atr_pct:.2f}%)")

    # 布林带
    bcfg = pre_cfg.get("bollinger", {})
    upper, mid, lower = calc_bollinger(close, bcfg.get("window", 20), bcfg.get("std", 2))
    print(f"  │ 布林带: {lower.iloc[-1]:.0f} / {mid.iloc[-1]:.0f} / {upper.iloc[-1]:.0f}")
    print(f"  └────────────────────────────────────────────┘")

    # ==================== 第二部分: Wyckoff 量价分析 ====================
    print(f"\n  ┌─ Wyckoff 量价分析 ─────────────────────────┐")

    # 阶段判断
    phase = wyckoff_phase(df, lookback=120)
    phase_icon = {
        "accumulation": "🟢吸筹", "markup": "🔵上涨",
        "distribution": "🔴派发", "markdown": "🟡下跌",
    }.get(phase.phase, "⚪")
    print(f"  │ 市场阶段: {phase_icon} (置信度{phase.confidence:.0%})")
    print(f"  │   {phase.description}")

    # 量价关系
    vp = analyze_volume_pattern(df, lookback=60)
    ratio = vp["up_down_ratio"]
    if ratio > 1.3:
        vp_label = "多方主导 🟢"
    elif ratio < 0.77:
        vp_label = "空方主导 🔴"
    else:
        vp_label = "均衡"
    print(f"  │ 量价关系: 上涨日量/下跌日量 = {ratio:.2f} ({vp_label})")
    print(f"  │   量能趋势(20日/40日): {vp['vol_trend']:.2f}  量价相关性: {vp['price_vol_corr']:+.2f}")

    # VSA 最近K线分析
    vsa_bars = vsa_scan(df, window=20)
    recent_notable = [b for b in vsa_bars[-10:] if b.strength >= 1]
    if recent_notable:
        print(f"  │ VSA信号 (近10日):")
        for bar in recent_notable[-3:]:
            bias_icon = "🟢" if bar.bias == "bullish" else "🔴" if bar.bias == "bearish" else "⚪"
            print(f"  │   {bias_icon} [{bar.bar_type}] {bar.description}")
    else:
        print(f"  │ VSA信号: 近期无显著量价信号")

    # 关键事件
    events = detect_climax(df, 60) + detect_spring_upthrust(df, 60) + detect_sos_sow(df, 60)
    if events:
        print(f"  │ Wyckoff事件 (近60日):")
        for evt in events[-3:]:
            evt_icon = "🟢" if evt["bias"] == "bullish" else "🔴"
            print(f"  │   {evt_icon} {evt['date']} {evt['type']}")
            print(f"  │     {evt['detail']}")

    print(f"  └────────────────────────────────────────────┘")

    # ==================== 第三部分: 持仓量分析 ====================
    print(f"\n  ┌─ 持仓量(OI)分析 ──────────────────────────┐")
    oi_sig = analyze_oi(df, window=5)
    if oi_sig:
        oi_icon = "🟢" if oi_sig.bias == "bullish" else "🔴"
        print(f"  │ {oi_icon} {oi_sig.label} ({oi_sig.strength})")
        print(f"  │   {oi_sig.description}")
    else:
        print(f"  │ 无持仓数据")

    oi_div = oi_divergence(df, window=20)
    if oi_div:
        print(f"  │ {oi_div}")
    print(f"  └────────────────────────────────────────────┘")

    # ==================== 第四部分: 支撑阻力 & 目标位 ====================

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

    print(f"\n  🧮 综合评分 (经典指标50分 + 量价分析35分 + 持仓15分)")
    section_labels = {
        "均线排列": "经典", "MACD": "经典", "RSI": "经典", "布林带": "经典", "动量": "经典",
        "Wyckoff阶段": "量价", "量价关系": "量价", "VSA信号": "量价",
        "持仓信号": "持仓",
    }
    for k, v in scores.items():
        section = section_labels.get(k, "")
        bar_len = int(abs(v))
        bar = "█" * bar_len + "░" * (15 - bar_len)
        sign = "+" if v > 0 else ""
        icon = "🟢" if v > 0 else "🔴" if v < 0 else "⚪"
        print(f"     {section:2s} {k:8s}: {sign}{v:5.1f}  {icon} {bar}")
    print(f"     {'─'*45}")
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
        rr = (tp1 - entry) / (entry - stop + 1e-12)
        print(f"     方向: 做多")
        print(f"     参考入场: {entry:.0f} (支撑位/布林下轨附近)")
        print(f"     止损位:   {stop:.0f} (入场价 - 2*ATR)")
        print(f"     止盈1:    {tp1:.0f} (第一阻力位)")
        print(f"     止盈2:    {tp2:.0f} (第二阻力位)")
        print(f"     盈亏比:   {rr:.2f}")
    else:
        entry = min(resistances[0] if resistances else last * 1.02, upper.iloc[-1])
        stop = entry + 2 * atr
        tp1 = supports[-1] if supports else last * 0.95
        tp2 = supports[-2] if len(supports) > 1 else last * 0.90
        rr = (entry - tp1) / (stop - entry + 1e-12)
        print(f"     方向: 做空")
        print(f"     参考入场: {entry:.0f} (阻力位/布林上轨附近)")
        print(f"     止损位:   {stop:.0f} (入场价 + 2*ATR)")
        print(f"     止盈1:    {tp1:.0f} (第一支撑位)")
        print(f"     止盈2:    {tp2:.0f} (第二支撑位)")
        print(f"     盈亏比:   {rr:.2f}")

    # 是否可操作：评分方向一致 且 |总分| > 20 且 盈亏比 > 1
    score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
    rr_ok = rr >= 1.0
    actionable = score_ok and rr_ok
    if score_ok and not rr_ok:
        print(f"  ⚠️ 评分达标但盈亏比({rr:.2f})不足1.0，暂不建议入场")

    # 构建评分原因摘要
    classical_keys = ["均线排列", "MACD", "RSI", "布林带", "动量"]
    wyckoff_keys = ["Wyckoff阶段", "量价关系", "VSA信号"]
    oi_keys = ["持仓信号"]
    classical_total = sum(scores.get(k, 0) for k in classical_keys)
    wyckoff_total = sum(scores.get(k, 0) for k in wyckoff_keys)
    oi_total = sum(scores.get(k, 0) for k in oi_keys)

    top_factors = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    reason_parts = [f"{k}{v:+.0f}" for k, v in top_factors[:3] if abs(v) >= 3]

    # Wyckoff 阶段
    phase_info = wyckoff_phase(df)
    wk_phase = phase_info.phase

    return {
        "symbol": symbol, "name": name, "direction": direction,
        "score": float(total), "actionable": actionable,
        "price": float(last),
        "entry": float(entry), "stop": float(stop),
        "tp1": float(tp1), "tp2": float(tp2),
        "rr": float(rr),
        "classical_score": float(classical_total),
        "wyckoff_score": float(wyckoff_total),
        "oi_score": float(oi_total),
        "wyckoff_phase": wk_phase,
        "reason": ", ".join(reason_parts) if reason_parts else "综合中性",
    }


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
