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

from ctp_log import get_log_path, get_logger
from data_cache import get_daily
from phase2_direction import choose_phase2_direction as _choose_phase2_direction

log = get_logger("pre_market")
from wyckoff import (
    wyckoff_phase, vsa_scan, analyze_oi, oi_divergence,
    analyze_volume_pattern, detect_climax, detect_spring_upthrust,
    detect_sos_sow, relative_volume, close_position,
    assess_reversal_status,
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
    """基于多维度寻找有意义的支撑阻力位，合并过近的价位"""
    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values
    current = closes[-1]
    atr = calc_atr(df).iloc[-1]
    min_gap = atr * 1.5  # 价位之间至少间隔 1.5×ATR

    # --- 多来源收集关键价位候选 ---
    candidates = []

    # 1) 局部极值（近期高低点转折）
    from scipy.signal import argrelextrema
    order = max(3, lookback // 20)
    local_highs_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    local_lows_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    for i in local_highs_idx:
        candidates.append(("pivot_h", highs[i]))
    for i in local_lows_idx:
        candidates.append(("pivot_l", lows[i]))

    # 2) KDE 价格密集区
    from scipy.stats import gaussian_kde
    try:
        all_prices = np.concatenate([highs, lows])
        kde = gaussian_kde(all_prices, bw_method=0.08)
        price_range = np.linspace(all_prices.min() * 0.95, all_prices.max() * 1.05, 500)
        density = kde(price_range)
        from scipy.signal import find_peaks as fp
        peaks, _ = fp(density, distance=30, prominence=0.0001)
        for p in peaks:
            candidates.append(("kde", price_range[p]))
    except Exception:
        pass

    # 3) 整数关口（心理价位）
    magnitude = 10 ** max(0, int(np.log10(current)) - 1)
    round_step = magnitude
    lo = current * 0.85
    hi = current * 1.15
    r = lo - (lo % round_step)
    while r <= hi:
        if abs(r - current) > min_gap * 0.5:
            candidates.append(("round", r))
        r += round_step

    # 4) 历史成交量加权价格（VWAP 近似）
    if "volume" in recent.columns and recent["volume"].sum() > 0:
        vwap = (recent["close"] * recent["volume"]).sum() / recent["volume"].sum()
        candidates.append(("vwap", vwap))

    # 5) 长周期关键位（250日高低点、中位数）
    long_df = df.tail(250)
    candidates.append(("long_high", long_df["high"].max()))
    candidates.append(("long_low", long_df["low"].min()))
    candidates.append(("long_mid", (long_df["high"].max() + long_df["low"].min()) / 2))

    # --- 合并过近的价位 ---
    raw_levels = sorted(set(v for _, v in candidates))
    merged = []
    for lvl in raw_levels:
        if merged and abs(lvl - merged[-1]) < min_gap:
            merged[-1] = (merged[-1] + lvl) / 2
        else:
            merged.append(lvl)

    supports = [p for p in merged if p < current - atr * 0.3]
    resistances = [p for p in merged if p > current + atr * 0.3]

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

    评分体系 (总计 ±105):
      经典指标 (55分):
        均线排列  15分  — 趋势方向
        MACD     10分  — 趋势动能
        RSI      10分  — 超买超卖
        布林带    10分  — 价格位置
        动量       5分  — 短期惯性
        价格位置   5分  — 历史高低点间的位置上下文
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

    # ===== 经典指标 (55分) =====

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

    # 6. 价格位置 (5) — 从 Phase 1 迁入，作为上下文而非逆势信号
    n = min(len(df), 300)
    recent_prices = df.tail(n)["close"]
    high_all = float(recent_prices.max())
    low_all = float(recent_prices.min())
    range_pct = (last - low_all) / (high_all - low_all + 1e-12) * 100
    if range_pct < 10:
        scores["价格位置"] = 5
    elif range_pct > 90:
        scores["价格位置"] = -5
    else:
        scores["价格位置"] = 0

    # ===== Wyckoff 量价分析 (35分) =====

    # 7. 阶段判断 (15)
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

    # 8. 量价关系 (10) — 上涨日成交量 vs 下跌日成交量
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

    # 9. VSA 最近K线 (10)
    vsa_bars = vsa_scan(df, window=20)
    recent_vsa = vsa_bars[-5:]
    bullish_strength = sum(b.strength for b in recent_vsa if b.bias == "bullish")
    bearish_strength = sum(b.strength for b in recent_vsa if b.bias == "bearish")
    vsa_net = bullish_strength - bearish_strength
    scores["VSA信号"] = np.clip(vsa_net * 3, -10, 10)

    # ===== 期货持仓分析 (15分) =====

    # 10. OI 信号 (15)
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


def resolve_phase2_direction(*, long_score: float, short_score: float, delta: float = 12.0) -> str:
    """Thin seam for future phase 2 direction selection."""
    return _choose_phase2_direction(long_score=long_score, short_score=short_score, delta=delta)


def build_trade_plan_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
) -> dict | None:
    """Build the Phase 2 trade plan from an already-fetched daily DataFrame."""
    if df is None or df.empty:
        return None

    close = df["close"]
    last = close.iloc[-1]

    scores = score_signals(df, direction, cfg)
    total = float(sum(scores.values()))

    classical_keys = ["均线排列", "MACD", "RSI", "布林带", "动量", "价格位置"]
    wyckoff_keys = ["Wyckoff阶段", "量价关系", "VSA信号"]
    oi_keys = ["持仓信号"]
    classical_total = sum(scores.get(k, 0) for k in classical_keys)
    wyckoff_total = sum(scores.get(k, 0) for k in wyckoff_keys)
    oi_total = sum(scores.get(k, 0) for k in oi_keys)
    top_factors = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    reason_parts = [f"{k}{v:+.0f}" for k, v in top_factors[:3] if abs(v) >= 3]

    phase_info = wyckoff_phase(df)
    wk_phase = phase_info.phase

    fib_high = df.tail(120)["high"].max()
    fib_low = df.tail(120)["low"].min()
    fib_range = fib_high - fib_low

    lookback = cfg.get("sr_lookback", 120)
    supports, resistances = find_support_resistance(df, lookback=lookback)

    fib_levels = cfg.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
    recent_high = df.tail(120)["high"].max()
    recent_low = df.tail(120)["low"].min()
    fib_targets = calc_fibonacci(recent_high, recent_low, direction, fib_levels)

    atr = calc_atr(df, cfg.get("atr_window", 14)).iloc[-1]
    reversal = assess_reversal_status(df, direction, lookback=60)

    if reversal["has_signal"]:
        sig_bar = reversal["signal_bar"]
        entry = last

        if direction == "long":
            stop = sig_bar["low"] - 0.5 * atr

            tp1_sr = next((r for r in resistances if r > entry + 0.5 * atr), None)
            tp1_fib = fib_low + fib_range * 0.618
            if tp1_fib is not None and tp1_fib <= entry:
                tp1_fib = None
            tp1_candidates = [v for v in [tp1_sr, tp1_fib] if v is not None]
            tp1 = min(tp1_candidates) if tp1_candidates else (resistances[0] if resistances else last * 1.03)

            tp2_sr = next((r for r in resistances if r > tp1 + 0.5 * atr), None)
            tp2_fib = fib_low + fib_range * 1.0
            if tp2_fib is not None and tp2_fib <= tp1:
                tp2_fib = None
            tp2_candidates = [v for v in [tp2_sr, tp2_fib] if v is not None]
            tp2 = min(tp2_candidates) if tp2_candidates else tp1 * 1.02

            rr = (tp1 - entry) / (entry - stop + 1e-12)
        else:
            stop = sig_bar["high"] + 0.5 * atr

            tp1_sr = next((s for s in reversed(supports) if s < entry - 0.5 * atr), None)
            tp1_fib = fib_high - fib_range * 0.618
            if tp1_fib is not None and tp1_fib >= entry:
                tp1_fib = None
            tp1_candidates = [v for v in [tp1_sr, tp1_fib] if v is not None]
            tp1 = max(tp1_candidates) if tp1_candidates else (supports[-1] if supports else last * 0.97)

            tp2_sr = next((s for s in reversed(supports) if s < tp1 - 0.5 * atr), None)
            tp2_fib = fib_high - fib_range * 1.0
            if tp2_fib is not None and tp2_fib >= tp1:
                tp2_fib = None
            tp2_candidates = [v for v in [tp2_sr, tp2_fib] if v is not None]
            tp2 = max(tp2_candidates) if tp2_candidates else tp1 * 0.98

            rr = (entry - tp1) / (stop - entry + 1e-12)

        score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
        rr_ok = rr >= 1.0
        actionable = score_ok and rr_ok
    else:
        entry, stop, tp1, tp2, rr = last, 0.0, 0.0, 0.0, 0.0
        actionable = False

    return {
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "score": float(total),
        "actionable": actionable,
        "price": float(last),
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "classical_score": float(classical_total),
        "wyckoff_score": float(wyckoff_total),
        "oi_score": float(oi_total),
        "wyckoff_phase": wk_phase,
        "reason": ", ".join(reason_parts) if reason_parts else "综合中性",
        "reversal_status": reversal,
        "support_levels": supports,
        "resistance_levels": resistances,
        "fib_targets": fib_targets,
        "scores": scores,
    }


def analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict | None:
    """
    对单个品种执行完整的盘前分析；长篇输出写入当前运行的日志文件。

    返回: {"symbol", "name", "direction", "score", "actionable", "entry", "stop", "tp1"} 或 None
    """
    log.info(f"\n{'='*60}")
    dir_icon = "🟢 做多" if direction == "long" else "🔴 做空"
    log.info(f"  {name} ({symbol})  {dir_icon}")
    log.info(f"  基本面判断: {cfg.get('reason', '')}")
    log.info(f"{'='*60}")

    # 拉取数据
    df = fetch_data(symbol, days=400)
    if df.empty:
        log.info("  ❌ 数据获取失败")
        return None

    close = df["close"]
    last = close.iloc[-1]
    prev = close.iloc[-2]
    change_pct = (last / prev - 1) * 100

    log.info(f"\n  📊 最新行情 ({df['date'].iloc[-1].date()})")
    log.info(f"     收盘价: {last:.0f}  涨跌: {change_pct:+.2f}%")
    log.info(f"     成交量: {df['volume'].iloc[-1]:,.0f}  持仓量: {df['oi'].iloc[-1]:,.0f}")

    plan = build_trade_plan_from_daily_df(
        symbol=symbol,
        name=name,
        direction=direction,
        df=df,
        cfg=cfg,
    )
    if plan is None:
        log.info("  ❌ 数据获取失败")
        return None

    reversal = plan["reversal_status"]
    log.info(f"\n  ┌─ 反转信号评估 ──────────────────────────────┐")
    log.info(f"  │ 当前阶段: {reversal['current_stage']}")
    log.info(f"  │ 下一步:   {reversal['next_expected']}")
    if reversal["all_events"]:
        log.info(f"  │ 近期事件:")
        for evt in reversal["all_events"][-3:]:
            icon = "🟢" if evt["bias"] == "bullish" else "🔴"
            log.info(f"  │   {icon} {evt['date']} {evt['signal']}: {evt['detail']}")
    else:
        log.info(f"  │ 近期无相关反转事件")
    log.info(f"  └────────────────────────────────────────────┘")

    log.info(f"\n  💡 交易建议")
    dir_cn = "做多" if direction == "long" else "做空"
    log.info(f"     方向: {dir_cn}")

    if reversal.get("suspect_events"):
        log.info(f"  │ ⚠️ 疑似信号（未达标，仅供参考）:")
        for evt in reversal["suspect_events"][-3:]:
            log.info(f"  │   ⚪ {evt['date']} {evt['signal']}: {evt['detail']}")

    if reversal["has_signal"]:
        sig = reversal["signal_type"]
        sig_date = reversal["signal_date"]
        sig_cn = {"Spring": "弹簧", "SOS": "强势突破", "SC": "卖方高潮",
                  "UT": "上冲回落", "SOW": "弱势跌破", "BC": "买方高潮",
                  "StopVol_Bull": "停止量(多)", "StopVol_Bear": "停止量(空)"}.get(sig, sig)

        log.info(f"     ✅ 入场信号: {sig_cn} ({sig_date})")
        log.info(f"     信号详情: {reversal['signal_detail']}")
        log.info(f"     参考入场: {plan['entry']:.0f} (当前价)")
        stop_basis = "信号K线最低 - 0.5ATR" if direction == "long" else "信号K线最高 + 0.5ATR"
        log.info(f"     止损位:   {plan['stop']:.0f} ({stop_basis})")
        log.info(f"     止盈1:    {plan['tp1']:.0f}")
        log.info(f"     止盈2:    {plan['tp2']:.0f}")
        log.info(f"     盈亏比:   {plan['rr']:.2f}")
        log.info(f"     置信度:   {reversal['confidence']:.0%}")

        if not plan["actionable"]:
            if plan["rr"] < 1.0:
                log.info(f"  ⚠️ 有入场信号但盈亏比({plan['rr']:.2f})不足1.0，谨慎入场")
            if (direction == "long" and plan["score"] <= 20) or (direction == "short" and plan["score"] >= -20):
                log.info(f"  ⚠️ 有入场信号但评分({plan['score']:+.0f})未达标，谨慎入场")
    else:
        log.info(f"     ⏳ 尚无有效反转信号，暂不建议入场")
        log.info(f"     当前阶段: {reversal['current_stage']}")
        log.info(f"     等待信号: {reversal['next_expected']}")
        if direction == "long":
            log.info(f"     入场条件: 出现Spring(假跌破收回)或SOS(放量突破)后再评估")
        else:
            log.info(f"     入场条件: 出现UT(假突破回落)或SOW(放量跌破)后再评估")

    return plan


def main():
    config = load_config()
    positions = config["positions"]
    pre_cfg = config["pre_market"]

    log.info("╔" + "═"*58 + "╗")
    log.info("║" + "盘前分析报告".center(50) + "║")
    log.info("║" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    log.info("╚" + "═"*58 + "╝")

    for symbol, pos_cfg in positions.items():
        merged_cfg = {**pre_cfg, **pos_cfg}
        analyze_one(
            symbol=symbol,
            name=pos_cfg["name"],
            direction=pos_cfg["direction"],
            cfg=merged_cfg,
        )

    log.info(f"\n{'='*60}")
    log.info("⚠️  以上为技术面分析，需结合基本面判断综合决策")
    log.info(f"{'='*60}")
    print(f"盘前分析详情已写入 {get_log_path()}")


if __name__ == "__main__":
    main()
