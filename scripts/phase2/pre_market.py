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
    PYTHONPATH=scripts python -m phase2.pre_market
"""

from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from shared.ctp_log import get_log_path, get_logger
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND
from data_cache import get_daily
from phase2.direction import choose_phase2_direction as _choose_phase2_direction

log = get_logger("pre_market")
from wyckoff import (
    wyckoff_phase, vsa_scan, analyze_oi, oi_divergence,
    analyze_volume_pattern, detect_climax, detect_spring_upthrust,
    detect_sos_sow, relative_volume, close_position,
    assess_reversal_status, REVERSAL_FRESH_SIGNAL_MAX_DAYS_AGO,
)


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
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
    all_ma_vals = [float(mas[w]) for w in ma_cfg if w in mas and np.isfinite(mas[w])]
    if all_ma_vals:
        # (a) 价格在均线上方/下方的数量: 全在上方=多头, 全在下方=空头
        above_count = sum(1 for m in all_ma_vals if last > m)
        price_vs_ma = (above_count / len(all_ma_vals) - 0.5) * 2  # -1~+1

        # (b) 均线排列方向: 短均 vs 长均
        short_mas = [float(mas[w]) for w in ma_cfg[:3] if w in mas and np.isfinite(mas[w])]
        long_mas = [float(mas[w]) for w in ma_cfg[3:] if w in mas and np.isfinite(mas[w])]
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


def _phase2_min_history(cfg: dict) -> int:
    return max(int(cfg.get("min_history_bars", 60)), 1)


def _score_gate(direction: str, total: float, threshold: float = 20.0) -> bool:
    if direction == "long":
        return total > threshold
    return total < -threshold


def _soft_countertrend_score_gate(direction: str, total: float, threshold: float = 15.0) -> bool:
    if direction == "long":
        return total > -threshold
    return total < threshold


def _directional_value_ok(direction: str, value: float) -> bool:
    if direction == "long":
        return value > 0
    return value < 0


def _reward_risk(direction: str, *, entry: float, target: float, stop: float) -> float:
    if direction == "long":
        return (target - entry) / (entry - stop + 1e-12)
    return (entry - target) / (stop - entry + 1e-12)


def _fresh_reversal_score_gate(direction: str, total: float, reversal: dict) -> bool:
    return bool(_fresh_reversal_gate_details(direction, total, reversal)["score_gate_passed"])


def _fresh_reversal_soft_countertrend_threshold(reversal: dict) -> float:
    signal_type = str(reversal.get("signal_type") or "")
    strength = float(reversal.get("signal_strength") or 0.0)
    threshold = 12.0
    if signal_type in {"SOS", "SOW"}:
        threshold += 8.0
    elif signal_type in {"Spring", "UT"}:
        threshold += 4.0
    if strength >= 0.85:
        threshold += 8.0
    elif strength >= 0.75:
        threshold += 4.0
    return threshold


def _fresh_reversal_gate_details(direction: str, total: float, reversal: dict) -> dict[str, float | int | bool]:
    signal_type = str(reversal.get("signal_type") or "")
    signal_days_ago_value = reversal.get("signal_days_ago")
    signal_days_ago = int(signal_days_ago_value) if signal_days_ago_value is not None else (0 if reversal.get("has_signal") else None)
    strength = float(reversal.get("signal_strength") or 0.0)
    details: dict[str, float | int | bool] = {
        "fresh_signal": False,
        "score_gate_passed": False,
        "signal_days_ago": signal_days_ago if signal_days_ago is not None else -1,
        "signal_strength": strength,
        "soft_countertrend_threshold": 0.0,
    }
    if not reversal.get("has_signal"):
        return details
    if signal_type not in {"Spring", "SOS", "UT", "SOW"}:
        return details
    if signal_days_ago is None or signal_days_ago > REVERSAL_FRESH_SIGNAL_MAX_DAYS_AGO:
        return details
    details["fresh_signal"] = True
    if strength < 0.65:
        return details
    if _score_gate(direction, total):
        details["score_gate_passed"] = True
        return details
    threshold = _fresh_reversal_soft_countertrend_threshold(reversal)
    details["soft_countertrend_threshold"] = float(threshold)
    details["score_gate_passed"] = bool(_soft_countertrend_score_gate(direction, total, threshold=threshold))
    return details


def _reversal_signal_is_fresh(reversal_status: dict) -> bool:
    signal_days_ago_value = reversal_status.get("signal_days_ago")
    if not reversal_status.get("has_signal"):
        return False
    if signal_days_ago_value is None:
        return True
    return int(signal_days_ago_value) <= REVERSAL_FRESH_SIGNAL_MAX_DAYS_AGO


def _trend_hold_gate(direction: str, total: float, trend_status: dict) -> bool:
    if not trend_status.get("phase_ok"):
        return False
    if trend_status.get("slope_ok") or trend_status.get("trend_indicator_ok"):
        return True
    return _soft_countertrend_score_gate(direction, total, threshold=10.0)


def _build_reversal_candidate_plan(
    *,
    direction: str,
    last: float,
    atr: float,
    total: float,
    reversal: dict,
    supports: list[float],
    resistances: list[float],
    fib_low: float,
    fib_high: float,
    fib_range: float,
) -> dict | None:
    if not reversal.get("has_signal"):
        return None
    gate_details = _fresh_reversal_gate_details(direction, total, reversal)

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

    rr = _reward_risk(direction, entry=entry, target=tp1, stop=stop)
    admission_rr = _reward_risk(direction, entry=entry, target=tp2, stop=stop)

    rr_ok = admission_rr >= 1.0
    score_ok = bool(gate_details["score_gate_passed"])
    return {
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "admission_rr": float(admission_rr),
        "actionable": bool(score_ok and rr_ok),
        "phase2_score_gate_passed": score_ok,
        "phase2_rr_gate_passed": rr_ok,
        "reversal_signal_fresh": bool(gate_details["fresh_signal"]),
        "reversal_signal_days_ago": int(gate_details["signal_days_ago"]),
        "reversal_signal_strength": float(gate_details["signal_strength"]),
        "entry_family": "reversal",
        "entry_signal_type": reversal.get("signal_type", ""),
        "entry_signal_detail": reversal.get("signal_detail", ""),
    }


def _collect_trend_ma_context(close: pd.Series, cfg: dict) -> tuple[dict[int, float], dict[int, float]]:
    trend_windows = cfg.get("trend_ma_windows") or cfg.get("ma_windows", [5, 10, 20])
    ma_values: dict[int, float] = {}
    ma_slopes: dict[int, float] = {}

    for window in trend_windows[:3]:
        ma_series = calc_ma(close, int(window)).dropna()
        if ma_series.empty:
            continue
        ma_values[int(window)] = float(ma_series.iloc[-1])
        anchor_idx = max(len(ma_series) - min(5, len(ma_series)), 0)
        ma_slopes[int(window)] = float(ma_series.iloc[-1] - ma_series.iloc[anchor_idx])

    return ma_values, ma_slopes


def _assess_trend_continuation(
    *,
    close: pd.Series,
    last: float,
    direction: str,
    scores: dict,
    total: float,
    atr: float,
    wk_phase: str,
    cfg: dict,
) -> dict:
    ma_values, ma_slopes = _collect_trend_ma_context(close, cfg)
    ma_items = sorted(ma_values.items())
    fast_ma = ma_items[0][1] if ma_items else last
    aligned_slopes = [s for s in ma_slopes.values() if not np.isnan(s)]
    trend_quality_values = {
        "均线排列": float(scores.get("均线排列", 0.0)),
        "MACD": float(scores.get("MACD", 0.0)),
        "动量": float(scores.get("动量", 0.0)),
    }

    if direction == "long":
        slope_ok = bool(aligned_slopes) and all(s > 0 for s in aligned_slopes)
        phase_ok = wk_phase in {"markup", "accumulation"}
        signal_type = "TrendBreak" if last >= fast_ma + atr else "Pullback"
        trend_bias = "bullish"
    else:
        slope_ok = bool(aligned_slopes) and all(s < 0 for s in aligned_slopes)
        phase_ok = wk_phase in {"markdown", "distribution"}
        signal_type = "TrendBreak" if last <= fast_ma - atr else "Pullback"
        trend_bias = "bearish"

    trend_indicator_ok = all(_directional_value_ok(direction, value) for value in trend_quality_values.values())
    score_ok = _score_gate(direction, total)
    has_signal = bool(score_ok and slope_ok and phase_ok and trend_indicator_ok)
    signal_detail = (
        f"{wk_phase} 阶段，评分{total:+.0f}，"
        f"均线斜率与{trend_bias}方向一致，当前按{signal_type}处理"
    ) if has_signal else ""

    return {
        "has_signal": has_signal,
        "signal_type": signal_type if has_signal else "",
        "signal_detail": signal_detail,
        "score_ok": score_ok,
        "trend_indicator_ok": trend_indicator_ok,
        "trend_quality_values": trend_quality_values,
        "phase_ok": phase_ok,
        "slope_ok": slope_ok,
        "phase": wk_phase,
        "ma_values": ma_values,
        "ma_slopes": ma_slopes,
    }


def _build_trend_candidate_plan(
    *,
    direction: str,
    last: float,
    atr: float,
    total: float,
    trend_status: dict,
    supports: list[float],
    resistances: list[float],
    fib_low: float,
    fib_high: float,
    fib_range: float,
) -> dict | None:
    if not trend_status.get("has_signal"):
        return None

    ma_values = list(trend_status.get("ma_values", {}).values())
    signal_type = trend_status.get("signal_type", "")
    entry = last

    if direction == "long":
        below_price = [v for v in ma_values if v < entry]
        if signal_type == "TrendBreak":
            stop = max(entry - atr, max(below_price) - 0.5 * atr) if below_price else entry - atr
        else:
            anchor = max(below_price) if below_price else entry - 0.5 * atr
            stop = anchor - 0.5 * atr

        tp1_sr = next((r for r in resistances if r > entry + 0.5 * atr), None)
        tp1_fib = fib_low + fib_range * 0.618
        if tp1_fib is not None and tp1_fib <= entry:
            tp1_fib = None
        tp1_candidates = [v for v in [tp1_sr, tp1_fib] if v is not None]
        tp1 = min(tp1_candidates) if tp1_candidates else (resistances[0] if resistances else entry * 1.03)

        tp2_sr = next((r for r in resistances if r > tp1 + 0.5 * atr), None)
        tp2_fib = fib_low + fib_range * 1.0
        if tp2_fib is not None and tp2_fib <= tp1:
            tp2_fib = None
        tp2_candidates = [v for v in [tp2_sr, tp2_fib] if v is not None]
        tp2 = min(tp2_candidates) if tp2_candidates else tp1 * 1.02

    else:
        above_price = [v for v in ma_values if v > entry]
        if signal_type == "TrendBreak":
            stop = min(entry + atr, min(above_price) + 0.5 * atr) if above_price else entry + atr
        else:
            anchor = min(above_price) if above_price else entry + 0.5 * atr
            stop = anchor + 0.5 * atr

        tp1_sr = next((s for s in reversed(supports) if s < entry - 0.5 * atr), None)
        tp1_fib = fib_high - fib_range * 0.618
        if tp1_fib is not None and tp1_fib >= entry:
            tp1_fib = None
        tp1_candidates = [v for v in [tp1_sr, tp1_fib] if v is not None]
        tp1 = max(tp1_candidates) if tp1_candidates else (supports[-1] if supports else entry * 0.97)

        tp2_sr = next((s for s in reversed(supports) if s < tp1 - 0.5 * atr), None)
        tp2_fib = fib_high - fib_range * 1.0
        if tp2_fib is not None and tp2_fib >= tp1:
            tp2_fib = None
        tp2_candidates = [v for v in [tp2_sr, tp2_fib] if v is not None]
        tp2 = max(tp2_candidates) if tp2_candidates else tp1 * 0.98

    rr = _reward_risk(direction, entry=entry, target=tp1, stop=stop)
    admission_rr = _reward_risk(direction, entry=entry, target=tp2, stop=stop)

    rr_ok = admission_rr >= 1.0
    score_ok = _score_gate(direction, total)
    return {
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "admission_rr": float(admission_rr),
        "actionable": bool(score_ok and rr_ok),
        "phase2_score_gate_passed": bool(score_ok),
        "phase2_rr_gate_passed": bool(rr_ok),
        "entry_family": "trend",
        "entry_signal_type": signal_type,
        "entry_signal_detail": trend_status.get("signal_detail", ""),
    }


def assess_active_trend_hold_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
) -> dict | None:
    """Evaluate whether an active trend trade should remain valid on the next day."""
    if df is None or df.empty:
        return None
    if len(df) < _phase2_min_history(cfg):
        return None

    close = df["close"]
    last = float(close.iloc[-1])
    atr = float(calc_atr(df, cfg.get("atr_window", 14)).iloc[-1])
    scores = score_signals(df, direction, cfg)
    total = float(sum(scores.values()))
    phase_info = wyckoff_phase(df)
    trend_status = _assess_trend_continuation(
        close=close,
        last=last,
        direction=direction,
        scores=scores,
        total=total,
        atr=atr,
        wk_phase=phase_info.phase,
        cfg=cfg,
    )
    hold_valid = _trend_hold_gate(direction, total, trend_status)
    return {
        "hold_valid": hold_valid,
        "score": total,
        "trend_status": trend_status,
    }


def assess_active_reversal_hold_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
) -> dict | None:
    """Evaluate whether an active reversal trade should remain valid on the next day."""
    if df is None or df.empty:
        return None
    if len(df) < _phase2_min_history(cfg):
        return None

    ctx = _build_plan_context(direction=direction, df=df, cfg=cfg)
    reversal = assess_reversal_status(df, direction, lookback=60)
    trend_status = _assess_trend_continuation(
        close=ctx["close"],
        last=float(ctx["last"]),
        direction=direction,
        scores=ctx["scores"],
        total=float(ctx["total"]),
        atr=float(ctx["atr"]),
        wk_phase=str(ctx["wk_phase"]),
        cfg=cfg,
    )
    trend_followthrough = bool(
        trend_status.get("phase_ok")
        and trend_status.get("slope_ok")
        and trend_status.get("trend_indicator_ok")
    )
    return {
        "hold_valid": bool(reversal.get("has_signal") or trend_followthrough),
        "score": float(ctx["total"]),
        "reversal_status": reversal,
        "trend_status": trend_status,
    }


def _choose_trade_candidate(candidates: list[dict | None]) -> dict | None:
    available = [candidate for candidate in candidates if candidate]
    if not available:
        return None
    return max(available, key=lambda item: (bool(item.get("actionable")), float(item.get("rr", 0.0))))


def _build_plan_context(*, direction: str, df: pd.DataFrame, cfg: dict) -> dict[str, object]:
    if df is None or df.empty:
        return {}
    if len(df) < _phase2_min_history(cfg):
        return {}

    close = df["close"]
    last = float(close.iloc[-1])

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

    atr = float(calc_atr(df, cfg.get("atr_window", 14)).iloc[-1])
    return {
        "close": close,
        "last": last,
        "scores": scores,
        "total": total,
        "classical_total": float(classical_total),
        "wyckoff_total": float(wyckoff_total),
        "oi_total": float(oi_total),
        "reason": ", ".join(reason_parts) if reason_parts else "综合中性",
        "wk_phase": wk_phase,
        "fib_high": float(fib_high),
        "fib_low": float(fib_low),
        "fib_range": float(fib_range),
        "fib_targets": fib_targets,
        "supports": supports,
        "resistances": resistances,
        "atr": atr,
    }


def _build_plan_payload(
    *,
    symbol: str,
    name: str,
    direction: str,
    ctx: dict[str, object],
    candidate: dict | None,
    strategy_family: str,
    reversal_status: dict,
    trend_status: dict,
) -> dict:
    if candidate:
        entry = float(candidate["entry"])
        stop = float(candidate["stop"])
        tp1 = float(candidate["tp1"])
        tp2 = float(candidate["tp2"])
        rr = float(candidate["rr"])
        admission_rr = float(candidate.get("admission_rr", rr))
        actionable = bool(candidate["actionable"])
        phase2_score_gate_passed = bool(candidate.get("phase2_score_gate_passed"))
        phase2_rr_gate_passed = bool(candidate.get("phase2_rr_gate_passed"))
        reversal_signal_fresh = bool(candidate.get("reversal_signal_fresh"))
        entry_family = str(candidate.get("entry_family") or "")
        entry_signal_type = str(candidate.get("entry_signal_type") or "")
        entry_signal_detail = str(candidate.get("entry_signal_detail") or "")
        strategy_value = strategy_family
    else:
        entry = float(ctx["last"])
        stop = 0.0
        tp1 = 0.0
        tp2 = 0.0
        rr = 0.0
        admission_rr = 0.0
        actionable = False
        phase2_score_gate_passed = False
        phase2_rr_gate_passed = False
        reversal_signal_fresh = _reversal_signal_is_fresh(reversal_status)
        entry_family = ""
        entry_signal_type = ""
        entry_signal_detail = ""
        strategy_value = ""

    return {
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "strategy_family": strategy_value,
        "score": float(ctx["total"]),
        "actionable": actionable,
        "price": float(ctx["last"]),
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "admission_rr": admission_rr,
        "phase2_score_gate_passed": phase2_score_gate_passed,
        "phase2_rr_gate_passed": phase2_rr_gate_passed,
        "reversal_signal_fresh": reversal_signal_fresh,
        "classical_score": float(ctx["classical_total"]),
        "wyckoff_score": float(ctx["wyckoff_total"]),
        "oi_score": float(ctx["oi_total"]),
        "wyckoff_phase": str(ctx["wk_phase"]),
        "reason": str(ctx["reason"]),
        "entry_family": entry_family,
        "entry_signal_type": entry_signal_type,
        "entry_signal_detail": entry_signal_detail,
        "reversal_status": reversal_status,
        "trend_status": trend_status,
        "support_levels": list(ctx["supports"]),
        "resistance_levels": list(ctx["resistances"]),
        "fib_targets": dict(ctx["fib_targets"]),
        "scores": dict(ctx["scores"]),
    }


def build_reversal_trade_plan_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
    allow_watch_plan: bool = False,
) -> dict | None:
    """Build a reversal-only Phase 2 trade plan from daily bars."""
    if df is None or df.empty:
        return None
    if len(df) < _phase2_min_history(cfg):
        return None

    ctx = _build_plan_context(direction=direction, df=df, cfg=cfg)
    reversal = assess_reversal_status(df, direction, lookback=60)
    trend_status = _assess_trend_continuation(
        close=ctx["close"],
        last=float(ctx["last"]),
        direction=direction,
        scores=ctx["scores"],
        total=float(ctx["total"]),
        atr=float(ctx["atr"]),
        wk_phase=str(ctx["wk_phase"]),
        cfg=cfg,
    )
    reversal_candidate = _build_reversal_candidate_plan(
        direction=direction,
        last=float(ctx["last"]),
        atr=float(ctx["atr"]),
        total=float(ctx["total"]),
        reversal=reversal,
        supports=ctx["supports"],
        resistances=ctx["resistances"],
        fib_low=float(ctx["fib_low"]),
        fib_high=float(ctx["fib_high"]),
        fib_range=float(ctx["fib_range"]),
    )
    if reversal_candidate is None:
        if not allow_watch_plan:
            return None
        return _build_plan_payload(
            symbol=symbol,
            name=name,
            direction=direction,
            ctx=ctx,
            candidate=None,
            strategy_family=STRATEGY_REVERSAL,
            reversal_status=reversal,
            trend_status=trend_status,
        )
    return _build_plan_payload(
        symbol=symbol,
        name=name,
        direction=direction,
        ctx=ctx,
        candidate=reversal_candidate,
        strategy_family=STRATEGY_REVERSAL,
        reversal_status=reversal,
        trend_status=trend_status,
    )


def build_trend_trade_plan_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
    allow_watch_plan: bool = False,
) -> dict | None:
    """Build a trend-only Phase 2 trade plan from daily bars."""
    if df is None or df.empty:
        return None
    if len(df) < _phase2_min_history(cfg):
        return None

    ctx = _build_plan_context(direction=direction, df=df, cfg=cfg)
    reversal = assess_reversal_status(df, direction, lookback=60)
    trend_status = _assess_trend_continuation(
        close=ctx["close"],
        last=float(ctx["last"]),
        direction=direction,
        scores=ctx["scores"],
        total=float(ctx["total"]),
        atr=float(ctx["atr"]),
        wk_phase=str(ctx["wk_phase"]),
        cfg=cfg,
    )
    trend_candidate = _build_trend_candidate_plan(
        direction=direction,
        last=float(ctx["last"]),
        atr=float(ctx["atr"]),
        total=float(ctx["total"]),
        trend_status=trend_status,
        supports=ctx["supports"],
        resistances=ctx["resistances"],
        fib_low=float(ctx["fib_low"]),
        fib_high=float(ctx["fib_high"]),
        fib_range=float(ctx["fib_range"]),
    )
    if trend_candidate is None:
        if not allow_watch_plan:
            return None
        return _build_plan_payload(
            symbol=symbol,
            name=name,
            direction=direction,
            ctx=ctx,
            candidate=None,
            strategy_family=STRATEGY_TREND,
            reversal_status=reversal,
            trend_status=trend_status,
        )
    return _build_plan_payload(
        symbol=symbol,
        name=name,
        direction=direction,
        ctx=ctx,
        candidate=trend_candidate,
        strategy_family=STRATEGY_TREND,
        reversal_status=reversal,
        trend_status=trend_status,
    )


def build_trade_plan_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame | None,
    cfg: dict,
) -> dict | None:
    """Build the legacy mixed Phase 2 trade plan from an already-fetched daily DataFrame."""
    if df is None or df.empty:
        return None
    if len(df) < _phase2_min_history(cfg):
        return None

    reversal_plan = build_reversal_trade_plan_from_daily_df(
        symbol=symbol,
        name=name,
        direction=direction,
        df=df,
        cfg=cfg,
    )
    trend_plan = build_trend_trade_plan_from_daily_df(
        symbol=symbol,
        name=name,
        direction=direction,
        df=df,
        cfg=cfg,
    )
    selected_plan = _choose_trade_candidate([reversal_plan, trend_plan])
    if selected_plan is not None:
        return selected_plan

    ctx = _build_plan_context(direction=direction, df=df, cfg=cfg)
    reversal = assess_reversal_status(df, direction, lookback=60)
    trend_status = _assess_trend_continuation(
        close=ctx["close"],
        last=float(ctx["last"]),
        direction=direction,
        scores=ctx["scores"],
        total=float(ctx["total"]),
        atr=float(ctx["atr"]),
        wk_phase=str(ctx["wk_phase"]),
        cfg=cfg,
    )
    return _build_plan_payload(
        symbol=symbol,
        name=name,
        direction=direction,
        ctx=ctx,
        candidate=None,
        strategy_family="",
        reversal_status=reversal,
        trend_status=trend_status,
    )


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

    # ==================== 第一部分: 经典技术指标 ====================
    log.info(f"\n  ┌─ 经典技术指标 ─────────────────────────────┐")
    ma_windows = cfg.get("ma_windows", [5, 10, 20, 60, 120])
    log.info(f"  │ 均线系统:")
    for w in ma_windows:
        ma_val = calc_ma(close, w).iloc[-1]
        diff_pct = (last / ma_val - 1) * 100
        above = "▲" if last > ma_val else "▼"
        log.info(f"  │   MA{w}: {ma_val:.0f}  ({above} {diff_pct:+.2f}%)")

    mcfg = cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_status = ("红柱放大 ↑" if hist.iloc[-1] > hist.iloc[-2] else "红柱缩小 ↓") \
        if hist.iloc[-1] > 0 else ("绿柱放大 ↓" if hist.iloc[-1] < hist.iloc[-2] else "绿柱缩小 ↑")
    log.info(f"  │ MACD: DIF={dif.iloc[-1]:.2f} DEA={dea.iloc[-1]:.2f} 柱={hist.iloc[-1]:.2f} [{hist_status}]")

    rsi = calc_rsi(close, cfg.get("rsi_window", 14)).iloc[-1]
    rsi_status = "超卖🔥" if rsi < 30 else "超买⚠️" if rsi > 70 else "中性"
    log.info(f"  │ RSI(14): {rsi:.1f}  ({rsi_status})")

    atr = calc_atr(df, cfg.get("atr_window", 14)).iloc[-1]
    atr_pct = atr / last * 100
    log.info(f"  │ ATR(14): {atr:.1f}  ({atr_pct:.2f}%)")

    bcfg = cfg.get("bollinger", {})
    upper, mid, lower = calc_bollinger(close, bcfg.get("window", 20), bcfg.get("std", 2))
    log.info(f"  │ 布林带: {lower.iloc[-1]:.0f} / {mid.iloc[-1]:.0f} / {upper.iloc[-1]:.0f}")
    log.info(f"  └────────────────────────────────────────────┘")

    # ==================== 第二部分: Wyckoff 量价分析 ====================
    log.info(f"\n  ┌─ Wyckoff 量价分析 ─────────────────────────┐")
    phase = wyckoff_phase(df, lookback=120)
    phase_icon = {
        "accumulation": "🟢吸筹", "markup": "🔵上涨",
        "distribution": "🔴派发", "markdown": "🟡下跌",
    }.get(phase.phase, "⚪")
    log.info(f"  │ 市场阶段: {phase_icon} (置信度{phase.confidence:.0%})")
    if getattr(phase, "description", ""):
        log.info(f"  │   {phase.description}")

    vp = analyze_volume_pattern(df, lookback=60)
    ratio = vp["up_down_ratio"]
    if ratio > 1.3:
        vp_label = "多方主导 🟢"
    elif ratio < 0.77:
        vp_label = "空方主导 🔴"
    else:
        vp_label = "均衡"
    log.info(f"  │ 量价关系: 上涨日量/下跌日量 = {ratio:.2f} ({vp_label})")
    log.info(f"  │   量能趋势(20日/40日): {vp['vol_trend']:.2f}  量价相关性: {vp['price_vol_corr']:+.2f}")

    vsa_bars = vsa_scan(df, window=20)
    recent_notable = [b for b in vsa_bars[-10:] if b.strength >= 1]
    if recent_notable:
        log.info(f"  │ VSA信号 (近10日):")
        for bar in recent_notable[-3:]:
            bias_icon = "🟢" if bar.bias == "bullish" else "🔴" if bar.bias == "bearish" else "⚪"
            log.info(f"  │   {bias_icon} [{bar.bar_type}] {bar.description}")
    else:
        log.info(f"  │ VSA信号: 近期无显著量价信号")

    events = detect_climax(df, 60) + detect_spring_upthrust(df, 60) + detect_sos_sow(df, 60)
    if events:
        log.info(f"  │ Wyckoff事件 (近60日):")
        for evt in events[-3:]:
            evt_icon = "🟢" if evt["bias"] == "bullish" else "🔴"
            log.info(f"  │   {evt_icon} {evt['date']} {evt['type']}")
            log.info(f"  │     {evt['detail']}")
    log.info(f"  └────────────────────────────────────────────┘")

    # ==================== 第三部分: 持仓量分析 ====================
    log.info(f"\n  ┌─ 持仓量(OI)分析 ──────────────────────────┐")
    oi_sig = analyze_oi(df, window=5)
    if oi_sig:
        oi_icon = "🟢" if getattr(oi_sig, "bias", "") == "bullish" else "🔴"
        log.info(f"  │ {oi_icon} {oi_sig.label} ({oi_sig.strength})")
        log.info(f"  │   {oi_sig.description}")
    else:
        log.info(f"  │ 无持仓数据")

    oi_div = oi_divergence(df, window=20)
    if oi_div:
        log.info(f"  │ {oi_div}")
    log.info(f"  └────────────────────────────────────────────┘")

    # ==================== 第四部分: 支撑阻力 & 目标位 ====================
    lookback = cfg.get("sr_lookback", 120)
    supports = plan["support_levels"]
    resistances = plan["resistance_levels"]
    log.info(f"\n  🔑 关键价位 (近{lookback}日)")
    if supports:
        log.info(f"     支撑位: {', '.join([f'{p:.0f}' for p in reversed(supports)])}")
    if resistances:
        log.info(f"     阻力位: {', '.join([f'{p:.0f}' for p in resistances])}")

    fib_targets = plan["fib_targets"]
    recent_high = df.tail(120)["high"].max()
    recent_low = df.tail(120)["low"].min()
    log.info(f"\n  🎯 斐波那契目标位 (近120日 高:{recent_high:.0f} 低:{recent_low:.0f})")
    if direction == "long":
        ordered_targets = sorted(fib_targets.items(), key=lambda x: x[1])
    else:
        ordered_targets = sorted(fib_targets.items(), key=lambda x: -x[1])
    for k, v in ordered_targets:
        marker = " ← 当前" if abs(v - last) / last < 0.01 else ""
        log.info(f"     {k}: {v:.0f}{marker}")

    # ==================== 第五部分: 综合评分 ====================
    scores = plan["scores"]
    total = plan["score"]
    log.info(f"\n  🧮 综合评分 (经典指标55分 + 量价分析35分 + 持仓15分)")
    section_labels = {
        "均线排列": "经典", "MACD": "经典", "RSI": "经典", "布林带": "经典", "动量": "经典",
        "价格位置": "经典",
        "Wyckoff阶段": "量价", "量价关系": "量价", "VSA信号": "量价",
        "持仓信号": "持仓",
    }
    for k, v in scores.items():
        section = section_labels.get(k, "")
        bar_len = min(int(abs(v)), 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        sign = "+" if v > 0 else ""
        icon = "🟢" if v > 0 else "🔴" if v < 0 else "⚪"
        log.info(f"     {section:2s} {k:8s}: {sign}{v:5.1f}  {icon} {bar}")
    log.info(f"     {'─'*45}")
    if direction == "long":
        if total > 30:
            score_hint = "→ ✅ 强多信号，可考虑入场"
        elif total > 10:
            score_hint = "→ 🟡 偏多，等待确认"
        elif total > -10:
            score_hint = "→ ⚪ 中性，继续观望"
        else:
            score_hint = "→ ❌ 偏空，不宜做多"
    else:
        if total < -30:
            score_hint = "→ ✅ 强空信号，可考虑入场"
        elif total < -10:
            score_hint = "→ 🟡 偏空，等待确认"
        elif total < 10:
            score_hint = "→ ⚪ 中性，继续观望"
        else:
            score_hint = "→ ❌ 偏多，不宜做空"
    log.info("     总分:   %+.1f / 105  %s", total, score_hint)

    reversal = plan["reversal_status"]
    trend = plan.get("trend_status", {})
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
    entry_family = plan.get("entry_family", "")
    entry_signal_type = plan.get("entry_signal_type", "")
    entry_signal_detail = plan.get("entry_signal_detail", "")

    if reversal.get("suspect_events"):
        log.info(f"  │ ⚠️ 疑似信号（未达标，仅供参考）:")
        for evt in reversal["suspect_events"][-3:]:
            log.info(f"  │   ⚪ {evt['date']} {evt['signal']}: {evt['detail']}")

    if entry_family == "trend":
        status_icon = "✅" if plan["actionable"] else "🟡"
        log.info(f"     {status_icon} 顺势入场: {entry_signal_type}")
        if entry_signal_detail:
            log.info(f"     信号详情: {entry_signal_detail}")
        log.info(f"     参考入场: {plan['entry']:.0f} (当前价)")
        stop_basis = "趋势回踩失效位" if direction == "long" else "趋势反弹失效位"
        log.info(f"     止损位:   {plan['stop']:.0f} ({stop_basis})")
        log.info(f"     止盈1:    {plan['tp1']:.0f}")
        log.info(f"     止盈2:    {plan['tp2']:.0f}")
        log.info(f"     第一止盈盈亏比: {plan['rr']:.2f}")
        log.info(f"     准入盈亏比:     {plan.get('admission_rr', plan['rr']):.2f}")
        if not plan["actionable"]:
            if not plan.get("phase2_rr_gate_passed", False):
                admission_rr = plan.get("admission_rr", plan["rr"])
                log.info(f"  ⚠️ 顺势信号存在，但准入盈亏比({admission_rr:.2f})不足1.0，谨慎入场")
            if (direction == "long" and plan["score"] <= 20) or (direction == "short" and plan["score"] >= -20):
                log.info(f"  ⚠️ 顺势信号存在，但评分({plan['score']:+.0f})未达标，谨慎入场")
    elif reversal["has_signal"]:
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
        log.info(f"     第一止盈盈亏比: {plan['rr']:.2f}")
        log.info(f"     准入盈亏比:     {plan.get('admission_rr', plan['rr']):.2f}")
        log.info(f"     置信度:   {reversal['confidence']:.0%}")

        if not plan["actionable"]:
            if not plan.get("phase2_rr_gate_passed", False):
                admission_rr = plan.get("admission_rr", plan["rr"])
                log.info(f"  ⚠️ 有入场信号但准入盈亏比({admission_rr:.2f})不足1.0，谨慎入场")
            if (direction == "long" and plan["score"] <= 20) or (direction == "short" and plan["score"] >= -20):
                log.info(f"  ⚠️ 有入场信号但评分({plan['score']:+.0f})未达标，谨慎入场")
    else:
        log.info(f"     ⏳ 尚无有效反转信号，暂不建议入场")
        log.info(f"     当前阶段: {reversal['current_stage']}")
        log.info(f"     等待信号: {reversal['next_expected']}")
        if trend.get("phase"):
            log.info(f"     趋势状态: {trend['phase']}")
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
