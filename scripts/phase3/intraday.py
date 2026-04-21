#!/usr/bin/env python3
"""
日内策略监控
========================================

盘中实时拉取分钟级数据，基于技术指标发出交易信号。

策略逻辑:
  做多品种 (如生猪):
    - 布林带下轨反弹 + RSI超卖回升 → 开多
    - 突破N日高点 + 放量 → 加仓
    - 跌破止损 或 触及布林上轨 → 平仓

  做空品种 (如多晶硅):
    - 布林带上轨回落 + RSI超买回落 → 开空
    - 跌破N日低点 + 放量 → 加仓
    - 涨破止损 或 触及布林下轨 → 平仓

使用方法:
    # 实时监控（每分钟刷新）
    PYTHONPATH=scripts python -m phase3.intraday

    # 只跑一次分析
    PYTHONPATH=scripts python -m phase3.intraday --once

    # 指定K线周期
    PYTHONPATH=scripts python -m phase3.intraday --period 15
"""

import time
import argparse
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import pandas as pd

from pathlib import Path

from shared.ctp_log import get_log_path, get_logger
from data_cache import get_minute

log = get_logger("intraday")
from wyckoff import vsa_scan, classify_vsa_bar, relative_volume, close_position
from phase3.live import TqPhase3Monitor, tqsdk_configured


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_minute_data(symbol: str, period: str = "5") -> pd.DataFrame:
    """通过缓存层获取分钟数据"""
    df = get_minute(symbol, period)
    return df if df is not None else pd.DataFrame()


# ========== 指标 ==========

def calc_ma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()

def calc_ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def calc_rsi(s: pd.Series, w: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def calc_atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(w).mean()

def calc_macd(s: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(s, fast)
    ema_slow = calc_ema(s, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist

def calc_bollinger(s: pd.Series, w=20, std_n=2):
    mid = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return mid + std_n * sd, mid, mid - std_n * sd

def calc_kdj(df: pd.DataFrame, n=9, m1=3, m2=3):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-12) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


# ========== 信号生成 ==========

def generate_signals(df: pd.DataFrame, direction: str, cfg: dict) -> list[dict]:
    """根据技术指标生成日内交易信号"""
    signals = []
    close = df["close"]
    volume = df["volume"]
    last_idx = len(df) - 1
    last = close.iloc[-1]

    bb_cfg = cfg.get("mean_reversion", {})
    bb_w = bb_cfg.get("bb_window", 20)
    bb_std = bb_cfg.get("bb_std", 2)
    rsi_os = bb_cfg.get("rsi_oversold", 30)
    rsi_ob = bb_cfg.get("rsi_overbought", 70)

    upper, mid, lower = calc_bollinger(close, bb_w, bb_std)
    rsi = calc_rsi(close, 14)
    atr = calc_atr(df, 14)
    vol_ma = calc_ma(volume, 20)
    k, d, j = calc_kdj(df)
    dif, dea, hist = calc_macd(close)

    vol_filter = cfg.get("volume_filter", 1.5)

    # 取最近标量值
    r = float(rsi.iloc[-1])
    r_prev = float(rsi.iloc[-2])
    u = float(upper.iloc[-1])
    m = float(mid.iloc[-1])
    lo = float(lower.iloc[-1])
    atr_val = float(atr.iloc[-1])
    vol_now = float(volume.iloc[-1])
    vol_avg = float(vol_ma.iloc[-1]) if not np.isnan(vol_ma.iloc[-1]) else 1.0
    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    k_now = float(k.iloc[-1])
    d_now = float(d.iloc[-1])
    j_now = float(j.iloc[-1])
    k_prev = float(k.iloc[-2])
    d_prev = float(d.iloc[-2])
    j_prev = float(j.iloc[-2])
    dif_now = float(dif.iloc[-1])
    close_prev = float(close.iloc[-2])
    mid_prev = float(mid.iloc[-2])

    if direction == "long":
        # 信号1: 布林下轨反弹 + RSI从超卖回升
        if last <= lo * 1.005 and r < 35 and r > r_prev:
            signals.append({
                "type": "开多",
                "strength": "强" if r < rsi_os else "中",
                "reason": f"触及布林下轨({lo:.0f}), RSI={r:.1f}↑ 超卖回升",
                "entry": last,
                "stop": last - 2 * atr_val,
                "target": m,
            })

        # 信号2: MACD金叉
        if hist_now > 0 and hist_prev <= 0:
            signals.append({
                "type": "开多",
                "strength": "中",
                "reason": f"MACD金叉, DIF={dif_now:.1f}上穿DEA",
                "entry": last,
                "stop": last - 2 * atr_val,
                "target": u,
            })

        # 信号3: KDJ金叉 + J值从超卖区域拐头
        if j_now > d_now and j_prev <= d_prev and j_now < 30:
            signals.append({
                "type": "开多",
                "strength": "中",
                "reason": f"KDJ金叉(J={j_now:.0f}), 低位金叉",
                "entry": last,
                "stop": last - 1.5 * atr_val,
                "target": m,
            })

        # 信号4: 放量突破均线
        if last > m and close_prev <= mid_prev and vol_now > vol_avg * vol_filter:
            signals.append({
                "type": "加多",
                "strength": "强",
                "reason": f"放量突破布林中轨({m:.0f}), 量比={vol_now/vol_avg:.1f}",
                "entry": last,
                "stop": m - atr_val,
                "target": u,
            })

        # 平仓信号
        if last >= u * 0.995:
            signals.append({
                "type": "平多",
                "strength": "中",
                "reason": f"触及布林上轨({u:.0f}), 考虑止盈",
            })

        if r > rsi_ob:
            signals.append({
                "type": "平多",
                "strength": "弱",
                "reason": f"RSI={r:.1f} 超买, 注意回落风险",
            })

    else:  # short
        # 信号1: 布林上轨受阻 + RSI从超买回落
        if last >= u * 0.995 and r > 65 and r < r_prev:
            signals.append({
                "type": "开空",
                "strength": "强" if r > rsi_ob else "中",
                "reason": f"触及布林上轨({u:.0f}), RSI={r:.1f}↓ 超买回落",
                "entry": last,
                "stop": last + 2 * atr_val,
                "target": m,
            })

        # 信号2: MACD死叉
        if hist_now < 0 and hist_prev >= 0:
            signals.append({
                "type": "开空",
                "strength": "中",
                "reason": f"MACD死叉, DIF={dif_now:.1f}下穿DEA",
                "entry": last,
                "stop": last + 2 * atr_val,
                "target": lo,
            })

        # 信号3: KDJ死叉 + J值从超买区域拐头
        if j_now < d_now and j_prev >= d_prev and j_now > 70:
            signals.append({
                "type": "开空",
                "strength": "中",
                "reason": f"KDJ死叉(J={j_now:.0f}), 高位死叉",
                "entry": last,
                "stop": last + 1.5 * atr_val,
                "target": m,
            })

        # 信号4: 放量跌破均线
        if last < m and close_prev >= mid_prev and vol_now > vol_avg * vol_filter:
            signals.append({
                "type": "加空",
                "strength": "强",
                "reason": f"放量跌破布林中轨({m:.0f}), 量比={vol_now/vol_avg:.1f}",
                "entry": last,
                "stop": m + atr_val,
                "target": lo,
            })

        # 平仓信号
        if last <= lo * 1.005:
            signals.append({
                "type": "平空",
                "strength": "中",
                "reason": f"触及布林下轨({lo:.0f}), 考虑止盈",
            })

        if r < rsi_os:
            signals.append({
                "type": "平空",
                "strength": "弱",
                "reason": f"RSI={r:.1f} 超卖, 注意反弹风险",
            })

    # ===== VSA 量价信号 =====
    vsa_bars = vsa_scan(df, window=20)
    last_vsa = vsa_bars[-1] if vsa_bars else None
    prev_vsa = vsa_bars[-2] if len(vsa_bars) >= 2 else None

    if last_vsa and last_vsa.strength >= 2:
        if direction == "long" and last_vsa.bias == "bullish":
            signals.append({
                "type": "开多",
                "strength": "强" if last_vsa.strength >= 3 else "中",
                "reason": f"VSA [{last_vsa.bar_type}] {last_vsa.description}",
                "entry": last,
                "stop": last - 2 * atr_val,
                "target": u,
            })
        elif direction == "long" and last_vsa.bias == "bearish" and last_vsa.strength >= 2:
            signals.append({
                "type": "平多",
                "strength": "中",
                "reason": f"VSA [{last_vsa.bar_type}] {last_vsa.description}",
            })
        elif direction == "short" and last_vsa.bias == "bearish":
            signals.append({
                "type": "开空",
                "strength": "强" if last_vsa.strength >= 3 else "中",
                "reason": f"VSA [{last_vsa.bar_type}] {last_vsa.description}",
                "entry": last,
                "stop": last + 2 * atr_val,
                "target": lo,
            })
        elif direction == "short" and last_vsa.bias == "bullish" and last_vsa.strength >= 2:
            signals.append({
                "type": "平空",
                "strength": "中",
                "reason": f"VSA [{last_vsa.bar_type}] {last_vsa.description}",
            })

    return signals


def print_dashboard(symbol: str, name: str, direction: str, df: pd.DataFrame, cfg: dict):
    """打印单品种日内面板"""
    close = df["close"]
    last = close.iloc[-1]
    last_time = df["datetime"].iloc[-1]
    day_open = df.iloc[0]["open"]
    day_high = df["high"].max()
    day_low = df["low"].min()
    day_change = (last / day_open - 1) * 100

    upper, mid, lower = calc_bollinger(close, 20, 2)
    rsi = calc_rsi(close, 14).iloc[-1]
    atr = calc_atr(df, 14).iloc[-1]
    k, d, j = calc_kdj(df)
    dif, dea, hist = calc_macd(close)

    dir_icon = "🟢多" if direction == "long" else "🔴空"

    log.info(f"\n┌─── {name}({symbol}) {dir_icon} ── {last_time.strftime('%H:%M')} ───┐")
    log.info(f"│ 价格: {last:.0f}  涨跌: {day_change:+.2f}%  高: {day_high:.0f}  低: {day_low:.0f}")
    log.info(f"│ BB: {lower.iloc[-1]:.0f} / {mid.iloc[-1]:.0f} / {upper.iloc[-1]:.0f}")
    log.info(f"│ RSI: {rsi:.1f}  KDJ: {k.iloc[-1]:.0f}/{d.iloc[-1]:.0f}/{j.iloc[-1]:.0f}")
    log.info(f"│ MACD: {dif.iloc[-1]:.1f}/{dea.iloc[-1]:.1f}  柱:{hist.iloc[-1]:.1f}")
    log.info(f"│ ATR: {atr:.1f} ({atr/last*100:.2f}%)")

    # VSA 最近K线分析
    vsa_bars = vsa_scan(df, window=20)
    last_vsa = vsa_bars[-1] if vsa_bars else None
    if last_vsa and last_vsa.strength >= 1:
        bias_icon = "🟢" if last_vsa.bias == "bullish" else "🔴" if last_vsa.bias == "bearish" else "⚪"
        log.info(f"│ VSA: {bias_icon} [{last_vsa.bar_type}] {'★' * last_vsa.strength}")

    # 相对成交量
    rel_vol = relative_volume(df, 20)
    rv = float(rel_vol.iloc[-1]) if not np.isnan(rel_vol.iloc[-1]) else 1.0
    vol_label = "放量🔥" if rv > 2 else "偏高" if rv > 1.3 else "缩量" if rv < 0.6 else ""
    if vol_label:
        log.info(f"│ 量比: {rv:.1f} {vol_label}")

    # 生成信号
    signals = generate_signals(df, direction, cfg)

    if signals:
        log.info(f"│")
        log.info(f"│ ⚡ 信号:")
        for sig in signals:
            strength_icon = {"强": "🔥", "中": "⚡", "弱": "💡"}.get(sig["strength"], "")
            log.info(f"│   {strength_icon} [{sig['type']}] {sig['reason']}")
            if "entry" in sig:
                log.info(f"│      入场:{sig['entry']:.0f}  止损:{sig.get('stop', 0):.0f}  目标:{sig.get('target', 0):.0f}")
    else:
        log.info(f"│")
        log.info(f"│ 🔇 无信号，继续观望")

    log.info(f"└{'─'*50}┘")


def run_once(period: str = "5"):
    """执行一次分析"""
    config = load_config()
    positions = config["positions"]
    intraday_cfg = config.get("intraday", {})

    now = datetime.now()
    log.info(f"\n{'═'*52}")
    log.info(f"  日内监控  {now.strftime('%Y-%m-%d %H:%M:%S')}  K线:{period}分钟")
    log.info(f"{'═'*52}")

    tq = config.get("tqsdk") or {}
    mon: Optional[TqPhase3Monitor] = None
    if tqsdk_configured(config):
        try:
            syms = list(positions.keys())
            mon = TqPhase3Monitor(
                account=str(tq["account"]).strip(),
                password=str(tq["password"]).strip(),
                symbols=syms,
                period=period,
            )
            mon.connect()
            mon.warmup()
            log.info("\n  数据源: TqSdk 实时K线")
        except Exception as e:
            log.info(f"\n  ⚠️ TqSdk 不可用，使用新浪分钟线: {e}")
            mon = None
    else:
        log.info("\n  数据源: 新浪分钟线（配置 tqsdk 账户可改用 TqSdk）")

    try:
        for symbol, pos_cfg in positions.items():
            try:
                if mon:
                    df = mon.minute_df(symbol)
                    if df is None:
                        df = pd.DataFrame()
                else:
                    df = fetch_minute_data(symbol, period)
                if df is None or df.empty:
                    log.info(f"\n  {pos_cfg['name']}: 无数据（非交易时段或合约无行情）")
                    continue
                print_dashboard(symbol, pos_cfg["name"], pos_cfg["direction"], df, intraday_cfg)
            except Exception as e:
                log.info(f"\n  {pos_cfg['name']}: ❌ {e}")
    finally:
        if mon:
            mon.close()


def run_loop(period: str = "5", interval: int = 60):
    """持续监控（单连接 TqSdk 时复用同一订阅，避免重复握手）"""
    print(f"日内监控详情写入 {get_log_path()}（本终端仅提示启动/停止）。")
    config = load_config()
    positions = config["positions"]
    intraday_cfg = config.get("intraday", {})
    tq = config.get("tqsdk") or {}

    log.info("🚀 日内策略监控启动")
    log.info(f"   K线周期: {period}分钟")
    log.info(f"   刷新间隔: {interval}秒")
    log.info(f"   按 Ctrl+C 停止\n")

    mon: Optional[TqPhase3Monitor] = None
    if tqsdk_configured(config):
        try:
            mon = TqPhase3Monitor(
                account=str(tq["account"]).strip(),
                password=str(tq["password"]).strip(),
                symbols=list(positions.keys()),
                period=period,
            )
            mon.connect()
            mon.warmup()
            log.info("   数据源: TqSdk 实时K线\n")
        except Exception as e:
            log.info(f"   ⚠️ TqSdk 不可用，使用新浪: {e}\n")
            mon = None
    else:
        log.info("   数据源: 新浪分钟线\n")

    try:
        while True:
            try:
                now = datetime.now()
                log.info(f"\n{'═'*52}")
                log.info(f"  日内监控  {now.strftime('%Y-%m-%d %H:%M:%S')}  K线:{period}分钟")
                log.info(f"{'═'*52}")

                for symbol, pos_cfg in positions.items():
                    try:
                        if mon:
                            df = mon.minute_df(symbol)
                        else:
                            df = fetch_minute_data(symbol, period)
                        if df is None or df.empty:
                            log.info(f"\n  {pos_cfg['name']}: 无数据")
                            continue
                        print_dashboard(symbol, pos_cfg["name"], pos_cfg["direction"], df, intraday_cfg)
                    except Exception as e:
                        log.info(f"\n  {pos_cfg['name']}: ❌ {e}")

                log.info("\n⏳ %s秒后刷新...", interval)
                if mon:
                    mon.wait_interval(float(interval))
                else:
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\n\n🛑 监控已停止")
                break
            except Exception as e:
                log.exception("错误: %s", e)
                time.sleep(10)
    finally:
        if mon:
            mon.close()


def main():
    parser = argparse.ArgumentParser(description="日内策略监控")
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"], help="K线周期(分钟)")
    parser.add_argument("--once", action="store_true", help="只运行一次")
    parser.add_argument("--interval", type=int, default=60, help="刷新间隔(秒)")
    args = parser.parse_args()

    if args.once:
        run_once(args.period)
    else:
        run_loop(args.period, args.interval)


if __name__ == "__main__":
    main()
