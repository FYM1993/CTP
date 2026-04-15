#!/usr/bin/env python3
"""
TqSdk 驱动的每日自动化交易工作流
========================================
核心流程：
Phase 0: 数据同步 - 同步所有品种历史日线数据
Phase 1: 市场筛选 - 基于基本面分类阈值和价格位置筛选候选品种
Phase 2: 深度分析 - 技术面 + 量价 + 持仓多维度评估，RRF 排名分流
Phase 3: 盘中监控 - 实时分钟线跟踪，捕捉入场与反转信号
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
#  1. 全局配置与日志初始化 (必须在导入子模块前)
# ============================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 交易时段定义
DAY_SESSION = (9, 0, 15, 0)
NIGHT_SESSION = (21, 0, 23, 30)

# RRF 算法参数
RRF_K = 10
DIRECTION_PENALTY = 0.5

# 报告目录
RESULT_DIR = Path(__file__).parent.parent / "data" / "reports"

# 延迟导入，防止日志器失效
from tqsdk import TqApi, TqAuth

# 内部模块导入
from data_cache import (
    get_all_symbols,
    get_daily_tq,
    get_hog_fundamentals,
    get_inventory,
    get_warehouse_receipt,
    prefetch_all_tq,
    to_tq_symbol,
)
from wyckoff import (
    analyze_oi,
    analyze_volume_pattern,
    assess_reversal_status,
    assess_trend_entry,
    wyckoff_phase,
)

# ============================================================
#  2. 基础工具函数
# ============================================================

def load_config() -> Dict[str, Any]:
    """加载项目的 config.yaml 配置文件"""
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_trading_hours() -> bool:
    """判断当前是否处于国内期货交易时间"""
    now = datetime.now()
    m = now.hour * 60 + now.minute
    is_day = (DAY_SESSION[0] * 60 + DAY_SESSION[1]) <= m <= (DAY_SESSION[2] * 60 + DAY_SESSION[3])
    is_night = (NIGHT_SESSION[0] * 60 + NIGHT_SESSION[1]) <= m <= (NIGHT_SESSION[2] * 60 + NIGHT_SESSION[3])
    return is_day or is_night


def _md_cell(val: Any) -> str:
    """Markdown 表格单元格转义"""
    if val is None:
        return "-"
    return str(val).replace("|", "｜").replace("\n", " ")


def _clean_numpy_types(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 Numpy 数据类型转换为 Python 原生类型"""
    clean_data = []
    for item in data:
        row = {}
        for k, v in item.items():
            if isinstance(v, (np.bool_, np.integer)):
                row[k] = int(v)
            elif isinstance(v, np.floating):
                row[k] = float(v)
            else:
                row[k] = v
        clean_data.append(row)
    return clean_data

# ============================================================
#  3. 技术指标计算工具 (原子化函数)
# ============================================================

def calc_ma(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均线 (MA)"""
    return series.rolling(n).mean()

def calc_ema(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均线 (EMA)"""
    return series.ewm(span=n, adjust=False).mean()

def calc_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """真实波幅均值 (ATR)"""
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指标"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    diff = ema_fast - ema_slow
    dea = calc_ema(diff, signal)
    hist = (diff - dea) * 2
    return diff, dea, hist

def calc_bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带 (Bollinger Bands)"""
    ma = calc_ma(series, n)
    std = series.rolling(n).std()
    return ma + k * std, ma, ma - k * std

def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ 指标"""
    low_min = df["low"].rolling(n).min()
    high_max = df["high"].rolling(n).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-12) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

# ============================================================
#  4. 核心分析引擎
# ============================================================

def _score_symbol_fundamental(sym: str, name: str, exchange: str, df: pd.DataFrame, default_threshold: float) -> Optional[Dict[str, Any]]:
    """Phase 1: 基本面评分逻辑"""
    cfg = load_config()
    fs = cfg.get("fundamental_screening", {})
    
    # 类别阈值
    cat_thresholds = fs.get("category_thresholds", {})
    cat = "none"
    for c, syms in fs.get("categories", {}).items():
        if sym in syms:
            cat = c
            break
    threshold = float(cat_thresholds.get(cat, default_threshold))
    
    try:
        last = float(df["close"].iloc[-1])
        r_300 = df.tail(300)
        h_all, l_all = r_300["close"].max(), r_300["close"].min()
        range_pct = (last - l_all) / (h_all - l_all + 1e-12) * 100

        inv = get_inventory(sym)
        receipt = get_warehouse_receipt(sym)
        hog = get_hog_fundamentals() if sym == "LH0" else None
        
        fund_score, details = 0.0, []
        if inv:
            chg = inv["inv_change_4wk"]
            if chg < -10: fund_score += 10
            elif chg < -3: fund_score += 5
            elif chg > 10: fund_score -= 10
            elif chg > 3: fund_score -= 5
            details.append(f"库存({chg:+.1f}%)")
        if receipt:
            rc = receipt["receipt_change"]
            if rc < 0: fund_score += 15
            elif rc > 0: fund_score -= 15
        if sym == "LH0" and (not hog or hog.get("profit_margin") is None):
            if range_pct < 15: fund_score += 20
            elif range_pct > 85: fund_score -= 20
        elif hog and hog.get("profit_margin") is not None:
            pm = hog["profit_margin"]
            if pm < -20: fund_score += 25
            elif pm < -10: fund_score += 15
            elif pm > 20: fund_score -= 25
            elif pm > 10: fund_score -= 15
            details.append(f"利润({pm:+.1f}%)")

        return {
            "symbol": sym, "name": name, "exchange": exchange, 
            "score": fund_score, "details": " | ".join(details), 
            "range_pct": range_pct, "category": cat
        }
    except Exception as e:
        logger.error(f"评分失败 {name}: {e}")
        return None

def _calculate_technical_scores(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    """计算多维度技术评分 (Phase 2)"""
    close = df["close"]
    last_price = close.iloc[-1]
    scores = {}

    # 1. 均线排列 (±15分)
    ma_windows = cfg.get("ma_windows", [5, 10, 20, 60, 120])
    mas = {w: calc_ma(close, w).iloc[-1] for w in ma_windows}
    valid_mas = [v for v in mas.values() if pd.notna(v)]
    if valid_mas:
        above_count = sum(1 for m in valid_mas if last_price > m)
        price_pos = (above_count / len(valid_mas) - 0.5) * 2
        short_ma_avg = np.mean([mas[w] for w in ma_windows[:3] if pd.notna(mas[w])])
        long_ma_avg = np.mean([mas[w] for w in ma_windows[3:] if pd.notna(mas[w])])
        alignment = np.sign(short_ma_avg / long_ma_avg - 1) if pd.notna(short_ma_avg) and pd.notna(long_ma_avg) else 0
        scores["均线排列"] = np.clip((price_pos * 0.7 + alignment * 0.3) * 15, -15, 15)

    # 2. MACD (±10分)
    _, _, hist = calc_macd(close)
    h_now = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2]) if len(hist) > 1 else 0
    if h_now > 0: scores["MACD"] = 10 if h_now > h_prev else 5
    elif h_now < 0: scores["MACD"] = -10 if h_now < h_prev else -5

    # 3. RSI (±10分)
    rsi_val = calc_rsi(close).iloc[-1]
    if rsi_val < 25: scores["RSI"] = 10
    elif rsi_val < 35: scores["RSI"] = 5
    elif rsi_val > 75: scores["RSI"] = -10
    elif rsi_val > 65: scores["RSI"] = -5

    # 4. 布林带位置 (±10分)
    upper, _, lower = calc_bollinger(close)
    u, l = float(upper.iloc[-1]), float(lower.iloc[-1])
    bb_pos = (last_price - l) / (u - l + 1e-12)
    scores["布林带"] = float(np.clip((0.5 - bb_pos) * 20, -10, 10))

    # 5. Wyckoff 阶段 (±15分)
    phase = wyckoff_phase(df, lookback=120)
    p_map = {"accumulation": 15, "markup": 8, "distribution": -15, "markdown": -8}
    scores["Wyckoff阶段"] = p_map.get(phase.phase, 0) * phase.confidence
    
    vp = analyze_volume_pattern(df, lookback=60)
    ratio = vp["up_down_ratio"]
    scores["量价关系"] = 10 if ratio > 1.5 else (-10 if ratio < 0.67 else 0)
    
    oi_sig = analyze_oi(df, window=5)
    if oi_sig:
        val = 15 if oi_sig.strength == "强势" else 5
        scores["持仓信号"] = val if oi_sig.bias == "bullish" else -val

    return scores

def analyze_one(symbol: str, name: str, direction: str, cfg: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """对单个品种执行完整的技术面深度分析"""
    if df.empty or len(df) < 60:
        return None
    last = df["close"].iloc[-1]
    
    scores = _calculate_technical_scores(df, cfg)
    total = sum(scores.values())
    
    reversal = assess_reversal_status(df, direction, lookback=60)
    trend = assess_trend_entry(df, direction, p2_score=total, p1_score=cfg.get("fund_screen_score", 0))
    
    # 修复 NoneType 错误: 使用 or {} 确保即使值为 None 也能调用 get
    active = reversal if reversal["has_signal"] else (trend if trend["has_signal"] else reversal)
    if reversal["has_signal"]:
        active["entry_mode"] = "reversal"
    
    atr = calc_atr(df).iloc[-1]
    entry = last
    if active.get("entry_mode") == "trend" and "stop_ref" in active:
        stop = active["stop_ref"]
    else:
        sig_bar = active.get("signal_bar") or {}
        if direction == "long":
            stop = sig_bar.get("low", last) - 0.5 * atr
        else:
            stop = sig_bar.get("high", last) + 0.5 * atr
    
    tp1 = entry + (3 * atr if direction == "long" else -3 * atr)
    rr = abs(tp1 - entry) / max(abs(entry - stop), 1e-12)
    
    score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
    actionable = score_ok and rr >= 1.0 and active.get("has_signal", False)

    return {
        "symbol": symbol, "name": name, "direction": direction, "score": float(total),
        "actionable": actionable, "price": float(last), "entry": float(entry), "stop": float(stop),
        "tp1": float(tp1), "rr": float(rr), "wyckoff_phase": wyckoff_phase(df).phase,
        "reason": ", ".join([f"{k}{v:+.0f}" for k, v in sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3] if abs(v) >= 3]),
        "reversal_status": active,
        "classical_score": float(sum(scores.get(k, 0) for k in ["均线排列", "MACD", "RSI", "布林带", "动量", "价格位置"])),
        "wyckoff_score": float(sum(scores.get(k, 0) for k in ["Wyckoff阶段", "量价关系", "VSA信号"])),
        "oi_score": float(scores.get("持仓信号", 0)),
    }

# ============================================================
#  5. Phase 执行逻辑
# ============================================================

def phase_1_screen_tq(api: TqApi, all_data: Dict[str, pd.DataFrame], threshold: float = 10) -> List[Dict[str, Any]]:
    logger.info("执行 Phase 1: 基本面筛选")
    cfg = load_config()
    fs = cfg.get("fundamental_screening") or {}
    default_thr = float(fs.get("default_threshold", threshold))
    candidates = []
    
    for info in get_all_symbols():
        sym = info["symbol"]
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            continue
        res = _score_symbol_fundamental(sym, info["name"], info["exchange"], df, default_thr)
        if res and (abs(res["score"]) >= default_thr or res["range_pct"] < 5 or res["range_pct"] > 95):
            res["entry_pool_reason"] = f"达标({res['score']:+.0f})" if abs(res["score"]) >= default_thr else "极端价位"
            if abs(res["score"]) >= default_thr:
                res["direction"] = "long" if res["score"] > 0 else "short"
            else:
                res["direction"] = "long" if res["range_pct"] < 50 else "short"
            candidates.append(res)
    return sorted(candidates, key=lambda x: abs(x["score"]), reverse=True)

def phase_2_premarket_tq(api: TqApi, candidates: List[Dict[str, Any]], config: Dict[str, Any], max_picks: int = 6) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Phase 2: 深度技术分析 + RRF 排名分流"""
    logger.info("执行 Phase 2: 深度分析")
    actionable, watchlist = [], []
    
    for cand in candidates:
        df = get_daily_tq(api, cand["symbol"], data_length=400)
        if df.empty or len(df) < 60:
            continue
        
        res = analyze_one(
            cand["symbol"], cand["name"], cand["direction"], 
            {**config.get("pre_market", {}), "fund_screen_score": cand["score"]}, 
            df
        )
        
        if res:
            # 继承 Phase 1 的基本面数据
            res["fund_screen_score"] = cand["score"]
            res["fund_range_pct"] = cand.get("range_pct")
            res["fund_details"] = cand.get("details")
            res["entry_pool_reason"] = cand.get("entry_pool_reason")
            
            # 检查方向一致性
            if np.sign(cand["score"]) != np.sign(res["score"]) and cand["score"] != 0:
                res["direction_conflict"] = True
                res["score_signs_support_direction"] = False
            else:
                res["direction_conflict"] = False
                res["score_signs_support_direction"] = True
            
            if res["actionable"]:
                actionable.append(res)
            else:
                watchlist.append(res)
    
    # RRF 排序
    def rrf_score(item, rank_list):
        try:
            rank = rank_list.index(item) + 1
            return 1.0 / (RRF_K + rank)
        except ValueError:
            return 0.0

    all_items = actionable + watchlist
    p1_rank = sorted(all_items, key=lambda x: abs(x["fund_screen_score"]), reverse=True)
    p2_rank = sorted(all_items, key=lambda x: abs(x["score"]), reverse=True)
    
    final_scores = []
    for item in all_items:
        score = rrf_score(item, p1_rank) + rrf_score(item, p2_rank)
        # 方向冲突惩罚
        if item.get("direction_conflict", False):
            score *= DIRECTION_PENALTY
        
        # 记录排名
        item["rank_p1"] = p1_rank.index(item) + 1
        item["rank_p2"] = p2_rank.index(item) + 1
        item["rrf_score"] = score
        final_scores.append((score, item))
    
    ranked = [x[1] for x in sorted(final_scores, key=lambda x: x[0], reverse=True)]
    return ranked[:max_picks], ranked[max_picks:max_picks+10]

# ============================================================
#  6. 报告与持久化
# ============================================================

def _today_md_path() -> Path:
    """当日 Markdown 报告路径"""
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.md"

def _today_json_path() -> Path:
    """当日 JSON 数据路径"""
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.json"

def _build_fund_summary_str(t: Dict[str, Any]) -> str:
    """构建基本面详情摘要字符串"""
    parts = []
    if t.get("fund_range_pct") is not None:
        parts.append(f"区间位{t['fund_range_pct']:.0f}%")
    if t.get("fund_inv_change") is not None:
        parts.append(f"库存4周{t['fund_inv_change']:+.1f}%")
    if t.get("fund_inv_percentile") is not None:
        parts.append(f"库存分位{t['fund_inv_percentile']:.0f}%")
    if t.get("fund_receipt_change") is not None:
        parts.append(f"仓单{t['fund_receipt_change']:+.0f}")
    if t.get("fund_seasonal") is not None and abs(t["fund_seasonal"]) > 0.5:
        parts.append(f"季节性{'偏多' if t['fund_seasonal'] > 0 else '偏空'}")
    if t.get("fund_hog_profit") is not None:
        parts.append(f"养殖利润{t['fund_hog_profit']:+.0f}%")
    fd = t.get("fund_details", "")
    if fd and not any(fd in p for p in parts):
        parts.append(fd)
    return ", ".join(parts) if parts else "-"

def save_targets(targets: List[Dict[str, Any]], watchlist: List[Dict[str, Any]]):
    """保存 Phase 2 结果到 JSON 和 Markdown"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _clean_numpy_types(targets)
    watchlist = _clean_numpy_types(watchlist)
    
    # 保存 JSON
    json_path = _today_json_path()
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "targets": targets,
        "watchlist": watchlist,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, ensure_ascii=False, indent=2, fp=f)
    
    # 生成 Markdown 报告
    md_path = _today_md_path()
    now = datetime.now()
    
    has_actionable = len(targets) > 0
    has_watchlist = len(watchlist) > 0
    
    lines = [
        f"# 每日交易建议 {now.strftime('%Y-%m-%d')}",
        "",
        f"> 生成时间: {now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    
    if has_actionable:
        lines.append("## 可操作品种一览")
    elif has_watchlist:
        lines.append("## 今日观望（尚无可操作品种）")
        lines.append("")
        lines.append("> **未满足可操作三要件**：① 有效入场信号 ② Phase 2 达标（做多>+20 / 做空<-20）③ 盈亏比≥1。")
        lines.append("> **方向**由 Phase 1 **入池理由**决定（基本面达标或极端价位），**不是** P1+P2 分数的合成结果。"
                    "若 RRF 旁有 **⚠️逆势**，表示 P1/P2 读数与计划方向未共振（常见于摸顶/抄底），"
                    "可在有确认信号时参与，但须自行加权风控。")
    else:
        lines.append("## 今日无信号")
        lines.append("")
        lines.append("> 全市场无极端品种，今日无操作/观望机会。")
    
    # 可操作品种表格
    if targets:
        lines.append("")
        lines.append("| 状态 | 合约 | 方向 | 入池理由 | 当前价 | 入场信号 | 入场价 | 止损 | 止盈1 | 盈亏比 | RRF | #P1 | #P2 | 基本面 | 技术面 | 信号强度 | 标签 | 基本面详情 |")
        lines.append("| :---: | :--- | :---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
        for t in targets:
            rev = t.get("reversal_status", {})
            sig_cn = {
                "Spring": "弹簧", "SOS": "强势突破", "SC": "卖方高潮",
                "UT": "上冲回落", "SOW": "弱势跌破", "BC": "买方高潮",
                "StopVol_Bull": "停止量", "StopVol_Bear": "停止量",
                "Pullback": "回撤", "TrendBreak": "突破"
            }.get(rev.get("signal_type", ""), rev.get("signal_type", ""))
            mode_tag = "🔄" if rev.get("entry_mode") == "trend" else ""
            signal_str = f"{mode_tag}{sig_cn} {rev['signal_date'][-5:]}" if rev.get("has_signal") else "⏳"
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = rev.get("signal_strength", 0)
            
            tag_cells = []
            if t.get("direction_conflict"):
                tag_cells.append("P1P2异号")
            if not t.get("score_signs_support_direction", True):
                tag_cells.append("逆势")
            tag_str = "/".join(tag_cells) if tag_cells else "-"
            
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            name_cell = _md_cell(f"{t['name']}(主力)")
            dir_cell = _md_cell(dir_icon)
            
            lines.append(
                f"| {_md_cell('✅入场')} | {name_cell} | {dir_cell} "
                f"| {epr} | {t['price']:.0f} | {_md_cell(signal_str)} | {t['entry']:.0f} | {t['stop']:.0f} "
                f"| {t['tp1']:.0f} | {t['rr']:.2f} "
                f"| **{rrf:.4f}** | {r1} | {r2} "
                f"| {t.get('fund_screen_score', 0):+.1f} | {t.get('score', 0):+.1f} "
                f"| {ss:.2f} | {_md_cell(tag_str)} | {_md_cell(_build_fund_summary_str(t))} |"
            )
    
    # 观望品种表格
    if watchlist:
        lines.append("")
        lines.append("| 合约 | 方向 | 入池理由 | 当前价 | 系统提示 | RRF | #P1 | #P2 | 基本面 | 技术面 | 信号强度 | 标签 | 基本面详情 |")
        lines.append("| :--- | :---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
        for t in watchlist:
            rev = t.get("reversal_status", {})
            next_exp = rev.get("next_expected", "等待反转信号")
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = rev.get("signal_strength", 0)
            
            wtags = []
            if t.get("direction_conflict"):
                wtags.append("P1P2异号")
            if not t.get("score_signs_support_direction", True):
                wtags.append("逆势")
            conflict_tag = f" ⚠️{'/'.join(wtags)}" if wtags else ""
            wtag_cell = "/".join(wtags) if wtags else "-"
            
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            rrf_cell = _md_cell(f"**{rrf:.4f}**{conflict_tag}")
            name_cell = _md_cell(f"{t['name']}(主力)")
            dir_cell = _md_cell(dir_icon)
            
            lines.append(
                f"| {name_cell} | {dir_cell} "
                f"| {epr} | {t['price']:.0f} | {_md_cell(f'⏳{next_exp}')} "
                f"| {rrf_cell} | {r1} | {r2} "
                f"| {t.get('fund_screen_score', 0):+.1f} | {t.get('score', 0):+.1f} "
                f"| {ss:.2f} | {_md_cell(wtag_cell)} | {_md_cell(_build_fund_summary_str(t))} |"
            )
    
    # 评分说明
    lines.extend([
        "",
        "## 评分说明",
        "",
        "### Phase 1 基本面筛选评分体系",
        "",
        "所有评分统一约定：**正分=偏多(做多机会)，负分=偏空(做空机会)**。分值越大信号越强。",
        "",
        "Phase 1 为纯基本面筛选，通过**各品种基本面门槛**或极端价格位（<5% 或 >95%）进入候选池。门槛由 config 分层配置（默认与各品种分类），并非单一固定「数值≥10」规则。",
        "",
        "| 指标 | 满分 | 数据来源 | 含义 |",
        "| :--- | ---: | :--- | :--- |",
        "| 库存变化 | ±20 | 东方财富(~30品种) | 去库=供不应求=+分(做多), 累库=供过于求=-分(做空) |",
        "| 库存分位 | ±10 | 东方财富 | 当前库存在52周范围内的位置。库存低位=+分(做多), 库存高位=-分(做空) |",
        "| 仓单变化 | ±15 | 上期所+广期所仓单 | 仓单减少=供应收紧=+分(做多), 仓单增加=供应压力=-分(做空) |",
        "| 季节性 | ±10 | 历史日线 | 当前月份的历史涨跌概率与平均收益率 |",
        "| 生猪专项 | ±15 | 卓创资讯 | 仅LH0: 养殖亏损=+分(价格低做多), 养殖盈利=-分(价格高做空) |",
        "",
        "### Phase 2 深度分析评分体系",
        "",
        "| 维度 | 满分 | 说明 |",
        "| :--- | ---: | :--- |",
        "| 技术面 | ±55 | 均线排列、MACD、RSI、布林带、动量、价格位置 |",
        "| 量价面 | ±35 | Wyckoff阶段、量价关系、VSA信号 |",
        "| 持仓面 | ±15 | OI价格四象限、OI背离 |",
        "",
        "可操作条件：① 有新鲜反转/顺势入场信号 ② Phase 2 总分达标(做多>+20/做空<-20) ③ 盈亏比≥1.0。",
        "**计划方向**来自 Phase 1 入池规则（见「入池理由」），与 P1/P2 分数符号可以不一致；若标签含 **逆势**，表示 P1/P2 未与计划方向共振，属反转/摸顶情境，须额外风控。",
        "",
        "**两阶段评分**：正分=看多, 负分=看空。P1 与 P2 同号时通常更可靠；RRF 对 P1/P2 异号有惩罚。",
        "",
        "### RRF 综合排名 (Reciprocal Rank Fusion)",
        "",
        "品种最终排名通过 RRF 算法融合 Phase 1 和 Phase 2 的排名，公式:",
        "",
        "> `RRF(d) = 1/(k + rank_P1) + 1/(k + rank_P2)`  (k=10)",
        "",
        "- 各 Phase 按 |评分| 绝对值降序独立排名(#1=最强)",
        "- RRF 不依赖评分的绝对值或尺度，只看排名位次",
        "- 两个 Phase 都排名靠前的品种，RRF 得分最高",
        "- 只有一个 Phase 排名高的品种，也能保留但排名靠后",
        "- **方向一致性惩罚**: 若 P1(基本面)与 P2(技术面)方向冲突(一正一负)，RRF 得分×0.5",
        "",
    ])
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"报告已保存: {md_path}")

def load_targets() -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """加载今日分析结果"""
    json_path = _today_json_path()
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["targets"], data["watchlist"]

# ============================================================
#  7. 实时监控与主入口
# ============================================================

def print_dashboard(t: Dict[str, Any]):
    """打印盘中监控面板"""
    last_price = t["klines"].close.iloc[-1]
    chg = (last_price / t["price"] - 1) * 100
    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {t['name']:4s} | 现价:{last_price:7.0f} | 幅度:{chg:+5.2f}% | 止损:{t['stop']:7.0f}", end="", flush=True)

def phase_3_intraday_tq(api: TqApi, targets: List[Dict[str, Any]], watchlist: List[Dict[str, Any]]):
    """Phase 3: 实时盘中监控"""
    if not is_trading_hours():
        logger.warning("⚠️ 当前非交易时段，跳过盘中监控")
        print(f"\n⚠️ 当前时间 {datetime.now().strftime('%H:%M')} 不在交易时段内")
        print(f"   日盘: 09:00-15:00")
        print(f"   夜盘: 21:00-23:30")
        print(f"\n✅ 盘前分析已完成，报告已保存至 {RESULT_DIR}")
        return
    
    logger.info("开始 Phase 3: 实时监控")
    for t in targets:
        t["klines"] = api.get_kline_serial(to_tq_symbol(t["symbol"]), 300, data_length=100)
    
    while is_trading_hours():
        api.wait_update()
        for t in targets:
            print_dashboard(t)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="恢复今日分析结果并进入监控")
    parser.add_argument("--threshold", type=float, default=10.0, help="基本面评分阈值")
    parser.add_argument("--skip-monitor", action="store_true", help="跳过盘中监控，仅生成报告")
    args = parser.parse_args()
    
    config = load_config()
    tq_cfg = config.get("tqsdk", {})
    
    print(f"\n🚀 启动工作流 (阈值: {args.threshold})")
    auth = TqAuth(tq_cfg["account"], tq_cfg["password"]) if tq_cfg.get("account") else None
    api = TqApi(auth=auth)
    
    try:
        if args.resume:
            result = load_targets()
            if result:
                targets, watchlist = result
                print(f"📂 已加载今日分析结果: {len(targets)} 个目标")
            else:
                logger.error("未找到今日分析结果")
                return
        else:
            all_symbols = get_all_symbols()
            all_data = prefetch_all_tq(api, all_symbols)
            
            candidates = phase_1_screen_tq(api, all_data, threshold=args.threshold)
            if not candidates:
                logger.info("无符合条件的品种")
                print("⚠️ 今日无基本面达标品种，程序退出。")
                return
            
            targets, watchlist = phase_2_premarket_tq(api, candidates, config)
            save_targets(targets, watchlist)
            print(f"✅ 分析完成，生成 {len(targets)} 个目标，{len(watchlist)} 个观察。")
        
        if not args.skip_monitor:
            phase_3_intraday_tq(api, targets, watchlist)
        else:
            print("\n📊 跳过盘中监控，报告已生成。")
    finally:
        api.close()


if __name__ == "__main__":
    main()
