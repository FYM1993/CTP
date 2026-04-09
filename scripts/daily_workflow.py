#!/usr/bin/env python3
"""
每日自动化交易工作流
========================================

完整流水线:
  Phase 0 — 数据预加载 (盘前)
    一次性拉取所有品种日线，缓存到本地 parquet
    当日首次运行走API (~60个请求)，后续运行零API调用

  Phase 1 — 全市场筛选 (盘前)
    基于缓存数据分析，零API调用

  Phase 2 — 盘前深度分析 (盘前)
    对筛选出的品种做深度分析，零API调用

  Phase 3 — 盘中实时监控 (09:00-15:00 / 21:00-23:30)
    分钟K线实时拉取（无法缓存）

使用方法:
    python scripts/daily_workflow.py                  # 完整流程
    python scripts/daily_workflow.py --no-monitor     # 只筛选+分析
    python scripts/daily_workflow.py --skip-screen    # 跳过筛选，用config品种
    python scripts/daily_workflow.py --resume         # 恢复今日分析，直接监控
"""

import sys
import time
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from data_cache import (
    get_all_symbols, get_daily, get_minute, get_inventory,
    get_warehouse_receipt, get_hog_fundamentals,
    get_seasonality, get_oi_structure,
    prefetch_all, get_request_count,
)
from pre_market import analyze_one, score_signals
from intraday import print_dashboard
from wyckoff import wyckoff_score


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  时间工具
# ============================================================

DAY_SESSION = (9, 0, 15, 0)
NIGHT_SESSION = (21, 0, 23, 30)

def is_trading_hours() -> bool:
    now = datetime.now()
    t = now.hour * 60 + now.minute
    return ((DAY_SESSION[0] * 60 + DAY_SESSION[1]) <= t <= (DAY_SESSION[2] * 60 + DAY_SESSION[3])) or \
           ((NIGHT_SESSION[0] * 60 + NIGHT_SESSION[1]) <= t <= (NIGHT_SESSION[2] * 60 + NIGHT_SESSION[3]))


def time_to_next_session() -> int:
    if is_trading_hours():
        return 0
    now = datetime.now()
    t = now.hour * 60 + now.minute
    day_start = DAY_SESSION[0] * 60 + DAY_SESSION[1]
    night_start = NIGHT_SESSION[0] * 60 + NIGHT_SESSION[1]
    if t < day_start:
        return (day_start - t) * 60
    elif t < night_start:
        return (night_start - t) * 60
    else:
        return (24 * 60 - t + day_start) * 60


# ============================================================
#  Phase 0: 数据预加载
# ============================================================

def phase_0_prefetch() -> dict[str, pd.DataFrame]:
    print("\n" + "▓" * 60)
    print("▓  Phase 0: 数据预加载")
    print("▓" * 60)

    symbols = get_all_symbols()
    before = get_request_count()
    data = prefetch_all(symbols)
    after = get_request_count()

    if after == before:
        print("  ✅ 全部命中本地缓存，零API调用")
    else:
        print(f"  ✅ 完成，本次API调用 {after - before} 次")

    return data


# ============================================================
#  Phase 1: 全市场筛选（纯本地计算）
# ============================================================

def phase_1_screen(all_data: dict[str, pd.DataFrame], threshold: float = 25) -> list[dict]:
    print("\n" + "▓" * 60)
    print("▓  Phase 1: 全市场筛选")
    print("▓" * 60)

    symbols = get_all_symbols()
    candidates = []

    for info in symbols:
        sym = info["symbol"]
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            continue

        result = _score_symbol(sym, info["name"], info["exchange"], df)
        if result and abs(result["score"]) >= threshold:
            result["direction"] = "long" if result["score"] < 0 else "short"
            candidates.append(result)

    candidates.sort(key=lambda x: abs(x["score"]), reverse=True)

    print(f"\n  ✅ 通过筛选 (|评分| ≥ {threshold}): {len(candidates)} 个品种\n")

    long_cands = [c for c in candidates if c["direction"] == "long"]
    short_cands = [c for c in candidates if c["direction"] == "short"]

    def _print_cand(c):
        line = (f"     {c['name']:6s} ({c['symbol']:5s})  "
                f"区间位{c['range_pct']:3.0f}%  RSI={c['rsi']:.0f}  "
                f"技术={c.get('tech_score', 0):+.0f}  基本面={c.get('fund_score', 0):+.0f}  "
                f"总分={c['score']:+.0f}")
        fd = c.get("fund_details", "")
        if fd:
            line += f"  [{fd}]"
        print(line)

    if long_cands:
        print(f"  🟢 做多候选:")
        for c in long_cands[:8]:
            _print_cand(c)
    if short_cands:
        print(f"  🔴 做空候选:")
        for c in short_cands[:8]:
            _print_cand(c)

    return candidates


def _score_symbol(sym: str, name: str, exchange: str, df: pd.DataFrame) -> dict | None:
    """
    对单品种计算筛选评分。

    评分维度:
      技术面: 价格位置(±30), RSI(±15), 均线趋势(±10), Wyckoff量价(±25)
      基本面: 库存水平(±20), 库存分位(±10), 仓单变化(±15),
              持仓结构(±10), 季节性(±10), 生猪专项(±15)
    """
    try:
        close = df["close"]
        last = float(close.iloc[-1])
        n = min(len(df), 300)
        recent = df.tail(n)

        high_all = float(recent["close"].max())
        low_all = float(recent["close"].min())
        range_pct = (last - low_all) / (high_all - low_all + 1e-12) * 100

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - 100 / (1 + gain / (loss + 1e-12))).iloc[-1])

        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma60 = float(close.rolling(min(60, len(close))).mean().iloc[-1])
        ma_trend = (ma5 / ma60 - 1) * 100

        ret_5d = (last / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0
        ret_20d = (last / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0

        wk = wyckoff_score(df) if len(df) > 60 else {}
        wk_phase = wk.get("phase", "")
        wk_composite = wk.get("composite", 0)

        # ---- 基本面数据采集 ----
        inv = get_inventory(sym)
        receipt = get_warehouse_receipt(sym)
        oi_struct = get_oi_structure(df)
        seasonal = get_seasonality(df)
        hog = get_hog_fundamentals() if sym == "LH0" else None

        # ---- 技术面评分 ----
        tech_score = 0.0

        if range_pct < 10: tech_score -= 30
        elif range_pct < 20: tech_score -= 20
        elif range_pct < 30: tech_score -= 10
        elif range_pct > 90: tech_score += 30
        elif range_pct > 80: tech_score += 20
        elif range_pct > 70: tech_score += 10

        if rsi < 25: tech_score -= 15
        elif rsi < 35: tech_score -= 8
        elif rsi > 75: tech_score += 15
        elif rsi > 65: tech_score += 8

        if ma_trend < -5: tech_score -= 10
        elif ma_trend < -2: tech_score -= 5
        elif ma_trend > 5: tech_score += 10
        elif ma_trend > 2: tech_score += 5

        tech_score += np.clip(wk_composite, -25, 25)

        # ---- 基本面评分 ----
        fund_score = 0.0
        fund_details = []

        # 库存变化 (±20)
        inv_4wk = None
        inv_pct = None
        if inv:
            inv_4wk = inv["inv_change_4wk"]
            inv_pct = inv.get("inv_percentile")
            cum = inv.get("inv_cumulating_weeks", 4)
            trend = inv.get("inv_trend", "持平")

            if inv_4wk > 10: fund_score += 10
            elif inv_4wk > 3: fund_score += 5
            elif inv_4wk < -10: fund_score -= 10
            elif inv_4wk < -3: fund_score -= 5
            if cum >= 6: fund_score += 10
            elif cum <= 2: fund_score -= 10

            fund_details.append(f"库存{trend}({inv_4wk:+.1f}%)")

        # 库存分位 (±10): 库存极高=做空信号，库存极低=做多信号
        if inv_pct is not None:
            if inv_pct > 85:
                fund_score += 10
                fund_details.append(f"库存高位{inv_pct:.0f}%")
            elif inv_pct > 70:
                fund_score += 5
            elif inv_pct < 15:
                fund_score -= 10
                fund_details.append(f"库存低位{inv_pct:.0f}%")
            elif inv_pct < 30:
                fund_score -= 5

        # 仓单 (±15): 仓单增加=供应压力=价格下行信号
        receipt_change = None
        if receipt:
            rc = receipt["receipt_change"]
            receipt_change = rc
            rt = receipt["receipt_total"]
            if rc > 0 and rt > 0:
                change_ratio = rc / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score += 15
                    fund_details.append(f"仓单大增{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score += 8
            elif rc < 0 and rt > 0:
                change_ratio = abs(rc) / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score -= 15
                    fund_details.append(f"仓单大减{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score -= 8

        # 持仓结构 (±10)
        oi_pct = None
        oi_vs_price = None
        if oi_struct:
            oi_pct = oi_struct["oi_percentile"]
            oi_vs_price = oi_struct["oi_vs_price"]
            if oi_vs_price == "增仓上涨":
                fund_score += 5
            elif oi_vs_price == "增仓下跌":
                fund_score += 5
            elif oi_vs_price == "减仓上涨":
                fund_score -= 5
            elif oi_vs_price == "减仓下跌":
                fund_score -= 5

            if oi_pct > 80:
                fund_score += 5
                fund_details.append(f"持仓高位{oi_pct:.0f}%")
            elif oi_pct < 20:
                fund_score -= 5

        # 季节性 (±10)
        seasonal_sig = None
        if seasonal:
            sig = seasonal["seasonal_signal"]
            seasonal_sig = sig
            avg_ret = seasonal["hist_avg_return"]
            if abs(sig) > 0.6:
                s = np.clip(sig * 10, -10, 10)
                fund_score += s
                direction = "上涨" if sig > 0 else "下跌"
                fund_details.append(f"季节性偏{direction}(历史{avg_ret:+.1f}%)")

        # 生猪专项 (±15)
        hog_profit = None
        if hog:
            if "profit_margin" in hog:
                pm = hog["profit_margin"]
                hog_profit = pm
                if pm < -15:
                    fund_score -= 15
                    fund_details.append(f"养殖亏损{pm:.0f}%")
                elif pm < -5:
                    fund_score -= 8
                elif pm > 20:
                    fund_score += 10
                elif pm > 10:
                    fund_score += 5

            if "price_trend" in hog:
                pt = hog["price_trend"]
                if pt < -3:
                    fund_score -= 5
                elif pt > 3:
                    fund_score += 5

        score = tech_score + fund_score

        return {
            "symbol": sym, "name": name, "exchange": exchange,
            "price": last, "range_pct": range_pct,
            "rsi": rsi, "ma_trend": ma_trend,
            "ret_5d": ret_5d, "ret_20d": ret_20d,
            "wyckoff_phase": wk_phase,
            "score": score,
            "tech_score": tech_score,
            "fund_score": fund_score,
            "fund_details": ", ".join(fund_details) if fund_details else "",
            "inv_change_4wk": inv_4wk,
            "inv_percentile": inv_pct,
            "receipt_change": receipt_change,
            "oi_percentile": oi_pct,
            "oi_vs_price": oi_vs_price,
            "seasonal_signal": seasonal_sig,
            "hog_profit": hog_profit,
        }
    except Exception as e:
        return None


# ============================================================
#  Phase 2: 盘前深度分析（读缓存）
# ============================================================

def phase_2_premarket(candidates: list[dict], config: dict, max_picks: int = 6) -> list[dict]:
    print("\n" + "▓" * 60)
    print("▓  Phase 2: 盘前深度分析")
    print("▓" * 60)

    pre_cfg = config.get("pre_market", {})
    actionable = []

    for cand in candidates:
        try:
            fund_parts = []
            if "range_pct" in cand:
                fund_parts.append(f"区间位{cand['range_pct']:.0f}%")
            if "rsi" in cand:
                fund_parts.append(f"RSI={cand['rsi']:.0f}")
            if cand.get("inv_change_4wk") is not None:
                fund_parts.append(f"库存4周{cand['inv_change_4wk']:+.1f}%")
            if cand.get("inv_percentile") is not None:
                fund_parts.append(f"库存分位{cand['inv_percentile']:.0f}%")
            if cand.get("receipt_change") is not None:
                fund_parts.append(f"仓单{cand['receipt_change']:+.0f}")
            if cand.get("oi_vs_price"):
                fund_parts.append(cand["oi_vs_price"])
            if cand.get("fund_details"):
                fund_parts.append(cand["fund_details"])
            fund_reason = ", ".join(fund_parts) if fund_parts else ""

            merged_cfg = {
                **pre_cfg,
                "reason": f"筛选评分{cand['score']:+.0f}(技术{cand.get('tech_score',0):+.0f}/基本面{cand.get('fund_score',0):+.0f}), {fund_reason}",
            }
            result = analyze_one(
                symbol=cand["symbol"],
                name=cand["name"],
                direction=cand["direction"],
                cfg=merged_cfg,
            )
            if result:
                result["fund_range_pct"] = cand.get("range_pct", 0)
                result["fund_rsi"] = cand.get("rsi", 50)
                result["fund_inv_change"] = cand.get("inv_change_4wk")
                result["fund_inv_percentile"] = cand.get("inv_percentile")
                result["fund_receipt_change"] = cand.get("receipt_change")
                result["fund_oi_vs_price"] = cand.get("oi_vs_price")
                result["fund_seasonal"] = cand.get("seasonal_signal")
                result["fund_hog_profit"] = cand.get("hog_profit")
                result["fund_screen_score"] = cand.get("score", 0)
                result["fund_tech_score"] = cand.get("tech_score", 0)
                result["fund_fund_score"] = cand.get("fund_score", 0)
                result["fund_details"] = cand.get("fund_details", "")
                if result["actionable"]:
                    actionable.append(result)
        except Exception as e:
            print(f"\n  ⚠️ {cand['name']} 分析失败: {e}")

    actionable.sort(key=lambda x: abs(x["score"]), reverse=True)
    actionable = actionable[:max_picks]

    print(f"\n{'='*60}")
    print(f"  📋 Phase 2 汇总: {len(actionable)} 个品种可操作")
    print(f"{'='*60}")

    if actionable:
        print(f"\n  {'品种':8s} {'代码':6s} {'方向':4s} {'评分':>6s} {'入场':>8s} {'止损':>8s} {'目标':>8s} {'盈亏比':>6s}")
        print(f"  {'─'*55}")
        for a in actionable:
            dir_str = "做多" if a["direction"] == "long" else "做空"
            print(f"  {a['name']:8s} {a['symbol']:6s} {dir_str:4s} "
                  f"{a['score']:>+5.0f} {a['entry']:>8.0f} {a['stop']:>8.0f} "
                  f"{a['tp1']:>8.0f} {a['rr']:>5.1f}")
    else:
        print("\n  ⚠️ 没有品种满足入场条件，今日观望")

    return actionable


# ============================================================
#  Phase 3: 盘中监控（分钟数据实时获取）
# ============================================================

def phase_3_intraday(targets: list[dict], config: dict, period: str = "5", interval: int = 60):
    if not targets:
        print("\n  没有可操作品种，跳过盘中监控")
        return

    intraday_cfg = config.get("intraday", {})

    print("\n" + "▓" * 60)
    print("▓  Phase 3: 盘中实时监控")
    print("▓" * 60)
    print(f"\n  监控品种: {', '.join(t['name'] for t in targets)}")
    print(f"  K线周期: {period}分钟  |  刷新: {interval}秒")

    wait = time_to_next_session()
    if wait > 0:
        open_time = (datetime.now() + timedelta(seconds=wait)).strftime("%H:%M")
        print(f"\n  ⏳ 距开盘还有 {wait // 60} 分钟 ({open_time})，等待中...")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print("  → 提前进入监控")

    print(f"\n  🚀 监控启动！按 Ctrl+C 停止\n")

    while True:
        try:
            now = datetime.now()
            print(f"\n{'═'*60}")
            print(f"  盘中监控  {now.strftime('%Y-%m-%d %H:%M:%S')}  K线:{period}分钟")
            print(f"{'═'*60}")

            for t in targets:
                try:
                    df = get_minute(t["symbol"], period)
                    if df is None or df.empty:
                        print(f"\n  {t['name']}: 无数据")
                        continue
                    print_dashboard(t["symbol"], t["name"], t["direction"], df, intraday_cfg)
                    print(f"│ 📌 盘前参考: 入场{t['entry']:.0f}  止损{t['stop']:.0f}  目标{t['tp1']:.0f}")
                except Exception as e:
                    print(f"\n  {t['name']}: ❌ {e}")

            if not is_trading_hours():
                print(f"\n  🔔 交易时段结束")
                break

            print(f"\n  ⏳ {interval}秒后刷新...", end="", flush=True)
            time.sleep(interval)
            print("\033[2J\033[H", end="")
        except KeyboardInterrupt:
            print("\n\n  🛑 监控已停止")
            break
        except Exception as e:
            print(f"\n  ❌ 错误: {e}")
            time.sleep(10)


# ============================================================
#  结果持久化
# ============================================================

RESULT_DIR = Path(__file__).parent.parent / "data" / "reports"


def _today_md_path() -> Path:
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.md"


def _today_json_path() -> Path:
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.json"


def _clean_numpy(targets: list[dict]) -> list[dict]:
    """numpy 类型转 Python 原生类型"""
    clean = []
    for t in targets:
        row = {}
        for k, v in t.items():
            if isinstance(v, (np.bool_, np.integer)):
                row[k] = int(v)
            elif isinstance(v, np.floating):
                row[k] = float(v)
            else:
                row[k] = v
        clean.append(row)
    return clean


def save_targets(targets: list[dict]):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _clean_numpy(targets)

    # 1) JSON（供 --resume 读取）
    json_path = _today_json_path()
    payload = {"date": datetime.now().strftime("%Y-%m-%d"), "targets": targets}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # 2) Markdown 报告
    md_path = _today_md_path()
    now = datetime.now()
    lines = [
        f"# 每日交易建议 {now.strftime('%Y-%m-%d')}",
        "",
        f"> 生成时间: {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 可操作品种一览",
        "",
        "| 合约 | 方向 | 当前价 | 建议入场价 | 止损价 | 止盈价1 | 止盈价2 | 盈亏比 | 综合评分 | 技术面 | 量价面 | 持仓面 | 基本面筛选 | 主要依据 |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]

    for t in targets:
        name = t["name"]
        # 从 symbol 提取合约月份，如 SH0 → 烧碱(主力)
        contract_label = f"{name}(主力)"
        direction_cn = "做多" if t["direction"] == "long" else "做空"
        direction_icon = "🟢 " + direction_cn if t["direction"] == "long" else "🔴 " + direction_cn

        score = t.get("score", 0)
        classical = t.get("classical_score", 0)
        wyckoff = t.get("wyckoff_score", 0)
        oi = t.get("oi_score", 0)
        reason = t.get("reason", "")
        phase = t.get("wyckoff_phase", "")
        if phase:
            phase_cn = {"accumulation": "吸筹", "distribution": "派发",
                        "markup": "上涨", "markdown": "下跌"}.get(phase, phase)
            reason = f"Wyckoff:{phase_cn}; {reason}"

        fund_parts = []
        fund_range = t.get("fund_range_pct")
        fund_inv = t.get("fund_inv_change")
        fund_inv_pct = t.get("fund_inv_percentile")
        fund_receipt = t.get("fund_receipt_change")
        fund_oi_vp = t.get("fund_oi_vs_price")
        fund_seasonal = t.get("fund_seasonal")
        fund_hog = t.get("fund_hog_profit")
        fd = t.get("fund_details", "")

        if fund_range is not None:
            fund_parts.append(f"区间位{fund_range:.0f}%")
        if fund_inv is not None:
            fund_parts.append(f"库存4周{fund_inv:+.1f}%")
        if fund_inv_pct is not None:
            fund_parts.append(f"库存分位{fund_inv_pct:.0f}%")
        if fund_receipt is not None:
            fund_parts.append(f"仓单{fund_receipt:+.0f}")
        if fund_oi_vp:
            fund_parts.append(fund_oi_vp)
        if fund_seasonal is not None and abs(fund_seasonal) > 0.5:
            direction = "偏多" if fund_seasonal > 0 else "偏空"
            fund_parts.append(f"季节性{direction}")
        if fund_hog is not None:
            fund_parts.append(f"养殖利润{fund_hog:+.0f}%")
        if fd and not any(fd in p for p in fund_parts):
            fund_parts.append(fd)
        fund_str = ", ".join(fund_parts) if fund_parts else "-"

        lines.append(
            f"| {contract_label} | {direction_icon} "
            f"| {t['price']:.0f} | {t['entry']:.0f} | {t['stop']:.0f} "
            f"| {t['tp1']:.0f} | {t['tp2']:.0f} | {t['rr']:.2f} "
            f"| {score:+.1f} | {classical:+.1f} | {wyckoff:+.1f} | {oi:+.1f} "
            f"| {fund_str} | {reason} |"
        )

    lines.append("")
    lines.append("## 评分说明")
    lines.append("")
    lines.append("### Phase 1 筛选评分体系")
    lines.append("")
    lines.append("初筛综合技术面和基本面两个大类评分，正分=偏空信号（做空机会），负分=偏多信号（做多机会）。")
    lines.append("")
    lines.append("#### 技术面维度 (±80)")
    lines.append("")
    lines.append("| 指标 | 满分 | 含义 |")
    lines.append("| :--- | ---: | :--- |")
    lines.append("| 价格区间位 | ±30 | 价格在近300日高低点间的位置。<10%极低位=-30, >90%极高位=+30 |")
    lines.append("| RSI(14) | ±15 | 超卖(<25)=-15, 超买(>75)=+15 |")
    lines.append("| 均线趋势 | ±10 | MA5/MA60比值。多头排列=+10, 空头排列=-10 |")
    lines.append("| Wyckoff量价 | ±25 | 综合阶段判定、量价关系、VSA信号 |")
    lines.append("")
    lines.append("#### 基本面维度 (±80)")
    lines.append("")
    lines.append("| 指标 | 满分 | 数据来源 | 含义 |")
    lines.append("| :--- | ---: | :--- | :--- |")
    lines.append("| 库存变化 | ±20 | 东方财富(~30品种) | 4周库存变化率+连续累库/去库周数。累库=供过于求=做空信号 |")
    lines.append("| 库存分位 | ±10 | 东方财富 | 当前库存在52周范围内的位置。>85%=库存高位=做空, <15%=库存低位=做多 |")
    lines.append("| 仓单变化 | ±15 | 上期所仓单 | 仓单增加=交割品供应增加=价格下行压力 |")
    lines.append("| 持仓结构 | ±10 | 日线OI数据 | OI+价格四象限分析。增仓上涨=多方主导, 增仓下跌=空方主导 |")
    lines.append("| 季节性 | ±10 | 历史日线 | 当前月份的历史涨跌概率与平均收益率 |")
    lines.append("| 生猪专项 | ±15 | 卓创资讯 | 仅LH0: 养殖利润率(猪价×120kg/成本)、价格趋势 |")
    lines.append("")
    lines.append("### Phase 2 深度分析评分体系")
    lines.append("")
    lines.append("| 维度 | 满分 | 说明 |")
    lines.append("| :--- | ---: | :--- |")
    lines.append("| 技术面 | ±50 | 均线排列、MACD、RSI、布林带、动量 |")
    lines.append("| 量价面 | ±35 | Wyckoff阶段、量价关系、VSA信号 |")
    lines.append("| 持仓面 | ±15 | OI价格四象限、OI背离 |")
    lines.append("")
    lines.append("做多品种：总分 > +20 且盈亏比 >= 1.0 视为可操作。")
    lines.append("做空品种：总分 < -20 且盈亏比 >= 1.0 视为可操作。")
    lines.append("")
    lines.append("### 关键术语")
    lines.append("")
    lines.append("- **盈亏比** = |入场价-止盈价1| / |入场价-止损价|。>=2理想，>=1可接受，<1不建议入场")
    lines.append("- **止盈价1**：最近的支撑/阻力位（保守目标）")
    lines.append("- **止盈价2**：第二道支撑/阻力位（进取目标）")
    lines.append("- **库存分位**：当前库存在近52周最高与最低之间的百分比位置")
    lines.append("- **仓单**：交易所注册仓单量。仓单增加意味着现货被注册为可交割品，反映卖方意愿增强")
    lines.append("- **持仓四象限**：价格涨+OI增=新多入场(偏多)、价格跌+OI增=新空入场(偏空)、价格涨+OI减=空头回补(中性偏多)、价格跌+OI减=多头平仓(中性偏空)")
    lines.append("- **季节性**：该品种在当前月份的历史平均涨跌幅和上涨概率")
    lines.append("")

    lines.append("## 品种详细建议")
    lines.append("")
    for t in targets:
        name = t["name"]
        direction_cn = "做多" if t["direction"] == "long" else "做空"
        lines.append(f"### {name}(主力) — {direction_cn}")
        lines.append("")

        lines.append("**交易参数**")
        lines.append("")
        lines.append(f"| 当前价 | 入场价 | 止损价 | 止盈1 | 止盈2 | 盈亏比 |")
        lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: |")
        lines.append(f"| {t['price']:.0f} | {t['entry']:.0f} | {t['stop']:.0f} "
                     f"| {t['tp1']:.0f} | {t['tp2']:.0f} | {t['rr']:.2f} |")
        lines.append("")

        lines.append("**评分分解**")
        lines.append("")
        lines.append(f"- Phase 2 综合: {t.get('score', 0):+.1f} "
                     f"(技术{t.get('classical_score', 0):+.1f} / "
                     f"量价{t.get('wyckoff_score', 0):+.1f} / "
                     f"持仓{t.get('oi_score', 0):+.1f})")
        fs = t.get("fund_screen_score")
        ft = t.get("fund_tech_score")
        ff = t.get("fund_fund_score")
        if fs is not None:
            lines.append(f"- Phase 1 筛选: {fs:+.0f} "
                         f"(技术{ft:+.0f} / 基本面{ff:+.0f})" if ft is not None else f"- Phase 1 筛选: {fs:+.0f}")
        lines.append("")

        lines.append("**基本面数据**")
        lines.append("")
        if t.get("fund_range_pct") is not None:
            lines.append(f"- 价格区间位: {t['fund_range_pct']:.0f}%（年度高低点间的位置）")
        if t.get("fund_inv_change") is not None:
            lines.append(f"- 库存4周变化: {t['fund_inv_change']:+.1f}%")
        if t.get("fund_inv_percentile") is not None:
            lines.append(f"- 库存52周分位: {t['fund_inv_percentile']:.0f}%")
        if t.get("fund_receipt_change") is not None:
            lines.append(f"- 仓单日增减: {t['fund_receipt_change']:+.0f}")
        if t.get("fund_oi_vs_price"):
            lines.append(f"- 持仓-价格关系: {t['fund_oi_vs_price']}")
        if t.get("fund_seasonal") is not None:
            sig = t["fund_seasonal"]
            if abs(sig) > 0.3:
                lines.append(f"- 季节性信号: {'偏多' if sig > 0 else '偏空'} (强度{abs(sig):.1f})")
        if t.get("fund_hog_profit") is not None:
            lines.append(f"- 养殖利润率: {t['fund_hog_profit']:+.0f}%")
        fd = t.get("fund_details", "")
        if fd:
            lines.append(f"- 其他: {fd}")
        lines.append("")

        phase = t.get("wyckoff_phase", "")
        if phase:
            phase_cn = {"accumulation": "吸筹", "distribution": "派发",
                        "markup": "上涨", "markdown": "下跌"}.get(phase, phase)
            lines.append(f"**Wyckoff阶段**: {phase_cn}")
            lines.append("")
        reason = t.get("reason", "")
        if reason:
            lines.append(f"**主要依据**: {reason}")
            lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  💾 报告已保存:")
    print(f"     📄 {md_path}")
    print(f"     📊 {json_path}")


def load_targets() -> list[dict] | None:
    json_path = _today_json_path()
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text())
        if payload.get("date") == datetime.now().strftime("%Y-%m-%d"):
            return payload["targets"]
    except Exception:
        pass
    return None


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="每日自动化交易工作流")
    parser.add_argument("--no-monitor", action="store_true", help="跳过盘中监控")
    parser.add_argument("--skip-screen", action="store_true", help="跳过筛选，用config.yaml品种")
    parser.add_argument("--resume", action="store_true", help="恢复今日分析，直接监控")
    parser.add_argument("--threshold", type=float, default=25, help="筛选阈值 (默认25)")
    parser.add_argument("--max-picks", type=int, default=6, help="最多跟踪品种数")
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"], help="K线周期")
    parser.add_argument("--interval", type=int, default=60, help="监控刷新间隔(秒)")
    args = parser.parse_args()

    config = load_config()
    now = datetime.now()

    print("╔" + "═" * 58 + "╗")
    print("║" + "每日交易工作流".center(48) + "║")
    print("║" + f"  {now.strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    print("╚" + "═" * 58 + "╝")

    # --- 恢复模式 ---
    if args.resume:
        targets = load_targets()
        if targets:
            print(f"\n  📂 恢复今日目标: {len(targets)} 个品种")
            for t in targets:
                dir_str = "做多" if t["direction"] == "long" else "做空"
                print(f"     {t['name']} ({t['symbol']}) {dir_str}  评分{t['score']:+.0f}")
            if not args.no_monitor:
                phase_3_intraday(targets, config, args.period, args.interval)
            return
        print("\n  ⚠️ 无今日缓存，重新筛选")

    # --- Phase 0: 数据预加载 ---
    if args.skip_screen:
        all_data = {}
        positions = config.get("positions", {})
        for sym in positions:
            df = get_daily(sym)
            if df is not None:
                all_data[sym] = df
        print(f"\n  ⏩ 跳过筛选，加载 config 中 {len(all_data)} 个品种")
    else:
        all_data = phase_0_prefetch()

    # --- Phase 1: 筛选 ---
    if args.skip_screen:
        positions = config.get("positions", {})
        candidates = []
        for sym, pos in positions.items():
            candidates.append({
                "symbol": sym, "name": pos["name"],
                "direction": pos["direction"],
                "score": -50 if pos["direction"] == "long" else 50,
                "range_pct": 0, "rsi": 50,
                "wyckoff_phase": "",
            })
    else:
        candidates = phase_1_screen(all_data, threshold=args.threshold)

    if not candidates:
        print("\n  😴 全市场没有极端品种，今日无操作机会")
        return

    # --- Phase 2: 深度分析 ---
    actionable = phase_2_premarket(candidates, config, max_picks=args.max_picks)

    if actionable:
        save_targets(actionable)

    # --- Phase 3: 盘中监控 ---
    if args.no_monitor:
        print(f"\n  ⏩ 跳过盘中监控 (--no-monitor)")
        api_total = get_request_count()
        print(f"  📊 本次运行总API调用: {api_total} 次")
        return

    phase_3_intraday(actionable, config, args.period, args.interval)


if __name__ == "__main__":
    main()
