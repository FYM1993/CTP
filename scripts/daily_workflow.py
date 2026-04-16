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

import time
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from ctp_log import get_log_path, get_logger

from data_cache import (
    get_all_symbols, get_daily, get_minute,
    prefetch_all, get_request_count,
    get_daily_with_live_bar,
)
from pre_market import analyze_one, resolve_phase2_direction, score_signals
from intraday import print_dashboard
from wyckoff import assess_reversal_status
from tqsdk_live import try_create_phase3_monitor
from fundamental_legacy import score_symbol_legacy
from fundamental_regime import score_symbol_regime

log = get_logger("workflow")


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
    symbols = get_all_symbols()
    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 0: 数据预加载")
    log.info("▓" * 60)

    before = get_request_count()
    data = prefetch_all(symbols)
    after = get_request_count()

    if after == before:
        log.info("  ✅ 全部命中本地缓存，零API调用")
    else:
        log.info("  ✅ 完成，本次API调用 %s 次", after - before)

    return data


# ============================================================
#  Phase 1: 全市场筛选（纯本地计算）
# ============================================================

def phase_1_screen(
    all_data: dict[str, pd.DataFrame],
    threshold: float = 10,
) -> list[dict]:
    cfg = load_config()
    fs = cfg.get("fundamental_screening") or {}
    p1_engine = (fs.get("p1_engine") or "legacy").strip().lower()
    if p1_engine not in ("legacy", "regime"):
        p1_engine = "legacy"
    score_fn = score_symbol_regime if p1_engine == "regime" else score_symbol_legacy

    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 1: 全市场基本面筛选")
    log.info(
        "▓  P1引擎: %s（config fundamental_screening.p1_engine；regime 为实验对照）",
        p1_engine,
    )
    log.info("▓" * 60)

    default_thr = (
        float(fs["default_threshold"])
        if fs.get("default_threshold") is not None
        else float(threshold)
    )
    sym_to_threshold: dict[str, float] = {}
    for cat_cfg in (fs.get("categories") or {}).values():
        if not isinstance(cat_cfg, dict):
            continue
        if cat_cfg.get("threshold") is None:
            continue
        t = float(cat_cfg["threshold"])
        for s in cat_cfg.get("symbols") or []:
            sym_to_threshold[str(s)] = t

    symbols = get_all_symbols()
    candidates = []

    for info in symbols:
        sym = info["symbol"]
        df = all_data.get(sym)
        if df is None or len(df) < 60:
            if sym == "LH0":
                log.info("[P1] 生猪 LH0: 日线不足 60 根，跳过评分")
            continue

        eff_threshold = sym_to_threshold.get(sym, default_thr)
        result = score_fn(
            sym, info["name"], info["exchange"], df, threshold=eff_threshold
        )
        if result is None:
            if sym == "LH0":
                log.warning("[P1] 生猪 LH0: 评分失败")
            continue

        if sym == "LH0":
            reg = result.get("lh_regime")
            reg_s = f" @regime={reg}" if reg else ""
            log.info(
                f"[P1] 生猪 LH0: {result['score']:+.0f} (门槛{eff_threshold:g}){reg_s}"
            )

        fund_pass = abs(result["score"]) >= eff_threshold
        extreme_price = result["range_pct"] < 5 or result["range_pct"] > 95

        if fund_pass or extreme_price:
            pool_parts = []
            if fund_pass:
                # 避免字符串含 ASCII |，否则 Markdown 表格列会错位
                pool_parts.append(
                    f"基本面达标(绝对值≥{eff_threshold:g}，当前{result['score']:+.0f})"
                )
            if extreme_price:
                pool_parts.append(f"极端价位(区间位{result['range_pct']:.0f}%)")
            result["entry_pool_reason"] = "；".join(pool_parts)

            if not fund_pass and result["score"] == 0:
                result["score"] = 1.0 if result["range_pct"] < 50 else -1.0

            if fund_pass and extreme_price:
                labels = ["双标签候选"]
                reversal_score = abs(result["score"])
                trend_score = abs(result["score"])
            elif extreme_price:
                labels = ["反转候选"]
                reversal_score = abs(result["score"])
                trend_score = 0.0
            else:
                labels = ["趋势候选"]
                reversal_score = 0.0
                trend_score = abs(result["score"])

            result["attention_score"] = abs(result["score"])
            result["reversal_score"] = reversal_score
            result["trend_score"] = trend_score
            result["labels"] = labels
            result["state_labels"] = []
            result["data_coverage"] = 1.0
            result["reason_summary"] = result.get("fund_details") or result["entry_pool_reason"]
            candidates.append(result)

    candidates.sort(key=lambda x: x.get("attention_score", 0), reverse=True)

    log.info(
        "\n  ✅ 通过筛选 (|基本面| 达各品种门槛，默认%g；或极端价格): %s 个品种\n",
        default_thr,
        len(candidates),
    )

    def _log_cand(c):
        extreme = " ⚡极端价格" if (c["range_pct"] < 5 or c["range_pct"] > 95) else ""
        label_text = "/".join(c.get("labels") or [])
        line = (f"     {c['name']:6s} ({c['symbol']:5s})  "
                f"区间位{c['range_pct']:3.0f}%  "
                f"基本面={c.get('fund_score', 0):+.0f}  "
                f"关注={c.get('attention_score', 0):.0f}  "
                f"标签={label_text}{extreme}")
        fd = c.get("fund_details", "")
        if fd:
            line += f"  [{fd}]"
        if c.get("lh_regime"):
            line += f"  @regime={c['lh_regime']}"
        log.info(line)

    if candidates:
        log.info("  👀 关注候选:")
        for c in candidates[:16]:
            _log_cand(c)

    return candidates


# ============================================================
#  Phase 2: 盘前深度分析（读缓存）
# ============================================================

def _phase_scores_support_direction(direction: str, fund: float, p2: float) -> bool:
    """
    约定：正分=偏多、负分=偏空。P1/P2 符号是否均支持计划交易方向（共振）。
    仅用于报告/标签「逆势」，**不**再作为可操作门槛（选项：允许逆势参与但须明示）。
    """
    if direction == "long":
        return fund > 0 and p2 > 0
    if direction == "short":
        return fund < 0 and p2 < 0
    return False


def build_phase1_summary(
    top_n: int,
    sort_field: str = "attention_score",
    label_field: str = "labels",
) -> dict[str, object]:
    """返回 Phase 1 机会发现摘要，供报告和 JSON payload 使用。"""
    sort_field_cn = {
        "attention_score": "关注优先级分",
        "reversal_score": "反转机会分",
        "trend_score": "趋势机会分",
    }
    label_values = {
        "labels": ["反转候选", "趋势候选", "双标签候选", "数据覆盖不足"],
        "state_labels": ["低位出清", "高位扩产", "紧平衡强化", "过剩深化"],
    }
    return {
        "阶段": "Phase 1 发现机会",
        "关注池上限": int(top_n),
        "排序字段": sort_field_cn.get(sort_field, "关注优先级分"),
        "标签字段": list(label_values.get(label_field, label_values["labels"])),
    }


def phase_2_premarket(
    candidates: list[dict],
    config: dict,
    max_picks: int = 6,
) -> tuple[list[dict], list[dict]]:
    """
    返回 (actionable, watchlist)：
      actionable: 满足入场条件（评分达标 + 盈亏比 >= 1.0）
      watchlist:  评分有倾向但不满足入场条件（今日观望）
    """
    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 2: 盘前深度分析")
    log.info("▓" * 60)

    pre_cfg = config.get("pre_market", {})
    actionable = []
    watchlist = []

    for cand in candidates:
        try:
            fund_parts = []
            if "range_pct" in cand:
                fund_parts.append(f"区间位{cand['range_pct']:.0f}%")
            if cand.get("inv_change_4wk") is not None:
                fund_parts.append(f"库存4周{cand['inv_change_4wk']:+.1f}%")
            if cand.get("inv_percentile") is not None:
                fund_parts.append(f"库存分位{cand['inv_percentile']:.0f}%")
            if cand.get("receipt_change") is not None:
                fund_parts.append(f"仓单{cand['receipt_change']:+.0f}")
            if cand.get("fund_details"):
                fund_parts.append(cand["fund_details"])
            fund_reason = ", ".join(fund_parts) if fund_parts else ""
            phase2_df = get_daily(cand["symbol"])
            phase2_scores = score_signals(phase2_df, "long", pre_cfg)
            long_score = float(sum(v for v in phase2_scores.values() if v > 0))
            short_score = float(sum(-v for v in phase2_scores.values() if v < 0))
            resolved_direction = resolve_phase2_direction(
                long_score=long_score,
                short_score=short_score,
                delta=float(pre_cfg.get("direction_delta", 12.0)),
            )
            analysis_direction = resolved_direction
            if analysis_direction == "watch":
                analysis_direction = "long" if long_score >= short_score else "short"

            merged_cfg = {
                **pre_cfg,
                "reason": f"基本面筛选{cand['score']:+.0f}, {fund_reason}",
                "fund_screen_score": cand.get("score", 0),
                "phase2_decision": resolved_direction,
            }
            result = analyze_one(
                symbol=cand["symbol"],
                name=cand["name"],
                direction=analysis_direction,
                cfg=merged_cfg,
            )
            if result:
                result["direction"] = analysis_direction
                result["attention_score"] = float(cand.get("attention_score", abs(cand.get("score", 0))))
                result["reversal_score"] = float(cand.get("reversal_score", 0.0))
                result["trend_score"] = float(cand.get("trend_score", 0.0))
                result["phase1_labels"] = list(cand.get("labels") or [])
                result["phase1_state_labels"] = list(cand.get("state_labels") or [])
                result["phase1_data_coverage"] = float(cand.get("data_coverage", 0.0))
                result["phase1_reason_summary"] = cand.get("reason_summary") or cand.get("entry_pool_reason", "")
                result.setdefault("labels", result["phase1_labels"])
                result.setdefault("state_labels", result["phase1_state_labels"])
                result.setdefault("data_coverage", result["phase1_data_coverage"])
                result.setdefault("reason_summary", result["phase1_reason_summary"])
                result["fund_range_pct"] = cand.get("range_pct", 0)
                result["fund_inv_change"] = cand.get("inv_change_4wk")
                result["fund_inv_percentile"] = cand.get("inv_percentile")
                result["fund_receipt_change"] = cand.get("receipt_change")
                result["fund_seasonal"] = cand.get("seasonal_signal")
                result["fund_hog_profit"] = cand.get("hog_profit")
                result["fund_screen_score"] = cand.get("score", 0)
                result["fund_details"] = cand.get("fund_details", "")
                result["signal_strength"] = result.get("reversal_status", {}).get("signal_strength", 0.0)
                result["phase2_decision"] = resolved_direction
                result["score_signs_support_direction"] = _phase_scores_support_direction(
                    analysis_direction,
                    float(result.get("fund_screen_score", 0)),
                    float(result.get("score", 0)),
                )
                result["entry_pool_reason"] = cand.get("entry_pool_reason", "")

                if result["actionable"]:
                    actionable.append(result)
                else:
                    watchlist.append(result)
        except Exception as e:
            log.warning("\n  ⚠️ %s 分析失败: %s", cand["name"], e)

    # RRF (Reciprocal Rank Fusion): 按排名融合 P1 和 P2
    # k=10 适合候选数 10~30 的场景，让 top 排名拉开差距
    RRF_K = 10
    DIRECTION_PENALTY = 0.5  # P1/P2 方向冲突时 RRF 打五折
    all_results = actionable + watchlist
    if all_results:
        p1_ranked = sorted(all_results, key=lambda x: abs(x.get("fund_screen_score", 0)), reverse=True)
        p2_ranked = sorted(all_results, key=lambda x: abs(x.get("score", 0)), reverse=True)
        p1_rank = {id(r): i + 1 for i, r in enumerate(p1_ranked)}
        p2_rank = {id(r): i + 1 for i, r in enumerate(p2_ranked)}
        for r in all_results:
            r1 = p1_rank[id(r)]
            r2 = p2_rank[id(r)]
            raw_rrf = 1.0 / (RRF_K + r1) + 1.0 / (RRF_K + r2)
            p1_score = r.get("fund_screen_score", 0)
            p2_score = r.get("score", 0)
            if p1_score != 0 and p2_score != 0 and (p1_score > 0) != (p2_score > 0):
                r["rrf_score"] = raw_rrf * DIRECTION_PENALTY
                r["direction_conflict"] = True
            else:
                r["rrf_score"] = raw_rrf
                r["direction_conflict"] = False
            r["rank_p1"] = r1
            r["rank_p2"] = r2

    actionable.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    actionable = actionable[:max_picks]
    watchlist.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    log.info(f"\n{'='*60}")
    act_count = len(actionable)
    watch_count = len(watchlist)
    log.info("  📋 Phase 2 汇总: %s 个可操作, %s 个观望", act_count, watch_count)
    log.info("%s", "=" * 60)

    if actionable:
        log.info(f"\n  {'':2s}{'品种':8s} {'代码':6s} {'方向':4s} {'RRF':>7s} {'#P1':>4s} {'#P2':>4s} {'基本面':>6s} {'技术面':>6s} {'信号':>6s} {'强度':>5s} {'入场':>8s} {'盈亏比':>6s}")
        log.info(f"  {'─'*88}")
        for a in actionable:
            rev = a.get("reversal_status", {})
            sig = rev.get("signal_type", "")
            sig_cn = {"Spring": "弹簧", "SOS": "SOS", "SC": "SC",
                      "UT": "UT", "SOW": "SOW", "BC": "BC",
                      "StopVol_Bull": "停止量", "StopVol_Bear": "停止量",
                      "Pullback": "回撤", "TrendBreak": "突破"}.get(sig, sig)
            dir_str = "做多" if a["direction"] == "long" else "做空"
            rrf = a.get("rrf_score", 0)
            r1 = a.get("rank_p1", 0)
            r2 = a.get("rank_p2", 0)
            p1 = a.get("fund_screen_score", 0)
            ss = a.get("signal_strength", 0)
            log.info(
                f"  🟢{a['name']:8s} {a['symbol']:6s} {dir_str:4s} "
                f"{rrf:>6.4f} {r1:>4d} {r2:>4d} {p1:>+5.0f} {a['score']:>+5.0f} "
                f"{sig_cn:6s} {ss:>4.2f} {a['entry']:>8.0f} {a['rr']:>5.1f}"
            )
    if watchlist:
        log.info(f"\n  {'':2s}{'品种':8s} {'代码':6s} {'方向':4s} {'RRF':>7s} {'#P1':>4s} {'#P2':>4s} {'基本面':>6s} {'技术面':>6s} {'强度':>5s} {'系统提示'}")
        log.info(f"  {'─'*88}")
        top_watch = watchlist[:max(3, max_picks - act_count)]
        for w in top_watch:
            rev = w.get("reversal_status", {})
            next_exp = rev.get("next_expected", "等待反转信号")
            dir_str = "做多" if w["direction"] == "long" else "做空"
            rrf = w.get("rrf_score", 0)
            r1 = w.get("rank_p1", 0)
            r2 = w.get("rank_p2", 0)
            p1 = w.get("fund_screen_score", 0)
            ss = w.get("signal_strength", 0)
            tags = []
            if w.get("direction_conflict"):
                tags.append("P1P2异号")
            if not w.get("score_signs_support_direction", True):
                tags.append("逆势")
            conflict = f" ⚠️{'/'.join(tags)}" if tags else ""
            pool_hint = (w.get("entry_pool_reason") or "")[:28]
            pool_suffix = f" | {pool_hint}" if pool_hint else ""
            log.info(
                f"  👀{w['name']:8s} {w['symbol']:6s} {dir_str:4s} "
                f"{rrf:>6.4f} {r1:>4d} {r2:>4d} {p1:>+5.0f} {w['score']:>+5.0f} "
                f"{ss:>4.2f}  {next_exp}{conflict}{pool_suffix}"
            )
    if not actionable and not watchlist:
        log.info("\n  ⚠️ 没有品种通过深度分析")

    return actionable, watchlist


# ============================================================
#  Phase 3: 盘中监控（分钟数据实时获取）
# ============================================================

def phase_3_intraday(
    targets: list[dict],
    config: dict,
    period: str = "5",
    interval: int = 60,
    watchlist: list[dict] | None = None,
):
    """
    盘中实时监控：对「可操作」品种拉分钟 K 并打印仪表盘；对「观望」品种用日线检测反转信号。

    Parameters
    ----------
    targets
        Phase 2 判定为可操作的品种列表（分钟级 K 线 + `print_dashboard`）。
    watchlist
        观望列表（可选）。非空时对每个品种做日线级 `assess_reversal_status`，
        并与上一轮 `prev_signals` 对比打印「新出现 / 持续 / 消失 / 无信号」等提示。
    config
        全局配置，读取 `intraday` 段传入 `print_dashboard`。
        若配置 `tqsdk.account` / `tqsdk.password` 且已安装 tqsdk，则 Phase 3 使用 TqSdk
        连续合约实时 K 线（分钟 + 日线）；否则回退新浪 `get_minute` + `get_daily_with_live_bar`。
    period
        分钟 K 周期字符串（例如 ``"5"`` 表示 5 分钟）。
    interval
        每轮监控结束后的等待秒数；使用 TqSdk 时用 ``wait_update(deadline=...)`` 推进行情。

    Behavior
    --------
    - 开盘前若不在交易时段，先 `time_to_next_session` 等待。
    - 每轮：先处理全部 `targets`，再处理 `watchlist`（若有）。
    - 观望品种每轮结束时在 ``finally`` 中写入 ``prev_signals[sym]``。
    """
    all_monitor = targets + (watchlist or [])
    if not all_monitor:
        log.info("\n  没有可操作/观望品种，跳过盘中监控")
        return

    intraday_cfg = config.get("intraday", {})

    print(f"盘中监控：终端仅打印观望品种反转/信号持续/消失；其余见 {get_log_path()}")
    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 3: 盘中实时监控")
    log.info("▓" * 60)
    act_names = ", ".join(t["name"] for t in targets) if targets else "无"
    watch_names = ", ".join(t["name"] for t in (watchlist or [])) if watchlist else "无"
    log.info("\n  可操作: %s", act_names)
    log.info("  信号观察: %s", watch_names)
    log.info("  K线周期: %s分钟  |  刷新: %s秒", period, interval)

    tq_mon = None
    try:
        tq_mon = try_create_phase3_monitor(config, targets, watchlist, period)
        if tq_mon:
            log.info("\n  数据源: TqSdk 实时K线（%s）", tq_mon.describe_subscription())
        else:
            log.info(
                "\n  数据源: 新浪分钟线 + 合成日线（配置 tqsdk 账户并 pip install tqsdk 可切换）"
            )
    except Exception as e:
        log.warning("\n  ⚠️ TqSdk 初始化失败，回退新浪: %s", e)
        tq_mon = None

    wait = time_to_next_session()
    if wait > 0:
        open_time = (datetime.now() + timedelta(seconds=wait)).strftime("%H:%M")
        log.info("\n  ⏳ 距开盘还有 %s 分钟 (%s)，等待中...", wait // 60, open_time)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            log.info("  → 提前进入监控")

    log.info("\n  🚀 监控启动！按 Ctrl+C 停止\n")

    SIG_CN = {"Spring": "弹簧", "SOS": "强势突破", "UT": "上冲回落", "SOW": "弱势跌破",
              "SC": "卖方高潮", "BC": "买方高潮", "StopVol_Bull": "停止量(多)", "StopVol_Bear": "停止量(空)",
              "Pullback": "顺势回撤", "TrendBreak": "顺势突破"}

    # 记录上轮每个品种的信号状态，用于检测信号出现/消失
    prev_signals: dict[str, bool] = {}

    cycle = 0
    ended_after_trading_hours = False
    try:
        while True:
            try:
                cycle += 1
                now = datetime.now()
                log.info(f"\n{'═'*60}")
                log.info(
                    "  盘中监控 #%s  %s  K线:%s分钟",
                    cycle,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    period,
                )
                log.info("%s", "═" * 60)

                # --- 可操作品种: 分钟级监控 ---
                for t in targets:
                    try:
                        if tq_mon:
                            df = tq_mon.minute_df(t["symbol"])
                        else:
                            df = get_minute(t["symbol"], period)
                        if df is None or df.empty:
                            log.info("\n  %s: 无数据", t["name"])
                            continue
                        print_dashboard(t["symbol"], t["name"], t["direction"], df, intraday_cfg)
                        rev = t.get("reversal_status", {})
                        if rev.get("has_signal"):
                            log.info(
                                "│ 📌 入场信号: %s %s  入场%s  止损%s  目标%s",
                                rev["signal_type"],
                                rev["signal_date"],
                                f"{t['entry']:.0f}",
                                f"{t['stop']:.0f}",
                                f"{t['tp1']:.0f}",
                            )
                    except Exception as e:
                        log.info("\n  %s: ❌ %s", t["name"], e)

                # --- 观望品种: 日线反转检测（TqSdk 用交易所日线序列含当日未完成K线；否则合成日线）---
                if watchlist:
                    log.info("\n  ── 盘中反转检测 ──")
                    for w in watchlist:
                        sym = w["symbol"]
                        has_signal = False
                        try:
                            if tq_mon:
                                live_df = tq_mon.daily_df(sym)
                            else:
                                live_df = get_daily_with_live_bar(sym, period)
                            if live_df is None or live_df.empty:
                                log.warning("  ⚠️ %s: 无数据", w["name"])
                                continue

                            rev = assess_reversal_status(live_df, w["direction"], lookback=60)
                            has_signal = bool(rev.get("has_signal", False))
                            had_signal = prev_signals.get(sym, False)

                            # 当日 K 线信息（TqSdk 日线末根为盘中未完成时即「实时日线」）
                            live_row = live_df.iloc[-1]
                            live_info = (f"当日 O:{live_row['open']:.0f} "
                                         f"H:{live_row['high']:.0f} L:{live_row['low']:.0f} "
                                         f"C:{live_row['close']:.0f}")

                            if has_signal and not had_signal:
                                sig_cn = SIG_CN.get(rev["signal_type"], rev["signal_type"])
                                sig_bar = rev.get("signal_bar", {})
                                dir_cn = "做多" if w["direction"] == "long" else "做空"

                                if w["direction"] == "long":
                                    confirm = f"收盘维持在 {sig_bar.get('low', 0):.0f} 以上"
                                else:
                                    confirm = f"收盘维持在 {sig_bar.get('high', 0):.0f} 以下"

                                aligned = w.get("score_signs_support_direction", True)
                                pool_r = w.get("entry_pool_reason") or ""
                                pool_line = f"       入池理由: {pool_r}" if pool_r else ""
                                if aligned:
                                    print(f"  🔔🔔 {w['name']} 出现反转信号（P1/P2 与计划方向共振）")
                                    print(f"       信号: {sig_cn} | 计划方向: {dir_cn}")
                                    if pool_line:
                                        print(pool_line)
                                    print(f"       {live_info}")
                                    print(f"       确认条件: {confirm}")
                                    print(f"       置信度: {rev['confidence']:.0%}")
                                else:
                                    print(f"  🔔 {w['name']} 出现反转信号（逆势：P1/P2 读数与计划方向未共振，见报告「入池理由/⚠️逆势」）")
                                    print(f"       信号: {sig_cn} | 计划方向: {dir_cn}")
                                    if pool_line:
                                        print(pool_line)
                                    print(f"       {live_info}")
                                    print(f"       确认条件: {confirm}")
                                    print(f"       置信度: {rev['confidence']:.0%}")

                            elif has_signal and had_signal:
                                sig_cn = SIG_CN.get(rev["signal_type"], rev["signal_type"])
                                al = w.get("score_signs_support_direction", True)
                                tag = "" if al else "（逆势未解除）"
                                print(f"  🟢 {w['name']}: {sig_cn}信号持续{tag} | {live_info}")

                            elif not has_signal and had_signal:
                                print(f"  ❌ {w['name']}: 信号已消失（价格回落），继续观望")
                                print(f"       {live_info}")

                            else:
                                stage = rev.get("current_stage", "")
                                events = rev.get("all_events", [])
                                today_str = now.strftime("%Y-%m-%d")
                                today_events = [e for e in events if e.get("date") == today_str]

                                if today_events:
                                    te = today_events[-1]
                                    te_cn = SIG_CN.get(te["signal"], te["signal"])
                                    log.info(
                                        "  👀 %s: 当日出现 %s（预备信号）| %s",
                                        w["name"],
                                        te_cn,
                                        live_info,
                                    )
                                else:
                                    log.info("  ⏳ %s: %s | %s", w["name"], stage, live_info)

                        except Exception as e:
                            log.warning("  ⚠️ %s: %s", w["name"], e)
                        finally:
                            prev_signals[sym] = has_signal

                if not is_trading_hours():
                    ended_after_trading_hours = True
                    print("\n  🔔 交易时段已结束，正在关闭行情连接…")
                    log.info(
                        "  非交易时段：正常结束盘中监控（随后断开 TqSdk，无需当作错误处理）"
                    )
                    break

                log.info("\n  ⏳ %s秒后刷新...", interval)
                if tq_mon:
                    tq_mon.wait_interval(float(interval))
                else:
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\n\n  🛑 监控已停止")
                break
            except Exception as e:
                log.exception("\n  ❌ 错误: %s", e)
                time.sleep(10)
    finally:
        if tq_mon:
            tq_mon.close()
            if ended_after_trading_hours:
                log.info("  TqSdk 行情连接已关闭。")


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


def save_targets(
    targets: list[dict],
    watchlist: list[dict] | None = None,
    phase1_summary: dict | None = None,
):
    """
    将 Phase 2 结果落盘：写入当日 ``_targets.json``（供 ``--resume``）与 ``_targets.md``（表格+说明+分品种详情）。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _clean_numpy(targets)
    if watchlist:
        watchlist = _clean_numpy(watchlist)
    else:
        watchlist = []
    if phase1_summary is None:
        phase1_summary = build_phase1_summary(top_n=max(len(targets), len(watchlist)))

    json_path = _today_json_path()
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "targets": targets,
        "watchlist": watchlist,
        "phase1_summary": phase1_summary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

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
        lines.append(
            "> **未满足可操作三要件**：① 有效入场信号 ② Phase 2 达标（做多>+20 / 做空<-20）③ 盈亏比≥1。"
        )
        lines.append(
            "> **方向**由 Phase 2 技术面总分解析得到；Phase 1 只负责给出关注候选、关注分与机会标签。"
            "若 RRF 旁有 **⚠️逆势**，表示 P1/P2 读数与计划方向未共振（常见于摸顶/抄底），"
            "可在有确认信号时参与，但须自行加权风控。"
        )
    else:
        lines.append("## 今日无信号")
        lines.append("")
        lines.append("> 全市场无极端品种，今日无操作/观望机会。")

    def _md_cell(val) -> str:
        """Markdown 表格单元格内若含 | 会拆列，统一换成全角竖线。"""
        if val is None:
            return "-"
        return str(val).replace("|", "｜")

    def _build_fund_str(t):
        fp = []
        if t.get("fund_range_pct") is not None:
            fp.append(f"区间位{t['fund_range_pct']:.0f}%")
        if t.get("fund_inv_change") is not None:
            fp.append(f"库存4周{t['fund_inv_change']:+.1f}%")
        if t.get("fund_inv_percentile") is not None:
            fp.append(f"库存分位{t['fund_inv_percentile']:.0f}%")
        if t.get("fund_receipt_change") is not None:
            fp.append(f"仓单{t['fund_receipt_change']:+.0f}")
        if t.get("fund_seasonal") is not None and abs(t["fund_seasonal"]) > 0.5:
            fp.append(f"季节性{'偏多' if t['fund_seasonal'] > 0 else '偏空'}")
        if t.get("fund_hog_profit") is not None:
            fp.append(f"养殖利润{t['fund_hog_profit']:+.0f}%")
        fd = t.get("fund_details", "")
        if fd and not any(fd in p for p in fp):
            fp.append(fd)
        return ", ".join(fp) if fp else "-"

    if targets:
        lines.append("")
        lines.append(
            "| 状态 | 合约 | 方向 | 关注理由 | 当前价 | 入场信号 | 入场价 | 止损 | 止盈1 | 盈亏比 | RRF | #P1 | #P2 | 关注分 | P2分 | 信号强度 | 机会标签 | Phase1摘要 |"
        )
        lines.append(
            "| :---: | :--- | :---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"
        )
        for t in targets:
            rev = t.get("reversal_status", {})
            sig_cn = {"Spring": "弹簧", "SOS": "强势突破", "SC": "卖方高潮",
                      "UT": "上冲回落", "SOW": "弱势跌破", "BC": "买方高潮",
                      "StopVol_Bull": "停止量", "StopVol_Bear": "停止量",
                      "Pullback": "回撤", "TrendBreak": "突破"}.get(
                rev.get("signal_type", ""), rev.get("signal_type", ""))
            mode_tag = "🔄" if rev.get("entry_mode") == "trend" else ""
            signal_str = f"{mode_tag}{sig_cn} {rev['signal_date'][-5:]}" if rev.get("has_signal") else "⏳"
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = t.get("signal_strength", 0)
            tag_cells = []
            if t.get("direction_conflict"):
                tag_cells.append("P1P2异号")
            if not t.get("score_signs_support_direction", True):
                tag_cells.append("逆势")
            tag_str = "/".join(tag_cells) if tag_cells else "-"
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            phase1_labels = "/".join(t.get("phase1_labels") or t.get("labels") or []) or "-"
            phase1_summary = _md_cell(t.get("phase1_reason_summary") or t.get("reason_summary") or "-")
            risk_suffix = f" ⚠️{tag_str}" if tag_cells else ""
            rrf_cell = _md_cell(f"**{rrf:.4f}**{risk_suffix}")
            name_cell = _md_cell(f"{t['name']}(主力)")
            dir_cell = _md_cell(dir_icon)
            lines.append(
                f"| {_md_cell('✅入场')} | {name_cell} | {dir_cell} "
                f"| {epr} | {t['price']:.0f} | {_md_cell(signal_str)} | {t['entry']:.0f} | {t['stop']:.0f} "
                f"| {t['tp1']:.0f} | {t['rr']:.2f} "
                f"| {rrf_cell} | {r1} | {r2} "
                f"| {t.get('attention_score', t.get('fund_screen_score', 0)):+.1f} | {t.get('score', 0):+.1f} "
                f"| {ss:.2f} | {_md_cell(phase1_labels)} | {phase1_summary} |"
            )

    if watchlist:
        lines.append("")
        lines.append(
            "| 合约 | 方向 | 关注理由 | 当前价 | 系统提示 | RRF | #P1 | #P2 | 关注分 | P2分 | 信号强度 | 机会标签 | Phase1摘要 |"
        )
        lines.append("| :--- | :---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
        for t in watchlist:
            rev = t.get("reversal_status", {})
            next_exp = rev.get("next_expected", "等待反转信号")
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = t.get("signal_strength", 0)
            wtags = []
            if t.get("direction_conflict"):
                wtags.append("P1P2异号")
            if not t.get("score_signs_support_direction", True):
                wtags.append("逆势")
            conflict_tag = f" ⚠️{'/'.join(wtags)}" if wtags else ""
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            phase1_labels = "/".join(t.get("phase1_labels") or t.get("labels") or []) or "-"
            phase1_summary = _md_cell(t.get("phase1_reason_summary") or t.get("reason_summary") or "-")
            rrf_cell = _md_cell(f"**{rrf:.4f}**{conflict_tag}")
            name_cell = _md_cell(f"{t['name']}(主力)")
            dir_cell = _md_cell(dir_icon)
            lines.append(
                f"| {name_cell} | {dir_cell} "
                f"| {epr} | {t['price']:.0f} | {_md_cell(f'⏳{next_exp}')} "
                f"| {rrf_cell} | {r1} | {r2} "
                f"| {t.get('attention_score', t.get('fund_screen_score', 0)):+.1f} | {t.get('score', 0):+.1f} "
                f"| {ss:.2f} | {_md_cell(phase1_labels)} | {phase1_summary} |"
            )

    lines.append("")
    lines.append("## 评分说明")
    lines.append("")
    lines.append("### Phase 1 机会发现评分体系")
    lines.append("")
    lines.append("所有评分统一约定：**正分=偏多机会，负分=偏空机会**。分值越大代表关注优先级越高。")
    lines.append("")
    lines.append(
        "Phase 1 负责发现机会并输出关注优先级，通过**各品种基本面门槛**或极端价格位（<5% 或 >95%）进入候选池。"
        "它不直接定方向，方向由 Phase 2 结合技术面与信号时机决定。门槛由 config 分层配置（默认与各品种分类），并非单一固定「数值≥10」规则。"
    )
    lines.append("")
    lines.append("| 指标 | 满分 | 数据来源 | 含义 |")
    lines.append("| :--- | ---: | :--- | :--- |")
    lines.append("| 库存变化 | ±20 | 东方财富(~30品种) | 去库=供不应求=+关注分, 累库=供过于求=-关注分 |")
    lines.append("| 库存分位 | ±10 | 东方财富 | 当前库存在52周范围内的位置。库存低位=更高关注优先级, 库存高位=更高空头关注优先级 |")
    lines.append("| 仓单变化 | ±15 | 上期所+广期所仓单 | 仓单减少=供应收紧=+关注分, 仓单增加=供应压力=-关注分 |")
    lines.append("| 季节性 | ±10 | 历史日线 | 当前月份的历史涨跌概率与平均收益率 |")
    lines.append("| 生猪专项 | ±15 | 卓创资讯 | 仅LH0: 养殖亏损抬升低位关注，养殖盈利抬升高位关注 |")
    lines.append("")
    lines.append("### Phase 2 定方向和时机评分体系")
    lines.append("")
    lines.append("| 维度 | 满分 | 说明 |")
    lines.append("| :--- | ---: | :--- |")
    lines.append("| 技术面 | ±55 | 均线排列、MACD、RSI、布林带、动量、价格位置 |")
    lines.append("| 量价面 | ±35 | Wyckoff阶段、量价关系、VSA信号 |")
    lines.append("| 持仓面 | ±15 | OI价格四象限、OI背离 |")
    lines.append("")
    lines.append(
        "可操作条件：① 有新鲜反转/顺势入场信号 ② Phase 2 总分达标(做多>+20/做空<-20) ③ 盈亏比≥1.0。"
    )
    lines.append(
        "**计划方向**来自 Phase 2 对技术面总分的解析；Phase 1 仅提供关注候选、关注分与机会标签。"
        "若标签含 **逆势**，表示 P1/P2 未与计划方向共振，属反转/摸顶情境，须额外风控。"
    )
    lines.append("")
    lines.append("**两阶段评分**：正分=偏多机会, 负分=偏空机会。P1 与 P2 同号时通常更可靠；RRF 对 P1/P2 异号有惩罚。")
    lines.append("")
    lines.append("### RRF 综合排名 (Reciprocal Rank Fusion)")
    lines.append("")
    lines.append("品种最终排名通过 RRF 算法融合 Phase 1 和 Phase 2 的排名，公式:")
    lines.append("")
    lines.append("> `RRF(d) = 1/(k + rank_P1) + 1/(k + rank_P2)`  (k=10)")
    lines.append("")
    lines.append("- 各 Phase 按 |评分| 绝对值降序独立排名(#1=最强)")
    lines.append("- RRF 不依赖评分的绝对值或尺度，只看排名位次")
    lines.append("- 两个 Phase 都排名靠前的品种，RRF 得分最高")
    lines.append("- 只有一个 Phase 排名高的品种，也能保留但排名靠后")
    lines.append("- **方向一致性惩罚**: 若 P1(基本面)与 P2(技术面)方向冲突(一正一负)，RRF 得分×0.5")
    lines.append("")
    lines.append("### 入场信号说明")
    lines.append("")
    lines.append("系统支持两种入场模式，反转信号优先级高于顺势信号：")
    lines.append("")
    lines.append("#### 模式一：Wyckoff 反转入场（抓顶抄底）")
    lines.append("")
    lines.append("适用于横盘/筑底/筑顶阶段，等待极端事件确认反转：")
    lines.append("")
    lines.append("**做多反转序列**: 下跌 → SC(卖方高潮) → 停止量 → Spring(弹簧) → SOS(强势突破)")
    lines.append("**做空反转序列**: 上涨 → BC(买方高潮) → 停止量 → UT(上冲回落) → SOW(弱势跌破)")
    lines.append("")
    lines.append("| 信号 | 含义 | 入场级别 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| SC/BC | 放量极端K线，趋势力量耗尽 | 预备信号，不入场 |")
    lines.append("| 停止量 | 放量但幅度窄，对手方被吸收 | 预备信号，不入场 |")
    lines.append("| Spring/UT | 假突破后收回，经典入场点 | **入场信号** |")
    lines.append("| SOS/SOW | 放量突破确认趋势反转 | **确认信号** |")
    lines.append("")
    lines.append("入场价 = 当前价（信号新鲜时入场） | 止损 = 信号K线极端价 ± 0.5ATR")
    lines.append("")
    lines.append("#### 模式二：顺势入场（趋势延续）")
    lines.append("")
    lines.append("适用于趋势已确立（Wyckoff=markup/markdown）且基本面+技术面一致时：")
    lines.append("")
    lines.append("| 信号 | 条件 | 信号强度 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| 回撤入场(Pullback) | 价格回到MA20附近 + 缩量反弹/回踩 + 未能站稳MA20 | 0.70 |")
    lines.append("| 突破入场(TrendBreak) | 放量(量比≥1.3)突破20日新高/新低 | 0.60 |")
    lines.append("")
    lines.append("前提条件：Wyckoff阶段匹配方向 + |P2评分|≥25 + Phase 2 方向明确")
    lines.append("")
    lines.append("止损 = 近5日极值 ± 0.5ATR | 回撤优先于突破（盈亏比更好）")
    lines.append("")
    lines.append("### 关键术语")
    lines.append("")
    lines.append("- **盈亏比** = |入场价-止盈价1| / |入场价-止损价|。>=2理想，>=1可接受，<1不建议入场")
    lines.append("- **止盈价1**：最近的支撑/阻力位或Fib回撤位（保守目标）")
    lines.append("- **止盈价2**：更远的支撑/阻力位或Fib回撤位（进取目标）")
    lines.append("- **库存分位**：当前库存在近52周最高与最低之间的百分比位置")
    lines.append("- **仓单**：交易所注册仓单量。仓单增加意味着现货被注册为可交割品，反映卖方意愿增强")
    lines.append("- **持仓四象限**：价格涨+OI增=新多入场(偏多)、价格跌+OI增=新空入场(偏空)、价格涨+OI减=空头回补(中性偏多)、价格跌+OI减=多头平仓(中性偏空)")
    lines.append("- **季节性**：该品种在当前月份的历史平均涨跌幅和上涨概率")
    lines.append("")

    def _build_detail(t, status):
        name = t["name"]
        direction_cn = "做多" if t["direction"] == "long" else "做空"
        status_text = f" — {status}" if status else ""
        lines.append(f"### {_md_cell(name)}(主力) — {direction_cn}{status_text}")
        lines.append("")
        epr = t.get("entry_pool_reason")
        if epr:
            lines.append(f"**关注理由（Phase 1）**: {_md_cell(epr)}")
            lines.append("")
        phase1_labels = t.get("phase1_labels") or t.get("labels") or []
        if phase1_labels:
            lines.append(f"**机会标签（Phase 1）**: {_md_cell('/'.join(phase1_labels))}")
            lines.append("")
        aln = t.get("score_signs_support_direction", True)
        if not aln:
            lines.append("**⚠️ 逆势**: P1/P2 综合分符号与计划方向未共振；表内「方向」来自 Phase 2 技术面解析，不是 Phase 1 分栏。")
            lines.append("")

        # --- 入场信号状态 ---
        rev = t.get("reversal_status", {})
        if rev.get("has_signal"):
            sig_cn = {"Spring": "弹簧(Spring)", "SOS": "强势突破(SOS)",
                      "SC": "卖方高潮(SC)", "UT": "上冲回落(UT)",
                      "SOW": "弱势跌破(SOW)", "BC": "买方高潮(BC)",
                      "StopVol_Bull": "停止量(多方力竭)", "StopVol_Bear": "停止量(空方力竭)",
                      "Pullback": "顺势回撤(Pullback)", "TrendBreak": "顺势突破(TrendBreak)",
                      }.get(rev["signal_type"], rev["signal_type"])
            mode_label = "[顺势]" if rev.get("entry_mode") == "trend" else "[反转]"
            lines.append(f"**入场信号**: {mode_label} {sig_cn} ({rev['signal_date']})")
            lines.append(f"- 信号详情: {_md_cell(rev.get('signal_detail', '') or '')}")
            lines.append(f"- 阶段判断: {_md_cell(rev['current_stage'])}")
            lines.append(f"- 置信度: {rev['confidence']:.0%}")
            lines.append("")

            lines.append("**交易参数**")
            lines.append("")
            lines.append(f"| 当前价 | 入场价 | 止损价 | 止盈1 | 止盈2 | 盈亏比 |")
            lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: |")
            lines.append(f"| {t['price']:.0f} | {t['entry']:.0f} | {t['stop']:.0f} "
                         f"| {t['tp1']:.0f} | {t['tp2']:.0f} | {t['rr']:.2f} |")
            lines.append(f"- 入场依据: 当前价（信号{rev['signal_date']}触发，新鲜可入）")
            if t["direction"] == "long":
                lines.append(f"- 止损依据: 信号K线最低点 - 0.5×ATR")
            else:
                lines.append(f"- 止损依据: 信号K线最高点 + 0.5×ATR")
        else:
            lines.append(f"**入场信号**: ⏳ 尚无有效入场信号")
            lines.append(f"- 当前阶段: {_md_cell(rev.get('current_stage', '未知'))}")
            lines.append(f"- 等待条件: {_md_cell(rev.get('next_expected', '等待入场信号'))}")
            suspect = rev.get("suspect_events", [])
            if suspect:
                lines.append(f"- ⚠️ 有{len(suspect)}个疑似信号（未通过上下文验证）:")
                for se in suspect[-3:]:
                    lines.append(
                        f"  - {_md_cell(se.get('date', ''))} {_md_cell(se.get('signal', ''))}: {_md_cell(se.get('detail', ''))}"
                    )
            if t["direction"] == "long":
                lines.append(f"- 反转入场: Spring(假跌破收回) 或 SOS(放量突破)")
                lines.append(f"- 顺势入场: 缩量回踩MA20企稳 或 放量突破前高")
            else:
                lines.append(f"- 反转入场: UT(假突破回落) 或 SOW(放量跌破)")
                lines.append(f"- 顺势入场: 缩量反弹MA20失败 或 放量跌破前低")
            lines.append("")
            lines.append("**交易参数**: 无（需等待信号触发后计算）")

        lines.append("")

        lines.append("**评分分解**")
        lines.append("")
        rrf = t.get("rrf_score", 0)
        r1 = t.get("rank_p1", "-")
        r2 = t.get("rank_p2", "-")
        conflict_note = " ⚠️ 方向冲突(已惩罚×0.5)" if t.get("direction_conflict") else ""
        lines.append(f"- **RRF 综合: {rrf:.4f}** (P1排名#{r1} + P2排名#{r2}){conflict_note}")
        lines.append(f"- Phase 2 综合: {t.get('score', 0):+.1f} "
                     f"(技术{t.get('classical_score', 0):+.1f} / "
                     f"量价{t.get('wyckoff_score', 0):+.1f} / "
                     f"持仓{t.get('oi_score', 0):+.1f})")
        fs = t.get("fund_screen_score")
        if fs is not None:
            lines.append(f"- Phase 1 关注分: {t.get('attention_score', fs):+.0f}")
        ss = t.get("signal_strength", 0)
        lines.append(f"- 信号强度: {ss:.2f}")
        if not t.get("actionable", True):
            rev_info = t.get("reversal_status", {})
            reasons = []
            if not rev_info.get("has_signal"):
                reasons.append("无有效入场信号")
            else:
                rr = t.get("rr", 0)
                score_val = t.get("score", 0)
                ddir = t.get("direction")
                if rr < 1.0:
                    reasons.append(f"盈亏比{rr:.2f}<1.0")
                if ddir == "long" and score_val <= 20:
                    reasons.append(f"P2={score_val:+.0f}未达做多阈值(>+20)")
                elif ddir == "short" and score_val >= -20:
                    reasons.append(f"P2={score_val:+.0f}未达做空阈值(<-20)")
            if reasons:
                lines.append(f"- ⚠️ 未达入场条件: {_md_cell(', '.join(reasons))}")
        lines.append("")

        lines.append("**基本面数据**")
        lines.append("")
        has_any = False
        if t.get("fund_range_pct") is not None:
            lines.append(f"- 价格区间位: {t['fund_range_pct']:.0f}%（年度高低点间的位置）")
            has_any = True
        if t.get("fund_inv_change") is not None:
            lines.append(f"- 库存4周变化: {t['fund_inv_change']:+.1f}%")
            has_any = True
        if t.get("fund_inv_percentile") is not None:
            lines.append(f"- 库存52周分位: {t['fund_inv_percentile']:.0f}%")
            has_any = True
        if t.get("fund_receipt_change") is not None:
            lines.append(f"- 仓单日增减: {t['fund_receipt_change']:+.0f}")
            has_any = True
        if t.get("fund_seasonal") is not None:
            sig = t["fund_seasonal"]
            if abs(sig) > 0.3:
                lines.append(f"- 季节性信号: {'偏多' if sig > 0 else '偏空'} (强度{abs(sig):.1f})")
                has_any = True
        if t.get("fund_hog_profit") is not None:
            lines.append(f"- 养殖利润率: {t['fund_hog_profit']:+.0f}%")
            has_any = True
        fd = t.get("fund_details", "")
        if fd:
            lines.append(f"- 其他: {_md_cell(fd)}")
            has_any = True
        if not has_any:
            lines.append("- 无可用基本面数据（该品种暂无库存/仓单API覆盖）")
        lines.append("")

        phase = t.get("wyckoff_phase", "")
        if phase:
            phase_cn = {"accumulation": "吸筹", "distribution": "派发",
                        "markup": "上涨", "markdown": "下跌"}.get(phase, phase)
            lines.append(f"**Wyckoff阶段**: {_md_cell(phase_cn)}")
            lines.append("")
        reason = t.get("reason", "")
        if reason:
            lines.append(f"**主要依据**: {_md_cell(reason)}")
            lines.append("")

    lines.append("## 品种详细建议")
    lines.append("")
    for t in targets:
        _build_detail(t, "")
    for t in watchlist:
        _build_detail(t, "观望")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("\n  💾 报告已保存:")
    log.info("     📄 %s", md_path)
    log.info("     📊 %s", json_path)


def load_targets() -> tuple[list[dict], list[dict]] | None:
    """
    读取当日 ``_targets.json``。成功且日期为今天时返回 ``(targets, watchlist)``（可为空列表）；
    文件缺失、解析失败或日期不符时返回 ``None``。
    """
    json_path = _today_json_path()
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text())
        if payload.get("date") == datetime.now().strftime("%Y-%m-%d"):
            targets = payload.get("targets") or []
            watchlist = payload.get("watchlist") or []
            return (targets, watchlist)
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=10,
        help="基本面筛选分数门槛 (默认10, 仅在 config.yaml 未定义 default_threshold 时生效)",
    )
    parser.add_argument("--max-picks", type=int, default=6, help="最多跟踪品种数")
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"], help="K线周期")
    parser.add_argument("--interval", type=int, default=60, help="监控刷新间隔(秒)")
    args = parser.parse_args()

    config = load_config()
    now = datetime.now()

    print(f"每日工作流：盘前/筛选/深度分析写入 {get_log_path()}；盘中终端仅输出观望反转警报。")
    log.info("╔" + "═" * 58 + "╗")
    log.info("║" + "每日交易工作流".center(48) + "║")
    log.info("║" + f"  {now.strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    log.info("╚" + "═" * 58 + "╝")

    # --- 恢复模式 ---
    if args.resume:
        snapshot = load_targets()
        if snapshot is not None:
            targets, watchlist_resumed = snapshot
            log.info(
                "\n  📂 恢复今日目标: %s 个可操作, %s 个观望",
                len(targets),
                len(watchlist_resumed),
            )
            if not targets and not watchlist_resumed:
                log.info(
                    "     （当日缓存无品种条目；仍将进入监控流程或按无标的退出）"
                )
            for t in targets:
                dir_str = "做多" if t["direction"] == "long" else "做空"
                log.info(
                    "     %s (%s) %s  评分%s",
                    t["name"],
                    t["symbol"],
                    dir_str,
                    f"{t['score']:+.0f}",
                )
            for w in watchlist_resumed:
                dir_str = "做多" if w["direction"] == "long" else "做空"
                log.info(
                    "     👀 %s (%s) %s  评分%s [观望]",
                    w["name"],
                    w["symbol"],
                    dir_str,
                    f"{w['score']:+.0f}",
                )
            if not args.no_monitor:
                phase_3_intraday(
                    targets,
                    config,
                    args.period,
                    args.interval,
                    watchlist=watchlist_resumed,
                )
            return
        log.info("\n  ⚠️ 无今日缓存，重新筛选")

    # --- Phase 0: 数据预加载 ---
    if args.skip_screen:
        all_data = {}
        positions = config.get("positions", {})
        for sym in positions:
            df = get_daily(sym)
            if df is not None:
                all_data[sym] = df
        log.info("\n  ⏩ 跳过筛选，加载 config 中 %s 个品种", len(all_data))
    else:
        all_data = phase_0_prefetch()

    # --- Phase 1: 筛选 ---
    if args.skip_screen:
        positions = config.get("positions", {})
        candidates = []
        for sym, pos in positions.items():
            score = 50 if pos["direction"] == "long" else -50
            candidates.append({
                "symbol": sym, "name": pos["name"],
                "score": score,
                "range_pct": 0,
                "attention_score": abs(score),
                "reversal_score": 0.0,
                "trend_score": abs(score),
                "labels": ["趋势候选"],
                "state_labels": [],
                "data_coverage": 1.0,
                "reason_summary": "配置文件(--skip-screen)",
                "entry_pool_reason": "配置文件(--skip-screen)",
                "phase1_labels": ["趋势候选"],
                "phase1_state_labels": [],
                "phase1_data_coverage": 1.0,
                "phase1_reason_summary": "配置文件(--skip-screen)",
            })
    else:
        candidates = phase_1_screen(all_data, threshold=args.threshold)

    if not candidates:
        log.info("\n  😴 全市场没有极端品种，今日无操作机会")
        return

    # --- Phase 2: 深度分析 ---
    actionable, watchlist_result = phase_2_premarket(
        candidates, config, max_picks=args.max_picks
    )

    top_watch = watchlist_result[:max(3, args.max_picks - len(actionable))]
    phase1_summary = build_phase1_summary(
        top_n=config.get("phase1", {}).get("top_n", args.max_picks),
        sort_field="attention_score",
        label_field="labels",
    )
    save_targets(actionable, top_watch, phase1_summary=phase1_summary)

    # --- Phase 3: 盘中监控 ---
    if args.no_monitor:
        log.info("\n  ⏩ 跳过盘中监控 (--no-monitor)")
        api_total = get_request_count()
        log.info("  📊 本次运行总API调用: %s 次", api_total)
        return

    phase_3_intraday(actionable, config, args.period, args.interval, watchlist=top_watch)


if __name__ == "__main__":
    main()
