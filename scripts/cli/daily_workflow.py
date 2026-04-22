#!/usr/bin/env python3
"""
每日自动化交易工作流
========================================

完整流水线:
  Phase 0 — 数据预加载 (盘前)
    一次性拉取所有品种日线，缓存到本地 parquet
    当日首次运行会远端抓取日线，后续优先复用当日缓存

  Phase 1 — 全市场筛选 (盘前)
    基于已加载的日线数据分析，不再重复抓取日线

  Phase 2 — 盘前深度分析 (盘前)
    对筛选出的品种做深度分析，不再重复抓取日线

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
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from shared.ctp_log import get_log_path, get_logger
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND

from data_cache import (
    get_all_symbols, get_daily, get_minute,
    prefetch_all_with_stats, get_request_count,
    format_prefetch_summary,
    get_daily_with_live_bar,
)
from phase2.pre_market import (
    analyze_one,
    build_trade_plan_from_daily_df,
    resolve_phase2_direction,
    score_signals,
)
from phase3.intraday import print_dashboard
from wyckoff import assess_reversal_status
from phase3.live import try_create_phase3_monitor
from phase1.pipeline import build_phase1_candidates, is_phase1_candidate_eligible, select_top_candidates
from strategy_reversal.screen import select_reversal_candidates
from strategy_trend.screen import build_trend_universe as build_trend_universe_candidates, select_trend_candidates
from strategy_reversal import pre_market as reversal_pre_market
from strategy_trend import pre_market as trend_pre_market

log = get_logger("workflow")


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
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

    data, stats = prefetch_all_with_stats(symbols)
    log.info("  ✅ %s", format_prefetch_summary(len(data), stats))

    return data


# ============================================================
#  Phase 1: 全市场筛选（纯本地计算）
# ============================================================

def _dominant_phase1_score(row: dict) -> float:
    return float(max(row.get("reversal_score") or 0.0, row.get("trend_score") or 0.0))


def _build_phase1_diagnostics(rows: list[dict], *, selected_rows: list[dict]) -> dict[str, object]:
    factor_hits = {
        "inventory": 0,
        "receipt": 0,
        "seasonality": 0,
        "hog": 0,
    }
    label_counts = {
        "reversal": 0,
        "trend": 0,
        "dual": 0,
        "low_coverage": 0,
    }
    threshold_counts: Counter[float] = Counter()
    num_eligible = 0
    blocked_rows: list[dict] = []
    blocked_reversal_labeled = 0

    for row in rows:
        if row.get("inv_change_4wk") is not None:
            factor_hits["inventory"] += 1
        if row.get("receipt_change") is not None:
            factor_hits["receipt"] += 1
        if row.get("seasonal_signal") is not None:
            factor_hits["seasonality"] += 1
        if row.get("hog_profit") is not None:
            factor_hits["hog"] += 1

        labels = set(row.get("labels") or [])
        if "双标签候选" in labels:
            label_counts["dual"] += 1
        elif "反转候选" in labels:
            label_counts["reversal"] += 1
        elif "趋势候选" in labels:
            label_counts["trend"] += 1
        if row.get("data_coverage", 1.0) < 0.5:
            label_counts["low_coverage"] += 1

        threshold_counts[float(row.get("entry_threshold") or 0.0)] += 1

        eligible = is_phase1_candidate_eligible(row)
        if eligible:
            num_eligible += 1
            continue

        blocked_rows.append(row)
        if "反转候选" in labels or "双标签候选" in labels:
            blocked_reversal_labeled += 1

    blocked_rows.sort(
        key=lambda row: (
            _dominant_phase1_score(row),
            float(row.get("attention_score") or 0.0),
        ),
        reverse=True,
    )
    blocked_preview = [
        {
            "symbol": str(row.get("symbol") or ""),
            "name": str(row.get("name") or ""),
            "dominant_score": _dominant_phase1_score(row),
            "entry_threshold": float(row.get("entry_threshold") or 0.0),
            "labels": list(row.get("labels") or []),
            "reason_summary": str(row.get("reason_summary") or ""),
        }
        for row in blocked_rows[:5]
    ]

    return {
        "num_rows": len(rows),
        "num_eligible": num_eligible,
        "num_selected": len(selected_rows),
        "factor_hits": factor_hits,
        "label_counts": label_counts,
        "threshold_counts": dict(sorted(threshold_counts.items())),
        "blocked_reversal_labeled": blocked_reversal_labeled,
        "blocked_preview": blocked_preview,
    }


def _log_phase1_diagnostics(diag: dict[str, object]) -> None:
    factor_hits = diag.get("factor_hits") or {}
    label_counts = diag.get("label_counts") or {}
    threshold_counts = diag.get("threshold_counts") or {}
    threshold_text = ", ".join(
        f"{threshold:.0f}分x{count}"
        for threshold, count in threshold_counts.items()
    ) or "无"

    log.info(
        "  🧪 Phase 1诊断: 原始候选%s 个, 门槛通过%s 个, TopN保留%s 个",
        diag.get("num_rows", 0),
        diag.get("num_eligible", 0),
        diag.get("num_selected", 0),
    )
    log.info(
        "     因子命中: 库存%s / 仓单%s / 季节性%s / 生猪%s",
        factor_hits.get("inventory", 0),
        factor_hits.get("receipt", 0),
        factor_hits.get("seasonality", 0),
        factor_hits.get("hog", 0),
    )
    log.info(
        "     标签统计: 反转%s / 趋势%s / 双标签%s / 覆盖不足%s",
        label_counts.get("reversal", 0),
        label_counts.get("trend", 0),
        label_counts.get("dual", 0),
        label_counts.get("low_coverage", 0),
    )
    log.info("     门槛分布: %s", threshold_text)

    blocked_reversal_labeled = int(diag.get("blocked_reversal_labeled", 0) or 0)
    if blocked_reversal_labeled:
        log.info("     反转标签但被门槛拦截: %s 个", blocked_reversal_labeled)

    blocked_preview = diag.get("blocked_preview") or []
    if blocked_preview:
        log.info("     未过门槛Top:")
        for row in blocked_preview:
            label_text = "/".join(row.get("labels") or []) or "无"
            log.info(
                "       %s (%s)  分数%.0f < 门槛%.0f  标签=%s  原因=%s",
                row.get("name", ""),
                row.get("symbol", ""),
                float(row.get("dominant_score") or 0.0),
                float(row.get("entry_threshold") or 0.0),
                label_text,
                row.get("reason_summary", ""),
            )


def phase_1_screen(
    all_data: dict[str, pd.DataFrame],
    threshold: float = 10,
) -> list[dict]:
    cfg = load_config()
    fs = cfg.get("fundamental_screening") or {}

    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 1: 全市场基本面筛选")
    log.info("▓  P1引擎: 关注优先级新管线（反转机会分 / 趋势机会分）")
    log.info("▓" * 60)
    rows = build_phase1_candidates(
        all_data=all_data,
        symbols=get_all_symbols(),
        threshold=threshold,
        config=cfg,
    )
    candidates = select_top_candidates(rows, top_n=int(fs.get("top_n") or 40))
    _log_phase1_diagnostics(_build_phase1_diagnostics(rows, selected_rows=candidates))
    reversal_seed_count = len(select_reversal_candidates(candidates, top_n=len(candidates)))

    log.info(
        "\n  ✅ Phase 1关注池（门槛通过）: %s 个品种，其中反转线%s 个\n",
        len(candidates),
        reversal_seed_count,
    )

    def _log_cand(c):
        label_text = "/".join(c.get("labels") or [])
        line = (f"     {c['name']:6s} ({c['symbol']:5s})  "
                f"区间位{c['range_pct']:3.0f}%  "
                f"反转={c.get('reversal_score', 0):.0f}  "
                f"趋势={c.get('trend_score', 0):.0f}  "
                f"关注={c.get('attention_score', 0):.0f}  "
                f"标签={label_text}")
        fd = c.get("fund_details", "")
        if fd:
            line += f"  [{fd}]"
        log.info(line)

    if candidates:
        log.info("  👀 关注候选:")
        for c in candidates[:16]:
            _log_cand(c)

    return candidates


def build_trend_universe(
    all_data: dict[str, pd.DataFrame],
    config: dict,
) -> list[dict]:
    return build_trend_universe_candidates(
        all_data=all_data,
        symbols=get_all_symbols(),
        config=config,
    )


# ============================================================
#  Phase 2: 盘前深度分析（读缓存）
# ============================================================

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


def _build_fund_reason(cand: dict) -> str:
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
    return ", ".join(fund_parts) if fund_parts else ""


def _merge_strategy_result(
    *,
    cand: dict,
    result: dict,
    resolved_direction: str,
    strategy_family: str,
) -> dict:
    merged = dict(result)
    merged["strategy_family"] = merged.get("strategy_family") or strategy_family
    merged["attention_score"] = float(cand.get("attention_score", abs(cand.get("score", 0))))
    merged["reversal_score"] = float(cand.get("reversal_score", 0.0))
    merged["trend_score"] = float(cand.get("trend_score", 0.0))
    merged["phase1_labels"] = list(cand.get("labels") or [])
    merged["phase1_state_labels"] = list(cand.get("state_labels") or [])
    merged["phase1_data_coverage"] = float(cand.get("data_coverage", 0.0))
    merged["phase1_reason_summary"] = cand.get("reason_summary") or cand.get("entry_pool_reason", "")
    merged.setdefault("labels", merged["phase1_labels"])
    merged.setdefault("state_labels", merged["phase1_state_labels"])
    merged.setdefault("data_coverage", merged["phase1_data_coverage"])
    merged.setdefault("reason_summary", merged["phase1_reason_summary"])
    merged["fund_range_pct"] = cand.get("range_pct", 0)
    merged["fund_inv_change"] = cand.get("inv_change_4wk")
    merged["fund_inv_percentile"] = cand.get("inv_percentile")
    merged["fund_receipt_change"] = cand.get("receipt_change")
    merged["fund_seasonal"] = cand.get("seasonal_signal")
    merged["fund_hog_profit"] = cand.get("hog_profit")
    merged["fund_screen_score"] = cand.get("strategy_score", cand.get("attention_score", cand.get("score", 0)))
    merged["fund_details"] = cand.get("fund_details", "")
    merged["signal_strength"] = merged.get("reversal_status", {}).get("signal_strength", 0.0)
    merged["phase2_decision"] = resolved_direction
    merged["entry_pool_reason"] = cand.get("entry_pool_reason", "")
    return merged


def _apply_strategy_rrf(rows: list[dict], *, sort_field: str) -> None:
    if not rows:
        return
    rrf_k = 10
    p1_ranked = sorted(rows, key=lambda item: item.get(sort_field, item.get("attention_score", 0)), reverse=True)
    p2_ranked = sorted(rows, key=lambda item: abs(item.get("score", 0)), reverse=True)
    p1_rank = {id(row): idx + 1 for idx, row in enumerate(p1_ranked)}
    p2_rank = {id(row): idx + 1 for idx, row in enumerate(p2_ranked)}
    for row in rows:
        rank_1 = p1_rank[id(row)]
        rank_2 = p2_rank[id(row)]
        row["rrf_score"] = 1.0 / (rrf_k + rank_1) + 1.0 / (rrf_k + rank_2)
        row["rank_p1"] = rank_1
        row["rank_p2"] = rank_2


def _run_strategy_phase2(
    candidates: list[dict],
    config: dict,
    *,
    max_picks: int,
    strategy_family: str,
    sort_field: str,
    plan_builder,
) -> tuple[list[dict], list[dict]]:
    pre_cfg = config.get("pre_market", {})
    actionable: list[dict] = []
    watchlist: list[dict] = []

    for cand in candidates:
        phase2_df = get_daily(cand["symbol"])
        if phase2_df is None or getattr(phase2_df, "empty", False):
            continue

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

        screen_label = "基本面筛选" if strategy_family == STRATEGY_REVERSAL else "趋势筛选"
        screen_score = float(cand.get("strategy_score", cand.get("attention_score", cand.get("score", 0))))
        screen_reason = cand.get("entry_pool_reason") or _build_fund_reason(cand)
        reason_parts = [f"{screen_label}{screen_score:+.0f}"]
        if screen_reason:
            reason_parts.append(screen_reason)

        merged_cfg = {
            **pre_cfg,
            "reason": ", ".join(reason_parts),
            "fund_screen_score": screen_score,
            "phase2_decision": resolved_direction,
        }
        result = plan_builder(
            symbol=cand["symbol"],
            name=cand["name"],
            direction=analysis_direction,
            df=phase2_df,
            cfg=merged_cfg,
            allow_watch_plan=True,
        )
        if result is None:
            continue

        merged_result = _merge_strategy_result(
            cand=cand,
            result=result,
            resolved_direction=resolved_direction,
            strategy_family=strategy_family,
        )
        if merged_result["actionable"]:
            actionable.append(merged_result)
        else:
            watchlist.append(merged_result)

    all_results = actionable + watchlist
    _apply_strategy_rrf(all_results, sort_field=sort_field)
    actionable.sort(key=lambda item: item.get("rrf_score", 0), reverse=True)
    watchlist.sort(key=lambda item: item.get("rrf_score", 0), reverse=True)
    return actionable[:max_picks], watchlist


def run_reversal_strategy_phase2(candidates: list[dict], config: dict, max_picks: int) -> tuple[list[dict], list[dict]]:
    selected = select_reversal_candidates(candidates, top_n=max(max_picks * 2, max_picks))
    return _run_strategy_phase2(
        selected,
        config,
        max_picks=max_picks,
        strategy_family=STRATEGY_REVERSAL,
        sort_field="reversal_score",
        plan_builder=reversal_pre_market.build_trade_plan_from_daily_df,
    )


def run_trend_strategy_phase2(candidates: list[dict], config: dict, max_picks: int) -> tuple[list[dict], list[dict]]:
    selected = select_trend_candidates(candidates, top_n=max(max_picks * 2, max_picks))
    return _run_strategy_phase2(
        selected,
        config,
        max_picks=max_picks,
        strategy_family=STRATEGY_TREND,
        sort_field="trend_score",
        plan_builder=trend_pre_market.build_trade_plan_from_daily_df,
    )


def run_dual_strategy_phase2(
    reversal_candidates: list[dict],
    trend_candidates: list[dict],
    config: dict,
    max_picks: int = 6,
) -> tuple[list[dict], list[dict], dict[str, dict[str, list[dict]]]]:
    reversal_actionable, reversal_watchlist = run_reversal_strategy_phase2(reversal_candidates, config, max_picks)
    trend_actionable, trend_watchlist = run_trend_strategy_phase2(trend_candidates, config, max_picks)

    actionable = sorted(
        reversal_actionable + trend_actionable,
        key=lambda item: item.get("rrf_score", 0),
        reverse=True,
    )[:max_picks]
    watchlist = sorted(
        reversal_watchlist + trend_watchlist,
        key=lambda item: item.get("rrf_score", 0),
        reverse=True,
    )
    grouped = {
        STRATEGY_REVERSAL: {
            "actionable": reversal_actionable,
            "watchlist": reversal_watchlist,
        },
        STRATEGY_TREND: {
            "actionable": trend_actionable,
            "watchlist": trend_watchlist,
        },
    }
    return actionable, watchlist, grouped


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
                result["fund_screen_score"] = cand.get("attention_score", cand.get("score", 0))
                result["fund_details"] = cand.get("fund_details", "")
                result["signal_strength"] = result.get("reversal_status", {}).get("signal_strength", 0.0)
                result["phase2_decision"] = resolved_direction
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
    all_results = actionable + watchlist
    if all_results:
        p1_ranked = sorted(all_results, key=lambda x: x.get("attention_score", x.get("fund_screen_score", 0)), reverse=True)
        p2_ranked = sorted(all_results, key=lambda x: abs(x.get("score", 0)), reverse=True)
        p1_rank = {id(r): i + 1 for i, r in enumerate(p1_ranked)}
        p2_rank = {id(r): i + 1 for i, r in enumerate(p2_ranked)}
        for r in all_results:
            r1 = p1_rank[id(r)]
            r2 = p2_rank[id(r)]
            raw_rrf = 1.0 / (RRF_K + r1) + 1.0 / (RRF_K + r2)
            r["rrf_score"] = raw_rrf
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
            entry_signal = _resolve_entry_signal(a)
            sig_cn = entry_signal["display_label"] if entry_signal["has_signal"] else "-"
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
            entry_signal = _resolve_entry_signal(w)
            if entry_signal["has_signal"]:
                family_cn = entry_signal["family_cn"]
                signal_label = entry_signal["display_label"]
                next_exp = f"已有{family_cn}{signal_label}，待入场条件改善"
            else:
                next_exp = rev.get("next_expected", _default_waiting_hint(w))
            dir_str = "做多" if w["direction"] == "long" else "做空"
            rrf = w.get("rrf_score", 0)
            r1 = w.get("rank_p1", 0)
            r2 = w.get("rank_p2", 0)
            p1 = w.get("fund_screen_score", 0)
            ss = w.get("signal_strength", 0)
            pool_hint = (w.get("entry_pool_reason") or "")[:28]
            pool_suffix = f" | {pool_hint}" if pool_hint else ""
            log.info(
                f"  👀{w['name']:8s} {w['symbol']:6s} {dir_str:4s} "
                f"{rrf:>6.4f} {r1:>4d} {r2:>4d} {p1:>+5.0f} {w['score']:>+5.0f} "
                f"{ss:>4.2f}  {next_exp}{pool_suffix}"
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
    盘中实时监控：对「可操作」品种拉分钟 K 并打印仪表盘；对「观望」品种做日线级信号检测。

    Parameters
    ----------
    targets
        Phase 2 判定为可操作的品种列表（分钟级 K 线 + `print_dashboard`）。
    watchlist
        观望列表（可选）。非空时对每个品种做日线级 `assess_reversal_status`，
        并与上一轮 `prev_signals` 对比打印「新出现 / 持续 / 消失 / 无信号」等提示；
        具体文案会按 Phase 2 的 `entry_family` 区分顺势/反转。
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

    print(f"盘中监控：终端仅打印观望品种信号出现/持续/消失；其余见 {get_log_path()}")
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
    pre_cfg = config.get("pre_market", {})

    # 记录上轮每个品种的信号状态，用于检测信号出现/消失
    prev_signals: dict[str, bool] = {}
    prev_watch_rows: dict[str, dict] = {}

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
                        entry_signal = _resolve_entry_signal(t)
                        if entry_signal["has_signal"]:
                            signal_date = str(entry_signal["signal_date"])
                            signal_date_part = f" {signal_date}" if signal_date else ""
                            log.info(
                                "│ 📌 入场信号: %s %s  入场%s  止损%s  目标%s",
                                entry_signal["display_label"],
                                signal_date_part,
                                f"{t['entry']:.0f}",
                                f"{t['stop']:.0f}",
                                f"{t['tp1']:.0f}",
                            )
                    except Exception as e:
                        log.info("\n  %s: ❌ %s", t["name"], e)

                # --- 观望品种: 日线信号检测（TqSdk 用交易所日线序列含当日未完成K线；否则合成日线）---
                if watchlist:
                    log.info("\n  ── 盘中信号检测 ──")
                    for w in watchlist:
                        sym = w["symbol"]
                        has_signal = False
                        current_watch = _normalize_phase2_plan_row(w)
                        try:
                            if tq_mon:
                                live_df = tq_mon.daily_df(sym)
                            else:
                                live_df = get_daily_with_live_bar(sym, period)
                            if live_df is None or live_df.empty:
                                log.warning("  ⚠️ %s: 无数据", w["name"])
                                continue

                            previous_watch = prev_watch_rows.get(sym, _normalize_phase2_plan_row(w))
                            live_plan = build_trade_plan_from_daily_df(
                                symbol=sym,
                                name=w["name"],
                                direction=w["direction"],
                                df=live_df,
                                cfg=pre_cfg,
                            )
                            current_watch = _merge_intraday_watch_plan(w, live_plan)
                            rev = current_watch.get("reversal_status")
                            if not isinstance(rev, dict):
                                rev = {}
                            entry_signal = _resolve_entry_signal(current_watch)
                            has_signal = bool(entry_signal["has_signal"])
                            had_signal = prev_signals.get(sym, False)
                            plan_changed = _intraday_watch_plan_changed(previous_watch, current_watch)
                            actionable_upgraded = bool(current_watch.get("actionable")) and not bool(previous_watch.get("actionable"))

                            # 当日 K 线信息（TqSdk 日线末根为盘中未完成时即「实时日线」）
                            live_row = live_df.iloc[-1]
                            live_info = (f"当日 O:{live_row['open']:.0f} "
                                         f"H:{live_row['high']:.0f} L:{live_row['low']:.0f} "
                                         f"C:{live_row['close']:.0f}")

                            if has_signal and not had_signal:
                                signal_noun, sig_cn = _intraday_watch_signal_text(current_watch, str(rev.get("signal_type") or ""))
                                sig_bar = rev.get("signal_bar", {})
                                dir_cn = "做多" if current_watch["direction"] == "long" else "做空"

                                if current_watch["direction"] == "long":
                                    confirm = f"收盘维持在 {sig_bar.get('low', 0):.0f} 以上"
                                else:
                                    confirm = f"收盘维持在 {sig_bar.get('high', 0):.0f} 以下"

                                pool_r = current_watch.get("entry_pool_reason") or ""
                                pool_line = f"       入池理由: {pool_r}" if pool_r else ""
                                print(f"  🔔🔔 {current_watch['name']} 出现{signal_noun}")
                                print(f"       信号: {sig_cn} | 计划方向: {dir_cn}")
                                if pool_line:
                                    print(pool_line)
                                print(f"       {live_info}")
                                print(f"       {_format_intraday_trade_plan(current_watch)}")
                                if plan_changed:
                                    print("       🔄 盘中重评后交易计划已更新")
                                if actionable_upgraded:
                                    print(
                                        f"       ✅ 盘中重评后转为可操作：评分{float(current_watch.get('score', 0.0)):+.0f} "
                                        f"准入RR {float(current_watch.get('admission_rr', current_watch.get('rr', 0.0))):.2f}"
                                    )
                                print(f"       确认条件: {confirm}")
                                print(f"       置信度: {rev['confidence']:.0%}")

                            elif has_signal and had_signal:
                                _, sig_cn = _intraday_watch_signal_text(current_watch, str(rev.get("signal_type") or ""))
                                print(f"  🟢 {current_watch['name']}: {sig_cn}信号持续 | {live_info}")
                                print(f"       {_format_intraday_trade_plan(current_watch)}")
                                if plan_changed:
                                    print("       🔄 盘中重评后交易计划已更新")
                                if actionable_upgraded:
                                    print(
                                        f"       ✅ 盘中重评后转为可操作：评分{float(current_watch.get('score', 0.0)):+.0f} "
                                        f"准入RR {float(current_watch.get('admission_rr', current_watch.get('rr', 0.0))):.2f}"
                                    )

                            elif not has_signal and had_signal:
                                print(f"  ❌ {current_watch['name']}: 信号已消失（价格回落），继续观望")
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
                            prev_watch_rows[sym] = current_watch

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

RESULT_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

ENTRY_SIGNAL_CN = {
    "Spring": "弹簧",
    "SOS": "强势突破",
    "SC": "卖方高潮",
    "UT": "上冲回落",
    "SOW": "弱势跌破",
    "BC": "买方高潮",
    "StopVol_Bull": "停止量",
    "StopVol_Bear": "停止量",
    "Pullback": "顺势回撤",
    "TrendBreak": "顺势突破",
}
ENTRY_FAMILY_CN = {
    "trend": "顺势",
    "reversal": "反转",
}
TREND_ENTRY_SIGNAL_TYPES = {"Pullback", "TrendBreak"}


def _today_md_path() -> Path:
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.md"


def _today_json_path() -> Path:
    return RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_targets.json"


def _clean_numpy(value):
    """递归将 numpy 类型转为 Python 原生类型。"""
    if isinstance(value, dict):
        return {k: _clean_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clean_numpy(item) for item in value)
    if isinstance(value, (np.bool_, np.integer)):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _normalize_phase2_plan_row(row: dict) -> dict:
    normalized = dict(row)
    reversal = normalized.get("reversal_status")
    if not isinstance(reversal, dict):
        reversal = {}

    entry_family = normalized.get("entry_family") or ""
    entry_signal_type = normalized.get("entry_signal_type") or ""
    legacy_signal_type = reversal.get("signal_type", "")
    legacy_entry_mode = reversal.get("entry_mode", "")
    if entry_family not in ENTRY_FAMILY_CN:
        if (
            legacy_entry_mode == "trend"
            or entry_signal_type in TREND_ENTRY_SIGNAL_TYPES
            or legacy_signal_type in TREND_ENTRY_SIGNAL_TYPES
        ):
            entry_family = "trend"
        # 旧数据没有 entry_family 时，只在确认存在反转信号时回填 reversal；
        # 其余情况保持空，避免把无明确信号的旧快照误判成任一入场家族。
        elif reversal.get("has_signal"):
            entry_family = "reversal"
        else:
            entry_family = ""

    if not entry_signal_type:
        if entry_family == "trend" and legacy_signal_type in TREND_ENTRY_SIGNAL_TYPES:
            entry_signal_type = legacy_signal_type
        elif entry_family == "reversal":
            entry_signal_type = legacy_signal_type

    entry_signal_detail = normalized.get("entry_signal_detail") or ""
    if not entry_signal_detail and entry_family in {"trend", "reversal"}:
        entry_signal_detail = reversal.get("signal_detail", "")

    normalized["entry_family"] = entry_family
    normalized["entry_signal_type"] = entry_signal_type
    normalized["entry_signal_detail"] = entry_signal_detail
    return normalized


def _normalize_phase2_plan_rows(rows: list[dict]) -> list[dict]:
    return [_normalize_phase2_plan_row(row) for row in rows]


def _resolve_entry_signal(row: dict) -> dict[str, str | bool]:
    normalized = _normalize_phase2_plan_row(row)
    reversal = normalized.get("reversal_status")
    if not isinstance(reversal, dict):
        reversal = {}

    entry_family = normalized.get("entry_family", "")
    signal_type = normalized.get("entry_signal_type", "")
    signal_detail = normalized.get("entry_signal_detail", "")
    signal_date = reversal.get("signal_date", "") if entry_family == "reversal" else ""
    has_signal = False
    if entry_family == "trend":
        has_signal = bool(signal_type or signal_detail)
    elif entry_family == "reversal":
        has_signal = bool(reversal.get("has_signal") or signal_type or signal_detail)

    signal_cn = ENTRY_SIGNAL_CN.get(signal_type, signal_type)
    family_cn = ENTRY_FAMILY_CN.get(entry_family, "")
    display_label = signal_cn or family_cn or signal_type
    return {
        "entry_family": entry_family,
        "family_cn": family_cn,
        "signal_type": signal_type,
        "signal_cn": signal_cn,
        "signal_detail": signal_detail,
        "signal_date": signal_date,
        "has_signal": has_signal,
        "display_label": display_label,
    }


def _format_entry_signal_for_table(row: dict) -> str:
    entry_signal = _resolve_entry_signal(row)
    if not entry_signal["has_signal"]:
        return "⏳"

    label = str(entry_signal["display_label"])
    if entry_signal["entry_family"] == "trend":
        return f"🔄{label}"

    signal_date = str(entry_signal["signal_date"])
    if signal_date:
        return f"{label} {signal_date[-5:]}"
    return label


def _format_entry_signal_for_detail(row: dict) -> str:
    entry_signal = _resolve_entry_signal(row)
    if not entry_signal["has_signal"]:
        return "⏳ 尚无有效入场信号"

    family_cn = str(entry_signal["family_cn"])
    family_prefix = f"[{family_cn}] " if family_cn else ""
    signal_date = str(entry_signal["signal_date"])
    if signal_date:
        return f"{family_prefix}{entry_signal['display_label']} ({signal_date})"
    return f"{family_prefix}{entry_signal['display_label']}"


def _default_waiting_hint(row: dict) -> str:
    entry_family = _resolve_entry_signal(row)["entry_family"]
    if entry_family == "trend":
        return "等待顺势信号"
    if entry_family == "reversal":
        return "等待反转信号"
    return "等待入场信号"


def _format_intraday_trade_plan(row: dict) -> str:
    try:
        entry = float(row["entry"])
        stop = float(row["stop"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])
        rr = float(row["rr"])
        admission_rr = float(row.get("admission_rr", rr))
    except (KeyError, TypeError, ValueError):
        return "交易计划: 盘前未生成完整交易计划"

    if min(entry, stop, tp1, tp2) <= 0:
        return "交易计划: 盘前未生成完整交易计划"

    return (
        f"交易计划: 入场{entry:.0f} 止损{stop:.0f} 止盈1 {tp1:.0f} "
        f"止盈2 {tp2:.0f} 第一止盈RR {rr:.2f} 准入RR {admission_rr:.2f}"
    )


def _intraday_watch_signal_text(row: dict, fallback_signal_type: str) -> tuple[str, str]:
    entry_signal = _resolve_entry_signal(row)
    entry_family = str(entry_signal["entry_family"])
    fallback_label = ENTRY_SIGNAL_CN.get(fallback_signal_type, fallback_signal_type)

    if entry_family == "trend":
        return "顺势信号", str(entry_signal["display_label"] or fallback_label or "顺势")
    if entry_family == "reversal":
        return "反转信号", str(fallback_label or entry_signal["display_label"] or "反转")
    return "入场信号", str(entry_signal["display_label"] or fallback_label or "入场")


def _merge_intraday_watch_plan(base_row: dict, live_plan: dict | None) -> dict:
    merged = _normalize_phase2_plan_row(base_row)
    if not isinstance(live_plan, dict):
        return merged
    merged.update(live_plan)
    return _normalize_phase2_plan_row(merged)


def _intraday_watch_plan_changed(previous_row: dict, current_row: dict) -> bool:
    return _format_intraday_trade_plan(previous_row) != _format_intraday_trade_plan(current_row)


def save_targets(
    targets: list[dict],
    watchlist: list[dict] | None = None,
    phase1_summary: dict | None = None,
    grouped_results: dict[str, dict[str, list[dict]]] | None = None,
):
    """
    将 Phase 2 结果落盘：写入当日 ``_targets.json``（供 ``--resume``）与 ``_targets.md``（表格+说明+分品种详情）。
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    targets = _normalize_phase2_plan_rows(_clean_numpy(targets))
    if watchlist:
        watchlist = _normalize_phase2_plan_rows(_clean_numpy(watchlist))
    else:
        watchlist = []
    if phase1_summary is None:
        phase1_summary = build_phase1_summary(top_n=max(len(targets), len(watchlist)))

    json_path = _today_json_path()
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "targets": targets,
        "watchlist": watchlist,
        "strategy_results": _clean_numpy(grouped_results or {}),
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
            "机会标签描述的是基本面机会类型，不直接等同于最终交易方向。"
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
            signal_str = _format_entry_signal_for_table(t)
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = t.get("signal_strength", 0)
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            phase1_labels = "/".join(t.get("phase1_labels") or t.get("labels") or []) or "-"
            phase1_summary = _md_cell(t.get("phase1_reason_summary") or t.get("reason_summary") or "-")
            rrf_cell = _md_cell(f"**{rrf:.4f}**")
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
            entry_signal = _resolve_entry_signal(t)
            if entry_signal["has_signal"]:
                family_cn = entry_signal["family_cn"]
                signal_label = entry_signal["display_label"]
                next_exp = f"已有{family_cn}{signal_label}，待入场条件改善"
            else:
                next_exp = rev.get("next_expected", _default_waiting_hint(t))
            dir_icon = "🟢 做多" if t["direction"] == "long" else "🔴 做空"
            rrf = t.get("rrf_score", 0)
            r1 = t.get("rank_p1", "-")
            r2 = t.get("rank_p2", "-")
            ss = t.get("signal_strength", 0)
            epr = _md_cell(t.get("entry_pool_reason") or "-")
            phase1_labels = "/".join(t.get("phase1_labels") or t.get("labels") or []) or "-"
            phase1_summary = _md_cell(t.get("phase1_reason_summary") or t.get("reason_summary") or "-")
            rrf_cell = _md_cell(f"**{rrf:.4f}**")
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
    lines.append("Phase 1 的三个分数均使用 `0-100` 区间；分值越大代表机会强度或关注优先级越高。")
    lines.append("")
    lines.append(
        "Phase 1 负责发现机会并输出关注优先级，不再直接给做多/做空建议。"
        "候选池由关注门槛控制，门槛沿用 config 分层配置，并映射为各品种的关注分入池线。"
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
        "机会标签描述的是基本面机会类型，不直接等同于最终交易方向。"
    )
    lines.append("")
    lines.append("**两阶段评分**：Phase 1 负责发现机会，Phase 2 负责解析方向与时机，两者通过 RRF 排名融合。")
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

        # --- 入场信号状态 ---
        rev = t.get("reversal_status", {})
        entry_signal = _resolve_entry_signal(t)
        if entry_signal["has_signal"]:
            signal_type = str(entry_signal["signal_type"])
            sig_cn = {
                "Spring": "弹簧(Spring)",
                "SOS": "强势突破(SOS)",
                "SC": "卖方高潮(SC)",
                "UT": "上冲回落(UT)",
                "SOW": "弱势跌破(SOW)",
                "BC": "买方高潮(BC)",
                "StopVol_Bull": "停止量(多方力竭)",
                "StopVol_Bear": "停止量(空方力竭)",
                "Pullback": "顺势回撤(Pullback)",
                "TrendBreak": "顺势突破(TrendBreak)",
            }.get(signal_type, signal_type)
            family_cn = entry_signal["family_cn"]
            family_prefix = f"[{family_cn}] " if family_cn else ""
            signal_date = str(entry_signal["signal_date"])
            signal_suffix = f" ({signal_date})" if signal_date else ""
            lines.append(f"**入场信号**: {family_prefix}{sig_cn}{signal_suffix}")
            lines.append(f"- 信号详情: {_md_cell(str(entry_signal.get('signal_detail', '') or ''))}")
            lines.append(f"- 阶段判断: {_md_cell(rev.get('current_stage', '未知'))}")
            lines.append(f"- 置信度: {float(rev.get('confidence', 0.0)):.0%}")
            lines.append("")

            lines.append("**交易参数**")
            lines.append("")
            lines.append(f"| 当前价 | 入场价 | 止损价 | 止盈1 | 止盈2 | 盈亏比 |")
            lines.append(f"| ---: | ---: | ---: | ---: | ---: | ---: |")
            lines.append(f"| {t['price']:.0f} | {t['entry']:.0f} | {t['stop']:.0f} "
                         f"| {t['tp1']:.0f} | {t['tp2']:.0f} | {t['rr']:.2f} |")
            if entry_signal["entry_family"] == "trend":
                lines.append(f"- 入场依据: 当前价（{sig_cn}条件满足，按顺势计划执行）")
                if t["direction"] == "long":
                    lines.append(f"- 止损依据: 趋势回踩失效位 - 0.5×ATR")
                else:
                    lines.append(f"- 止损依据: 趋势反弹失效位 + 0.5×ATR")
            else:
                signal_date = rev.get("signal_date", "")
                if signal_date:
                    lines.append(f"- 入场依据: 当前价（信号{signal_date}触发，新鲜可入）")
                else:
                    lines.append(f"- 入场依据: 当前价（反转信号触发，新鲜可入）")
                if t["direction"] == "long":
                    lines.append(f"- 止损依据: 信号K线最低点 - 0.5×ATR")
                else:
                    lines.append(f"- 止损依据: 信号K线最高点 + 0.5×ATR")
        else:
            lines.append(f"**入场信号**: {_format_entry_signal_for_detail(t)}")
            lines.append(f"- 当前阶段: {_md_cell(rev.get('current_stage', '未知'))}")
            lines.append(f"- 等待条件: {_md_cell(rev.get('next_expected', _default_waiting_hint(t)))}")
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
        lines.append(f"- **RRF 综合: {rrf:.4f}** (P1排名#{r1} + P2排名#{r2})")
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
            entry_signal_now = _resolve_entry_signal(t)
            reasons = []
            if not entry_signal_now["has_signal"]:
                reasons.append("无有效入场信号")
            else:
                rr = t.get("rr", 0)
                admission_rr = t.get("admission_rr", rr)
                score_val = t.get("score", 0)
                ddir = t.get("direction")
                if not t.get("phase2_rr_gate_passed", rr >= 1.0):
                    reasons.append(f"准入盈亏比{admission_rr:.2f}<1.0")
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
            targets = _normalize_phase2_plan_rows(payload.get("targets") or [])
            watchlist = _normalize_phase2_plan_rows(payload.get("watchlist") or [])
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
        reversal_candidates = []
        trend_candidates = candidates
    else:
        reversal_candidates = phase_1_screen(all_data, threshold=args.threshold)
        trend_candidates = build_trend_universe(all_data, config)

    if not reversal_candidates and not trend_candidates:
        log.info("\n  😴 今日无可跟踪品种")
        return

    # --- Phase 2: 深度分析 ---
    actionable, watchlist_result, grouped_results = run_dual_strategy_phase2(
        reversal_candidates, trend_candidates, config, max_picks=args.max_picks
    )

    top_watch = watchlist_result[:max(3, args.max_picks - len(actionable))]
    phase1_summary = build_phase1_summary(
        top_n=config.get("phase1", {}).get("top_n", args.max_picks),
        sort_field="attention_score",
        label_field="labels",
    )
    save_targets(
        actionable,
        top_watch,
        phase1_summary=phase1_summary,
        grouped_results=grouped_results,
    )

    # --- Phase 3: 盘中监控 ---
    if args.no_monitor:
        log.info("\n  ⏩ 跳过盘中监控 (--no-monitor)")
        api_total = get_request_count()
        log.info("  📊 本次运行累计AkShare请求: %s 次", api_total)
        return

    phase_3_intraday(actionable, config, args.period, args.interval, watchlist=top_watch)


if __name__ == "__main__":
    main()
