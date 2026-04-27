#!/usr/bin/env python3
"""
每日自动化交易工作流
========================================

完整流水线:
  Phase 0 — 数据预加载 (盘前)
    一次性拉取所有品种日线，缓存到本地 parquet
    当日首次运行会远端抓取日线，后续优先复用当日缓存

  Phase 1 — 机会发现 (盘前)
    负责缩小观察范围，找出值得继续跟踪的品种

  Phase 2 — 交易计划 (盘前)
    负责确定交易方向、策略类型和原始交易计划

  Phase 3 — 盘中计划更新 (09:00-15:00 / 21:00-23:30)
    负责在盘中更新第二阶段已经确定的计划，不改换策略

  Phase 4 — 持仓管理
    负责基于盘中快照和持仓成本给出持仓建议

使用方法:
    python scripts/daily_workflow.py                  # 完整流程
    python scripts/daily_workflow.py --no-monitor     # 只筛选+分析
    python scripts/daily_workflow.py --skip-screen    # 跳过筛选，用config品种
    python scripts/daily_workflow.py --resume         # 恢复今日分析，直接监控
"""

import time
import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from shared.ctp_log import get_log_path, get_logger
from shared.config_loader import load_yaml_config
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND
from shared.workflow_contract import build_workflow_contract_lines, build_workflow_contract_summary
from shared.workflow_wording import (
    daily_entry_plan_label,
    phase3_terminal_scope,
    watchlist_update_header,
)

from data_cache import (
    get_all_symbols, get_completed_daily as get_daily, get_minute,
    prefetch_all_with_stats, get_request_count,
    format_prefetch_summary,
    get_daily_with_live_bar,
)
from phase2.pre_market import (
    TREND_DIRECTION_SCORE_KEYS,
    analyze_one,
    assess_active_reversal_hold_from_daily_df,
    assess_active_trend_hold_from_daily_df,
    build_trade_plan_from_daily_df,
    reversal_direction_score_summary,
    resolve_phase2_direction,
    score_signals,
    trend_direction_score_summary,
)
from phase3.intraday import print_dashboard
from wyckoff import assess_reversal_status
from phase3.live import try_create_phase3_monitor
from phase1.pipeline import build_phase1_candidates, is_phase1_candidate_eligible, select_top_candidates
from market.contract_specs import builtin_contract_spec
from strategy_reversal.screen import select_reversal_candidates
from strategy_trend.screen import build_trend_universe as build_trend_universe_candidates, select_trend_candidates
from strategy_reversal import pre_market as reversal_pre_market
from strategy_trend import pre_market as trend_pre_market
from holdings_advice import (
    analyze_holding_record,
    build_holdings_summary_line,
    default_holdings_root,
    find_daily_holdings_workbook,
    format_holding_log_block,
    format_holding_terminal_alert,
    format_plan_line,
    holding_alert_state_key,
    load_holding_contexts,
    should_emit_terminal_alert,
)

log = get_logger("workflow")


def load_config() -> dict:
    return load_yaml_config(__file__)


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

    data, stats = prefetch_all_with_stats(symbols, completed_only=True)
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


def _phase1_direction_cn(value: object) -> str:
    direction = str(value or "").strip().lower()
    if direction == "long":
        return "做多"
    if direction == "short":
        return "做空"
    return "观察"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _phase1_story_details(row: dict, family: str) -> dict:
    details = row.get("phase1_score_details")
    if isinstance(details, dict):
        story = details.get(family)
        if isinstance(story, dict):
            return story
    if family == "reversal":
        return {
            "direction": row.get("reversal_direction"),
            "score": row.get("reversal_score"),
            "drivers": [row.get("reason_summary") or row.get("fund_details") or "暂无拆解"],
        }
    return {
        "direction": row.get("trend_direction"),
        "score": row.get("trend_score"),
        "drivers": [row.get("reason_summary") or row.get("fund_details") or "暂无拆解"],
    }


def _format_phase1_story_log(row: dict, family: str, *, prefix: str) -> str:
    story = _phase1_story_details(row, family)
    details = row.get("phase1_score_details") if isinstance(row.get("phase1_score_details"), dict) else {}
    coverage = details.get("coverage") if isinstance(details, dict) else {}
    if not isinstance(coverage, dict):
        coverage = {}

    label = "反转故事线" if family == "reversal" else "趋势故事线"
    up_label = "低位反转" if family == "reversal" else "上行趋势"
    down_label = "高位反转" if family == "reversal" else "下行趋势"
    direction = _phase1_direction_cn(story.get("direction"))
    score = _as_float(story.get("score"), _as_float(row.get(f"{family}_score")))
    up_score = story.get("up_score")
    down_score = story.get("down_score")

    pieces = [f"{prefix} {label}: {direction} {score:.0f}分"]
    if up_score is not None and down_score is not None:
        pieces.append(
            f"内部={up_label}{_as_float(up_score):.0f}/{down_label}{_as_float(down_score):.0f}"
        )
    if coverage.get("data_coverage") is not None:
        pieces.append(f"覆盖{_as_float(coverage.get('data_coverage')) * 100:.0f}%")
    if coverage.get("shrink") is not None:
        pieces.append(f"折减{_as_float(coverage.get('shrink')):.2f}")

    raw_drivers = story.get("drivers") or []
    if isinstance(raw_drivers, str):
        drivers = [raw_drivers]
    else:
        drivers = [str(item) for item in raw_drivers if str(item or "").strip()]
    driver_text = "；".join(drivers[:6]) if drivers else "暂无拆解"
    pieces.append(f"指标: {driver_text}")
    return "  ".join(pieces)


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
        if c.get("entry_threshold") is not None:
            line += f"  门槛={_as_float(c.get('entry_threshold')):.0f}"
        fd = c.get("fund_details", "")
        if fd:
            line += f"  [{fd}]"
        log.info(line)
        log.info(_format_phase1_story_log(c, "reversal", prefix="      ├─"))
        log.info(_format_phase1_story_log(c, "trend", prefix="      └─"))

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


def _phase1_trend_labeled_candidates(candidates: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in candidates:
        labels = set(row.get("labels") or [])
        if "趋势候选" in labels or "双标签候选" in labels:
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("trend_score") or 0.0), reverse=True)
    return rows


def _phase2_trend_component_text(scores: dict) -> str:
    components = sorted(
        (
            (str(key), _as_float(scores.get(key)))
            for key in TREND_DIRECTION_SCORE_KEYS
            if abs(_as_float(scores.get(key))) >= 3.0
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return " / ".join(f"{key}{value:+.0f}" for key, value in components[:4]) or "趋势指标无明显贡献"


def _log_trend_bridge_diagnostics(
    *,
    phase1_candidates: list[dict],
    trend_candidates: list[dict],
    all_data: dict[str, pd.DataFrame],
    config: dict,
) -> None:
    phase1_trend_rows = _phase1_trend_labeled_candidates(phase1_candidates)
    if not phase1_trend_rows:
        return

    trend_symbols = {str(row.get("symbol") or "") for row in trend_candidates}
    pre_cfg = config.get("pre_market") or {}
    min_history = max(int(pre_cfg.get("min_history_bars", 60)), 1)
    min_trend_score = 60.0

    log.info(
        "  🔎 趋势线衔接: Phase 1趋势标签%s 个, 盘前趋势技术池%s 个（趋势技术入池线%.0f分）",
        len(phase1_trend_rows),
        len(trend_candidates),
        min_trend_score,
    )

    for row in phase1_trend_rows[:10]:
        symbol = str(row.get("symbol") or "")
        name = str(row.get("name") or symbol)
        p1_trend = _as_float(row.get("trend_score"))
        frame = all_data.get(symbol)

        if frame is None or getattr(frame, "empty", False):
            log.info(
                "     %s (%s)  Phase1趋势%.0f -> 盘前趋势无数据  未入池: 日线缺失",
                name,
                symbol,
                p1_trend,
            )
            continue
        if len(frame) < min_history:
            log.info(
                "     %s (%s)  Phase1趋势%.0f -> 盘前趋势无数据  未入池: 历史长度%s < %s",
                name,
                symbol,
                p1_trend,
                len(frame),
                min_history,
            )
            continue

        scores = score_signals(frame.reset_index(drop=True), "long", pre_cfg)
        summary = trend_direction_score_summary(
            scores,
            delta=float(pre_cfg.get("direction_delta", 12.0)),
        )
        p2_trend = _as_float(summary.get("score"))
        long_score = _as_float(summary.get("long_score"))
        short_score = _as_float(summary.get("short_score"))
        direction = str(summary.get("direction") or "")

        if symbol in trend_symbols:
            status = "已进入趋势线"
        elif p2_trend < min_trend_score:
            status = f"未入池: 盘前趋势分{p2_trend:.0f} < {min_trend_score:.0f}"
        elif direction == "watch":
            status = "未入池: 多空分差不足，方向只够观察"
        else:
            status = "未入池: 未被趋势技术池返回"

        log.info(
            "     %s (%s)  Phase1趋势%.0f -> 盘前趋势%.0f（%s；多%.0f/空%.0f）  %s  指标: %s",
            name,
            symbol,
            p1_trend,
            p2_trend,
            _phase1_direction_cn(direction),
            long_score,
            short_score,
            status,
            _phase2_trend_component_text(scores),
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


def _normalize_plan_direction(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"long", "多", "做多"}:
        return "long"
    if text in {"short", "空", "做空"}:
        return "short"
    return ""


def _direction_cn(direction: str) -> str:
    return "做多" if direction == "long" else "做空"


def _append_downgrade_reason(existing: str, reason: str) -> str:
    if not reason:
        return existing
    if not existing:
        return reason
    return f"{existing}；{reason}"


def _fundamental_conflict_reason(
    *,
    cand: dict,
    result: dict,
    strategy_family: str,
    config: dict,
) -> str:
    trade_direction = _normalize_plan_direction(result.get("direction"))
    if trade_direction not in {"long", "short"}:
        return ""

    if strategy_family == STRATEGY_REVERSAL:
        phase1_direction = _normalize_plan_direction(cand.get("reversal_direction"))
    else:
        phase1_direction = _normalize_plan_direction(cand.get("trend_direction"))

    reasons: list[str] = []
    if phase1_direction in {"long", "short"} and phase1_direction != trade_direction:
        reasons.append(
            f"基本面方向{_direction_cn(phase1_direction)}与交易方向{_direction_cn(trade_direction)}冲突"
        )

    symbol = str(cand.get("symbol") or result.get("symbol") or "")
    configured_direction = ""
    position_cfg = (config.get("positions") or {}).get(symbol)
    if isinstance(position_cfg, dict):
        configured_direction = _normalize_plan_direction(position_cfg.get("direction"))
    if configured_direction in {"long", "short"} and configured_direction != trade_direction:
        reasons.append(
            f"配置方向{_direction_cn(configured_direction)}与交易方向{_direction_cn(trade_direction)}冲突"
        )

    return "；".join(reasons)


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


EVIDENCE_DOMAIN_CN = {
    "technical_trend": "技术趋势",
    "positioning_oi": "持仓资金",
    "inventory_supply": "库存供需",
    "warehouse_receipt": "仓单压力",
    "seasonality_profit": "季节利润",
}

FUNDAMENTAL_DOMAIN_CN = {
    "inventory": "库存",
    "warehouse_receipt": "仓单",
    "spot_basis": "现货/基差",
    "margin_cost": "成本/利润",
    "supply": "供给",
    "demand": "需求",
    "seasonality": "季节性",
}

MARKET_STAGE_CN = {
    "clear_trend": "明确趋势",
    "fundamental_turn": "基本面拐点",
    "technical_rebound_only": "技术反弹",
    "conflict": "方向冲突",
    "observation": "观察",
}

CONFLUENCE_QUALITY_CN = {
    "conflict": "方向冲突",
    "single_domain": "单域支撑",
    "supported": "多域支撑",
    "overlapping": "同类重叠",
    "independent": "独立共振",
}


def _normalize_evidence_domains(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []

    domains: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        domains.append(name)
    return domains


def _candidate_evidence_domains(*, cand: dict, strategy_family: str, direction: str) -> list[str]:
    domains = _normalize_evidence_domains(
        cand.get("trend_evidence_domains" if strategy_family == STRATEGY_TREND else "reversal_evidence_domains")
    )
    if domains:
        return domains

    oi_vs_price = str(cand.get("oi_vs_price") or "")
    seasonal_signal = _float_or_none(cand.get("seasonal_signal"))
    hog_profit = _float_or_none(cand.get("hog_profit"))
    inv_change = _float_or_none(cand.get("inv_change_4wk"))
    inv_percentile = _float_or_none(cand.get("inv_percentile"))
    receipt_change = _float_or_none(cand.get("receipt_change"))
    fallback: list[str] = []

    if strategy_family == STRATEGY_TREND:
        fallback.append("technical_trend")
        if (direction == "long" and oi_vs_price == "增仓上涨") or (direction == "short" and oi_vs_price == "增仓下跌"):
            fallback.append("positioning_oi")
        if direction == "long" and (
            (inv_change is not None and inv_change < 0) or (inv_percentile is not None and inv_percentile <= 30)
        ):
            fallback.append("inventory_supply")
        if direction == "short" and (
            (inv_change is not None and inv_change > 0) or (inv_percentile is not None and inv_percentile >= 70)
        ):
            fallback.append("inventory_supply")
        if (direction == "long" and receipt_change is not None and receipt_change < 0) or (
            direction == "short" and receipt_change is not None and receipt_change > 0
        ):
            fallback.append("warehouse_receipt")
        if (direction == "long" and seasonal_signal is not None and seasonal_signal > 0) or (
            direction == "short" and seasonal_signal is not None and seasonal_signal < 0
        ):
            fallback.append("seasonality_profit")
        return _normalize_evidence_domains(fallback)

    if (direction == "long" and receipt_change is not None and receipt_change < 0) or (
        direction == "short" and receipt_change is not None and receipt_change > 0
    ):
        fallback.append("warehouse_receipt")
    if (direction == "long" and inv_change is not None and inv_change <= 0) or (
        direction == "short" and inv_change is not None and inv_change >= 0
    ) or (
        direction == "long" and inv_percentile is not None and inv_percentile >= 80
    ) or (
        direction == "short" and inv_percentile is not None and inv_percentile <= 20
    ):
        fallback.append("inventory_supply")
    if (direction == "long" and oi_vs_price == "减仓下跌") or (direction == "short" and oi_vs_price == "减仓上涨"):
        fallback.append("positioning_oi")
    if hog_profit is not None:
        fallback.append("seasonality_profit")
    return _normalize_evidence_domains(fallback)


def _candidate_strategy_direction(cand: dict, *, strategy_family: str) -> str:
    if strategy_family == STRATEGY_REVERSAL:
        return _normalize_plan_direction(cand.get("reversal_direction"))
    if strategy_family == STRATEGY_TREND:
        return _normalize_plan_direction(cand.get("trend_direction"))
    return ""


def _candidate_primary_strategy_family(cand: dict) -> str:
    family = str(cand.get("strategy_family") or "")
    if family in {STRATEGY_REVERSAL, STRATEGY_TREND}:
        return family
    labels = set(cand.get("labels") or [])
    if "趋势候选" in labels and "反转候选" not in labels and "双标签候选" not in labels:
        return STRATEGY_TREND
    reversal_score = float(cand.get("reversal_score") or 0.0)
    trend_score = float(cand.get("trend_score") or 0.0)
    if trend_score > reversal_score:
        return STRATEGY_TREND
    return STRATEGY_REVERSAL


def _pre_market_cfg_for_symbol(config: dict, symbol: str) -> dict:
    pre_cfg = dict(config.get("pre_market") or {})
    position = (config.get("positions") or {}).get(symbol)
    if not isinstance(position, dict):
        return pre_cfg

    contract_values = {
        key: position[key]
        for key in ("multiplier", "margin_rate", "pricetick", "commission")
        if position.get(key) is not None
    }
    if not contract_values:
        return pre_cfg

    raw_specs = pre_cfg.get("contract_specs") or {}
    specs = {str(key): dict(value) for key, value in raw_specs.items() if isinstance(value, dict)}
    symbol_spec = dict(specs.get(symbol) or {})
    for key, value in contract_values.items():
        symbol_spec.setdefault(key, value)
    specs[symbol] = symbol_spec
    pre_cfg["contract_specs"] = specs
    return pre_cfg


def _contract_multiplier_for_symbol(pre_cfg: dict, symbol: str) -> float:
    spec = builtin_contract_spec(symbol)
    raw_specs = pre_cfg.get("contract_specs") or {}
    configured = raw_specs.get(symbol) if isinstance(raw_specs, dict) else None
    if isinstance(configured, dict):
        spec.update(configured)
    return _as_float(spec.get("multiplier") or pre_cfg.get("contract_multiplier"), 0.0)


def _strategy_direction_from_scores(
    *,
    scores: dict,
    strategy_family: str,
    pre_cfg: dict,
) -> tuple[str, str, dict[str, float | str]]:
    delta = float(pre_cfg.get("direction_delta", 12.0))
    if strategy_family == STRATEGY_TREND:
        summary = trend_direction_score_summary(scores, delta=delta)
    else:
        summary = reversal_direction_score_summary(scores, delta=delta)
    long_score = float(summary.get("long_score") or 0.0)
    short_score = float(summary.get("short_score") or 0.0)
    resolved_direction = resolve_phase2_direction(
        long_score=long_score,
        short_score=short_score,
        delta=delta,
    )
    analysis_direction = resolved_direction
    if analysis_direction == "watch":
        analysis_direction = "long" if long_score >= short_score else "short"
    return resolved_direction, analysis_direction, summary


def _opposite_direction(direction: str) -> str:
    if direction == "long":
        return "short"
    if direction == "short":
        return "long"
    return ""


def _resolve_strategy_phase2_direction(
    *,
    cand: dict,
    scores: dict,
    strategy_family: str,
    pre_cfg: dict,
) -> tuple[str, str, dict[str, float | str]]:
    score_direction, score_analysis_direction, summary = _strategy_direction_from_scores(
        scores=scores,
        strategy_family=strategy_family,
        pre_cfg=pre_cfg,
    )
    candidate_direction = _candidate_strategy_direction(cand, strategy_family=strategy_family)
    if candidate_direction in {"long", "short"}:
        summary["candidate_direction"] = candidate_direction
        summary["score_direction"] = score_direction
        if score_direction in {"long", "short"} and score_direction == _opposite_direction(candidate_direction):
            summary["candidate_direction_conflict"] = True
            summary["direction"] = "watch"
            return "watch", candidate_direction, summary
        return score_direction, candidate_direction, summary
    summary["score_direction"] = score_direction
    return score_direction, score_analysis_direction, summary


def _phase2_direction_gate_reason(direction_summary: dict[str, float | str]) -> str:
    if not bool(direction_summary.get("candidate_direction_conflict")):
        return ""
    candidate_direction = str(direction_summary.get("candidate_direction") or "")
    score_direction = str(direction_summary.get("score_direction") or direction_summary.get("direction") or "")
    if candidate_direction not in {"long", "short"} or score_direction not in {"long", "short"}:
        return ""
    return (
        f"日线深度方向分数已明显转为{_direction_cn(score_direction)}，"
        f"上游候选{_direction_cn(candidate_direction)}仅保留观察"
    )


def _apply_phase2_direction_gate(row: dict, direction_summary: dict[str, float | str]) -> dict:
    row["phase2_direction_long_score"] = float(direction_summary.get("long_score") or 0.0)
    row["phase2_direction_short_score"] = float(direction_summary.get("short_score") or 0.0)
    row["phase2_score_direction"] = str(direction_summary.get("score_direction") or direction_summary.get("direction") or "")
    candidate_direction = str(direction_summary.get("candidate_direction") or "")
    if candidate_direction:
        row["phase2_candidate_direction"] = candidate_direction
    reason = _phase2_direction_gate_reason(direction_summary)
    if reason:
        row["actionable"] = False
        row["phase2_direction_conflict"] = True
        row["downgrade_reason"] = _append_downgrade_reason(
            str(row.get("downgrade_reason") or ""),
            reason,
        )
    return row


def _fundamental_extreme_state(*, cand: dict, direction: str) -> tuple[bool, list[str]]:
    inv_change = _float_or_none(cand.get("inv_change_4wk"))
    inv_percentile = _float_or_none(cand.get("inv_percentile"))
    receipt_change = _float_or_none(cand.get("receipt_change"))
    hog_profit = _float_or_none(cand.get("hog_profit"))
    reasons: list[str] = []

    if direction == "long":
        if inv_percentile is not None and inv_percentile >= 80:
            reasons.append("高库存")
        if inv_change is not None and inv_change > 0:
            reasons.append("累库")
        if receipt_change is not None and receipt_change > 0:
            reasons.append("仓单增加")
        if hog_profit is not None and hog_profit <= -30:
            reasons.append("利润深亏")
    else:
        if inv_percentile is not None and inv_percentile <= 20:
            reasons.append("低库存")
        if inv_change is not None and inv_change < 0:
            reasons.append("去库")
        if receipt_change is not None and receipt_change < 0:
            reasons.append("仓单减少")
        if hog_profit is not None and hog_profit >= 30:
            reasons.append("利润高位")

    return bool(reasons), reasons


def _fundamental_marginal_turn(*, cand: dict, direction: str) -> tuple[bool, list[str]]:
    inv_change = _float_or_none(cand.get("inv_change_4wk"))
    receipt_change = _float_or_none(cand.get("receipt_change"))
    hog_profit = _float_or_none(cand.get("hog_profit"))
    hog_profit_change = _float_or_none(cand.get("hog_profit_change"))
    hog_price_trend = _float_or_none(cand.get("hog_price_trend"))
    reasons: list[str] = []

    if direction == "long":
        if inv_change is not None and inv_change <= 0:
            reasons.append("去库启动")
        if receipt_change is not None and receipt_change < 0:
            reasons.append("仓单回落")
        if hog_profit is not None and hog_profit <= -30 and (
            (hog_profit_change is not None and hog_profit_change > 0)
            or (hog_price_trend is not None and hog_price_trend > 0)
        ):
            reasons.append("利润压力缓和")
    else:
        if inv_change is not None and inv_change >= 0:
            reasons.append("库存回升")
        if receipt_change is not None and receipt_change > 0:
            reasons.append("仓单回升")
        if hog_profit is not None and hog_profit >= 30 and (
            (hog_profit_change is not None and hog_profit_change < 0)
            or (hog_price_trend is not None and hog_price_trend < 0)
        ):
            reasons.append("利润高位回落")

    return bool(reasons), reasons


def _candidate_has_snapshot_fundamental_gate(cand: dict) -> bool:
    return any(
        key in cand
        for key in (
            "fundamental_reversal_confirmed",
            "fundamental_extreme_state_confirmed",
            "fundamental_marginal_turn_confirmed",
            "fundamental_coverage_score",
            "fundamental_domains_missing",
        )
    )


def _fundamental_reversal_gate(*, cand: dict, result: dict) -> dict[str, object]:
    direction = _normalize_plan_direction(result.get("direction"))
    if direction not in {"long", "short"}:
        return {
            "extreme_state_confirmed": False,
            "marginal_turn_confirmed": False,
            "reversal_confirmed": False,
            "downgrade_reason": "方向未明确，暂不判断基本面拐点",
        }

    coverage_score = cand.get("fundamental_coverage_score")
    coverage_status = cand.get("fundamental_coverage_status")
    domains_present = list(cand.get("fundamental_domains_present") or [])
    domains_missing = list(cand.get("fundamental_domains_missing") or [])
    missing_reasons = list(cand.get("fundamental_missing_domain_reasons") or [])

    if _candidate_has_snapshot_fundamental_gate(cand):
        extreme_confirmed = bool(cand.get("fundamental_extreme_state_confirmed"))
        extreme_reasons = list(cand.get("fundamental_extreme_state_reasons") or [])
        turn_confirmed = bool(cand.get("fundamental_marginal_turn_confirmed"))
        turn_reasons = list(cand.get("fundamental_marginal_turn_reasons") or [])
        reversal_confirmed = bool(cand.get("fundamental_reversal_confirmed"))
    else:
        extreme_confirmed, extreme_reasons = _fundamental_extreme_state(cand=cand, direction=direction)
        turn_confirmed, turn_reasons = _fundamental_marginal_turn(cand=cand, direction=direction)
        reversal_confirmed = extreme_confirmed and turn_confirmed

    if reversal_confirmed:
        reason = ""
    elif missing_reasons:
        reason = (
            "基本面数据不足："
            f"{'/'.join(missing_reasons)}，仅保留观察"
        )
    elif extreme_confirmed and not turn_confirmed:
        reason = (
            "基本面极端仍在但边际未转向："
            f"{'/'.join(extreme_reasons) or '结构压力未缓解'}，归为逆基本面反弹观察"
        )
    elif turn_confirmed and not extreme_confirmed:
        reason = (
            "基本面尚未进入极端失衡："
            f"{'/'.join(turn_reasons) or '仅有边际改善'}，暂不作为均值回归剧本"
        )
    else:
        reason = "未见明确基本面拐点：当前更像技术反弹，暂归为观察"

    return {
        "extreme_state_confirmed": extreme_confirmed,
        "extreme_state_reasons": extreme_reasons,
        "marginal_turn_confirmed": turn_confirmed,
        "marginal_turn_reasons": turn_reasons,
        "reversal_confirmed": reversal_confirmed,
        "coverage_score": coverage_score,
        "coverage_status": coverage_status,
        "domains_present": domains_present,
        "domains_missing": domains_missing,
        "missing_domain_reasons": missing_reasons,
        "downgrade_reason": reason,
    }


def _merge_strategy_result(
    *,
    cand: dict,
    result: dict,
    resolved_direction: str,
    strategy_family: str,
    config: dict,
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
    merged["fund_hog_price_trend"] = cand.get("hog_price_trend")
    merged["fund_hog_profit_change"] = cand.get("hog_profit_change")
    merged["oi_vs_price"] = cand.get("oi_vs_price")
    merged["oi_20d_change"] = cand.get("oi_20d_change")
    merged["oi_percentile"] = cand.get("oi_percentile")
    merged["fund_screen_score"] = cand.get("strategy_score", cand.get("attention_score", cand.get("score", 0)))
    merged["fund_details"] = cand.get("fund_details", "")
    merged["signal_strength"] = merged.get("reversal_status", {}).get("signal_strength", 0.0)
    merged["phase2_decision"] = resolved_direction
    merged["entry_pool_reason"] = cand.get("entry_pool_reason", "")
    merged["strategy_role"] = "strategy_pool_member"
    merged["evidence_domains"] = _candidate_evidence_domains(
        cand=cand,
        strategy_family=strategy_family,
        direction=_normalize_plan_direction(merged.get("direction")),
    )
    conflict_reason = _fundamental_conflict_reason(
        cand=cand,
        result=merged,
        strategy_family=strategy_family,
        config=config,
    )
    merged["fundamental_conflict"] = bool(conflict_reason)
    if conflict_reason:
        merged["actionable"] = False
        merged["downgrade_reason"] = _append_downgrade_reason(
            str(merged.get("downgrade_reason") or ""),
            conflict_reason,
        )
    if strategy_family == STRATEGY_REVERSAL:
        gate = _fundamental_reversal_gate(cand=cand, result=merged)
        merged["fundamental_extreme_state_confirmed"] = bool(gate["extreme_state_confirmed"])
        merged["fundamental_extreme_state_reasons"] = list(gate.get("extreme_state_reasons") or [])
        merged["fundamental_marginal_turn_confirmed"] = bool(gate["marginal_turn_confirmed"])
        merged["fundamental_marginal_turn_reasons"] = list(gate.get("marginal_turn_reasons") or [])
        merged["fundamental_reversal_confirmed"] = bool(gate["reversal_confirmed"])
        if gate.get("coverage_score") is not None:
            merged["fundamental_coverage_score"] = float(gate.get("coverage_score") or 0.0)
        if gate.get("coverage_status") is not None:
            merged["fundamental_coverage_status"] = str(gate.get("coverage_status") or "")
        merged["fundamental_domains_present"] = list(gate.get("domains_present") or cand.get("fundamental_domains_present") or [])
        merged["fundamental_domains_missing"] = list(gate.get("domains_missing") or cand.get("fundamental_domains_missing") or [])
        merged["fundamental_missing_domain_reasons"] = list(
            gate.get("missing_domain_reasons") or cand.get("fundamental_missing_domain_reasons") or []
        )
        reversal_reason = str(gate.get("downgrade_reason") or "")
        if not merged["fundamental_reversal_confirmed"] and reversal_reason:
            merged["actionable"] = False
            merged["strategy_role"] = "counter_fundamental_rebound_watch"
            merged["downgrade_reason"] = _append_downgrade_reason(
                str(merged.get("downgrade_reason") or ""),
                reversal_reason,
            )
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


def _positive_int_floor(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        return 0
    return max(int(math.floor(value)), 0)


def _phase2_position_weight(row: dict) -> float:
    weight = abs(_as_float(row.get("score"), 0.0))
    return weight if weight > 0 else 1.0


def _apply_portfolio_position_sizing(rows: list[dict], config: dict) -> None:
    if not rows:
        log.info("  💰 资金管理: 无可执行品种，跳过仓位分配")
        return
    pre_cfg = config.get("pre_market") or {}
    account_equity = _as_float(pre_cfg.get("account_equity"), 0.0)
    portfolio_pct = _as_float(pre_cfg.get("portfolio_max_margin_pct") or pre_cfg.get("max_margin_pct"), 0.0)
    if account_equity <= 0 or portfolio_pct <= 0:
        return

    margin_pool = max(account_equity * portfolio_pct - _as_float(pre_cfg.get("portfolio_margin_used_cny"), 0.0), 0.0)
    common_risk_budget = _as_float(pre_cfg.get("risk_per_trade_cny"), 0.0)
    if common_risk_budget <= 0:
        common_risk_budget = account_equity * _as_float(pre_cfg.get("risk_per_trade_pct"), 0.0)
    log.info(
        "  💰 资金管理: 可执行%d个, 账户权益%.0f, 组合保证金池%.0f, 单笔止损预算%.0f",
        len(rows),
        account_equity,
        margin_pool,
        common_risk_budget,
    )
    weights = [_phase2_position_weight(row) for row in rows]
    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = float(len(rows))
        weights = [1.0 for _ in rows]

    for row, weight in zip(rows, weights, strict=False):
        margin_per_lot = _as_float(row.get("margin_per_lot"), 0.0)
        if margin_per_lot <= 0:
            entry = _as_float(row.get("entry"), 0.0)
            multiplier = _as_float(row.get("contract_multiplier"), 0.0)
            margin_rate = _as_float(row.get("margin_rate"), 0.0)
            margin_per_lot = entry * multiplier * margin_rate if entry > 0 and multiplier > 0 and margin_rate > 0 else 0.0
            row["margin_per_lot"] = float(margin_per_lot)

        score_margin_budget = margin_pool * weight / total_weight if total_weight > 0 else 0.0
        score_lots = _positive_int_floor(score_margin_budget / margin_per_lot) if margin_per_lot > 0 else 0
        portfolio_lots = _positive_int_floor(margin_pool / margin_per_lot) if margin_per_lot > 0 else 0
        risk_budget = _as_float(row.get("risk_budget"), 0.0)
        if risk_budget <= 0:
            risk_budget = _as_float(pre_cfg.get("risk_per_trade_cny"), 0.0)
        if risk_budget <= 0:
            risk_budget = account_equity * _as_float(pre_cfg.get("risk_per_trade_pct"), 0.0)
        risk_lots = int(row.get("max_lots_by_risk") or row.get("risk_lots") or 0)
        risk_per_lot = _as_float(row.get("risk_per_lot"), 0.0)
        if risk_budget > 0 and risk_lots <= 0 and risk_per_lot > 0:
            risk_lots = _positive_int_floor(risk_budget / risk_per_lot)

        caps = [score_lots, portfolio_lots]
        if risk_budget > 0:
            caps.append(risk_lots)
        suggested_lots = min(caps) if caps else 0

        row["position_weight"] = float(weight)
        row["position_weight_pct"] = float(weight / total_weight) if total_weight > 0 else 0.0
        row["portfolio_margin_budget"] = float(margin_pool)
        row["score_margin_budget"] = float(score_margin_budget)
        row["risk_budget"] = float(risk_budget)
        row["score_lots"] = int(score_lots)
        row["portfolio_lots"] = int(portfolio_lots)
        row["risk_lots"] = int(risk_lots)
        row["suggested_lots"] = int(suggested_lots)
        row["suggested_margin"] = float(suggested_lots * margin_per_lot)
        row["suggested_stop_risk"] = float(suggested_lots * risk_per_lot)
        row["phase2_position_size_passed"] = bool(suggested_lots >= 1)
        log.info(
            "     %s (%s) Phase2分%.1f 权重%.1f%% 预算%.0f 每手保证金%.0f 每手止损%.0f "
            "手数: 分数仓位%d手 / 组合上限%d手 / 止损风险%d手 -> 建议%d手",
            row.get("name") or row.get("symbol") or "",
            row.get("symbol") or "",
            _as_float(row.get("score"), 0.0),
            float(weight / total_weight * 100.0) if total_weight > 0 else 0.0,
            score_margin_budget,
            margin_per_lot,
            risk_per_lot,
            score_lots,
            portfolio_lots,
            risk_lots,
            suggested_lots,
        )
        if suggested_lots < 1:
            row["actionable"] = False
            row["downgrade_reason"] = _append_downgrade_reason(
                str(row.get("downgrade_reason") or ""),
                "账户仓位和止损风险约束下建议手数为0",
            )
            log.info(
                "     资金管理降级: %s (%s) 建议手数0，移入观察；原因=账户仓位和止损风险约束下建议手数为0",
                row.get("name") or row.get("symbol") or "",
                row.get("symbol") or "",
            )


def _run_strategy_phase2(
    candidates: list[dict],
    config: dict,
    *,
    max_picks: int,
    strategy_family: str,
    sort_field: str,
    plan_builder,
) -> tuple[list[dict], list[dict]]:
    actionable: list[dict] = []
    watchlist: list[dict] = []

    for cand in candidates:
        pre_cfg = _pre_market_cfg_for_symbol(config, str(cand["symbol"]))
        phase2_df = get_daily(cand["symbol"])
        if phase2_df is None or getattr(phase2_df, "empty", False):
            continue

        phase2_scores = score_signals(phase2_df, "long", pre_cfg)
        resolved_direction, analysis_direction, direction_summary = _resolve_strategy_phase2_direction(
            cand=cand,
            scores=phase2_scores,
            strategy_family=strategy_family,
            pre_cfg=pre_cfg,
        )

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
            config=config,
        )
        merged_result = _apply_phase2_direction_gate(merged_result, direction_summary)
        if merged_result["actionable"]:
            actionable.append(merged_result)
        else:
            watchlist.append(merged_result)

    all_results = actionable + watchlist
    _apply_strategy_rrf(all_results, sort_field=sort_field)
    actionable.sort(key=lambda item: item.get("rrf_score", 0), reverse=True)
    watchlist.sort(key=lambda item: item.get("rrf_score", 0), reverse=True)
    return actionable[:max_picks], watchlist


def _is_actionable(row: dict) -> bool:
    return bool(row.get("actionable", False))


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


def _strategy_hit(row: dict) -> dict:
    return {
        "strategy_family": str(row.get("strategy_family") or ""),
        "direction": str(row.get("direction") or ""),
        "actionable": _is_actionable(row),
        "score": float(row.get("score") or 0.0),
        "rrf_score": float(row.get("rrf_score") or 0.0),
        "entry_family": str(row.get("entry_family") or ""),
        "entry_signal_type": str(row.get("entry_signal_type") or ""),
        "evidence_domains": _normalize_evidence_domains(row.get("evidence_domains")),
    }


def _row_priority(row: dict) -> tuple[int, float, float]:
    return (
        1 if _is_actionable(row) else 0,
        float(row.get("rrf_score") or 0.0),
        abs(float(row.get("score") or 0.0)),
    )


def _row_market_stage_priority(row: dict, market_stage: str) -> tuple[int, int, float, float]:
    family = str(row.get("strategy_family") or "")
    entry_family = str(row.get("entry_family") or "")
    family_priority = 0
    if market_stage == "clear_trend" and (family == STRATEGY_TREND or entry_family == "trend"):
        family_priority = 1
    elif market_stage in {"fundamental_turn", "technical_rebound_only"} and (
        family == STRATEGY_REVERSAL or entry_family == "reversal"
    ):
        family_priority = 1
    actionable, rrf_score, raw_score = _row_priority(row)
    return (family_priority, actionable, rrf_score, raw_score)


def _market_stage(rows: list[dict], hits: list[dict], directions: set[str]) -> str:
    if len(directions) > 1:
        return "conflict"

    trend_rows = [
        row
        for row in rows
        if str(row.get("strategy_family") or "") == STRATEGY_TREND or str(row.get("entry_family") or "") == "trend"
    ]
    if any(_is_actionable(row) for row in trend_rows):
        return "clear_trend"

    reversal_rows = [
        row
        for row in rows
        if str(row.get("strategy_family") or "") == STRATEGY_REVERSAL or str(row.get("entry_family") or "") == "reversal"
    ]
    if any(
        bool(row.get("fundamental_reversal_confirmed"))
        or ("fundamental_reversal_confirmed" not in row and _is_actionable(row))
        for row in reversal_rows
    ):
        return "fundamental_turn"
    if reversal_rows:
        return "technical_rebound_only"
    return "observation"


def _representative_row(rows: list[dict], market_stage: str) -> dict:
    if market_stage == "clear_trend":
        candidates = [
            row
            for row in rows
            if str(row.get("strategy_family") or "") == STRATEGY_TREND or str(row.get("entry_family") or "") == "trend"
        ]
        if candidates:
            return max(candidates, key=_row_priority)
    if market_stage == "fundamental_turn":
        candidates = [row for row in rows if bool(row.get("fundamental_reversal_confirmed"))]
        if candidates:
            return max(candidates, key=_row_priority)
    if market_stage == "technical_rebound_only":
        candidates = [
            row
            for row in rows
            if str(row.get("strategy_family") or "") == STRATEGY_REVERSAL or str(row.get("entry_family") or "") == "reversal"
        ]
        if candidates:
            return max(candidates, key=_row_priority)
    return max(rows, key=_row_priority)


def _selected_playbook(row: dict, market_stage: str) -> str:
    if market_stage == "clear_trend":
        signal_type = str(row.get("entry_signal_type") or "")
        if signal_type == "Pullback":
            return "trend_pullback"
        return "trend_continuation"
    if market_stage == "fundamental_turn":
        return "fundamental_mean_reversion"
    if market_stage == "technical_rebound_only":
        return "technical_rebound_watch"
    if market_stage == "conflict":
        return "conflict_watch"
    return "observation"


def _confluence_quality(*, hits: list[dict], directions: set[str], independent_evidence_count: int) -> str:
    if len(directions) > 1:
        return "conflict"
    if len(hits) <= 1:
        return "supported" if independent_evidence_count >= 2 else "single_domain"
    if independent_evidence_count <= 1:
        return "overlapping"
    return "independent"


def _independent_evidence_summary(hits: list[dict]) -> tuple[list[str], int]:
    domains: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        for domain in _normalize_evidence_domains(hit.get("evidence_domains")):
            if domain in seen:
                continue
            seen.add(domain)
            domains.append(domain)
    summary = [EVIDENCE_DOMAIN_CN.get(domain, domain) for domain in domains]
    return summary, len(domains)


def _unselected_playbook_reason(*, market_stage: str, hits: list[dict]) -> str:
    has_trend = any(hit["strategy_family"] == STRATEGY_TREND or hit["entry_family"] == "trend" for hit in hits)
    has_reversal = any(hit["strategy_family"] == STRATEGY_REVERSAL or hit["entry_family"] == "reversal" for hit in hits)
    confirmed_directions = {
        hit["direction"]
        for hit in hits
        if hit.get("actionable") and hit.get("direction") in {"long", "short"}
    }
    if len(confirmed_directions) == 1 and market_stage in {"clear_trend", "fundamental_turn"}:
        confirmed_direction = next(iter(confirmed_directions))
        has_opposite_watch = any(
            not hit.get("actionable")
            and hit.get("direction") in {"long", "short"}
            and hit.get("direction") != confirmed_direction
            for hit in hits
        )
        if has_opposite_watch:
            return "存在未确认的反向观察项，仅作为风险提示，不否决已确认交易剧本"

    if market_stage == "clear_trend" and has_reversal:
        return "存在明确趋势，基本面同向只作为共振增强，不单独切换到均值回归剧本"
    if market_stage == "fundamental_turn" and has_trend:
        return "当前无明确趋势，且基本面已出现边际转向，优先按均值回归剧本执行"
    if market_stage == "technical_rebound_only":
        return "未见基本面拐点，当前只按技术反弹观察"
    if market_stage == "conflict":
        return "趋势与基本面方向冲突，暂不选择交易剧本"
    return ""


def _market_stage_rank(value: str) -> int:
    return {
        "clear_trend": 3,
        "fundamental_turn": 2,
        "technical_rebound_only": 1,
        "observation": 0,
        "conflict": -1,
    }.get(value, 0)


def _direction_set(rows: list[dict]) -> set[str]:
    return {
        str(row.get("direction") or "")
        for row in rows
        if str(row.get("direction") or "") in {"long", "short"}
    }


def _merge_strategy_pool_rows(rows: list[dict]) -> dict:
    confirmed_rows = [row for row in rows if _is_actionable(row)]
    direction_rows = confirmed_rows if confirmed_rows else rows
    directions = _direction_set(direction_rows)
    market_stage = _market_stage(rows, [], directions)
    ordered_rows = sorted(rows, key=lambda row: _row_market_stage_priority(row, market_stage), reverse=True)
    hits = [_strategy_hit(row) for row in ordered_rows]
    confirmed_hits = [hit for hit in hits if bool(hit.get("actionable"))]
    confluence_hits = confirmed_hits if confirmed_hits else hits
    market_stage = _market_stage(rows, hits, directions)
    representative = _representative_row(rows, market_stage)
    merged = dict(representative)
    merged["strategy_pool_hits"] = hits
    merged["strategy_hit_count"] = len(hits)
    merged["confirmed_strategy_pool_hits"] = confirmed_hits
    merged["confirmed_strategy_hit_count"] = len(confirmed_hits)
    merged["market_stage"] = market_stage
    merged["independent_evidence_summary"], merged["independent_evidence_count"] = _independent_evidence_summary(
        confluence_hits
    )
    merged["confluence_quality"] = _confluence_quality(
        hits=confluence_hits,
        directions=directions,
        independent_evidence_count=int(merged["independent_evidence_count"]),
    )
    merged["unselected_playbook_reason"] = _unselected_playbook_reason(market_stage=market_stage, hits=hits)

    if len(directions) > 1:
        merged["actionable"] = False
        merged["strategy_resonance"] = "conflict"
        merged["market_stage"] = "conflict"
        merged["selected_playbook"] = "conflict_watch"
        merged["downgrade_reason"] = _append_downgrade_reason(
            str(merged.get("downgrade_reason") or ""),
            "策略池方向冲突，需等待趋势和基本面重新同向",
        )
        return merged

    resonance_hits = confirmed_hits if confirmed_hits else hits
    if len(resonance_hits) > 1:
        merged["strategy_resonance"] = "same_direction"
    else:
        merged["strategy_resonance"] = "single_strategy"
    merged["selected_playbook"] = _selected_playbook(merged, market_stage)
    if market_stage == "technical_rebound_only":
        merged["actionable"] = False
    return merged


def _merge_strategy_pool(
    *,
    trend_actionable: list[dict],
    trend_watchlist: list[dict],
    reversal_actionable: list[dict],
    reversal_watchlist: list[dict],
    max_picks: int,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in trend_actionable + trend_watchlist + reversal_actionable + reversal_watchlist:
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        grouped.setdefault(symbol, []).append(row)

    merged_rows = [_merge_strategy_pool_rows(rows) for rows in grouped.values()]
    actionable = [row for row in merged_rows if _is_actionable(row)]
    watchlist = [row for row in merged_rows if not _is_actionable(row)]

    actionable.sort(
        key=lambda row: (
            _market_stage_rank(str(row.get("market_stage") or "")),
            int(row.get("independent_evidence_count") or 0),
            1 if row.get("confluence_quality") == "independent" else 0,
            float(row.get("rrf_score") or 0.0),
        ),
        reverse=True,
    )
    watchlist.sort(
        key=lambda row: (
            1 if row.get("strategy_resonance") == "conflict" else 0,
            _market_stage_rank(str(row.get("market_stage") or "")),
            int(row.get("independent_evidence_count") or 0),
            float(row.get("rrf_score") or 0.0),
        ),
        reverse=True,
    )
    return actionable[:max_picks], watchlist


def run_dual_strategy_phase2(
    reversal_candidates: list[dict],
    trend_candidates: list[dict],
    config: dict,
    max_picks: int = 6,
) -> tuple[list[dict], list[dict], dict[str, dict[str, list[dict]]]]:
    reversal_actionable, reversal_watchlist = run_reversal_strategy_phase2(reversal_candidates, config, max_picks)
    trend_actionable, trend_watchlist = run_trend_strategy_phase2(trend_candidates, config, max_picks)

    actionable, watchlist = _merge_strategy_pool(
        trend_actionable=trend_actionable,
        trend_watchlist=trend_watchlist,
        reversal_actionable=reversal_actionable,
        reversal_watchlist=reversal_watchlist,
        max_picks=max_picks,
    )
    phase2_pre_sizing_actionable = [dict(row) for row in actionable]
    _apply_portfolio_position_sizing(actionable, config)
    sized_watchlist = [row for row in actionable if not _is_actionable(row)]
    actionable = [row for row in actionable if _is_actionable(row)]
    if sized_watchlist:
        watchlist = sized_watchlist + watchlist
    grouped = {
        STRATEGY_REVERSAL: {
            "actionable": reversal_actionable,
            "watchlist": reversal_watchlist,
        },
        STRATEGY_TREND: {
            "actionable": trend_actionable,
            "watchlist": trend_watchlist,
        },
        "phase2_pre_sizing": {
            "actionable": phase2_pre_sizing_actionable,
            "watchlist": [],
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
      actionable: 底层字段名沿用 actionable，含义是“今天已进入可执行状态”
      watchlist:  评分有倾向，但今天仍在等待确认
    """
    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 2: 盘前深度分析")
    log.info("▓" * 60)

    actionable = []
    watchlist = []

    for cand in candidates:
        try:
            pre_cfg = _pre_market_cfg_for_symbol(config, str(cand["symbol"]))
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
            strategy_family = _candidate_primary_strategy_family(cand)
            resolved_direction, analysis_direction, direction_summary = _resolve_strategy_phase2_direction(
                cand=cand,
                scores=phase2_scores,
                strategy_family=strategy_family,
                pre_cfg=pre_cfg,
            )

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
                result = _apply_phase2_direction_gate(result, direction_summary)

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
    log.info("  📋 Phase 2 汇总: %s 个可执行, %s 个等待确认", act_count, watch_count)
    log.info("%s", "=" * 60)

    if actionable:
        log.info(f"\n  {'':2s}{'品种':8s} {'代码':6s} {'方向':4s} {'RRF':>7s} {'#P1':>4s} {'#P2':>4s} {'基本面':>6s} {'技术面':>6s} {'确认':>6s} {'强度':>5s} {'执行价':>8s} {'盈亏比':>6s}")
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
            next_exp = _format_watchlist_story_status(w)
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
    盘中计划更新。

    这一步只负责更新第二阶段已经确定的交易计划，不负责改换策略类型。
    盘前被定义为顺势的品种，盘中仍按顺势框架更新；盘前被定义为反转的品种，
    盘中仍按反转框架更新。若盘前没有达到入场条件，但盘中出现了更好的入场时机，
    也只能在原有策略框架内补抓机会。

    Parameters
    ----------
    targets
        Phase 2 判定为可执行的品种列表（分钟级 K 线 + `print_dashboard`）。
    watchlist
        等待确认列表（可选）。非空时对每个品种用最新盘中日线重评交易计划，
        并与上一轮状态对比打印“确认新出现 / 持续 / 回退”等提示。
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
    - 等待确认品种每轮结束时记录本轮确认状态，供下一轮判断“新出现 / 持续 / 回退”。
    - 同一轮盘中快照会继续传给第四阶段，保证新开仓计划更新和持仓建议使用同一份输入。
    """
    prev_hold_actions: dict[str, str] = {}
    all_monitor = targets + (watchlist or [])
    if not all_monitor:
        log.info("\n  没有可执行/等待确认品种，先检查持仓建议")
        prev_hold_actions = run_phase_4_holdings(
            config,
            period=period,
            emit_terminal=True,
            previous_actions=prev_hold_actions,
            log_header=True,
        )
        if not prev_hold_actions:
            log.info("\n  没有可执行/等待确认品种，也没有持仓建议，跳过盘中监控")
            return

    intraday_cfg = config.get("intraday", {})

    _print_and_log(phase3_terminal_scope(get_log_path()))
    log.info("\n" + "▓" * 60)
    log.info("▓  Phase 3: 盘中实时监控")
    log.info("▓" * 60)
    act_names = ", ".join(t["name"] for t in targets) if targets else "无"
    watch_names = ", ".join(t["name"] for t in (watchlist or [])) if watchlist else "无"
    log.info("\n  可执行: %s", act_names)
    log.info("  确认观察: %s", watch_names)
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

    # 记录上轮每个品种的确认状态，用于检测确认出现/回退
    prev_signals: dict[str, bool] = {}
    prev_watch_rows: dict[str, dict] = {}
    cycle = 0
    ended_after_trading_hours = False
    try:
        while True:
            try:
                cycle += 1
                now = datetime.now()
                current_cycle_live_frames: dict[str, pd.DataFrame] = {}
                log.info(f"\n{'═'*60}")
                log.info(
                    "  盘中监控 #%s  %s  K线:%s分钟",
                    cycle,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    period,
                )
                log.info("%s", "═" * 60)

                # --- 可执行品种: 分钟级监控 ---
                for t in targets:
                    try:
                        if tq_mon:
                            df = tq_mon.minute_df(t["symbol"])
                        else:
                            df = get_minute(t["symbol"], period)
                        if df is None or df.empty:
                            log.info("\n  %s: 无数据", t["name"])
                            continue
                        entry_signal = _resolve_entry_signal(t)
                        if entry_signal["has_signal"]:
                            signal_date = str(entry_signal["signal_date"])
                            signal_date_part = f" {signal_date}" if signal_date else ""
                            log.info(
                                "│ 📌 %s: %s %s  入场%s  止损%s  目标%s",
                                daily_entry_plan_label(),
                                entry_signal["display_label"],
                                signal_date_part,
                                f"{t['entry']:.0f}",
                                f"{t['stop']:.0f}",
                                f"{t['tp1']:.0f}",
                            )
                            execution_line = _format_intraday_execution_line(t, df)
                            if execution_line:
                                log.info(execution_line)
                        print_dashboard(
                            t["symbol"],
                            t["name"],
                            t["direction"],
                            df,
                            intraday_cfg,
                            entry_family=str(entry_signal["entry_family"] or ""),
                        )
                    except Exception as e:
                        log.info("\n  %s: ❌ %s", t["name"], e)

                # --- 等待确认品种: 日线确认检测（TqSdk 用交易所日线序列含当日未完成K线；否则合成日线）---
                if watchlist:
                    log.info("\n  ── %s ──", watchlist_update_header())
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
                            current_cycle_live_frames[sym] = live_df

                            previous_watch = prev_watch_rows.get(sym, _normalize_phase2_plan_row(w))
                            live_plan = _build_strategy_scoped_trade_plan(
                                base_row=w,
                                symbol=sym,
                                name=w["name"],
                                direction=w["direction"],
                                df=live_df,
                                cfg=pre_cfg,
                                allow_watch_plan=True,
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
                                dir_cn = "做多" if current_watch["direction"] == "long" else "做空"
                                confirm = _intraday_watch_confirmation_text(current_watch)

                                pool_r = current_watch.get("entry_pool_reason") or ""
                                pool_line = f"       入池理由: {pool_r}" if pool_r else ""
                                _print_and_log(f"  🔔🔔 {current_watch['name']} 出现{signal_noun}")
                                _print_and_log(f"       确认: {sig_cn} | 计划方向: {dir_cn}")
                                if pool_line:
                                    _print_and_log(pool_line)
                                _print_and_log(f"       {live_info}")
                                _print_and_log(f"       {_format_intraday_trade_plan(current_watch)}")
                                if plan_changed:
                                    _print_and_log("       🔄 盘中重评后交易计划已更新")
                                if actionable_upgraded:
                                    _print_and_log(
                                        f"       ✅ 盘中重评后转为可执行：评分{float(current_watch.get('score', 0.0)):+.0f} "
                                        f"准入RR {float(current_watch.get('admission_rr', current_watch.get('rr', 0.0))):.2f}"
                                    )
                                _print_and_log(f"       确认条件: {confirm}")
                                if entry_signal["entry_family"] == "reversal" and rev.get("confidence") is not None:
                                    _print_and_log(f"       置信度: {float(rev['confidence']):.0%}")

                            elif has_signal and had_signal:
                                _, sig_cn = _intraday_watch_signal_text(current_watch, str(rev.get("signal_type") or ""))
                                _print_and_log(f"  🟢 {current_watch['name']}: {sig_cn}确认持续 | {live_info}")
                                _print_and_log(f"       {_format_intraday_trade_plan(current_watch)}")
                                if plan_changed:
                                    _print_and_log("       🔄 盘中重评后交易计划已更新")
                                if actionable_upgraded:
                                    _print_and_log(
                                        f"       ✅ 盘中重评后转为可执行：评分{float(current_watch.get('score', 0.0)):+.0f} "
                                        f"准入RR {float(current_watch.get('admission_rr', current_watch.get('rr', 0.0))):.2f}"
                                    )

                            elif not has_signal and had_signal:
                                waiting_text = _default_waiting_hint(current_watch)
                                _print_and_log(f"  ❌ {current_watch['name']}: 确认已回退，重新{waiting_text}")
                                _print_and_log(f"       {live_info}")

                            else:
                                entry_family = str(_resolve_entry_signal(current_watch)["entry_family"])
                                if entry_family == "reversal":
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
                                        waiting_text = stage or _default_waiting_hint(current_watch)
                                        log.info("  ⏳ %s: %s | %s", w["name"], waiting_text, live_info)
                                else:
                                    waiting_text = _default_waiting_hint(current_watch)
                                    log.info("  ⏳ %s: %s | %s", w["name"], waiting_text, live_info)

                        except Exception as e:
                            log.warning("  ⚠️ %s: %s", w["name"], e)
                        finally:
                            prev_signals[sym] = has_signal
                            prev_watch_rows[sym] = current_watch

                prev_hold_actions = run_phase_4_holdings(
                    config,
                    period=period,
                    emit_terminal=True,
                    previous_actions=prev_hold_actions,
                    live_frames=current_cycle_live_frames,
                    log_header=False,
                )

                if not is_trading_hours():
                    ended_after_trading_hours = True
                    _print_and_log("\n  🔔 交易时段已结束，正在关闭行情连接…")
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
                _print_and_log("\n\n  🛑 监控已停止")
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
        return f"⏳ {_default_waiting_hint(row)}"

    family_cn = str(entry_signal["family_cn"])
    family_prefix = f"[{family_cn}] " if family_cn else ""
    signal_date = str(entry_signal["signal_date"])
    if signal_date:
        return f"{family_prefix}{entry_signal['display_label']} ({signal_date})"
    return f"{family_prefix}{entry_signal['display_label']}"


def _default_waiting_hint(row: dict) -> str:
    entry_family = _resolve_entry_signal(row)["entry_family"]
    if entry_family == "trend":
        return "等待顺势确认"
    if entry_family == "reversal":
        return "等待反转确认"
    return "等待交易故事确认"


def _format_watchlist_story_status(row: dict) -> str:
    entry_signal = _resolve_entry_signal(row)
    if entry_signal["has_signal"]:
        family_cn = str(entry_signal["family_cn"] or "交易故事")
        signal_label = str(entry_signal["display_label"] or "")
        if signal_label:
            return f"已出现{family_cn}确认（{signal_label}），待执行条件改善"
        return f"已出现{family_cn}确认，待执行条件改善"

    if entry_signal["entry_family"] == "trend":
        return _default_waiting_hint(row)

    reversal = row.get("reversal_status")
    if not isinstance(reversal, dict):
        reversal = {}
    if entry_signal["entry_family"] == "reversal":
        return str(reversal.get("next_expected") or _default_waiting_hint(row))
    return _default_waiting_hint(row)


def _resolve_strategy_plan_mode(row: dict) -> str:
    """根据原始计划的中文语义，判断这笔机会属于顺势框架还是反转框架。"""
    normalized = _normalize_phase2_plan_row(row)
    strategy_family = str(normalized.get("strategy_family") or "").strip()
    entry_family = str(normalized.get("entry_family") or "").strip()
    if strategy_family == STRATEGY_TREND:
        return "trend"
    if strategy_family == STRATEGY_REVERSAL:
        return "reversal"
    if entry_family in {"trend", "reversal"}:
        return entry_family
    return ""


def _build_strategy_scoped_trade_plan(
    *,
    base_row: dict,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame,
    cfg: dict,
    allow_watch_plan: bool = False,
) -> dict | None:
    """
    按原始计划所属的策略框架重算交易计划。

    中文原则：
    - 第二阶段先确定“这笔机会属于哪种策略框架”。
    - 第三阶段和第四阶段只能在这个既定框架里更新计划。
    - 只有原始记录缺少策略归属时，才退回通用重评逻辑。
    """
    mode = _resolve_strategy_plan_mode(base_row)
    kwargs = {
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "df": df,
        "cfg": cfg,
    }
    if mode == "trend":
        return trend_pre_market.build_trade_plan_from_daily_df(
            **kwargs,
            allow_watch_plan=allow_watch_plan,
        )
    if mode == "reversal":
        return reversal_pre_market.build_trade_plan_from_daily_df(
            **kwargs,
            allow_watch_plan=allow_watch_plan,
        )
    return build_trade_plan_from_daily_df(**kwargs)


def run_phase_4_holdings(
    config: dict,
    *,
    period: str,
    emit_terminal: bool,
    previous_actions: dict[str, str] | None = None,
    live_frames: dict[str, pd.DataFrame] | None = None,
    log_header: bool = True,
) -> dict[str, str]:
    """
    持仓管理阶段。

    这一步不负责重新选品种，也不负责改换策略。它只使用与第三阶段相同的盘中快照，
    再结合原始计划和真实持仓成本，输出主结论（继续持有 / 减仓观察 / 平仓）
    以及必要的附加保护动作。
    """
    previous_actions = previous_actions or {}
    live_frames = live_frames or {}
    workbook_path = find_daily_holdings_workbook(
        root_dir=default_holdings_root(),
        trade_date=datetime.now().strftime("%Y-%m-%d"),
    )
    if workbook_path is None:
        if log_header:
            log.info("\n  未找到今日持仓文件，跳过 Phase 4")
        return previous_actions

    if log_header:
        log.info("\n" + "▓" * 60)
        log.info("▓  Phase 4: 持仓建议")
        log.info("▓" * 60)

    try:
        contexts = load_holding_contexts(workbook_path)
    except Exception as exc:
        log.warning("  ⚠️ 持仓文件读取失败: %s", exc)
        return previous_actions

    if not contexts:
        log.info("  今日持仓文件为空，跳过 Phase 4")
        return {}

    results: list[dict] = []
    actions: dict[str, str] = {}
    for context in contexts:
        recommendation = context.get("recommendation")
        if not recommendation:
            recommendation = {
                "record_id": context.get("record_id"),
                "symbol": (context.get("holding") or {}).get("symbol"),
                "name": (context.get("holding") or {}).get("name"),
                "direction": (context.get("holding") or {}).get("direction"),
                "entry": (context.get("holding") or {}).get("entry_price"),
            }

        holding = dict(context["holding"])
        symbol = str(holding.get("symbol") or "")
        name = str(holding.get("name") or recommendation.get("name") or symbol)
        direction = str(holding.get("direction") or recommendation.get("direction") or "")
        pre_cfg = _pre_market_cfg_for_symbol(config, symbol)

        live_df = live_frames.get(symbol)
        if live_df is None or live_df.empty:
            live_df = get_daily_with_live_bar(symbol, period)
        if live_df is None or live_df.empty:
            log.warning("  ⚠️ %s: 无法获取持仓行情，跳过", name)
            continue

        plan_mode = _resolve_strategy_plan_mode(recommendation)
        if plan_mode == "trend":
            hold_eval = assess_active_trend_hold_from_daily_df(
                symbol=symbol,
                name=name,
                direction=direction,
                df=live_df,
                cfg=pre_cfg,
            )
        elif plan_mode == "reversal":
            hold_eval = assess_active_reversal_hold_from_daily_df(
                symbol=symbol,
                name=name,
                direction=direction,
                df=live_df,
                cfg=pre_cfg,
            )
        else:
            hold_eval = {"hold_valid": True, "snapshot_missing": True}

        if plan_mode in {"trend", "reversal"}:
            current_plan = _build_strategy_scoped_trade_plan(
                base_row=recommendation,
                symbol=symbol,
                name=name,
                direction=direction,
                df=live_df,
                cfg=pre_cfg,
                allow_watch_plan=True,
            ) or {}
        else:
            current_plan = dict(recommendation)

        holding["name"] = name
        result = analyze_holding_record(
            holding=holding,
            recommendation=recommendation,
            current_plan=current_plan,
            hold_eval=hold_eval,
            current_price=float(live_df.iloc[-1]["close"]),
            minimum_tick=1.0,
            account_equity=_as_float(pre_cfg.get("account_equity"), 0.0),
            risk_per_trade_pct=_as_float(pre_cfg.get("risk_per_trade_pct"), 0.0),
            contract_multiplier=_contract_multiplier_for_symbol(pre_cfg, symbol),
        )
        result["original_plan_line"] = format_plan_line("原始计划", result["original_plan"])
        result["current_plan_line"] = format_plan_line("当前重评", result["current_plan"])
        for line in format_holding_log_block(result).splitlines():
            log.info(line)
        if emit_terminal and should_emit_terminal_alert(result, previous_actions):
            _print_and_log(format_holding_terminal_alert(result))

        record_id = str(result.get("record_id") or "")
        actions[record_id] = holding_alert_state_key(result)
        results.append(result)

    if results:
        log.info("  %s", build_holdings_summary_line(results))
    return actions


def _print_and_log(message: str) -> None:
    print(message)
    log.info(message)


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


def _format_intraday_execution_line(row: dict, minute_df: pd.DataFrame) -> str:
    entry_signal = _resolve_entry_signal(row)
    entry_family = str(entry_signal["entry_family"])
    direction = str(row.get("direction") or "")
    entry = float(row.get("entry") or 0.0)
    stop = float(row.get("stop") or 0.0)
    if entry <= 0 or stop <= 0 or minute_df is None or minute_df.empty:
        return ""

    last = float(minute_df["close"].iloc[-1])
    session_high = float(minute_df["high"].max())
    session_low = float(minute_df["low"].min())

    if entry_family == "trend":
        return "│ 🎯 " + _trend_execution_status_text(
            direction=direction,
            signal_type=str(entry_signal["signal_type"] or ""),
            entry=entry,
            stop=stop,
            last=last,
            session_high=session_high,
            session_low=session_low,
        )

    if entry_family == "reversal":
        reversal = row.get("reversal_status")
        if not isinstance(reversal, dict):
            reversal = {}
        return "│ 🎯 " + _reversal_execution_status_text(
            direction=direction,
            entry=entry,
            stop=stop,
            last=last,
            session_high=session_high,
            session_low=session_low,
            reversal=reversal,
        )

    return ""


def _trend_execution_status_text(
    *,
    direction: str,
    signal_type: str,
    entry: float,
    stop: float,
    last: float,
    session_high: float,
    session_low: float,
) -> str:
    if direction == "long":
        if session_low <= stop:
            return f"顺势执行: 今日已触及止损{stop:.0f}，原顺势计划暂不执行"
        if signal_type == "Pullback":
            if last >= entry:
                return f"顺势执行: 回踩后已重新站回入场{entry:.0f}上方，可按顺势回踩执行"
            return f"顺势执行: 等待回踩后重新站回入场{entry:.0f}上方，且不跌破止损{stop:.0f}"
        if last >= entry or session_high >= entry:
            return f"顺势执行: 价格已触及入场{entry:.0f}附近，继续观察能否站稳且不跌破止损{stop:.0f}"
        return f"顺势执行: 等待价格重新站上入场{entry:.0f}，且不跌破止损{stop:.0f}"

    if session_high >= stop:
        return f"顺势执行: 今日已触及止损{stop:.0f}，原顺势计划暂不执行"
    if signal_type == "Pullback":
        if last <= entry:
            return f"顺势执行: 反弹后已重新压回入场{entry:.0f}下方，可按顺势回撤执行"
        return f"顺势执行: 等待反弹后重新压回入场{entry:.0f}下方，且不突破止损{stop:.0f}"
    if last <= entry or session_low <= entry:
        return f"顺势执行: 价格已触及入场{entry:.0f}附近，继续观察能否压回且不突破止损{stop:.0f}"
    return f"顺势执行: 等待价格重新回到入场{entry:.0f}下方，且不突破止损{stop:.0f}"


def _reversal_execution_status_text(
    *,
    direction: str,
    entry: float,
    stop: float,
    last: float,
    session_high: float,
    session_low: float,
    reversal: dict,
) -> str:
    signal_bar = reversal.get("signal_bar") or {}
    if direction == "long":
        confirm = float(signal_bar.get("low") or stop)
        if session_low <= stop:
            return f"反转执行: 今日已触及止损{stop:.0f}，原反转计划失效"
        if last >= entry and session_low > confirm:
            return f"反转执行: 已守住确认位{confirm:.0f}并站回入场{entry:.0f}上方，可按反转执行"
        return f"反转执行: 等待守住确认位{confirm:.0f}后重新站回入场{entry:.0f}上方"

    confirm = float(signal_bar.get("high") or stop)
    if session_high >= stop:
        return f"反转执行: 今日已触及止损{stop:.0f}，原反转计划失效"
    if last <= entry and session_high < confirm:
        return f"反转执行: 已守住确认位{confirm:.0f}并压回入场{entry:.0f}下方，可按反转执行"
    return f"反转执行: 等待守住确认位{confirm:.0f}后重新压回入场{entry:.0f}下方"


def _intraday_watch_signal_text(row: dict, fallback_signal_type: str) -> tuple[str, str]:
    entry_signal = _resolve_entry_signal(row)
    entry_family = str(entry_signal["entry_family"])
    fallback_label = ENTRY_SIGNAL_CN.get(fallback_signal_type, fallback_signal_type)

    if entry_family == "trend":
        return "顺势确认", str(entry_signal["display_label"] or fallback_label or "顺势")
    if entry_family == "reversal":
        return "反转确认", str(fallback_label or entry_signal["display_label"] or "反转")
    return "交易故事确认", str(entry_signal["display_label"] or fallback_label or "入场")


def _intraday_watch_confirmation_text(row: dict) -> str:
    entry_signal = _resolve_entry_signal(row)
    entry_family = str(entry_signal["entry_family"])
    direction = str(row.get("direction") or "")
    entry = float(row.get("entry") or 0.0)
    stop = float(row.get("stop") or 0.0)
    reversal = row.get("reversal_status")
    if not isinstance(reversal, dict):
        reversal = {}

    if entry_family == "trend" and entry > 0 and stop > 0:
        signal_type = str(entry_signal["signal_type"] or "")
        if direction == "long":
            if signal_type == "Pullback":
                return f"回踩不跌破止损{stop:.0f}，并重新站回入场{entry:.0f}上方"
            return f"价格继续站稳入场{entry:.0f}上方，且不跌破止损{stop:.0f}"
        if signal_type == "Pullback":
            return f"反弹不突破止损{stop:.0f}，并重新压回入场{entry:.0f}下方"
        return f"价格继续压在入场{entry:.0f}下方，且不突破止损{stop:.0f}"

    signal_bar = reversal.get("signal_bar") or {}
    if direction == "long" and signal_bar.get("low") is not None:
        return f"收盘维持在 {float(signal_bar['low']):.0f} 以上"
    if direction == "short" and signal_bar.get("high") is not None:
        return f"收盘维持在 {float(signal_bar['high']):.0f} 以下"
    return _default_waiting_hint(row)


def _merge_intraday_watch_plan(base_row: dict, live_plan: dict | None) -> dict:
    base = _normalize_phase2_plan_row(base_row)
    if not isinstance(live_plan, dict):
        return base
    live = _normalize_phase2_plan_row(live_plan)
    merged = dict(base)
    merged.update(live)
    for key in ("strategy_family", "entry_family"):
        if not str(merged.get(key) or "").strip():
            merged[key] = base.get(key, "")
    return _normalize_phase2_plan_row(merged)


def _intraday_watch_plan_changed(previous_row: dict, current_row: dict) -> bool:
    return _format_intraday_trade_plan(previous_row) != _format_intraday_trade_plan(current_row)


def save_targets(
    targets: list[dict],
    watchlist: list[dict] | None = None,
    phase1_summary: dict | None = None,
    phase2_pre_sizing_candidates: list[dict] | None = None,
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
    if phase2_pre_sizing_candidates:
        phase2_pre_sizing_candidates = _normalize_phase2_plan_rows(_clean_numpy(phase2_pre_sizing_candidates))
    else:
        phase2_pre_sizing_candidates = []
    if phase1_summary is None:
        phase1_summary = build_phase1_summary(top_n=max(len(targets), len(watchlist)))

    json_path = _today_json_path()
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "targets": targets,
        "watchlist": watchlist,
        "phase2_pre_sizing_candidates": phase2_pre_sizing_candidates,
        "strategy_results": _clean_numpy(grouped_results or {}),
        "phase1_summary": phase1_summary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    md_path = _today_md_path()
    now = datetime.now()

    has_actionable = len(targets) > 0
    has_watchlist = len(watchlist) > 0
    has_phase2_pre_sizing_candidates = len(phase2_pre_sizing_candidates) > 0

    lines = [
        f"# 每日交易建议 {now.strftime('%Y-%m-%d')}",
        "",
        f"> 生成时间: {now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    if has_actionable:
        lines.append("## 可执行品种一览")
    elif has_watchlist:
        lines.append("## 今日等待确认（尚无可执行品种）")
        lines.append("")
        lines.append(
            "> **未满足可执行三要件**：① 交易故事已确认 ② Phase 2 达标（做多>+20 / 做空<-20）③ 盈亏比≥1。"
        )
        lines.append(
            "> **方向**由 Phase 2 技术面总分解析得到；Phase 1 只负责给出关注候选、关注分与机会标签。"
            "机会标签描述的是基本面机会类型，不直接等同于最终交易方向。"
        )
    elif has_phase2_pre_sizing_candidates:
        lines.append("## 今日无资金管理后可执行/等待确认品种")
        lines.append("")
        lines.append("> Phase 2 策略已有候选，但资金管理后没有最终可开仓/等待确认条目。")
    else:
        lines.append("## 今日无信号")
        lines.append("")
        lines.append("> 全市场无极端品种，今日无操作/等待确认机会。")

    def _md_cell(val) -> str:
        """Markdown 表格单元格内若含 | 会拆列，统一换成全角竖线。"""
        if val is None:
            return "-"
        return str(val).replace("|", "｜")

    def _format_risk_control_note(t: dict) -> str:
        try:
            risk_budget = float(t.get("risk_budget") or 0.0)
            risk_per_lot = float(t.get("risk_per_lot") or 0.0)
            score_margin_budget = float(t.get("score_margin_budget") or 0.0)
            score_lots = int(t.get("score_lots") or 0)
            portfolio_lots = int(t.get("portfolio_lots") or 0)
            risk_lots = int(t.get("risk_lots") or 0)
            suggested_lots = int(t.get("suggested_lots") or 0)
            margin_per_lot = float(t.get("margin_per_lot") or 0.0)
        except (TypeError, ValueError):
            return ""
        if score_margin_budget > 0 or score_lots > 0 or portfolio_lots > 0 or risk_lots > 0:
            note = (
                f"Phase2权重预算{score_margin_budget:.0f}元，"
                f"分数仓位{score_lots}手，组合上限{portfolio_lots}手"
            )
            if risk_budget > 0:
                note += f"，止损风险上限{risk_lots}手"
            note += f"，建议{suggested_lots}手"
            if margin_per_lot > 0:
                note += f"；每手保证金约{margin_per_lot:.0f}元"
            if risk_per_lot > 0:
                note += f"，每手止损约{risk_per_lot:.0f}元"
            return note
        if risk_budget <= 0 or risk_per_lot <= 0:
            return ""
        if suggested_lots > 0:
            lots_text = f"建议不超过{suggested_lots}手"
        else:
            lots_text = "当前不满足单笔止损预算"
        note = f"单笔止损预算{risk_budget:.0f}元，每手止损约{risk_per_lot:.0f}元，{lots_text}"
        if margin_per_lot > 0:
            note += f"；每手保证金约{margin_per_lot:.0f}元"
        return note

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

    def _phase2_pre_sizing_status(row: dict, final_status: dict[str, str]) -> str:
        symbol = str(row.get("symbol") or "")
        if symbol and symbol in final_status:
            return final_status[symbol]
        return "资金管理后未保留"

    def _render_phase2_pre_sizing_pool() -> None:
        if not phase2_pre_sizing_candidates:
            return
        final_status = {
            str(row.get("symbol") or ""): "最终可执行"
            for row in targets
            if str(row.get("symbol") or "")
        }
        final_status.update(
            {
                str(row.get("symbol") or ""): "资金管理后观察/等待确认"
                for row in watchlist
                if str(row.get("symbol") or "")
            }
        )
        lines.append("")
        lines.append("## Phase2资金管理前候选池")
        lines.append("")
        lines.append("| 合约 | 代码 | 方向 | P2分 | 入场 | 止损 | 止盈1 | RR | 每手保证金 | 每手止损 | 后续状态 |")
        lines.append("| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
        for row in phase2_pre_sizing_candidates:
            name = _md_cell(f"{row.get('name', row.get('symbol', ''))}(主力)")
            symbol = _md_cell(row.get("symbol") or "")
            direction = _direction_cn(str(row.get("direction") or ""))
            lines.append(
                f"| {name} | {symbol} | {direction} "
                f"| {_as_float(row.get('score')):+.0f} "
                f"| {_as_float(row.get('entry')):.0f} "
                f"| {_as_float(row.get('stop')):.0f} "
                f"| {_as_float(row.get('tp1')):.0f} "
                f"| {_as_float(row.get('rr')):.2f} "
                f"| {_as_float(row.get('margin_per_lot')):.0f} "
                f"| {_as_float(row.get('risk_per_lot')):.0f} "
                f"| {_phase2_pre_sizing_status(row, final_status)} |"
            )

    def _fundamental_domain_list(domains) -> str:
        labels = [
            FUNDAMENTAL_DOMAIN_CN.get(str(domain), str(domain))
            for domain in (domains or [])
        ]
        return " / ".join(labels) if labels else "无"

    def _has_fundamental_snapshot_fields(t: dict) -> bool:
        return any(
            key in t
            for key in (
                "fundamental_reversal_confirmed",
                "fundamental_coverage_score",
                "fundamental_domains_present",
                "fundamental_domains_missing",
                "fundamental_missing_domain_reasons",
            )
        )

    def _append_fundamental_snapshot_lines(t: dict) -> bool:
        if not _has_fundamental_snapshot_fields(t):
            return False

        confirmed = bool(t.get("fundamental_reversal_confirmed"))
        coverage = _float_or_none(t.get("fundamental_coverage_score"))
        status = str(t.get("fundamental_coverage_status") or "")
        status_cn = {
            "complete": "完整",
            "partial": "不完整",
            "missing": "缺失",
        }.get(status, "不完整" if coverage is not None and coverage < 1.0 else "完整")

        lines.append(f"**基本面反转**: {'已确认' if confirmed else '未确认，仅观察'}")
        if coverage is not None:
            lines.append(f"- 基本面覆盖: {status_cn} ({coverage * 100:.0f}%)")
        else:
            lines.append(f"- 基本面覆盖: {status_cn}")
        lines.append(f"- 已有产业证据: {_md_cell(_fundamental_domain_list(t.get('fundamental_domains_present')))}")
        lines.append(f"- 缺失产业证据: {_md_cell(_fundamental_domain_list(t.get('fundamental_domains_missing')))}")

        extreme_reasons = list(t.get("fundamental_extreme_state_reasons") or [])
        turn_reasons = list(t.get("fundamental_marginal_turn_reasons") or [])
        if extreme_reasons:
            lines.append(f"- 极值状态: {_md_cell(' / '.join(str(item) for item in extreme_reasons))}")
        elif not bool(t.get("fundamental_extreme_state_confirmed")):
            lines.append("- 极值状态: 未确认")
        if turn_reasons:
            lines.append(f"- 边际转向: {_md_cell(' / '.join(str(item) for item in turn_reasons))}")
        elif not bool(t.get("fundamental_marginal_turn_confirmed")):
            lines.append("- 边际转向: 未确认")

        missing_reasons = list(t.get("fundamental_missing_domain_reasons") or [])
        if missing_reasons and not confirmed:
            lines.append(f"- 不建议开仓: {_md_cell(' / '.join(str(item) for item in missing_reasons))}")
        lines.append("")
        return True

    if targets:
        lines.append("")
        lines.append(
            "| 状态 | 合约 | 方向 | 建议手数 | 关注理由 | 当前价 | 交易故事确认 | 入场价 | 止损 | 止盈1 | 盈亏比 | RRF | #P1 | #P2 | 关注分 | P2分 | 信号强度 | 机会标签 | Phase1摘要 |"
        )
        lines.append(
            "| :---: | :--- | :---: | ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"
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
            suggested_lots = int(t.get("suggested_lots") or 0)
            lines.append(
                f"| {_md_cell('✅可执行')} | {name_cell} | {dir_cell} "
                f"| {suggested_lots}手 | {epr} | {t['price']:.0f} | {_md_cell(signal_str)} | {t['entry']:.0f} | {t['stop']:.0f} "
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
            next_exp = _format_watchlist_story_status(t)
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

    _render_phase2_pre_sizing_pool()

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
        "可执行条件：① 有新鲜反转/顺势确认 ② Phase 2 总分达标(做多>+20/做空<-20) ③ 盈亏比≥1.0。"
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
    lines.append("### 交易故事确认说明")
    lines.append("")
    lines.append("系统采用策略池：各策略独立命中，多个同方向命中代表共振增强；反方向命中代表策略冲突。")
    lines.append("同方向共振只提高关注等级；最终交易只能选择一个交易剧本执行，不能混用不同剧本的入场、止损和止盈。")
    lines.append("")
    lines.append("#### 模式一：Wyckoff 反转确认（抓顶抄底）")
    lines.append("")
    lines.append("适用于横盘/筑底/筑顶阶段，等待极端事件确认反转：")
    lines.append("")
    lines.append("**做多反转序列**: 下跌 → SC(卖方高潮) → 停止量 → Spring(弹簧) → SOS(强势突破)")
    lines.append("**做空反转序列**: 上涨 → BC(买方高潮) → 停止量 → UT(上冲回落) → SOW(弱势跌破)")
    lines.append("")
    lines.append("| 信号 | 含义 | 确认级别 |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| SC/BC | 放量极端K线，趋势力量耗尽 | 预备信号，不入场 |")
    lines.append("| 停止量 | 放量但幅度窄，对手方被吸收 | 预备信号，不入场 |")
    lines.append("| Spring/UT | 假突破后收回，经典确认点 | **反转确认** |")
    lines.append("| SOS/SOW | 放量突破确认趋势反转 | **确认信号** |")
    lines.append("")
    lines.append("入场价 = 当前价（确认新鲜时执行） | 止损 = 信号K线极端价 ± 0.5ATR")
    lines.append("")
    lines.append("#### 模式二：顺势确认（趋势延续）")
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
        strategy_hits = t.get("strategy_pool_hits") or []
        if strategy_hits:
            resonance_cn = {
                "same_direction": "同方向共振",
                "conflict": "方向冲突",
                "single_strategy": "单策略命中",
            }.get(str(t.get("strategy_resonance") or ""), "策略池命中")
            playbook_cn = {
                "trend_continuation": "趋势延续",
                "trend_pullback": "趋势回撤",
                "fundamental_mean_reversion": "基本面均值回归",
                "technical_rebound_watch": "技术反弹观察",
                "conflict_watch": "冲突观察",
                "observation": "观察",
            }.get(str(t.get("selected_playbook") or ""), "观察")
            market_stage_cn = MARKET_STAGE_CN.get(str(t.get("market_stage") or ""), "观察")
            confluence_cn = CONFLUENCE_QUALITY_CN.get(str(t.get("confluence_quality") or ""), "单域支撑")
            evidence_summary = t.get("independent_evidence_summary") or []
            evidence_count = int(t.get("independent_evidence_count") or 0)
            unselected_reason = str(t.get("unselected_playbook_reason") or "")
            lines.append(f"**策略池命中**: {resonance_cn}，执行剧本: {playbook_cn}")
            lines.append(f"**市场阶段**: {market_stage_cn}")
            lines.append(f"**共振质量**: {confluence_cn}（{evidence_count}个独立证据）")
            if evidence_summary:
                lines.append(f"**独立证据**: {_md_cell(' / '.join(str(item) for item in evidence_summary))}")
            if unselected_reason:
                lines.append(f"**未选剧本原因**: {_md_cell(unselected_reason)}")
            for hit in strategy_hits:
                family_cn = {
                    STRATEGY_TREND: "趋势跟随",
                    STRATEGY_REVERSAL: "基本面均值回归",
                }.get(str(hit.get("strategy_family") or ""), "其他策略")
                hit_dir = _direction_cn(str(hit.get("direction") or ""))
                hit_state = "可执行" if bool(hit.get("actionable")) else "观察"
                hit_score = float(hit.get("score") or 0.0)
                lines.append(f"- {family_cn}: {hit_dir}，{hit_state}，评分{hit_score:+.0f}")
            lines.append("")

        # --- 交易故事确认状态 ---
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
            lines.append(f"**交易故事确认**: {family_prefix}{sig_cn}{signal_suffix}")
            lines.append(f"- 确认详情: {_md_cell(str(entry_signal.get('signal_detail', '') or ''))}")
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
                actionable_now = _is_actionable(t)
                if signal_date:
                    if actionable_now:
                        lines.append(f"- 入场依据: 当前价（确认于{signal_date}触发，新鲜可执行）")
                    else:
                        lines.append(f"- 入场依据: 当前价（确认于{signal_date}触发，但未达执行条件）")
                else:
                    if actionable_now:
                        lines.append(f"- 入场依据: 当前价（反转确认触发，新鲜可执行）")
                    else:
                        lines.append(f"- 入场依据: 当前价（反转确认已触发，但未达执行条件）")
                if t["direction"] == "long":
                    lines.append(f"- 止损依据: 信号K线最低点 - 0.5×ATR")
                else:
                    lines.append(f"- 止损依据: 信号K线最高点 + 0.5×ATR")
            if t.get("entry_plan_type") == "extended_target":
                lines.append("- 计划类型: 依赖第二止盈的延展计划")
            management_note = str(t.get("management_note") or "")
            if management_note:
                lines.append(f"- 仓位管理: {_md_cell(management_note)}")
            risk_note = _format_risk_control_note(t)
            if risk_note:
                lines.append(f"- 仓位建议: {_md_cell(risk_note)}")
        else:
            lines.append(f"**交易故事确认**: {_format_entry_signal_for_detail(t)}")
            lines.append(f"- 当前阶段: {_md_cell(rev.get('current_stage', '未知'))}")
            lines.append(f"- 等待条件: {_md_cell(rev.get('next_expected') or _default_waiting_hint(t))}")
            suspect = rev.get("suspect_events", [])
            if suspect:
                lines.append(f"- ⚠️ 有{len(suspect)}个疑似信号（未通过上下文验证）:")
                for se in suspect[-3:]:
                    lines.append(
                        f"  - {_md_cell(se.get('date', ''))} {_md_cell(se.get('signal', ''))}: {_md_cell(se.get('detail', ''))}"
                    )
            if t["direction"] == "long":
                lines.append(f"- 反转确认: Spring(假跌破收回) 或 SOS(放量突破)")
                lines.append(f"- 顺势确认: 缩量回踩MA20企稳 或 放量突破前高")
            else:
                lines.append(f"- 反转确认: UT(假突破回落) 或 SOW(放量跌破)")
                lines.append(f"- 顺势确认: 缩量反弹MA20失败 或 放量跌破前低")
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
        if not _is_actionable(t):
            entry_signal_now = _resolve_entry_signal(t)
            reasons = []
            downgrade_reason = str(t.get("downgrade_reason") or "")
            if downgrade_reason:
                reasons.append(downgrade_reason)
            if not entry_signal_now["has_signal"]:
                reasons.append("交易故事尚未确认")
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
                lines.append(f"- ⚠️ 未达执行条件: {_md_cell(', '.join(reasons))}")
        lines.append("")

        lines.append("**基本面数据**")
        lines.append("")
        has_any = False
        if _append_fundamental_snapshot_lines(t):
            has_any = True
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
        _build_detail(t, "等待确认")

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

    print(f"{build_workflow_contract_summary()} 详细日志见 {get_log_path()}。")
    log.info("╔" + "═" * 58 + "╗")
    log.info("║" + "每日交易工作流".center(48) + "║")
    log.info("║" + f"  {now.strftime('%Y-%m-%d %H:%M')}".ljust(56) + "║")
    log.info("╚" + "═" * 58 + "╝")
    log.info("  %s", build_workflow_contract_summary())
    for line in build_workflow_contract_lines():
        log.info("  - %s", line)

    # --- 恢复模式 ---
    if args.resume:
        snapshot = load_targets()
        if snapshot is not None:
            targets, watchlist_resumed = snapshot
            log.info(
                "\n  📂 恢复今日目标: %s 个可执行, %s 个等待确认",
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
                    "     👀 %s (%s) %s  评分%s [等待确认]",
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
            else:
                run_phase_4_holdings(config, period=args.period, emit_terminal=False)
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
                "trend_direction": pos["direction"],
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
        _log_trend_bridge_diagnostics(
            phase1_candidates=reversal_candidates,
            trend_candidates=trend_candidates,
            all_data=all_data,
            config=config,
        )

    if not reversal_candidates and not trend_candidates:
        log.info("\n  😴 今日无可跟踪品种")
        run_phase_4_holdings(config, period=args.period, emit_terminal=False)
        if not args.no_monitor:
            phase_3_intraday([], config, args.period, args.interval, watchlist=[])
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
        phase2_pre_sizing_candidates=(grouped_results.get("phase2_pre_sizing") or {}).get("actionable") or [],
        grouped_results=grouped_results,
    )

    run_phase_4_holdings(config, period=args.period, emit_terminal=False)

    # --- Phase 3: 盘中监控 ---
    if args.no_monitor:
        log.info("\n  ⏩ 跳过盘中监控 (--no-monitor)")
        api_total = get_request_count()
        log.info("  📊 本次运行累计AkShare请求: %s 次", api_total)
        return

    phase_3_intraday(actionable, config, args.period, args.interval, watchlist=top_watch)


if __name__ == "__main__":
    main()
