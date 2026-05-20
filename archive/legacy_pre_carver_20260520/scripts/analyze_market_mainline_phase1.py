from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_single_campaign_trader import CampaignBacktestResult, run_parameter_sweep, sniper_entry_score
from analyze_split_execution_factors import _finite_float
from analyze_trend_opportunity_quality import (
    DEFAULT_MARKET_CACHE_DIR,
    DEFAULT_PHASE23_CACHE_DIR,
    collect_trend_opportunity_rows,
)
from backtest.market_mainline_phase1 import (
    MarketMainlineParams,
    build_market_mainline_state_rows,
    commodity_group,
    discover_cached_symbols,
    load_market_data,
)


DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/market_mainline_phase1_2022_2025")

BUCKET_ORDER = {"missing": 0, "low": 1, "medium": 2, "high": 3, "crowded": 4}


@dataclass(frozen=True)
class MainlineAnalysisParams:
    transition_window_days: int = 20
    reset_low_days: int = 5
    start_year: int = 2022
    end_year: int = 2025
    profile: str = "aggressive"


@dataclass(frozen=True)
class MainlineOutputPaths:
    states_csv: Path
    events_csv: Path
    episodes_csv: Path
    trades_csv: Path
    representative_trades_csv: Path
    stage_summary_csv: Path
    policy_csv: Path
    campaign_csv: Path
    campaign_event_csv: Path
    equity_csv: Path
    summary_json: Path
    report_md: Path


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _normalize_bucket(value: Any) -> str:
    bucket = str(value or "").strip()
    return bucket if bucket in BUCKET_ORDER else "missing"


def _latest_row_before(rows: list[dict[str, Any]], date_value: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if str(row.get("date") or "") <= str(date_value)]
    return candidates[-1] if candidates else None


def _transition_type(prev_bucket: str, bucket: str) -> str:
    previous = _normalize_bucket(prev_bucket)
    current = _normalize_bucket(bucket)
    if previous == "missing" or previous == current:
        return ""
    return f"{previous}_to_{current}"


def _group_state_rows(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in state_rows:
        group = str(row.get("group") or row.get("mainline_group") or "").strip()
        direction = str(row.get("direction") or "").strip()
        date_value = str(row.get("date") or "").strip()
        if not group or not direction or not date_value:
            continue
        key = (group, direction, date_value)
        if key not in by_key:
            item = dict(row)
            item["symbol"] = ""
            item["group"] = group
            by_key[key] = item
    return [by_key[key] for key in sorted(by_key)]


def build_mainline_episodes(state_rows: list[dict[str, Any]], *, reset_low_days: int = 5) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _group_state_rows(state_rows):
        grouped.setdefault((str(row.get("group") or ""), str(row.get("direction") or "")), []).append(row)

    episodes: list[dict[str, Any]] = []
    for (group, direction), rows in sorted(grouped.items()):
        active: dict[str, Any] | None = None
        low_streak = 0
        previous_bucket = "low"
        for row in sorted(rows, key=lambda item: str(item.get("date") or "")):
            bucket = _normalize_bucket(row.get("bucket"))
            date_value = str(row.get("date") or "")
            if active is None:
                if bucket in {"medium", "high", "crowded"}:
                    active = {
                        "episode_id": f"{group}_{direction}_{date_value.replace('-', '')}",
                        "group": group,
                        "direction": direction,
                        "start_date": date_value,
                        "start_bucket": bucket,
                        "confirm_date": date_value if bucket in {"high", "crowded"} else "",
                        "end_date": "",
                        "end_reason": "",
                        "max_score": _finite_float(row.get("score"), math.nan),
                        "max_bucket": bucket,
                    }
                    low_streak = 0
                previous_bucket = bucket
                continue

            active["max_score"] = max(_finite_float(active.get("max_score"), 0.0), _finite_float(row.get("score"), 0.0))
            if BUCKET_ORDER[bucket] > BUCKET_ORDER[_normalize_bucket(active.get("max_bucket"))]:
                active["max_bucket"] = bucket
            if bucket in {"high", "crowded"} and not str(active.get("confirm_date") or ""):
                active["confirm_date"] = date_value

            if bucket == "low":
                low_streak += 1
            else:
                low_streak = 0
            if low_streak >= int(reset_low_days):
                active["end_date"] = date_value
                active["end_reason"] = f"low_{int(reset_low_days)}d_reset"
                episodes.append(active)
                active = None
                low_streak = 0
            previous_bucket = bucket
        if active is not None:
            episodes.append(active)
    return episodes


def _episodes_by_key(episodes: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode in episodes:
        grouped.setdefault((str(episode.get("group") or ""), str(episode.get("direction") or "")), []).append(episode)
    for values in grouped.values():
        values.sort(key=lambda item: str(item.get("start_date") or ""))
    return grouped


def _episode_for_date(episodes: list[dict[str, Any]], date_value: str) -> dict[str, Any] | None:
    target = pd.Timestamp(date_value)
    for episode in episodes:
        start = pd.Timestamp(episode.get("start_date"))
        end_value = str(episode.get("end_date") or "")
        end = pd.Timestamp(end_value) if end_value else pd.Timestamp.max
        if start <= target <= end:
            return episode
    return None


def build_mainline_events(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _group_state_rows(state_rows):
        grouped.setdefault((str(row.get("group") or ""), str(row.get("direction") or "")), []).append(row)
    events: list[dict[str, Any]] = []
    for (group, direction), rows in sorted(grouped.items()):
        prev_bucket = "missing"
        prev_score = math.nan
        for row in sorted(rows, key=lambda item: str(item.get("date") or "")):
            bucket = _normalize_bucket(row.get("bucket"))
            event_type = _transition_type(prev_bucket, bucket)
            if event_type:
                date_value = str(row.get("date") or "")
                events.append(
                    {
                        "event_id": f"{group}_{direction}_{date_value.replace('-', '')}_{event_type}",
                        "symbol": "",
                        "group": group,
                        "direction": direction,
                        "date": date_value,
                        "event_type": event_type,
                        "from_bucket": prev_bucket,
                        "to_bucket": bucket,
                        "prev_score": float(prev_score) if math.isfinite(prev_score) else math.nan,
                        "score": _finite_float(row.get("score"), math.nan),
                    }
                )
            prev_bucket = bucket
            prev_score = _finite_float(row.get("score"), math.nan)
    return events


def _stage_for_trade(state: dict[str, Any] | None, event: dict[str, Any] | None, days_since: int | None, window_days: int) -> str:
    bucket = _normalize_bucket((state or {}).get("bucket"))
    event_type = str((event or {}).get("event_type") or "")
    if event_type == "low_to_medium" and days_since is not None and days_since <= int(window_days):
        return "trial_low_to_medium"
    if event_type in {"medium_to_high", "medium_to_crowded", "low_to_high"} and days_since is not None and days_since <= int(window_days):
        return "confirm_medium_to_high"
    if bucket == "crowded":
        return "crowded_late"
    if bucket == "high":
        return "late_high"
    if bucket == "medium":
        return "medium_continuation"
    if bucket == "low":
        return "low_state"
    return "no_state"


def _copy_mainline_state_to_trade(row: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    if state is None:
        return out
    out["legacy_trend_opportunity_quality_score"] = row.get("trend_opportunity_quality_score")
    out["legacy_trend_structure_health_score"] = row.get("trend_structure_health_score")
    out["mainline_score"] = _finite_float(state.get("mainline_score"), math.nan)
    out["mainline_bucket"] = _normalize_bucket(state.get("bucket"))
    out["mainline_group"] = str(state.get("group") or "")
    out["mainline_leadership_score"] = _finite_float(state.get("mainline_leadership_score"), math.nan)
    out["mainline_capital_score"] = _finite_float(state.get("mainline_capital_score"), math.nan)
    out["mainline_resonance_score"] = _finite_float(state.get("mainline_resonance_score"), math.nan)
    out["mainline_structure_score"] = _finite_float(state.get("mainline_structure_score"), math.nan)
    out["mainline_crowding_score"] = _finite_float(state.get("mainline_crowding_score"), math.nan)
    out["mainline_leadership_rank_pct"] = _finite_float(state.get("leadership_rank_pct"), math.nan)
    out["mainline_group_breadth"] = _finite_float(state.get("group_breadth"), math.nan)
    out["term_structure_status"] = str(state.get("term_structure_status") or "not_available_in_continuous_cache")
    out["state_score"] = out["mainline_score"]
    out["state_bucket"] = out["mainline_bucket"]
    out["trend_opportunity_quality_score"] = out["mainline_score"]
    out["trend_structure_health_score"] = out["mainline_structure_score"]
    out["trend_participation_score"] = out["mainline_capital_score"]
    out["trend_remaining_space_score"] = max(0.0, 100.0 - _finite_float(state.get("mainline_crowding_score"), 0.0))
    out["trend_opportunity_bucket"] = out["mainline_bucket"]
    return out


def assign_trades_to_mainline_stages(
    trades: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    transition_window_days: int,
    episodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    states_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    events_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _group_state_rows(state_rows):
        states_by_key.setdefault((str(row.get("group") or ""), str(row.get("direction") or "")), []).append(row)
    for row in events:
        events_by_key.setdefault((str(row.get("group") or ""), str(row.get("direction") or "")), []).append(row)
    for values in states_by_key.values():
        values.sort(key=lambda item: str(item.get("date") or ""))
    for values in events_by_key.values():
        values.sort(key=lambda item: str(item.get("date") or ""))
    episodes_by_key = _episodes_by_key(episodes or build_mainline_episodes(state_rows))

    out: list[dict[str, Any]] = []
    seq_by_stage_event: Counter[tuple[str, str]] = Counter()
    for trade in sorted(trades, key=lambda item: (str(item.get("entry_time") or ""), str(item.get("symbol") or ""))):
        symbol = str(trade.get("symbol") or "").upper()
        group = commodity_group(symbol)
        direction = str(trade.get("direction") or "")
        entry_date = _date_str(trade.get("entry_time"))
        key = (group, direction)
        state = _latest_row_before(states_by_key.get(key, []), entry_date)
        event = _latest_row_before(events_by_key.get(key, []), entry_date)
        episode = _episode_for_date(episodes_by_key.get(key, []), entry_date)
        days_since = None
        if event is not None:
            days_since = int((pd.Timestamp(entry_date) - pd.Timestamp(event["date"])).days)
        stage = _stage_for_trade(state, event, days_since, transition_window_days)
        if episode is not None:
            confirm_date = str(episode.get("confirm_date") or "")
            start_date = str(episode.get("start_date") or "")
            current_bucket = _normalize_bucket((state or {}).get("bucket"))
            if current_bucket == "crowded":
                stage = "crowded_late"
            elif current_bucket == "high" and confirm_date and entry_date > confirm_date:
                stage = "late_high"
            elif confirm_date and entry_date >= confirm_date:
                stage = "confirm_medium_to_high"
            elif start_date and entry_date >= start_date:
                stage = "trial_low_to_medium"
        event_id = str((episode or {}).get("episode_id") or (event or {}).get("event_id") or "")
        seq_key = (stage, event_id or f"{group}_{direction}_{entry_date}")
        seq_by_stage_event[seq_key] += 1
        row = _copy_mainline_state_to_trade(dict(trade), state)
        row.update(
            {
                "transition_event_id": event_id,
                "transition_event_type": str((event or {}).get("event_type") or ""),
                "mainline_episode_id": str((episode or {}).get("episode_id") or ""),
                "mainline_episode_start_date": str((episode or {}).get("start_date") or ""),
                "mainline_episode_confirm_date": str((episode or {}).get("confirm_date") or ""),
                "days_since_transition": days_since if days_since is not None else "",
                "transition_stage": stage,
                "transition_trade_seq": int(seq_by_stage_event[seq_key]),
            }
        )
        out.append(row)
    return out


def _representative_rank(row: dict[str, Any]) -> tuple[int, int, float, str, str]:
    stage = str(row.get("transition_stage") or "")
    if stage == "confirm_medium_to_high":
        stage_rank = 0
    elif stage == "trial_low_to_medium":
        stage_rank = 1
    elif stage in {"late_high", "crowded_late"}:
        stage_rank = 2
    else:
        stage_rank = 3
    symbol = str(row.get("symbol") or "").upper()
    leader = str(row.get("group_leader_symbol") or "").upper()
    leader_rank = 0 if leader and symbol == leader else 1
    return (
        stage_rank,
        leader_rank,
        -float(sniper_entry_score(row)),
        str(row.get("entry_time") or ""),
        symbol,
    )


def select_representative_episode_trades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("transition_event_id") or row.get("mainline_episode_id") or "")
        stage = str(row.get("transition_stage") or "")
        if not episode_id or stage in {"no_state", "low_state", "medium_continuation"}:
            continue
        grouped.setdefault(episode_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        candidates = sorted(grouped[episode_id], key=_representative_rank)
        if candidates:
            item = dict(candidates[0])
            item["representative_selection_reason"] = "group_episode_representative"
            selected.append(item)
    return sorted(selected, key=lambda item: (str(item.get("entry_time") or ""), str(item.get("symbol") or "")))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _pct(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [_finite_float(row.get(field), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else math.nan


def _stage_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage in sorted({str(row.get("transition_stage") or "") for row in rows}):
        stage_rows = [row for row in rows if str(row.get("transition_stage") or "") == stage]
        if not stage_rows:
            continue
        wins = sum(1 for row in stage_rows if _finite_float(row.get("pnl_ratio"), 0.0) > 0)
        early = sum(1 for row in stage_rows if bool(row.get("early_stop")))
        tp2 = sum(1 for row in stage_rows if str(row.get("exit_reason") or "") == "tp2")
        out.append(
            {
                "transition_stage": stage,
                "trades": len(stage_rows),
                "win_rate": float(wins / len(stage_rows)),
                "early_stop_rate": float(early / len(stage_rows)),
                "tp2_rate": float(tp2 / len(stage_rows)),
                "avg_pnl_ratio": _mean(stage_rows, "pnl_ratio"),
                "avg_mainline_score": _mean(stage_rows, "mainline_score"),
                "avg_leadership_score": _mean(stage_rows, "mainline_leadership_score"),
                "avg_capital_score": _mean(stage_rows, "mainline_capital_score"),
                "avg_resonance_score": _mean(stage_rows, "mainline_resonance_score"),
                "avg_crowding_score": _mean(stage_rows, "mainline_crowding_score"),
            }
        )
    return out


def _policy_rank(row: dict[str, Any]) -> tuple[float, float]:
    return (_finite_float(row.get("return_pct"), 0.0), -_finite_float(row.get("max_drawdown_pct"), 0.0))


def _year_rows(selected: CampaignBacktestResult) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_year: dict[int, list[dict[str, Any]]] = {}
    for campaign in selected.campaigns:
        by_year.setdefault(int(pd.Timestamp(campaign["entry_time"]).year), []).append(campaign)
    for year in sorted(by_year):
        rows = by_year[year]
        wins = sum(1 for row in rows if _finite_float(row.get("net_pnl"), 0.0) > 0)
        out.append(
            {
                "year": year,
                "campaigns": len(rows),
                "return_contribution_pct": sum(_finite_float(row.get("net_pnl"), 0.0) for row in rows)
                / max(_finite_float(selected.summary.get("initial_equity"), 1.0), 1.0),
                "win_rate": float(wins / len(rows)) if rows else 0.0,
                "tp2_campaigns": sum(1 for row in rows if str(row.get("exit_reason") or "") == "tp2"),
                "early_campaign_exits": sum(
                    1 for row in rows if str(row.get("exit_reason") or "") not in {"tp2", "stop", "daily_reverse_confirmed"}
                ),
            }
        )
    return out


def render_markdown(
    *,
    diagnostics: dict[str, Any],
    stage_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    selected: CampaignBacktestResult,
) -> str:
    top_return = sorted(policy_rows, key=_policy_rank, reverse=True)[:12]
    controlled = [
        row
        for row in sorted(policy_rows, key=_policy_rank, reverse=True)
        if _finite_float(row.get("max_drawdown_pct"), 1.0) <= 0.30
    ][:12]
    margin50 = [
        row
        for row in sorted(policy_rows, key=_policy_rank, reverse=True)
        if _finite_float(row.get("max_margin_pct_observed"), 9.0) <= 0.50
    ][:8]
    margin30 = [
        row
        for row in sorted(policy_rows, key=_policy_rank, reverse=True)
        if _finite_float(row.get("max_margin_pct_observed"), 9.0) <= 0.30
    ][:8]
    lines = [
        "# Phase1 盘面主线识别回测 2022-2025",
        "",
        "口径：这是回测分析，不改变实盘默认规则。Phase1 只用价格、成交量、持仓、全市场相对强弱、板块共振识别主线状态；Phase2 入场候选保持原样。",
        "",
        "## 数据边界",
        "",
        f"- 候选交易：{diagnostics.get('trade_count', 0)} 笔。",
        f"- 主线状态：{diagnostics.get('state_rows', 0)} 条；主线状态跃迁：{diagnostics.get('events', 0)} 次；板块主线波段：{diagnostics.get('episodes', 0)} 段。",
        f"- 代表机会：{diagnostics.get('representative_trades', diagnostics.get('assigned_trades', 0))} 笔；波段重置口径：连续 {diagnostics.get('reset_low_days', 0)} 天低状态。",
        f"- 参与相对强弱排名的品种：{diagnostics.get('market_symbols', 0)} 个。",
        "- 期限结构：当前连续主力缓存不足以可靠还原跨月价差，因此本版只记录缺失诊断，不计入主线总分。",
        "",
        "## 收益最高参数",
        "",
        "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | 提前退出 | TP2捕获 | 最大占用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_return:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['campaigns']} | {_pct(row['win_rate'])} | {row['early_campaign_exits']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 回撤不超过 30% 的较优参数",
            "",
            "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | 提前退出 | TP2捕获 | 最大占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in controlled:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['campaigns']} | {_pct(row['win_rate'])} | {row['early_campaign_exits']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 保证金占用不超过 50% 的较优参数",
            "",
            "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | TP2捕获 | 平均占用 | 最大占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in margin50:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | "
            f"{_pct(row['max_drawdown_pct'])} | {row['campaigns']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} | {_pct(row['max_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 保证金占用不超过 30% 的较优参数",
            "",
            "| 参数 | 收益 | 年化 | 最大回撤 | 交易数 | 胜率 | TP2捕获 | 平均占用 | 最大占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in margin30:
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row.get('annualized_return_pct'))} | "
            f"{_pct(row['max_drawdown_pct'])} | {row['campaigns']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} | {_pct(row['max_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 选中参数的年份表现",
            "",
            "| 年份 | 交易数 | 收益贡献 | 胜率 | 提前退出 | TP2数 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _year_rows(selected):
        lines.append(
            f"| {row['year']} | {row['campaigns']} | {_pct(row['return_contribution_pct'])} | "
            f"{_pct(row['win_rate'])} | {row['early_campaign_exits']} | {row['tp2_campaigns']} |"
        )
    lines.extend(
        [
            "",
            "## 主线阶段与交易质量",
            "",
            "| 阶段 | 笔数 | 胜率 | 早止损率 | TP2率 | 平均收益 | 主线分 | 领先度 | 资金确认 | 板块共振 | 拥挤度 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stage_rows:
        lines.append(
            f"| {row['transition_stage']} | {row['trades']} | {_pct(row['win_rate'])} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | {_num(row['avg_mainline_score'])} | "
            f"{_num(row['avg_leadership_score'])} | {_num(row['avg_capital_score'])} | "
            f"{_num(row['avg_resonance_score'])} | {_num(row['avg_crowding_score'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前解释",
            "",
            "- low→medium 是主线初现，只适合试错；medium→high 是市场确认，才允许主动上仓位。",
            "- crowded 不是更强的买点，而是主线被市场广泛发现后的降仓状态。",
            "- 本版 Phase1 的弱点在期限结构缺失；如果后续补齐跨月合约或可靠基差历史，需要单独验证它是否能进一步减少假主线。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_mainline_phase1(
    *,
    trade_rows: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    params: MainlineAnalysisParams,
) -> dict[str, Any]:
    state_rows = build_market_mainline_state_rows(
        market_data,
        params=MarketMainlineParams(start_year=params.start_year, end_year=params.end_year),
    )
    events = build_mainline_events(state_rows)
    episodes = build_mainline_episodes(state_rows, reset_low_days=params.reset_low_days)
    assigned = assign_trades_to_mainline_stages(
        trade_rows.to_dict("records"),
        state_rows,
        events,
        transition_window_days=params.transition_window_days,
        episodes=episodes,
    )
    representative = select_representative_episode_trades(assigned) if params.profile.startswith("selective") else assigned
    stage_rows = _stage_summary_rows(representative)
    policy_rows, selected = run_parameter_sweep(representative, state_rows, profile=params.profile)
    return {
        "diagnostics": {
            "trade_count": int(len(trade_rows)),
            "state_rows": int(len(state_rows)),
            "events": int(len(events)),
            "episodes": int(len(episodes)),
            "assigned_trades": int(len(assigned)),
            "representative_trades": int(len(representative)),
            "market_symbols": int(len(market_data)),
            "profile": params.profile,
            "reset_low_days": int(params.reset_low_days),
        },
        "states": state_rows,
        "events": events,
        "episodes": episodes,
        "assigned_trades": assigned,
        "representative_trades": representative,
        "stage_summary": stage_rows,
        "policy_rows": policy_rows,
        "selected": selected,
    }


def write_outputs(report: dict[str, Any], output_prefix: Path) -> MainlineOutputPaths:
    prefix = Path(output_prefix)
    selected: CampaignBacktestResult = report["selected"]
    paths = MainlineOutputPaths(
        states_csv=prefix.with_name(f"{prefix.name}_states.csv"),
        events_csv=prefix.with_name(f"{prefix.name}_events.csv"),
        episodes_csv=prefix.with_name(f"{prefix.name}_episodes.csv"),
        trades_csv=prefix.with_name(f"{prefix.name}_trades.csv"),
        representative_trades_csv=prefix.with_name(f"{prefix.name}_representative_trades.csv"),
        stage_summary_csv=prefix.with_name(f"{prefix.name}_stage_summary.csv"),
        policy_csv=prefix.with_name(f"{prefix.name}_policy.csv"),
        campaign_csv=prefix.with_name(f"{prefix.name}_campaigns.csv"),
        campaign_event_csv=prefix.with_name(f"{prefix.name}_campaign_events.csv"),
        equity_csv=prefix.with_name(f"{prefix.name}_equity.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.states_csv, report["states"])
    _write_csv(paths.events_csv, report["events"])
    _write_csv(paths.episodes_csv, report["episodes"])
    _write_csv(paths.trades_csv, report["assigned_trades"])
    _write_csv(paths.representative_trades_csv, report["representative_trades"])
    _write_csv(paths.stage_summary_csv, report["stage_summary"])
    _write_csv(paths.policy_csv, report["policy_rows"])
    _write_csv(paths.campaign_csv, selected.campaigns)
    _write_csv(paths.campaign_event_csv, selected.events)
    _write_csv(paths.equity_csv, selected.equity_curve)
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(
        json.dumps(
            {
                "diagnostics": report["diagnostics"],
                "selected_summary": selected.summary,
                "policy_rows": report["policy_rows"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    paths.report_md.write_text(
        render_markdown(
            diagnostics=report["diagnostics"],
            stage_rows=report["stage_summary"],
            policy_rows=report["policy_rows"],
            selected=selected,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a market-mainline Phase1 detector with unchanged Phase2 entries")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_PHASE23_CACHE_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--transition-window-days", type=int, default=20)
    parser.add_argument("--reset-low-days", type=int, default=5)
    parser.add_argument(
        "--profile",
        choices=("baseline", "aggressive", "selective", "selective_constrained"),
        default="aggressive",
    )
    args = parser.parse_args()

    years = [int(year) for year in args.years]
    params = MainlineAnalysisParams(
        transition_window_days=int(args.transition_window_days),
        reset_low_days=int(args.reset_low_days),
        start_year=min(years),
        end_year=max(years),
        profile=str(args.profile),
    )
    trade_rows = collect_trend_opportunity_rows(
        years=years,
        phase23_cache_dir=args.phase23_cache_dir,
        market_cache_dir=args.market_cache_dir,
    )
    trade_symbols = set(trade_rows["symbol"].dropna().astype(str).str.upper().tolist()) if len(trade_rows) else set()
    market_symbols = set(discover_cached_symbols(args.market_cache_dir)) | trade_symbols
    market_data = load_market_data(market_symbols, args.market_cache_dir)
    report = analyze_mainline_phase1(trade_rows=trade_rows, market_data=market_data, params=params)
    paths = write_outputs(report, args.output_prefix)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
