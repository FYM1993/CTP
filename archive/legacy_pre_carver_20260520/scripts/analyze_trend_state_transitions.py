from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_phase1_phase2_split_grid import _candidate_for_grid
from analyze_split_execution_factors import _finite_float, _hold_days, _read_market_frame
from analyze_trend_opportunity_quality import (
    DEFAULT_MARKET_CACHE_DIR,
    DEFAULT_PHASE23_CACHE_DIR,
    collect_trend_opportunity_rows,
)
from backtest.account_runner import AccountBacktestConfig, AccountCandidate, run_account_backtest
from backtest.trend_opportunity_quality import trend_budget_margin_pct, trend_opportunity_quality_from_daily


DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/trend_state_transitions_2022_2025")


@dataclass(frozen=True)
class TransitionParams:
    transition_window_days: int = 20
    start_year: int = 2022
    end_year: int = 2025


@dataclass(frozen=True)
class TransitionPolicy:
    name: str
    trial_budget_margin_pct: float = 0.08
    confirm_budget_margin_pct: float = 0.30
    include_trial: bool = True
    include_confirm: bool = True
    include_late_high: bool = False
    max_trade_seq_per_event: int = 1
    split_initial_fraction: float = 1.0
    max_portfolio_margin_pct: float = 0.30


@dataclass(frozen=True)
class TransitionOutputPaths:
    states_csv: Path
    events_csv: Path
    trades_csv: Path
    stage_summary_csv: Path
    bucket_summary_csv: Path
    policy_csv: Path
    summary_json: Path
    report_md: Path


def bucket_for_score(score: Any) -> str:
    value = _finite_float(score, math.nan)
    if not math.isfinite(value):
        return "missing"
    if value >= 65.0:
        return "high"
    if value >= 45.0:
        return "medium"
    return "low"


def _date_str(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _state_row(symbol: str, daily_df: pd.DataFrame, idx: int, *, direction: str) -> dict[str, Any]:
    visible = daily_df.iloc[: idx + 1].copy()
    quality = trend_opportunity_quality_from_daily(visible, direction=direction)
    score = _finite_float(quality.get("trend_opportunity_quality_score"), math.nan)
    return {
        "symbol": str(symbol).upper(),
        "direction": direction,
        "date": _date_str(daily_df.iloc[idx]["date"]),
        "score": float(score) if math.isfinite(score) else math.nan,
        "bucket": bucket_for_score(score),
        "structure_bucket": str(quality.get("trend_structure_bucket") or ""),
        "direction_stability_score": _finite_float(quality.get("trend_direction_stability_score"), 0.0),
        "structure_health_score": _finite_float(quality.get("trend_structure_health_score"), 0.0),
        "remaining_space_score": _finite_float(quality.get("trend_remaining_space_score"), 0.0),
        "volatility_fit_score": _finite_float(quality.get("trend_volatility_fit_score"), 0.0),
        "participation_score": _finite_float(quality.get("trend_participation_score"), 0.0),
    }


def build_directional_state_rows(
    *,
    symbols: Iterable[str],
    market_cache_dir: Path = DEFAULT_MARKET_CACHE_DIR,
    start_year: int = 2022,
    end_year: int = 2025,
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for symbol in sorted({str(item).upper() for item in symbols}):
        daily_df = _read_market_frame(symbol, "daily", Path(market_cache_dir), cache)
        if daily_df.empty or "date" not in daily_df:
            continue
        data = daily_df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)
        for idx in range(len(data)):
            year = data.iloc[idx]["date"].date().year
            if year < int(start_year) or year > int(end_year):
                continue
            rows.append(_state_row(symbol, data, idx, direction="long"))
            rows.append(_state_row(symbol, data, idx, direction="short"))
    return rows


def _transition_type(prev_bucket: str, bucket: str) -> str:
    previous = "low" if prev_bucket in {"missing", ""} else str(prev_bucket)
    current = "low" if bucket in {"missing", ""} else str(bucket)
    if previous == current:
        return ""
    return f"{previous}_to_{current}"


def build_transition_events(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in state_rows:
        grouped.setdefault((str(row.get("symbol") or "").upper(), str(row.get("direction") or "")), []).append(row)
    for (symbol, direction), rows in sorted(grouped.items()):
        prev_bucket = ""
        prev_score = math.nan
        for row in sorted(rows, key=lambda item: str(item.get("date") or "")):
            bucket = str(row.get("bucket") or "missing")
            event_type = _transition_type(prev_bucket, bucket) if prev_bucket else ""
            if event_type:
                date_value = str(row.get("date") or "")
                events.append(
                    {
                        "event_id": f"{symbol}_{direction}_{date_value.replace('-', '')}_{event_type}",
                        "symbol": symbol,
                        "direction": direction,
                        "date": date_value,
                        "event_type": event_type,
                        "from_bucket": "low" if prev_bucket == "missing" else prev_bucket,
                        "to_bucket": "low" if bucket == "missing" else bucket,
                        "prev_score": float(prev_score) if math.isfinite(prev_score) else math.nan,
                        "score": _finite_float(row.get("score"), math.nan),
                    }
                )
            prev_bucket = bucket
            prev_score = _finite_float(row.get("score"), math.nan)
    return events


def _latest_row_before(rows: list[dict[str, Any]], entry_date: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if str(row.get("date") or "") <= entry_date]
    return candidates[-1] if candidates else None


def _stage_for_trade(state: dict[str, Any] | None, event: dict[str, Any] | None, days_since: int | None, window_days: int) -> str:
    bucket = str((state or {}).get("bucket") or "")
    event_type = str((event or {}).get("event_type") or "")
    if event_type == "low_to_medium" and days_since is not None and days_since <= int(window_days):
        return "trial_low_to_medium"
    if event_type == "medium_to_high" and days_since is not None and days_since <= int(window_days):
        return "confirm_medium_to_high"
    if bucket == "high":
        return "late_high"
    if bucket == "medium":
        return "medium_continuation"
    if bucket == "low":
        return "low_state"
    return "no_state"


def assign_trades_to_transition_stages(
    trades: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    transition_window_days: int = 20,
) -> list[dict[str, Any]]:
    states_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    events_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in state_rows:
        states_by_key.setdefault((str(row.get("symbol") or "").upper(), str(row.get("direction") or "")), []).append(row)
    for row in events:
        events_by_key.setdefault((str(row.get("symbol") or "").upper(), str(row.get("direction") or "")), []).append(row)
    for values in states_by_key.values():
        values.sort(key=lambda item: str(item.get("date") or ""))
    for values in events_by_key.values():
        values.sort(key=lambda item: str(item.get("date") or ""))

    out: list[dict[str, Any]] = []
    seq_by_stage_event: dict[tuple[str, str], int] = {}
    for trade in sorted(trades, key=lambda item: (str(item.get("entry_time") or ""), str(item.get("symbol") or ""))):
        symbol = str(trade.get("symbol") or "").upper()
        direction = str(trade.get("direction") or "")
        entry_date = _date_str(trade.get("entry_time"))
        key = (symbol, direction)
        state = _latest_row_before(states_by_key.get(key, []), entry_date)
        event = _latest_row_before(events_by_key.get(key, []), entry_date)
        days_since = None
        if event is not None:
            days_since = int((pd.Timestamp(entry_date) - pd.Timestamp(event["date"])).days)
        stage = _stage_for_trade(state, event, days_since, transition_window_days)
        event_id = str((event or {}).get("event_id") or "")
        seq_key = (stage, event_id or f"{symbol}_{direction}_{entry_date}")
        seq_by_stage_event[seq_key] = seq_by_stage_event.get(seq_key, 0) + 1
        row = dict(trade)
        row.update(
            {
                "state_bucket": str((state or {}).get("bucket") or ""),
                "state_score": _finite_float((state or {}).get("score"), math.nan),
                "transition_event_id": event_id,
                "transition_event_type": str((event or {}).get("event_type") or ""),
                "days_since_transition": days_since if days_since is not None else "",
                "transition_stage": stage,
                "transition_trade_seq": seq_by_stage_event[seq_key],
            }
        )
        out.append(row)
    return out


def candidate_for_transition_policy(row: dict[str, Any], policy: TransitionPolicy) -> AccountCandidate:
    candidate = _candidate_for_grid(row)
    stage = str(row.get("transition_stage") or "")
    if stage == "trial_low_to_medium":
        budget = float(policy.trial_budget_margin_pct)
    elif stage == "confirm_medium_to_high":
        budget = float(policy.confirm_budget_margin_pct)
    else:
        budget = trend_budget_margin_pct(_finite_float(row.get("trend_opportunity_quality_score"), 0.0))
    return replace(
        candidate,
        phase1_story_budget_margin_pct=budget,
        phase1_story_budget_source="transition_stage",
    )


def _policy_candidates(rows: list[dict[str, Any]], policy: TransitionPolicy) -> list[AccountCandidate]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        stage = str(row.get("transition_stage") or "")
        seq = int(row.get("transition_trade_seq") or 0)
        if seq > int(policy.max_trade_seq_per_event):
            continue
        if stage == "trial_low_to_medium" and bool(policy.include_trial):
            selected.append(row)
        elif stage == "confirm_medium_to_high" and bool(policy.include_confirm):
            selected.append(row)
        elif stage == "late_high" and bool(policy.include_late_high):
            selected.append(row)
    return [candidate_for_transition_policy(row, policy) for row in selected]


def _config_for_policy(policy: TransitionPolicy) -> AccountBacktestConfig:
    split_fraction = float(policy.split_initial_fraction)
    return AccountBacktestConfig(
        max_portfolio_margin_pct=float(policy.max_portfolio_margin_pct),
        position_budget_source="phase1_story",
        reserve_planned_second_entry_margin=True,
        risk_per_trade_pct=0.015,
        split_initial_fraction=split_fraction,
        split_second_entry_trigger="none" if split_fraction >= 1.0 else "tp1",
        commission_multiplier=1.01,
        scope=f"trend transition policy: {policy.name}",
    )


def iter_transition_policies() -> Iterable[TransitionPolicy]:
    yield TransitionPolicy("confirm30_only", include_trial=False, include_confirm=True, include_late_high=False)
    yield TransitionPolicy("trial8_confirm30", include_trial=True, include_confirm=True, include_late_high=False)
    yield TransitionPolicy("trial8_confirm30_seq2", include_trial=True, include_confirm=True, include_late_high=False, max_trade_seq_per_event=2)
    yield TransitionPolicy("trial8_confirm30_first70", include_trial=True, include_confirm=True, include_late_high=False, split_initial_fraction=0.70)
    yield TransitionPolicy("trial8_confirm30_late_high", include_trial=True, include_confirm=True, include_late_high=True, max_trade_seq_per_event=1)


def _avg_margin_pct(equity_curve: list[dict[str, Any]]) -> float:
    values = [
        _finite_float(row.get("active_margin"), 0.0) / _finite_float(row.get("equity"), 1.0)
        for row in equity_curve
        if _finite_float(row.get("equity"), 0.0) > 0
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _early_stop_count(trades: list[dict[str, Any]]) -> int:
    return sum(1 for trade in trades if str(trade.get("actual_exit_reason")) == "stop" and _hold_days(trade) < 3.0)


def _stage_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage in sorted({str(row.get("transition_stage") or "") for row in rows}):
        stage_rows = [row for row in rows if str(row.get("transition_stage") or "") == stage]
        if not stage_rows:
            continue
        wins = sum(1 for row in stage_rows if _finite_float(row.get("pnl_ratio"), 0.0) > 0)
        early = sum(1 for row in stage_rows if bool(row.get("early_stop")))
        tp2 = sum(1 for row in stage_rows if str(row.get("exit_reason")) == "tp2")
        out.append(
            {
                "transition_stage": stage,
                "trades": len(stage_rows),
                "win_rate": float(wins / len(stage_rows)),
                "early_stop_rate": float(early / len(stage_rows)),
                "tp2_rate": float(tp2 / len(stage_rows)),
                "avg_pnl_ratio": float(sum(_finite_float(row.get("pnl_ratio"), 0.0) for row in stage_rows) / len(stage_rows)),
                "avg_state_score": float(sum(_finite_float(row.get("state_score"), 0.0) for row in stage_rows) / len(stage_rows)),
            }
        )
    return out


def _group_value(row: dict[str, Any], field: str) -> str:
    if field == "year":
        return str(int(pd.Timestamp(row.get("entry_time")).year))
    value = str(row.get(field) or "").strip()
    return value or "missing"


def _stage_group_summary_rows(rows: list[dict[str, Any]], fields: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields:
        values = sorted({_group_value(row, field) for row in rows})
        for value in values:
            group_rows = [row for row in rows if _group_value(row, field) == value]
            for stage_row in _stage_summary_rows(group_rows):
                item = dict(stage_row)
                item["group_field"] = field
                item["group_value"] = value
                out.append(item)
    return out


def _policy_row(
    *,
    rows: list[dict[str, Any]],
    policy: TransitionPolicy,
    group_field: str,
    group_value: Any,
    candidate_tp2_count: int,
) -> dict[str, Any]:
    source = rows
    if group_field == "year":
        source = [row for row in rows if int(pd.Timestamp(row.get("entry_time")).year) == int(group_value)]
    candidates = _policy_candidates(source, policy)
    result = run_account_backtest(candidates, _config_for_policy(policy))
    accepted = int(result.summary.get("accepted_trades", 0))
    early = _early_stop_count(result.trades)
    tp2 = sum(1 for trade in result.trades if str(trade.get("actual_exit_reason")) == "tp2")
    return {
        "policy": policy.name,
        "group_field": group_field,
        "group_value": str(group_value),
        "candidate_trades": len(candidates),
        "accepted_trades": accepted,
        "return_pct": float(result.summary.get("return_pct", 0.0)),
        "net_profit": float(result.summary.get("net_profit", 0.0)),
        "max_drawdown_pct": float(result.summary.get("max_drawdown_realized_pct", 0.0)),
        "early_stop_trades": early,
        "early_stop_rate": float(early / accepted) if accepted else 0.0,
        "tp2_trades": tp2,
        "tp2_capture_rate": float(tp2 / candidate_tp2_count) if candidate_tp2_count else 0.0,
        "avg_margin_pct_observed": _avg_margin_pct(result.equity_curve),
        "max_margin_pct_observed": float(result.summary.get("max_margin_pct_observed", 0.0)),
        "trial_budget_margin_pct": float(policy.trial_budget_margin_pct),
        "confirm_budget_margin_pct": float(policy.confirm_budget_margin_pct),
        "max_trade_seq_per_event": int(policy.max_trade_seq_per_event),
        "split_initial_fraction": float(policy.split_initial_fraction),
    }


def _policy_rows(rows: list[dict[str, Any]], policies: Iterable[TransitionPolicy], years: Iterable[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_tp2 = sum(1 for row in rows if str(row.get("exit_reason")) == "tp2")
    for policy in policies:
        out.append(
            _policy_row(
                rows=rows,
                policy=policy,
                group_field="all",
                group_value="all",
                candidate_tp2_count=all_tp2,
            )
        )
        for year in years:
            year_tp2 = sum(
                1
                for row in rows
                if int(pd.Timestamp(row.get("entry_time")).year) == int(year) and str(row.get("exit_reason")) == "tp2"
            )
            out.append(
                _policy_row(
                    rows=rows,
                    policy=policy,
                    group_field="year",
                    group_value=year,
                    candidate_tp2_count=year_tp2,
                )
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
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


def _money(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:,.0f}"


def render_markdown(
    *,
    assigned_trades: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> str:
    all_policy = [row for row in policy_rows if row.get("group_field") == "all"]
    policy_years = [row for row in policy_rows if row.get("group_field") == "year"]
    stage_years = [row for row in bucket_rows if row.get("group_field") == "year"]

    def append_bucket_table(title: str, group_field: str, min_trades: int = 10) -> None:
        rows = [
            row
            for row in bucket_rows
            if row.get("group_field") == group_field and int(row.get("trades") or 0) >= int(min_trades)
        ]
        if not rows:
            return
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 分桶 | 阶段 | 笔数 | 胜率 | 早止损率 | TP2率 | 平均收益率 | 平均状态分 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda item: (str(item.get("group_value") or ""), str(item.get("transition_stage") or ""))):
            lines.append(
                f"| {row['group_value']} | {row['transition_stage']} | {row['trades']} | {_pct(row['win_rate'])} | "
                f"{_pct(row['early_stop_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
                f"{_num(row['avg_state_score'], 2)} |"
            )

    lines = [
        "# 趋势状态跃迁验证 2022-2025",
        "",
        "口径：只做回测分析，不改实盘默认规则。Phase1 趋势状态按 low/medium/high 分桶；low→medium 视为趋势故事萌芽，medium→high 视为趋势确认，high 持续视为后排趋势状态。交易只使用入场当日前可见的最近状态和最近跃迁归类。",
        "",
        "## 交易阶段分布",
        "",
        "| 阶段 | 笔数 | 胜率 | 早止损率 | TP2率 | 平均收益率 | 平均状态分 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row['transition_stage']} | {row['trades']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['early_stop_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_state_score'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## 账户级策略对照",
            "",
            "| 策略 | 候选 | 接受 | 收益 | 最大回撤 | 早止损率 | TP2捕获 | 平均占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_policy:
        lines.append(
            f"| {row['policy']} | {row['candidate_trades']} | {row['accepted_trades']} | "
            f"{_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 分年份：账户级策略对照",
            "",
            "| 策略 | 年份 | 候选 | 接受 | 收益 | 最大回撤 | 早止损率 | TP2捕获 | 平均占用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(policy_years, key=lambda item: (str(item["policy"]), str(item["group_value"]))):
        lines.append(
            f"| {row['policy']} | {row['group_value']} | {row['candidate_trades']} | {row['accepted_trades']} | "
            f"{_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 分年份：交易阶段结构",
            "",
            "| 年份 | 阶段 | 笔数 | 胜率 | 早止损率 | TP2率 | 平均收益率 | 平均状态分 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(stage_years, key=lambda item: (str(item["group_value"]), str(item["transition_stage"]))):
        lines.append(
            f"| {row['group_value']} | {row['transition_stage']} | {row['trades']} | {_pct(row['win_rate'])} | "
            f"{_pct(row['early_stop_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_state_score'], 2)} |"
        )
    append_bucket_table("分第一 RR 桶：交易阶段结构", "entry_rr_bucket")
    append_bucket_table("分中期质量桶：交易阶段结构", "medium_quality_bucket")
    append_bucket_table("分入场路径桶：交易阶段结构", "entry_path_bucket")
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- low→medium 试错只代表趋势故事初现，不等于已经值得满仓。",
            "- medium→high 确认只代表状态跃迁，不包含后续止盈止损优化。",
            "- high 持续后的后排票单独列出，便于判断是否应该限制追单次数。",
            "- 本报告不改变实盘默认配置。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_transitions(
    *,
    trade_rows: pd.DataFrame,
    state_rows: list[dict[str, Any]],
    params: TransitionParams,
) -> dict[str, Any]:
    events = build_transition_events(state_rows)
    assigned = assign_trades_to_transition_stages(
        trade_rows.to_dict("records"),
        state_rows,
        events,
        transition_window_days=params.transition_window_days,
    )
    years = list(range(int(params.start_year), int(params.end_year) + 1))
    stage_rows = _stage_summary_rows(assigned)
    bucket_rows = _stage_group_summary_rows(
        assigned,
        ["year", "entry_rr_bucket", "medium_quality_bucket", "entry_path_bucket"],
    )
    policy_rows = _policy_rows(assigned, iter_transition_policies(), years)
    return {
        "diagnostics": {
            "trade_count": int(len(trade_rows)),
            "state_rows": int(len(state_rows)),
            "transition_events": int(len(events)),
            "assigned_trades": int(len(assigned)),
        },
        "states": state_rows,
        "events": events,
        "assigned_trades": assigned,
        "stage_summary": stage_rows,
        "bucket_summary": bucket_rows,
        "policy_rows": policy_rows,
    }


def write_outputs(report: dict[str, Any], output_prefix: Path) -> TransitionOutputPaths:
    prefix = Path(output_prefix)
    paths = TransitionOutputPaths(
        states_csv=prefix.with_name(f"{prefix.name}_states.csv"),
        events_csv=prefix.with_name(f"{prefix.name}_events.csv"),
        trades_csv=prefix.with_name(f"{prefix.name}_trades.csv"),
        stage_summary_csv=prefix.with_name(f"{prefix.name}_stage_summary.csv"),
        bucket_summary_csv=prefix.with_name(f"{prefix.name}_bucket_summary.csv"),
        policy_csv=prefix.with_name(f"{prefix.name}_policy.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.states_csv, report["states"])
    _write_csv(paths.events_csv, report["events"])
    _write_csv(paths.trades_csv, report["assigned_trades"])
    _write_csv(paths.stage_summary_csv, report["stage_summary"])
    _write_csv(paths.bucket_summary_csv, report["bucket_summary"])
    _write_csv(paths.policy_csv, report["policy_rows"])
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    paths.report_md.write_text(
        render_markdown(
            assigned_trades=report["assigned_trades"],
            stage_rows=report["stage_summary"],
            bucket_rows=report["bucket_summary"],
            policy_rows=report["policy_rows"],
        ),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase1 trend state transitions")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_PHASE23_CACHE_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--transition-window-days", type=int, default=20)
    args = parser.parse_args()
    params = TransitionParams(
        transition_window_days=int(args.transition_window_days),
        start_year=min(args.years),
        end_year=max(args.years),
    )
    trade_rows = collect_trend_opportunity_rows(
        years=args.years,
        phase23_cache_dir=args.phase23_cache_dir,
        market_cache_dir=args.market_cache_dir,
    )
    symbols = sorted(trade_rows["symbol"].dropna().astype(str).str.upper().unique().tolist())
    state_rows = build_directional_state_rows(
        symbols=symbols,
        market_cache_dir=args.market_cache_dir,
        start_year=params.start_year,
        end_year=params.end_year,
    )
    report = analyze_transitions(trade_rows=trade_rows, state_rows=state_rows, params=params)
    paths = write_outputs(report, args.output_prefix)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
