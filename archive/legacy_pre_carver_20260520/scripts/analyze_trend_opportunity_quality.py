from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_split_execution_factors import (
    _as_bool,
    _candidate_from_row,
    _finite_float,
    _hold_days,
    _market_cache_path,
    _read_market_frame,
    _trade_row_from_cache_record,
)
from backtest.account_runner import AccountBacktestConfig, AccountCandidate, run_account_backtest
from backtest.execution_quality import visible_daily_before_entry
from backtest.trend_opportunity_quality import (
    trend_budget_margin_pct,
    trend_opportunity_bucket,
    trend_opportunity_quality_from_daily,
)


DEFAULT_PHASE23_CACHE_DIR = Path("data/cache/backtest/phase23_task4_split_v1")
DEFAULT_MARKET_CACHE_DIR = Path("data/cache/backtest")
DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/trend_opportunity_quality_2022_2025")

ACCOUNT_POLICIES = (
    "phase2_budget_fixed_full",
    "uniform_budget_fixed_full",
    "trend_quality_budget_fixed_full",
    "trend_quality_budget_trigger_release",
    "trend_quality_budget_trigger_release_reserved",
    "realistic_gate40_trigger_release_reserved",
)
GROUP_FIELDS = (
    "year",
    "entry_rr_bucket",
    "phase2_score_bucket",
    "medium_quality_bucket",
    "trend_opportunity_bucket",
    "trend_structure_bucket",
    "entry_adverse_bucket",
    "entry_signal_type",
)


@dataclass(frozen=True)
class TrendOpportunityOutputPaths:
    details_csv: Path
    factor_summary_csv: Path
    account_policy_csv: Path
    summary_json: Path
    report_md: Path


def phase2_score_bucket(score: Any) -> str:
    value = abs(_finite_float(score))
    if not math.isfinite(value):
        return "missing"
    if value < 40.0:
        return "lt_40"
    if value < 50.0:
        return "40_50"
    if value < 60.0:
        return "50_60"
    return "60_plus"


def phase2_score_for_target_margin_pct(target_margin_pct: float) -> float:
    pct = max(0.03, min(0.30, float(target_margin_pct)))
    return round(25.0 + pct / 0.30 * 35.0, 4)


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


def collect_trend_opportunity_rows(
    *,
    years: Iterable[int],
    phase23_cache_dir: Path = DEFAULT_PHASE23_CACHE_DIR,
    market_cache_dir: Path = DEFAULT_MARKET_CACHE_DIR,
) -> pd.DataFrame:
    wanted_years = {int(year) for year in years}
    market_cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(phase23_cache_dir).glob("*_phase23_cache.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbol = str(payload.get("symbol") or "").upper()
        daily_df = _read_market_frame(symbol, "daily", Path(market_cache_dir), market_cache)
        minute_df = _read_market_frame(symbol, "minute", Path(market_cache_dir), market_cache)
        for trade in ((payload.get("result") or {}).get("trades") or []):
            entry_time = pd.Timestamp(trade.get("entry_time"))
            if wanted_years and int(entry_time.year) not in wanted_years:
                continue
            row = _trade_row_from_cache_record(trade, daily_df=daily_df, minute_df=minute_df)
            quality = trend_opportunity_quality_from_daily(
                visible_daily_before_entry(daily_df, str(trade.get("entry_time") or "")),
                direction=str(trade.get("direction") or ""),
            )
            row.update(quality)
            row["phase2_score_bucket"] = phase2_score_bucket(row.get("phase2_score"))
            row["trend_opportunity_bucket"] = trend_opportunity_bucket(row.get("trend_opportunity_quality_score"))
            row["trend_budget_margin_pct"] = trend_budget_margin_pct(row.get("trend_opportunity_quality_score"))
            rows.append(row)
    return pd.DataFrame(rows)


def _candidate_for_policy(row: dict[str, Any], policy: str) -> AccountCandidate:
    candidate = _candidate_from_row(row)
    if policy == "phase2_budget_fixed_full":
        return candidate
    if policy == "uniform_budget_fixed_full":
        return replace(candidate, phase2_score=phase2_score_for_target_margin_pct(0.30))
    quality_score = _finite_float(row.get("trend_opportunity_quality_score"), 0.0)
    budget_pct = trend_budget_margin_pct(quality_score)
    return replace(
        candidate,
        phase1_story_budget_margin_pct=budget_pct,
        phase1_story_budget_source="trend_story",
        medium_term_quality_score=quality_score,
        medium_term_entry_location_score=_finite_float(row.get("trend_structure_health_score"), 0.0),
    )


def _avg_margin_pct(equity_curve: list[dict[str, Any]]) -> float:
    values = [
        _finite_float(row.get("active_margin"), 0.0) / _finite_float(row.get("equity"), 1.0)
        for row in equity_curve
        if _finite_float(row.get("equity"), 0.0) > 0
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _account_early_stop_count(trades: list[dict[str, Any]]) -> int:
    return sum(1 for trade in trades if str(trade.get("actual_exit_reason")) == "stop" and _hold_days(trade) < 3.0)


def _config_for_policy(policy: str, *, max_portfolio_margin_pct: float) -> AccountBacktestConfig:
    trend_policies = {
        "trend_quality_budget_fixed_full",
        "trend_quality_budget_trigger_release",
        "trend_quality_budget_trigger_release_reserved",
        "realistic_gate40_trigger_release_reserved",
    }
    if policy not in trend_policies:
        return AccountBacktestConfig(
            max_portfolio_margin_pct=float(max_portfolio_margin_pct),
            risk_per_trade_pct=0.015,
            split_initial_fraction=1.0,
            split_second_entry_trigger="none",
            commission_multiplier=1.01,
            scope=f"trend opportunity budget validation: {policy}",
        )
    if policy == "trend_quality_budget_fixed_full":
        return AccountBacktestConfig(
            max_portfolio_margin_pct=float(max_portfolio_margin_pct),
            position_budget_source="phase1_story",
            risk_per_trade_pct=0.015,
            split_initial_fraction=1.0,
            split_second_entry_trigger="none",
            commission_multiplier=1.01,
            scope=f"trend opportunity budget validation: {policy}",
        )
    reserve_second_margin = policy in {
        "trend_quality_budget_trigger_release_reserved",
        "realistic_gate40_trigger_release_reserved",
    }
    min_phase2_abs_score = 40.0 if policy == "realistic_gate40_trigger_release_reserved" else 0.0
    return AccountBacktestConfig(
        max_portfolio_margin_pct=float(max_portfolio_margin_pct),
        position_budget_source="phase1_story",
        reserve_planned_second_entry_margin=reserve_second_margin,
        min_phase1_story_budget_margin_pct=0.15 if policy == "realistic_gate40_trigger_release_reserved" else 0.0,
        min_phase2_abs_score=min_phase2_abs_score,
        risk_per_trade_pct=0.015,
        execution_policy="conditional",
        conditional_direct_quality_min=65.0,
        conditional_direct_entry_rr_min=0.0,
        conditional_direct_admission_rr_min=0.0,
        conditional_direct_entry_adverse_r_max=0.10,
        conditional_scale_quality_min=45.0,
        conditional_scale_entry_rr_min=0.0,
        conditional_scale_admission_rr_min=0.0,
        conditional_scale_entry_adverse_r_max=0.50,
        conditional_scale_initial_fraction=0.70,
        conditional_trial_quality_min=0.0,
        conditional_trial_admission_rr_min=0.0,
        conditional_trial_entry_adverse_r_max=1.00,
        conditional_trial_initial_fraction=0.30,
        commission_multiplier=1.01,
        scope=f"trend opportunity budget validation: {policy}",
    )


def _account_policy_row(
    *,
    rows: pd.DataFrame,
    group_field: str,
    group_value: Any,
    policy: str,
    max_portfolio_margin_pct: float,
) -> dict[str, Any]:
    source = rows[rows[group_field].astype(str) == str(group_value)].copy() if group_field in rows else rows.copy()
    candidates = [_candidate_for_policy(row, policy) for row in source.to_dict("records")]
    result = run_account_backtest(
        candidates,
        _config_for_policy(policy, max_portfolio_margin_pct=max_portfolio_margin_pct),
    )
    candidate_tp2 = int((source["exit_reason"].astype(str) == "tp2").sum()) if len(source) else 0
    account_tp2 = sum(1 for trade in result.trades if str(trade.get("actual_exit_reason")) == "tp2")
    return {
        "group_field": group_field,
        "group_value": str(group_value),
        "policy": policy,
        "max_portfolio_margin_pct": float(max_portfolio_margin_pct),
        "candidate_trades": int(len(source)),
        "accepted_trades": int(result.summary.get("accepted_trades", 0)),
        "return_pct": float(result.summary.get("return_pct", 0.0)),
        "net_profit": float(result.summary.get("net_profit", 0.0)),
        "max_drawdown_pct": float(result.summary.get("max_drawdown_realized_pct", 0.0)),
        "early_stop_trades": int(_account_early_stop_count(result.trades)),
        "tp2_trades": int(account_tp2),
        "tp2_capture_rate": float(account_tp2 / candidate_tp2) if candidate_tp2 else 0.0,
        "max_margin_pct_observed": float(result.summary.get("max_margin_pct_observed", 0.0)),
        "avg_margin_pct_observed": _avg_margin_pct(result.equity_curve),
        "total_fees": float(result.summary.get("total_fees", 0.0)),
        "split_target_lots": int(result.summary.get("split_target_lots", 0)),
        "split_first_entry_lots": int(result.summary.get("split_first_entry_lots", 0)),
        "split_second_entry_filled_lots": int(result.summary.get("split_second_entry_filled_lots", 0)),
        "split_second_entry_unfilled_lots": int(result.summary.get("split_second_entry_unfilled_lots", 0)),
    }


def _ordered_group_values(rows: pd.DataFrame, field: str) -> list[Any]:
    if field == "phase2_score_bucket":
        order = ("lt_40", "40_50", "50_60", "60_plus")
        return [value for value in order if value in set(rows[field].astype(str))]
    if field == "trend_opportunity_bucket":
        order = ("missing", "low", "medium", "high")
        return [value for value in order if value in set(rows[field].astype(str))]
    if field == "trend_structure_bucket":
        order = ("missing", "damaged", "overextended", "unclear", "healthy_continuation")
        return [value for value in order if value in set(rows[field].astype(str))]
    if field == "entry_adverse_bucket":
        order = ("missing", "no_adverse", "mild_adverse", "stretched", "extreme")
        return [value for value in order if value in set(rows[field].astype(str))]
    return sorted(rows[field].dropna().astype(str).unique().tolist())


def build_account_policy_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for max_margin in (0.30, 0.50):
        all_fixed_net = 0.0
        for policy in ACCOUNT_POLICIES:
            row = _account_policy_row(
                rows=rows,
                group_field="all",
                group_value="all",
                policy=policy,
                max_portfolio_margin_pct=max_margin,
            )
            if policy == "phase2_budget_fixed_full":
                all_fixed_net = float(row["net_profit"])
            row["net_profit_delta_vs_phase2"] = float(row["net_profit"] - all_fixed_net)
            out.append(row)
        for field in ("year", "trend_opportunity_bucket", "trend_structure_bucket", "phase2_score_bucket"):
            if field not in rows:
                continue
            for value in _ordered_group_values(rows, field):
                fixed_net = 0.0
                for policy in ACCOUNT_POLICIES:
                    row = _account_policy_row(
                        rows=rows,
                        group_field=field,
                        group_value=value,
                        policy=policy,
                        max_portfolio_margin_pct=max_margin,
                    )
                    if policy == "phase2_budget_fixed_full":
                        fixed_net = float(row["net_profit"])
                    row["net_profit_delta_vs_phase2"] = float(row["net_profit"] - fixed_net)
                    out.append(row)
    return out


def _mean(frame: pd.DataFrame, field: str) -> float:
    if field not in frame:
        return math.nan
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def _mean_abs(frame: pd.DataFrame, field: str) -> float:
    if field not in frame:
        return math.nan
    values = pd.to_numeric(frame[field], errors="coerce").dropna().abs()
    return float(values.mean()) if not values.empty else math.nan


def _rate(frame: pd.DataFrame, field: str) -> float:
    if field not in frame or len(frame) == 0:
        return 0.0
    return float(frame[field].astype(bool).mean())


def _factor_summary_row(frame: pd.DataFrame, *, group_field: str, group_value: Any) -> dict[str, Any]:
    return {
        "group_field": group_field,
        "group_value": str(group_value),
        "trades": int(len(frame)),
        "early_stop_rate": _rate(frame, "early_stop"),
        "tp1_hit_rate": _rate(frame, "tp1_hit"),
        "tp2_rate": _rate(frame, "tp2_winner"),
        "avg_pnl_ratio": _mean(frame, "pnl_ratio"),
        "avg_phase2_score": _mean(frame, "phase2_score"),
        "avg_phase2_abs_score": _mean_abs(frame, "phase2_score"),
        "avg_old_medium_quality_score": _mean(frame, "medium_term_quality_score"),
        "avg_trend_quality_score": _mean(frame, "trend_opportunity_quality_score"),
        "avg_direction_stability_score": _mean(frame, "trend_direction_stability_score"),
        "avg_structure_health_score": _mean(frame, "trend_structure_health_score"),
        "avg_remaining_space_score": _mean(frame, "trend_remaining_space_score"),
        "avg_volatility_fit_score": _mean(frame, "trend_volatility_fit_score"),
        "avg_participation_score": _mean(frame, "trend_participation_score"),
        "avg_entry_adverse_deviation_r": _mean(frame, "entry_adverse_deviation_r"),
        "avg_mae_3d_r": _mean(frame, "mae_3d_r"),
        "avg_mfe_3d_r": _mean(frame, "mfe_3d_r"),
    }


def build_factor_summary_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = [_factor_summary_row(rows, group_field="all", group_value="all")]
    for field in GROUP_FIELDS:
        if field not in rows:
            continue
        for value in _ordered_group_values(rows, field):
            group = rows[rows[field].astype(str) == str(value)].copy()
            out.append(_factor_summary_row(group, group_field=field, group_value=value))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _rows_where(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out = rows
    for key, value in filters.items():
        out = [row for row in out if str(row.get(key)) == str(value)]
    return out


def _first_row(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    matches = _rows_where(rows, **filters)
    return matches[0] if matches else {}


def render_markdown(
    *,
    rows: pd.DataFrame,
    factor_rows: list[dict[str, Any]],
    account_rows: list[dict[str, Any]],
) -> str:
    all_summary = _factor_summary_row(rows, group_field="all", group_value="all")
    all_account = _rows_where(account_rows, group_field="all", group_value="all")
    phase2_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="phase2_budget_fixed_full",
    )
    uniform_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="uniform_budget_fixed_full",
    )
    quality_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="trend_quality_budget_fixed_full",
    )
    trigger_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="trend_quality_budget_trigger_release",
    )
    reserved_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="trend_quality_budget_trigger_release_reserved",
    )
    realistic_30 = _first_row(
        all_account,
        max_portfolio_margin_pct=0.3,
        policy="realistic_gate40_trigger_release_reserved",
    )
    quality_factors = _rows_where(factor_rows, group_field="trend_opportunity_bucket")
    structure_factors = _rows_where(factor_rows, group_field="trend_structure_bucket")
    phase2_factors = _rows_where(factor_rows, group_field="phase2_score_bucket")
    adverse_factors = _rows_where(factor_rows, group_field="entry_adverse_bucket")
    year_account = _rows_where(account_rows, group_field="year", max_portfolio_margin_pct=0.3)
    lines = [
        "# 中期趋势预算层验证 2022-2025",
        "",
        "口径：固定现有 Phase23 趋势入场样本，不改变默认交易规则。Phase2 在本报告中视为短线量价触发分；新增中期趋势机会质量只用于分析候选预算层。手续费按交易所口径乘以 1.01。",
        "",
        "## 总览",
        "",
        f"- 候选交易：{len(rows)} 笔。",
        f"- TP1 命中率：{_pct(all_summary['tp1_hit_rate'])}。",
        f"- TP2 捕获率：{_pct(all_summary['tp2_rate'])}。",
        f"- 早止损率：{_pct(all_summary['early_stop_rate'])}。",
        f"- 平均 Phase2 短线触发强度：{_num(all_summary['avg_phase2_abs_score'], 2)}。",
        f"- 平均旧中期候选分：{_num(all_summary['avg_old_medium_quality_score'], 2)}。",
        f"- 平均新中期趋势机会分：{_num(all_summary['avg_trend_quality_score'], 2)}。",
        "",
        "## 当前结论",
        "",
        f"- Phase2 作为预算层明显偏弱：30% 保证金口径下，统一预算比 Phase2 预算多 {_money(_finite_float(uniform_30.get('net_profit'), 0.0) - _finite_float(phase2_30.get('net_profit'), 0.0))} 净利润，且最大回撤更低。",
        f"- 单独使用第一版中期趋势预算，比 Phase2 预算多 {_money(_finite_float(quality_30.get('net_profit'), 0.0) - _finite_float(phase2_30.get('net_profit'), 0.0))} 净利润，但最大回撤从 {_pct(phase2_30.get('max_drawdown_pct'))} 升到 {_pct(quality_30.get('max_drawdown_pct'))}；它还不是足够稳定的满仓排序器。",
        f"- 不预留二笔预算的触发释放会把首笔省下的保证金拿去接更多票，容易失真；预留二笔预算后，30%口径收益 {_pct(reserved_30.get('return_pct'))}、最大回撤 {_pct(reserved_30.get('max_drawdown_pct'))}、接受交易 {_num(reserved_30.get('accepted_trades'), 0)} 笔。",
        f"- 更接近真实交易的 `realistic_gate40_trigger_release_reserved` 不追求打满保证金：平均资金占用 {_pct(realistic_30.get('avg_margin_pct_observed'))}，最大回撤降到 {_pct(realistic_30.get('max_drawdown_pct'))}，但 TP2 捕获只有 {_pct(realistic_30.get('tp2_capture_rate'))}，说明单纯用 Phase2>=40 过滤会错过不少趋势。",
        "- 但第一版中期趋势分桶本身还没有明显区分收益结构：medium 和 high 的早止损、TP2、平均收益非常接近，说明评分卡目前更多是在确认“这些样本大多已经有趋势背景”，还没有足够强的预算排序能力。",
        "- 触发质量仍然是最清楚的执行层信息：真实触发偏离越大，早止损显著升高，TP1/TP2 显著下降。",
        "",
        "## 账户级预算口径对照",
        "",
        "| 保证金上限 | 预算/执行口径 | 收益 | 最大回撤 | 净利润 | 相对Phase2口径 | 早止损 | TP2捕获 | 最大资金占用 | 二笔成交手数 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(all_account, key=lambda item: (float(item["max_portfolio_margin_pct"]), str(item["policy"]))):
        lines.append(
            f"| {_pct(row['max_portfolio_margin_pct'])} | {row['policy']} | {_pct(row['return_pct'])} | "
            f"{_pct(row['max_drawdown_pct'])} | {_money(row['net_profit'])} | "
            f"{_money(row['net_profit_delta_vs_phase2'])} | {row['early_stop_trades']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} | "
            f"{row['split_second_entry_filled_lots']} |"
        )
    lines.extend(
        [
            "",
            "## 中期趋势机会分桶",
            "",
            "| 趋势机会桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 趋势分 | 结构分 | 3日MAE | 3日MFE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_trend_quality_score'], 2)} | {_num(row['avg_structure_health_score'], 2)} | "
            f"{_num(row['avg_mae_3d_r'], 3)} | {_num(row['avg_mfe_3d_r'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 趋势结构健康度",
            "",
            "| 结构桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 方向稳定 | 结构健康 | 剩余空间 | 波动适配 | 参与度 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in structure_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_direction_stability_score'], 2)} | {_num(row['avg_structure_health_score'], 2)} | "
            f"{_num(row['avg_remaining_space_score'], 2)} | {_num(row['avg_volatility_fit_score'], 2)} | "
            f"{_num(row['avg_participation_score'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Phase2 短线触发分对照",
            "",
            "| Phase2桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 新趋势分 | 结构分 | 入场偏离R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phase2_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_trend_quality_score'], 2)} | {_num(row['avg_structure_health_score'], 2)} | "
            f"{_num(row['avg_entry_adverse_deviation_r'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 触发质量对照",
            "",
            "| 触发偏离桶 | 笔数 | 早止损率 | TP1率 | TP2率 | 平均收益率 | 趋势分 | 结构分 | 3日MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adverse_factors:
        lines.append(
            f"| {row['group_value']} | {row['trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['tp2_rate'])} | {_pct(row['avg_pnl_ratio'])} | "
            f"{_num(row['avg_trend_quality_score'], 2)} | {_num(row['avg_structure_health_score'], 2)} | "
            f"{_num(row['avg_mae_3d_r'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 分年份账户对照",
            "",
            "| 年份 | 预算/执行口径 | 收益 | 最大回撤 | 净利润 | 相对Phase2口径 | 早止损 | TP2捕获 | 最大资金占用 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(year_account, key=lambda item: (str(item["group_value"]), str(item["policy"]))):
        lines.append(
            f"| {row['group_value']} | {row['policy']} | {_pct(row['return_pct'])} | "
            f"{_pct(row['max_drawdown_pct'])} | {_money(row['net_profit'])} | "
            f"{_money(row['net_profit_delta_vs_phase2'])} | {row['early_stop_trades']} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['max_margin_pct_observed'])} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 本报告只验证预算层语义：中期趋势质量是否比 Phase2 更适合决定理论资金暴露。",
            "- 新分数只用入场前日线数据；入场后路径仍只用于验证结果，不参与定义。",
            "- `trend_quality_budget_trigger_release` 是候选执行参数：中期趋势决定预算，真实触发偏离决定首笔释放比例；它不是默认实盘规则。",
            "- `trend_quality_budget_trigger_release_reserved` 会把计划二笔加仓预算视为已占用策略预算，避免首笔小仓后把释放出的额度拿去买更多票。",
            "- `realistic_gate40_trigger_release_reserved` 额外要求中期趋势预算至少 15%、Phase2 触发强度至少 40；这是粗过滤验证，不是最终入场规则。",
            "- 旧中期候选分仍列作对照，不直接等同于这次的新趋势机会评分卡。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_trend_opportunity_quality(rows: pd.DataFrame) -> dict[str, Any]:
    factor_rows = build_factor_summary_rows(rows)
    account_rows = build_account_policy_rows(rows)
    return {
        "diagnostics": {
            "trade_count": int(len(rows)),
            "years": {str(year): int(count) for year, count in rows.groupby("year").size().items()},
            "phase2_score_buckets": {
                str(bucket): int(count) for bucket, count in rows.groupby("phase2_score_bucket").size().items()
            },
            "trend_opportunity_buckets": {
                str(bucket): int(count) for bucket, count in rows.groupby("trend_opportunity_bucket").size().items()
            },
            "trend_structure_buckets": {
                str(bucket): int(count) for bucket, count in rows.groupby("trend_structure_bucket").size().items()
            },
        },
        "factor_summary": factor_rows,
        "account_policy": account_rows,
    }


def write_outputs(rows: pd.DataFrame, report: dict[str, Any], output_prefix: Path) -> TrendOpportunityOutputPaths:
    prefix = Path(output_prefix)
    paths = TrendOpportunityOutputPaths(
        details_csv=prefix.with_name(f"{prefix.name}_details.csv"),
        factor_summary_csv=prefix.with_name(f"{prefix.name}_factor_summary.csv"),
        account_policy_csv=prefix.with_name(f"{prefix.name}_account_policy.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.details_csv, rows)
    _write_csv(paths.factor_summary_csv, report["factor_summary"])
    _write_csv(paths.account_policy_csv, report["account_policy"])
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    paths.report_md.write_text(
        render_markdown(rows=rows, factor_rows=report["factor_summary"], account_rows=report["account_policy"]),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate medium-term trend opportunity quality as budget layer")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_PHASE23_CACHE_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()
    rows = collect_trend_opportunity_rows(
        years=args.years,
        phase23_cache_dir=args.phase23_cache_dir,
        market_cache_dir=args.market_cache_dir,
    )
    report = analyze_trend_opportunity_quality(rows)
    paths = write_outputs(rows, report, args.output_prefix)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
