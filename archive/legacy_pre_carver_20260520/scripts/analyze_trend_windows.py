from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_phase1_phase2_split_grid import _candidate_for_grid
from analyze_split_execution_factors import _finite_float, _read_market_frame, _hold_days
from analyze_trend_opportunity_quality import (
    DEFAULT_MARKET_CACHE_DIR,
    DEFAULT_PHASE23_CACHE_DIR,
    collect_trend_opportunity_rows,
)
from backtest.account_runner import AccountBacktestConfig, run_account_backtest
from backtest.trend_opportunity_quality import trend_opportunity_quality_from_daily


DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/trend_window_activation_2022_2025")


@dataclass(frozen=True)
class WindowParams:
    min_quality_score: float = 65.0
    min_consecutive_high_days: int = 5
    end_grace_days: int = 5
    start_year: int = 2022
    end_year: int = 2025


@dataclass(frozen=True)
class WindowPolicy:
    max_portfolio_margin_pct: float
    max_trades_per_window: int
    split_initial_fraction: float


@dataclass(frozen=True)
class WindowOutputPaths:
    windows_csv: Path
    trades_csv: Path
    policy_csv: Path
    conviction_csv: Path
    yearly_csv: Path
    summary_json: Path
    report_md: Path


def _policy_label(policy: WindowPolicy) -> str:
    margin = int(round(float(policy.max_portfolio_margin_pct) * 100))
    first = int(round(float(policy.split_initial_fraction) * 100))
    return f"m{margin}_window{int(policy.max_trades_per_window)}_first{first}"


def iter_window_policies() -> Iterable[WindowPolicy]:
    for max_margin in (0.30, 0.50):
        for max_trades in (1, 2):
            for first_fraction in (1.0, 0.70, 0.50):
                yield WindowPolicy(max_margin, max_trades, first_fraction)


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _date_str(value: Any) -> str:
    return _as_date(value).isoformat()


def _daily_dates(start: str, end: str) -> set[str]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        return set()
    return {item.date().isoformat() for item in pd.date_range(start_ts.date(), end_ts.date(), freq="D")}


def _state_row(symbol: str, daily_df: pd.DataFrame, idx: int, *, min_quality_score: float) -> dict[str, Any]:
    visible = daily_df.iloc[: idx + 1].copy()
    long_q = trend_opportunity_quality_from_daily(visible, direction="long")
    short_q = trend_opportunity_quality_from_daily(visible, direction="short")
    long_score = _finite_float(long_q.get("trend_opportunity_quality_score"), -math.inf)
    short_score = _finite_float(short_q.get("trend_opportunity_quality_score"), -math.inf)
    if short_score > long_score:
        direction = "short"
        quality = short_q
        score = short_score
    else:
        direction = "long"
        quality = long_q
        score = long_score
    structure = str(quality.get("trend_structure_bucket") or "")
    return {
        "symbol": str(symbol).upper(),
        "date": _date_str(daily_df.iloc[idx]["date"]),
        "direction": direction,
        "score": float(score) if math.isfinite(score) else math.nan,
        "is_high": bool(math.isfinite(score) and score >= float(min_quality_score) and structure == "healthy_continuation"),
        "structure_bucket": structure,
        "direction_stability_score": _finite_float(quality.get("trend_direction_stability_score"), 0.0),
        "structure_health_score": _finite_float(quality.get("trend_structure_health_score"), 0.0),
        "remaining_space_score": _finite_float(quality.get("trend_remaining_space_score"), 0.0),
        "volatility_fit_score": _finite_float(quality.get("trend_volatility_fit_score"), 0.0),
        "participation_score": _finite_float(quality.get("trend_participation_score"), 0.0),
    }


def build_trend_state_rows(
    *,
    symbols: Iterable[str],
    market_cache_dir: Path = DEFAULT_MARKET_CACHE_DIR,
    params: WindowParams = WindowParams(),
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
            row_date = data.iloc[idx]["date"].date()
            if row_date.year < int(params.start_year) or row_date.year > int(params.end_year):
                continue
            rows.append(_state_row(symbol, data, idx, min_quality_score=params.min_quality_score))
    return rows


def _finalize_window(
    *,
    symbol: str,
    direction: str,
    signal_start_date: str,
    active_from_date: str,
    end_date: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    high_rows = [row for row in rows if bool(row.get("is_high"))]
    signal_rows = [
        row
        for row in high_rows
        if str(signal_start_date) <= str(row.get("date") or "") <= str(active_from_date)
    ]
    activation_rows = [row for row in high_rows if str(row.get("date") or "") == str(active_from_date)]
    dates = _daily_dates(signal_start_date, end_date)
    scores = [_finite_float(row.get("score"), math.nan) for row in high_rows]
    signal_scores = [_finite_float(row.get("score"), math.nan) for row in signal_rows]
    activation_score = _finite_float(activation_rows[-1].get("score"), 0.0) if activation_rows else 0.0
    window_id = f"{symbol}_{direction}_{active_from_date.replace('-', '')}"
    return {
        "window_id": window_id,
        "symbol": symbol,
        "direction": direction,
        "signal_start_date": signal_start_date,
        "active_from_date": active_from_date,
        "end_date": end_date,
        "calendar_days": len(dates),
        "high_days": len(high_rows),
        "signal_high_days": len(signal_rows),
        "activation_score": float(activation_score),
        "signal_avg_score": float(sum(signal_scores) / len(signal_scores)) if signal_scores else 0.0,
        "signal_min_score": float(min(signal_scores)) if signal_scores else 0.0,
        "avg_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "max_score": float(max(scores)) if scores else 0.0,
    }


def identify_trend_windows(
    state_rows: list[dict[str, Any]],
    *,
    min_consecutive_high_days: int = 5,
    end_grace_days: int = 5,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in state_rows:
        by_symbol.setdefault(str(row.get("symbol") or "").upper(), []).append(row)

    for symbol, rows in sorted(by_symbol.items()):
        streak_direction = ""
        streak_start = ""
        streak_rows: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        active_rows: list[dict[str, Any]] = []
        last_high_date = ""
        misses = 0

        for row in sorted(rows, key=lambda item: str(item.get("date") or "")):
            row_date = str(row.get("date") or "")
            direction = str(row.get("direction") or "")
            is_high = bool(row.get("is_high")) and direction in {"long", "short"}

            if active is not None:
                if is_high and direction == str(active["direction"]):
                    active_rows.append(row)
                    last_high_date = row_date
                    misses = 0
                    continue
                if is_high and direction != str(active["direction"]):
                    windows.append(
                        _finalize_window(
                            symbol=symbol,
                            direction=str(active["direction"]),
                            signal_start_date=str(active["signal_start_date"]),
                            active_from_date=str(active["active_from_date"]),
                            end_date=last_high_date,
                            rows=active_rows,
                        )
                    )
                    active = None
                    active_rows = []
                    misses = 0
                else:
                    misses += 1
                    if misses <= int(end_grace_days):
                        active_rows.append(row)
                        continue
                    windows.append(
                        _finalize_window(
                            symbol=symbol,
                            direction=str(active["direction"]),
                            signal_start_date=str(active["signal_start_date"]),
                            active_from_date=str(active["active_from_date"]),
                            end_date=last_high_date,
                            rows=active_rows,
                        )
                    )
                    active = None
                    active_rows = []
                    misses = 0

            if not is_high:
                streak_direction = ""
                streak_start = ""
                streak_rows = []
                continue
            if direction != streak_direction:
                streak_direction = direction
                streak_start = row_date
                streak_rows = [row]
            else:
                streak_rows.append(row)
            if len(streak_rows) >= int(min_consecutive_high_days):
                active = {
                    "direction": direction,
                    "signal_start_date": streak_start,
                    "active_from_date": row_date,
                }
                active_rows = list(streak_rows)
                last_high_date = row_date
                misses = 0
                streak_direction = ""
                streak_start = ""
                streak_rows = []

        if active is not None:
            windows.append(
                _finalize_window(
                    symbol=symbol,
                    direction=str(active["direction"]),
                    signal_start_date=str(active["signal_start_date"]),
                    active_from_date=str(active["active_from_date"]),
                    end_date=last_high_date,
                    rows=active_rows,
                )
            )
    return windows


def assign_trades_to_windows(
    trades: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    *,
    max_trades_per_window: int = 1,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for window in windows:
        indexed.setdefault((str(window["symbol"]).upper(), str(window["direction"])), []).append(window)
    for values in indexed.values():
        values.sort(key=lambda item: str(item["active_from_date"]))

    assigned: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for trade in sorted(trades, key=lambda item: (str(item.get("entry_time") or ""), str(item.get("symbol") or ""))):
        symbol = str(trade.get("symbol") or "").upper()
        direction = str(trade.get("direction") or "")
        entry_date = _date_str(trade.get("entry_time"))
        matches = [
            window
            for window in indexed.get((symbol, direction), [])
            if str(window["active_from_date"]) <= entry_date <= str(window["end_date"])
        ]
        if not matches:
            continue
        window = matches[-1]
        window_id = str(window["window_id"])
        seq = counts.get(window_id, 0) + 1
        counts[window_id] = seq
        if seq > int(max_trades_per_window):
            continue
        row = dict(trade)
        row["window_id"] = window_id
        row["window_trade_seq"] = seq
        row["window_active_from_date"] = window["active_from_date"]
        row["window_end_date"] = window["end_date"]
        assigned.append(row)
    return assigned


def _account_config_for_policy(policy: WindowPolicy) -> AccountBacktestConfig:
    split_fraction = float(policy.split_initial_fraction)
    return AccountBacktestConfig(
        max_portfolio_margin_pct=float(policy.max_portfolio_margin_pct),
        position_budget_source="phase1_story",
        reserve_planned_second_entry_margin=True,
        risk_per_trade_pct=0.015,
        split_initial_fraction=split_fraction,
        split_second_entry_trigger="none" if split_fraction >= 1.0 else "tp1",
        commission_multiplier=1.01,
        scope=f"trend window activation: {_policy_label(policy)}",
    )


def _avg_margin_pct(equity_curve: list[dict[str, Any]]) -> float:
    values = [
        _finite_float(row.get("active_margin"), 0.0) / _finite_float(row.get("equity"), 1.0)
        for row in equity_curve
        if _finite_float(row.get("equity"), 0.0) > 0
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _early_stop_count(trades: list[dict[str, Any]]) -> int:
    return sum(1 for trade in trades if str(trade.get("actual_exit_reason")) == "stop" and _hold_days(trade) < 3.0)


def _run_policy_for_trades(
    *,
    trades: list[dict[str, Any]],
    policy: WindowPolicy,
    group_field: str,
    group_value: Any,
    candidate_tp2_count: int,
) -> dict[str, Any]:
    candidates = [_candidate_for_grid(row) for row in trades]
    result = run_account_backtest(candidates, _account_config_for_policy(policy))
    accepted = int(result.summary.get("accepted_trades", 0))
    tp2 = sum(1 for trade in result.trades if str(trade.get("actual_exit_reason")) == "tp2")
    early = _early_stop_count(result.trades)
    planned_second = int(result.summary.get("split_planned_second_entry_lots", 0))
    filled_second = int(result.summary.get("split_second_entry_filled_lots", 0))
    return {
        "policy": _policy_label(policy),
        "group_field": group_field,
        "group_value": str(group_value),
        "max_portfolio_margin_pct": float(policy.max_portfolio_margin_pct),
        "max_trades_per_window": int(policy.max_trades_per_window),
        "split_initial_fraction": float(policy.split_initial_fraction),
        "window_trades": len(trades),
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
        "split_first_entry_lots": int(result.summary.get("split_first_entry_lots", 0)),
        "split_planned_second_entry_lots": planned_second,
        "split_second_entry_filled_lots": filled_second,
        "split_second_entry_unfilled_lots": int(result.summary.get("split_second_entry_unfilled_lots", 0)),
        "second_entry_fill_rate": float(filled_second / planned_second) if planned_second else 0.0,
    }


def _window_yearly_rows(windows: list[dict[str, Any]], trades: list[dict[str, Any]], years: Iterable[int]) -> list[dict[str, Any]]:
    window_days_by_year: dict[int, set[str]] = {int(year): set() for year in years}
    windows_by_year: dict[int, int] = {int(year): 0 for year in years}
    for window in windows:
        dates = _daily_dates(str(window["active_from_date"]), str(window["end_date"]))
        touched_years = {pd.Timestamp(item).year for item in dates}
        for year in touched_years:
            if year in window_days_by_year:
                windows_by_year[year] += 1
        for item in dates:
            year = pd.Timestamp(item).year
            if year in window_days_by_year:
                window_days_by_year[year].add(item)
    trades_by_year: dict[int, int] = {int(year): 0 for year in years}
    first_trades_by_year: dict[int, int] = {int(year): 0 for year in years}
    later_trades_by_year: dict[int, int] = {int(year): 0 for year in years}
    for trade in trades:
        year = pd.Timestamp(trade.get("entry_time")).year
        if year not in trades_by_year:
            continue
        trades_by_year[year] += 1
        if int(trade.get("window_trade_seq") or 0) == 1:
            first_trades_by_year[year] += 1
        else:
            later_trades_by_year[year] += 1
    rows: list[dict[str, Any]] = []
    for year in years:
        rows.append(
            {
                "year": int(year),
                "windows": windows_by_year.get(int(year), 0),
                "any_window_days": len(window_days_by_year.get(int(year), set())),
                "window_months": len({item[:7] for item in window_days_by_year.get(int(year), set())}),
                "window_trade_candidates": trades_by_year.get(int(year), 0),
                "first_window_trade_candidates": first_trades_by_year.get(int(year), 0),
                "later_window_trade_candidates": later_trades_by_year.get(int(year), 0),
            }
        )
    return rows


def _conviction_scenarios() -> list[tuple[str, str, Any]]:
    return [
        ("base", "连续high窗口", lambda window: True),
        ("signal_avg75", "启动前连续high均分>=75", lambda window: float(window["signal_avg_score"]) >= 75.0),
        ("activation80", "启动当日分>=80", lambda window: float(window["activation_score"]) >= 80.0),
        ("activation85", "启动当日分>=85", lambda window: float(window["activation_score"]) >= 85.0),
    ]


def build_conviction_rows(
    *,
    windows: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    years: Iterable[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_tp2 = sum(1 for trade in trades if str(trade.get("exit_reason")) == "tp2")
    for scenario, description, predicate in _conviction_scenarios():
        filtered_windows = [window for window in windows if predicate(window)]
        window_days_by_year: dict[int, set[str]] = {int(year): set() for year in years}
        for window in filtered_windows:
            for item in _daily_dates(str(window["active_from_date"]), str(window["end_date"])):
                year = pd.Timestamp(item).year
                if year in window_days_by_year:
                    window_days_by_year[year].add(item)
        for max_trades in (1, 2):
            policy = WindowPolicy(0.30, max_trades, 1.0)
            selected = assign_trades_to_windows(trades, filtered_windows, max_trades_per_window=max_trades)
            result = _run_policy_for_trades(
                trades=selected,
                policy=policy,
                group_field="all",
                group_value="all",
                candidate_tp2_count=all_tp2,
            )
            result.update(
                {
                    "scenario": scenario,
                    "description": description,
                    "window_count": len(filtered_windows),
                    "any_window_days_avg": (
                        sum(len(value) for value in window_days_by_year.values()) / len(window_days_by_year)
                        if window_days_by_year
                        else 0.0
                    ),
                    "window_months_avg": (
                        sum(len({item[:7] for item in value}) for value in window_days_by_year.values())
                        / len(window_days_by_year)
                        if window_days_by_year
                        else 0.0
                    ),
                }
            )
            out.append(result)
    return out


def build_policy_rows(
    *,
    trades: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    policies: Iterable[WindowPolicy],
    years: Iterable[int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidate_tp2_all = sum(1 for trade in trades if str(trade.get("exit_reason")) == "tp2")
    for policy in policies:
        selected = assign_trades_to_windows(trades, windows, max_trades_per_window=policy.max_trades_per_window)
        out.append(
            _run_policy_for_trades(
                trades=selected,
                policy=policy,
                group_field="all",
                group_value="all",
                candidate_tp2_count=candidate_tp2_all,
            )
        )
        for year in years:
            year_trades = [row for row in selected if pd.Timestamp(row.get("entry_time")).year == int(year)]
            candidate_tp2_year = sum(1 for trade in trades if pd.Timestamp(trade.get("entry_time")).year == int(year) and str(trade.get("exit_reason")) == "tp2")
            out.append(
                _run_policy_for_trades(
                    trades=year_trades,
                    policy=policy,
                    group_field="year",
                    group_value=year,
                    candidate_tp2_count=candidate_tp2_year,
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
    windows: list[dict[str, Any]],
    window_trades: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
) -> str:
    all30 = [
        row
        for row in policy_rows
        if row.get("group_field") == "all" and abs(float(row.get("max_portfolio_margin_pct", 0.0)) - 0.30) < 1e-9
    ]
    year_policy = [
        row
        for row in policy_rows
        if row.get("group_field") == "year" and row.get("policy") == "m30_window1_first100"
    ]
    conviction_rows = sorted(
        [row for row in policy_rows if row.get("group_field") == "conviction"],
        key=lambda item: (str(item.get("scenario")), int(item.get("max_trades_per_window", 0))),
    )
    lines = [
        "# 强趋势窗口启动验证 2022-2025",
        "",
        "口径：只做回测分析，不改实盘默认规则。Phase1 high 连续出现才启动趋势交易模式，high 消失达到宽限天数或方向切换则关闭窗口；窗口内只允许前 1-2 笔趋势入场，Phase2 仍只是窗口内的入场触发。",
        "",
        "## 窗口概览",
        "",
        f"- 窗口数量：{len(windows)} 个。",
        f"- 窗口内候选交易：{len(window_trades)} 笔。",
        "",
        "| 年份 | 窗口数 | 有窗口天数 | 覆盖月份数 | 窗口内候选 | 第一笔候选 | 后续候选 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in yearly_rows:
        lines.append(
            f"| {row['year']} | {row['windows']} | {row['any_window_days']} | {row['window_months']} | "
            f"{row['window_trade_candidates']} | {row['first_window_trade_candidates']} | {row['later_window_trade_candidates']} |"
        )
    lines.extend(
        [
            "",
            "## 账户级窗口策略",
            "",
            "| 口径 | 收益 | 最大回撤 | 窗口内交易 | 接受交易 | 早止损率 | TP2捕获 | 平均占用 | 二笔成交率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(all30, key=lambda item: (int(item["max_trades_per_window"]), -float(item["split_initial_fraction"]))):
        lines.append(
            f"| {row['policy']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['window_trades']} | {row['accepted_trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} | {_pct(row['second_entry_fill_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 分年份：每窗口只做第一笔，首笔100%",
            "",
            "| 年份 | 收益 | 最大回撤 | 窗口内交易 | 接受交易 | 早止损率 | TP2捕获 | 平均占用 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(year_policy, key=lambda item: str(item["group_value"])):
        lines.append(
            f"| {row['group_value']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | "
            f"{row['window_trades']} | {row['accepted_trades']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
        )
    if conviction_rows:
        lines.extend(
            [
                "",
                "## 高确信窗口过滤",
                "",
                "| 场景 | 每窗口笔数 | 窗口数 | 平均有窗口天数/年 | 平均覆盖月份/年 | 收益 | 最大回撤 | 窗口内交易 | TP2捕获 | 平均占用 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in conviction_rows:
            lines.append(
                f"| {row['description']} | {row['max_trades_per_window']} | {row['window_count']} | "
                f"{_num(row['any_window_days_avg'], 1)} | {_num(row['window_months_avg'], 1)} | "
                f"{_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | {row['window_trades']} | "
                f"{_pct(row['tp2_capture_rate'])} | {_pct(row['avg_margin_pct_observed'])} |"
            )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 窗口由日线 Phase1 趋势故事质量生成，不使用入场后的交易结果。",
            "- 高确信过滤只使用窗口启动当日及启动前连续 high 段已经可见的分数，不使用窗口结束后的均分或峰值。",
            "- 当前窗口启动条件是连续 high 的机械定义，后续可继续测试方向稳定、过度延伸、参与度和窗口冷却期。",
            "- 这份报告只用于判断何时启动趋势交易模式，不进入实盘默认配置。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_trend_windows(
    *,
    trade_rows: pd.DataFrame,
    state_rows: list[dict[str, Any]],
    params: WindowParams,
    policies: Iterable[WindowPolicy] | None = None,
) -> dict[str, Any]:
    windows = identify_trend_windows(
        state_rows,
        min_consecutive_high_days=params.min_consecutive_high_days,
        end_grace_days=params.end_grace_days,
    )
    trades = trade_rows.to_dict("records")
    window_trades = assign_trades_to_windows(trades, windows, max_trades_per_window=99)
    years = list(range(int(params.start_year), int(params.end_year) + 1))
    policy_rows = build_policy_rows(trades=trades, windows=windows, policies=policies or iter_window_policies(), years=years)
    conviction_rows = build_conviction_rows(windows=windows, trades=trades, years=years)
    for row in conviction_rows:
        row["group_field"] = "conviction"
    report_policy_rows = policy_rows + conviction_rows
    yearly_rows = _window_yearly_rows(windows, window_trades, years)
    return {
        "diagnostics": {
            "trade_count": int(len(trade_rows)),
            "state_rows": int(len(state_rows)),
            "window_count": int(len(windows)),
            "window_trade_candidates": int(len(window_trades)),
        },
        "windows": windows,
        "window_trades": window_trades,
        "policy_rows": report_policy_rows,
        "conviction_rows": conviction_rows,
        "yearly_rows": yearly_rows,
    }


def write_outputs(report: dict[str, Any], output_prefix: Path) -> WindowOutputPaths:
    prefix = Path(output_prefix)
    paths = WindowOutputPaths(
        windows_csv=prefix.with_name(f"{prefix.name}_windows.csv"),
        trades_csv=prefix.with_name(f"{prefix.name}_trades.csv"),
        policy_csv=prefix.with_name(f"{prefix.name}_policy.csv"),
        conviction_csv=prefix.with_name(f"{prefix.name}_conviction.csv"),
        yearly_csv=prefix.with_name(f"{prefix.name}_yearly.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.windows_csv, report["windows"])
    _write_csv(paths.trades_csv, report["window_trades"])
    _write_csv(paths.policy_csv, report["policy_rows"])
    _write_csv(paths.conviction_csv, report["conviction_rows"])
    _write_csv(paths.yearly_csv, report["yearly_rows"])
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    paths.report_md.write_text(
        render_markdown(
            windows=report["windows"],
            window_trades=report["window_trades"],
            policy_rows=report["policy_rows"],
            yearly_rows=report["yearly_rows"],
        ),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze when to activate trend trading windows")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_PHASE23_CACHE_DIR)
    parser.add_argument("--market-cache-dir", type=Path, default=DEFAULT_MARKET_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--min-quality-score", type=float, default=65.0)
    parser.add_argument("--min-consecutive-high-days", type=int, default=5)
    parser.add_argument("--end-grace-days", type=int, default=5)
    args = parser.parse_args()
    params = WindowParams(
        min_quality_score=float(args.min_quality_score),
        min_consecutive_high_days=int(args.min_consecutive_high_days),
        end_grace_days=int(args.end_grace_days),
        start_year=min(args.years),
        end_year=max(args.years),
    )
    trade_rows = collect_trend_opportunity_rows(
        years=args.years,
        phase23_cache_dir=args.phase23_cache_dir,
        market_cache_dir=args.market_cache_dir,
    )
    symbols = sorted(trade_rows["symbol"].dropna().astype(str).str.upper().unique().tolist())
    state_rows = build_trend_state_rows(symbols=symbols, market_cache_dir=args.market_cache_dir, params=params)
    report = analyze_trend_windows(trade_rows=trade_rows, state_rows=state_rows, params=params)
    paths = write_outputs(report, args.output_prefix)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
