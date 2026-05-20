from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from analyze_split_execution_factors import _candidate_from_row, _finite_float, _hold_days
from analyze_trend_opportunity_quality import (
    DEFAULT_MARKET_CACHE_DIR,
    DEFAULT_PHASE23_CACHE_DIR,
    collect_trend_opportunity_rows,
    trend_budget_margin_pct,
)
from backtest.account_runner import AccountBacktestConfig, AccountCandidate, run_account_backtest


DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/phase1_phase2_split_grid_2022_2025")


@dataclass(frozen=True)
class GridSpec:
    max_portfolio_margin_pct: float
    phase1_budget_min: float
    phase2_score_min: float
    split_initial_fraction: float


@dataclass(frozen=True)
class GridOutputPaths:
    grid_csv: Path
    summary_json: Path
    report_md: Path


def iter_grid_specs(
    *,
    phase1_budget_mins: Iterable[float] = (0.0, 0.15, 0.30),
    phase2_score_mins: Iterable[float] = (0.0, 35.0, 40.0, 45.0, 50.0, 55.0),
    split_initial_fractions: Iterable[float] = (1.0, 0.70, 0.50, 0.30),
    max_portfolio_margin_pcts: Iterable[float] = (0.30, 0.50),
) -> Iterable[GridSpec]:
    for max_margin in max_portfolio_margin_pcts:
        for phase1_min in phase1_budget_mins:
            for phase2_min in phase2_score_mins:
                for split_fraction in split_initial_fractions:
                    yield GridSpec(
                        float(max_margin),
                        float(phase1_min),
                        float(phase2_min),
                        float(split_fraction),
                    )


def _grid_label(spec: GridSpec) -> str:
    margin = int(round(float(spec.max_portfolio_margin_pct) * 100))
    phase1 = int(round(float(spec.phase1_budget_min) * 100))
    phase2 = int(round(float(spec.phase2_score_min)))
    first = int(round(float(spec.split_initial_fraction) * 100))
    return f"m{margin}_p1>={phase1}_p2>={phase2}_first{first}"


def _candidate_for_grid(row: dict[str, Any]) -> AccountCandidate:
    candidate = _candidate_from_row(row)
    quality_score = _finite_float(row.get("trend_opportunity_quality_score"), 0.0)
    return replace(
        candidate,
        phase1_story_budget_margin_pct=trend_budget_margin_pct(quality_score),
        phase1_story_budget_source="trend_story",
        medium_term_quality_score=quality_score,
        medium_term_entry_location_score=_finite_float(row.get("trend_structure_health_score"), 0.0),
    )


def _config_for_grid(spec: GridSpec) -> AccountBacktestConfig:
    split_fraction = float(spec.split_initial_fraction)
    return AccountBacktestConfig(
        max_portfolio_margin_pct=float(spec.max_portfolio_margin_pct),
        position_budget_source="phase1_story",
        reserve_planned_second_entry_margin=True,
        min_phase1_story_budget_margin_pct=float(spec.phase1_budget_min),
        min_phase2_abs_score=float(spec.phase2_score_min),
        risk_per_trade_pct=0.015,
        split_initial_fraction=split_fraction,
        split_second_entry_trigger="none" if split_fraction >= 1.0 else "tp1",
        commission_multiplier=1.01,
        scope=f"phase1/phase2 threshold split grid: {_grid_label(spec)}",
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


def _row_for_result(*, source: pd.DataFrame, spec: GridSpec, group_field: str, group_value: Any) -> dict[str, Any]:
    candidates = [_candidate_for_grid(row) for row in source.to_dict("records")]
    result = run_account_backtest(candidates, _config_for_grid(spec))
    candidate_tp2 = int((source["exit_reason"].astype(str) == "tp2").sum()) if len(source) else 0
    account_tp2 = sum(1 for trade in result.trades if str(trade.get("actual_exit_reason")) == "tp2")
    early_stop_trades = _account_early_stop_count(result.trades)
    accepted = int(result.summary.get("accepted_trades", 0))
    planned_second = int(result.summary.get("split_planned_second_entry_lots", 0))
    filled_second = int(result.summary.get("split_second_entry_filled_lots", 0))
    skipped = result.summary.get("skipped_reasons") or {}
    return {
        "grid": _grid_label(spec),
        "group_field": group_field,
        "group_value": str(group_value),
        "max_portfolio_margin_pct": float(spec.max_portfolio_margin_pct),
        "phase1_budget_min": float(spec.phase1_budget_min),
        "phase2_score_min": float(spec.phase2_score_min),
        "split_initial_fraction": float(spec.split_initial_fraction),
        "candidate_trades": int(len(source)),
        "accepted_trades": accepted,
        "accepted_rate": float(accepted / len(source)) if len(source) else 0.0,
        "return_pct": float(result.summary.get("return_pct", 0.0)),
        "net_profit": float(result.summary.get("net_profit", 0.0)),
        "max_drawdown_pct": float(result.summary.get("max_drawdown_realized_pct", 0.0)),
        "early_stop_trades": int(early_stop_trades),
        "early_stop_rate": float(early_stop_trades / accepted) if accepted else 0.0,
        "tp2_trades": int(account_tp2),
        "tp2_capture_rate": float(account_tp2 / candidate_tp2) if candidate_tp2 else 0.0,
        "max_margin_pct_observed": float(result.summary.get("max_margin_pct_observed", 0.0)),
        "avg_margin_pct_observed": _avg_margin_pct(result.equity_curve),
        "split_target_lots": int(result.summary.get("split_target_lots", 0)),
        "split_first_entry_lots": int(result.summary.get("split_first_entry_lots", 0)),
        "split_planned_second_entry_lots": planned_second,
        "split_second_entry_filled_lots": filled_second,
        "split_second_entry_unfilled_lots": int(result.summary.get("split_second_entry_unfilled_lots", 0)),
        "second_entry_fill_rate": float(filled_second / planned_second) if planned_second else 0.0,
        "skipped_phase1_story": int(skipped.get("phase1_story_below_trade_threshold", 0)),
        "skipped_phase2_trigger": int(skipped.get("phase2_trigger_below_trade_threshold", 0)),
        "skipped_portfolio_margin": int(skipped.get("portfolio_margin_full_lower_score", 0)),
        "skipped_reasons": json.dumps(skipped, ensure_ascii=False, sort_keys=True),
    }


def build_grid_rows(rows: pd.DataFrame, specs: Iterable[GridSpec] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs or iter_grid_specs():
        all_row = _row_for_result(source=rows, spec=spec, group_field="all", group_value="all")
        year_rows: list[dict[str, Any]] = []
        for year in sorted(rows["year"].dropna().astype(int).unique().tolist()):
            source = rows[rows["year"].astype(int) == int(year)].copy()
            year_rows.append(_row_for_result(source=source, spec=spec, group_field="year", group_value=year))
        all_row["positive_years"] = sum(1 for row in year_rows if float(row["return_pct"]) > 0)
        all_row["worst_year_return_pct"] = min((float(row["return_pct"]) for row in year_rows), default=0.0)
        all_row["max_year_drawdown_pct"] = max((float(row["max_drawdown_pct"]) for row in year_rows), default=0.0)
        out.append(all_row)
        out.extend(year_rows)
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


def _all_rows(grid_rows: list[dict[str, Any]], *, max_margin: float = 0.30) -> list[dict[str, Any]]:
    return [
        row
        for row in grid_rows
        if row.get("group_field") == "all" and abs(float(row.get("max_portfolio_margin_pct", 0.0)) - max_margin) < 1e-9
    ]


def _row_matches(row: dict[str, Any], **filters: Any) -> bool:
    for key, value in filters.items():
        if str(row.get(key)) != str(value):
            return False
    return True


def _table_for(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 组合 | 收益 | 最大回撤 | 最差年份 | 正收益年份 | 接受率 | 早止损率 | TP2捕获 | 平均占用 | 二笔成交率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['grid']} | {_pct(row['return_pct'])} | {_pct(row['max_drawdown_pct'])} | "
            f"{_pct(row.get('worst_year_return_pct'))} | {_num(row.get('positive_years'), 0)}/4 | "
            f"{_pct(row['accepted_rate'])} | {_pct(row['early_stop_rate'])} | {_pct(row['tp2_capture_rate'])} | "
            f"{_pct(row['avg_margin_pct_observed'])} | {_pct(row['second_entry_fill_rate'])} |"
        )
    return lines


def render_markdown(grid_rows: list[dict[str, Any]]) -> str:
    all30 = _all_rows(grid_rows, max_margin=0.30)
    controlled = [
        row
        for row in all30
        if float(row["max_drawdown_pct"]) <= 0.20 and int(row.get("positive_years") or 0) >= 4
    ]
    controlled = sorted(controlled, key=lambda row: float(row["return_pct"]), reverse=True)[:10]
    return_leaders = sorted(all30, key=lambda row: float(row["return_pct"]), reverse=True)[:10]
    phase2_sweep = sorted(
        [
            row
            for row in all30
            if _row_matches(row, phase1_budget_min=0.15, split_initial_fraction=0.5)
        ],
        key=lambda row: float(row["phase2_score_min"]),
    )
    split_sweep = sorted(
        [
            row
            for row in all30
            if _row_matches(row, phase1_budget_min=0.15, phase2_score_min=40.0)
        ],
        key=lambda row: float(row["split_initial_fraction"]),
        reverse=True,
    )
    phase1_sweep = sorted(
        [
            row
            for row in all30
            if _row_matches(row, phase2_score_min=40.0, split_initial_fraction=0.5)
        ],
        key=lambda row: float(row["phase1_budget_min"]),
    )
    lines = [
        "# Phase1/Phase2 阈值与分仓比例网格 2022-2025",
        "",
        "口径：固定现有 Phase23 趋势入场样本，不改默认交易规则。Phase1 趋势故事预算决定交易资格和理论预算，Phase2 短线触发分只作为入场门槛，分仓比例决定首笔释放；所有分仓口径都预留计划二笔加仓预算，不把首笔省下的保证金拿去买更多票。",
        "",
        "## 当前读数",
        "",
        "- 当前样本里的 Phase1 趋势预算主要是 medium/high 两档，所以 Phase1 门槛真正有区分力的是是否只保留 high 桶。",
        "- Phase2 门槛越高，交易数量和 TP2 捕获通常下降；它更像短线入场过滤，不适合单独决定趋势机会是否存在。",
        "- 首笔比例越低，账户控制通常更好，但二笔成交率低时会牺牲趋势捕获；这需要和 TP1 后加仓条件一起看。",
        "",
        "## 控制优先候选",
        "",
    ]
    lines.extend(_table_for(controlled))
    lines.extend(["", "## 收益领先候选", ""])
    lines.extend(_table_for(return_leaders))
    lines.extend(["", "## Phase2 门槛扫描（Phase1>=15%，首笔50%）", ""])
    lines.extend(_table_for(phase2_sweep))
    lines.extend(["", "## 首笔比例扫描（Phase1>=15%，Phase2>=40）", ""])
    lines.extend(_table_for(split_sweep))
    lines.extend(["", "## Phase1 门槛扫描（Phase2>=40，首笔50%）", ""])
    lines.extend(_table_for(phase1_sweep))
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 这是参数网格报告，不是默认实盘规则。",
            "- Phase1 门槛使用趋势故事预算档位；当前样本没有 low 桶，后续需要在更宽的候选池里继续验证。",
            "- Phase2 门槛只代表短线触发强度，不代表中期趋势质量。",
            "- 分仓比例只验证首笔释放，二笔仍按当前 TP1 触发加仓口径。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_grid(rows: pd.DataFrame) -> dict[str, Any]:
    grid_rows = build_grid_rows(rows)
    return {
        "diagnostics": {
            "trade_count": int(len(rows)),
            "years": {str(year): int(count) for year, count in rows.groupby("year").size().items()},
            "grid_count": int(len([row for row in grid_rows if row.get("group_field") == "all"])),
        },
        "grid_rows": grid_rows,
    }


def write_outputs(report: dict[str, Any], output_prefix: Path) -> GridOutputPaths:
    prefix = Path(output_prefix)
    paths = GridOutputPaths(
        grid_csv=prefix.with_name(f"{prefix.name}_grid.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.grid_csv, report["grid_rows"])
    paths.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    paths.report_md.write_text(render_markdown(report["grid_rows"]), encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Phase1/Phase2 thresholds and split fractions")
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
    report = analyze_grid(rows)
    paths = write_outputs(report, args.output_prefix)
    print(json.dumps(asdict(paths), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
