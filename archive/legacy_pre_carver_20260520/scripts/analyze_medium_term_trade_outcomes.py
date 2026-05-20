from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from scipy.stats import mannwhitneyu


DEFAULT_TRADES = Path("data/reports/backtest/trend_account_2025_phase2score_margin30_risk_0p015_trades.csv")
DEFAULT_PHASE2_DETAILS = Path("data/reports/backtest/phase2_score_forward_return_long_trend_core_2025_details.csv")
DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/medium_term_trade_outcome_validation_2025")

PRIMARY_SCORE_FIELD = "medium_term_quality_score"
SCORE_FIELDS = [
    "medium_term_quality_score",
    "medium_term_momentum_score",
    "medium_term_efficiency_score",
    "medium_term_trend_state_score",
    "medium_term_freshness_score",
    "medium_term_regime_score",
    "short_term_price_volume_score",
    "trend_structure_score",
    "risk_reward_ratio",
]
PHASE2_DETAIL_FIELDS = [
    "symbol",
    "direction",
    "trade_date",
    "phase2_state",
    "score",
    "abs_score",
    "score_bucket",
    "short_term_price_volume_score",
    "trend_structure_score",
    "trend_phase_score",
    "risk_reward_ratio",
    "admission_reward_risk",
    "medium_term_quality_score",
    "medium_term_momentum_score",
    "medium_term_efficiency_score",
    "medium_term_trend_state_score",
    "medium_term_freshness_score",
    "medium_term_regime_score",
    "medium_term_quality_status",
    "entry_price",
    "forward_5d_return",
    "forward_10d_return",
    "forward_20d_return",
    "forward_5d_max_favorable_return",
    "forward_10d_max_favorable_return",
    "forward_20d_max_favorable_return",
    "forward_5d_max_adverse_return",
    "forward_10d_max_adverse_return",
    "forward_20d_max_adverse_return",
]


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _score_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna()


def _summary_stats(frame: pd.DataFrame, field: str) -> dict[str, float | int | None]:
    values = _score_series(frame, field)
    if values.empty:
        return {
            "rows": int(len(frame)),
            "score_rows": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "rows": int(len(frame)),
        "score_rows": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
    }


def pairwise_auc(positive_scores: Iterable[float], negative_scores: Iterable[float]) -> float | None:
    positives = [_finite_float(value) for value in positive_scores]
    negatives = [_finite_float(value) for value in negative_scores]
    positives = [value for value in positives if math.isfinite(value)]
    negatives = [value for value in negatives if math.isfinite(value)]
    if not positives or not negatives:
        return None
    wins = 0.0
    pairs = 0
    for positive in positives:
        for negative in negatives:
            pairs += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / pairs) if pairs else None


def _mannwhitney_greater_pvalue(positive_scores: pd.Series, negative_scores: pd.Series) -> float | None:
    positives = pd.to_numeric(positive_scores, errors="coerce").dropna()
    negatives = pd.to_numeric(negative_scores, errors="coerce").dropna()
    if len(positives) < 2 or len(negatives) < 2:
        return None
    try:
        return float(mannwhitneyu(positives, negatives, alternative="greater").pvalue)
    except ValueError:
        return None


def _outcome_label(row: pd.Series, *, early_stop_days: int, long_hold_days: int) -> str:
    hold_days = _finite_float(row.get("hold_days"), default=0.0)
    net_pnl = _finite_float(row.get("net_pnl"), default=0.0)
    exit_reason = str(row.get("actual_exit_reason") or "")
    if exit_reason == "stop" and hold_days < early_stop_days:
        return "early_stop"
    if hold_days >= long_hold_days and net_pnl > 0:
        return "long_hold_winner"
    return "other"


def build_joined_trade_outcomes(
    trades: pd.DataFrame,
    phase2_details: pd.DataFrame,
    *,
    early_stop_days: int = 3,
    long_hold_days: int = 14,
) -> pd.DataFrame:
    trades = trades.copy()
    details = phase2_details.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["actual_exit_time"] = pd.to_datetime(trades["actual_exit_time"])
    trades["entry_date"] = trades["entry_time"].dt.normalize()
    trades["hold_days"] = (trades["actual_exit_time"] - trades["entry_time"]).dt.total_seconds() / 86400.0

    details["trade_date"] = pd.to_datetime(details["trade_date"])
    details["entry_date"] = details["trade_date"].dt.normalize()
    selected_fields = [field for field in PHASE2_DETAIL_FIELDS if field in details.columns]
    details = details[selected_fields + ["entry_date"]].copy()

    joined = trades.merge(
        details,
        on=["symbol", "direction", "entry_date"],
        how="left",
        suffixes=("", "_phase2_detail"),
    )
    joined["phase2_detail_matched"] = joined[PRIMARY_SCORE_FIELD].notna()
    joined["outcome_label"] = joined.apply(
        _outcome_label,
        axis=1,
        early_stop_days=early_stop_days,
        long_hold_days=long_hold_days,
    )
    joined["is_early_stop"] = joined["outcome_label"] == "early_stop"
    joined["is_long_hold_winner"] = joined["outcome_label"] == "long_hold_winner"
    return joined


def _compare_outcomes(joined: pd.DataFrame, field: str) -> dict[str, Any]:
    early = joined[joined["outcome_label"] == "early_stop"]
    long_win = joined[joined["outcome_label"] == "long_hold_winner"]
    early_scores = _score_series(early, field)
    long_scores = _score_series(long_win, field)
    early_stats = _summary_stats(early, field)
    long_stats = _summary_stats(long_win, field)
    median_difference = None
    mean_difference = None
    if early_stats["median"] is not None and long_stats["median"] is not None:
        median_difference = float(long_stats["median"] - early_stats["median"])
    if early_stats["mean"] is not None and long_stats["mean"] is not None:
        mean_difference = float(long_stats["mean"] - early_stats["mean"])
    auc = pairwise_auc(long_scores, early_scores)
    pvalue = _mannwhitney_greater_pvalue(long_scores, early_scores)
    return {
        "field": field,
        "early_stop": early_stats,
        "long_hold_winner": long_stats,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "pairwise_auc": auc,
        "mannwhitney_greater_pvalue": pvalue,
    }


def _threshold_rows(joined: pd.DataFrame, thresholds: Iterable[float], field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = pd.to_numeric(joined[field], errors="coerce") if field in joined.columns else pd.Series(dtype=float)
    for threshold in thresholds:
        kept = joined[scores >= threshold].copy()
        early_stop_count = int((kept["outcome_label"] == "early_stop").sum())
        long_hold_winner_count = int((kept["outcome_label"] == "long_hold_winner").sum())
        trade_count = int(len(kept))
        rows.append(
            {
                "threshold": float(threshold),
                "trade_count": trade_count,
                "early_stop_count": early_stop_count,
                "early_stop_rate": float(early_stop_count / trade_count) if trade_count else None,
                "long_hold_winner_count": long_hold_winner_count,
                "long_hold_winner_rate": float(long_hold_winner_count / trade_count) if trade_count else None,
                "net_pnl": float(pd.to_numeric(kept.get("net_pnl", pd.Series(dtype=float)), errors="coerce").sum())
                if trade_count
                else 0.0,
                "avg_net_pnl": float(pd.to_numeric(kept.get("net_pnl", pd.Series(dtype=float)), errors="coerce").mean())
                if trade_count
                else None,
            }
        )
    return rows


def _filter_result_row(
    kept: pd.DataFrame,
    *,
    medium_threshold: float,
    rr_threshold: float,
    total_trades: int,
) -> dict[str, Any]:
    early_stop_count = int((kept["outcome_label"] == "early_stop").sum())
    long_hold_winner_count = int((kept["outcome_label"] == "long_hold_winner").sum())
    trade_count = int(len(kept))
    net_pnl = pd.to_numeric(kept.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
    return {
        "medium_term_quality_min": float(medium_threshold),
        "risk_reward_ratio_min": float(rr_threshold),
        "trade_count": trade_count,
        "trade_keep_rate": float(trade_count / total_trades) if total_trades else None,
        "early_stop_count": early_stop_count,
        "early_stop_rate": float(early_stop_count / trade_count) if trade_count else None,
        "long_hold_winner_count": long_hold_winner_count,
        "long_hold_winner_rate": float(long_hold_winner_count / trade_count) if trade_count else None,
        "net_pnl": float(net_pnl.sum()) if trade_count else 0.0,
        "avg_net_pnl": float(net_pnl.mean()) if trade_count else None,
    }


def analyze_filter_combinations(
    joined: pd.DataFrame,
    *,
    medium_thresholds: Iterable[float] = (0.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    rr_thresholds: Iterable[float] = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0),
) -> list[dict[str, Any]]:
    if PRIMARY_SCORE_FIELD not in joined.columns or "risk_reward_ratio" not in joined.columns:
        return []
    medium_scores = pd.to_numeric(joined[PRIMARY_SCORE_FIELD], errors="coerce")
    rr_scores = pd.to_numeric(joined["risk_reward_ratio"], errors="coerce")
    rows: list[dict[str, Any]] = []
    total_trades = int(len(joined))
    for medium_threshold in medium_thresholds:
        for rr_threshold in rr_thresholds:
            kept = joined[(medium_scores >= medium_threshold) & (rr_scores >= rr_threshold)]
            rows.append(
                _filter_result_row(
                    kept,
                    medium_threshold=float(medium_threshold),
                    rr_threshold=float(rr_threshold),
                    total_trades=total_trades,
                )
            )
    return rows


def _separation_label(comparison: dict[str, Any]) -> str:
    auc = comparison.get("pairwise_auc")
    pvalue = comparison.get("mannwhitney_greater_pvalue")
    median_difference = comparison.get("median_difference")
    if auc is None or median_difference is None:
        return "样本不足"
    if auc >= 0.60 and median_difference > 0 and (pvalue is None or pvalue < 0.05):
        return "有统计区分力"
    if auc >= 0.55 and median_difference > 0:
        return "有弱区分力"
    return "未显示有效区分力"


def analyze_joined_trade_outcomes(
    joined: pd.DataFrame,
    *,
    score_fields: Iterable[str] = SCORE_FIELDS,
    thresholds: Iterable[float] = (20.0, 30.0, 40.0, 50.0, 60.0),
) -> dict[str, Any]:
    detail_matched = (
        joined["phase2_detail_matched"].astype(bool)
        if "phase2_detail_matched" in joined.columns
        else pd.Series([True] * len(joined), index=joined.index)
    )
    comparisons = [_compare_outcomes(joined, field) for field in score_fields if field in joined.columns]
    primary = next(item for item in comparisons if item["field"] == PRIMARY_SCORE_FIELD)
    outcome_counts = joined["outcome_label"].value_counts().to_dict()
    report = {
        "diagnostics": {
            "trade_count": int(len(joined)),
            "matched_phase2_detail_count": int(detail_matched.sum()),
            "unmatched_phase2_detail_count": int((~detail_matched).sum()),
            "early_stop_count": int(outcome_counts.get("early_stop", 0)),
            "long_hold_winner_count": int(outcome_counts.get("long_hold_winner", 0)),
            "other_count": int(outcome_counts.get("other", 0)),
        },
        "primary_score_field": PRIMARY_SCORE_FIELD,
        "primary_comparison": primary,
        "separation_label": _separation_label(primary),
        "component_comparisons": comparisons,
        "thresholds": _threshold_rows(joined, thresholds, PRIMARY_SCORE_FIELD),
        "filter_combinations": analyze_filter_combinations(joined),
    }
    return report


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def render_markdown(report: dict[str, Any], *, early_stop_days: int, long_hold_days: int) -> str:
    diagnostics = report["diagnostics"]
    primary = report["primary_comparison"]
    lines = [
        "# Medium-Term Score Trade Outcome Validation",
        "",
        "## 业务问题",
        "",
        f"- 早止损定义: 入场后 {early_stop_days} 天内触发止损。",
        f"- 长持仓盈利定义: 持仓至少 {long_hold_days} 天且净收益为正。",
        "- 验证目标: 中期趋势成熟度能否区分早止损亏损与长持仓盈利。",
        "",
        "## 样本覆盖",
        "",
        f"- 交易数: {diagnostics['trade_count']}",
        f"- 成功匹配 Phase2 明细: {diagnostics['matched_phase2_detail_count']}",
        f"- 早止损: {diagnostics['early_stop_count']}",
        f"- 长持仓盈利: {diagnostics['long_hold_winner_count']}",
        f"- 其他: {diagnostics['other_count']}",
        "",
        "## 核心结论",
        "",
        f"- 中期评分区分结果: {report['separation_label']}",
        f"- 早止损中期评分中位数: {_num(primary['early_stop']['median'])}",
        f"- 长持仓盈利中期评分中位数: {_num(primary['long_hold_winner']['median'])}",
        f"- 中位数差: {_num(primary['median_difference'])}",
        f"- Pairwise AUC: {_num(primary['pairwise_auc'], 3)}",
        f"- Mann-Whitney 单侧 p 值: {_num(primary['mannwhitney_greater_pvalue'], 4)}",
        "",
        "## 分数阈值观察",
        "",
        "| 中期分阈值 | 保留交易 | 早止损率 | 长持仓盈利率 | 净收益 | 平均净收益 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["thresholds"]:
        lines.append(
            "| "
            f"{_num(row['threshold'], 0)} | "
            f"{row['trade_count']} | "
            f"{_pct(row['early_stop_rate'])} | "
            f"{_pct(row['long_hold_winner_rate'])} | "
            f"{_num(row['net_pnl'])} | "
            f"{_num(row['avg_net_pnl'])} |"
        )

    lines.extend(
        [
            "",
            "## 中期分 × 收益风险比组合筛选",
            "",
            "这是基于既有交易结果的交易级反事实筛选，不包含跳过交易后的资金再分配。",
            "",
            "| 中期分下限 | 收益风险比下限 | 保留交易 | 保留率 | 早止损率 | 长持仓盈利率 | 净收益 | 平均净收益 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    combo_rows = sorted(
        report.get("filter_combinations", []),
        key=lambda row: (
            -float(row.get("net_pnl") or 0.0),
            float(row.get("trade_count") or 0),
        ),
    )[:12]
    for row in combo_rows:
        lines.append(
            "| "
            f"{_num(row['medium_term_quality_min'], 0)} | "
            f"{_num(row['risk_reward_ratio_min'], 1)} | "
            f"{row['trade_count']} | "
            f"{_pct(row['trade_keep_rate'])} | "
            f"{_pct(row['early_stop_rate'])} | "
            f"{_pct(row['long_hold_winner_rate'])} | "
            f"{_num(row['net_pnl'])} | "
            f"{_num(row['avg_net_pnl'])} |"
        )

    lines.extend(
        [
            "",
            "## 组件对比",
            "",
            "| 评分字段 | 早止损中位数 | 长持仓盈利中位数 | 中位数差 | AUC | p 值 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["component_comparisons"]:
        lines.append(
            "| "
            f"{row['field']} | "
            f"{_num(row['early_stop']['median'])} | "
            f"{_num(row['long_hold_winner']['median'])} | "
            f"{_num(row['median_difference'])} | "
            f"{_num(row['pairwise_auc'], 3)} | "
            f"{_num(row['mannwhitney_greater_pvalue'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 这是回测审计报告，不改变默认策略、实盘流程或仓位规则。",
            "- 如果区分力不足，下一步应转向入场触发、早期止损和持仓管理归因，而不是继续简单提高中期分权重。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether medium-term scores separate early stops from long winners")
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--phase2-details", type=Path, default=DEFAULT_PHASE2_DETAILS)
    parser.add_argument("--early-stop-days", type=int, default=3)
    parser.add_argument("--long-hold-days", type=int, default=14)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trades = pd.read_csv(args.trades)
    details = pd.read_csv(args.phase2_details)
    joined = build_joined_trade_outcomes(
        trades,
        details,
        early_stop_days=args.early_stop_days,
        long_hold_days=args.long_hold_days,
    )
    report = analyze_joined_trade_outcomes(joined)
    markdown = render_markdown(
        report,
        early_stop_days=args.early_stop_days,
        long_hold_days=args.long_hold_days,
    )

    prefix = args.output_prefix
    _write_text(prefix.with_suffix(".md"), markdown)
    _write_text(prefix.with_suffix(".json"), json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    joined.to_csv(prefix.with_name(prefix.name + "_joined.csv"), index=False)
    pd.DataFrame(report["thresholds"]).to_csv(prefix.with_name(prefix.name + "_thresholds.csv"), index=False)
    pd.DataFrame(report["filter_combinations"]).to_csv(
        prefix.with_name(prefix.name + "_filter_combinations.csv"),
        index=False,
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
