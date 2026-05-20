from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CACHE_DIR = Path("data/cache/backtest/phase23_task4_split_v1")
DEFAULT_OUTPUT_PREFIX = Path("data/reports/backtest/tp1_take_profit_effect_2022_2025")


@dataclass(frozen=True)
class TP1OutputPaths:
    details_csv: Path
    summary_json: Path
    by_exit_reason_csv: Path
    by_first_rr_bucket_csv: Path
    report_md: Path


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def calc_pnl_ratio(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "short":
        return (float(entry_price) - float(exit_price)) / float(entry_price)
    return (float(exit_price) - float(entry_price)) / float(entry_price)


def _bucket_first_rr(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    if number < 1.0:
        return "lt_1.0"
    if number < 1.5:
        return "1.0_1.5"
    if number < 2.0:
        return "1.5_2.0"
    return "2.0_plus"


def prepare_tp1_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    for field in (
        "entry_price",
        "exit_price",
        "pnl_ratio",
        "entry_rr",
        "entry_admission_rr",
        "tp1_exit_fraction",
        "tp1_exit_price",
        "phase2_score",
        "medium_term_quality_score",
    ):
        if field in rows.columns:
            rows[field] = pd.to_numeric(rows[field], errors="coerce")
    rows["tp1_hit"] = rows["tp1_hit"].apply(_as_bool) if "tp1_hit" in rows.columns else False
    rows["entry_time"] = pd.to_datetime(rows["entry_time"])
    rows["exit_time"] = pd.to_datetime(rows["exit_time"])
    rows["hold_days"] = (rows["exit_time"] - rows["entry_time"]).dt.total_seconds() / 86400.0

    first_leg_ratios: list[float] = []
    final_leg_ratios: list[float] = []
    final_leg_r_values: list[float] = []
    realized_r_values: list[float] = []
    continuation_values: list[float] = []
    reduction_delta_values: list[float] = []
    opportunity_cost_values: list[float] = []
    reduction_helped_values: list[bool] = []
    for row in rows.to_dict("records"):
        if not _as_bool(row.get("tp1_hit")):
            first_leg_ratios.append(math.nan)
            final_leg_ratios.append(math.nan)
            final_leg_r_values.append(math.nan)
            realized_r_values.append(math.nan)
            continuation_values.append(math.nan)
            reduction_delta_values.append(math.nan)
            opportunity_cost_values.append(math.nan)
            reduction_helped_values.append(False)
            continue
        first_leg = calc_pnl_ratio(
            str(row.get("direction") or ""),
            _finite_float(row.get("entry_price")),
            _finite_float(row.get("tp1_exit_price")),
        )
        final_leg = calc_pnl_ratio(
            str(row.get("direction") or ""),
            _finite_float(row.get("entry_price")),
            _finite_float(row.get("exit_price")),
        )
        entry_rr = _finite_float(row.get("entry_rr"))
        realized = _finite_float(row.get("pnl_ratio"))
        if first_leg <= 0 or not math.isfinite(entry_rr):
            final_r = math.nan
            realized_r = math.nan
            continuation = math.nan
            delta = math.nan
            opportunity_cost = math.nan
        else:
            final_r = final_leg / first_leg * entry_rr
            realized_r = realized / first_leg * entry_rr
            continuation = final_r - entry_rr
            delta = realized_r - final_r
            opportunity_cost = final_r - realized_r
        first_leg_ratios.append(first_leg)
        final_leg_ratios.append(final_leg)
        final_leg_r_values.append(final_r)
        realized_r_values.append(realized_r)
        continuation_values.append(continuation)
        reduction_delta_values.append(delta)
        opportunity_cost_values.append(opportunity_cost)
        reduction_helped_values.append(bool(math.isfinite(delta) and delta > 0))
    rows["tp1_first_leg_pnl_ratio"] = first_leg_ratios
    rows["final_leg_pnl_ratio"] = final_leg_ratios
    rows["final_leg_r"] = final_leg_r_values
    rows["realized_r"] = realized_r_values
    rows["post_tp1_continuation_r"] = continuation_values
    rows["tp1_reduction_delta_r"] = reduction_delta_values
    rows["tp1_opportunity_cost_r"] = opportunity_cost_values
    rows["tp1_reduction_helped"] = reduction_helped_values
    rows["tp1_reduction_cost"] = rows["tp1_opportunity_cost_r"] > 0
    rows["post_tp1_continued"] = rows["post_tp1_continuation_r"] > 0
    rows["entry_rr_bucket"] = rows["entry_rr"].apply(_bucket_first_rr)
    return rows


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().sum())


def _rate(frame: pd.DataFrame, field: str) -> float:
    return float(frame[field].mean()) if len(frame) else 0.0


def _group_row(group: pd.DataFrame, *, label_field: str, label: str) -> dict[str, Any]:
    return {
        label_field: label,
        "tp1_hit_trades": int(len(group)),
        "post_tp1_continued_trades": int(group["post_tp1_continued"].sum()),
        "post_tp1_continued_rate": _rate(group, "post_tp1_continued"),
        "tp1_reduction_helped_trades": int(group["tp1_reduction_helped"].sum()),
        "tp1_reduction_helped_rate": _rate(group, "tp1_reduction_helped"),
        "tp1_reduction_cost_trades": int(group["tp1_reduction_cost"].sum()),
        "tp1_reduction_cost_rate": _rate(group, "tp1_reduction_cost"),
        "avg_post_tp1_continuation_r": _mean(group["post_tp1_continuation_r"]),
        "avg_tp1_reduction_delta_r": _mean(group["tp1_reduction_delta_r"]),
        "total_tp1_reduction_delta_r": _sum(group["tp1_reduction_delta_r"]),
        "avg_tp1_opportunity_cost_r": _mean(group["tp1_opportunity_cost_r"]),
        "total_tp1_opportunity_cost_r": _sum(group["tp1_opportunity_cost_r"]),
        "avg_final_leg_r": _mean(group["final_leg_r"]),
        "avg_realized_r": _mean(group["realized_r"]),
    }


def by_exit_reason_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    hit = rows[rows["tp1_hit"]].copy()
    return [
        _group_row(group, label_field="exit_reason", label=str(reason))
        for reason, group in hit.groupby("exit_reason", dropna=False)
    ]


def by_first_rr_bucket_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    hit = rows[rows["tp1_hit"]].copy()
    order = ["lt_1.0", "1.0_1.5", "1.5_2.0", "2.0_plus", "missing"]
    out: list[dict[str, Any]] = []
    for bucket in order:
        group = hit[hit["entry_rr_bucket"] == bucket]
        if group.empty:
            continue
        out.append(_group_row(group, label_field="entry_rr_bucket", label=bucket))
    return out


def analyze_tp1_effect(rows: pd.DataFrame) -> dict[str, Any]:
    hit = rows[rows["tp1_hit"]].copy()
    return {
        "total_trades": int(len(rows)),
        "tp1_hit_trades": int(len(hit)),
        "tp1_hit_rate": float(len(hit) / len(rows)) if len(rows) else 0.0,
        "post_tp1_continued_trades": int(hit["post_tp1_continued"].sum()),
        "post_tp1_continued_rate": _rate(hit, "post_tp1_continued"),
        "tp1_reduction_helped_trades": int(hit["tp1_reduction_helped"].sum()),
        "tp1_reduction_helped_rate": _rate(hit, "tp1_reduction_helped"),
        "tp1_reduction_cost_trades": int(hit["tp1_reduction_cost"].sum()),
        "tp1_reduction_cost_rate": _rate(hit, "tp1_reduction_cost"),
        "avg_post_tp1_continuation_r": _mean(hit["post_tp1_continuation_r"]),
        "avg_tp1_reduction_delta_r": _mean(hit["tp1_reduction_delta_r"]),
        "total_tp1_reduction_delta_r": _sum(hit["tp1_reduction_delta_r"]),
        "avg_tp1_opportunity_cost_r": _mean(hit["tp1_opportunity_cost_r"]),
        "total_tp1_opportunity_cost_r": _sum(hit["tp1_opportunity_cost_r"]),
        "avg_final_leg_r": _mean(hit["final_leg_r"]),
        "avg_realized_r": _mean(hit["realized_r"]),
        "by_exit_reason": by_exit_reason_rows(rows),
        "by_first_rr_bucket": by_first_rr_bucket_rows(rows),
    }


def _parse_years(raw: str) -> set[int]:
    return {int(item.strip()) for item in str(raw).split(",") if item.strip()}


def collect_trade_rows_from_cache(cache_dir: Path, *, years: set[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(cache_dir).glob("*_phase23_cache.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for trade in ((payload.get("result") or {}).get("trades") or []):
            entry_time = pd.Timestamp(trade.get("entry_time"))
            if years and int(entry_time.year) not in years:
                continue
            meta = trade.get("meta") or {}
            records.append(
                {
                    "year": int(entry_time.year),
                    "trade_id": trade.get("trade_id", ""),
                    "symbol": trade.get("symbol", ""),
                    "direction": trade.get("direction", ""),
                    "entry_time": trade.get("entry_time", ""),
                    "exit_time": trade.get("exit_time", ""),
                    "exit_reason": trade.get("exit_reason", ""),
                    "entry_price": trade.get("entry_price", 0.0),
                    "exit_price": trade.get("exit_price", 0.0),
                    "pnl_ratio": trade.get("pnl_ratio", 0.0),
                    "tp1_hit": trade.get("tp1_hit", False),
                    "entry_rr": meta.get("entry_rr", meta.get("execution_rr", meta.get("rr", 0.0))),
                    "entry_admission_rr": meta.get(
                        "entry_admission_rr",
                        meta.get("execution_admission_rr", meta.get("admission_rr", 0.0)),
                    ),
                    "tp1_exit_fraction": meta.get("tp1_exit_fraction", 0.0),
                    "tp1_exit_price": meta.get("tp1_exit_price", 0.0),
                    "tp1_exit_time": meta.get("tp1_exit_time", ""),
                    "phase2_score": meta.get("phase2_score", 0.0),
                    "management_profile": meta.get("management_profile", ""),
                    "protective_stop": meta.get("protective_stop", 0.0),
                }
            )
    return pd.DataFrame(records)


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


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
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


def _num(value: Any) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TP1 止盈减仓效果验证",
        "",
        "口径：只看 Phase23 候选交易。对已经触发 TP1 的交易，比较“当前 TP1 减仓后的实现收益”和“假设不在 TP1 减仓、持有同一退出点”的收益差。正数表示 TP1 减仓保护了收益，负数表示 TP1 减仓牺牲了后续趋势收益。",
        "",
        "## 总览",
        "",
        f"- 候选交易数：{report['total_trades']}",
        f"- TP1 触发数：{report['tp1_hit_trades']}（{_pct(report['tp1_hit_rate'])}）",
        f"- TP1 后继续走强数：{report['post_tp1_continued_trades']}（{_pct(report['post_tp1_continued_rate'])}）",
        f"- TP1 减仓保护收益数：{report['tp1_reduction_helped_trades']}（{_pct(report['tp1_reduction_helped_rate'])}）",
        f"- TP1 减仓牺牲收益数：{report['tp1_reduction_cost_trades']}（{_pct(report['tp1_reduction_cost_rate'])}）",
        f"- 平均 TP1 后续延伸：{_num(report['avg_post_tp1_continuation_r'])} R",
        f"- 平均 TP1 减仓收益差：{_num(report['avg_tp1_reduction_delta_r'])} R",
        "",
        "## 按退出原因",
        "",
        "| 退出原因 | TP1触发 | 后续走强率 | 减仓保护率 | 减仓牺牲率 | 平均后续延伸R | 平均减仓收益差R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_exit_reason"]:
        lines.append(
            f"| {row['exit_reason']} | {row['tp1_hit_trades']} | {_pct(row['post_tp1_continued_rate'])} | "
            f"{_pct(row['tp1_reduction_helped_rate'])} | {_pct(row['tp1_reduction_cost_rate'])} | "
            f"{_num(row['avg_post_tp1_continuation_r'])} | {_num(row['avg_tp1_reduction_delta_r'])} |"
        )
    lines.extend(
        [
            "",
            "## 按第一 RR 桶",
            "",
            "| 第一RR桶 | TP1触发 | 后续走强率 | 减仓保护率 | 减仓牺牲率 | 平均后续延伸R | 平均减仓收益差R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["by_first_rr_bucket"]:
        lines.append(
            f"| {row['entry_rr_bucket']} | {row['tp1_hit_trades']} | {_pct(row['post_tp1_continued_rate'])} | "
            f"{_pct(row['tp1_reduction_helped_rate'])} | {_pct(row['tp1_reduction_cost_rate'])} | "
            f"{_num(row['avg_post_tp1_continuation_r'])} | {_num(row['avg_tp1_reduction_delta_r'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame, report: dict[str, Any], output_prefix: Path) -> TP1OutputPaths:
    prefix = Path(output_prefix)
    paths = TP1OutputPaths(
        details_csv=prefix.with_name(f"{prefix.name}_details.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        by_exit_reason_csv=prefix.with_name(f"{prefix.name}_by_exit_reason.csv"),
        by_first_rr_bucket_csv=prefix.with_name(f"{prefix.name}_by_first_rr_bucket.csv"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.details_csv, rows)
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    _write_csv(paths.by_exit_reason_csv, report["by_exit_reason"])
    _write_csv(paths.by_first_rr_bucket_csv, report["by_first_rr_bucket"])
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether TP1 partial profit taking helps trend trades")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--phase23-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw = collect_trade_rows_from_cache(args.phase23_cache_dir, years=_parse_years(args.years))
    rows = prepare_tp1_rows(raw)
    report = analyze_tp1_effect(rows)
    paths = write_outputs(rows, report, args.output_prefix)
    for path in asdict(paths).values():
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
