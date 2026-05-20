from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from compare_trend_backtest_params import _config_for_combo
from run_account_trend_param_sweep import (
    _collect_combo_candidates,
    _config_with_fundamental_mode,
    _resolved_fundamental_mode,
    load_config,
    load_trend_case_frames_once,
)


FIRST_RR_FIELD = "entry_rr"
SECOND_RR_FIELD = "entry_admission_rr"
MEDIUM_QUALITY_FIELD = "medium_term_quality_score"
FIRST_RR_THRESHOLDS = (1.0, 1.5, 2.0)
SECOND_RR_THRESHOLDS = (2.0, 2.5, 3.0)


@dataclass(frozen=True)
class RROutputPaths:
    details_csv: Path
    score_diagnostics_csv: Path
    first_rr_buckets_csv: Path
    second_rr_buckets_csv: Path
    cross_buckets_csv: Path
    medium_quality_conditioned_csv: Path
    yearly_score_diagnostics_csv: Path
    summary_json: Path
    report_md: Path


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _bucket(value: Any, thresholds: tuple[float, ...]) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "missing"
    previous: float | None = None
    for threshold in thresholds:
        if number < threshold:
            return f"lt_{threshold:.1f}" if previous is None else f"{previous:.1f}_{threshold:.1f}"
        previous = threshold
    return f"{thresholds[-1]:.1f}_plus"


def _bucket_label(value: Any, *, field: str) -> str:
    thresholds = FIRST_RR_THRESHOLDS if field == FIRST_RR_FIELD else SECOND_RR_THRESHOLDS
    return _bucket(value, thresholds)


def _score_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").dropna()


def pairwise_auc(positive_scores: Iterable[Any], negative_scores: Iterable[Any]) -> float | None:
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


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def spearman(values: Iterable[Any], targets: Iterable[Any]) -> float | None:
    pairs = [
        (_finite_float(value), _finite_float(target))
        for value, target in zip(values, targets)
        if math.isfinite(_finite_float(value)) and math.isfinite(_finite_float(target))
    ]
    if len(pairs) < 3:
        return None
    x = [item[0] for item in pairs]
    y = [item[1] for item in pairs]
    rx = _rankdata(x)
    ry = _rankdata(y)
    avg_x = sum(rx) / len(rx)
    avg_y = sum(ry) / len(ry)
    cov = sum((a - avg_x) * (b - avg_y) for a, b in zip(rx, ry))
    var_x = sum((a - avg_x) ** 2 for a in rx)
    var_y = sum((b - avg_y) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return None
    return float(cov / math.sqrt(var_x * var_y))


def prepare_rr_rows(
    frame: pd.DataFrame,
    *,
    early_stop_days: int = 3,
    long_hold_days: int = 14,
) -> pd.DataFrame:
    out = frame.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"])
    out["planned_exit_time"] = pd.to_datetime(out["planned_exit_time"])
    out["hold_days"] = (out["planned_exit_time"] - out["entry_time"]).dt.total_seconds() / 86400.0
    for field in (FIRST_RR_FIELD, SECOND_RR_FIELD, MEDIUM_QUALITY_FIELD, "pnl_ratio_price", "phase2_score"):
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors="coerce")
    out["tp1_hit"] = _as_bool_series(out["tp1_hit"]) if "tp1_hit" in out.columns else False
    out["is_early_stop"] = (
        out["planned_exit_reason"].astype(str).eq("stop")
        & (out["hold_days"] < float(early_stop_days))
    )
    out["is_long_hold_winner"] = (
        (out["hold_days"] >= float(long_hold_days))
        & (pd.to_numeric(out.get("pnl_ratio_price", 0.0), errors="coerce") > 0)
    )
    out["is_tp2_winner"] = (
        out["planned_exit_reason"].astype(str).eq("tp2")
        & (pd.to_numeric(out.get("pnl_ratio_price", 0.0), errors="coerce") > 0)
    )
    out["entry_rr_bucket"] = out[FIRST_RR_FIELD].apply(lambda value: _bucket_label(value, field=FIRST_RR_FIELD))
    out["entry_admission_rr_bucket"] = out[SECOND_RR_FIELD].apply(
        lambda value: _bucket_label(value, field=SECOND_RR_FIELD)
    )
    return out


def _rate(frame: pd.DataFrame, field: str) -> float:
    return float(frame[field].mean()) if len(frame) else 0.0


def _avg(frame: pd.DataFrame, field: str) -> float | None:
    values = _score_series(frame, field)
    if values.empty:
        return None
    return float(values.mean())


def _bucket_row(frame: pd.DataFrame, *, bucket_field: str, bucket: str) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "trade_count": int(len(frame)),
        "early_stop_count": int(frame["is_early_stop"].sum()),
        "early_stop_rate": _rate(frame, "is_early_stop"),
        "tp1_hit_rate": _rate(frame, "tp1_hit"),
        "long_hold_winner_count": int(frame["is_long_hold_winner"].sum()),
        "long_hold_winner_rate": _rate(frame, "is_long_hold_winner"),
        "tp2_winner_count": int(frame["is_tp2_winner"].sum()),
        "tp2_winner_rate": _rate(frame, "is_tp2_winner"),
        "avg_pnl_ratio_price": _avg(frame, "pnl_ratio_price"),
        "avg_entry_rr": _avg(frame, FIRST_RR_FIELD),
        "avg_entry_admission_rr": _avg(frame, SECOND_RR_FIELD),
        bucket_field: bucket,
    }


def rr_bucket_rows(rows: pd.DataFrame, *, field: str) -> list[dict[str, Any]]:
    bucket_field = "entry_rr_bucket" if field == FIRST_RR_FIELD else "entry_admission_rr_bucket"
    out: list[dict[str, Any]] = []
    order = ["missing", "lt_1.0", "1.0_1.5", "1.5_2.0", "2.0_plus"]
    if field == SECOND_RR_FIELD:
        order = ["missing", "lt_2.0", "2.0_2.5", "2.5_3.0", "3.0_plus"]
    for bucket in order:
        group = rows[rows[bucket_field] == bucket]
        if group.empty:
            continue
        out.append(_bucket_row(group, bucket_field=bucket_field, bucket=bucket))
    return out


def cross_rr_bucket_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped = rows.groupby(["entry_rr_bucket", "entry_admission_rr_bucket"], dropna=False)
    for (first_bucket, second_bucket), group in grouped:
        row = _bucket_row(group, bucket_field="entry_rr_bucket", bucket=str(first_bucket))
        row["entry_admission_rr_bucket"] = str(second_bucket)
        out.append(row)
    return sorted(out, key=lambda row: (row["entry_rr_bucket"], row["entry_admission_rr_bucket"]))


def _score_diagnostics(rows: pd.DataFrame, field: str) -> dict[str, Any]:
    early = rows[rows["is_early_stop"]]
    survived = rows[~rows["is_early_stop"]]
    tp1 = rows[rows["tp1_hit"]]
    no_tp1 = rows[~rows["tp1_hit"]]
    long_winner = rows[rows["is_long_hold_winner"]]
    not_long_winner = rows[~rows["is_long_hold_winner"]]
    tp2_winner = rows[rows["is_tp2_winner"]]
    not_tp2_winner = rows[~rows["is_tp2_winner"]]
    return {
        "field": field,
        "score_rows": int(len(_score_series(rows, field))),
        "survive_early_auc": pairwise_auc(survived[field], early[field]) if field in rows else None,
        "tp1_hit_auc": pairwise_auc(tp1[field], no_tp1[field]) if field in rows else None,
        "long_hold_winner_auc": pairwise_auc(long_winner[field], not_long_winner[field]) if field in rows else None,
        "tp2_winner_auc": pairwise_auc(tp2_winner[field], not_tp2_winner[field]) if field in rows else None,
        "spearman_pnl_ratio": spearman(rows[field], rows["pnl_ratio_price"]) if field in rows else None,
        "spearman_hold_days": spearman(rows[field], rows["hold_days"]) if field in rows else None,
        "early_stop_mean": _avg(early, field),
        "tp1_hit_mean": _avg(tp1, field),
        "long_hold_winner_mean": _avg(long_winner, field),
        "tp2_winner_mean": _avg(tp2_winner, field),
    }


def medium_quality_conditioned_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    bands = [
        ("lt_50", rows[MEDIUM_QUALITY_FIELD] < 50),
        ("50_60", (rows[MEDIUM_QUALITY_FIELD] >= 50) & (rows[MEDIUM_QUALITY_FIELD] < 60)),
        ("60_plus", rows[MEDIUM_QUALITY_FIELD] >= 60),
    ]
    out: list[dict[str, Any]] = []
    for band, mask in bands:
        group = rows[mask].copy()
        if group.empty:
            continue
        first = _score_diagnostics(group, FIRST_RR_FIELD)
        second = _score_diagnostics(group, SECOND_RR_FIELD)
        out.append(
            {
                "medium_quality_band": band,
                "trade_count": int(len(group)),
                "early_stop_rate": _rate(group, "is_early_stop"),
                "tp1_hit_rate": _rate(group, "tp1_hit"),
                "long_hold_winner_rate": _rate(group, "is_long_hold_winner"),
                "tp2_winner_rate": _rate(group, "is_tp2_winner"),
                "entry_rr_tp1_auc": first["tp1_hit_auc"],
                "entry_rr_survive_early_auc": first["survive_early_auc"],
                "entry_admission_rr_long_hold_auc": second["long_hold_winner_auc"],
                "entry_admission_rr_tp2_auc": second["tp2_winner_auc"],
            }
        )
    return out


def yearly_score_diagnostic_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if "year" not in rows.columns:
        return []
    out: list[dict[str, Any]] = []
    for year, group in rows.groupby("year", dropna=False):
        for field in (FIRST_RR_FIELD, SECOND_RR_FIELD, MEDIUM_QUALITY_FIELD, "phase2_score"):
            if field not in group.columns:
                continue
            row = _score_diagnostics(group, field)
            row["year"] = int(year)
            out.append(row)
    return out


def analyze_rr_guidance(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(rows)),
        "early_stop_count": int(rows["is_early_stop"].sum()),
        "tp1_hit_count": int(rows["tp1_hit"].sum()),
        "long_hold_winner_count": int(rows["is_long_hold_winner"].sum()),
        "tp2_winner_count": int(rows["is_tp2_winner"].sum()),
        "score_diagnostics": {
            FIRST_RR_FIELD: _score_diagnostics(rows, FIRST_RR_FIELD),
            SECOND_RR_FIELD: _score_diagnostics(rows, SECOND_RR_FIELD),
            MEDIUM_QUALITY_FIELD: _score_diagnostics(rows, MEDIUM_QUALITY_FIELD),
            "phase2_score": _score_diagnostics(rows, "phase2_score") if "phase2_score" in rows else {},
        },
        "first_rr_buckets": rr_bucket_rows(rows, field=FIRST_RR_FIELD),
        "second_rr_buckets": rr_bucket_rows(rows, field=SECOND_RR_FIELD),
        "cross_buckets": cross_rr_bucket_rows(rows),
        "medium_quality_conditioned": medium_quality_conditioned_rows(rows),
        "yearly_score_diagnostics": yearly_score_diagnostic_rows(rows),
    }


def candidates_to_frame(candidates: Iterable[Any], *, year: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": int(year),
                "trade_id": candidate.trade_id,
                "symbol": candidate.symbol,
                "name": candidate.name,
                "direction": candidate.direction,
                "entry_time": candidate.entry_time,
                "planned_exit_time": candidate.planned_exit_time,
                "planned_exit_reason": candidate.planned_exit_reason,
                "pnl_ratio_price": candidate.pnl_ratio_price,
                "tp1_hit": candidate.tp1_hit,
                "entry_rr": candidate.entry_rr,
                "entry_admission_rr": candidate.entry_admission_rr,
                "medium_term_quality_score": candidate.medium_term_quality_score,
                "medium_term_entry_location_score": candidate.medium_term_entry_location_score,
                "entry_adverse_deviation_r": candidate.entry_adverse_deviation_r,
                "phase2_score": candidate.phase2_score,
            }
            for candidate in candidates
        ]
    )


def legacy_phase23_config(config: dict[str, Any]) -> dict[str, Any]:
    return _config_for_combo(
        config,
        min_stop_atr=None,
        max_entry_adverse_r=None,
        min_first_target_rr=None,
        second_target_rr_relax_threshold=None,
        relaxed_first_target_rr=None,
        entry_confirmation_bars=None,
        initial_stop_grace_bars=None,
    )


def _parse_years(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def collect_candidate_rows(
    *,
    years: Iterable[int],
    case_group: str,
    phase23_cache_dir: Path,
    case_limit: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    loaded_config = load_config()
    config = _config_with_fundamental_mode(loaded_config, _resolved_fundamental_mode(loaded_config, None))
    config = legacy_phase23_config(config)
    frames: list[pd.DataFrame] = []
    for year in years:
        loaded_cases, _minute_bars = load_trend_case_frames_once(
            case_group=case_group,
            year=int(year),
            config=config,
            case_limit=case_limit,
            progress=progress,
        )
        candidates, diagnostics = _collect_combo_candidates(
            loaded_cases=loaded_cases,
            year=int(year),
            config=config,
            phase23_cache_dir=phase23_cache_dir,
            require_phase23_cache=True,
        )
        if progress:
            print(
                f"year {year}: candidates={len(candidates)} "
                f"cache_hits={diagnostics['phase23_cache_hits']} cache_misses={diagnostics['phase23_cache_misses']}",
                flush=True,
            )
        frames.append(candidates_to_frame(candidates, year=int(year)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def _num(value: Any, digits: int = 3) -> str:
    number = _finite_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _diagnostic_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report["score_diagnostics"].values())


def write_outputs(rows: pd.DataFrame, report: dict[str, Any], output_prefix: Path) -> RROutputPaths:
    prefix = Path(output_prefix)
    paths = RROutputPaths(
        details_csv=prefix.with_name(f"{prefix.name}_details.csv"),
        score_diagnostics_csv=prefix.with_name(f"{prefix.name}_score_diagnostics.csv"),
        first_rr_buckets_csv=prefix.with_name(f"{prefix.name}_first_rr_buckets.csv"),
        second_rr_buckets_csv=prefix.with_name(f"{prefix.name}_second_rr_buckets.csv"),
        cross_buckets_csv=prefix.with_name(f"{prefix.name}_cross_buckets.csv"),
        medium_quality_conditioned_csv=prefix.with_name(f"{prefix.name}_medium_quality_conditioned.csv"),
        yearly_score_diagnostics_csv=prefix.with_name(f"{prefix.name}_yearly_score_diagnostics.csv"),
        summary_json=prefix.with_name(f"{prefix.name}_summary.json"),
        report_md=prefix.with_name(f"{prefix.name}_report.md"),
    )
    _write_csv(paths.details_csv, rows)
    _write_csv(paths.score_diagnostics_csv, _diagnostic_rows(report))
    _write_csv(paths.first_rr_buckets_csv, report["first_rr_buckets"])
    _write_csv(paths.second_rr_buckets_csv, report["second_rr_buckets"])
    _write_csv(paths.cross_buckets_csv, report["cross_buckets"])
    _write_csv(paths.medium_quality_conditioned_csv, report["medium_quality_conditioned"])
    _write_csv(paths.yearly_score_diagnostics_csv, report["yearly_score_diagnostics"])
    paths.summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")
    return paths


def render_markdown(report: dict[str, Any]) -> str:
    first = report["score_diagnostics"][FIRST_RR_FIELD]
    second = report["score_diagnostics"][SECOND_RR_FIELD]
    medium = report["score_diagnostics"][MEDIUM_QUALITY_FIELD]
    lines = [
        "# RR 指导力验证",
        "",
        "验证目的：判断第一目标 RR 是否适合指导早期存活/TP1，第二目标 RR 是否适合指导长持仓/TP2，并观察它们在中期质量分层后是否还有额外解释力。",
        "",
        "## 总览",
        "",
        f"- 候选交易数：{report['rows']}",
        f"- 早止损数：{report['early_stop_count']}",
        f"- TP1 命中数：{report['tp1_hit_count']}",
        f"- 长持仓盈利数：{report['long_hold_winner_count']}",
        f"- TP2 盈利数：{report['tp2_winner_count']}",
        "",
        "## 核心 AUC",
        "",
        "| 指标 | 早期存活 AUC | TP1 AUC | 长持仓盈利 AUC | TP2 盈利 AUC | PnL Spearman |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| 第一 RR | {_num(first['survive_early_auc'])} | {_num(first['tp1_hit_auc'])} | "
            f"{_num(first['long_hold_winner_auc'])} | {_num(first['tp2_winner_auc'])} | {_num(first['spearman_pnl_ratio'])} |"
        ),
        (
            f"| 第二 RR | {_num(second['survive_early_auc'])} | {_num(second['tp1_hit_auc'])} | "
            f"{_num(second['long_hold_winner_auc'])} | {_num(second['tp2_winner_auc'])} | {_num(second['spearman_pnl_ratio'])} |"
        ),
        (
            f"| 中期质量 | {_num(medium['survive_early_auc'])} | {_num(medium['tp1_hit_auc'])} | "
            f"{_num(medium['long_hold_winner_auc'])} | {_num(medium['tp2_winner_auc'])} | {_num(medium['spearman_pnl_ratio'])} |"
        ),
        "",
        "## 第一 RR 分层",
        "",
        "| 第一 RR 桶 | 笔数 | 早止损率 | TP1 命中率 | 长持仓盈利率 | TP2 盈利率 | 平均收益率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["first_rr_buckets"]:
        lines.append(
            f"| {row['entry_rr_bucket']} | {row['trade_count']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['long_hold_winner_rate'])} | "
            f"{_pct(row['tp2_winner_rate'])} | {_pct(row['avg_pnl_ratio_price'])} |"
        )
    lines.extend(
        [
            "",
            "## 第二 RR 分层",
            "",
            "| 第二 RR 桶 | 笔数 | 早止损率 | TP1 命中率 | 长持仓盈利率 | TP2 盈利率 | 平均收益率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["second_rr_buckets"]:
        lines.append(
            f"| {row['entry_admission_rr_bucket']} | {row['trade_count']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_pct(row['long_hold_winner_rate'])} | "
            f"{_pct(row['tp2_winner_rate'])} | {_pct(row['avg_pnl_ratio_price'])} |"
        )
    lines.extend(
        [
            "",
            "## 中期质量内的 RR 指导力",
            "",
            "| 中期质量 | 笔数 | 早止损率 | TP1 命中率 | 第一 RR TP1 AUC | 第二 RR 长持仓 AUC | 第二 RR TP2 AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["medium_quality_conditioned"]:
        lines.append(
            f"| {row['medium_quality_band']} | {row['trade_count']} | {_pct(row['early_stop_rate'])} | "
            f"{_pct(row['tp1_hit_rate'])} | {_num(row['entry_rr_tp1_auc'])} | "
            f"{_num(row['entry_admission_rr_long_hold_auc'])} | {_num(row['entry_admission_rr_tp2_auc'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether first/second RR guide trend trade outcomes")
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--case-group", default="long_trend_core")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--phase23-cache-dir", type=Path, default=Path("data/cache/backtest/phase23_task4_split_v1"))
    parser.add_argument("--early-stop-days", type=int, default=3)
    parser.add_argument("--long-hold-days", type=int, default=14)
    parser.add_argument("--output-prefix", type=Path, default=Path("data/reports/backtest/rr_guidance_2022_2025"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    years = _parse_years(args.years)
    raw_rows = collect_candidate_rows(
        years=years,
        case_group=args.case_group,
        phase23_cache_dir=args.phase23_cache_dir,
        case_limit=args.case_limit,
        progress=not args.quiet,
    )
    rows = prepare_rr_rows(
        raw_rows,
        early_stop_days=args.early_stop_days,
        long_hold_days=args.long_hold_days,
    )
    report = analyze_rr_guidance(rows)
    paths = write_outputs(rows, report, args.output_prefix)
    for path in asdict(paths).values():
        print(f"wrote: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
