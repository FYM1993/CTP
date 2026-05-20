from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_YEARS = (2022, 2023, 2024, 2025)
DEFAULT_REPORT_DIR = Path("data/reports/backtest")
DEFAULT_CACHE_DIR = Path("tmp/phase23_rr_sweep_cache")
DEFAULT_OUTPUT_PREFIX = DEFAULT_REPORT_DIR / "trend_account_2022_2025_rr_attribution"
LOW_FIRST_RR = 1.5
HIGH_SECOND_RR = 2.5
EARLY_STOP_DAYS = 3.0


DETAIL_FIELDS = [
    "year",
    "trade_key",
    "symbol",
    "name",
    "direction",
    "entry_time",
    "comparison",
    "attribution",
    "strict_skip_reason",
    "legacy_lots",
    "strict_lots",
    "legacy_exit_reason",
    "strict_exit_reason",
    "legacy_net_pnl",
    "strict_net_pnl",
    "net_pnl_delta",
    "phase2_score",
    "entry_rr",
    "entry_admission_rr",
    "low_first_high_second",
    "early_stop",
    "hold_days",
    "adverse_entry_deviation_r",
    "planned_entry_ref",
    "planned_stop",
    "planned_tp1",
    "planned_tp2",
    "entry_signal_type",
    "entry_signal_detail",
    "candidate_trade_id",
]


SUMMARY_FIELDS = [
    "year",
    "legacy_trades",
    "strict_trades",
    "common_trades",
    "legacy_only_trades",
    "strict_only_trades",
    "legacy_only_winner_trades",
    "legacy_only_winner_net_pnl",
    "legacy_only_loser_trades",
    "legacy_only_loser_net_pnl",
    "legacy_only_low_rr_winner_trades",
    "legacy_only_low_rr_winner_net_pnl",
    "legacy_only_low_rr_loser_trades",
    "legacy_only_low_rr_loser_net_pnl",
    "legacy_only_path_winner_trades",
    "legacy_only_path_winner_net_pnl",
    "legacy_only_path_loser_trades",
    "legacy_only_path_loser_net_pnl",
    "legacy_only_net_pnl",
    "strict_only_net_pnl",
    "common_net_pnl_delta",
    "strict_minus_legacy_net_pnl_reconciled",
    "low_first_high_second_trades",
    "low_first_high_second_winner_trades",
    "low_first_high_second_loser_trades",
    "low_first_high_second_early_stop_trades",
    "low_first_high_second_net_pnl",
    "low_first_high_second_avg_adverse_entry_deviation_r",
    "metadata_trade_rows",
    "metadata_rows_with_entry_rr",
    "metadata_coverage_pct",
]


def _trade_key(row: dict[str, Any]) -> str:
    return f"{row.get('symbol', '')}|{row.get('direction', '')}|{row.get('entry_time', '')}"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    try:
        if value is None or value == "":
            return None
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _hold_days(row: dict[str, Any]) -> float:
    entry = _parse_dt(row.get("entry_time"))
    exit_ = _parse_dt(row.get("actual_exit_time") or row.get("planned_exit_time"))
    if entry is None or exit_ is None:
        return 0.0
    return max((exit_ - entry).total_seconds() / 86400.0, 0.0)


def _is_early_stop(row: dict[str, Any]) -> bool:
    return str(row.get("actual_exit_reason") or "") == "stop" and _hold_days(row) < EARLY_STOP_DAYS


def _is_low_first_high_second(row: dict[str, Any]) -> bool:
    entry_rr = _as_optional_float(row.get("entry_rr"))
    second_rr = _as_optional_float(row.get("entry_admission_rr"))
    return entry_rr is not None and second_rr is not None and entry_rr < LOW_FIRST_RR and second_rr >= HIGH_SECOND_RR


def _adverse_entry_deviation_r(row: dict[str, Any]) -> float | str:
    deviation = _as_optional_float(row.get("entry_trigger_deviation"))
    if deviation is None:
        return ""
    direction = str(row.get("direction") or "")
    entry = _as_optional_float(row.get("entry_price"))
    stop = _as_optional_float(row.get("planned_stop") or row.get("initial_stop_price"))
    if entry is None or stop is None:
        return ""
    risk = abs(entry - stop)
    if risk <= 0:
        return ""
    signed_adverse = deviation if direction == "long" else -deviation
    return float(signed_adverse / risk)


def _index_by_trade_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out.setdefault(_trade_key(row), row)
    return out


def _decision_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _index_by_trade_key(rows)


def _candidate_meta_from_cache_trade(trade: dict[str, Any]) -> dict[str, Any]:
    meta = dict(trade.get("meta") or {})
    return {
        "candidate_trade_id": trade.get("trade_id", ""),
        "planned_entry_ref": meta.get("planned_entry_ref", ""),
        "planned_stop": meta.get("planned_stop") or meta.get("initial_stop_price") or "",
        "planned_tp1": meta.get("planned_tp1") or meta.get("initial_tp1_price") or "",
        "planned_tp2": meta.get("planned_tp2") or meta.get("initial_tp2_price") or "",
        "entry_rr": meta.get("entry_rr") or meta.get("execution_rr") or meta.get("rr") or "",
        "entry_admission_rr": (
            meta.get("entry_admission_rr")
            or meta.get("execution_admission_rr")
            or meta.get("admission_rr")
            or ""
        ),
        "entry_trigger_deviation": meta.get("entry_trigger_deviation", ""),
        "entry_trigger_deviation_pct": meta.get("entry_trigger_deviation_pct", ""),
        "entry_signal_type": meta.get("entry_signal_type", ""),
        "entry_signal_detail": meta.get("entry_signal_detail", ""),
        "initial_stop_price": meta.get("initial_stop_price", ""),
        "phase2_score": meta.get("phase2_score", ""),
        "_cache_exit_time": trade.get("exit_time", ""),
        "_cache_exit_reason": trade.get("exit_reason", ""),
    }


def load_phase23_meta_index(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not cache_dir.exists():
        return {}
    for path in sorted(cache_dir.glob("*phase23_cache.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for trade in ((payload.get("result") or {}).get("trades") or []):
            key = _trade_key(trade)
            meta = _candidate_meta_from_cache_trade(trade)
            if meta not in index[key]:
                index[key].append(meta)
    return dict(index)


def _choose_meta(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    planned_exit_time = str(row.get("planned_exit_time") or "")
    actual_exit_time = str(row.get("actual_exit_time") or "")
    for candidate in candidates:
        if planned_exit_time and str(candidate.get("_cache_exit_time") or "") == planned_exit_time:
            return candidate
    for candidate in candidates:
        if actual_exit_time and str(candidate.get("_cache_exit_time") or "") == actual_exit_time:
            return candidate
    return candidates[0]


def enrich_with_phase23_meta(
    rows: list[dict[str, Any]],
    meta_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        meta = _choose_meta(out, meta_index.get(_trade_key(out), []))
        if meta:
            for key, value in meta.items():
                if key.startswith("_"):
                    continue
                if out.get(key) in (None, ""):
                    out[key] = value
        enriched.append(out)
    return enriched


def _row_name(*rows: dict[str, Any] | None) -> str:
    for row in rows:
        if row and row.get("name"):
            return str(row["name"])
    return ""


def _base_detail_row(
    *,
    year: int,
    key: str,
    legacy: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    comparison: str,
    attribution: str,
) -> dict[str, Any]:
    source = legacy or strict or decision or {}
    source_for_meta = legacy or strict or {}
    legacy_pnl = _as_float(legacy.get("net_pnl")) if legacy else 0.0
    strict_pnl = _as_float(strict.get("net_pnl")) if strict else 0.0
    if legacy and strict:
        pnl_delta: float | str = strict_pnl - legacy_pnl
    elif legacy:
        pnl_delta = -legacy_pnl
    else:
        pnl_delta = strict_pnl
    return {
        "year": int(year),
        "trade_key": key,
        "symbol": source.get("symbol", ""),
        "name": _row_name(legacy, strict, decision),
        "direction": source.get("direction", ""),
        "entry_time": source.get("entry_time", ""),
        "comparison": comparison,
        "attribution": attribution,
        "strict_skip_reason": decision.get("skip_reason", "") if decision else "",
        "legacy_lots": legacy.get("lots", "") if legacy else "",
        "strict_lots": strict.get("lots", "") if strict else "",
        "legacy_exit_reason": legacy.get("actual_exit_reason", "") if legacy else "",
        "strict_exit_reason": strict.get("actual_exit_reason", "") if strict else "",
        "legacy_net_pnl": legacy.get("net_pnl", "") if legacy else "",
        "strict_net_pnl": strict.get("net_pnl", "") if strict else "",
        "net_pnl_delta": pnl_delta,
        "phase2_score": source_for_meta.get("phase2_score", ""),
        "entry_rr": source_for_meta.get("entry_rr", ""),
        "entry_admission_rr": source_for_meta.get("entry_admission_rr", ""),
        "low_first_high_second": _is_low_first_high_second(source_for_meta),
        "early_stop": _is_early_stop(source_for_meta),
        "hold_days": _hold_days(source_for_meta),
        "adverse_entry_deviation_r": _adverse_entry_deviation_r(source_for_meta),
        "planned_entry_ref": source_for_meta.get("planned_entry_ref", ""),
        "planned_stop": source_for_meta.get("planned_stop", ""),
        "planned_tp1": source_for_meta.get("planned_tp1", ""),
        "planned_tp2": source_for_meta.get("planned_tp2", ""),
        "entry_signal_type": source_for_meta.get("entry_signal_type", ""),
        "entry_signal_detail": source_for_meta.get("entry_signal_detail", ""),
        "candidate_trade_id": source_for_meta.get("candidate_trade_id", ""),
    }


def analyze_run_pair(
    *,
    year: int,
    legacy_trades: list[dict[str, Any]],
    strict_trades: list[dict[str, Any]],
    strict_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy_by_key = _index_by_trade_key(legacy_trades)
    strict_by_key = _index_by_trade_key(strict_trades)
    strict_decision_by_key = _decision_index(strict_decisions)

    detail: list[dict[str, Any]] = []
    common_delta = 0.0
    strict_only_net = 0.0
    legacy_only_net = 0.0
    legacy_only_winner_net = 0.0
    legacy_only_loser_net = 0.0
    legacy_only_winners = 0
    legacy_only_losers = 0
    legacy_only_low_rr_winner_net = 0.0
    legacy_only_low_rr_loser_net = 0.0
    legacy_only_path_winner_net = 0.0
    legacy_only_path_loser_net = 0.0
    legacy_only_low_rr_winners = 0
    legacy_only_low_rr_losers = 0
    legacy_only_path_winners = 0
    legacy_only_path_losers = 0

    for key in sorted(set(legacy_by_key) | set(strict_by_key)):
        legacy = legacy_by_key.get(key)
        strict = strict_by_key.get(key)
        decision = strict_decision_by_key.get(key)
        if legacy and strict:
            delta = _as_float(strict.get("net_pnl")) - _as_float(legacy.get("net_pnl"))
            common_delta += delta
            attribution = "共同交易仓位/退出差异" if abs(delta) > 1e-9 else "共同交易无差异"
            detail.append(
                _base_detail_row(
                    year=year,
                    key=key,
                    legacy=legacy,
                    strict=strict,
                    decision=decision,
                    comparison="common",
                    attribution=attribution,
                )
            )
        elif legacy:
            pnl = _as_float(legacy.get("net_pnl"))
            legacy_only_net += pnl
            low_rr = _as_float(legacy.get("entry_rr"), 999.0) < LOW_FIRST_RR
            if pnl > 0:
                legacy_only_winners += 1
                legacy_only_winner_net += pnl
                if low_rr:
                    legacy_only_low_rr_winners += 1
                    legacy_only_low_rr_winner_net += pnl
                    attribution = "低RR错杀趋势机会"
                else:
                    legacy_only_path_winners += 1
                    legacy_only_path_winner_net += pnl
                    attribution = "路径错过盈利机会"
            elif pnl < 0:
                legacy_only_losers += 1
                legacy_only_loser_net += pnl
                if low_rr:
                    legacy_only_low_rr_losers += 1
                    legacy_only_low_rr_loser_net += pnl
                    attribution = "低RR过滤有效"
                else:
                    legacy_only_path_losers += 1
                    legacy_only_path_loser_net += pnl
                    attribution = "路径避开亏损"
            else:
                attribution = "过滤中性"
            detail.append(
                _base_detail_row(
                    year=year,
                    key=key,
                    legacy=legacy,
                    strict=None,
                    decision=decision,
                    comparison="legacy_only",
                    attribution=attribution,
                )
            )
        elif strict:
            pnl = _as_float(strict.get("net_pnl"))
            strict_only_net += pnl
            attribution = "资金释放后新增盈利" if pnl > 0 else "资金释放后新增亏损" if pnl < 0 else "资金释放后新增中性"
            detail.append(
                _base_detail_row(
                    year=year,
                    key=key,
                    legacy=None,
                    strict=strict,
                    decision=decision,
                    comparison="strict_only",
                    attribution=attribution,
                )
            )

    low_high = [row for row in legacy_trades if _is_low_first_high_second(row)]
    low_high_adverse = [
        float(value)
        for value in (_adverse_entry_deviation_r(row) for row in low_high)
        if isinstance(value, float)
    ]
    metadata_rows = len(legacy_trades) + len(strict_trades)
    metadata_with_rr = sum(1 for row in legacy_trades + strict_trades if _as_optional_float(row.get("entry_rr")) is not None)
    summary = {
        "year": int(year),
        "legacy_trades": len(legacy_trades),
        "strict_trades": len(strict_trades),
        "common_trades": sum(1 for row in detail if row["comparison"] == "common"),
        "legacy_only_trades": sum(1 for row in detail if row["comparison"] == "legacy_only"),
        "strict_only_trades": sum(1 for row in detail if row["comparison"] == "strict_only"),
        "legacy_only_winner_trades": legacy_only_winners,
        "legacy_only_winner_net_pnl": legacy_only_winner_net,
        "legacy_only_loser_trades": legacy_only_losers,
        "legacy_only_loser_net_pnl": legacy_only_loser_net,
        "legacy_only_low_rr_winner_trades": legacy_only_low_rr_winners,
        "legacy_only_low_rr_winner_net_pnl": legacy_only_low_rr_winner_net,
        "legacy_only_low_rr_loser_trades": legacy_only_low_rr_losers,
        "legacy_only_low_rr_loser_net_pnl": legacy_only_low_rr_loser_net,
        "legacy_only_path_winner_trades": legacy_only_path_winners,
        "legacy_only_path_winner_net_pnl": legacy_only_path_winner_net,
        "legacy_only_path_loser_trades": legacy_only_path_losers,
        "legacy_only_path_loser_net_pnl": legacy_only_path_loser_net,
        "legacy_only_net_pnl": legacy_only_net,
        "strict_only_net_pnl": strict_only_net,
        "common_net_pnl_delta": common_delta,
        "strict_minus_legacy_net_pnl_reconciled": strict_only_net - legacy_only_net + common_delta,
        "low_first_high_second_trades": len(low_high),
        "low_first_high_second_winner_trades": sum(1 for row in low_high if _as_float(row.get("net_pnl")) > 0),
        "low_first_high_second_loser_trades": sum(1 for row in low_high if _as_float(row.get("net_pnl")) < 0),
        "low_first_high_second_early_stop_trades": sum(1 for row in low_high if _is_early_stop(row)),
        "low_first_high_second_net_pnl": sum(_as_float(row.get("net_pnl")) for row in low_high),
        "low_first_high_second_avg_adverse_entry_deviation_r": (
            sum(low_high_adverse) / len(low_high_adverse) if low_high_adverse else 0.0
        ),
        "metadata_trade_rows": metadata_rows,
        "metadata_rows_with_entry_rr": metadata_with_rr,
        "metadata_coverage_pct": metadata_with_rr / metadata_rows if metadata_rows else 0.0,
    }
    return detail, summary


def _paths_for_year(report_dir: Path, year: int) -> tuple[Path, Path, Path]:
    stem = report_dir / f"trend_account_{year}_phase2score_margin30_risk_0p015_rr_legacy_vs_1p5"
    return (
        stem.with_name(f"{stem.name}_combo_001_trades.csv"),
        stem.with_name(f"{stem.name}_combo_002_trades.csv"),
        stem.with_name(f"{stem.name}_combo_002_decisions.csv"),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_money(value: Any) -> str:
    return f"{_as_float(value):,.0f}"


def _fmt_pct(value: Any) -> str:
    return f"{_as_float(value):.1%}"


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    attribution: str,
    sort_field: str,
    reverse: bool,
    limit: int = 8,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("attribution") == attribution]
    return sorted(selected, key=lambda row: _as_float(row.get(sort_field)), reverse=reverse)[:limit]


def _render_top_table(rows: list[dict[str, Any]], pnl_field: str) -> list[str]:
    if not rows:
        return ["- 无"]
    lines = ["| 年份 | 品种 | 方向 | 入场时间 | 收益 | 第一RR | 第二RR | 退出 |", "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        exit_reason = row.get("legacy_exit_reason") or row.get("strict_exit_reason") or ""
        pnl = row.get(pnl_field)
        lines.append(
            "| {year} | {symbol} | {direction} | {entry_time} | {pnl} | {rr:.2f} | {second:.2f} | {exit_reason} |".format(
                year=row.get("year", ""),
                symbol=row.get("symbol", ""),
                direction=row.get("direction", ""),
                entry_time=row.get("entry_time", ""),
                pnl=_fmt_money(pnl),
                rr=_as_float(row.get("entry_rr")),
                second=_as_float(row.get("entry_admission_rr")),
                exit_reason=exit_reason,
            )
        )
    return lines


def render_markdown(summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# 趋势账户 1.5 RR 过滤归因（2022-2025）",
        "",
        "口径：对比 legacy 趋势账户与第一目标 RR>=1.5 的账户结果。正数代表严格规则相对 legacy 增利，负数代表减利。",
        "",
        "## 年度收益差拆解",
        "",
        "| 年份 | 低RR错杀 | 低RR避亏 | 路径错过盈利 | 路径避开亏损 | 严格新增单 | 共同单差异 | 严格-legacy合计 | 低一高二单 | 元数据覆盖 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {year} | {low_miss} / {low_miss_pnl} | {low_avoid} / {low_avoid_pnl} | {path_miss} / {path_miss_pnl} | {path_avoid} / {path_avoid_pnl} | {new} / {new_pnl} | {common} | {total} | {lh} / {lh_pnl} | {coverage} |".format(
                year=row["year"],
                low_miss=row["legacy_only_low_rr_winner_trades"],
                low_miss_pnl=_fmt_money(row["legacy_only_low_rr_winner_net_pnl"]),
                low_avoid=row["legacy_only_low_rr_loser_trades"],
                low_avoid_pnl=_fmt_money(row["legacy_only_low_rr_loser_net_pnl"]),
                path_miss=row["legacy_only_path_winner_trades"],
                path_miss_pnl=_fmt_money(row["legacy_only_path_winner_net_pnl"]),
                path_avoid=row["legacy_only_path_loser_trades"],
                path_avoid_pnl=_fmt_money(row["legacy_only_path_loser_net_pnl"]),
                new=row["strict_only_trades"],
                new_pnl=_fmt_money(row["strict_only_net_pnl"]),
                common=_fmt_money(row["common_net_pnl_delta"]),
                total=_fmt_money(row["strict_minus_legacy_net_pnl_reconciled"]),
                lh=row["low_first_high_second_trades"],
                lh_pnl=_fmt_money(row["low_first_high_second_net_pnl"]),
                coverage=_fmt_pct(row["metadata_coverage_pct"]),
            )
        )

    all_total = sum(_as_float(row["strict_minus_legacy_net_pnl_reconciled"]) for row in summary_rows)
    missed = sum(_as_float(row["legacy_only_winner_net_pnl"]) for row in summary_rows)
    avoided = sum(_as_float(row["legacy_only_loser_net_pnl"]) for row in summary_rows)
    low_rr_missed = sum(_as_float(row["legacy_only_low_rr_winner_net_pnl"]) for row in summary_rows)
    low_rr_avoided = sum(_as_float(row["legacy_only_low_rr_loser_net_pnl"]) for row in summary_rows)
    path_missed = sum(_as_float(row["legacy_only_path_winner_net_pnl"]) for row in summary_rows)
    path_avoided = sum(_as_float(row["legacy_only_path_loser_net_pnl"]) for row in summary_rows)
    strict_only = sum(_as_float(row["strict_only_net_pnl"]) for row in summary_rows)
    common = sum(_as_float(row["common_net_pnl_delta"]) for row in summary_rows)
    low_high_pnl = sum(_as_float(row["low_first_high_second_net_pnl"]) for row in summary_rows)
    low_high_count = sum(int(row["low_first_high_second_trades"]) for row in summary_rows)
    low_high_early = sum(int(row["low_first_high_second_early_stop_trades"]) for row in summary_rows)

    lines.extend(
        [
            "",
            "## 汇总结论",
            "",
            f"- 四年合计严格规则相对 legacy 的拆解差额：{_fmt_money(all_total)}。",
            f"- legacy-only 盈利单合计：{_fmt_money(missed)}，其中第一目标 RR<1.5 的直接错杀为 {_fmt_money(low_rr_missed)}，其余 {_fmt_money(path_missed)} 更像账户路径变化导致的错过。",
            f"- legacy-only 亏损单合计：{_fmt_money(avoided)}，其中第一目标 RR<1.5 的直接过滤有效为 {_fmt_money(low_rr_avoided)}，其余 {_fmt_money(path_avoided)} 是路径变化避开的亏损。",
            f"- 严格规则释放资金后新增单合计：{_fmt_money(strict_only)}；共同单仓位/退出差异合计：{_fmt_money(common)}。",
            f"- legacy 中“第一目标低于1.5、第二目标不低于2.5”的单共 {low_high_count} 笔，合计 {_fmt_money(low_high_pnl)}，其中 {low_high_early} 笔为3日内早止损。",
            "",
            "## 低RR直接错杀盈利单",
            "",
        ]
    )
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="低RR错杀趋势机会", sort_field="legacy_net_pnl", reverse=True),
            "legacy_net_pnl",
        )
    )
    lines.extend(["", "## 路径错过盈利单", ""])
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="路径错过盈利机会", sort_field="legacy_net_pnl", reverse=True),
            "legacy_net_pnl",
        )
    )
    lines.extend(["", "## 低RR过滤有效单", ""])
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="低RR过滤有效", sort_field="legacy_net_pnl", reverse=False),
            "legacy_net_pnl",
        )
    )
    lines.extend(["", "## 路径避开亏损单", ""])
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="路径避开亏损", sort_field="legacy_net_pnl", reverse=False),
            "legacy_net_pnl",
        )
    )
    lines.extend(["", "## 严格规则新增单：最大盈利", ""])
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="资金释放后新增盈利", sort_field="strict_net_pnl", reverse=True),
            "strict_net_pnl",
        )
    )
    lines.extend(["", "## 严格规则新增单：最大亏损", ""])
    lines.extend(
        _render_top_table(
            _top_rows(detail_rows, attribution="资金释放后新增亏损", sort_field="strict_net_pnl", reverse=False),
            "strict_net_pnl",
        )
    )
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    *,
    years: tuple[int, ...],
    report_dir: Path,
    cache_dir: Path,
    output_prefix: Path,
) -> dict[str, Any]:
    meta_index = load_phase23_meta_index(cache_dir)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    low_high_rows: list[dict[str, Any]] = []
    for year in years:
        legacy_path, strict_path, decisions_path = _paths_for_year(report_dir, year)
        legacy = enrich_with_phase23_meta(_load_csv_rows(legacy_path), meta_index)
        strict = enrich_with_phase23_meta(_load_csv_rows(strict_path), meta_index)
        decisions = _load_csv_rows(decisions_path)
        detail, summary = analyze_run_pair(
            year=year,
            legacy_trades=legacy,
            strict_trades=strict,
            strict_decisions=decisions,
        )
        detail_rows.extend(detail)
        summary_rows.append(summary)
        for row in legacy:
            if _is_low_first_high_second(row):
                out = dict(row)
                out["year"] = int(year)
                out["trade_key"] = _trade_key(row)
                out["early_stop"] = _is_early_stop(row)
                out["hold_days"] = _hold_days(row)
                out["adverse_entry_deviation_r"] = _adverse_entry_deviation_r(row)
                low_high_rows.append(out)

    payload = {
        "years": list(years),
        "thresholds": {
            "low_first_rr": LOW_FIRST_RR,
            "high_second_rr": HIGH_SECOND_RR,
            "early_stop_days": EARLY_STOP_DAYS,
        },
        "summary": summary_rows,
    }
    _write_csv(output_prefix.with_name(f"{output_prefix.name}_detail.csv"), detail_rows, DETAIL_FIELDS)
    _write_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), summary_rows, SUMMARY_FIELDS)
    low_high_fields = [
        "year",
        "trade_key",
        "symbol",
        "name",
        "direction",
        "entry_time",
        "actual_exit_time",
        "actual_exit_reason",
        "net_pnl",
        "lots",
        "phase2_score",
        "entry_rr",
        "entry_admission_rr",
        "early_stop",
        "hold_days",
        "adverse_entry_deviation_r",
        "planned_entry_ref",
        "planned_stop",
        "planned_tp1",
        "planned_tp2",
        "entry_signal_type",
        "entry_signal_detail",
        "candidate_trade_id",
    ]
    _write_csv(output_prefix.with_name(f"{output_prefix.name}_low_first_high_second.csv"), low_high_rows, low_high_fields)
    output_prefix.with_name(f"{output_prefix.name}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_prefix.with_name(f"{output_prefix.name}.md").write_text(
        render_markdown(summary_rows, detail_rows),
        encoding="utf-8",
    )
    return payload


def _parse_years(value: str) -> tuple[int, ...]:
    years = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not years:
        raise argparse.ArgumentTypeError("at least one year is required")
    return years


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze trend account RR filter attribution")
    parser.add_argument("--years", type=_parse_years, default=DEFAULT_YEARS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_analysis(
        years=args.years,
        report_dir=args.report_dir,
        cache_dir=args.cache_dir,
        output_prefix=args.output_prefix,
    )
    print(f"wrote attribution report for years={','.join(str(year) for year in payload['years'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
