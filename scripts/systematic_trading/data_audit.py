from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SourceAudit:
    source_name: str
    files: pd.DataFrame
    summary: dict[str, object]


def symbol_from_market_file(path: Path) -> str:
    daily_match = re.match(r"daily_([A-Z0-9]+)_\d{8}_(?:live|final)\.parquet$", path.name)
    if daily_match:
        return daily_match.group(1)
    backtest_match = re.match(r"([a-z0-9]+)_\d{8}_\d{8}_.+_daily\.parquet$", path.name)
    if backtest_match:
        return backtest_match.group(1).upper()
    return path.stem.split("_")[0].upper()


def _normalize_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    frame = raw.copy()
    if "date" not in frame.columns and "datetime" in frame.columns:
        frame["date"] = frame["datetime"]
    if "open_interest" not in frame.columns and "oi" in frame.columns:
        frame["open_interest"] = frame["oi"]
    if "open_interest" not in frame.columns and "hold" in frame.columns:
        frame["open_interest"] = frame["hold"]
    return frame


def _file_audit_row(
    *,
    source_name: str,
    path: Path,
    min_start: pd.Timestamp,
    min_end: pd.Timestamp,
    min_rows: int,
) -> dict[str, object]:
    issues: list[str] = []
    symbol = symbol_from_market_file(path)
    try:
        frame = _normalize_frame(path)
    except Exception as exc:
        return {
            "source": source_name,
            "symbol": symbol,
            "file": str(path),
            "rows": 0,
            "start_date": "",
            "end_date": "",
            "has_open_interest": False,
            "is_tradeable": False,
            "issues": f"read_error:{exc}",
        }

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        issues.append("missing_columns:" + ",".join(missing))
        return {
            "source": source_name,
            "symbol": symbol,
            "file": str(path),
            "rows": int(len(frame)),
            "start_date": "",
            "end_date": "",
            "has_open_interest": "open_interest" in frame.columns,
            "is_tradeable": False,
            "issues": ",".join(issues),
        }

    dates = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    close = pd.to_numeric(frame["close"], errors="coerce")
    start_date = dates.min()
    end_date = dates.max()
    if dates.isna().any():
        issues.append("invalid_date")
    if not dates.is_monotonic_increasing:
        issues.append("date_not_sorted")
    if dates.duplicated().any():
        issues.append("duplicate_dates")
    if len(frame) < min_rows:
        issues.append("short_history")
    if pd.isna(start_date) or start_date > min_start:
        issues.append("history_starts_after_required_date")
    if pd.isna(end_date) or end_date < min_end:
        issues.append("latest_bar_before_required_date")
    if close.isna().any():
        issues.append("missing_close")
    if (close <= 0).any():
        issues.append("non_positive_close")
    if "open_interest" not in frame.columns:
        issues.append("missing_open_interest")

    return {
        "source": source_name,
        "symbol": symbol,
        "file": str(path),
        "rows": int(len(frame)),
        "start_date": "" if pd.isna(start_date) else start_date.date().isoformat(),
        "end_date": "" if pd.isna(end_date) else end_date.date().isoformat(),
        "has_open_interest": "open_interest" in frame.columns,
        "is_tradeable": len(issues) == 0,
        "issues": ",".join(issues),
    }


def audit_source_directory(
    *,
    source_name: str,
    directory: Path,
    pattern: str,
    min_start: pd.Timestamp,
    min_end: pd.Timestamp,
    min_rows: int,
) -> SourceAudit:
    files = sorted(directory.glob(pattern))
    rows = [
        _file_audit_row(
            source_name=source_name,
            path=path,
            min_start=min_start,
            min_end=min_end,
            min_rows=min_rows,
        )
        for path in files
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "source",
                "symbol",
                "file",
                "rows",
                "start_date",
                "end_date",
                "has_open_interest",
                "is_tradeable",
                "issues",
            ]
        )
    summary = {
        "source": source_name,
        "files": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "tradeable_files": int(frame["is_tradeable"].sum()) if "is_tradeable" in frame else 0,
        "earliest_start": "" if frame.empty else str(frame["start_date"].replace("", pd.NA).dropna().min() or ""),
        "latest_end": "" if frame.empty else str(frame["end_date"].replace("", pd.NA).dropna().max() or ""),
        "min_rows": 0 if frame.empty else int(frame["rows"].min()),
        "max_rows": 0 if frame.empty else int(frame["rows"].max()),
    }
    return SourceAudit(source_name=source_name, files=frame, summary=summary)


def source_verdict(audits: list[SourceAudit], min_tradeable_symbols: int = 1) -> str:
    symbols: set[str] = set()
    for audit in audits:
        if audit.files.empty:
            continue
        tradeable = audit.files.loc[audit.files["is_tradeable"], "symbol"].dropna().astype(str)
        symbols.update(tradeable.tolist())
    return "PASS" if len(symbols) >= min_tradeable_symbols else "BLOCKED"


def write_data_source_report(path: Path, audits: list[SourceAudit], min_tradeable_symbols: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict = source_verdict(audits, min_tradeable_symbols=min_tradeable_symbols)
    tradeable_symbols: set[str] = set()
    for audit in audits:
        if not audit.files.empty:
            tradeable_symbols.update(audit.files.loc[audit.files["is_tradeable"], "symbol"].dropna().astype(str))
    lines = [
        "# 数据口径审计报告",
        "",
        f"- 总体结论: {verdict}",
        f"- 最低可用品种数要求: {min_tradeable_symbols}",
        f"- 当前可用品种数: {len(tradeable_symbols)}",
        "",
        "## 业务含义",
        "",
        "这份报告只审计本地行情文件是否足以支撑系统化期货回测。它不证明数据与交易软件完全一致；交易软件价格一致性仍需要人工抽样或外部行情源校验。",
        "",
        "## 数据源摘要",
        "",
        "| 数据源 | 文件数 | 品种数 | 可用文件数 | 最早日期 | 最新日期 | 最少行数 | 最多行数 |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for audit in audits:
        summary = audit.summary
        lines.append(
            "| {source} | {files} | {symbols} | {tradeable_files} | {earliest_start} | {latest_end} | {min_rows} | {max_rows} |".format(
                **summary
            )
        )
    lines.extend(["", "## 主要问题", ""])
    for audit in audits:
        bad = audit.files.loc[~audit.files["is_tradeable"]].copy()
        lines.append(f"### {audit.source_name}")
        if bad.empty:
            lines.append("")
            lines.append("- 未发现阻断问题。")
            lines.append("")
            continue
        counts = bad["issues"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
        for issue, count in counts.items():
            if issue:
                lines.append(f"- {issue}: {int(count)}")
        lines.append("")
        sample = bad.head(12)
        lines.append("| 品种 | 起始 | 截止 | 行数 | 问题 |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for _, row in sample.iterrows():
            lines.append(
                f"| {row['symbol']} | {row['start_date']} | {row['end_date']} | {int(row['rows'])} | {row['issues']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
