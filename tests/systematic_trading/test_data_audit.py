from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.data_audit import audit_source_directory, write_data_source_report  # noqa: E402


def _write_daily(path: Path, dates: pd.DatetimeIndex) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 100 + len(dates)),
            "high": range(101, 101 + len(dates)),
            "low": range(99, 99 + len(dates)),
            "close": range(100, 100 + len(dates)),
            "volume": 1000,
            "oi": 5000,
        }
    ).to_parquet(path)


def test_audit_source_directory_flags_stale_and_short_history(tmp_path: Path) -> None:
    _write_daily(
        tmp_path / "daily_AG0_20260508_live.parquet",
        pd.date_range("2025-01-01", "2025-01-10", freq="B"),
    )

    audit = audit_source_directory(
        source_name="live_daily_cache",
        directory=tmp_path,
        pattern="daily_*_live.parquet",
        min_start=pd.Timestamp("2022-01-01"),
        min_end=pd.Timestamp("2026-05-19"),
        min_rows=260,
    )

    assert audit.summary["files"] == 1
    assert audit.summary["tradeable_files"] == 0
    assert "history_starts_after_required_date" in audit.files.loc[0, "issues"]
    assert "latest_bar_before_required_date" in audit.files.loc[0, "issues"]
    assert "short_history" in audit.files.loc[0, "issues"]


def test_write_data_source_report_states_blocked_verdict(tmp_path: Path) -> None:
    _write_daily(
        tmp_path / "daily_AG0_20260508_live.parquet",
        pd.date_range("2025-01-01", "2025-01-10", freq="B"),
    )
    audit = audit_source_directory(
        source_name="live_daily_cache",
        directory=tmp_path,
        pattern="daily_*_live.parquet",
        min_start=pd.Timestamp("2022-01-01"),
        min_end=pd.Timestamp("2026-05-19"),
        min_rows=260,
    )
    report_path = tmp_path / "audit.md"

    write_data_source_report(report_path, [audit])

    text = report_path.read_text()
    assert "数据口径审计报告" in text
    assert "BLOCKED" in text
    assert "live_daily_cache" in text
