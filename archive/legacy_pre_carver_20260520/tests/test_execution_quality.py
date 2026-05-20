from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.execution_quality import medium_term_quality_from_daily, visible_daily_before_entry  # noqa: E402


def test_visible_daily_before_entry_excludes_entry_date() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "close": [100.0, 101.0, 102.0],
        }
    )

    visible = visible_daily_before_entry(daily, "2025-01-03 09:00:00")

    assert visible["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-01-02"]


def test_medium_term_quality_scores_persistent_trend_and_entry_location() -> None:
    orderly = pd.DataFrame(
        {
            "date": pd.date_range("2024-09-01", periods=160, freq="D"),
            "open": [100.0 + idx * 0.35 for idx in range(160)],
            "high": [101.0 + idx * 0.35 for idx in range(160)],
            "low": [99.0 + idx * 0.35 for idx in range(160)],
            "close": [100.0 + idx * 0.35 for idx in range(160)],
            "volume": [1000 + idx for idx in range(160)],
            "oi": [500 + idx for idx in range(160)],
        }
    )
    flat = orderly.copy()
    flat["close"] = [100.0 + (idx % 2) * 0.2 for idx in range(160)]
    flat["open"] = flat["close"]
    flat["high"] = flat["close"] + 1.0
    flat["low"] = flat["close"] - 1.0
    overextended = orderly.copy()
    overextended.loc[150:, "close"] = [float(orderly.iloc[149]["close"]) + 8.0 + idx * 2.5 for idx in range(10)]
    overextended.loc[150:, "open"] = overextended.loc[150:, "close"]
    overextended.loc[150:, "high"] = overextended.loc[150:, "close"] + 1.0
    overextended.loc[150:, "low"] = overextended.loc[150:, "close"] - 1.0

    strong = medium_term_quality_from_daily(orderly, direction="long", trend_phase="markup")
    weak = medium_term_quality_from_daily(flat, direction="long", trend_phase="neutral")
    stretched = medium_term_quality_from_daily(overextended, direction="long", trend_phase="markup")

    assert strong["medium_term_quality_status"] == "candidate_v3"
    assert strong["medium_term_quality_score"] >= 60.0
    assert weak["medium_term_quality_score"] <= 45.0
    assert stretched["medium_term_quality_score"] < strong["medium_term_quality_score"]
    assert stretched["medium_term_entry_location_score"] < strong["medium_term_entry_location_score"]
