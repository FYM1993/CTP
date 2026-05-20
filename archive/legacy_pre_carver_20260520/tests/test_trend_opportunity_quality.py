from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.trend_opportunity_quality import (  # noqa: E402
    trend_budget_margin_pct,
    trend_opportunity_quality_from_daily,
)


def _frame_from_close(close: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000.0 + idx * 2.0 for idx in range(len(close))],
            "oi": [500.0 + idx * 1.5 for idx in range(len(close))],
        }
    )


def test_scores_orderly_pullback_as_healthy_trend_opportunity() -> None:
    close = [100.0 + idx * 0.45 for idx in range(140)]
    close.extend([163.0, 162.2, 161.8, 162.5, 163.4, 164.1, 165.0, 166.0, 166.8, 167.5])
    quality = trend_opportunity_quality_from_daily(_frame_from_close(close), direction="long")

    assert quality["trend_opportunity_quality_status"] == "candidate_v1"
    assert quality["trend_structure_bucket"] == "healthy_continuation"
    assert quality["trend_opportunity_bucket"] == "high"
    assert quality["trend_structure_health_score"] >= 70.0
    assert trend_budget_margin_pct(quality["trend_opportunity_quality_score"]) == 0.30


def test_penalizes_strong_but_overextended_entry_structure() -> None:
    orderly = [100.0 + idx * 0.35 for idx in range(150)]
    stretched = orderly[:-8] + [orderly[-9] + 8.0 + idx * 3.0 for idx in range(8)]

    healthy = trend_opportunity_quality_from_daily(_frame_from_close(orderly), direction="long")
    overextended = trend_opportunity_quality_from_daily(_frame_from_close(stretched), direction="long")

    assert overextended["trend_direction_stability_score"] >= healthy["trend_direction_stability_score"] - 10.0
    assert overextended["trend_structure_bucket"] == "overextended"
    assert overextended["trend_structure_health_score"] < healthy["trend_structure_health_score"]
    assert overextended["trend_budget_margin_pct"] < healthy["trend_budget_margin_pct"]


def test_penalizes_trend_after_recent_structure_break() -> None:
    close = [100.0 + idx * 0.35 for idx in range(130)]
    close.extend([145.0, 143.0, 140.0, 136.0, 132.0, 128.0, 126.0, 124.0, 123.0, 122.0])
    quality = trend_opportunity_quality_from_daily(_frame_from_close(close), direction="long")

    assert quality["trend_structure_bucket"] == "damaged"
    assert quality["trend_opportunity_bucket"] == "low"
    assert quality["trend_structure_health_score"] <= 35.0
    assert quality["trend_budget_margin_pct"] <= 0.08


def test_scores_orderly_short_trend_in_trade_direction() -> None:
    close = [180.0 - idx * 0.35 for idx in range(150)]
    quality = trend_opportunity_quality_from_daily(_frame_from_close(close), direction="short")

    assert quality["trend_opportunity_bucket"] in {"medium", "high"}
    assert quality["trend_direction_stability_score"] >= 60.0
    assert quality["trend_structure_bucket"] == "healthy_continuation"
