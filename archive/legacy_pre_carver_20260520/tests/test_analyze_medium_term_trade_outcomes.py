from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module():
    return importlib.import_module("analyze_medium_term_trade_outcomes")


def test_build_joined_trade_outcomes_labels_early_stops_and_long_hold_winners() -> None:
    analysis = _module()
    trades = pd.DataFrame(
        [
            {
                "symbol": "AG0",
                "name": "白银",
                "direction": "long",
                "entry_time": "2025-01-02 09:31:00",
                "actual_exit_time": "2025-01-03 09:31:00",
                "actual_exit_reason": "stop",
                "net_pnl": -1000.0,
                "phase2_score": 35.0,
            },
            {
                "symbol": "AG0",
                "name": "白银",
                "direction": "long",
                "entry_time": "2025-01-04 09:31:00",
                "actual_exit_time": "2025-01-24 09:31:00",
                "actual_exit_reason": "tp2",
                "net_pnl": 5000.0,
                "phase2_score": 40.0,
            },
            {
                "symbol": "JM0",
                "name": "焦煤",
                "direction": "short",
                "entry_time": "2025-01-06 09:31:00",
                "actual_exit_time": "2025-01-13 09:31:00",
                "actual_exit_reason": "stop",
                "net_pnl": -300.0,
                "phase2_score": 45.0,
            },
        ]
    )
    details = pd.DataFrame(
        [
            {
                "symbol": "AG0",
                "direction": "long",
                "trade_date": "2025-01-02",
                "medium_term_quality_score": 20.0,
                "medium_term_momentum_score": 25.0,
                "medium_term_efficiency_score": 30.0,
            },
            {
                "symbol": "AG0",
                "direction": "long",
                "trade_date": "2025-01-04",
                "medium_term_quality_score": 80.0,
                "medium_term_momentum_score": 82.0,
                "medium_term_efficiency_score": 78.0,
            },
            {
                "symbol": "JM0",
                "direction": "short",
                "trade_date": "2025-01-06",
                "medium_term_quality_score": 50.0,
                "medium_term_momentum_score": 55.0,
                "medium_term_efficiency_score": 45.0,
            },
        ]
    )

    joined = analysis.build_joined_trade_outcomes(trades, details, early_stop_days=3, long_hold_days=14)

    assert len(joined) == 3
    assert joined["phase2_detail_matched"].tolist() == [True, True, True]
    assert joined["outcome_label"].tolist() == ["early_stop", "long_hold_winner", "other"]
    assert joined["hold_days"].tolist() == [1.0, 20.0, 7.0]


def test_analyze_trade_outcomes_reports_medium_term_separation() -> None:
    analysis = _module()
    joined = pd.DataFrame(
        [
            {"outcome_label": "early_stop", "medium_term_quality_score": 20.0, "net_pnl": -1000.0},
            {"outcome_label": "early_stop", "medium_term_quality_score": 30.0, "net_pnl": -800.0},
            {"outcome_label": "long_hold_winner", "medium_term_quality_score": 70.0, "net_pnl": 3000.0},
            {"outcome_label": "long_hold_winner", "medium_term_quality_score": 80.0, "net_pnl": 5000.0},
            {"outcome_label": "other", "medium_term_quality_score": 55.0, "net_pnl": 100.0},
        ]
    )

    report = analysis.analyze_joined_trade_outcomes(joined, thresholds=(40.0, 60.0))

    comparison = report["primary_comparison"]
    assert comparison["early_stop"]["rows"] == 2
    assert comparison["long_hold_winner"]["rows"] == 2
    assert comparison["median_difference"] == pytest.approx(50.0)
    assert comparison["pairwise_auc"] == pytest.approx(1.0)
    assert report["thresholds"][0]["threshold"] == 40.0
    assert report["thresholds"][0]["early_stop_rate"] == pytest.approx(0.0)
    assert report["thresholds"][1]["long_hold_winner_rate"] == pytest.approx(1.0)


def test_analyze_filter_combinations_crosses_medium_score_and_risk_reward() -> None:
    analysis = _module()
    joined = pd.DataFrame(
        [
            {
                "outcome_label": "early_stop",
                "medium_term_quality_score": 80.0,
                "risk_reward_ratio": 1.0,
                "net_pnl": -1000.0,
            },
            {
                "outcome_label": "long_hold_winner",
                "medium_term_quality_score": 75.0,
                "risk_reward_ratio": 3.0,
                "net_pnl": 5000.0,
            },
            {
                "outcome_label": "other",
                "medium_term_quality_score": 45.0,
                "risk_reward_ratio": 3.5,
                "net_pnl": 500.0,
            },
        ]
    )

    combos = analysis.analyze_filter_combinations(
        joined,
        medium_thresholds=(0.0, 60.0),
        rr_thresholds=(0.0, 2.0),
    )

    selected = [
        row
        for row in combos
        if row["medium_term_quality_min"] == 60.0 and row["risk_reward_ratio_min"] == 2.0
    ][0]
    assert selected["trade_count"] == 1
    assert selected["early_stop_count"] == 0
    assert selected["long_hold_winner_count"] == 1
    assert selected["net_pnl"] == pytest.approx(5000.0)
