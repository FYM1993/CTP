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
    return importlib.import_module("analyze_rr_guidance")


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "A0",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "planned_exit_time": "2025-01-03 09:00:00",
                "planned_exit_reason": "stop",
                "pnl_ratio_price": -0.03,
                "tp1_hit": False,
                "entry_rr": 0.8,
                "entry_admission_rr": 2.8,
                "medium_term_quality_score": 70.0,
            },
            {
                "symbol": "B0",
                "direction": "long",
                "entry_time": "2025-01-04 09:00:00",
                "planned_exit_time": "2025-01-24 09:00:00",
                "planned_exit_reason": "tp2",
                "pnl_ratio_price": 0.12,
                "tp1_hit": True,
                "entry_rr": 1.8,
                "entry_admission_rr": 3.4,
                "medium_term_quality_score": 72.0,
            },
            {
                "symbol": "C0",
                "direction": "short",
                "entry_time": "2025-02-02 09:00:00",
                "planned_exit_time": "2025-02-12 09:00:00",
                "planned_exit_reason": "tp1",
                "pnl_ratio_price": 0.04,
                "tp1_hit": True,
                "entry_rr": 1.4,
                "entry_admission_rr": 2.1,
                "medium_term_quality_score": 45.0,
            },
            {
                "symbol": "D0",
                "direction": "short",
                "entry_time": "2025-03-02 09:00:00",
                "planned_exit_time": "2025-03-03 09:00:00",
                "planned_exit_reason": "stop",
                "pnl_ratio_price": -0.02,
                "tp1_hit": False,
                "entry_rr": 1.1,
                "entry_admission_rr": 1.7,
                "medium_term_quality_score": 42.0,
            },
        ]
    )


def test_prepare_rr_rows_labels_early_stop_and_long_hold_winner() -> None:
    analysis = _module()

    rows = analysis.prepare_rr_rows(_sample_rows(), early_stop_days=3, long_hold_days=14)

    assert rows["hold_days"].tolist() == [1.0, 20.0, 10.0, 1.0]
    assert rows["is_early_stop"].tolist() == [True, False, False, True]
    assert rows["is_long_hold_winner"].tolist() == [False, True, False, False]
    assert rows["is_tp2_winner"].tolist() == [False, True, False, False]


def test_analyze_rr_guidance_reports_first_and_second_rr_roles() -> None:
    analysis = _module()
    raw = _sample_rows()
    raw["year"] = [2025, 2025, 2026, 2026]
    rows = analysis.prepare_rr_rows(raw, early_stop_days=3, long_hold_days=14)

    report = analysis.analyze_rr_guidance(rows)

    first = report["score_diagnostics"]["entry_rr"]
    second = report["score_diagnostics"]["entry_admission_rr"]
    assert first["tp1_hit_auc"] == pytest.approx(1.0)
    assert first["survive_early_auc"] == pytest.approx(1.0)
    assert second["tp2_winner_auc"] == pytest.approx(1.0)
    assert second["long_hold_winner_auc"] == pytest.approx(1.0)
    assert report["rows"] == 4
    assert [row["year"] for row in report["yearly_score_diagnostics"]][:2] == [2025, 2025]


def test_rr_bucket_outputs_cross_first_and_second_rr() -> None:
    analysis = _module()
    rows = analysis.prepare_rr_rows(_sample_rows(), early_stop_days=3, long_hold_days=14)

    cross = analysis.cross_rr_bucket_rows(rows)

    selected = [
        row
        for row in cross
        if row["entry_rr_bucket"] == "1.5_2.0" and row["entry_admission_rr_bucket"] == "3.0_plus"
    ][0]
    assert selected["trade_count"] == 1
    assert selected["tp2_winner_rate"] == pytest.approx(1.0)


def test_legacy_phase23_config_removes_experiment_overrides_for_cache_matching() -> None:
    analysis = _module()

    config = {
        "pre_market": {
            "base": True,
            "trend_min_first_target_rr": 1.5,
            "trend_entry_confirmation_bars": 2,
        },
        "intraday": {},
    }

    legacy = analysis.legacy_phase23_config(config)

    assert legacy["pre_market"]["base"] is True
    assert "trend_min_first_target_rr" not in legacy["pre_market"]
    assert "trend_entry_confirmation_bars" not in legacy["pre_market"]
