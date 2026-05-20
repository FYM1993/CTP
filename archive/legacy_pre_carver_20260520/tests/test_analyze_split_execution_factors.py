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
    return importlib.import_module("analyze_split_execution_factors")


def test_path_metrics_detect_trial_then_confirm_before_tp1() -> None:
    analysis = _module()
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 10:00:00",
                    "2025-01-02 11:00:00",
                    "2025-01-03 09:00:00",
                ]
            ),
            "open": [100.0, 98.0, 96.0, 105.0],
            "high": [101.0, 99.0, 104.0, 111.0],
            "low": [99.0, 94.0, 95.0, 104.0],
            "close": [100.0, 96.0, 103.0, 110.0],
            "volume": [1, 1, 1, 1],
        }
    )
    trade = {
        "direction": "long",
        "entry_time": "2025-01-02 09:00:00",
        "exit_time": "2025-01-03 10:00:00",
        "entry_price": 100.0,
        "initial_stop_price": 90.0,
        "tp1_exit_time": "2025-01-03 09:00:00",
        "tp1_hit": True,
        "exit_reason": "tp2",
    }

    metrics = analysis.compute_path_metrics_for_trade(trade, minute)

    assert metrics["mae_1d_r"] == pytest.approx(0.6)
    assert metrics["mfe_3d_r"] == pytest.approx(1.1)
    assert metrics["touched_minus_0p5r_before_tp1"] is True
    assert metrics["touched_minus_1r_before_tp1"] is False
    assert metrics["tp1_pre_mae_r"] == pytest.approx(0.6)
    assert metrics["entry_path_bucket"] == "trial_then_confirm"


def test_path_metrics_do_not_count_before_tp1_touch_when_tp1_never_hits() -> None:
    analysis = _module()
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00", "2025-01-02 10:00:00"]),
            "open": [100.0, 95.0],
            "high": [101.0, 96.0],
            "low": [99.0, 89.0],
            "close": [95.0, 90.0],
            "volume": [1, 1],
        }
    )
    trade = {
        "direction": "long",
        "entry_time": "2025-01-02 09:00:00",
        "exit_time": "2025-01-02 10:00:00",
        "entry_price": 100.0,
        "initial_stop_price": 90.0,
        "tp1_exit_time": "",
        "tp1_hit": False,
        "exit_reason": "stop",
    }

    metrics = analysis.compute_path_metrics_for_trade(trade, minute)

    assert metrics["mae_1d_r"] == pytest.approx(1.1)
    assert metrics["touched_minus_0p5r_before_tp1"] is False
    assert metrics["touched_minus_1r_before_tp1"] is False
    assert metrics["entry_path_bucket"] == "early_failure"


def test_account_policy_rows_compare_full_and_split_for_bucket() -> None:
    analysis = _module()
    rows = pd.DataFrame(
        [
            {
                "trade_id": "A0-1",
                "symbol": "A0",
                "name": "A",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "exit_time": "2025-01-05 09:00:00",
                "exit_reason": "tp2",
                "entry_price": 100.0,
                "exit_price": 120.0,
                "pnl_ratio": 0.20,
                "tp1_hit": True,
                "tp1_exit_time": "2025-01-03 09:00:00",
                "tp1_exit_price": 110.0,
                "phase2_score": 65.0,
                "entry_rr": 1.0,
                "entry_admission_rr": 2.0,
                "medium_term_quality_score": 65.0,
                "medium_term_entry_location_score": 80.0,
                "entry_adverse_deviation_r": 0.0,
                "risk_per_lot": 100.0,
                "margin_per_lot": 1_000.0,
                "notional_per_lot": 1_000.0,
                "multiplier": 10.0,
                "entry_path_bucket": "fast_favorable",
            },
            {
                "trade_id": "B0-1",
                "symbol": "B0",
                "name": "B",
                "direction": "long",
                "entry_time": "2025-01-02 09:01:00",
                "exit_time": "2025-01-03 09:00:00",
                "exit_reason": "stop",
                "entry_price": 100.0,
                "exit_price": 90.0,
                "pnl_ratio": -0.10,
                "tp1_hit": False,
                "tp1_exit_time": "",
                "tp1_exit_price": 0.0,
                "phase2_score": 65.0,
                "entry_rr": 1.0,
                "entry_admission_rr": 2.0,
                "medium_term_quality_score": 45.0,
                "medium_term_entry_location_score": 20.0,
                "entry_adverse_deviation_r": 0.8,
                "risk_per_lot": 100.0,
                "margin_per_lot": 1_000.0,
                "notional_per_lot": 1_000.0,
                "multiplier": 10.0,
                "entry_path_bucket": "early_failure",
            },
        ]
    )

    policy_rows = analysis.account_policy_rows_for_group(
        rows,
        group_field="entry_path_bucket",
        group_value="fast_favorable",
        initial_equity=100_000.0,
    )

    policies = {row["policy"]: row for row in policy_rows}
    assert {"fixed_full", "split_30_tp1", "split_50_tp1", "split_70_tp1"} <= set(policies)
    assert policies["fixed_full"]["candidate_trades"] == 1
    assert policies["fixed_full"]["tp2_capture_rate"] == pytest.approx(1.0)
    assert policies["fixed_full"]["max_margin_pct_observed"] > 0
    assert policies["split_30_tp1"]["split_second_entry_filled_lots"] > 0
    assert policies["split_30_tp1"]["net_profit_delta_vs_fixed"] < 0


def test_entry_adverse_bucket_labels_trigger_slippage_in_r() -> None:
    analysis = _module()

    assert analysis.entry_adverse_bucket(0.0) == "no_adverse"
    assert analysis.entry_adverse_bucket(0.25) == "mild_adverse"
    assert analysis.entry_adverse_bucket(0.75) == "stretched"
    assert analysis.entry_adverse_bucket(1.25) == "extreme"
