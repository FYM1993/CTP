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
    return importlib.import_module("analyze_tp1_take_profit_effect")


def test_prepare_tp1_rows_measures_continuation_and_reduction_cost() -> None:
    analysis = _module()
    raw = pd.DataFrame(
        [
            {
                "symbol": "A0",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "exit_time": "2025-01-10 09:00:00",
                "exit_reason": "tp2",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "pnl_ratio": 0.0835,
                "tp1_hit": True,
                "entry_rr": 1.0,
                "tp1_exit_fraction": 0.33,
                "tp1_exit_price": 105.0,
            },
            {
                "symbol": "B0",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "exit_time": "2025-01-03 09:00:00",
                "exit_reason": "stop",
                "entry_price": 100.0,
                "exit_price": 99.0,
                "pnl_ratio": 0.0095,
                "tp1_hit": True,
                "entry_rr": 1.0,
                "tp1_exit_fraction": 0.33,
                "tp1_exit_price": 105.0,
            },
        ]
    )

    rows = analysis.prepare_tp1_rows(raw)

    assert rows.loc[0, "final_leg_r"] == pytest.approx(2.0)
    assert rows.loc[0, "post_tp1_continuation_r"] == pytest.approx(1.0)
    assert rows.loc[0, "tp1_reduction_delta_r"] == pytest.approx(-0.33)
    assert bool(rows.loc[0, "tp1_reduction_helped"]) is False
    assert rows.loc[1, "final_leg_r"] == pytest.approx(-0.2)
    assert rows.loc[1, "post_tp1_continuation_r"] == pytest.approx(-1.2)
    assert rows.loc[1, "tp1_reduction_delta_r"] == pytest.approx(0.39)
    assert bool(rows.loc[1, "tp1_reduction_helped"]) is True


def test_analyze_tp1_effect_reports_helped_and_cost_counts() -> None:
    analysis = _module()
    raw = pd.DataFrame(
        [
            {
                "symbol": "A0",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "exit_time": "2025-01-10 09:00:00",
                "exit_reason": "tp2",
                "entry_price": 100.0,
                "exit_price": 110.0,
                "pnl_ratio": 0.0835,
                "tp1_hit": True,
                "entry_rr": 1.0,
                "tp1_exit_fraction": 0.33,
                "tp1_exit_price": 105.0,
            },
            {
                "symbol": "B0",
                "direction": "long",
                "entry_time": "2025-01-02 09:00:00",
                "exit_time": "2025-01-03 09:00:00",
                "exit_reason": "stop",
                "entry_price": 100.0,
                "exit_price": 99.0,
                "pnl_ratio": 0.0095,
                "tp1_hit": True,
                "entry_rr": 1.0,
                "tp1_exit_fraction": 0.33,
                "tp1_exit_price": 105.0,
            },
        ]
    )

    report = analysis.analyze_tp1_effect(analysis.prepare_tp1_rows(raw))

    assert report["tp1_hit_trades"] == 2
    assert report["post_tp1_continued_trades"] == 1
    assert report["tp1_reduction_helped_trades"] == 1
    assert report["tp1_reduction_cost_trades"] == 1
    assert report["avg_post_tp1_continuation_r"] == pytest.approx(-0.1)
