from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_phase1_phase2_split_grid import (  # noqa: E402
    GridSpec,
    _candidate_for_grid,
    _config_for_grid,
    _grid_label,
    iter_grid_specs,
)


def test_iter_grid_specs_covers_phase1_phase2_thresholds_and_split_fractions() -> None:
    specs = list(
        iter_grid_specs(
            phase1_budget_mins=(0.15, 0.30),
            phase2_score_mins=(0.0, 40.0),
            split_initial_fractions=(1.0, 0.50),
            max_portfolio_margin_pcts=(0.30,),
        )
    )

    assert len(specs) == 8
    assert GridSpec(0.30, 0.15, 40.0, 0.50) in specs
    assert GridSpec(0.30, 0.30, 0.0, 1.00) in specs


def test_config_for_grid_uses_story_budget_gate_trigger_gate_and_reserved_split_budget() -> None:
    split_config = _config_for_grid(GridSpec(0.30, 0.15, 40.0, 0.50))
    full_config = _config_for_grid(GridSpec(0.30, 0.30, 45.0, 1.00))

    assert split_config.position_budget_source == "phase1_story"
    assert split_config.reserve_planned_second_entry_margin is True
    assert split_config.min_phase1_story_budget_margin_pct == pytest.approx(0.15)
    assert split_config.min_phase2_abs_score == pytest.approx(40.0)
    assert split_config.split_initial_fraction == pytest.approx(0.50)
    assert split_config.split_second_entry_trigger == "tp1"
    assert full_config.split_second_entry_trigger == "none"


def test_candidate_for_grid_keeps_phase2_score_and_adds_phase1_story_budget() -> None:
    candidate = _candidate_for_grid(
        {
            "symbol": "AU0",
            "name": "黄金",
            "direction": "long",
            "entry_time": "2025-01-02 09:00:00",
            "exit_time": "2025-01-03 09:00:00",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "exit_reason": "tp2",
            "phase2_score": 35.0,
            "risk_per_lot": 500.0,
            "margin_per_lot": 10_000.0,
            "notional_per_lot": 1_000.0,
            "multiplier": 10.0,
            "pnl_ratio": 0.10,
            "tp1_hit": True,
            "trend_opportunity_quality_score": 72.0,
            "trend_structure_health_score": 80.0,
        }
    )

    assert candidate.phase2_score == 35.0
    assert candidate.phase1_story_budget_margin_pct == pytest.approx(0.30)
    assert candidate.phase1_story_budget_source == "trend_story"
    assert candidate.medium_term_quality_score == pytest.approx(72.0)


def test_grid_label_is_human_readable_for_report_rows() -> None:
    assert _grid_label(GridSpec(0.30, 0.15, 40.0, 0.50)) == "m30_p1>=15_p2>=40_first50"
