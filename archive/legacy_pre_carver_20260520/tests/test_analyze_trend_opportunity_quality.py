from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_trend_opportunity_quality import (  # noqa: E402
    _candidate_for_policy,
    _config_for_policy,
    phase2_score_bucket,
    phase2_score_for_target_margin_pct,
    trend_budget_margin_pct,
    trend_opportunity_bucket,
)


def test_phase2_score_bucket_uses_absolute_short_term_trigger_strength() -> None:
    assert phase2_score_bucket(-38.0) == "lt_40"
    assert phase2_score_bucket(45.0) == "40_50"
    assert phase2_score_bucket(55.0) == "50_60"
    assert phase2_score_bucket(67.0) == "60_plus"


def test_trend_budget_mapping_keeps_budget_layer_separate_from_phase2_score() -> None:
    assert trend_opportunity_bucket(35.0) == "low"
    assert trend_opportunity_bucket(55.0) == "medium"
    assert trend_opportunity_bucket(72.0) == "high"
    assert trend_budget_margin_pct(35.0) == 0.03
    assert trend_budget_margin_pct(55.0) == 0.15
    assert trend_budget_margin_pct(72.0) == 0.30


def test_phase2_surrogate_score_round_trips_account_margin_budget() -> None:
    assert phase2_score_for_target_margin_pct(0.03) == 28.5
    assert phase2_score_for_target_margin_pct(0.15) == 42.5
    assert phase2_score_for_target_margin_pct(0.30) == 60.0


def test_trend_quality_policy_uses_phase1_story_budget_not_phase2_surrogate() -> None:
    row = {
        "symbol": "AU0",
        "name": "黄金",
        "direction": "long",
        "entry_time": "2025-01-02 09:00:00",
        "exit_time": "2025-01-03 09:00:00",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "exit_reason": "tp2",
        "phase2_score": 30.0,
        "risk_per_lot": 500.0,
        "margin_per_lot": 10_000.0,
        "notional_per_lot": 1_000.0,
        "multiplier": 10.0,
        "pnl_ratio": 0.10,
        "tp1_hit": True,
        "trend_opportunity_quality_score": 72.0,
        "trend_structure_health_score": 80.0,
    }

    candidate = _candidate_for_policy(row, "trend_quality_budget_fixed_full")
    config = _config_for_policy("trend_quality_budget_fixed_full", max_portfolio_margin_pct=0.30)

    assert candidate.phase2_score == 30.0
    assert candidate.phase1_story_budget_margin_pct == 0.30
    assert candidate.phase1_story_budget_source == "trend_story"
    assert config.position_budget_source == "phase1_story"
