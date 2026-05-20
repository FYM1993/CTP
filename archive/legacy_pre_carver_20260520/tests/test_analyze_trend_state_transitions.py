from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_trend_state_transitions import (  # noqa: E402
    TransitionPolicy,
    assign_trades_to_transition_stages,
    build_transition_events,
    bucket_for_score,
    candidate_for_transition_policy,
)


def test_bucket_for_score_maps_phase1_trend_states() -> None:
    assert bucket_for_score(float("nan")) == "missing"
    assert bucket_for_score(30.0) == "low"
    assert bucket_for_score(55.0) == "medium"
    assert bucket_for_score(72.0) == "high"


def test_build_transition_events_tracks_low_medium_high_state_changes() -> None:
    state_rows = [
        {"symbol": "AU0", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 38.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-02", "bucket": "medium", "score": 52.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 70.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-04", "bucket": "high", "score": 72.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-05", "bucket": "medium", "score": 60.0},
    ]

    events = build_transition_events(state_rows)

    assert [(row["event_type"], row["date"]) for row in events] == [
        ("low_to_medium", "2025-01-02"),
        ("medium_to_high", "2025-01-03"),
        ("high_to_medium", "2025-01-05"),
    ]
    assert events[1]["event_id"] == "AU0_long_20250103_medium_to_high"


def test_assign_trades_to_transition_stages_uses_latest_visible_transition() -> None:
    state_rows = [
        {"symbol": "AU0", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 38.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-02", "bucket": "medium", "score": 52.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 70.0},
        {"symbol": "AU0", "direction": "long", "date": "2025-02-15", "bucket": "high", "score": 74.0},
    ]
    events = build_transition_events(state_rows)
    trades = [
        {"trade_id": "trial", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-02 09:00:00"},
        {"trade_id": "confirm", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-03 09:00:00"},
        {"trade_id": "late", "symbol": "AU0", "direction": "long", "entry_time": "2025-02-15 09:00:00"},
    ]

    assigned = assign_trades_to_transition_stages(trades, state_rows, events, transition_window_days=20)

    assert [row["transition_stage"] for row in assigned] == [
        "trial_low_to_medium",
        "confirm_medium_to_high",
        "late_high",
    ]
    assert [row["transition_trade_seq"] for row in assigned] == [1, 1, 1]


def test_candidate_for_transition_policy_uses_small_trial_budget_and_full_confirm_budget() -> None:
    base_row = {
        "symbol": "AU0",
        "name": "黄金",
        "direction": "long",
        "entry_time": "2025-01-02 09:00:00",
        "exit_time": "2025-01-03 09:00:00",
        "exit_reason": "tp2",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "phase2_score": 42.0,
        "risk_per_lot": 500.0,
        "margin_per_lot": 10_000.0,
        "notional_per_lot": 1_000.0,
        "multiplier": 10.0,
        "pnl_ratio": 0.10,
        "tp1_hit": True,
        "trend_opportunity_quality_score": 55.0,
        "trend_structure_health_score": 70.0,
    }
    policy = TransitionPolicy("trial8_confirm30", trial_budget_margin_pct=0.08, confirm_budget_margin_pct=0.30)
    trial = candidate_for_transition_policy({**base_row, "transition_stage": "trial_low_to_medium"}, policy)
    confirm = candidate_for_transition_policy({**base_row, "transition_stage": "confirm_medium_to_high"}, policy)

    assert trial.phase1_story_budget_margin_pct == pytest.approx(0.08)
    assert confirm.phase1_story_budget_margin_pct == pytest.approx(0.30)
