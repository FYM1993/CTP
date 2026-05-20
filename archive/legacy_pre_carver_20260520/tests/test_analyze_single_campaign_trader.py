from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_single_campaign_trader import (  # noqa: E402
    CampaignParams,
    iter_campaign_params,
    run_single_campaign_backtest,
    sniper_entry_score,
)


def _trade(
    trade_id: str,
    *,
    entry_time: str = "2025-01-01 09:00:00",
    exit_time: str = "2025-01-12 15:00:00",
    entry_price: float = 100.0,
    exit_price: float = 120.0,
    stage: str = "trial_low_to_medium",
) -> dict:
    return {
        "trade_id": trade_id,
        "symbol": "AU0",
        "name": "AU0",
        "direction": "long",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": "tp2",
        "phase2_score": 50.0,
        "state_score": 55.0,
        "entry_admission_rr": 3.0,
        "entry_rr": 2.0,
        "trend_opportunity_quality_score": 55.0,
        "trend_structure_health_score": 55.0,
        "trend_remaining_space_score": 55.0,
        "trend_participation_score": 55.0,
        "risk_per_lot": 10.0,
        "margin_per_lot": 1000.0,
        "notional_per_lot": entry_price * 10.0,
        "multiplier": 10.0,
        "commission_per_lot": 0.0,
        "commission_rate": 0.0,
        "transition_stage": stage,
    }


def _state(date: str, bucket: str, close: float, *, score: float | None = None) -> dict:
    return {
        "symbol": "AU0",
        "direction": "long",
        "date": date,
        "bucket": bucket,
        "score": 50.0 if score is None else score,
        "close": close,
    }


def test_single_campaign_allows_only_one_active_trade() -> None:
    trades = [
        _trade("first", entry_time="2025-01-01 09:00:00", exit_time="2025-01-12 15:00:00"),
        _trade("overlap", entry_time="2025-01-03 09:00:00", exit_time="2025-01-08 15:00:00"),
    ]
    states = [
        _state("2025-01-03", "medium", 102.0),
        _state("2025-01-05", "high", 108.0),
        _state("2025-01-08", "high", 110.0),
    ]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(trial_confirm_days=2, late_high_days=20),
    )

    assert result.summary["campaigns"] == 1
    assert result.summary["skipped_busy"] == 1
    assert result.campaigns[0]["trade_id"] == "first"


def test_campaign_scales_up_on_medium_and_high_then_reduces_late_high() -> None:
    trades = [_trade("scale")]
    states = [
        _state("2025-01-03", "medium", 105.0),
        _state("2025-01-05", "high", 110.0),
        _state("2025-01-10", "high", 120.0),
    ]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(
            trial_margin_pct=0.05,
            medium_margin_pct=0.15,
            high_margin_pct=0.30,
            late_high_margin_pct=0.10,
            trial_confirm_days=2,
            late_high_days=3,
        ),
    )

    actions = [(event["action"], event["target_lots"]) for event in result.events]
    assert ("open", 50) in actions
    assert ("raise_to_medium", 150) in actions
    assert ("raise_to_high", 300) in actions
    assert ("reduce_late_high", 100) in actions
    assert result.campaigns[0]["max_lots"] == 300
    assert result.campaigns[0]["final_lots"] == 0


def test_trial_exits_when_trend_does_not_confirm() -> None:
    trades = [_trade("failed_trial", exit_time="2025-01-12 15:00:00")]
    states = [
        _state("2025-01-02", "medium", 101.0),
        _state("2025-01-04", "low", 96.0),
    ]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(trial_confirm_days=2),
    )

    assert result.summary["campaigns"] == 1
    assert result.campaigns[0]["exit_reason"] == "trial_failed_state_low"
    assert result.campaigns[0]["exit_time"].startswith("2025-01-04")
    assert result.summary["early_campaign_exits"] == 1


def test_campaign_reduces_when_phase1_state_becomes_crowded() -> None:
    trades = [_trade("crowded")]
    states = [
        _state("2025-01-03", "medium", 105.0),
        _state("2025-01-05", "crowded", 112.0),
    ]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(
            trial_margin_pct=0.05,
            medium_margin_pct=0.30,
            high_margin_pct=0.50,
            late_high_margin_pct=0.10,
            trial_confirm_days=2,
        ),
    )

    actions = [(event["action"], event["target_lots"]) for event in result.events]
    assert ("raise_to_medium", 300) in actions
    assert ("reduce_crowded", 100) in actions
    assert result.summary["reduce_crowded_cut_events"] == 1


def test_confirm_start_uses_high_budget_for_direct_mainline_entry() -> None:
    trades = [
        _trade(
            "confirm",
            stage="confirm_medium_to_high",
            entry_time="2025-01-03 09:00:00",
            exit_time="2025-01-08 15:00:00",
        )
    ]
    states = [_state("2025-01-04", "high", 106.0)]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(
            trial_margin_pct=0.05,
            medium_margin_pct=0.30,
            high_margin_pct=0.50,
            max_margin_pct=1.0,
            allow_confirm_start=True,
            trial_confirm_days=2,
        ),
    )

    assert result.events[0]["action"] == "open"
    assert result.events[0]["target_lots"] == 500
    assert result.campaigns[0]["reached_high"] is True


def test_margin_cap_reduces_position_when_equity_drawdown_lifts_observed_usage() -> None:
    trades = [
        _trade(
            "cap_guard",
            stage="confirm_medium_to_high",
            entry_time="2025-01-01 09:00:00",
            exit_time="2025-01-12 15:00:00",
            entry_price=100.0,
            exit_price=100.0,
        )
    ]
    states = [_state("2025-01-02", "high", 20.0)]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(
            high_margin_pct=0.50,
            max_margin_pct=0.50,
            allow_confirm_start=True,
            risk_per_trade_pct=1.0,
            trial_confirm_days=1,
        ),
    )

    cap_events = [event for event in result.events if event["action"] == "reduce_margin_cap"]
    assert cap_events
    assert cap_events[0]["target_lots"] == 300
    assert result.summary["reduce_margin_cap_events"] == 1
    assert result.summary["max_margin_pct_observed"] <= 0.50


def test_max_campaigns_per_year_limits_sniper_trades() -> None:
    trades = [
        _trade("one", entry_time="2025-01-01 09:00:00", exit_time="2025-01-03 15:00:00"),
        _trade("two", entry_time="2025-02-01 09:00:00", exit_time="2025-02-03 15:00:00"),
        _trade("three", entry_time="2026-01-01 09:00:00", exit_time="2026-01-03 15:00:00"),
    ]
    states = [
        _state("2025-01-02", "high", 110.0),
        _state("2025-02-02", "high", 110.0),
        _state("2026-01-02", "high", 110.0),
    ]

    result = run_single_campaign_backtest(
        trades,
        states,
        CampaignParams(max_campaigns_per_year=1, trial_confirm_days=1),
    )

    assert [row["trade_id"] for row in result.campaigns] == ["one", "three"]
    assert result.summary["skipped_year_quota"] == 1


def test_min_sniper_score_filters_weak_entry_context_without_looking_at_outcome() -> None:
    weak = _trade("weak")
    strong = _trade("strong", entry_time="2025-02-01 09:00:00", exit_time="2025-02-03 15:00:00")
    strong.update(
        {
            "state_score": 82.0,
            "phase2_score": 75.0,
            "entry_admission_rr": 6.0,
            "trend_opportunity_quality_score": 80.0,
            "trend_structure_health_score": 85.0,
            "trend_remaining_space_score": 90.0,
            "trend_participation_score": 80.0,
        }
    )
    states = [
        _state("2025-01-02", "high", 110.0),
        _state("2025-02-02", "high", 110.0),
    ]

    result = run_single_campaign_backtest(
        [weak, strong],
        states,
        CampaignParams(min_sniper_score=70.0, trial_confirm_days=1),
    )

    assert sniper_entry_score(strong) > sniper_entry_score(weak)
    assert [row["trade_id"] for row in result.campaigns] == ["strong"]
    assert result.summary["skipped_sniper_score"] == 1


def test_min_mainline_score_filters_without_yearly_quota() -> None:
    weak = _trade("weak", entry_time="2025-01-01 09:00:00", exit_time="2025-01-03 15:00:00")
    weak["mainline_score"] = 62.0
    strong = _trade("strong", entry_time="2025-02-01 09:00:00", exit_time="2025-02-03 15:00:00")
    strong["mainline_score"] = 78.0
    states = [
        _state("2025-01-02", "high", 110.0),
        _state("2025-02-02", "high", 110.0),
    ]

    result = run_single_campaign_backtest(
        [weak, strong],
        states,
        CampaignParams(min_mainline_score=70.0, max_campaigns_per_year=0, trial_confirm_days=1),
    )

    assert [row["trade_id"] for row in result.campaigns] == ["strong"]
    assert result.summary["skipped_mainline_score"] == 1
    assert result.summary["skipped_year_quota"] == 0


def test_transition_event_quota_limits_repeated_attempts_without_yearly_quota() -> None:
    first = _trade("first", entry_time="2025-01-01 09:00:00", exit_time="2025-01-03 15:00:00")
    first["transition_event_id"] = "AU0_long_20250101_medium_to_high"
    second = _trade("second", entry_time="2025-02-01 09:00:00", exit_time="2025-02-03 15:00:00")
    second["transition_event_id"] = "AU0_long_20250101_medium_to_high"
    states = [
        _state("2025-01-02", "high", 110.0),
        _state("2025-02-02", "high", 110.0),
    ]

    result = run_single_campaign_backtest(
        [first, second],
        states,
        CampaignParams(max_campaigns_per_transition_event=1, max_campaigns_per_year=0, trial_confirm_days=1),
    )

    assert [row["trade_id"] for row in result.campaigns] == ["first"]
    assert result.summary["skipped_transition_event_quota"] == 1
    assert result.summary["skipped_year_quota"] == 0


def test_selective_constrained_params_keep_margin_caps_as_hard_budget() -> None:
    params = list(iter_campaign_params("selective_constrained"))

    assert params
    assert {round(item.max_margin_pct, 2) for item in params} == {0.30, 0.50}
    assert all(item.high_margin_pct <= item.max_margin_pct for item in params)
    assert all(item.medium_margin_pct <= item.max_margin_pct for item in params)
    assert all(item.trial_margin_pct <= item.max_margin_pct for item in params)
    assert all(item.late_high_margin_pct <= item.high_margin_pct for item in params)
    assert all(item.allow_confirm_start for item in params)
    assert all(item.max_campaigns_per_year == 0 for item in params)
    assert all(item.max_campaigns_per_transition_event == 1 for item in params)
