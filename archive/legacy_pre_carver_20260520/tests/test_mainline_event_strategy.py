from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mainline_event_strategy import (  # noqa: E402
    MainlineEventParams,
    classify_event_state,
    run_event_strategy,
)


def _weekly_row(
    *,
    status: str,
    latest_stage: str = "early_trial",
    active_days: int = 2,
    latest_score: float = 62.0,
    crowding: float = 0.0,
) -> dict:
    return {
        "weekly_status": status,
        "latest_stage": latest_stage,
        "active_days": active_days,
        "latest_mainline_score": latest_score,
        "crowding_change": crowding,
    }


def _state_row(
    date: str,
    symbol: str,
    close: float,
    *,
    group: str = "metals",
    direction: str = "long",
    bucket: str = "high",
    score: float = 75.0,
    breadth: float = 1.0,
    count: int = 2,
    leader: str | None = None,
) -> dict:
    return {
        "date": date,
        "symbol": symbol,
        "close": close,
        "group": group,
        "direction": direction,
        "bucket": bucket,
        "mainline_score": score,
        "score": score,
        "mainline_capital_score": 80.0,
        "mainline_resonance_score": 80.0,
        "mainline_crowding_score": 0.0,
        "mainline_leadership_score": score,
        "group_breadth": breadth,
        "group_symbol_count": count,
        "group_leader_symbol": leader or symbol,
        "group_best_symbol_score": score,
        "symbol_setup_score": score,
        "retrace_atr": 1.0,
    }


def test_event_state_maps_lifecycle_to_position_exposure() -> None:
    params = MainlineEventParams()

    assert (
        classify_event_state(_weekly_row(status="improving_consensus", latest_stage="early_trial", latest_score=58.0), params).state
        == "sprouting"
    )
    assert classify_event_state(_weekly_row(status="improving_consensus", latest_stage="confirming"), params).state == "confirmed"
    assert classify_event_state(_weekly_row(status="persistent_mainline", latest_stage="established", latest_score=78.0), params).state == "markup"
    assert classify_event_state(_weekly_row(status="late_cycle_risk", latest_stage="crowded", latest_score=86.0), params).state == "crowded"

    assert (
        classify_event_state(
            _weekly_row(status="improving_consensus", latest_stage="early_trial", latest_score=58.0), params
        ).target_exposure
        == params.sprouting_exposure
    )
    assert classify_event_state(_weekly_row(status="improving_consensus", latest_stage="confirming"), params).target_exposure == params.confirmed_exposure
    assert classify_event_state(_weekly_row(status="persistent_mainline", latest_stage="established", latest_score=78.0), params).target_exposure == params.markup_exposure
    assert classify_event_state(_weekly_row(status="late_cycle_risk", latest_stage="crowded", latest_score=86.0), params).target_exposure == params.crowded_exposure


def test_event_strategy_holds_one_mainline_leader_until_trend_is_lost() -> None:
    rows = []
    prices = [100, 102, 104, 108, 112, 110, 109]
    for idx, price in enumerate(prices, start=1):
        date = f"2025-01-0{idx}"
        bucket = "medium" if idx <= 2 else "high"
        score = 60.0 if idx <= 2 else 76.0
        if idx >= 6:
            bucket = "low"
            score = 48.0
        rows.append(_state_row(date, "AU0", price, bucket=bucket, score=score, leader="AU0"))
        rows.append(_state_row(date, "AG0", price * 0.8, bucket=bucket, score=score, leader="AU0"))
        rows.append(_state_row(date, "RB0", 200 - idx, group="black", direction="short", bucket="low", score=45.0, leader="RB0"))
        rows.append(_state_row(date, "HC0", 180 - idx, group="black", direction="short", bucket="low", score=45.0, leader="RB0"))

    result = run_event_strategy(rows, MainlineEventParams(top_n=1, sprouting_exposure=0.5, confirmed_exposure=1.0, markup_exposure=2.0))

    assert result.summary["max_simultaneous_positions"] == 1
    assert result.summary["trades"] >= 1
    assert {trade["entry_symbol"] for trade in result.trades}.issubset({"AU0", "RB0"})
    assert any(trade["exit_reason"] in {"mainline_lost", "rotated_to_stronger_mainline"} for trade in result.trades)
