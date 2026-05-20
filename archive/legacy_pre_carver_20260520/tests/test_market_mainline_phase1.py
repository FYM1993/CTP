from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_market_mainline_phase1 import (  # noqa: E402
    assign_trades_to_mainline_stages,
    build_mainline_events,
    build_mainline_episodes,
    select_representative_episode_trades,
)
from backtest.market_mainline_phase1 import MarketMainlineParams, build_market_mainline_state_rows  # noqa: E402


def _daily_frame(start: str, periods: int, *, first: float, daily_change: float, volume_step: float, oi_step: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    close = [first + daily_change * idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [1000.0 + volume_step * idx for idx in range(periods)],
            "oi": [5000.0 + oi_step * idx for idx in range(periods)],
        }
    )


def test_market_mainline_scores_group_leadership_and_capital_confirmation() -> None:
    market_data = {
        "AU0": _daily_frame("2024-10-01", 130, first=100.0, daily_change=1.0, volume_step=20.0, oi_step=15.0),
        "AG0": _daily_frame("2024-10-01", 130, first=80.0, daily_change=0.7, volume_step=15.0, oi_step=10.0),
        "CU0": _daily_frame("2024-10-01", 130, first=120.0, daily_change=-0.1, volume_step=1.0, oi_step=1.0),
    }

    rows = build_market_mainline_state_rows(market_data, params=MarketMainlineParams(start_year=2025, end_year=2025))
    latest_long = [
        row for row in rows if row["symbol"] == "AU0" and row["direction"] == "long" and row["date"] == "2025-02-07"
    ][0]
    latest_short = [
        row for row in rows if row["symbol"] == "AU0" and row["direction"] == "short" and row["date"] == "2025-02-07"
    ][0]

    assert latest_long["mainline_score"] > latest_short["mainline_score"]
    assert latest_long["mainline_capital_score"] >= 70.0
    assert latest_long["mainline_resonance_score"] >= 70.0
    assert latest_long["bucket"] in {"medium", "high", "crowded"}
    assert latest_long["term_structure_status"] == "not_available_in_continuous_cache"


def test_market_mainline_handles_symbols_with_uneven_history_lengths() -> None:
    market_data = {
        "AU0": _daily_frame("2024-10-01", 130, first=100.0, daily_change=1.0, volume_step=20.0, oi_step=15.0),
        "AG0": _daily_frame("2024-10-03", 128, first=80.0, daily_change=0.7, volume_step=15.0, oi_step=10.0),
        "CU0": _daily_frame("2024-10-02", 129, first=120.0, daily_change=-0.1, volume_step=1.0, oi_step=1.0),
    }

    rows = build_market_mainline_state_rows(market_data, params=MarketMainlineParams(start_year=2025, end_year=2025))

    assert rows
    assert {row["symbol"] for row in rows} == {"AU0", "AG0", "CU0"}
    assert {row["direction"] for row in rows} == {"long", "short"}


def test_assign_trades_maps_market_mainline_transitions_to_trial_and_confirm() -> None:
    state_rows = [
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 42.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-02", "bucket": "medium", "score": 58.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 72.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-04", "bucket": "crowded", "score": 75.0},
    ]
    events = build_mainline_events(state_rows)
    trades = [
        {"trade_id": "trial", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-02 09:00:00"},
        {"trade_id": "confirm", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-03 09:00:00"},
        {"trade_id": "crowded", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-04 09:00:00"},
    ]

    assigned = assign_trades_to_mainline_stages(trades, state_rows, events, transition_window_days=20)

    assert [row["transition_stage"] for row in assigned] == [
        "trial_low_to_medium",
        "confirm_medium_to_high",
        "crowded_late",
    ]
    assert assigned[0]["transition_event_id"].startswith("precious_metals_long_")


def test_mainline_events_are_group_level_not_symbol_level() -> None:
    state_rows = [
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 40.0},
        {"symbol": "AG0", "group": "precious_metals", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 40.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-02", "bucket": "high", "score": 78.0},
        {"symbol": "AG0", "group": "precious_metals", "direction": "long", "date": "2025-01-02", "bucket": "high", "score": 78.0},
    ]

    events = build_mainline_events(state_rows)

    assert len(events) == 1
    assert events[0]["event_id"] == "precious_metals_long_20250102_low_to_high"


def test_mainline_episode_merges_group_state_noise_until_sustained_low_reset() -> None:
    state_rows = [
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 40.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-02", "bucket": "medium", "score": 62.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 78.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-04", "bucket": "medium", "score": 66.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-05", "bucket": "low", "score": 45.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-06", "bucket": "medium", "score": 61.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-07", "bucket": "low", "score": 42.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-08", "bucket": "low", "score": 41.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-09", "bucket": "medium", "score": 63.0},
    ]

    episodes = build_mainline_episodes(state_rows, reset_low_days=2)

    assert [(row["episode_id"], row["start_date"], row["confirm_date"], row["end_date"]) for row in episodes] == [
        ("precious_metals_long_20250102", "2025-01-02", "2025-01-03", "2025-01-08"),
        ("precious_metals_long_20250109", "2025-01-09", "", ""),
    ]


def test_trades_in_same_group_episode_share_one_mainline_id() -> None:
    state_rows = [
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-01", "bucket": "low", "score": 40.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-02", "bucket": "medium", "score": 62.0},
        {"symbol": "AU0", "group": "precious_metals", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 78.0},
        {"symbol": "AG0", "group": "precious_metals", "direction": "long", "date": "2025-01-03", "bucket": "high", "score": 78.0},
    ]
    episodes = build_mainline_episodes(state_rows, reset_low_days=2)
    trades = [
        {"trade_id": "gold", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-03 09:00:00"},
        {"trade_id": "silver", "symbol": "AG0", "direction": "long", "entry_time": "2025-01-04 09:00:00"},
    ]

    assigned = assign_trades_to_mainline_stages(trades, state_rows, [], transition_window_days=20, episodes=episodes)

    assert [row["transition_event_id"] for row in assigned] == [
        "precious_metals_long_20250102",
        "precious_metals_long_20250102",
    ]
    assert [row["mainline_group"] for row in assigned] == ["precious_metals", "precious_metals"]


def test_representative_selection_keeps_one_trade_per_group_episode() -> None:
    rows = [
        {
            "trade_id": "first_non_leader",
            "symbol": "AU0",
            "entry_time": "2025-01-03 09:00:00",
            "transition_event_id": "precious_metals_long_20250102",
            "transition_stage": "confirm_medium_to_high",
            "group_leader_symbol": "AG0",
            "mainline_score": 82.0,
            "state_score": 82.0,
            "phase2_score": 70.0,
            "entry_admission_rr": 4.0,
            "trend_opportunity_quality_score": 82.0,
            "trend_structure_health_score": 80.0,
            "trend_remaining_space_score": 75.0,
            "trend_participation_score": 78.0,
        },
        {
            "trade_id": "leader",
            "symbol": "AG0",
            "entry_time": "2025-01-04 09:00:00",
            "transition_event_id": "precious_metals_long_20250102",
            "transition_stage": "confirm_medium_to_high",
            "group_leader_symbol": "AG0",
            "mainline_score": 82.0,
            "state_score": 82.0,
            "phase2_score": 68.0,
            "entry_admission_rr": 4.0,
            "trend_opportunity_quality_score": 82.0,
            "trend_structure_health_score": 80.0,
            "trend_remaining_space_score": 75.0,
            "trend_participation_score": 78.0,
        },
    ]

    selected = select_representative_episode_trades(rows)

    assert [row["trade_id"] for row in selected] == ["leader"]
