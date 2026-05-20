from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_trend_windows import (  # noqa: E402
    WindowPolicy,
    _policy_label,
    assign_trades_to_windows,
    identify_trend_windows,
)


def test_identify_trend_windows_starts_after_consecutive_high_days_and_uses_grace_to_end() -> None:
    state_rows = [
        {"symbol": "AU0", "date": "2025-01-01", "direction": "long", "is_high": True, "score": 70.0},
        {"symbol": "AU0", "date": "2025-01-02", "direction": "long", "is_high": True, "score": 72.0},
        {"symbol": "AU0", "date": "2025-01-03", "direction": "long", "is_high": True, "score": 75.0},
        {"symbol": "AU0", "date": "2025-01-04", "direction": "long", "is_high": False, "score": 62.0},
        {"symbol": "AU0", "date": "2025-01-05", "direction": "long", "is_high": False, "score": 61.0},
        {"symbol": "AU0", "date": "2025-01-06", "direction": "long", "is_high": False, "score": 60.0},
    ]

    windows = identify_trend_windows(state_rows, min_consecutive_high_days=3, end_grace_days=2)

    assert len(windows) == 1
    assert windows[0]["window_id"] == "AU0_long_20250103"
    assert windows[0]["signal_start_date"] == "2025-01-01"
    assert windows[0]["active_from_date"] == "2025-01-03"
    assert windows[0]["end_date"] == "2025-01-03"
    assert windows[0]["high_days"] == 3
    assert windows[0]["calendar_days"] == 3
    assert windows[0]["activation_score"] == 75.0
    assert round(windows[0]["signal_avg_score"], 2) == 72.33


def test_identify_trend_windows_ends_on_direction_flip_and_can_start_new_window() -> None:
    state_rows = [
        {"symbol": "AG0", "date": "2025-01-01", "direction": "long", "is_high": True, "score": 70.0},
        {"symbol": "AG0", "date": "2025-01-02", "direction": "long", "is_high": True, "score": 71.0},
        {"symbol": "AG0", "date": "2025-01-03", "direction": "short", "is_high": True, "score": 72.0},
        {"symbol": "AG0", "date": "2025-01-04", "direction": "short", "is_high": True, "score": 74.0},
    ]

    windows = identify_trend_windows(state_rows, min_consecutive_high_days=2, end_grace_days=1)

    assert [row["direction"] for row in windows] == ["long", "short"]
    assert windows[0]["end_date"] == "2025-01-02"
    assert windows[1]["active_from_date"] == "2025-01-04"


def test_assign_trades_to_windows_keeps_only_first_n_trades_per_symbol_window() -> None:
    windows = [
        {
            "window_id": "AU0_long_20250103",
            "symbol": "AU0",
            "direction": "long",
            "active_from_date": "2025-01-03",
            "end_date": "2025-01-20",
        }
    ]
    trades = [
        {"trade_id": "a", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-03 09:00:00"},
        {"trade_id": "b", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-04 09:00:00"},
        {"trade_id": "c", "symbol": "AU0", "direction": "long", "entry_time": "2025-01-05 09:00:00"},
        {"trade_id": "d", "symbol": "AU0", "direction": "short", "entry_time": "2025-01-06 09:00:00"},
    ]

    assigned = assign_trades_to_windows(trades, windows, max_trades_per_window=2)

    assert [row["trade_id"] for row in assigned] == ["a", "b"]
    assert [row["window_trade_seq"] for row in assigned] == [1, 2]
    assert all(row["window_id"] == "AU0_long_20250103" for row in assigned)


def test_policy_label_describes_window_trade_limit_and_first_fraction() -> None:
    assert _policy_label(WindowPolicy(0.30, 1, 1.0)) == "m30_window1_first100"
    assert _policy_label(WindowPolicy(0.30, 2, 0.70)) == "m30_window2_first70"
