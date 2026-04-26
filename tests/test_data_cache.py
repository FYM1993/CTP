from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data_cache import completed_daily_frame  # noqa: E402


def test_completed_daily_frame_drops_today_bar_during_trading_session() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-23", "2026-04-24"]),
            "open": [100.0, 101.0],
            "high": [102.0, 110.0],
            "low": [99.0, 98.0],
            "close": [101.0, 109.0],
            "volume": [1000.0, 300.0],
            "oi": [5000.0, 5050.0],
        }
    )

    completed = completed_daily_frame(df, now=datetime(2026, 4, 24, 10, 30))

    assert completed["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-23"]


def test_completed_daily_frame_drops_same_date_live_bar_during_night_session() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-14", "2026-04-15"]),
            "open": [100.0, 101.0],
            "high": [102.0, 110.0],
            "low": [99.0, 98.0],
            "close": [101.0, 109.0],
            "volume": [1000.0, 300.0],
            "oi": [5000.0, 5050.0],
        }
    )

    completed = completed_daily_frame(df, now=datetime(2026, 4, 15, 22, 0))

    assert completed["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-14"]


def test_completed_daily_frame_keeps_prior_completed_bar_before_open() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-23", "2026-04-24"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1200.0],
            "oi": [5000.0, 5050.0],
        }
    )

    completed = completed_daily_frame(df, now=datetime(2026, 4, 25, 8, 30))

    assert completed["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-23", "2026-04-24"]


def test_completed_daily_frame_drops_today_bar_before_close_even_before_open() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-23", "2026-04-24"]),
            "open": [100.0, 101.0],
            "high": [102.0, 110.0],
            "low": [99.0, 98.0],
            "close": [101.0, 109.0],
            "volume": [1000.0, 300.0],
            "oi": [5000.0, 5050.0],
        }
    )

    completed = completed_daily_frame(df, now=datetime(2026, 4, 24, 8, 30))

    assert completed["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-23"]
