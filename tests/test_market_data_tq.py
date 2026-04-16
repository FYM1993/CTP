from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from market_data_tq import (  # noqa: E402
    klines_to_daily_frame,
    symbol_to_tq_main,
)


def test_symbol_to_tq_main_uses_exchange_table():
    assert symbol_to_tq_main("LH0", "dce") == "KQ.m@DCE.lh"


def test_klines_to_daily_frame_renames_columns():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-04-15"]),
            "open": [100],
            "high": [110],
            "low": [90],
            "close": [105],
            "volume": [1234],
            "close_oi": [888],
        }
    )
    out = klines_to_daily_frame(frame)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert float(out.iloc[0]["oi"]) == 888.0
