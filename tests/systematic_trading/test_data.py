from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.data import audit_market_data, load_daily_cache_file, load_market_data_file  # noqa: E402


def test_load_daily_cache_file_normalizes_columns(tmp_path: Path) -> None:
    path = tmp_path / "daily_AG0_20260508_live.parquet"
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1200],
            "open_interest": [5000, 5100],
        }
    ).to_parquet(path)

    frame = load_daily_cache_file(path)

    assert list(frame.columns) == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]
    assert frame["symbol"].iloc[0] == "AG0"
    assert str(frame["date"].iloc[0].date()) == "2024-01-02"


def test_load_daily_cache_file_maps_oi_to_open_interest(tmp_path: Path) -> None:
    path = tmp_path / "daily_JD0_20260508_live.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1200],
            "oi": [5000, 5100],
        }
    ).to_parquet(path)

    frame = load_daily_cache_file(path)

    assert frame["open_interest"].tolist() == [5000, 5100]


def test_load_market_data_file_accepts_systematic_tq_main_file(tmp_path: Path) -> None:
    path = tmp_path / "AG0_tq_main_daily.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-18", "2026-05-19"]),
            "symbol": ["AG0", "AG0"],
            "name": ["白银", "白银"],
            "exchange": ["shfe", "shfe"],
            "tq_symbol": ["KQ.m@SHFE.ag", "KQ.m@SHFE.ag"],
            "open": [8000.0, 8010.0],
            "high": [8020.0, 8030.0],
            "low": [7990.0, 8000.0],
            "close": [8010.0, 8020.0],
            "volume": [1000, 1200],
            "open_interest": [5000, 5100],
        }
    ).to_parquet(path)

    frame = load_market_data_file(path)

    assert frame["symbol"].tolist() == ["AG0", "AG0"]
    assert frame["open_interest"].tolist() == [5000, 5100]


def test_audit_market_data_flags_short_history() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AG0", "AG0"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1200],
            "open_interest": [5000, 5100],
        }
    )

    audit = audit_market_data({"AG0": data}, min_history_days=260)

    assert audit.loc[0, "symbol"] == "AG0"
    assert audit.loc[0, "rows"] == 2
    assert bool(audit.loc[0, "is_tradeable"]) is False
    assert "short_history" in audit.loc[0, "issues"]
