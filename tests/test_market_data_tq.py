from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from market_data_tq import (  # noqa: E402
    klines_to_daily_frame,
    symbol_to_tq_main,
)
import data_cache  # noqa: E402
import market_data_tq  # noqa: E402


def test_symbol_to_tq_main_uses_exchange_table():
    assert symbol_to_tq_main("LH0", "dce") == "KQ.m@DCE.lh"


def test_symbol_to_tq_main_raises_value_error_for_unknown_exchange():
    try:
        symbol_to_tq_main("LH0", "xxx")
    except ValueError as exc:
        assert str(exc) == "未知交易所: xxx"
    else:
        raise AssertionError("expected ValueError")


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


def test_klines_to_daily_frame_defaults_oi_to_zero_when_missing():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-04-15"]),
            "open": [100],
            "high": [110],
            "low": [90],
            "close": [105],
            "volume": [1234],
        }
    )
    out = klines_to_daily_frame(frame)
    assert list(out["oi"]) == [0.0]


def test_get_daily_uses_tq_daily_standardized_columns(monkeypatch, tmp_path):
    tq_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-15"]),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1234.0],
            "oi": [888.0],
        }
    )

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(
        data_cache,
        "fetch_daily_from_tq",
        lambda symbol, days: tq_frame.copy(),
        raising=False,
    )

    def _raise_if_api_used(*args, **kwargs):
        raise AssertionError("fallback API should not be used when Tq data is available")

    monkeypatch.setattr(data_cache, "_fetch_daily_from_api", _raise_if_api_used)

    out = data_cache.get_daily("LH0", 30)

    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert float(out.iloc[0]["close"]) == 105.0


def test_choose_cache_suffix_keeps_same_day_data_live_before_close(monkeypatch):
    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-15 10:00:00").to_pydatetime()

    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_is_after_close", lambda: False)

    df = pd.DataFrame({"date": pd.to_datetime(["2026-04-15"])})

    assert data_cache._choose_cache_suffix(df) == "live"


def test_get_daily_ignores_stale_live_cache_after_close(monkeypatch, tmp_path):
    today = pd.Timestamp("2026-04-15")
    live_path = tmp_path / f"daily_LH0_{today.strftime('%Y%m%d')}_live.parquet"
    stale_frame = pd.DataFrame(
        {
            "date": [today],
            "open": [90.0],
            "high": [95.0],
            "low": [85.0],
            "close": [92.0],
            "volume": [100.0],
            "oi": [50.0],
        }
    )
    stale_frame.to_parquet(live_path, index=False)

    fresh_frame = pd.DataFrame(
        {
            "date": [today],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1234.0],
            "oi": [888.0],
        }
    )

    class FakeDatetime:
        @classmethod
        def now(cls):
            return today.to_pydatetime()

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_is_after_close", lambda: True)
    monkeypatch.setattr(data_cache, "fetch_daily_from_tq", lambda symbol, days: fresh_frame.copy())

    out = data_cache.get_daily("LH0", 30)

    assert float(out.iloc[0]["close"]) == 105.0


def test_get_daily_ignores_live_cache_on_weekend(monkeypatch, tmp_path):
    friday = pd.Timestamp("2026-04-17")
    saturday = pd.Timestamp("2026-04-18")
    live_path = tmp_path / f"daily_LH0_{saturday.strftime('%Y%m%d')}_live.parquet"
    stale_frame = pd.DataFrame(
        {
            "date": [friday],
            "open": [90.0],
            "high": [95.0],
            "low": [85.0],
            "close": [92.0],
            "volume": [100.0],
            "oi": [50.0],
        }
    )
    live_path.parent.mkdir(parents=True, exist_ok=True)
    stale_frame.to_parquet(live_path, index=False)

    fresh_frame = pd.DataFrame(
        {
            "date": [friday],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1234.0],
            "oi": [888.0],
        }
    )

    class FakeDatetime:
        @classmethod
        def now(cls):
            return saturday.to_pydatetime()

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "fetch_daily_from_tq", lambda symbol, days: fresh_frame.copy())

    out = data_cache.get_daily("LH0", 30)

    assert float(out.iloc[0]["close"]) == 105.0


def test_fetch_daily_from_tq_standardizes_oi_from_close_oi(monkeypatch):
    fake_tqsdk = ModuleType("tqsdk")

    class FakeTqAuth:
        def __init__(self, account, password):
            self.account = account
            self.password = password

    class FakeTqApi:
        def __init__(self, auth=None):
            self.auth = auth
            self.closed = False

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            assert symbol == "KQ.m@DCE.lh"
            assert duration_seconds == 86400
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-04-15"]),
                    "open": [100.0],
                    "high": [110.0],
                    "low": [90.0],
                    "close": [105.0],
                    "volume": [1234.0],
                    "close_oi": [888.0],
                }
            )

        def close(self):
            self.closed = True

    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqApi = FakeTqApi

    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)

    out = market_data_tq.fetch_daily_from_tq("LH0", "dce", 30, "acct", "pwd")

    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert float(out.iloc[0]["oi"]) == 888.0
