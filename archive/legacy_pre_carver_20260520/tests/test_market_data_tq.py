from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from market.tq import (  # noqa: E402
    klines_to_daily_frame,
    resolve_tq_continuous_symbol,
    resolve_tq_continuous_symbol_info,
    symbol_to_tq_main,
    tq_underlying_to_continuous,
)
import data_cache  # noqa: E402
from market import tq as market_data_tq  # noqa: E402


def test_symbol_to_tq_main_uses_exchange_table():
    assert symbol_to_tq_main("LH0", "dce") == "KQ.m@DCE.lh"


def test_symbol_to_tq_main_uses_uppercase_product_for_czce():
    assert symbol_to_tq_main("TA0", "czce") == "KQ.m@CZCE.TA"


def test_tq_underlying_to_continuous_uses_returned_product_case():
    assert tq_underlying_to_continuous("DCE.lh2509") == "KQ.m@DCE.lh"
    assert tq_underlying_to_continuous("CZCE.TA609") == "KQ.m@CZCE.TA"


def test_resolve_tq_continuous_symbol_prefers_query_cont_quotes():
    class FakeApi:
        def __init__(self):
            self.calls = []

        def query_cont_quotes(self, exchange_id=None, product_id=None):
            self.calls.append((exchange_id, product_id))
            if exchange_id == "DCE" and product_id == "lh":
                return ["DCE.lh2509"]
            return []

    api = FakeApi()

    out = resolve_tq_continuous_symbol("LH0", "dce", api=api)

    assert out == "KQ.m@DCE.lh"
    assert api.calls == [("DCE", "LH"), ("DCE", "lh")]


def test_resolve_tq_continuous_symbol_info_records_query_source():
    class FakeApi:
        def query_cont_quotes(self, exchange_id=None, product_id=None):
            if exchange_id == "DCE" and product_id == "lh":
                return ["DCE.lh2509"]
            return []

    info = resolve_tq_continuous_symbol_info("LH0", "dce", api=FakeApi())

    assert info == {
        "continuous_symbol": "KQ.m@DCE.lh",
        "source": "query_cont_quotes",
        "underlying_symbol": "DCE.lh2509",
    }


def test_resolve_tq_continuous_symbol_falls_back_to_local_mapping_when_query_empty():
    class FakeApi:
        def query_cont_quotes(self, exchange_id=None, product_id=None):
            return []

    out = resolve_tq_continuous_symbol("TA0", "czce", api=FakeApi())

    assert out == "KQ.m@CZCE.TA"


def test_resolve_tq_continuous_symbol_info_records_fallback_source():
    class FakeApi:
        def query_cont_quotes(self, exchange_id=None, product_id=None):
            return []

    info = resolve_tq_continuous_symbol_info("TA0", "czce", api=FakeApi())

    assert info == {
        "continuous_symbol": "KQ.m@CZCE.TA",
        "source": "fallback",
        "underlying_symbol": "",
    }


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


def test_get_daily_skips_akshare_fallback_when_tq_is_configured_but_empty(monkeypatch, tmp_path):
    warnings = []

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "_tq_credentials", lambda config=None: ("acct", "pwd"))
    monkeypatch.setattr(data_cache, "fetch_daily_from_tq", lambda symbol, days: None)
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("configured Tq path should not fallback")),
    )
    monkeypatch.setattr(data_cache._log, "warning", lambda msg, *args: warnings.append(msg % args))

    out = data_cache.get_daily("PK0", 30)

    assert out is None
    assert any("PK0" in message and "TqSdk无数据，跳过AkShare回退" in message for message in warnings)


def test_choose_cache_suffix_keeps_same_day_data_live_before_close(monkeypatch):
    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-15 10:00:00").to_pydatetime()

    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_is_after_close", lambda: False)

    df = pd.DataFrame({"date": pd.to_datetime(["2026-04-15"])})

    assert data_cache._choose_cache_suffix(df) == "live"


def test_choose_cache_suffix_keeps_same_day_data_live_during_night_session(monkeypatch):
    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-15 22:00:00").to_pydatetime()

    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)

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


def test_get_daily_keeps_live_cache_during_night_session(monkeypatch, tmp_path):
    today = pd.Timestamp("2026-04-15")
    live_path = tmp_path / f"daily_LH0_{today.strftime('%Y%m%d')}_live.parquet"
    cached_frame = pd.DataFrame(
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
    live_path.parent.mkdir(parents=True, exist_ok=True)
    cached_frame.to_parquet(live_path, index=False)

    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-15 22:00:00").to_pydatetime()

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(
        data_cache,
        "fetch_daily_from_tq",
        lambda symbol, days: (_ for _ in ()).throw(AssertionError("night session should reuse live cache")),
    )

    out = data_cache.get_daily("LH0", 30)

    assert float(out.iloc[0]["close"]) == 92.0


def test_get_daily_refreshes_cache_when_requested_history_exceeds_cached_rows(monkeypatch, tmp_path):
    today = pd.Timestamp("2026-04-15")
    final_path = tmp_path / f"daily_LH0_{today.strftime('%Y%m%d')}_final.parquet"
    cached_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-14", "2026-04-15"]),
            "open": [90.0, 91.0],
            "high": [95.0, 96.0],
            "low": [85.0, 86.0],
            "close": [92.0, 93.0],
            "volume": [100.0, 101.0],
            "oi": [50.0, 51.0],
        }
    )
    cached_frame.to_parquet(final_path, index=False)

    fresh_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-11", "2026-04-12", "2026-04-13", "2026-04-14", "2026-04-15"]),
            "open": [80.0, 81.0, 82.0, 83.0, 84.0],
            "high": [85.0, 86.0, 87.0, 88.0, 89.0],
            "low": [75.0, 76.0, 77.0, 78.0, 79.0],
            "close": [82.0, 83.0, 84.0, 85.0, 86.0],
            "volume": [100.0, 101.0, 102.0, 103.0, 104.0],
            "oi": [50.0, 51.0, 52.0, 53.0, 54.0],
        }
    )

    class FakeDatetime:
        @classmethod
        def now(cls):
            return today.to_pydatetime()

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_tq_credentials", lambda config=None: ("acct", "pwd"))
    monkeypatch.setattr(data_cache, "fetch_daily_from_tq", lambda symbol, days: fresh_frame.copy())

    out = data_cache.get_daily("LH0", 5)

    assert len(out) == 5
    assert float(out.iloc[0]["close"]) == 82.0


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
            self.query_calls = []

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

        def query_cont_quotes(self, exchange_id=None, product_id=None):
            self.query_calls.append((exchange_id, product_id))
            if exchange_id == "DCE" and product_id == "lh":
                return ["DCE.lh2509"]
            return []

        def close(self):
            self.closed = True

    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqApi = FakeTqApi

    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)

    out = market_data_tq.fetch_daily_from_tq("LH0", "dce", 30, "acct", "pwd")

    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert float(out.iloc[0]["oi"]) == 888.0


def test_fetch_daily_from_tq_waits_for_initial_update(monkeypatch):
    fake_tqsdk = ModuleType("tqsdk")

    class FakeTqAuth:
        def __init__(self, account, password):
            self.account = account
            self.password = password

    class FakeTqApi:
        instances = []

        def __init__(self, auth=None):
            self.auth = auth
            self.closed = False
            self.wait_calls = 0
            self.frame = None
            FakeTqApi.instances.append(self)

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            assert symbol == "KQ.m@DCE.lh"
            assert duration_seconds == 86400
            self.frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-04-15"]),
                    "open": [100.0],
                    "high": [110.0],
                    "low": [90.0],
                    "close": [float("nan")],
                    "volume": [1234.0],
                    "close_oi": [888.0],
                }
            )
            return self.frame

        def wait_update(self, deadline=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                self.frame.loc[0, "close"] = 105.0
                return True
            return False

        def close(self):
            self.closed = True

    fake_tqsdk.TqAuth = FakeTqAuth
    fake_tqsdk.TqApi = FakeTqApi

    monkeypatch.setitem(sys.modules, "tqsdk", fake_tqsdk)

    out = market_data_tq.fetch_daily_from_tq("LH0", "dce", 30, "acct", "pwd")

    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 105.0
    assert FakeTqApi.instances[0].wait_calls >= 1
    assert FakeTqApi.instances[0].closed is True


def test_prefetch_all_reuses_single_tq_session(monkeypatch, tmp_path):
    class FakeApi:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    created_apis = []
    used_apis = []

    def fake_create_tq_api(account, password):
        assert account == "acct"
        assert password == "pwd"
        api = FakeApi()
        created_apis.append(api)
        return api

    def fake_tq_fetch(symbol, exchange, days, account, password, api=None):
        used_apis.append(api)
        return pd.DataFrame(
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
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(data_cache, "_create_tq_api", fake_create_tq_api, raising=False)
    monkeypatch.setattr(data_cache, "_tq_fetch_daily_from_tq", fake_tq_fetch)
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fallback to AkShare")),
    )

    out = data_cache.prefetch_all(
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ]
    )

    assert set(out) == {"LH0", "M0"}
    assert len(created_apis) == 1
    assert used_apis == [created_apis[0], created_apis[0]]
    assert created_apis[0].closed is True


def test_prefetch_all_does_not_create_tq_session_when_all_symbols_hit_cache(monkeypatch, tmp_path):
    cache_frame = pd.DataFrame(
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
    for symbol in ("LH0", "M0"):
        cache_frame.to_parquet(
            tmp_path / f"daily_{symbol}_20260417_live.parquet",
            index=False,
        )

    create_calls = []

    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-17 10:57:04").to_pydatetime()

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(
        data_cache,
        "_create_tq_api",
        lambda *args, **kwargs: create_calls.append((args, kwargs)),
        raising=False,
    )

    out = data_cache.prefetch_all(
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ]
    )

    assert set(out) == {"LH0", "M0"}
    assert create_calls == []


def test_prefetch_all_skips_per_symbol_tq_retry_after_session_init_failure(monkeypatch, tmp_path):
    fallback_calls = []

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(
        data_cache,
        "_create_tq_api",
        lambda account, password: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        data_cache,
        "fetch_daily_from_tq",
        lambda symbol, days: (_ for _ in ()).throw(AssertionError("should not retry per symbol Tq fetch")),
    )

    def fake_fallback(symbol, days):
        fallback_calls.append((symbol, days))
        return pd.DataFrame(
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

    monkeypatch.setattr(data_cache, "_fetch_daily_from_api", fake_fallback)

    out = data_cache.prefetch_all(
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ]
    )

    assert set(out) == {"LH0", "M0"}
    assert fallback_calls == [("LH0", 400), ("M0", 400)]


def test_prefetch_all_skips_akshare_fallback_when_tq_session_is_available_but_symbol_empty(
    monkeypatch,
    tmp_path,
):
    warnings = []

    class FakeApi:
        def close(self):
            pass

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(data_cache, "_create_tq_api", lambda account, password: FakeApi(), raising=False)
    monkeypatch.setattr(data_cache, "_fetch_daily_from_tq_with_api", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("per-symbol empty Tq should not fallback")),
    )
    monkeypatch.setattr(data_cache._log, "warning", lambda msg, *args: warnings.append(msg % args))

    out = data_cache.prefetch_all(
        symbols=[{"symbol": "PK0", "name": "花生", "exchange": "czce"}]
    )

    assert out == {}
    assert any("PK0" in message and "TqSdk无数据，跳过AkShare回退" in message for message in warnings)


def test_fetch_daily_from_api_logs_retry_and_uses_socket_timeout(monkeypatch):
    warnings = []
    timeout_values = []
    sleep_calls = []

    @contextmanager
    def fake_socket_timeout(timeout):
        timeout_values.append(timeout)
        yield

    monkeypatch.setattr(data_cache, "_socket_timeout", fake_socket_timeout, raising=False)
    monkeypatch.setattr(data_cache, "_throttle", lambda: None)
    monkeypatch.setattr(data_cache.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(
        data_cache.aks,
        "futures_main_sina",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("api down")),
    )
    monkeypatch.setattr(data_cache._log, "warning", lambda msg, *args: warnings.append(msg % args))

    out = data_cache._fetch_daily_from_api("PK0", 400, retries=2)

    assert out is None
    assert timeout_values == [8.0, 8.0]
    assert sleep_calls == [2]
    assert len(warnings) == 2
    assert "PK0" in warnings[0]
    assert "AkShare日线获取失败" in warnings[0]


def test_prefetch_all_logs_when_global_tq_is_unavailable_and_falls_back_to_akshare(monkeypatch, tmp_path):
    warnings = []

    monkeypatch.setattr(data_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data_cache, "_cache_cleaned", False)
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(
        data_cache,
        "_create_tq_api",
        lambda account, password: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_api",
        lambda symbol, days: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-15"]),
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "volume": [1234.0],
                "oi": [888.0],
            }
        ),
    )
    monkeypatch.setattr(data_cache._log, "warning", lambda msg, *args: warnings.append(msg % args))

    out = data_cache.prefetch_all(
        symbols=[{"symbol": "PK0", "name": "花生", "exchange": "czce"}]
    )

    assert set(out) == {"PK0"}
    assert any("TqSdk 初始化失败" in message for message in warnings)


def test_prefetch_all_with_stats_reports_tq_fetches_and_non_today_bars(monkeypatch, tmp_path):
    infos = []

    class FakeApi:
        def close(self):
            pass

    class FakeDatetime:
        @classmethod
        def now(cls):
            return pd.Timestamp("2026-04-22 09:23:39").to_pydatetime()

    fetched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-21"]),
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
    monkeypatch.setattr(data_cache, "datetime", FakeDatetime)
    monkeypatch.setattr(data_cache, "_load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(data_cache, "_create_tq_api", lambda account, password: FakeApi(), raising=False)
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_tq_with_api",
        lambda *args, **kwargs: fetched.copy(),
    )
    monkeypatch.setattr(
        data_cache,
        "_fetch_daily_from_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fallback to AkShare")),
    )
    monkeypatch.setattr(data_cache._log, "info", lambda msg, *args: infos.append(msg % args if args else msg))

    out, stats = data_cache.prefetch_all_with_stats(
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ]
    )

    assert set(out) == {"LH0", "M0"}
    assert stats.cache_hits == 0
    assert stats.tq_fetches == 2
    assert stats.akshare_fetches == 0
    assert stats.latest_bar_not_today == 2
    assert any("TqSdk拉取2" in message for message in infos)
    assert any("最新bar非今日2" in message for message in infos)


def test_get_hog_fundamentals_includes_spot_price_percentile(monkeypatch):
    monkeypatch.setattr(data_cache, "_hog_cache", None)
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_core",
        lambda symbol="外三元": pd.DataFrame({"date": ["2026-04-16", "2026-04-17", "2026-04-18", "2026-04-19"], "value": [20.0, 15.0, 10.0, 12.0]}),
    )
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_cost",
        lambda symbol="玉米": pd.DataFrame({"date": ["2026-04-19"], "value": [2400.0]}),
    )
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_supply",
        lambda symbol="猪肉批发价": pd.DataFrame({"date": ["2026-04-19"], "value": [100.0]}),
    )

    out = data_cache.get_hog_fundamentals()

    assert out is not None
    assert out["price"] == 12.0
    assert round(out["spot_price_percentile"], 2) == 20.0


def test_get_inventory_historical_replay_returns_none(monkeypatch):
    monkeypatch.setattr(
        data_cache.aks,
        "futures_inventory_em",
        lambda symbol="a": (_ for _ in ()).throw(AssertionError("历史库存回放不应请求东方财富")),
    )

    out = data_cache.get_inventory("A0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out is None


def test_get_inventory_live_uses_eastmoney_source(monkeypatch):
    monkeypatch.setattr(
        data_cache.aks,
        "futures_inventory_em",
        lambda symbol="a": pd.DataFrame(
            {
                "日期": [f"2026-04-{day:02d}" for day in range(10, 20)],
                "库存": [70.0, 72.0, 75.0, 78.0, 80.0, 85.0, 90.0, 100.0, 130.0, 120.0],
                "增减": [0.0, 2.0, 3.0, 3.0, 2.0, 5.0, 5.0, 10.0, 30.0, -10.0],
            }
        ),
    )

    out = data_cache.get_inventory("A0")

    assert out is not None
    assert out["inv_now"] == 120.0
    assert round(out["inv_change_4wk"], 2) == 41.18
    assert out["inv_cumulating_weeks"] == 7
    assert out["inv_trend"] == "累库"


def test_get_spot_basis_maps_internal_symbol_to_commodity_code(monkeypatch):
    seen: list[tuple[str, list[str]]] = []

    def fake_spot_price(date="20240430", vars_list=None):
        seen.append((date, list(vars_list or [])))
        return pd.DataFrame(
            {
                "var": ["RB"],
                "sp": [3520.0],
                "dom_price": [3480.0],
                "dom_basis": [40.0],
                "dom_basis_rate": [1.15],
                "date": ["20250401"],
            }
        )

    monkeypatch.setattr(data_cache.aks, "futures_spot_price", fake_spot_price)

    out = data_cache.get_spot_basis("RB0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert seen == [("20250401", ["RB"])]
    assert out == {
        "commodity_code": "RB",
        "spot_price": 3520.0,
        "dominant_contract_price": 3480.0,
        "basis": 40.0,
        "basis_rate": 1.15,
        "data_date": "20250401",
        "source": "akshare_futures_spot_price",
    }


def test_get_spot_basis_accepts_akshare_normalized_schema(monkeypatch):
    def fake_spot_price(date="20240430", vars_list=None):
        return pd.DataFrame(
            {
                "date": ["20250401"],
                "symbol": ["RB"],
                "spot_price": [3181.33],
                "dominant_contract_price": [3170.0],
                "dom_basis": [-11.33],
                "dom_basis_rate": [-0.003561],
            }
        )

    monkeypatch.setattr(data_cache.aks, "futures_spot_price", fake_spot_price)

    out = data_cache.get_spot_basis("RB0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out == {
        "commodity_code": "RB",
        "spot_price": 3181.33,
        "dominant_contract_price": 3170.0,
        "basis": -11.33,
        "basis_rate": -0.003561,
        "data_date": "20250401",
        "source": "akshare_futures_spot_price",
    }


def test_get_spot_basis_returns_none_when_symbol_row_missing(monkeypatch):
    monkeypatch.setattr(
        data_cache.aks,
        "futures_spot_price",
        lambda date="20240430", vars_list=None: pd.DataFrame(
            {
                "var": ["HC"],
                "sp": [3600.0],
                "dom_price": [3550.0],
                "dom_basis": [50.0],
                "dom_basis_rate": [1.4],
                "date": ["20250401"],
            }
        ),
    )

    out = data_cache.get_spot_basis("RB0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out is None


def test_get_spot_basis_returns_none_when_source_fails(monkeypatch):
    def fail_spot_price(date="20240430", vars_list=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(data_cache.aks, "futures_spot_price", fail_spot_price)

    out = data_cache.get_spot_basis("RB0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out is None


def test_get_warehouse_receipt_uses_as_of_date_snapshot(monkeypatch):
    monkeypatch.setattr(data_cache, "_shfe_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_gfex_receipt_cache", {})
    seen: list[str] = []

    def fake_shfe(date="20200702"):
        seen.append(date)
        return {
            "天然橡胶": pd.DataFrame(
                {
                    "WRTWGHTS": [20.0, 30.0],
                    "WRTCHANGE": [2.0, -1.0],
                }
            )
        }

    monkeypatch.setattr(data_cache.aks, "futures_shfe_warehouse_receipt", fake_shfe)

    out = data_cache.get_warehouse_receipt("RU0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out == {"receipt_total": 50.0, "receipt_change": 1.0, "exchange": "SHFE"}
    assert seen == ["20250401"]


def test_get_warehouse_receipt_supports_dce_symbols(monkeypatch):
    monkeypatch.setattr(data_cache, "_shfe_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_gfex_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_dce_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_czce_receipt_cache", {})
    seen: list[str] = []

    def fake_dce(date="20200702"):
        seen.append(date)
        return {
            "豆粕": pd.DataFrame(
                {
                    "今日仓单量": [100.0, 30.0],
                    "增减": [5.0, -2.0],
                }
            )
        }

    monkeypatch.setattr(data_cache.aks, "futures_dce_warehouse_receipt", fake_dce)

    out = data_cache.get_warehouse_receipt("M0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out == {"receipt_total": 130.0, "receipt_change": 3.0, "exchange": "DCE"}
    assert seen == ["20250401"]


def test_get_warehouse_receipt_supports_czce_symbols(monkeypatch):
    monkeypatch.setattr(data_cache, "_shfe_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_gfex_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_dce_receipt_cache", {})
    monkeypatch.setattr(data_cache, "_czce_receipt_cache", {})
    seen: list[str] = []

    def fake_czce(date="20200702"):
        seen.append(date)
        return {
            "PTA": pd.DataFrame(
                {
                    "仓单数量": [80.0, 20.0],
                    "当日增减": [-6.0, 1.0],
                }
            )
        }

    monkeypatch.setattr(data_cache.aks, "futures_czce_warehouse_receipt", fake_czce)

    out = data_cache.get_warehouse_receipt("TA0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out == {"receipt_total": 100.0, "receipt_change": -5.0, "exchange": "CZCE"}
    assert seen == ["20250401"]


def test_get_warehouse_receipt_prefers_czce_total_row_with_split_quantity_columns(monkeypatch):
    monkeypatch.setattr(data_cache, "_czce_receipt_cache", {})

    monkeypatch.setattr(
        data_cache.aks,
        "futures_czce_warehouse_receipt",
        lambda date="20200702": {
            "PTA": pd.DataFrame(
                {
                    "仓库编号": ["小计", "总计"],
                    "仓单数量(完税)": [9930.0, 173273.0],
                    "仓单数量(保税)": [0.0, 11.0],
                    "当日增减": [0.0, -1920.0],
                }
            )
        },
    )

    out = data_cache.get_warehouse_receipt("TA0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out == {"receipt_total": 173284.0, "receipt_change": -1920.0, "exchange": "CZCE"}


def test_get_warehouse_receipt_returns_none_for_unexpected_exchange_shape(monkeypatch):
    monkeypatch.setattr(data_cache, "_dce_receipt_cache", {})
    monkeypatch.setattr(
        data_cache.aks,
        "futures_dce_warehouse_receipt",
        lambda date="20200702": {"豆粕": pd.DataFrame({"仓库": ["A"]})},
    )

    out = data_cache.get_warehouse_receipt("M0", as_of_date=pd.Timestamp("2025-04-01").date())

    assert out is None


def test_get_hog_fundamentals_respects_as_of_date(monkeypatch):
    monkeypatch.setattr(data_cache, "_hog_cache", None)
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_core",
        lambda symbol="外三元": pd.DataFrame(
            {
                "date": ["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04"],
                "value": [16.0, 14.0, 10.0, 12.0],
            }
        ),
    )
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_cost",
        lambda symbol="玉米": pd.DataFrame(
            {
                "date": ["2025-04-01", "2025-04-03", "2025-04-04"],
                "value": [2600.0, 2500.0, 2400.0],
            }
        ),
    )
    monkeypatch.setattr(
        data_cache.aks,
        "futures_hog_supply",
        lambda symbol="猪肉批发价": pd.DataFrame(
            {
                "date": ["2025-04-01", "2025-04-03", "2025-04-04"],
                "value": [90.0, 95.0, 100.0],
            }
        ),
    )

    out = data_cache.get_hog_fundamentals(as_of_date=pd.Timestamp("2025-04-03").date())

    assert out is not None
    assert out["price"] == 10.0
    assert out["price_5d_ago"] == 16.0
    assert round(out["price_trend"], 2) == -37.5
    assert round(out["spot_price_percentile"], 2) == 0.0
    assert out["cost"] == 2500.0
    assert out["supply"] == 95.0
