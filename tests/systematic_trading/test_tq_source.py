from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.tq_source import (  # noqa: E402
    discover_main_continuous_instruments,
    instrument_from_tq_info_row,
    klines_to_daily_frame,
    symbol_to_tq_main,
)


def test_symbol_to_tq_main_uses_lowercase_for_dce() -> None:
    assert symbol_to_tq_main("LH0", "dce") == "KQ.m@DCE.lh"


def test_symbol_to_tq_main_uses_uppercase_for_czce() -> None:
    assert symbol_to_tq_main("TA0", "czce") == "KQ.m@CZCE.TA"


def test_klines_to_daily_frame_maps_close_oi_to_open_interest() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-05-19"]),
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1234.0],
            "close_oi": [888.0],
        }
    )

    out = klines_to_daily_frame(frame, symbol="LH0", name="生猪", exchange="dce", tq_symbol="KQ.m@DCE.lh")

    assert list(out.columns) == [
        "date",
        "symbol",
        "name",
        "exchange",
        "tq_symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
    ]
    assert float(out.iloc[0]["open_interest"]) == 888.0


def test_instrument_from_tq_info_row_builds_internal_symbol_and_continuous_symbol() -> None:
    row = pd.Series(
        {
            "instrument_id": "DCE.j2609",
            "instrument_name": "焦炭2609",
            "exchange_id": "DCE",
            "product_id": "j",
        }
    )

    instrument = instrument_from_tq_info_row(row)

    assert instrument == {
        "symbol": "J0",
        "exchange": "dce",
        "name": "焦炭",
        "tq_symbol": "KQ.m@DCE.j",
        "underlying_symbol": "DCE.j2609",
    }


def test_discover_main_continuous_instruments_uses_query_cont_quotes_and_symbol_info() -> None:
    class FakeApi:
        def query_cont_quotes(self):
            return ["DCE.j2609", "CZCE.TA609"]

        def query_symbol_info(self, symbols):
            assert symbols == ["DCE.j2609", "CZCE.TA609"]
            return pd.DataFrame(
                [
                    {
                        "instrument_id": "DCE.j2609",
                        "instrument_name": "焦炭2609",
                        "exchange_id": "DCE",
                        "product_id": "j",
                    },
                    {
                        "instrument_id": "CZCE.TA609",
                        "instrument_name": "PTA609",
                        "exchange_id": "CZCE",
                        "product_id": "TA",
                    },
                ]
            )

    instruments = discover_main_continuous_instruments(FakeApi())

    assert instruments == [
        {
            "symbol": "TA0",
            "exchange": "czce",
            "name": "PTA",
            "tq_symbol": "KQ.m@CZCE.TA",
            "underlying_symbol": "CZCE.TA609",
        },
        {
            "symbol": "J0",
            "exchange": "dce",
            "name": "焦炭",
            "tq_symbol": "KQ.m@DCE.j",
            "underlying_symbol": "DCE.j2609",
        },
    ]
