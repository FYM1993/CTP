from __future__ import annotations

import re

import pandas as pd

EXCHANGE_UPPER = {
    "dce": "DCE",
    "czce": "CZCE",
    "shfe": "SHFE",
    "ine": "INE",
    "gfex": "GFEX",
}


def symbol_to_tq_main(symbol: str, exchange: str) -> str:
    match = re.match(r"^([A-Za-z]+)0$", symbol)
    if not match:
        raise ValueError(f"不支持的主连代码: {symbol}")
    return f"KQ.m@{EXCHANGE_UPPER[exchange.lower()]}.{match.group(1).lower()}"


def klines_to_daily_frame(klines: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(klines["datetime"]).dt.normalize(),
            "open": pd.to_numeric(klines["open"], errors="coerce"),
            "high": pd.to_numeric(klines["high"], errors="coerce"),
            "low": pd.to_numeric(klines["low"], errors="coerce"),
            "close": pd.to_numeric(klines["close"], errors="coerce"),
            "volume": pd.to_numeric(klines["volume"], errors="coerce"),
            "oi": pd.to_numeric(klines.get("close_oi", 0.0), errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["close"]).reset_index(drop=True)
