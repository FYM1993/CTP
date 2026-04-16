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
    exchange_upper = EXCHANGE_UPPER.get(exchange.lower())
    if not exchange_upper:
        raise ValueError(f"未知交易所: {exchange}")
    return f"KQ.m@{exchange_upper}.{match.group(1).lower()}"


def klines_to_daily_frame(klines: pd.DataFrame) -> pd.DataFrame:
    close_oi = klines.get("close_oi")
    if close_oi is None:
        close_oi = pd.Series(0.0, index=klines.index)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(klines["datetime"]).dt.normalize(),
            "open": pd.to_numeric(klines["open"], errors="coerce"),
            "high": pd.to_numeric(klines["high"], errors="coerce"),
            "low": pd.to_numeric(klines["low"], errors="coerce"),
            "close": pd.to_numeric(klines["close"], errors="coerce"),
            "volume": pd.to_numeric(klines["volume"], errors="coerce"),
            "oi": pd.to_numeric(close_oi, errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["close"]).reset_index(drop=True)
