from __future__ import annotations

import re
import time

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


def create_tq_api(account: str, password: str):
    from tqsdk import TqApi, TqAuth

    return TqApi(auth=TqAuth(account, password))


def _wait_for_ready_daily_frame(api, klines: pd.DataFrame, wait_timeout: float) -> pd.DataFrame | None:
    frame = klines_to_daily_frame(klines)
    if len(frame) > 0:
        return frame

    wait_update = getattr(api, "wait_update", None)
    if wait_update is None:
        return None

    deadline = time.time() + max(wait_timeout, 0.0)
    while time.time() < deadline:
        updated = wait_update(deadline=deadline)
        frame = klines_to_daily_frame(klines)
        if len(frame) > 0:
            return frame
        if not updated:
            break
    return None


def fetch_daily_from_tq(
    symbol: str,
    exchange: str,
    days: int,
    account: str,
    password: str,
    api=None,
    wait_timeout: float = 5.0,
) -> pd.DataFrame | None:
    try:
        tq_symbol = symbol_to_tq_main(symbol.strip(), exchange)
    except Exception:
        return None

    owns_api = api is None
    try:
        if api is None:
            api = create_tq_api(account, password)
        klines = api.get_kline_serial(tq_symbol, 86400, data_length=max(days, 30))
        if klines is None or len(klines) < 1:
            return None
        frame = _wait_for_ready_daily_frame(api, klines, wait_timeout)
        if frame is None or len(frame) < 1:
            return None
        return frame.tail(days).reset_index(drop=True)
    except Exception:
        return None
    finally:
        try:
            if owns_api and api is not None:
                api.close()
        except Exception:
            pass
