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


def _parse_symbol_components(symbol: str, exchange: str) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z]+)0$", symbol)
    if not match:
        raise ValueError(f"不支持的主连代码: {symbol}")
    exchange_upper = EXCHANGE_UPPER.get(exchange.lower())
    if not exchange_upper:
        raise ValueError(f"未知交易所: {exchange}")
    return exchange_upper, match.group(1)


def _product_id_candidates(symbol: str, exchange: str) -> tuple[str, list[str]]:
    exchange_upper, product = _parse_symbol_components(symbol, exchange)
    preferred = product.upper() if exchange_upper == "CZCE" else product.lower()
    candidates = []
    for item in (product, preferred, product.lower(), product.upper()):
        if item not in candidates:
            candidates.append(item)
    return exchange_upper, candidates


def symbol_to_tq_main(symbol: str, exchange: str) -> str:
    exchange_upper, product = _parse_symbol_components(symbol, exchange)
    product_code = product.upper() if exchange_upper == "CZCE" else product.lower()
    return f"KQ.m@{exchange_upper}.{product_code}"


def tq_underlying_to_continuous(underlying_symbol: str) -> str:
    if underlying_symbol.startswith("KQ.m@"):
        return underlying_symbol
    parts = underlying_symbol.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"不支持的Tq合约代码: {underlying_symbol}")
    exchange_id, instrument_id = parts
    match = re.match(r"^([A-Za-z]+)", instrument_id)
    if not match:
        raise ValueError(f"不支持的Tq合约代码: {underlying_symbol}")
    return f"KQ.m@{exchange_id.upper()}.{match.group(1)}"


def _query_cont_underlying(api, symbol: str, exchange: str) -> str | None:
    query_cont_quotes = getattr(api, "query_cont_quotes", None)
    if not callable(query_cont_quotes):
        return None

    exchange_upper, candidates = _product_id_candidates(symbol, exchange)
    for product_id in candidates:
        try:
            symbols = query_cont_quotes(exchange_id=exchange_upper, product_id=product_id)
        except Exception:
            continue
        for item in symbols or []:
            if item:
                return str(item)
    return None


def _resolve_from_quote(api, fallback_symbol: str, wait_timeout: float) -> str | None:
    get_quote = getattr(api, "get_quote", None)
    if not callable(get_quote):
        return None

    try:
        quote = get_quote(fallback_symbol)
    except Exception:
        return None

    underlying_symbol = getattr(quote, "underlying_symbol", "") or ""
    if underlying_symbol:
        try:
            return tq_underlying_to_continuous(underlying_symbol)
        except ValueError:
            return None

    wait_update = getattr(api, "wait_update", None)
    if not callable(wait_update):
        return None

    deadline = time.time() + max(wait_timeout, 0.0)
    while time.time() < deadline:
        updated = wait_update(deadline=deadline)
        underlying_symbol = getattr(quote, "underlying_symbol", "") or ""
        if underlying_symbol:
            try:
                return tq_underlying_to_continuous(underlying_symbol)
            except ValueError:
                return None
        if not updated:
            break
    return None


def resolve_tq_continuous_symbol(
    symbol: str,
    exchange: str,
    api=None,
    wait_timeout: float = 2.0,
) -> str:
    fallback_symbol = symbol_to_tq_main(symbol, exchange)
    if api is None:
        return fallback_symbol

    underlying_symbol = _query_cont_underlying(api, symbol, exchange)
    if underlying_symbol:
        try:
            return tq_underlying_to_continuous(underlying_symbol)
        except ValueError:
            pass

    resolved_from_quote = _resolve_from_quote(api, fallback_symbol, wait_timeout)
    if resolved_from_quote:
        return resolved_from_quote
    return fallback_symbol


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
    owns_api = api is None
    try:
        if api is None:
            api = create_tq_api(account, password)
        tq_symbol = resolve_tq_continuous_symbol(
            symbol.strip(),
            exchange,
            api=api,
            wait_timeout=min(wait_timeout, 2.0),
        )
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
