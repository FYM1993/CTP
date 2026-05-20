from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

SYSTEMATIC_FUTURES_UNIVERSE: list[dict[str, str]] = [
    {"symbol": "V0", "exchange": "dce", "name": "PVC"},
    {"symbol": "P0", "exchange": "dce", "name": "棕榈油"},
    {"symbol": "B0", "exchange": "dce", "name": "豆二"},
    {"symbol": "M0", "exchange": "dce", "name": "豆粕"},
    {"symbol": "I0", "exchange": "dce", "name": "铁矿石"},
    {"symbol": "JD0", "exchange": "dce", "name": "鸡蛋"},
    {"symbol": "L0", "exchange": "dce", "name": "塑料"},
    {"symbol": "PP0", "exchange": "dce", "name": "聚丙烯"},
    {"symbol": "Y0", "exchange": "dce", "name": "豆油"},
    {"symbol": "C0", "exchange": "dce", "name": "玉米"},
    {"symbol": "A0", "exchange": "dce", "name": "豆一"},
    {"symbol": "J0", "exchange": "dce", "name": "焦炭"},
    {"symbol": "JM0", "exchange": "dce", "name": "焦煤"},
    {"symbol": "CS0", "exchange": "dce", "name": "淀粉"},
    {"symbol": "EG0", "exchange": "dce", "name": "乙二醇"},
    {"symbol": "EB0", "exchange": "dce", "name": "苯乙烯"},
    {"symbol": "PG0", "exchange": "dce", "name": "液化石油气"},
    {"symbol": "LH0", "exchange": "dce", "name": "生猪"},
    {"symbol": "FB0", "exchange": "dce", "name": "纤维板"},
    {"symbol": "TA0", "exchange": "czce", "name": "PTA"},
    {"symbol": "OI0", "exchange": "czce", "name": "菜油"},
    {"symbol": "RM0", "exchange": "czce", "name": "菜粕"},
    {"symbol": "SR0", "exchange": "czce", "name": "白糖"},
    {"symbol": "CF0", "exchange": "czce", "name": "棉花"},
    {"symbol": "MA0", "exchange": "czce", "name": "甲醇"},
    {"symbol": "FG0", "exchange": "czce", "name": "玻璃"},
    {"symbol": "SF0", "exchange": "czce", "name": "硅铁"},
    {"symbol": "SM0", "exchange": "czce", "name": "锰硅"},
    {"symbol": "AP0", "exchange": "czce", "name": "苹果"},
    {"symbol": "CJ0", "exchange": "czce", "name": "红枣"},
    {"symbol": "UR0", "exchange": "czce", "name": "尿素"},
    {"symbol": "SA0", "exchange": "czce", "name": "纯碱"},
    {"symbol": "PK0", "exchange": "czce", "name": "花生"},
    {"symbol": "CY0", "exchange": "czce", "name": "棉纱"},
    {"symbol": "SH0", "exchange": "czce", "name": "烧碱"},
    {"symbol": "PX0", "exchange": "czce", "name": "对二甲苯"},
    {"symbol": "CU0", "exchange": "shfe", "name": "铜"},
    {"symbol": "AL0", "exchange": "shfe", "name": "铝"},
    {"symbol": "ZN0", "exchange": "shfe", "name": "沪锌"},
    {"symbol": "PB0", "exchange": "shfe", "name": "铅"},
    {"symbol": "NI0", "exchange": "shfe", "name": "镍"},
    {"symbol": "SN0", "exchange": "shfe", "name": "锡"},
    {"symbol": "AU0", "exchange": "shfe", "name": "黄金"},
    {"symbol": "AG0", "exchange": "shfe", "name": "白银"},
    {"symbol": "RB0", "exchange": "shfe", "name": "螺纹钢"},
    {"symbol": "HC0", "exchange": "shfe", "name": "热轧卷板"},
    {"symbol": "RU0", "exchange": "shfe", "name": "天然橡胶"},
    {"symbol": "FU0", "exchange": "shfe", "name": "燃料油"},
    {"symbol": "BU0", "exchange": "shfe", "name": "沥青"},
    {"symbol": "SP0", "exchange": "shfe", "name": "纸浆"},
    {"symbol": "SS0", "exchange": "shfe", "name": "不锈钢"},
    {"symbol": "AO0", "exchange": "shfe", "name": "氧化铝"},
    {"symbol": "SC0", "exchange": "ine", "name": "原油"},
    {"symbol": "NR0", "exchange": "ine", "name": "20号胶"},
    {"symbol": "LU0", "exchange": "ine", "name": "低硫燃料油"},
    {"symbol": "BC0", "exchange": "ine", "name": "国际铜"},
    {"symbol": "SI0", "exchange": "gfex", "name": "工业硅"},
    {"symbol": "LC0", "exchange": "gfex", "name": "碳酸锂"},
    {"symbol": "PS0", "exchange": "gfex", "name": "多晶硅"},
]

EXCHANGE_UPPER = {
    "dce": "DCE",
    "czce": "CZCE",
    "shfe": "SHFE",
    "ine": "INE",
    "gfex": "GFEX",
}


def symbol_to_tq_main(symbol: str, exchange: str) -> str:
    product = symbol.removesuffix("0")
    exchange_upper = EXCHANGE_UPPER[exchange.lower()]
    product_code = product.upper() if exchange_upper == "CZCE" else product.lower()
    return f"KQ.m@{exchange_upper}.{product_code}"


def _strip_contract_digits(name: str) -> str:
    return re.sub(r"\d+$", "", str(name)).strip()


def instrument_from_tq_info_row(row: pd.Series) -> dict[str, str]:
    exchange_id = str(row["exchange_id"]).upper()
    product_id = str(row["product_id"])
    product_code = product_id.upper()
    return {
        "symbol": f"{product_code}0",
        "exchange": exchange_id.lower(),
        "name": _strip_contract_digits(str(row.get("instrument_name") or product_code)),
        "tq_symbol": f"KQ.m@{exchange_id}.{product_id}",
        "underlying_symbol": str(row["instrument_id"]),
    }


def discover_main_continuous_instruments(api) -> list[dict[str, str]]:
    underlying_symbols = list(api.query_cont_quotes())
    if not underlying_symbols:
        return []
    info = api.query_symbol_info(underlying_symbols)
    instruments = [instrument_from_tq_info_row(row) for _, row in info.iterrows()]
    return sorted(instruments, key=lambda item: (item["exchange"], item["symbol"]))


def klines_to_daily_frame(
    klines: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    exchange: str,
    tq_symbol: str,
) -> pd.DataFrame:
    close_oi = klines.get("close_oi")
    if close_oi is None:
        close_oi = pd.Series(0.0, index=klines.index)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(klines["datetime"]).dt.tz_localize(None).dt.normalize(),
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "tq_symbol": tq_symbol,
            "open": pd.to_numeric(klines["open"], errors="coerce"),
            "high": pd.to_numeric(klines["high"], errors="coerce"),
            "low": pd.to_numeric(klines["low"], errors="coerce"),
            "close": pd.to_numeric(klines["close"], errors="coerce"),
            "volume": pd.to_numeric(klines["volume"], errors="coerce"),
            "open_interest": pd.to_numeric(close_oi, errors="coerce").fillna(0.0),
        }
    )
    out = out.dropna(subset=["date", "close"])
    out = out.loc[out["close"] > 0]
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def load_tq_credentials(config_path: Path) -> tuple[str, str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tq_config = config.get("tqsdk") or {}
    account = str(tq_config.get("account") or "").strip()
    password = str(tq_config.get("password") or "").strip()
    if not account or not password:
        raise ValueError(f"TqSdk account/password missing in {config_path}")
    return account, password


def filter_universe(symbols: Iterable[str] | None = None) -> list[dict[str, str]]:
    if symbols is None:
        return list(SYSTEMATIC_FUTURES_UNIVERSE)
    wanted = {symbol.upper() for symbol in symbols}
    return [item for item in SYSTEMATIC_FUTURES_UNIVERSE if item["symbol"].upper() in wanted]


def fetch_tq_daily_with_api(
    api,
    instrument: dict[str, str],
    *,
    days: int,
    wait_timeout: float,
) -> pd.DataFrame:
    tq_symbol = instrument.get("tq_symbol") or symbol_to_tq_main(instrument["symbol"], instrument["exchange"])
    klines = api.get_kline_serial(tq_symbol, 86400, data_length=max(days, 30))
    deadline = time.time() + max(wait_timeout, 1.0)
    while time.time() < deadline:
        frame = klines_to_daily_frame(
            klines,
            symbol=instrument["symbol"],
            name=instrument["name"],
            exchange=instrument["exchange"],
            tq_symbol=tq_symbol,
        )
        if not frame.empty:
            return frame.tail(days).reset_index(drop=True)
        api.wait_update(deadline=deadline)
    return pd.DataFrame()
