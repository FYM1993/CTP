"""
数据缓存层
========================================

解决的问题:
  新浪API免费接口有频率限制（短时间>50次会被封IP 1-5分钟）。
  日线数据每天只更新一次，没必要每次分析都远程拉取。

架构:
  API调用 → 本地parquet缓存 → 分析脚本读缓存

  所有脚本通过本模块获取数据，不再直接调用 akshare。
  首次获取写入 data/cache/{symbol}_{date}.parquet，
  同一天再次请求直接读本地文件，零API调用。

缓存策略:
  日线: 每日只拉取一次，当日有效
  分钟: 不缓存（实时数据）
  库存: 每日缓存
  品种列表: 内置列表 + 每日尝试更新一次
"""

import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import akshare as aks

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

# 请求计数器，用于自动限速
_request_count = 0
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.5  # 两次请求之间最少间隔(秒)

# 日线缓存只在收盘后生效 —
# 15:30之前拉取的数据不含当日收盘价，不应作为当天缓存
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


def _is_after_close() -> bool:
    """是否已过当日收盘时间（15:30）"""
    now = datetime.now()
    return now.hour > MARKET_CLOSE_HOUR or (
        now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE
    )


def _throttle():
    """全局限速：确保两次API请求之间至少间隔 MIN_REQUEST_INTERVAL 秒"""
    global _request_count, _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()
    _request_count += 1


def get_request_count() -> int:
    return _request_count


def _cache_path(symbol: str, data_type: str = "daily") -> Path:
    """
    生成缓存文件路径。

    收盘后(15:30+): data/cache/daily_CU0_20260409_final.parquet (稳定缓存)
    盘中(15:30前):  data/cache/daily_CU0_20260409_live.parquet  (临时缓存)
    """
    today = datetime.now().strftime("%Y%m%d")
    suffix = "final" if _is_after_close() else "live"
    return CACHE_DIR / f"{data_type}_{symbol}_{today}_{suffix}.parquet"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    for f in CACHE_DIR.glob("*.parquet"):
        if today not in f.name:
            f.unlink()
    # 收盘后清理盘中临时缓存，强制用收盘数据
    if _is_after_close():
        for f in CACHE_DIR.glob(f"*_{today}_live.parquet"):
            f.unlink()


# ============================================================
#  日线数据
# ============================================================

def get_daily(symbol: str, days: int = 400) -> pd.DataFrame | None:
    """
    获取日线数据（优先读缓存）

    返回标准化的 DataFrame: date, open, high, low, close, volume, oi
    """
    _ensure_cache_dir()
    cache = _cache_path(symbol, "daily")

    # 读缓存
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if len(df) > 0:
                return df.tail(days)
        except Exception:
            cache.unlink(missing_ok=True)

    # 缓存不存在，从API获取
    df = _fetch_daily_from_api(symbol, days)
    if df is not None and len(df) > 0:
        try:
            df.to_parquet(cache, index=False)
        except Exception:
            pass
        return df.tail(days)
    return None


def _fetch_daily_from_api(symbol: str, days: int, retries: int = 3) -> pd.DataFrame | None:
    """从新浪API获取日线数据，带重试"""
    for attempt in range(retries):
        try:
            _throttle()
            start = pd.Timestamp.now() - pd.Timedelta(days=days * 2)
            df = aks.futures_main_sina(symbol=symbol, start_date=start.strftime("%Y%m%d"))
            df = df.rename(columns={
                "日期": "date", "开盘价": "open", "最高价": "high",
                "最低价": "low", "收盘价": "close",
                "成交量": "volume", "持仓量": "oi",
            })
            for c in ["open", "high", "low", "close", "volume", "oi"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            if len(df) >= 30:
                return df
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 + attempt * 3)
    return None


# ============================================================
#  分钟数据（不缓存，实时获取）
# ============================================================

def get_minute(symbol: str, period: str = "5") -> pd.DataFrame | None:
    """获取分钟K线，不缓存"""
    try:
        _throttle()
        df = aks.futures_zh_minute_sina(symbol=symbol, period=period)
        for c in ["open", "high", "low", "close", "volume", "hold"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


# ============================================================
#  库存数据（每日缓存）
# ============================================================

INVENTORY_NAME_MAP = {
    "CU0": "铜", "AL0": "铝", "ZN0": "锌", "PB0": "铅", "NI0": "镍",
    "SN0": "锡", "SS0": "不锈钢", "AU0": "黄金", "AG0": "白银",
    "RB0": "螺纹钢", "HC0": "热卷",
    "RU0": "天然橡胶", "FU0": "燃料油", "BU0": "沥青", "SP0": "纸浆",
    "I0": "铁矿石", "J0": "焦炭", "JM0": "焦煤",
    "M0": "豆粕", "Y0": "豆油", "P0": "棕榈油",
    "A0": "豆一", "B0": "豆二", "C0": "玉米", "CS0": "玉米淀粉",
    "JD0": "鸡蛋", "LH0": "生猪", "V0": "PVC", "L0": "塑料",
    "PP0": "聚丙烯", "EB0": "苯乙烯", "EG0": "乙二醇",
    "PG0": "LPG", "TA0": "PTA", "MA0": "甲醇",
    "CF0": "棉花", "SR0": "白糖", "OI0": "菜油",
    "RM0": "菜粕", "FG0": "玻璃", "SA0": "纯碱",
    "UR0": "尿素", "SF0": "硅铁", "SM0": "锰硅",
    "AP0": "苹果", "CJ0": "红枣", "PK0": "花生",
    "SI0": "工业硅", "LC0": "碳酸锂",
}


def get_inventory(symbol: str) -> dict | None:
    """获取库存数据（东方财富API，单独域名不受新浪限流影响）"""
    inv_name = INVENTORY_NAME_MAP.get(symbol)
    if not inv_name:
        return None
    try:
        df = aks.futures_inventory_em(symbol=inv_name)
        if df.empty or len(df) < 10:
            return None

        df["库存"] = pd.to_numeric(df["库存"], errors="coerce")
        df["增减"] = pd.to_numeric(df["增减"], errors="coerce")
        df = df.dropna(subset=["库存"])

        inv_now = float(df["库存"].iloc[-1])
        inv_4wk = float(df["库存"].iloc[-5]) if len(df) >= 5 else inv_now

        return {
            "inv_now": inv_now,
            "inv_change_4wk": (inv_now / (inv_4wk + 1e-12) - 1) * 100,
            "inv_cumulating_weeks": int((df["增减"].tail(8) > 0).sum()),
        }
    except Exception:
        return None


# ============================================================
#  品种列表
# ============================================================

EXCLUDE = {"IF0", "IH0", "IC0", "IM0", "TF0", "T0", "TS0", "TL0"}

BUILTIN_SYMBOLS = [
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


def get_all_symbols() -> list[dict]:
    """获取品种列表。优先用内置列表（避免浪费API调用）"""
    return [s for s in BUILTIN_SYMBOLS if s["symbol"] not in EXCLUDE]


def prefetch_all(symbols: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    """
    批量预加载所有品种日线数据。

    首次调用从API获取并缓存，后续调用直接读本地文件。
    返回: {symbol: DataFrame}
    """
    if symbols is None:
        symbols = get_all_symbols()

    _ensure_cache_dir()
    data = {}
    api_calls = 0
    cache_hits = 0

    for i, info in enumerate(symbols):
        sym = info["symbol"]
        print(f"\r  [{i+1}/{len(symbols)}] {info['name']:8s}", end="", flush=True)

        cache = _cache_path(sym, "daily")
        if cache.exists():
            try:
                df = pd.read_parquet(cache)
                if len(df) > 0:
                    data[sym] = df
                    cache_hits += 1
                    continue
            except Exception:
                cache.unlink(missing_ok=True)

        df = _fetch_daily_from_api(sym, days=400)
        if df is not None and len(df) > 0:
            try:
                df.to_parquet(cache, index=False)
            except Exception:
                pass
            data[sym] = df
            api_calls += 1
        else:
            api_calls += 1

    print(f"\n  📊 加载完成: {len(data)}个品种 (缓存命中{cache_hits}, API调用{api_calls})")
    return data
