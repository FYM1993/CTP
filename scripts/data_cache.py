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

_request_count = 0
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.5

MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

_cache_cleaned = False


def _is_after_close() -> bool:
    now = datetime.now()
    return now.hour > MARKET_CLOSE_HOUR or (
        now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE
    )


def _throttle():
    global _request_count, _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()
    _request_count += 1


def get_request_count() -> int:
    return _request_count


def _cache_paths(symbol: str, data_type: str = "daily") -> tuple[Path, Path]:
    """返回 (final缓存路径, live缓存路径)"""
    today = datetime.now().strftime("%Y%m%d")
    return (
        CACHE_DIR / f"{data_type}_{symbol}_{today}_final.parquet",
        CACHE_DIR / f"{data_type}_{symbol}_{today}_live.parquet",
    )


def _choose_cache_suffix(df: pd.DataFrame) -> str:
    """
    根据数据内容决定缓存类型。

    只有当数据包含今天的日期时才标记为 final（永久缓存）。
    否则标记为 live（临时缓存），下次运行时会被清理并重新拉取。
    周末不是交易日，任何数据都视为 final。
    """
    today_str = datetime.now().strftime("%Y%m%d")
    is_weekend = datetime.now().weekday() >= 5

    if is_weekend:
        return "final"

    if df is not None and len(df) > 0:
        latest = pd.Timestamp(df["date"].iloc[-1]).strftime("%Y%m%d")
        if latest == today_str:
            return "final"

    return "live"


def _ensure_cache_dir():
    """
    初始化缓存目录。每次进程启动只执行一次清理：
    1. 删除非今日的旧缓存
    2. 收盘后(工作日)删除 _live 临时缓存，强制下次拉取最新数据
    """
    global _cache_cleaned
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _cache_cleaned:
        return
    _cache_cleaned = True

    today = datetime.now().strftime("%Y%m%d")
    for f in CACHE_DIR.glob("*.parquet"):
        if today not in f.name:
            f.unlink()
    if _is_after_close() and datetime.now().weekday() < 5:
        for f in CACHE_DIR.glob(f"*_{today}_live.parquet"):
            f.unlink()


# ============================================================
#  日线数据
# ============================================================

def get_daily(symbol: str, days: int = 400) -> pd.DataFrame | None:
    """
    获取日线数据（优先读缓存）

    返回标准化的 DataFrame: date, open, high, low, close, volume, oi

    缓存策略:
      final: 数据已包含当日收盘价 → 当日不再重新拉取
      live:  数据尚未包含当日 → 下次运行时(收盘后)会被清理并重试
    """
    _ensure_cache_dir()
    final_path, live_path = _cache_paths(symbol, "daily")

    if final_path.exists():
        try:
            df = pd.read_parquet(final_path)
            if len(df) > 0:
                return df.tail(days)
        except Exception:
            final_path.unlink(missing_ok=True)

    if live_path.exists():
        try:
            df = pd.read_parquet(live_path)
            if len(df) > 0:
                return df.tail(days)
        except Exception:
            live_path.unlink(missing_ok=True)

    df = _fetch_daily_from_api(symbol, days)
    if df is not None and len(df) > 0:
        suffix = _choose_cache_suffix(df)
        save_path = final_path if suffix == "final" else live_path
        try:
            df.to_parquet(save_path, index=False)
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
    """
    获取库存数据（东方财富API）。

    返回:
      inv_now: 当前库存
      inv_change_4wk: 4周库存变化 (%)
      inv_cumulating_weeks: 最近8周累库周数
      inv_percentile: 库存在52周内的分位数 (0~100, 越高库存越高)
      inv_trend: 库存趋势方向 ("累库"/"去库"/"持平")
    """
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
        inv_change = (inv_now / (inv_4wk + 1e-12) - 1) * 100

        n52 = min(len(df), 52)
        inv_52w = df["库存"].tail(n52)
        low52 = float(inv_52w.min())
        high52 = float(inv_52w.max())
        inv_pct = (inv_now - low52) / (high52 - low52 + 1e-12) * 100

        cum_weeks = int((df["增减"].tail(8) > 0).sum())
        if cum_weeks >= 5:
            trend = "累库"
        elif cum_weeks <= 2:
            trend = "去库"
        else:
            trend = "持平"

        return {
            "inv_now": inv_now,
            "inv_change_4wk": inv_change,
            "inv_cumulating_weeks": cum_weeks,
            "inv_percentile": inv_pct,
            "inv_trend": trend,
        }
    except Exception:
        return None


# ============================================================
#  仓单数据（上期所）
# ============================================================

SHFE_RECEIPT_MAP = {
    "CU0": "铜", "AL0": "铝", "ZN0": "锌", "PB0": "铅", "NI0": "镍",
    "SN0": "锡", "AU0": "黄金", "AG0": "白银",
    "RB0": "螺纹钢", "HC0": "热轧卷板",
    "RU0": "天然橡胶", "FU0": "燃料油", "BU0": "沥青", "SP0": "纸浆",
    "SS0": "不锈钢", "AO0": "氧化铝",
}

_shfe_receipt_cache: dict | None = None


def get_warehouse_receipt(symbol: str) -> dict | None:
    """
    获取上期所仓单数据。

    返回:
      receipt_total: 仓单总量
      receipt_change: 仓单增减
    """
    global _shfe_receipt_cache
    name = SHFE_RECEIPT_MAP.get(symbol)
    if not name:
        return None

    if _shfe_receipt_cache is None:
        try:
            _shfe_receipt_cache = aks.futures_shfe_warehouse_receipt()
        except Exception:
            _shfe_receipt_cache = {}
            return None

    df = _shfe_receipt_cache.get(name)
    if df is None or not hasattr(df, "empty") or df.empty:
        return None

    try:
        total = pd.to_numeric(df["WRTWGHTS"], errors="coerce").sum()
        change = pd.to_numeric(df["WRTCHANGE"], errors="coerce").sum()
        return {
            "receipt_total": float(total),
            "receipt_change": float(change),
        }
    except Exception:
        return None


# ============================================================
#  生猪专项数据
# ============================================================

_hog_cache: dict | None = None


def get_hog_fundamentals() -> dict | None:
    """
    获取生猪专项基本面数据。

    返回:
      price: 最新猪价 (元/kg)
      price_5d_ago: 5天前猪价
      price_trend: 价格趋势 (%)
      cost: 养殖成本 (元/头)
      profit_margin: 利润率估算 (正值=盈利)
      supply: 供应指数(最新可得)
    """
    global _hog_cache
    if _hog_cache is not None:
        return _hog_cache

    result = {}
    try:
        df = aks.futures_hog_core()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        result["price"] = float(df["value"].iloc[-1])
        n5 = min(len(df), 6)
        result["price_5d_ago"] = float(df["value"].iloc[-n5])
        result["price_trend"] = (result["price"] / result["price_5d_ago"] - 1) * 100
    except Exception:
        pass

    try:
        df = aks.futures_hog_cost()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        result["cost"] = float(df["value"].iloc[-1])
        if "price" in result and result["cost"] > 0:
            est_revenue = result["price"] * 120
            result["profit_margin"] = (est_revenue / result["cost"] - 1) * 100
    except Exception:
        pass

    try:
        df = aks.futures_hog_supply()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        result["supply"] = float(df["value"].iloc[-1])
    except Exception:
        pass

    if result:
        _hog_cache = result
        return result
    return None


# ============================================================
#  季节性分析（从历史日线数据计算）
# ============================================================

def get_seasonality(df: pd.DataFrame) -> dict | None:
    """
    从日线数据中计算季节性特征。

    对比当前月份在历史上的涨跌统计。

    返回:
      month: 当前月份
      hist_avg_return: 历史同月平均月度收益率 (%)
      hist_up_pct: 历史同月上涨概率 (%)
      current_month_return: 当月已实现收益率 (%)
      seasonal_signal: 与历史方向一致时的信号强度 (-1~+1)
    """
    if df is None or len(df) < 250:
        return None

    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["month"] = d["date"].dt.month
        d["year"] = d["date"].dt.year

        monthly = d.groupby(["year", "month"])["close"].agg(["first", "last"])
        monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100

        now = datetime.now()
        cur_month = now.month
        hist = monthly.loc[monthly.index.get_level_values("month") == cur_month, "ret"]

        if len(hist) < 2:
            return None

        avg_ret = float(hist.mean())
        up_pct = float((hist > 0).mean() * 100)

        cur_data = d[
            (d["date"].dt.month == cur_month)
            & (d["date"].dt.year == now.year)
        ]
        if len(cur_data) >= 2:
            cur_ret = (float(cur_data["close"].iloc[-1]) / float(cur_data["close"].iloc[0]) - 1) * 100
        else:
            cur_ret = 0.0

        if abs(avg_ret) < 0.5:
            signal = 0.0
        elif avg_ret > 0:
            signal = min(up_pct / 100, 1.0)
        else:
            signal = -min((100 - up_pct) / 100, 1.0)

        return {
            "month": cur_month,
            "hist_avg_return": avg_ret,
            "hist_up_pct": up_pct,
            "current_month_return": cur_ret,
            "seasonal_signal": signal,
        }
    except Exception:
        return None


# ============================================================
#  持仓结构分析（从日线 OI 数据计算）
# ============================================================

def get_oi_structure(df: pd.DataFrame) -> dict | None:
    """
    从日线 OI 数据分析持仓结构。

    返回:
      oi_now: 当前持仓量
      oi_percentile: OI 在300日内的分位数 (0~100)
      oi_20d_change: 20日 OI 变化率 (%)
      oi_vs_price: OI-价格协同判断 ("增仓上涨"/"增仓下跌"/"减仓上涨"/"减仓下跌")
    """
    if df is None or "oi" not in df.columns or len(df) < 60:
        return None

    try:
        oi = pd.to_numeric(df["oi"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        oi_now = float(oi.iloc[-1])
        n = min(len(oi), 300)
        oi_range = oi.tail(n)
        oi_lo = float(oi_range.min())
        oi_hi = float(oi_range.max())
        oi_pct = (oi_now - oi_lo) / (oi_hi - oi_lo + 1e-12) * 100

        oi_20d_ago = float(oi.iloc[-21]) if len(oi) > 21 else oi_now
        oi_change = (oi_now / (oi_20d_ago + 1e-12) - 1) * 100

        price_20d_ago = float(close.iloc[-21]) if len(close) > 21 else float(close.iloc[-1])
        price_change = float(close.iloc[-1]) / (price_20d_ago + 1e-12) - 1

        oi_up = oi_change > 2
        price_up = price_change > 0.01
        if oi_up and price_up:
            oi_vs_price = "增仓上涨"
        elif oi_up and not price_up:
            oi_vs_price = "增仓下跌"
        elif not oi_up and price_up:
            oi_vs_price = "减仓上涨"
        else:
            oi_vs_price = "减仓下跌"

        return {
            "oi_now": oi_now,
            "oi_percentile": oi_pct,
            "oi_20d_change": oi_change,
            "oi_vs_price": oi_vs_price,
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
    live_count = 0

    for i, info in enumerate(symbols):
        sym = info["symbol"]
        print(f"\r  [{i+1}/{len(symbols)}] {info['name']:8s}", end="", flush=True)

        final_path, live_path = _cache_paths(sym, "daily")

        cached_df = None
        for p in (final_path, live_path):
            if p.exists():
                try:
                    cached_df = pd.read_parquet(p)
                    if len(cached_df) > 0:
                        break
                    cached_df = None
                except Exception:
                    p.unlink(missing_ok=True)
                    cached_df = None

        if cached_df is not None:
            data[sym] = cached_df
            cache_hits += 1
            continue

        df = _fetch_daily_from_api(sym, days=400)
        api_calls += 1
        if df is not None and len(df) > 0:
            suffix = _choose_cache_suffix(df)
            save_path = final_path if suffix == "final" else live_path
            if suffix == "live":
                live_count += 1
            try:
                df.to_parquet(save_path, index=False)
            except Exception:
                pass
            data[sym] = df

    has_today = api_calls - live_count
    msg = f"\n  📊 加载完成: {len(data)}个品种 (缓存{cache_hits}, API{api_calls}"
    if live_count > 0:
        msg += f", {live_count}个尚无今日数据"
    msg += ")"
    print(msg)
    return data
