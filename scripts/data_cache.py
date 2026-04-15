"""
TqSdk 数据获取与基本面数据层
========================================

本模块负责：
1. 使用 TqSdk 获取高频行情、日线、分钟线数据。
2. 使用 AkShare 获取基本面数据（库存、仓单、生猪专项），因 TqSdk 不提供此类数据。
3. 品种列表管理与 TqSdk 符号映射。
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as aks
import numpy as np
import pandas as pd
from tqsdk import TqApi

# 配置日志
logger = logging.getLogger(__name__)

# ============================================================
#  1. 品种列表与映射
# ============================================================

# 映射表：项目缩写 -> TqSdk 交易所前缀与品种名
# 注意：TqSdk 主连格式为 KQ.m@EXCHANGE.symbol
SYMBOL_MAP = {
    # DCE
    "V0": "KQ.m@DCE.v", "P0": "KQ.m@DCE.p", "B0": "KQ.m@DCE.b", "M0": "KQ.m@DCE.m",
    "I0": "KQ.m@DCE.i", "JD0": "KQ.m@DCE.jd", "L0": "KQ.m@DCE.l", "PP0": "KQ.m@DCE.pp",
    "Y0": "KQ.m@DCE.y", "C0": "KQ.m@DCE.c", "A0": "KQ.m@DCE.a", "J0": "KQ.m@DCE.j",
    "JM0": "KQ.m@DCE.jm", "CS0": "KQ.m@DCE.cs", "EG0": "KQ.m@DCE.eg", "EB0": "KQ.m@DCE.eb",
    "PG0": "KQ.m@DCE.pg", "LH0": "KQ.m@DCE.lh", "FB0": "KQ.m@DCE.fb",
    # CZCE
    "TA0": "KQ.m@CZCE.TA", "OI0": "KQ.m@CZCE.OI", "RM0": "KQ.m@CZCE.RM", "SR0": "KQ.m@CZCE.SR",
    "CF0": "KQ.m@CZCE.CF", "MA0": "KQ.m@CZCE.MA", "FG0": "KQ.m@CZCE.FG", "SF0": "KQ.m@CZCE.SF",
    "SM0": "KQ.m@CZCE.SM", "AP0": "KQ.m@CZCE.AP", "CJ0": "KQ.m@CZCE.CJ", "UR0": "KQ.m@CZCE.UR",
    "SA0": "KQ.m@CZCE.SA", "PK0": "KQ.m@CZCE.PK", "CY0": "KQ.m@CZCE.CY", "SH0": "KQ.m@CZCE.SH",
    "PX0": "KQ.m@CZCE.PX",
    # SHFE
    "CU0": "KQ.m@SHFE.cu", "AL0": "KQ.m@SHFE.al", "ZN0": "KQ.m@SHFE.zn", "PB0": "KQ.m@SHFE.pb",
    "NI0": "KQ.m@SHFE.ni", "SN0": "KQ.m@SHFE.sn", "AU0": "KQ.m@SHFE.au", "AG0": "KQ.m@SHFE.ag",
    "RB0": "KQ.m@SHFE.rb", "HC0": "KQ.m@SHFE.hc", "RU0": "KQ.m@SHFE.ru", "FU0": "KQ.m@SHFE.fu",
    "BU0": "KQ.m@SHFE.bu", "SP0": "KQ.m@SHFE.sp", "SS0": "KQ.m@SHFE.ss", "AO0": "KQ.m@SHFE.ao",
    # INE
    "SC0": "KQ.m@INE.sc", "NR0": "KQ.m@INE.nr", "LU0": "KQ.m@INE.lu", "BC0": "KQ.m@INE.bc",
    # GFEX
    "SI0": "KQ.m@GFEX.si", "LC0": "KQ.m@GFEX.lc", "PS0": "KQ.m@GFEX.ps",
}

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
    {"symbol": "SC0", "exchange": "ine", "name": "原油"},
    {"symbol": "LC0", "exchange": "gfex", "name": "碳酸锂"},
    {"symbol": "SI0", "exchange": "gfex", "name": "工业硅"},
]

def get_all_symbols() -> List[Dict[str, str]]:
    """获取所有支持的品种列表"""
    return BUILTIN_SYMBOLS

def to_tq_symbol(symbol: str) -> str:
    """内部缩写转 TqSdk 符号"""
    return SYMBOL_MAP.get(symbol, symbol)

# ============================================================
#  2. TqSdk 行情数据
# ============================================================

def get_daily_tq(api: TqApi, symbol: str, data_length: int = 400) -> pd.DataFrame:
    """获取日线数据并标准化格式"""
    tq_sym = to_tq_symbol(symbol)
    klines = api.get_kline_serial(tq_sym, 86400, data_length=data_length)
    df = klines.copy()
    df["date"] = pd.to_datetime(df["datetime"] / 1e9, unit="s")
    df = df.rename(columns={"close_oi": "oi"})
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

def get_minute_tq(api: TqApi, symbol: str, period: int = 5, data_length: int = 200) -> pd.DataFrame:
    """获取分钟线数据"""
    tq_sym = to_tq_symbol(symbol)
    klines = api.get_kline_serial(tq_sym, period * 60, data_length=data_length)
    df = klines.copy()
    df["datetime"] = pd.to_datetime(df["datetime"] / 1e9, unit="s")
    return df.sort_values("datetime").reset_index(drop=True)

def get_quote_tq(api: TqApi, symbol: str):
    """获取实时行情"""
    return api.get_quote(to_tq_symbol(symbol))

def prefetch_all_tq(api: TqApi, symbols: List[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    """批量同步所有品种的日线数据 (修复异步等待逻辑)"""
    klines_map = {}
    data = {}
    
    logger.info(f"TqSdk 开始同步 {len(symbols)} 个品种...")
    # 1. 发起所有订阅
    for info in symbols:
        sym = info["symbol"]
        tq_sym = to_tq_symbol(sym)
        # 获取 kline 对象（此时 kline.empty 为 True）
        klines_map[sym] = api.get_kline_serial(tq_sym, 86400, data_length=400)
    
    # 2. 循环等待数据填充 (最长等待 30 秒)
    start_wait = time.time()
    while time.time() - start_wait < 30:
        api.wait_update()
        # 检查是否所有品种的最后一根 K 线都已经有数据（非 NaN）
        all_ready = True
        for k in klines_map.values():
            if k.empty or pd.isna(k.close.iloc[-1]):
                all_ready = False
                break
        if all_ready:
            break
            
    # 3. 转换为标准 DataFrame
    for sym, klines in klines_map.items():
        if not klines.empty:
            df = klines.copy()
            df["date"] = pd.to_datetime(df["datetime"] / 1e9, unit="s")
            df = df.rename(columns={"close_oi": "oi"})
            data[sym] = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            
    logger.info(f"TqSdk 同步完成: 成功 {len(data)} 个，失败 {len(symbols) - len(data)} 个")
    return data

# ============================================================
#  3. 基本面数据 (AkShare 来源)
# ============================================================

INVENTORY_NAME_MAP = {
    "CU0": "铜", "AL0": "铝", "ZN0": "锌", "PB0": "铅", "NI0": "镍",
    "SN0": "锡", "SS0": "不锈钢", "AU0": "黄金", "AG0": "白银",
    "RB0": "螺纹钢", "HC0": "热卷", "RU0": "天然橡胶", "FU0": "燃料油",
    "BU0": "沥青", "SP0": "纸浆", "I0": "铁矿石", "J0": "焦炭", "JM0": "焦煤",
    "M0": "豆粕", "Y0": "豆油", "P0": "棕榈油", "A0": "豆一", "B0": "豆二",
    "C0": "玉米", "CS0": "玉米淀粉", "JD0": "鸡蛋", "LH0": "生猪", "V0": "PVC",
    "L0": "塑料", "PP0": "聚丙烯", "EB0": "苯乙烯", "EG0": "乙二醇",
    "PG0": "LPG", "TA0": "PTA", "MA0": "甲醇", "CF0": "棉花", "SR0": "白糖",
    "OI0": "菜油", "RM0": "菜粕", "FG0": "玻璃", "SA0": "纯碱", "UR0": "尿素",
    "SF0": "硅铁", "SM0": "锰硅", "AP0": "苹果", "CJ0": "红枣", "PK0": "花生",
}

def get_inventory(symbol: str) -> Optional[Dict[str, Any]]:
    """获取库存数据（东方财富API）"""
    inv_name = INVENTORY_NAME_MAP.get(symbol)
    if not inv_name: return None
    try:
        df = aks.futures_inventory_em(symbol=inv_name)
        if df.empty or len(df) < 10: return None
        df["库存"] = pd.to_numeric(df["库存"], errors="coerce")
        df = df.dropna(subset=["库存"])
        inv_now = float(df["库存"].iloc[-1])
        inv_4wk = float(df["库存"].iloc[-5]) if len(df) >= 5 else inv_now
        inv_change = (inv_now / (inv_4wk + 1e-12) - 1) * 100
        n52 = min(len(df), 52)
        inv_52w = df["库存"].tail(n52)
        low52, high52 = float(inv_52w.min()), float(inv_52w.max())
        inv_pct = (inv_now - low52) / (high52 - low52 + 1e-12) * 100
        return {"inv_now": inv_now, "inv_change_4wk": inv_change, "inv_percentile": inv_pct}
    except Exception: return None

SHFE_RECEIPT_MAP = {
    "CU0": "铜", "AL0": "铝", "ZN0": "锌", "PB0": "铅", "NI0": "镍",
    "SN0": "锡", "AU0": "黄金", "AG0": "白银", "RB0": "螺纹钢",
    "HC0": "热轧卷板", "RU0": "天然橡胶", "FU0": "燃料油", "BU0": "沥青",
    "SP0": "纸浆", "SS0": "不锈钢", "AO0": "氧化铝",
}

def get_warehouse_receipt(symbol: str) -> Optional[Dict[str, Any]]:
    """获取仓单数据摘要 (上期所)"""
    shfe_name = SHFE_RECEIPT_MAP.get(symbol)
    if not shfe_name: return None
    try:
        df = aks.futures_sjsh_receipt(symbol=shfe_name)
        if df.empty or len(df) < 2: return None
        # 提取最新一天的仓单
        latest = pd.to_numeric(df["receipt"].iloc[-1], errors="coerce")
        prev = pd.to_numeric(df["receipt"].iloc[-2], errors="coerce")
        return {"receipt_now": float(latest), "receipt_change": float(latest - prev)}
    except Exception: return None

_hog_cache = None
def get_hog_fundamentals() -> Optional[Dict[str, Any]]:
    """获取生猪专项数据 (价格、成本、利润)"""
    global _hog_cache
    if _hog_cache: return _hog_cache
    try:
        result = {"data_status": "full"}
        # 1. 生猪现货指数
        price_df = aks.futures_hog_index()
        if not price_df.empty:
            result["price"] = float(price_df["value"].iloc[-1])
        
        # 2. 养殖成本
        cost_df = aks.futures_hog_cost()
        if not cost_df.empty:
            result["cost"] = float(cost_df["value"].iloc[-1])
            result["data_status"] = "full"
        else:
            result["data_status"] = "partial"

        # 3. 计算利润率
        if "price" in result and "cost" in result and result["cost"] > 0:
            # 预估收入 = 现货价格 * 120 (假设120kg出栏)
            result["profit_margin"] = (result["price"] * 120 / result["cost"] - 1) * 100
        
        _hog_cache = result
        return result
    except Exception: return None

def get_seasonality(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """计算季节性评分"""
    if df is None or len(df) < 250: return None
    try:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["month"] = d["date"].dt.month
        d["year"] = d["date"].dt.year
        monthly = d.groupby(["year", "month"])["close"].last().pct_change() * 100
        cur_month = datetime.now().month
        hist = monthly.xs(cur_month, level="month").dropna()
        if len(hist) < 2: return None
        avg_ret = float(hist.mean())
        up_pct = float((hist > 0).mean() * 100)
        return {"seasonal_signal": np.sign(avg_ret) * (up_pct/100), "hist_avg_return": avg_ret}
    except Exception: return None
