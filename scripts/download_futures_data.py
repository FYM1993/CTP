#!/usr/bin/env python3
"""
在线下载期货主力合约数据（AKShare）
=====================================

覆盖6大交易所82个品种，数据来自新浪财经，完全免费。

使用方法:
    # 下载全部品种
    python scripts/download_futures_data.py

    # 下载指定品种
    python scripts/download_futures_data.py --symbols RB0 CU0 AU0

    # 指定起始日期
    python scripts/download_futures_data.py --start-date 20180101

    # 输出为parquet格式
    python scripts/download_futures_data.py --format parquet
"""

import time
import argparse
from pathlib import Path
from datetime import datetime

import akshare as aks
import pandas as pd


# 交易所中文名映射
EXCHANGE_NAME = {
    "shfe": "上期所",
    "dce": "大商所",
    "czce": "郑商所",
    "cffex": "中金所",
    "ine": "能源所",
    "gfex": "广期所",
}


def get_all_symbols() -> pd.DataFrame:
    """获取所有期货品种列表"""
    return aks.futures_display_main_sina()


def download_one(symbol: str, start_date: str) -> pd.DataFrame | None:
    """下载单个品种的主力合约数据"""
    try:
        df = aks.futures_main_sina(symbol=symbol, start_date=start_date)
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "日期": "datetime",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "持仓量": "open_interest",
            "动态结算价": "settle",
        })

        for col in ["open", "high", "low", "close", "volume", "open_interest", "settle"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["symbol"] = symbol
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"    ❌ {symbol} 下载失败: {e}")
        return None


def download_all(
    symbols: list[str] | None = None,
    start_date: str = "20140101",
    output_dir: str = "data/futures",
    output_format: str = "csv",
    sleep_sec: float = 0.5,
) -> None:
    """批量下载期货数据"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📥 下载期货主力合约数据（AKShare）")
    print("=" * 60)
    print(f"起始日期: {start_date}")
    print(f"输出目录: {output_path}")
    print(f"输出格式: {output_format}")

    # 获取品种列表
    all_info = get_all_symbols()

    if symbols:
        all_info = all_info[all_info["symbol"].isin(symbols)]

    print(f"品种数量: {len(all_info)}")
    print()

    success = 0
    failed = []
    all_dfs = []

    for idx, row in all_info.iterrows():
        sym = row["symbol"]
        exchange = row["exchange"]
        name = row["name"]
        ex_name = EXCHANGE_NAME.get(exchange, exchange)

        print(f"  [{idx+1}/{len(all_info)}] {sym} ({name}, {ex_name})", end=" ... ")

        df = download_one(sym, start_date)

        if df is not None and len(df) > 0:
            df["exchange"] = exchange
            df["name"] = name
            all_dfs.append(df)
            print(f"✅ {len(df)} 条 ({df['datetime'].min().date()} ~ {df['datetime'].max().date()})")
            success += 1
        else:
            print("⚠️ 无数据")
            failed.append(sym)

        time.sleep(sleep_sec)

    if not all_dfs:
        print("\n❌ 没有下载到任何数据")
        return

    # 合并
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    # 保存汇总文件
    print("\n" + "=" * 60)
    print("💾 保存数据")
    print("=" * 60)

    if output_format == "csv":
        out_file = output_path / "futures_main.csv"
        combined.to_csv(out_file, index=False)
    else:
        out_file = output_path / "futures_main.parquet"
        combined.to_parquet(out_file, index=False)

    print(f"✅ 汇总文件: {out_file} ({out_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   总记录: {len(combined):,}")
    print(f"   日期范围: {combined['datetime'].min().date()} ~ {combined['datetime'].max().date()}")
    print(f"   成功: {success}, 失败: {len(failed)}")

    if failed:
        print(f"   失败品种: {', '.join(failed)}")

    # 按品种分别保存
    variety_dir = output_path / "by_symbol"
    variety_dir.mkdir(exist_ok=True)

    for sym in combined["symbol"].unique():
        sym_df = combined[combined["symbol"] == sym]
        if output_format == "csv":
            sym_df.to_csv(variety_dir / f"{sym}.csv", index=False)
        else:
            sym_df.to_parquet(variety_dir / f"{sym}.parquet", index=False)

    print(f"✅ 按品种保存: {variety_dir}/")

    print("\n" + "=" * 60)
    print("✅ 下载完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="下载期货主力合约数据")
    parser.add_argument("--symbols", nargs="+", help="指定品种 (如 RB0 CU0)")
    parser.add_argument("--start-date", default="20140101", help="起始日期 (默认: 20140101)")
    parser.add_argument("--output-dir", default="data/futures", help="输出目录")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="输出格式")
    parser.add_argument("--sleep", type=float, default=0.5, help="请求间隔秒数")
    args = parser.parse_args()

    download_all(
        symbols=args.symbols,
        start_date=args.start_date,
        output_dir=args.output_dir,
        output_format=args.format,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()
