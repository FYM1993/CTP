"""
连续合约构建模块
================

从单个期货合约数据构建连续合约（主力合约序列）。

支持方式：
- volume: 按成交量切换（最主流）
- oi: 按持仓量切换
"""

import os
from pathlib import Path
from datetime import datetime

import polars as pl


def load_raw_contracts(data_dir: str, symbol: str) -> pl.DataFrame:
    """
    加载单个品种的所有合约数据

    数据格式要求（CSV）：
    - datetime, symbol, open, high, low, close, volume, turnover, open_interest
    - symbol 格式: RB2401, IF2312 等

    也支持直接从 vnpy AlphaLab 的 parquet 目录加载。
    """
    data_path = Path(data_dir)

    # 尝试从 CSV 目录加载
    csv_dir = data_path / "raw" / symbol
    if csv_dir.exists():
        dfs = []
        for csv_file in sorted(csv_dir.glob("*.csv")):
            df = pl.read_csv(
                csv_file,
                try_parse_dates=True,
                dtypes={"symbol": pl.Utf8}
            )
            dfs.append(df)

        if dfs:
            return pl.concat(dfs).sort(["datetime", "symbol"])

    # 尝试从 parquet 加载（vnpy AlphaLab 格式）
    parquet_dir = data_path / "daily"
    if parquet_dir.exists():
        dfs = []
        for f in sorted(parquet_dir.glob(f"{symbol}*.parquet")):
            df = pl.read_parquet(f)
            # 从文件名提取合约代码
            vt_symbol = f.stem
            df = df.with_columns(pl.lit(vt_symbol).alias("symbol"))
            dfs.append(df)

        if dfs:
            return pl.concat(dfs).sort(["datetime", "symbol"])

    raise FileNotFoundError(f"未找到 {symbol} 的数据，检查 {data_dir}")


def build_continuous(
    raw_df: pl.DataFrame,
    method: str = "volume",
    switch_threshold: float = 0.8,
    min_active_days: int = 10,
) -> pl.DataFrame:
    """
    构建连续合约

    Args:
        raw_df: 原始合约数据，必须包含 datetime, symbol, open, high, low, close,
                volume, open_interest 列
        method: 切换方式 "volume" 或 "oi"
        switch_threshold: 切换阈值（新合约的成交量/持仓量超过旧合约的此比例时切换）
        min_active_days: 合约最少活跃天数（过滤掉极不活跃的合约）

    Returns:
        连续合约数据（单个 DataFrame，symbol 为品种代号）
    """
    sort_col = "volume" if method == "volume" else "open_interest"

    # 按日期排序
    df = raw_df.sort("datetime")

    # 获取所有交易日
    all_dates = df.select("datetime").unique().sort("datetime")

    records = []
    current_symbol = None

    for row in all_dates.iter_rows(named=True):
        dt = row["datetime"]

        # 当日各合约数据
        day_df = df.filter(pl.col("datetime") == dt)

        if day_df.is_empty():
            continue

        # 找当日主力（成交量/持仓量最大的合约）
        day_df = day_df.sort(sort_col, descending=True)
        top_contract = day_df.row(0, named=True)
        top_symbol = top_contract["symbol"]
        top_value = top_contract[sort_col]

        if current_symbol is None:
            # 首日，直接选定
            current_symbol = top_symbol
        else:
            # 检查是否需要切换
            current_day = day_df.filter(pl.col("symbol") == current_symbol)

            if current_day.is_empty():
                # 当前合约已退市，切换到新主力
                current_symbol = top_symbol
            else:
                current_value = current_day.row(0, named=True)[sort_col]

                if current_value and top_value:
                    if top_value > current_value * (1 + (1 - switch_threshold)):
                        # 新合约的量足够大，切换
                        current_symbol = top_symbol

        # 记录当日数据
        contract_data = day_df.filter(pl.col("symbol") == current_symbol)
        if not contract_data.is_empty():
            row_data = contract_data.row(0, named=True)
            records.append({
                "datetime": dt,
                "open": row_data["open"],
                "high": row_data["high"],
                "low": row_data["low"],
                "close": row_data["close"],
                "volume": row_data.get("volume", 0),
                "turnover": row_data.get("turnover", 0),
                "open_interest": row_data.get("open_interest", 0),
                "contract": current_symbol,  # 记录实际合约
            })

    continuous_df = pl.DataFrame(records).sort("datetime")

    # 过滤掉数据量过少的片段
    # （通常连续合约前后可能有数据缺失）
    if len(continuous_df) < min_active_days:
        print(f"  ⚠️ 数据量不足 {min_active_days} 天，跳过")
        return pl.DataFrame()

    return continuous_df


def prepare_all_continuous(
    data_dir: str,
    symbols: list[str],
    method: str = "volume",
    switch_threshold: float = 0.8,
) -> dict[str, pl.DataFrame]:
    """
    为所有品种构建连续合约

    Returns:
        {品种代号: 连续合约DataFrame}
    """
    results = {}

    for symbol in symbols:
        print(f"📊 构建 {symbol} 连续合约...")

        try:
            raw_df = load_raw_contracts(data_dir, symbol)
            continuous_df = build_continuous(
                raw_df,
                method=method,
                switch_threshold=switch_threshold,
            )

            if not continuous_df.is_empty():
                results[symbol] = continuous_df

                # 保存到文件
                out_path = Path(data_dir) / "continuous"
                out_path.mkdir(parents=True, exist_ok=True)
                continuous_df.write_parquet(out_path / f"{symbol}.parquet")

                n_days = len(continuous_df)
                start = continuous_df["datetime"][0]
                end = continuous_df["datetime"][-1]
                print(f"  ✅ {symbol}: {n_days} 天, {start} ~ {end}")
            else:
                print(f"  ❌ {symbol}: 数据为空")

        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    return results


def load_continuous(data_dir: str, symbol: str) -> pl.DataFrame:
    """加载已保存的连续合约数据"""
    path = Path(data_dir) / "continuous" / f"{symbol}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    raise FileNotFoundError(f"连续合约文件不存在: {path}")


if __name__ == "__main__":
    import yaml
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["data_dir"]
    symbols = config["symbols"]
    method = config.get("continuous_method", "volume")
    threshold = config.get("switch_threshold", 0.8)

    print("=" * 50)
    print("🔧 构建连续合约")
    print("=" * 50)

    results = prepare_all_continuous(
        data_dir=data_dir,
        symbols=symbols,
        method=method,
        switch_threshold=threshold,
    )

    print(f"\n✅ 完成，共处理 {len(results)} 个品种")
