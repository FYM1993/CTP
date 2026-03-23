"""
连续合约构建模块
================

从单个期货合约数据构建连续合约（主力合约序列）。

支持方式：
- volume: 按成交量切换（最主流）
- oi: 按持仓量切换
"""

import re
from pathlib import Path

import polars as pl


def load_raw_contracts(data_dir: str, symbol: str) -> pl.DataFrame:
    """
    加载单个品种的所有合约数据

    数据格式要求（CSV）：
    - datetime, symbol, open, high, low, close, volume, turnover, open_interest
    - symbol 格式: RB2401, IF2312 等

    支持的数据源（按优先级）：
    1. cleaned/SHFE/shfe_all.csv - 清洗后的完整数据
    2. raw/{symbol}/*.csv - 按品种分目录的原始CSV
    3. daily/{symbol}*.parquet - vnpy AlphaLab parquet格式
    """
    data_path = Path(data_dir)

    # 优先尝试从清洗后的 SHFE 数据加载
    shfe_csv = data_path / "cleaned" / "SHFE" / "shfe_all.csv"
    if shfe_csv.exists():
        # 读取完整数据并筛选目标品种
        df = pl.read_csv(shfe_csv, try_parse_dates=True)
        # 品种代码：从 AG2401.SHFE 提取 AG
        variety_pattern = f"{symbol.upper()}"
        df_filtered = df.filter(
            pl.col("symbol").str.starts_with(variety_pattern + "2") | 
            pl.col("symbol").str.starts_with(variety_pattern + "1") |
            pl.col("symbol").str.starts_with(variety_pattern + "0")
        )
        if len(df_filtered) > 0:
            # 移除 .SHFE 后缀以匹配期望格式
            df_filtered = df_filtered.with_columns([
                pl.col("symbol").str.replace(r"\.SHFE$", "")
            ])
            return df_filtered.sort(["datetime", "symbol"])

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
        raw_df: 单品种原始合约数据，必须包含 datetime, symbol, open, high, low, close,
                volume, open_interest 列
        method: 切换方式 "volume" 或 "oi"
        switch_threshold: 切换阈值（新合约的量/持仓量达到旧合约的此比例时切换）
        min_active_days: 合约最少活跃天数（过滤掉极不活跃的合约）

    Returns:
        连续合约数据（单个 DataFrame，symbol 为品种代号）
    """
    sort_col = "volume" if method == "volume" else "open_interest"

    # ========== 防御性检查: 确保单品种数据 ==========
    variety_set = set(
        m.group(1)
        for s in raw_df["symbol"].unique().to_list()
        if (m := re.match(r'([A-Z]+)', s))
    )
    if len(variety_set) > 1:
        raise ValueError(
            f"输入数据包含多个品种 {variety_set}，请按品种分别调用"
        )

    # ========== 构建 (datetime, symbol) → volume 查找字典 ==========
    # 用于 O(1) 查询当前主力的量，避免反复 filter 大表
    lookup = {}
    for row in raw_df.select(["datetime", "symbol", sort_col]).iter_rows(named=True):
        lookup[(row["datetime"], row["symbol"])] = row[sort_col] or 0

    # ========== 每日 top 合约（向量化）==========
    daily_top = (
        raw_df
        .group_by("datetime", maintain_order=True)
        .agg(
            pl.col("symbol").sort_by(pl.col(sort_col), descending=True).first().alias("top_symbol"),
        )
        .sort("datetime")
    )

    # ========== 模拟主力合约切换 ==========
    dates = daily_top["datetime"].to_list()
    top_symbols = daily_top["top_symbol"].to_list()

    current_symbol = None
    dominant_symbols = []

    for i in range(len(dates)):
        dt = dates[i]
        top_sym = top_symbols[i]

        if current_symbol is None:
            current_symbol = top_sym
        elif top_sym != current_symbol:
            current_val = lookup.get((dt, current_symbol), 0)
            top_val = lookup.get((dt, top_sym), 0)

            if current_val == 0:
                # 当前合约已退市
                current_symbol = top_sym
            elif top_val > current_val * switch_threshold:
                current_symbol = top_sym

        dominant_symbols.append(current_symbol)

    # ========== join 提取连续合约数据 ==========
    dominant_df = pl.DataFrame({
        "datetime": dates,
        "contract": dominant_symbols,
    })

    continuous_df = (
        raw_df
        .join(
            dominant_df,
            left_on=["datetime", "symbol"],
            right_on=["datetime", "contract"],
            how="inner",
        )
        .select(["datetime", "open", "high", "low", "close", "volume", "turnover", "open_interest", "symbol"])
        .rename({"symbol": "contract"})
        .sort("datetime")
    )

    # 过滤掉数据量过少的片段
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