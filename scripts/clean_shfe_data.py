#!/usr/bin/env python3
"""
清洗上海期货交易所历史数据
=============================

将从上期所官网下载的 Excel 格式行情数据转换为 vnpy.alpha 支持的格式。

使用方法:
    python scripts/clean_shfe_data.py
"""

from pathlib import Path
import pandas as pd
import polars as pl


def parse_shfe_excel(file_path: Path) -> pd.DataFrame:
    """
    解析单个上期所 Excel 文件
    
    Args:
        file_path: Excel 文件路径（.xls 或 .xlsx）
        
    Returns:
        清洗后的 DataFrame
    """
    print(f"  处理文件: {file_path.name}")
    
    try:
        # 读取 Excel，不指定 header
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
        
        # 查找包含"合约"或"Contract"且后面跟着"日期"或"Date"的行作为表头
        header_row_idx = None
        for idx, row in df_raw.iterrows():
            if idx > 20:  # 只检查前20行
                break
            row_list = [str(x) for x in row if pd.notna(x)]
            row_str = ' '.join(row_list)
            # 必须同时包含合约和日期关键字
            has_contract = '合约' in row_str or 'contract' in row_str.lower()
            has_date = '日期' in row_str or 'date' in row_str.lower()
            if has_contract and has_date:
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            print(f"    ⚠️ 未找到表头行，跳过")
            return None
        
        # 使用找到的行作为header重新读取
        df = pd.read_excel(file_path, sheet_name=0, header=header_row_idx)
        
        # 过滤空行
        df = df.dropna(how='all')
        
        # 查找合约和日期列
        contract_col = None
        date_col = None
        
        for col in df.columns:
            col_str = str(col).strip()
            if '合约' in col_str or 'contract' in col_str.lower() or '品种' in col_str:
                contract_col = col
            elif '日期' in col_str or 'date' in col_str.lower():
                date_col = col
        
        if contract_col is None or date_col is None:
            print(f"    ⚠️ 未找到合约或日期列，跳过")
            return None
        
        # 标准化列名
        col_mapping = {}
        for col in df.columns:
            col_str = str(col).strip().lower()
            if col == contract_col:
                col_mapping[col] = 'symbol'
            elif col == date_col:
                col_mapping[col] = 'date'
            elif '开盘' in col_str or col_str == 'open':
                col_mapping[col] = 'open'
            elif '最高' in col_str or col_str == 'high':
                col_mapping[col] = 'high'
            elif '最低' in col_str or col_str == 'low':
                col_mapping[col] = 'low'
            elif '收盘' in col_str or col_str == 'close':
                col_mapping[col] = 'close'
            elif '成交量' in col_str or col_str == 'volume':
                col_mapping[col] = 'volume'
            elif '成交金额' in col_str or '成交额' in col_str or col_str in ['amount', 'turnover']:
                col_mapping[col] = 'turnover'
            elif '持仓' in col_str or col_str in ['oi', 'open interest']:
                col_mapping[col] = 'open_interest'
        
        df = df.rename(columns=col_mapping)
        
        # 只保留需要的列
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 
                        'volume', 'turnover', 'open_interest']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # 过滤空行
        df = df.dropna(subset=['symbol', 'date'], how='any')
        df = df[df['symbol'] != '']
        df = df[df['symbol'].notna()]
        
        # 向前填充合约代码（处理合并单元格的情况）
        df['symbol'] = df['symbol'].replace('', pd.NA).ffill()
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date'])
        
        # 转换数值列
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # 标准化合约代码: 转大写并添加交易所后缀
        df['symbol'] = df['symbol'].str.upper().str.strip() + '.SHFE'
        
        # 重命名 date 为 datetime
        df = df.rename(columns={'date': 'datetime'})
        
        # 排序
        df = df.sort_values(['datetime', 'symbol']).reset_index(drop=True)
        
        print(f"    ✅ 成功读取 {len(df)} 条记录")
        return df
        
    except Exception as e:
        print(f"    ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_all_shfe_data(
    input_dir: str = "data/SH",
    output_dir: str = "data/cleaned/SHFE",
    output_format: str = "csv"
) -> None:
    """
    批量清洗所有上期所数据文件
    
    Args:
        input_dir: 原始 Excel 文件目录
        output_dir: 输出目录
        output_format: 输出格式 "csv" 或 "parquet"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 清洗上海期货交易所历史数据")
    print("=" * 60)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"输出格式: {output_format}")
    print()
    
    # 查找所有 Excel 文件
    excel_files = sorted(list(input_path.glob("*.xls")) + list(input_path.glob("*.xlsx")))
    
    if not excel_files:
        print("❌ 未找到任何 Excel 文件")
        return
    
    print(f"找到 {len(excel_files)} 个文件")
    print()
    
    all_dataframes = []
    
    # 逐个处理文件
    for file_path in excel_files:
        df = parse_shfe_excel(file_path)
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("\n❌ 没有成功读取任何数据")
        return
    
    # 合并所有数据
    print("\n" + "=" * 60)
    print("📊 合并数据")
    print("=" * 60)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.sort_values(['datetime', 'symbol']).reset_index(drop=True)
    
    # 去重
    original_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['datetime', 'symbol'], keep='first')
    duplicate_count = original_len - len(combined_df)
    
    print(f"总记录数: {len(combined_df):,}")
    if duplicate_count > 0:
        print(f"去重删除: {duplicate_count:,} 条")
    
    # 统计信息
    date_range = (
        combined_df['datetime'].min(),
        combined_df['datetime'].max()
    )
    unique_symbols = combined_df['symbol'].nunique()
    
    print(f"日期范围: {date_range[0].date()} ~ {date_range[1].date()}")
    print(f"合约数量: {unique_symbols}")
    print()
    
    # 转换为 Polars 以便后续处理
    combined_pl = pl.from_pandas(combined_df)
    
    # 输出数据
    print("=" * 60)
    print("💾 保存数据")
    print("=" * 60)
    
    if output_format == "csv":
        output_file = output_path / "shfe_all.csv"
        combined_pl.write_csv(output_file)
        print(f"✅ CSV 文件已保存: {output_file}")
    
    elif output_format == "parquet":
        output_file = output_path / "shfe_all.parquet"
        combined_pl.write_parquet(output_file)
        print(f"✅ Parquet 文件已保存: {output_file}")
    
    # 额外：按品种分别保存
    print("\n按品种保存...")
    varieties_dir = output_path / "by_variety"
    varieties_dir.mkdir(exist_ok=True)
    
    # 提取品种代码
    combined_pl = combined_pl.with_columns([
        pl.col('symbol').str.extract(r'([A-Z]+)', 1).alias('variety')
    ])
    
    variety_list = combined_pl['variety'].unique().sort().to_list()
    
    for variety in variety_list:
        if variety is None or variety == '':
            continue
        
        variety_df = combined_pl.filter(pl.col('variety') == variety)
        variety_df = variety_df.drop('variety')
        
        if output_format == "csv":
            variety_file = varieties_dir / f"{variety}.csv"
            variety_df.write_csv(variety_file)
        else:
            variety_file = varieties_dir / f"{variety}.parquet"
            variety_df.write_parquet(variety_file)
        
        print(f"  ✅ {variety}: {len(variety_df):,} 条 -> {variety_file.name}")
    
    print()
    print("=" * 60)
    print("✅ 数据清洗完成！")
    print("=" * 60)
    print()
    print("后续步骤：")
    print("  1. 查看清洗后的数据:")
    print(f"     {output_file}")
    print()
    print("  2. 构建连续合约:")
    print("     python scripts/build_continuous.py")
    print()
    print("  3. 或直接运行研究流程:")
    print("     python scripts/research_workflow.py")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="清洗上海期货交易所历史数据")
    parser.add_argument(
        "--input-dir", 
        default="data/SH",
        help="原始 Excel 文件目录 (默认: data/SH)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/cleaned/SHFE",
        help="输出目录 (默认: data/cleaned/SHFE)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="输出格式 (默认: csv)"
    )
    
    args = parser.parse_args()
    
    clean_all_shfe_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
