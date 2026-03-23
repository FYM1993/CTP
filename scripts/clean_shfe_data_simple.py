#!/usr/bin/env python3
"""
清洗上海期货交易所历史数据 - 简化版
使用 xlrd + CSV 输出
"""

from pathlib import Path
import xlrd
import csv
from datetime import datetime


def parse_date(date_value):
    """解析日期"""
    if isinstance(date_value, float):
        date_value = int(date_value)
    
    date_str = str(date_value).strip()
    
    if len(date_str) == 8:
        try:
            return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        except:
            return None
    return None


def parse_shfe_excel_simple(file_path: Path):
    """解析Excel文件 - 简化版"""
    print(f"  处理: {file_path.name}")
    
    try:
        wb = xlrd.open_workbook(file_path, formatting_info=False)
        sheet = wb.sheet_by_index(0)
        
        # 查找表头
        header_idx = None
        for i in range(min(10, sheet.nrows)):
            row = sheet.row_values(i)
            row_str = ' '.join([str(x) for x in row])
            if ('合约' in row_str or 'Contract' in row_str) and ('日期' in row_str or 'Date' in row_str):
                header_idx = i
                break
        
        if header_idx is None:
            print(f"    ⚠️ 未找到表头")
            return []
        
        headers = sheet.row_values(header_idx)
        
        # 找到各列的索引
        col_map = {}
        for idx, h in enumerate(headers):
            h_str = str(h).strip().lower()
            if '合约' in h_str or 'contract' in h_str:
                col_map['symbol'] = idx
            elif '日期' in h_str or 'date' in h_str:
                col_map['date'] = idx
            elif '开盘' in h_str or h_str == 'open':
                col_map['open'] = idx
            elif '最高' in h_str or h_str == 'high':
                col_map['high'] = idx
            elif '最低' in h_str or h_str == 'low':
                col_map['low'] = idx
            elif '收盘' in h_str or h_str == 'close':
                col_map['close'] = idx
            elif '成交量' in h_str or h_str == 'volume':
                col_map['volume'] = idx
            elif '成交金额' in h_str or '成交额' in h_str or h_str in ['amount', 'turnover']:
                col_map['turnover'] = idx
            elif '持仓' in h_str or h_str in ['oi', 'open interest']:
                col_map['open_interest'] = idx
        
        if 'symbol' not in col_map or 'date' not in col_map:
            print(f"    ⚠️ 缺少必要列")
            return []
        
        # 读取数据
        data_rows = []
        current_symbol = None
        
        for i in range(header_idx + 1, sheet.nrows):
            row = sheet.row_values(i)
            
            # 合约代码
            symbol_val = row[col_map['symbol']] if col_map['symbol'] < len(row) else ''
            if symbol_val and str(symbol_val).strip():
                current_symbol = str(symbol_val).strip().upper()
            
            if not current_symbol:
                continue
            
            # 日期
            date_val = row[col_map['date']] if col_map['date'] < len(row) else ''
            date_str = parse_date(date_val)
            if not date_str:
                continue
            
            # 构建行数据
            data_row = {
                'datetime': date_str,
                'symbol': current_symbol + '.SHFE',
                'open': float(row[col_map.get('open', -1)]) if col_map.get('open', -1) < len(row) and row[col_map.get('open', -1)] else 0,
                'high': float(row[col_map.get('high', -1)]) if col_map.get('high', -1) < len(row) and row[col_map.get('high', -1)] else 0,
                'low': float(row[col_map.get('low', -1)]) if col_map.get('low', -1) < len(row) and row[col_map.get('low', -1)] else 0,
                'close': float(row[col_map.get('close', -1)]) if col_map.get('close', -1) < len(row) and row[col_map.get('close', -1)] else 0,
                'volume': float(row[col_map.get('volume', -1)]) if col_map.get('volume', -1) < len(row) and row[col_map.get('volume', -1)] else 0,
                'turnover': float(row[col_map.get('turnover', -1)]) if col_map.get('turnover', -1) < len(row) and row[col_map.get('turnover', -1)] else 0,
                'open_interest': float(row[col_map.get('open_interest', -1)]) if col_map.get('open_interest', -1) < len(row) and row[col_map.get('open_interest', -1)] else 0,
            }
            
            data_rows.append(data_row)
        
        print(f"    ✅ 读取 {len(data_rows)} 条")
        return data_rows
        
    except Exception as e:
        print(f"    ❌ 失败: {e}")
        return []


def main():
    input_dir = Path("data/SH")
    output_dir = Path("data/cleaned/SHFE")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 清洗上海期货交易所历史数据")
    print("=" * 60)
    
    # 只处理 xls 文件（xlrd 不支持新版 xlsx）
    files = sorted(input_dir.glob("*.xls"))
    print(f"找到 {len(files)} 个 .xls 文件\n")
    
    all_data = []
    
    for f in files:
        rows = parse_shfe_excel_simple(f)
        all_data.extend(rows)
    
    if not all_data:
        print("\n❌ 没有读取到数据")
        return
    
    # 去重并排序
    print(f"\n总记录数: {len(all_data):,}")
    
    # 按 datetime+symbol 去重
    seen = set()
    unique_data = []
    for row in all_data:
        key = (row['datetime'], row['symbol'])
        if key not in seen:
            seen.add(key)
            unique_data.append(row)
    
    print(f"去重后: {len(unique_data):,}")
    
    # 排序
    unique_data.sort(key=lambda x: (x['datetime'], x['symbol']))
    
    # 保存 CSV
    output_file = output_dir / "shfe_all.csv"
    fieldnames = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'open_interest']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_data)
    
    print(f"\n✅ 已保存: {output_file}")
    print(f"   日期范围: {unique_data[0]['datetime']} ~ {unique_data[-1]['datetime']}")
    print(f"   合约数: {len(set(r['symbol'] for r in unique_data))}")


if __name__ == "__main__":
    main()
