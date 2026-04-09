# 期货 CTA 策略系统

基于基本面筛选 + 技术面择时的期货交易辅助系统。

## 功能

- **盘前分析** — 综合技术指标给出入场时机建议和目标价位预估
- **日内监控** — 实时拉取分钟K线，自动生成交易信号

## 项目结构

```
ctp/
├── config.yaml          # 策略配置（品种、参数）
├── requirements.txt     # 依赖
├── scripts/
│   ├── pre_market.py    # 盘前分析
│   ├── intraday.py      # 日内策略监控
│   └── download_futures_data.py  # 批量下载历史数据
├── data/                # 数据缓存（.gitignore）
└── results/             # 分析结果（.gitignore）
```

## 快速开始

```bash
pip install -r requirements.txt
```

### 盘前分析

每日开盘前运行，获取技术面综合评分和操作建议：

```bash
python scripts/pre_market.py
```

输出包括：
- 多周期均线排列（MA5/10/20/60/120）
- 支撑/阻力位
- RSI、MACD、布林带状态
- Fibonacci 目标价位
- 综合评分和入场建议

### 日内监控

盘中实时监控，自动检测交易信号：

```bash
# 持续监控（每60秒刷新）
python scripts/intraday.py

# 单次分析
python scripts/intraday.py --once

# 指定K线周期
python scripts/intraday.py --period 15

# 自定义刷新间隔
python scripts/intraday.py --interval 30
```

信号类型：
- 布林带 + RSI 超买超卖反转
- MACD 金叉/死叉
- KDJ 交叉确认
- 放量突破/跌破均线

## 当前跟踪品种

| 品种 | 代码 | 方向 | 逻辑 |
|------|------|------|------|
| 生猪 | LH0 | 做多 | 基本面超跌，等待技术确认入场 |
| 多晶硅 | PS0 | 做空 | 库存过多，需求无增长 |

在 `config.yaml` 的 `positions` 中增删品种即可。

## 配置说明

`config.yaml` 包含三部分参数：

- **positions** — 品种列表及方向、合约乘数等
- **pre_market** — 盘前分析的均线窗口、ATR/RSI/MACD/Bollinger 参数、Fibonacci 级别
- **intraday** — 日内策略的突破窗口、均值回归参数、量能过滤、风控阈值
