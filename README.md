# 期货量化研究系统

基于 vnpy Alpha 模块的期货量化研究和回测系统。

## 项目结构

```
futures_quant/
├── config.yaml                # 配置文件（所有参数）
├── requirements.txt           # 依赖
├── scripts/
│   └── research_workflow.py   # 主研究流程（一键运行）
├── src/
│   ├── continuous.py          # 连续合约构建
│   ├── alpha_futures.py       # 期货Alpha158因子集
│   └── model_lgb.py           # LightGBM模型封装
├── data/                      # 数据目录
│   ├── raw/                   # 原始合约数据（CSV）
│   │   └── {品种}/            # 每个品种一个子目录
│   │       └── *.csv          # 各合约CSV
│   ├── continuous/            # 连续合约（自动生成）
│   └── lab/                   # AlphaLab数据（自动生成）
└── results/                   # 回测结果（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将期货日线数据放入 `data/raw/{品种}/` 目录，CSV 格式：

```csv
datetime,symbol,open,high,low,close,volume,turnover,open_interest
2020-01-02,RB2001,3500,3520,3490,3510,100000,350000000,500000
```

或者如果你已有 vnpy 的 parquet 数据，放入 `data/daily/` 目录。

### 3. 修改配置

编辑 `config.yaml`：

- `symbols`: 选择要交易的品种
- `segments`: 设置训练/验证/测试时间段
- `model.lightgbm`: 调整模型参数
- `backtest`: 设置回测参数和手续费

### 4. 运行

```bash
cd futures_quant
python scripts/research_workflow.py
```

或使用自定义配置：

```bash
python scripts/research_workflow.py my_config.yaml
```

## 因子集说明

### Alpha158（158个标准量价因子）

与 vnpy / qlib 的 Alpha158 一致，包含：
- K线形态因子（9个）
- 价格位置因子（4个）
- 时序动量/均值/波动/极值因子（60个）
- 量价关联因子（60个）
- 成交量因子（25个）

### 期货特有因子（额外 ~33个）

在 Alpha158 基础上扩展：
- **持仓量因子**: 变化率、均值比、趋势
- **波动率因子**: ATR、收益率波动率、偏度
- **量能因子**: 成交额变化、量价同步性
- **收益分布因子**: 连涨占比、最大涨跌幅

### 标签定义

- `ret_1`: T+1 收益率
- `ret_3`: T+3 收益率（T+1 买入持有到 T+3，默认）
- `ret_5`: T+5 收益率

## 回测说明

回测使用 vnpy 的 `BacktestingEngine`，支持：
- 多空双向交易
- 按合约规格计算手续费
- 滑点模拟
- 逐日盯市盈亏计算
- Sharpe Ratio、最大回撤等统计指标

## 与你的 stock 项目对比

| | stock（A股） | futures_quant（期货） |
|---|---|---|
| 框架 | Qlib | vnpy Alpha |
| 因子 | Alpha158 | Alpha158 + 期货因子 |
| 模型 | DEnsemble / LGB | LightGBM |
| 数据 | qlib bin | polars Parquet |
| 回测 | Qlib 内置 | vnpy BacktestingEngine |
| 交易方向 | 仅做多 | 多空双向 |
| 执行层 | 需额外对接 | vnpy CTA 直接可用 |
