# 期货 CTA 策略系统

这是一个以“基本面发现机会 + 盘前双路径定方向/定入场 + 盘中实时监控”为核心的期货交易辅助系统。

系统只做策略辅助，不直接替代人工决策。当前真实流程分三段：

- Phase 1：从基本面里发现机会，输出机会优先级，不直接给最终做多/做空建议。
- Phase 2：先解析方向，再并行评估反转入场和趋势延续入场，输出可操作或观望结果。
- Phase 3：盘中监控已选目标和观望目标，只负责实时跟踪，不再承担日线定方向。

## 每日工作流

```
全市场扫描 → 盘前深度分析 → 盘中实时监控
      Phase 1              Phase 2          Phase 3
```

一条命令跑完整流程：

```bash
python scripts/daily_workflow.py
```

常用参数：

```bash
# 完整流程
python scripts/daily_workflow.py

# 只看分析，不进入盘中监控
python scripts/daily_workflow.py --no-monitor

# 只用 config.yaml 里的品种，跳过全市场筛选
python scripts/daily_workflow.py --skip-screen

# 恢复今日已有分析，直接进入盘中监控
python scripts/daily_workflow.py --resume

# 调整 Phase 1 基本面门槛（默认 10；若 config.yaml 已配置则以配置为准）
python scripts/daily_workflow.py --threshold 25

# 最多跟踪 N 个品种
python scripts/daily_workflow.py --max-picks 4

# 指定 K 线周期
python scripts/daily_workflow.py --period 5
python scripts/daily_workflow.py --period 15
```

定时任务示例：

```bash
crontab -e
# 每个交易日 8:30 自动启动
30 8 * * 1-5 cd /path/to/ctp && python scripts/daily_workflow.py >> logs/$(date +\%Y\%m\%d).log 2>&1
```

## 策略说明

### Phase 1：基本面发现机会

Phase 1 只做“是否值得进入 Phase 2 深挖”的判断，不直接输出最终做多/做空建议。

当前 Phase 1 的核心输出是：

- `反转机会分`
- `趋势机会分`
- `关注优先级分`
- `机会标签`
- `Phase1摘要`

当前主要看这些基本面和位置因子：

- 库存变化与库存分位
- 仓单变化
- 季节性
- 生猪专项利润/周期逻辑
- 价格相对长期均衡位置

当前代码会把 `反转机会分` 和 `趋势机会分` 一起用于候选筛选和排序，再汇总成 `关注优先级分`。标签层会区分 `反转候选`、`趋势候选`、`双标签候选`、`数据覆盖不足`，摘要用于给 Phase 2 提供盘前上下文。

### Phase 2：盘前双路径定方向与入场

Phase 2 的真实逻辑是先解析方向，再并行评估两条入场路径：

1. 先由技术面评分解析方向，得到 `long / short / watch`。
2. 再分别评估反转入场和趋势延续入场。
3. 两条路径都要同时满足方向分数门槛和盈亏比门槛，才会被判定为可操作。

方向层主要看这些指标：

- 均线
- MACD
- RSI
- 布林带
- 动量
- 价格位置
- Wyckoff 阶段
- 量价关系
- VSA
- OI 结构

方向判定规则目前是：

- `long_score >= 20` 且 `long_score - short_score >= 12` => `long`
- `short_score >= 20` 且 `short_score - long_score >= 12` => `short`
- 其他情况 => `watch`

反转入场走 `assess_reversal_status()` 这条链路：

- 多头侧关注 `SC / StopVol_Bull / Spring / SOS`
- 空头侧关注 `BC / StopVol_Bear / UT / SOW`

趋势延续入场走当前的趋势状态评估：

- 多头侧看回踩企稳后重新转强，或放量突破
- 空头侧看反弹受阻后重新转弱，或放量跌破
- 在代码里，这两类状态会被整理成 `Pullback` / `TrendBreak` 这类入场标签

两条路径都会计算自己的入场、止损、止盈和 `rr`，并以：

- 方向分数门槛
- 盈亏比门槛 `RR >= 1.0`

作为是否 `actionable` 的最终条件。当前输出里会保留：

- `direction`
- `score`
- `actionable`
- `entry_family`
- `entry_signal_type`
- `entry_signal_detail`
- `entry`
- `stop`
- `tp1`
- `tp2`
- `rr`

Phase 2 的报告会分成可操作和观望两类。观望目标如果暂时不满足入场条件，会继续留给 Phase 3 盘中盯盘。

### Phase 3：盘中监控

Phase 3 只负责盘中实时监控，不负责日线定方向。

当前真实逻辑是：

- 信号周期默认是 `5m`
- 运行时会高频轮询当前进行中的 `5m bar`
- 终端重点打印观望目标的新信号、持续信号、信号消失
- 更细的分钟级变化写入日志文件

可操作目标会在盘中持续刷新分钟级图表和入场信息；观望目标会持续跑反转检测。若配置了 `tqsdk.account / password` 且安装了 `tqsdk`，则优先使用 TqSdk 实时订阅，否则回退到本地分钟线/合成日线。

## 回测

当前回测只验证 Phase 2 / Phase 3，不回测 Phase 1。

回测逻辑是：

- 用日线生成 Phase 2 交易计划
- 用 `1m` 推进时间
- 用当前进行中的 `5m bar` 判断盘中信号
- 支持单段回测和滚动回测

单段回测：

```bash
# 内置 case
python scripts/run_backtest.py --case lh0_long

# 覆盖日期区间
python scripts/run_backtest.py --case ps0_short --start 2025-04-01 --end 2025-12-31
```

滚动回测：

```bash
# walk-forward：180 天训练窗，60 天验证窗，60 天滚动步长
python scripts/run_backtest.py --case lh0_long --rolling --train-days 180 --validation-days 60 --step-days 60

# 指定区间后再做 walk-forward
python scripts/run_backtest.py --case lh0_long --start 2025-01-01 --end 2025-06-30 --rolling --train-days 120 --validation-days 30 --step-days 30
```

回测会打印收益汇总和诊断字段。关键诊断字段包括：

- `diag_phase2_actionable_days`
- `diag_phase2_reject_no_signal_days`
- `diag_total_phase2_actionable_reversal_days`
- `diag_total_phase2_actionable_trend_days`
- `diag_total_trades_opened_reversal`
- `diag_total_trades_opened_trend`

滚动回测还会输出每个窗口的对应诊断前缀，方便定位是哪一段时间没有信号、哪一段时间真正开出了反转单或趋势单。

## 安装

```bash
pip install -r requirements.txt
```

## 配置说明

`config.yaml` 里主要包含这些段：

- `data`：数据源类型、历史起始日期、输出目录等基础数据配置
- `tqsdk`：TqSdk 实时订阅和回测配置
- `positions`：手动指定的品种和方向，供 `--skip-screen` 使用
- `pre_market`：Phase 2 分析参数，包括均线、ATR、RSI、MACD、布林带和斐波那契目标位
- `fundamental_screening`：Phase 1 机会发现门槛和品种分组
- `intraday`：Phase 3 日内参数，包括量能过滤、均值回归和风控参数

## 项目结构

```text
ctp/
├── README.md
├── config.yaml
├── requirements.txt
├── scripts/
│   ├── daily_workflow.py
│   ├── run_backtest.py
│   ├── data_cache.py
│   ├── wyckoff.py
│   ├── download_futures_data.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── cases.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── phase23.py
│   │   └── rolling.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── daily_workflow.py
│   ├── market/
│   │   ├── __init__.py
│   │   ├── fundamental_data.py
│   │   ├── fundamental_edb_map.py
│   │   └── tq.py
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── factors.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   └── scoring.py
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── direction.py
│   │   └── pre_market.py
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── intraday.py
│   │   ├── live.py
│   └── shared/
│       ├── __init__.py
│       └── ctp_log.py
├── data/
└── logs/
```

## 备注

- 根目录真正对外的入口是 `scripts/daily_workflow.py` 和 `scripts/run_backtest.py`。
- 如果你只想看盘前策略，优先看 `scripts/phase1/`、`scripts/phase2/`。
- 如果你只想看盘中逻辑，优先看 `scripts/phase3/`。
