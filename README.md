# 期货 CTA 策略系统

这是一个以 `Phase 0 共享数据准备 + 双策略独立执行 + 上层统一聚合` 为核心的期货交易辅助系统。

系统当前实现分成两条独立策略线：

- 策略 A：`reversal_fundamental`，基本面反转
- 策略 B：`trend_following`，趋势跟随

系统只做策略辅助，不直接替代人工决策。

## 阅读前提

如果 `README`、旧对话结论、口头描述与仓库代码/测试/真实命令结果冲突，以当前仓库代码为准。

当前代码已经明确的架构边界是：

- Strategy A = 基本面反转
- Strategy B = 趋势跟随
- Phase 1 只有基本面因素，只服务 Strategy A
- Strategy A 链路：`Phase 0 -> Phase 1 -> Phase 2A -> Phase 3`
- Strategy B 链路：`Phase 0 -> 趋势 universe / 价格技术筛选 -> Phase 2B -> Phase 3`
- Strategy B 不依赖 Phase 1

## 对外入口

根目录当前真正对外的命令入口是：

- `python scripts/daily_workflow.py`
- `python scripts/run_backtest.py`

其中：

- [scripts/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/daily_workflow.py) 只是一个很薄的 wrapper，实际主逻辑在 [scripts/cli/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/cli/daily_workflow.py)
- 回测主入口在 [scripts/run_backtest.py](/Users/yimin.fu/PythonProjects/ctp/scripts/run_backtest.py)

## 每日工作流

```text
全市场扫描 -> 盘前深度分析 -> 盘中实时监控
    Phase 0         Phase 1/2          Phase 3
```

一条命令跑完整流程：

```bash
python scripts/daily_workflow.py
```

常用参数：

```bash
# 完整流程
python scripts/daily_workflow.py

# 只跑盘前，不进入盘中监控
python scripts/daily_workflow.py --no-monitor

# 跳过全市场筛选，只用 config.yaml 里的 positions
python scripts/daily_workflow.py --skip-screen

# 恢复今日已有分析，直接进入盘中监控
python scripts/daily_workflow.py --resume

# 调整 Phase 1 基本面 threshold 原始值
python scripts/daily_workflow.py --threshold 25

# 聚合后最多保留 N 个可执行目标
python scripts/daily_workflow.py --max-picks 4

# 指定 Phase 3 使用的分钟周期
python scripts/daily_workflow.py --period 5
python scripts/daily_workflow.py --period 15
```

定时任务示例：

```bash
crontab -e
30 8 * * 1-5 cd /path/to/ctp && python scripts/daily_workflow.py >> logs/$(date +\%Y\%m\%d).log 2>&1
```

## 当前策略实现

### Phase 0：共享数据准备

对应代码：[scripts/cli/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/cli/daily_workflow.py)

- `phase_0_prefetch()` 会先拉全市场日线并写入本地 parquet 缓存
- 当日后续重复运行优先命中缓存
- Phase 0 只负责准备数据，不做策略判断

### Strategy A：基本面反转

对应模块：

- Universe / screen: [scripts/phase1/pipeline.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase1/pipeline.py)、[scripts/strategy_reversal/screen.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_reversal/screen.py)
- Phase 2A: [scripts/phase2/pre_market.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase2/pre_market.py)
- Backtest wrapper: [scripts/strategy_reversal/backtest.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_reversal/backtest.py)

#### A-1. Phase 1 基本面筛选

Strategy A 的 live/backtest 主路径是：

1. `build_phase1_candidates()`
2. `select_top_candidates()`
3. `select_reversal_candidates()`

其中 `run_phase1_pipeline()` 只是前两步的便捷封装，不包含最终的 reversal line 过滤。

Phase 1 当前输出这几类字段：

- `reversal_score`
- `trend_score`
- `attention_score`
- `labels`
- `state_labels`
- `reason_summary`
- `entry_threshold`

Phase 1 当前使用的因子族：

- 库存：`get_inventory()`
- 仓单：`get_warehouse_receipt()`
- 季节性：`get_seasonality()`
- 生猪专项：`get_hog_fundamentals()`，只对 `LH0` 生效
- 历史代理：`_historical_proxy_scores()`，基于价格位置、均线结构、20/60 日动量、OI 结构

实现细节：

- live 模式下，只有当库存/仓单/季节性/生猪因子全部缺失时，才会退化到历史代理
- 在 `as_of_date` 回放模式下，历史代理会启用
- 如果库存/仓单/季节性/生猪因子都缺失，也会退化到历史代理
- `data_coverage` 当前是 4 槽 shrink 模型，不是“5 个因子族逐项计数”：
  - 基础价格上下文固定算 1 槽
  - 库存命中算 1 槽
  - 仓单命中算 1 槽
  - `季节性 / 生猪专项` 合并算 1 槽
  - 历史代理只有分数足够强时才补进覆盖度
- 最终分数会乘 `shrink = 0.6 + 0.4 * data_coverage`
- 当前反转打分对价格位置使用更宽的低位/高位软区间；`LH0` 还会额外参考生猪现货在历史区间中的位置，而不是只看期货主力价格

标签规则：

- `reversal_score >= 60` 且 `trend_score >= 60` -> `双标签候选`
- 仅 `reversal_score >= 60` -> `反转候选`
- 仅 `trend_score >= 60` -> `趋势候选`
- `data_coverage < 0.5` -> 额外加 `数据覆盖不足`

入池门槛：

- `entry_threshold = 55 + raw_threshold`
- `config.yaml` 里默认 `fundamental_screening.default_threshold = 10`
- 所以大多数品种默认 `entry_threshold = 65`
- 特定分类会覆盖这个门槛：
  - `high_sensitivity`：`LH0`，门槛 `58`
  - `agricultural`：门槛 `63`
  - `low_sensitivity`：门槛 `70`

Universe 规则：

- `select_top_candidates()` 只保留满足 `is_phase1_candidate_eligible()` 的品种
- `fundamental_screening.top_n` 未配置时默认取 `40`
- A 策略最终只从 `labels` 含 `反转候选` 或 `双标签候选` 的品种里选
- [scripts/strategy_reversal/screen.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_reversal/screen.py) 会按 `reversal_score` 排序

日常工作流里的 `phase_1_screen()` 日志当前打印的是“Phase 1 关注池（门槛通过）”，不是 A 策略最终 seed；最终 seed 还要再经过 `select_reversal_candidates()`。

一句话概括：

`Phase 1` 是 Strategy A 的 seeded universe，不直接下单，也不服务 Strategy B。

#### A-2. Phase 2A 反转确认

对应函数：

- `build_reversal_trade_plan_from_daily_df()`
- `assess_reversal_status()`

反转事件识别来自 [scripts/wyckoff.py](/Users/yimin.fu/PythonProjects/ctp/scripts/wyckoff.py)，当前关注：

- 多头目标信号：`SC`、`StopVol_Bull`、`Spring`、`SOS`
- 空头目标信号：`BC`、`StopVol_Bear`、`UT`、`SOW`

其中真正的确认型事件是：

- 多头：`Spring`、`SOS`
- 空头：`UT`、`SOW`

当前确认新鲜度定义：

- `REVERSAL_FRESH_SIGNAL_MAX_DAYS_AGO = 2`
- 也就是 `signal_days_ago <= 2` 才算新鲜反转确认

当前 Phase 2A 的可执行条件：

- 存在有效反转确认
- 确认类型属于 `Spring/SOS/UT/SOW`
- `signal_strength >= 0.65`
- 分数 gate 通过
- `rr >= 1.0`

其中分数 gate 不是一个固定阈值：

- 先尝试常规 `_score_gate()`
- 如果常规分数不过，但属于新鲜反转确认，则允许走 soft countertrend gate
- soft threshold 会随 `signal_type` 和 `signal_strength` 提高或放宽
  - `SOS/SOW` 比 `Spring/UT` 更强
  - `signal_strength >= 0.85` 比 `0.75` 更强

底层返回里还会带这些技术字段，供聚合、调试和回测复用：

- `actionable`：是否已进入可执行状态
- `entry / stop / tp1 / tp2 / rr`
- `entry_family = reversal`
- `entry_signal_type`
- `phase2_score_gate_passed`
- `phase2_rr_gate_passed`
- `reversal_signal_fresh`
- `reversal_status.current_stage`
- `reversal_status.next_expected`

这意味着 Phase 2A 不只是给一个 yes/no，它还会保留观察级别诊断，说明当前卡在：

- 还没有确认
- 确认新鲜，但分数 gate 没过
- 确认新鲜，但 rr 没过
- 已有旧确认，但过了执行窗口

#### A-3. Phase 3 共享日内执行

Strategy A 的盘中执行不再自己写一套日内开仓逻辑，而是复用共享引擎：

- [scripts/phase3/intraday.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase3/intraday.py)
- [scripts/backtest/phase23.py](/Users/yimin.fu/PythonProjects/ctp/scripts/backtest/phase23.py)

当前 Phase 3 的做法是：

- 先拿到 Phase 2A 的日线计划
- 对已经可执行的目标，固定按“日线计划 -> 执行状态 -> 分钟级执行辅助观察”输出
- 分钟级观察只负责补充短线节奏和执行风险，不再充当独立下单判断
- 对观察名单，会提示“确认出现 / 确认持续 / 确认回退”

共享日内信号当前主要基于：

- 布林带
- RSI
- MACD
- KDJ
- 量能过滤

### Strategy B：趋势跟随

对应模块：

- Universe / screen: [scripts/strategy_trend/screen.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_trend/screen.py)
- Phase 2B: [scripts/phase2/pre_market.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase2/pre_market.py)
- Backtest wrapper: [scripts/strategy_trend/backtest.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_trend/backtest.py)

#### B-1. 趋势 universe

Strategy B 不依赖 Phase 1。

当前 universe 直接来自：

- `build_trend_universe()`
- `select_trend_candidates()`

具体逻辑：

- 对每个品种直接跑 `score_signals()`
- 汇总出 `long_score` / `short_score`
- 用 `resolve_phase2_direction()` 判断技术方向
- `trend_score = max(long_score, short_score)`

入池条件：

- `trend_score >= 60`
- `resolved_direction != watch`

候选列表里的关键技术字段：

- `trend_score`
- `attention_score = trend_score`
- `trend_direction`
- `labels = ["趋势候选"]`
- `entry_pool_reason = 趋势技术分达标（...）`

Strategy B 的 universe 只基于价格/技术结构，不读 Phase 1 的 `labels`。

#### B-2. Phase 2B 趋势确认

对应函数：

- `build_trend_trade_plan_from_daily_df()`
- `_assess_trend_continuation()`

当前 P2B 会检查：

- `phase_ok`
  - 多头要求 Wyckoff phase 属于 `markup` 或 `accumulation`
  - 空头要求属于 `markdown` 或 `distribution`
- `slope_ok`
  - 核心均线斜率方向一致
- `trend_indicator_ok`
  - `均线排列`、`MACD`、`动量` 三类证据方向一致
- `score_ok`
  - 总分 gate 通过

当前确认类型只有两类：

- `Pullback`
- `TrendBreak`

趋势计划的可执行条件：

- 有趋势延续确认
- `score_ok`
- `rr >= 1.0`

输出字段包括：

- `entry_family = trend`
- `entry_signal_type = Pullback | TrendBreak`
- `entry / stop / tp1 / tp2 / rr`

### 聚合层

对应代码：[scripts/cli/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/cli/daily_workflow.py)

当前聚合不是把两条策略混成一套逻辑，而是：

1. 分别跑 A 的 `Phase 1 -> Phase 2A`
2. 分别跑 B 的 `trend universe -> Phase 2B`
3. 合并两条策略线的可执行目标和等待确认列表
4. 保留 `strategy_family`、`entry_family`、`entry_signal_type` 等归因字段

盘前深度分析阶段：

- A 策略先从 reversal candidates 里取前 `max(2*max_picks, max_picks)` 个做 P2A
- B 策略先从 trend candidates 里取前 `max(2*max_picks, max_picks)` 个做 P2B
- 每条策略线内部会用 RRF 融合 screen 分数和 Phase 2 分数
- 最终全局可执行目标再按 `rrf_score` 合并，并截断到 `max_picks`

聚合结果里保留的关键技术字段包括：

- `strategy_family`
- `direction`
- `score`
- `actionable`：是否已进入可执行状态
- `entry_family`
- `entry_signal_type`
- `entry_signal_detail`
- `entry / stop / tp1 / tp2 / rr`
- `phase1_labels`
- `phase1_reason_summary`

## Phase 3：盘中监控

对应代码：[scripts/cli/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/cli/daily_workflow.py)、[scripts/phase3/intraday.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase3/intraday.py)

盘中分成两类对象：

- 可执行目标：拉分钟线，按“日线计划 -> 执行状态 -> 分钟级执行辅助观察”输出
- 观察名单：拉日线，提示对应交易故事的“确认出现 / 持续 / 回退”

当前实现里，观察名单已经按交易故事分流：

- 顺势机会输出顺势确认和顺势确认条件
- 反转机会输出反转确认和反转确认条件
- 分钟级提示只保留为执行辅助观察，不再和日线级主判断抢角色

## 回测

### 1. 单 case 回测

```bash
python scripts/run_backtest.py --case lh0_reversal_long
python scripts/run_backtest.py --case ps0_reversal_short --start 2025-04-01 --end 2025-12-31
python scripts/run_backtest.py --case lh0_trend_short
```

兼容旧 case 名：

- `lh0_long` -> `lh0_reversal_long`
- `lh0_short` -> `lh0_trend_short`
- `ps0_short` -> `ps0_reversal_short`

支持 walk-forward：

```bash
python scripts/run_backtest.py --case lh0_reversal_long --rolling --train-days 180 --validation-days 60 --step-days 60
python scripts/run_backtest.py --case lh0_trend_short --start 2025-01-01 --end 2025-06-30 --rolling --train-days 120 --validation-days 30 --step-days 30
```

### 2. 回测诊断输出

当前单 case / rolling mode 支持直接打印 Phase 2 与交易明细调试信息：

```bash
python scripts/run_backtest.py --case lh0_reversal_long --debug-phase2
python scripts/run_backtest.py --case ps0_reversal_short --rolling --debug-phase2 --window-index 1
```

当前诊断会打印：

- `phase2_actionable_days`：进入可执行状态的交易日数
- `phase2_non_actionable_days`：仍在等待确认的交易日数
- `phase2_reject_*`
- `trades_opened*`
- rolling 模式下每个 window 的 Phase 2 / Phase 3 汇总

当前 `phase2_state` 取值来自回测 debug 汇总，常见包括：

- `actionable`：当天进入可执行状态
- `no_signal`
- `score_gate`
- `rr_gate`
- `score_gate+rr_gate`
- `no_plan`
- `duplicate_signal`
- `history_insufficient`

另外，reversal 诊断还会单列：

- `fresh_signal_but_score_blocked_days`
- `fresh_signal_but_rr_blocked_days`

### 3. 回测里的 Phase 3 和退出

回测当前逻辑是：

- 先用日线计划判断当天是否已经达到可执行条件
- 再把当日分钟线逐 bar 喂给 `generate_signals()`
- 分钟级结果只作为执行辅助观察，真正的开仓仍以前面已经成立的日线计划为前提

当前回测 exit reason 主要包括：

- `stop`
- `tp2`
- `phase2_invalidated`
- `end_of_data`

## 配置说明

当前主要配置段：

- `data`
- `tqsdk`
- `positions`
- `pre_market`
- `fundamental_screening`
- `intraday`

其中：

- `pre_market` 控制 Phase 2 评分和方向解析
- `fundamental_screening` 控制 Strategy A 的 Phase 1 门槛和品种分类
- `intraday` 控制共享 Phase 3 分钟级入场/风控参数

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
│   │   ├── cases.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── phase23.py
│   │   ├── rolling.py
│   │   └── universe.py
│   ├── cli/
│   │   └── daily_workflow.py
│   ├── market/
│   │   ├── fundamental_data.py
│   │   ├── fundamental_edb_map.py
│   │   └── tq.py
│   ├── phase1/
│   │   ├── factors.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   └── scoring.py
│   ├── phase2/
│   │   ├── direction.py
│   │   └── pre_market.py
│   ├── phase3/
│   │   ├── intraday.py
│   │   └── live.py
│   ├── strategy_reversal/
│   │   ├── backtest.py
│   │   ├── intraday.py
│   │   ├── pre_market.py
│   │   └── screen.py
│   ├── strategy_trend/
│   │   ├── backtest.py
│   │   ├── intraday.py
│   │   ├── pre_market.py
│   │   └── screen.py
│   └── shared/
│       ├── ctp_log.py
│       └── strategy.py
├── data/
├── logs/
└── tests/
```

## 快速定位代码

- 只看 Strategy A：优先看 [scripts/phase1/pipeline.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase1/pipeline.py)、[scripts/strategy_reversal/screen.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_reversal/screen.py)、[scripts/phase2/pre_market.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase2/pre_market.py)
- 只看 Strategy B：优先看 [scripts/strategy_trend/screen.py](/Users/yimin.fu/PythonProjects/ctp/scripts/strategy_trend/screen.py)、[scripts/phase2/pre_market.py](/Users/yimin.fu/PythonProjects/ctp/scripts/phase2/pre_market.py)
- 只看日常工作流编排：优先看 [scripts/cli/daily_workflow.py](/Users/yimin.fu/PythonProjects/ctp/scripts/cli/daily_workflow.py)
- 只看回测：优先看 [scripts/run_backtest.py](/Users/yimin.fu/PythonProjects/ctp/scripts/run_backtest.py)、[scripts/backtest/universe.py](/Users/yimin.fu/PythonProjects/ctp/scripts/backtest/universe.py)、[scripts/backtest/phase23.py](/Users/yimin.fu/PythonProjects/ctp/scripts/backtest/phase23.py)
