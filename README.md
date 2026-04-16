# 期货 CTA 策略系统

基于「机会发现 + 方向/时机判断 + 实时监控」的期货交易辅助系统。

- Phase 1：发现机会，输出关注优先级和机会标签，不直接定方向
- Phase 2：定方向和入场时机，给出可操作/观望结果
- Phase 3：盘中实时监控，跟踪已选目标和观望信号

## 每日工作流

```
全市场扫描(76品种) → 盘前深度分析 → 盘中实时监控
      Phase 1              Phase 2          Phase 3
```

一条命令完成全流程：

```bash
python scripts/daily_workflow.py
```

### 完整流程说明

| 阶段 | 时间 | 做什么 |
|------|------|--------|
| Phase 1 | 盘前 | 扫描76个品种，按关注分、极端价位和基本面条件发现机会，输出机会标签 |
| Phase 2 | 盘前 | 对候选品种做方向与时机分析，输出入场价/止损/目标位，标记可操作或观望 |
| Phase 3 | 09:00-15:00 | 对可操作品种拉取分钟K线，并对观望品种做实时反转提示 |

### 常用参数

```bash
# 完整流程
python scripts/daily_workflow.py

# 只看分析，不进入盘中监控
python scripts/daily_workflow.py --no-monitor

# 用 config.yaml 已有品种，跳过全市场筛选
python scripts/daily_workflow.py --skip-screen

# 恢复今日已有分析，直接进入盘中监控
python scripts/daily_workflow.py --resume

# 调整 Phase 1 基本面门槛（默认10；若 config.yaml 已配置则以配置为准）
python scripts/daily_workflow.py --threshold 25

# 最多跟踪N个品种
python scripts/daily_workflow.py --max-picks 4

# 指定K线周期
python scripts/daily_workflow.py --period 15
```

### 定时任务 (crontab)

```bash
crontab -e
# 每个交易日 8:30 自动启动
30 8 * * 1-5 cd /path/to/ctp && python scripts/daily_workflow.py >> logs/$(date +\%Y\%m\%d).log 2>&1
```

## 高级用法

```bash
# 只做筛选和分析，不监控（快速查看机会）
python scripts/daily_workflow.py --no-monitor

# 只分析 config.yaml 指定品种（跳过全市场扫描）
python scripts/daily_workflow.py --skip-screen --no-monitor

# 恢复今日分析结果，直接进入监控
python scripts/daily_workflow.py --resume

# 自定义筛选数量和监控参数
python scripts/daily_workflow.py --max-picks 10 --period 15 --interval 30
```

## 当前策略

README 这里描述的是当前主入口已经落地的策略行为，不把“设计稿里的目标形态”和“已经接入的真实逻辑”混在一起。

### Phase 1：基本面发现机会

Phase 1 当前不直接给做多/做空结论，只回答“这个品种值不值得进 Phase 2 深挖”。

主入口 `scripts/daily_workflow.py` 目前的 Phase 1 流程是：

1. 先跑基本面引擎，默认以 `legacy` 为主，`LH0` 可切 `regime`。
2. 候选入池条件满足任一即可：
   - 基本面绝对分达到该品种门槛
   - 价格处于极端区间位（`<5%` 或 `>95%`）
3. 入池后统一生成这些 Phase 1 字段：
   - `关注分`
   - `反转机会分`
   - `趋势机会分`
   - `机会标签`
   - `Phase1摘要`

当前桥接规则如下：

| 入池原因 | 机会标签 | 分数字段 |
|------|------|------|
| 基本面达标 + 极端价格 | `双标签候选` | `反转机会分` 和 `趋势机会分` 都继承当前 P1 分值 |
| 仅极端价格 | `反转候选` | 只给 `反转机会分` |
| 仅基本面达标 | `趋势候选` | 只给 `趋势机会分` |

当前实现里，`关注分 = abs(Phase 1 原始分数)`，用于候选排序和报告展示。

Phase 1 当前主要使用这些信息：

- 价格区间位置
- 库存4周变化
- 库存分位
- 仓单变化
- 季节性
- 生猪专项利润/周期逻辑

### Phase 2：技术面定方向和时机

Phase 2 才真正决定 `long / short / watch`。

当前流程是：

1. 先用 `score_signals()` 分别计算技术面多头分和空头分
2. 再用 `scripts/phase2_direction.py` 解析方向
3. 最后把解析出的方向交给 `analyze_one()` 生成交易参数

当前方向规则：

- `long_score >= 20` 且 `long_score - short_score >= 12` => `long`
- `short_score >= 20` 且 `short_score - long_score >= 12` => `short`
- 否则 => `watch`

Phase 2 当前技术面来自三类信息：

- 经典指标：均线、MACD、RSI、布林带、动量、价格位置
- Wyckoff / VSA：阶段识别、关键事件、量价关系
- 持仓结构：价格与 OI 四象限、OI 背离

可操作品种必须同时满足：

- 有新鲜入场信号
- Phase 2 分数达标
- 盈亏比 `>= 1`

### 排名、报告与风险提示

当前报告会同时输出：

- 可操作列表
- 观望列表
- JSON 报告
- Markdown 报告

报告里的关键字段包括：

- `关注分`
- `P2分`
- `机会标签`
- `Phase1摘要`
- `RRF`
- `#P1 / #P2`
- 风险提示：`⚠️P1P2异号`、`⚠️逆势`

当前 RRF 仍按 `Phase 1 原始分` 与 `Phase 2 分数` 的排名融合，用于最终展示顺序。

### Phase 3：实时监控

Phase 3 负责两件事：

- 对可操作品种做分钟级监控
- 对观望品种做日线级反转信号跟踪

若已配置 `tqsdk.account / password`，Phase 3 优先使用 `TqSdk` 实时订阅；否则回退新浪分钟线。

### 数据源优先级

- 价格、日线、分钟线、持仓量：优先使用 `TqSdk`
- 非价量类基本面：优先尝试 `EDB`
- `EDB` 缺失或覆盖不到时：回退 `AkShare`

### 当前落地状态

`scripts/phase1_models.py`、`scripts/phase1_scoring.py`、`scripts/phase1_pipeline.py`、`scripts/fundamental_data.py`、`scripts/market_data_tq.py` 这些新模块已经落地并有测试，但主入口当前仍处在“现有基本面引擎 + 新机会发现语义 + 新报告字段”的过渡状态。

## 项目结构

```text
ctp/
├── config.yaml                  # 策略配置（Phase 1 关注优先级 / Phase 2 定方向）
├── requirements.txt             # 依赖
├── scripts/
│   ├── daily_workflow.py        # 每日自动化工作流（主入口）
│   ├── pre_market.py            # Phase 2 技术面与交易参数分析
│   ├── intraday.py              # Phase 3 盘中监控
│   ├── phase2_direction.py      # Phase 2 自主方向判定
│   ├── phase1_models.py         # Phase 1 结构化类型
│   ├── phase1_scoring.py        # Phase 1 评分函数
│   ├── phase1_pipeline.py       # Phase 1 候选排序
│   ├── market_data_tq.py        # TqSdk 日线/主连映射
│   ├── fundamental_data.py      # EDB -> AkShare 基本面入口
│   ├── tqsdk_live.py            # TqSdk 实时订阅封装
│   ├── wyckoff.py               # Wyckoff量价分析引擎
│   └── download_futures_data.py # 批量下载历史数据
├── data/                        # 运行时缓存（.gitignore）
└── logs/                        # 日志（.gitignore）
```

## 配置说明

`config.yaml` 包含：

- **positions** — 手动指定的品种及方向（`--skip-screen` 时使用）
- **pre_market** — Phase 2 分析参数：均线窗口、ATR/RSI/MACD/Bollinger、Fibonacci 级别
- **fundamental_screening** — Phase 1 机会发现门槛和品种分组
- **phase1** — Phase 1 报告摘要和候选池相关参数，例如 `top_n`
- **intraday** — Phase 3 日内参数：突破窗口、均值回归、量能过滤、风控阈值

## 安装

```bash
pip install -r requirements.txt
```
