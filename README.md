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

# 调整筛选灵敏度（默认25，越大越严格）
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

## 分析体系

### 评分维度 (盘前分析 ±100分)

| 层级 | 指标 | 权重 | 来源 |
|------|------|------|------|
| 经典指标 | 均线排列 | 15 | MA5/10/20 vs MA60/120 |
| 经典指标 | MACD | 10 | DIF/DEA/柱状图方向 |
| 经典指标 | RSI(14) | 10 | 超买>70 / 超卖<30 |
| 经典指标 | 布林带位置 | 10 | 价格在带内百分位 |
| 经典指标 | 动量 | 5 | 5日+20日涨跌幅 |
| **Wyckoff量价** | **市场阶段** | **15** | 吸筹/上涨/派发/下跌 |
| **Wyckoff量价** | **量价关系** | **10** | 上涨日量 vs 下跌日量 |
| **Wyckoff量价** | **VSA信号** | **10** | 每根K线的量价行为分类 |
| **期货持仓** | **OI信号** | **15** | 价格+持仓量四象限 |

### 数据源优先级

- 价格、K线和持仓量优先使用 `TqSdk`
- 非价量类基本面优先使用 `EDB`
- `EDB` 缺失或覆盖不到时，再回退 `AkShare`

### Wyckoff 理论

- **市场阶段**: 自动识别吸筹(Accumulation)/派发(Distribution)/上涨(Markup)/下跌(Markdown)
- **关键事件**: 卖方高潮(SC)、弹簧(Spring)、上冲回落(UT)、强势信号(SOS)、弱势信号(SOW)
- **VSA分类**: 停止量、无需求、无供给、上涨受阻、下跌受托、买方/卖方高潮

### 期货持仓量分析

| 价格 | 持仓量 | 信号 | 含义 |
|------|--------|------|------|
| 涨 | 增 | 新多入场 | 强势看多 |
| 涨 | 减 | 空头回补 | 弱势看多 |
| 跌 | 增 | 新空入场 | 强势看空 |
| 跌 | 减 | 多头平仓 | 弱势看空(可能见底) |

## 项目结构

```
ctp/
├── config.yaml                  # 策略配置（Phase 1 关注优先级 / Phase 2 定方向）
├── requirements.txt             # 依赖
├── scripts/
│   ├── daily_workflow.py        # 每日自动化工作流（主入口，含 Phase 1 全市场筛选）
│   ├── pre_market.py            # 盘前深度分析
│   ├── intraday.py              # 盘中实时监控
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
- **phase1** — Phase 1 报告/候选池参数，例如 `top_n`
- **intraday** — Phase 3 日内参数：突破窗口、均值回归、量能过滤、风控阈值

## 安装

```bash
pip install -r requirements.txt
```
