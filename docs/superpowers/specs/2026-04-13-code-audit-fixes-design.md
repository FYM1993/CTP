# 量化系统代码审计修复设计

## 背景

对 Phase 1/2 评分体系重构后的代码进行全面审计（4 个核心模块），发现 6 个 Critical 级别 bug 和 7 个 Important 级别问题。本文档描述所有修复的设计方案。

## 问题清单

### Critical（必须修复）

| ID | 模块 | 问题 | 根因 |
|----|------|------|------|
| C1 | pre_market.py | `actionable` 未使用 `signal_strength >= 0.5` 门控 | 仅用 `has_signal`，与 spec 不一致 |
| C2 | wyckoff.py | `has_signal` 独立于 `signal_strength` 设置 | 应由 `signal_strength >= 0.5` 派生 |
| C3 | pre_market.py | `score_signals` 中 momentum 访问 `iloc[-21]` 无长度检查 | 短数据 IndexError |
| C4 | wyckoff.py | `assess_reversal_status` 使用 `df.iloc[-1]` 无空检查 | 空 DataFrame 崩溃 |
| C5 | data_cache.py | `get_hog_fundamentals` 的 `price_trend` 除零 | `price_5d_ago` 可能为 0 |
| C6 | data_cache.py | `get_seasonality` 月度收益 `last/first` 除零 | `first` 可能为 0 |

### Important（应修复）

| ID | 模块 | 问题 |
|----|------|------|
| I1 | daily_workflow.py | Phase 1 `_score_symbol` 仍计算 RSI/MA/returns（违反"纯基本面"原则） |
| I2 | wyckoff.py | SOS 强前序判定：SC(priority=1) 也能触发 priority=4 的 SOS |
| I3 | pre_market.py | `score_signals` 的 `direction` 参数声明但从未使用 |
| I4 | pre_market.py | NaN 传播：Bollinger/MA 可能产生 NaN 导致 `sum(scores)` 为 NaN |
| I5 | data_cache.py | GFEX 仓单非交易日返回空数据 |
| I6 | data_cache.py | `inv_percentile` docstring "分位数"实为 min-max 归一化 |
| I7 | wyckoff.py | `signal_strength` 的 prep boost(+0.1) 对所有 entry 生效，应仅限 SOS/SOW |

### 额外发现（Spec 自查阶段发现）

| ID | 模块 | 问题 | 根因 |
|----|------|------|------|
| C7 | daily_workflow.py | 季节性评分符号与其他维度不一致 | 其他维度用正=偏空约定，季节性用正=偏多，经 `-fund_score` 反转后方向反了 |

**C7 详细说明**：

所有非季节性维度的本地 `fund_score` 约定是 **正值=供应压力/偏空**：
- 累库 `inv_4wk > 10` → `fund_score += 10`
- 库存高位 `inv_pct > 85` → `fund_score += 10`
- 仓单大增 → `fund_score += 15`

但季节性使用 **正值=偏多**：
- `sig > 0`（历史偏上涨）→ `fund_score += sig*10`（正值=偏多）

经过 `score = -fund_score` 后，看多的季节性信号被反转为负分=做空方向。**这是一个预存 bug**。

方向约定清理（D1）会自然修复此问题：统一为正=偏多后，季节性保持原样，其余维度反转。

### 设计层清理

| ID | 模块 | 问题 |
|----|------|------|
| D1 | daily_workflow.py | `fund_score` 本地变量用反向约定（正=偏空），通过 `-fund_score` 反转 |
| D2 | data_cache.py | 未使用变量 `yesterday_col`、`has_today` |

## 修复设计

### 第 1 层：防御性编码

#### 1.1 `pre_market.py` — `score_signals` 数据长度保护

**文件**: `scripts/pre_market.py`，`score_signals` 函数

**修改**:
- 在函数开头获取 `n = len(df)`
- MACD 维度：`hist.iloc[-2]` 访问前检查 `n >= 2`，不足则 MACD 得分 = 0
- 动量维度：`close.iloc[-6]` 和 `close.iloc[-21]` 前检查 `n >= 21`，不足则动量得分 = 0
- 所有维度的计算结果统一用 `safe_score(value)` 确保非 NaN：

```python
def _safe(v: float) -> float:
    return float(v) if pd.notna(v) and np.isfinite(v) else 0.0
```

#### 1.2 `wyckoff.py` — `assess_reversal_status` 空数据保护

**文件**: `scripts/wyckoff.py`，`assess_reversal_status` 函数

**修改**: 函数开头加入早返回：

```python
if len(df) < 21:
    return {
        "has_signal": False, "signal_strength": 0.0,
        "reversal_type": "", "latest_signal": "",
        "all_events": [], "entry_events": [], "prep_events": [],
        "status_summary": "数据不足"
    }
```

21 是 `_find_events_with_bars` 内循环 `range(20, len)` 的最小有意义长度。

#### 1.3 `data_cache.py` — 除零保护

**文件**: `scripts/data_cache.py`

**`get_hog_fundamentals`**: `price_trend` 计算加保护：

```python
p5 = result["price_5d_ago"]
result["price_trend"] = (result["price"] / (p5 + 1e-12) - 1) * 100 if p5 > 0 else 0.0
```

**`get_seasonality`**: 月度收益计算过滤零值：

```python
monthly["first"] = monthly["first"].replace(0, np.nan)
monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
monthly = monthly.dropna(subset=["ret"])
```

#### 1.4 `data_cache.py` — GFEX 非交易日容错

**文件**: `scripts/data_cache.py`，`get_warehouse_receipt` GFEX 分支

**修改**: 日期回退逻辑，最多尝试 T ~ T-3：

```python
from datetime import timedelta

for offset in range(4):
    d = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
    try:
        cache = aks.futures_gfex_warehouse_receipt(date=d)
        if cache:  # 非空字典
            _gfex_receipt_cache = cache
            break
    except Exception:
        continue
else:
    _gfex_receipt_cache = {}
```

### 第 2 层：Spec-代码对齐

#### 2.1 `has_signal` 由 `signal_strength` 派生

**文件**: `scripts/wyckoff.py`，`assess_reversal_status` 函数

**修改**: 移除各分支中独立设置 `has_signal: True/False` 的逻辑。在函数末尾（返回前）统一设置：

```python
result["has_signal"] = result["signal_strength"] >= 0.5
```

具体来说，fresh_entry 分支只负责设置 `signal_strength`（基于 priority 映射），不再直接设 `has_signal`。stale entry / prep-only / suspect-only 分支同样只设 `signal_strength`。

#### 2.2 `actionable` 加入 `signal_strength` 门控

**文件**: `scripts/pre_market.py`，`analyze_one` 函数

**修改**: 在 `has_signal` 分支内的 actionable 判定中加入 strength 检查：

```python
strength_ok = reversal.get("signal_strength", 0) >= 0.5
actionable = score_ok and rr_ok and strength_ok
```

同步更新 `reason` 字符串，当 strength 不足时输出具体原因。

#### 2.3 SOS/SOW 前序优先级跟踪

**文件**: `scripts/wyckoff.py`，`_find_events_with_bars` 函数

**修改**: 将 `seen_signals: set[str]` 改为 `precursor_priority: dict[str, int]`，记录每个前序信号的最高优先级：

```python
precursor_priority = {}  # {"SC": 1, "Spring": 3, ...}

# 记录时:
if priority >= 1:
    precursor_priority[signal_name] = max(precursor_priority.get(signal_name, 0), priority)

# SOS 判定时:
bullish_keys = {"SC", "Spring", "StopVol_Bull"}
max_pre = max((precursor_priority.get(k, -1) for k in bullish_keys), default=-1)
if max_pre >= 2:
    sospriority = 4  # 强确认
elif has_weak_precursor:
    sospriority = 2  # 弱确认
else:
    sospriority = 0  # 疑似
```

`suspect_signals` 集合保留不变，用于弱确认路径。

#### 2.4 `signal_strength` prep boost 仅限 SOS/SOW

**文件**: `scripts/wyckoff.py`，`assess_reversal_status` 函数

**修改**: prep_events boost 条件收紧：

```python
if prep_events and fresh_entry["signal"] in ("SOS", "SOW"):
    base_strength = min(base_strength + 0.1, 0.95)
```

其他 entry（Spring/UT 等）有 prep_events 时不额外 boost，因为 Spring/UT 本身就是底部/顶部确认信号，不需要再叠加 prep 加成。

#### 2.5 `_score_symbol` 方向约定清理

**文件**: `scripts/daily_workflow.py`，`_score_symbol` 函数

**修改**: 统一所有维度为"正=偏多"约定，移除 `-fund_score` 反转。

**需要反转的维度**（当前用正=偏空，改为正=偏多）：

| 场景 | 修改前 | 修改后 | 经济逻辑 |
|------|--------|--------|---------|
| 去库 (inv_4wk < -10) | `fund_score -= 10` | `fund_score += 10` | 供应减少=看多 |
| 累库 (inv_4wk > 10) | `fund_score += 10` | `fund_score -= 10` | 供应增加=看空 |
| 去库 (inv_4wk < -3) | `fund_score -= 5` | `fund_score += 5` | 同上(弱) |
| 累库 (inv_4wk > 3) | `fund_score += 5` | `fund_score -= 5` | 同上(弱) |
| 库存低位 (< 15%) | `fund_score -= 10` | `fund_score += 10` | 低库存=看多 |
| 库存低位 (< 30%) | `fund_score -= 5` | `fund_score += 5` | 同上(弱) |
| 库存高位 (> 85%) | `fund_score += 10` | `fund_score -= 10` | 高库存=看空 |
| 库存高位 (> 70%) | `fund_score += 5` | `fund_score -= 5` | 同上(弱) |
| 累库周数多 (≥6) | `fund_score += 10` | `fund_score -= 10` | 持续累库=看空 |
| 累库周数少 (≤2) | `fund_score -= 10` | `fund_score += 10` | 持续去库=看多 |
| 仓单大增 (>5%) | `fund_score += 15` | `fund_score -= 15` | 交割品增加=看空 |
| 仓单小增 (>1%) | `fund_score += 8` | `fund_score -= 8` | 同上(弱) |
| 仓单大减 (>5%) | `fund_score -= 15` | `fund_score += 15` | 交割品减少=看多 |
| 仓单小减 (>1%) | `fund_score -= 8` | `fund_score += 8` | 同上(弱) |
| 生猪亏损 (pm<-15) | `fund_score -= 15` | `fund_score += 15` | 亏损→减产→看多 |
| 生猪亏损 (pm<-5) | `fund_score -= 8` | `fund_score += 8` | 同上(弱) |
| 生猪盈利 (pm>20) | `fund_score += 10` | `fund_score -= 10` | 盈利→增产→看空 |
| 生猪盈利 (pm>10) | `fund_score += 5` | `fund_score -= 5` | 同上(弱) |
| 生猪价格跌 (pt<-3) | `fund_score -= 5` | `fund_score += 5` | 低价→减产→看多 |
| 生猪价格涨 (pt>3) | `fund_score += 5` | `fund_score -= 5` | 高价→增产→看空 |

**保持不变的维度**（已经是正=偏多）：

| 场景 | 当前代码 | 说明 |
|------|---------|------|
| 季节性偏多 (sig>0) | `fund_score += sig*10` (正) | 历史上涨月份=看多，已正确 |
| 季节性偏空 (sig<0) | `fund_score += sig*10` (负) | 历史下跌月份=看空，已正确 |

**注意**：季节性在当前代码中使用的是 **正=偏多** 约定，与其他维度的 **正=偏空** 不一致。这是一个 **预存 bug (C7)**：当前经 `-fund_score` 反转后，季节性的方向被错误反转。统一约定后此 bug 自然消除。

最终：`score = fund_score`（移除取反），`result["fund_score"] = fund_score`。

输出行为变更：
- **库存/仓单/累库/生猪维度**：最终方向不变（反转约定 + 移除取反 = 两次取反抵消）
- **季节性维度**：方向**修正**（不再被错误反转），即看多月份真正产出正分=做多

### 第 3 层：架构清理

#### 3.1 `_score_symbol` 移除技术指标

**文件**: `scripts/daily_workflow.py`

**移除**: RSI(14) 计算（L188-191）、MA5/MA60 trend 计算（L193-195）、ret_5d/ret_20d 计算（L197-198）。

**移除返回字段**: `rsi`、`ma_trend`、`ret_5d`、`ret_20d`、`wyckoff_phase`。

**保留**: `range_pct`（极端价格安全网需要）。

**下游影响**:
- Phase 2 打印中的 `fund_rsi` 字段：改为直接从 Phase 2 计算结果中获取（`analyze_one` 已计算 RSI）
- `save_targets` 报告中的相关字段：移除或改用 Phase 2 数据

#### 3.2 `score_signals` 移除 `direction` 参数

**文件**: `scripts/pre_market.py`

**修改**: 函数签名从 `score_signals(df, direction, cfg)` 改为 `score_signals(df, cfg)`。

**调用方更新**: `pre_market.py` 内的 `analyze_one` 调用 `score_signals(df, cfg=cfg)` 即可。

#### 3.3 Docstring 修正

**文件**: `scripts/data_cache.py`，`get_inventory` docstring

**修改**: `inv_percentile: 库存在52周内的分位数 (0~100, 越高库存越高)` → `inv_percentile: 库存在52周高低区间的位置 (0~100, min-max归一化, 越高库存越高)`

#### 3.4 未使用变量清理

**文件**: `scripts/data_cache.py`

- 移除 `yesterday_col = "昨日仓单量"`（GFEX 仓单处理中）
- 移除 `has_today = api_calls - live_count`（`prefetch_all` 中）

## 文件影响范围

| 文件 | 改动量级 | 改动内容 |
|------|---------|---------|
| `scripts/daily_workflow.py` | 中等 | 方向约定反转 + 移除技术指标 + 报告字段适配 |
| `scripts/pre_market.py` | 小 | actionable 门控 + direction 参数移除 + NaN 保护 + 长度检查 |
| `scripts/wyckoff.py` | 中等 | has_signal 派生 + SOS 前序优先级 + prep boost 限定 + 空数据保护 |
| `scripts/data_cache.py` | 小 | 除零保护 + GFEX 日期容错 + docstring + 死代码清理 |

## 风险与缓解

1. **方向约定反转可能引入符号错误**：需要逐项验证每个 `+=` / `-=` 的反转。库存/仓单等维度的最终输出方向不变（反转+移除取反=两次取反抵消），但季节性维度会被**修正**（之前方向是反的）。需验证修正后的季节性方向符合预期
2. **`score_signals` 移除 `direction` 参数**：需检查所有调用方，确认无遗漏
3. **SOS 前序优先级改变**：可能影响历史信号的回测结果，SC(priority=1) 不再能触发强 SOS
4. **GFEX 日期回退**：额外 API 调用可能增加延迟，但最多 4 次，可接受
5. **季节性方向修正 (C7)**：之前看多季节性信号会产出做空方向（bug），修正后可能改变部分品种的推荐方向。这是正确的修复，但需关注首日运行报告的方向变化
