# Phase 1/2 评分体系重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Phase 1 重构为纯基本面筛选，Phase 2 承接全部技术面分析，并修复 Wyckoff 反转信号级联失败。

**Architecture:** Phase 1 (`_score_symbol`) 移除所有技术面维度，仅保留库存/仓单/季节性/生猪基本面，阈值降至 10。新增极端价格安全网。Phase 2 (`score_signals`) 新增"价格位置"维度。`wyckoff.py` 新增带力弹簧路径、SOS 弱前序匹配、`signal_strength` 连续值。`daily_workflow.py` 适配新字段和 RRF 融合。

**Tech Stack:** Python, pandas, numpy

---

### Task 1: Phase 1 重构 — `_score_symbol` 改为纯基本面

**Files:**
- Modify: `scripts/daily_workflow.py:159-360` (`_score_symbol` 函数)
- Modify: `scripts/daily_workflow.py:111-156` (`phase_1_screen` 函数)

- [ ] **Step 1: 移除 `_score_symbol` 中的技术面评分**

在 `scripts/daily_workflow.py` 的 `_score_symbol` 函数中，删除 `tech_score` 的所有计算逻辑（价格区间位、RSI、均线趋势、Wyckoff 量价），以及 `wyckoff_score` 的导入调用。保留 `range_pct` 的**计算**（后续用于安全网和输出），但不纳入评分。

移除持仓结构(OI)评分（`get_oi_structure` 调用和 `oi_vs_price`/`oi_percentile` 评分逻辑）。

最终评分公式从 `score = -(tech_score + fund_score)` 简化为 `score = -fund_score`（保持正=做多、负=做空的约定）。

保留 `range_pct`、`rsi`、`ret_5d`、`ret_20d` 的计算和返回值（Phase 2 和输出需要），但它们不再影响 `score`。

```python
def _score_symbol(sym: str, name: str, exchange: str, df: pd.DataFrame) -> dict | None:
    """
    对单品种计算筛选评分（纯基本面）。

    评分维度:
      基本面: 库存水平(±20), 库存分位(±10), 仓单变化(±15),
              季节性(±10), 生猪专项(±15)
    """
    try:
        close = df["close"]
        last = float(close.iloc[-1])
        n = min(len(df), 300)
        recent = df.tail(n)

        high_all = float(recent["close"].max())
        low_all = float(recent["close"].min())
        range_pct = (last - low_all) / (high_all - low_all + 1e-12) * 100

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - 100 / (1 + gain / (loss + 1e-12))).iloc[-1])

        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma60 = float(close.rolling(min(60, len(close))).mean().iloc[-1])
        ma_trend = (ma5 / ma60 - 1) * 100

        ret_5d = (last / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0
        ret_20d = (last / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0

        # ---- 基本面数据采集 ----
        inv = get_inventory(sym)
        receipt = get_warehouse_receipt(sym)
        seasonal = get_seasonality(df)
        hog = get_hog_fundamentals() if sym == "LH0" else None

        # ---- 基本面评分 ----
        fund_score = 0.0
        fund_details = []

        # 库存变化 (±20) — 逻辑不变
        inv_4wk = None
        inv_pct = None
        if inv:
            inv_4wk = inv["inv_change_4wk"]
            inv_pct = inv.get("inv_percentile")
            cum = inv.get("inv_cumulating_weeks", 4)
            trend = inv.get("inv_trend", "持平")

            if inv_4wk > 10: fund_score += 10
            elif inv_4wk > 3: fund_score += 5
            elif inv_4wk < -10: fund_score -= 10
            elif inv_4wk < -3: fund_score -= 5
            if cum >= 6: fund_score += 10
            elif cum <= 2: fund_score -= 10

            fund_details.append(f"库存{trend}({inv_4wk:+.1f}%)")

        # 库存分位 (±10) — 逻辑不变
        if inv_pct is not None:
            if inv_pct > 85:
                fund_score += 10
                fund_details.append(f"库存高位{inv_pct:.0f}%")
            elif inv_pct > 70:
                fund_score += 5
            elif inv_pct < 15:
                fund_score -= 10
                fund_details.append(f"库存低位{inv_pct:.0f}%")
            elif inv_pct < 30:
                fund_score -= 5

        # 仓单 (±15) — 逻辑不变
        receipt_change = None
        if receipt:
            rc = receipt["receipt_change"]
            receipt_change = rc
            rt = receipt["receipt_total"]
            if rc > 0 and rt > 0:
                change_ratio = rc / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score += 15
                    fund_details.append(f"仓单大增{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score += 8
            elif rc < 0 and rt > 0:
                change_ratio = abs(rc) / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score -= 15
                    fund_details.append(f"仓单大减{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score -= 8

        # 季节性 (±10) — 逻辑不变
        seasonal_sig = None
        if seasonal:
            sig = seasonal["seasonal_signal"]
            seasonal_sig = sig
            avg_ret = seasonal["hist_avg_return"]
            if abs(sig) > 0.6:
                s = np.clip(sig * 10, -10, 10)
                fund_score += s
                direction = "上涨" if sig > 0 else "下跌"
                fund_details.append(f"季节性偏{direction}(历史{avg_ret:+.1f}%)")

        # 生猪专项 (±15) — 逻辑不变
        hog_profit = None
        if hog:
            if "profit_margin" in hog:
                pm = hog["profit_margin"]
                hog_profit = pm
                if pm < -15:
                    fund_score -= 15
                    fund_details.append(f"养殖亏损{pm:.0f}%")
                elif pm < -5:
                    fund_score -= 8
                elif pm > 20:
                    fund_score += 10
                elif pm > 10:
                    fund_score += 5

            if "price_trend" in hog:
                pt = hog["price_trend"]
                if pt < -3:
                    fund_score -= 5
                elif pt > 3:
                    fund_score += 5

        # 纯基本面: 正分=看多(做多), 负分=看空(做空)
        score = -fund_score

        return {
            "symbol": sym, "name": name, "exchange": exchange,
            "price": last, "range_pct": range_pct,
            "rsi": rsi, "ma_trend": ma_trend,
            "ret_5d": ret_5d, "ret_20d": ret_20d,
            "wyckoff_phase": "",
            "score": score,
            "fund_score": -fund_score,
            "fund_details": ", ".join(fund_details) if fund_details else "",
            "inv_change_4wk": inv_4wk,
            "inv_percentile": inv_pct,
            "receipt_change": receipt_change,
            "seasonal_signal": seasonal_sig,
            "hog_profit": hog_profit,
        }
    except Exception as e:
        return None
```

- [ ] **Step 2: 修改 `phase_1_screen` — 降低阈值 + 极端价格安全网**

在 `scripts/daily_workflow.py` 的 `phase_1_screen` 函数中：

1. 默认阈值参数从 `threshold: float = 25` 改为 `threshold: float = 10`
2. 在 `abs(result["score"]) >= threshold` 的判断后，增加安全网：即使基本面分数不达标，如果 `range_pct < 5 或 > 95`，也将品种加入候选池
3. 移除返回字典中的 `tech_score` 字段（不再有技术面评分），保持 `score` 和 `fund_score`

```python
def phase_1_screen(all_data: dict[str, pd.DataFrame], threshold: float = 10) -> list[dict]:
    # ... 循环内 ...
        result = _score_symbol(sym, info["name"], info["exchange"], df)
        if result is None:
            continue

        fund_pass = abs(result["score"]) >= threshold
        extreme_price = result["range_pct"] < 5 or result["range_pct"] > 95

        if fund_pass or extreme_price:
            if fund_pass:
                result["direction"] = "long" if result["score"] > 0 else "short"
            else:
                # 安全网: 无基本面方向，由价格极端位决定
                result["direction"] = "long" if result["range_pct"] < 50 else "short"
                result["score"] = 1.0 if result["direction"] == "long" else -1.0
            candidates.append(result)
```

4. 更新打印输出，去掉 `tech_score` 列，更新标题文案为"基本面筛选"。

```python
    print(f"\n  ✅ 通过筛选 (|基本面| ≥ {threshold} 或极端价格): {len(candidates)} 个品种\n")

    # ...
    def _print_cand(c):
        extreme = " ⚡极端价格" if (c["range_pct"] < 5 or c["range_pct"] > 95) else ""
        line = (f"     {c['name']:6s} ({c['symbol']:5s})  "
                f"区间位{c['range_pct']:3.0f}%  "
                f"基本面={c.get('fund_score', 0):+.0f}  "
                f"总分={c['score']:+.0f}{extreme}")
        fd = c.get("fund_details", "")
        if fd:
            line += f"  [{fd}]"
        print(line)
```

- [ ] **Step 3: 更新 `main` 中的 threshold 参数默认值**

在 `scripts/daily_workflow.py:1015` 的 argparse 中，将 `--threshold` 的 default 从 25 改为 10：

```python
    parser.add_argument("--threshold", type=float, default=10, help="筛选阈值 (默认10)")
```

- [ ] **Step 4: 移除 `_score_symbol` 中不再需要的 import**

在 `scripts/daily_workflow.py:47`，从 import 行中移除 `wyckoff_score`（如果它仅被 `_score_symbol` 使用），以及移除 `get_oi_structure`：

```python
from data_cache import (
    get_all_symbols, get_daily, get_minute, get_inventory,
    get_warehouse_receipt, get_hog_fundamentals,
    get_seasonality,
    prefetch_all, get_request_count,
    get_daily_with_live_bar,
)
```

检查 `wyckoff_score` 是否在其他地方使用（`phase_3_intraday` 等），如果没有则也从 import 中移除。`assess_reversal_status` 仍被 Phase 2/3 使用，保留。

- [ ] **Step 5: 验证 Phase 1 修改**

Run: `cd /Users/yimin.fu/PythonProjects/ctp && python -c "from scripts.daily_workflow import _score_symbol; print('import ok')"`
Expected: `import ok`，无报错

- [ ] **Step 6: Commit**

```bash
git add scripts/daily_workflow.py
git commit -m "refactor: Phase 1 becomes pure fundamental screening

Remove technical dimensions (range_pct, RSI, MA trend, Wyckoff, OI)
from _score_symbol. Lower threshold from 25 to 10. Add extreme price
safety net for commodities without fundamental data coverage."
```

---

### Task 2: Phase 2 增强 — `score_signals` 新增价格位置维度

**Files:**
- Modify: `scripts/pre_market.py:174-319` (`score_signals` 函数)

- [ ] **Step 1: 在 `score_signals` 中新增"价格位置"维度**

在 `scripts/pre_market.py` 的 `score_signals` 函数中，在"动量"维度之后、Wyckoff 量价分析之前，新增"价格位置"维度(±5)：

```python
    # 6. 价格位置 (5) — 从 Phase 1 迁入，作为上下文而非逆势信号
    n = min(len(df), 300)
    recent_prices = df.tail(n)["close"]
    high_all = float(recent_prices.max())
    low_all = float(recent_prices.min())
    range_pct = (last - low_all) / (high_all - low_all + 1e-12) * 100
    if range_pct < 10:
        scores["价格位置"] = 5
    elif range_pct > 90:
        scores["价格位置"] = -5
    else:
        scores["价格位置"] = 0
```

- [ ] **Step 2: 更新 `score_signals` 的文档注释**

更新函数的 docstring，将总分改为反映新的构成：

```python
    """
    综合评分: -100(强空) ~ +100(强多)

    评分体系 (总计 ±105):
      经典指标 (55分):
        均线排列  15分  — 趋势方向
        MACD     10分  — 趋势动能
        RSI      10分  — 超买超卖
        布林带    10分  — 价格位置
        动量       5分  — 短期惯性
        价格位置   5分  — 历史高低点间的位置上下文
      Wyckoff 量价 (35分):
        阶段判断  15分  — 吸筹/派发/上涨/下跌
        量价关系  10分  — 上涨日量 vs 下跌日量
        VSA信号  10分  — 最近K线的量价行为
      期货持仓 (15分):
        OI信号   15分  — 价格+持仓量四象限
    """
```

- [ ] **Step 3: 更新 `analyze_one` 中的打印和 section_labels**

在 `scripts/pre_market.py` 的 `analyze_one` 函数中（约 `score_signals` 调用后的打印区域），更新 `section_labels` 字典，添加"价格位置"：

```python
    section_labels = {
        "均线排列": "经典", "MACD": "经典", "RSI": "经典", "布林带": "经典", "动量": "经典",
        "价格位置": "经典",
        "Wyckoff阶段": "量价", "量价关系": "量价", "VSA信号": "量价",
        "持仓信号": "持仓",
    }
```

同时更新 `classical_keys` 列表：

```python
    classical_keys = ["均线排列", "MACD", "RSI", "布林带", "动量", "价格位置"]
```

- [ ] **Step 4: 验证 Phase 2 修改**

Run: `cd /Users/yimin.fu/PythonProjects/ctp && python -c "from scripts.pre_market import score_signals; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: Commit**

```bash
git add scripts/pre_market.py
git commit -m "feat: add price-position dimension to Phase 2 scoring

New ±5 scoring dimension based on range_pct (300-day high/low position).
Extreme low (<10%) gives +5 (bullish context), extreme high (>90%)
gives -5 (bearish context). Neutral in between."
```

---

### Task 3: Wyckoff 带力弹簧 — `_find_events_with_bars` Spring/UT 增强

**Files:**
- Modify: `scripts/wyckoff.py:678-908` (`_find_events_with_bars` 函数)

- [ ] **Step 1: 在 Spring 检测中新增"带力弹簧"路径**

在 `scripts/wyckoff.py` 的 `_find_events_with_bars` 函数中，找到 Spring 检测区域（约第 765-795 行）。在现有的经典 Spring（`context_ok and vol_ok and close_ok`）分支和疑似 Spring（`else`）分支之间，插入"带力弹簧"路径。

将当前的 if/else 结构改为三分支：

```python
        # ====== Spring ======
        if row["low"] < prev_low and row["close"] > prev_low:
            depth = (prev_low - row["low"]) / (row["close"] + 1e-12) * 100
            if depth > 0.2:
                trend_slowed = ret_10d > -0.02
                context_ok = has_trading_range or trend_slowed
                vol_ok = rv < 1.5
                close_ok = cp > 0.5

                if context_ok and vol_ok and close_ok:
                    # 经典 Spring: 缩量试探 + 收盘偏高
                    events.append({
                        "signal": "Spring", "date": date_str, "bias": "bullish",
                        "bar": bar,
                        "detail": f"跌破前低{prev_low:.0f}后收回{cp:.0%}位, "
                                  f"缩量(量比{rv:.1f}), {'横盘中' if has_trading_range else '趋势放缓'}",
                        "ref_level": prev_low, "priority": 3,
                    })
                    seen_signals.add("Spring")
                elif context_ok and rv >= 1.5 and cp > 0.6:
                    # 带力弹簧: 放量但强力收回（主力底部吸筹）
                    events.append({
                        "signal": "Spring", "date": date_str, "bias": "bullish",
                        "bar": bar,
                        "detail": f"带力弹簧: 跌破前低{prev_low:.0f}后强力收回{cp:.0%}位, "
                                  f"放量(量比{rv:.1f}), {'横盘中' if has_trading_range else '趋势放缓'}",
                        "ref_level": prev_low, "priority": 2,
                    })
                    seen_signals.add("Spring")
                else:
                    reasons = []
                    if not context_ok:
                        reasons.append("无横盘且下跌未放缓")
                    if not vol_ok and cp <= 0.6:
                        reasons.append(f"放量(量比{rv:.1f})且收盘不够强({cp:.0%})")
                    elif not vol_ok:
                        reasons.append(f"放量(量比{rv:.1f})非缩量试探")
                    if not close_ok and rv < 1.5:
                        reasons.append(f"收盘偏低({cp:.0%})")
                    events.append({
                        "signal": "Spring", "date": date_str, "bias": "bullish",
                        "bar": bar,
                        "detail": f"疑似Spring: 跌破前低{prev_low:.0f}后收回, 但{'; '.join(reasons)}",
                        "ref_level": prev_low, "priority": 0,
                    })
```

- [ ] **Step 2: 在 Upthrust 检测中新增"带力上冲回落"路径**

对称地，在 Upthrust 检测区域（约第 800-830 行），添加相同的三分支结构：

```python
        # ====== Upthrust ======
        if row["high"] > prev_high and row["close"] < prev_high:
            depth = (row["high"] - prev_high) / (row["close"] + 1e-12) * 100
            if depth > 0.2:
                trend_slowed = ret_10d < 0.02
                context_ok = has_trading_range or trend_slowed
                vol_ok = rv < 1.5
                close_ok = cp < 0.5

                if context_ok and vol_ok and close_ok:
                    events.append({
                        "signal": "UT", "date": date_str, "bias": "bearish",
                        "bar": bar,
                        "detail": f"突破前高{prev_high:.0f}后回落{cp:.0%}位, "
                                  f"缩量(量比{rv:.1f}), {'横盘中' if has_trading_range else '趋势放缓'}",
                        "ref_level": prev_high, "priority": 3,
                    })
                    seen_signals.add("UT")
                elif context_ok and rv >= 1.5 and cp < 0.4:
                    # 带力上冲回落: 放量但强力回落（主力顶部派发）
                    events.append({
                        "signal": "UT", "date": date_str, "bias": "bearish",
                        "bar": bar,
                        "detail": f"带力UT: 突破前高{prev_high:.0f}后强力回落{cp:.0%}位, "
                                  f"放量(量比{rv:.1f}), {'横盘中' if has_trading_range else '趋势放缓'}",
                        "ref_level": prev_high, "priority": 2,
                    })
                    seen_signals.add("UT")
                else:
                    reasons = []
                    if not context_ok:
                        reasons.append("无横盘且上涨未放缓")
                    if not vol_ok and cp >= 0.4:
                        reasons.append(f"放量(量比{rv:.1f})且回落不够深({cp:.0%})")
                    elif not vol_ok:
                        reasons.append(f"放量(量比{rv:.1f})")
                    if not close_ok and rv < 1.5:
                        reasons.append(f"收盘偏高({cp:.0%})")
                    events.append({
                        "signal": "UT", "date": date_str, "bias": "bearish",
                        "bar": bar,
                        "detail": f"疑似UT: 突破前高{prev_high:.0f}后回落, 但{'; '.join(reasons)}",
                        "ref_level": prev_high, "priority": 0,
                    })
```

- [ ] **Step 3: Commit**

```bash
git add scripts/wyckoff.py
git commit -m "feat: add power-spring and power-UT detection paths

New priority=2 signal path for high-volume (rv>=1.5) Spring/UT
where close position is strong (cp>0.6 for Spring, cp<0.4 for UT).
These represent institutional accumulation/distribution and should
not be rejected just because volume is high."
```

---

### Task 4: SOS/SOW 弱前序匹配

**Files:**
- Modify: `scripts/wyckoff.py:678-908` (`_find_events_with_bars` 函数)

- [ ] **Step 1: 收集疑似信号到单独集合**

在 `_find_events_with_bars` 函数中（约第 696 行），在现有 `seen_signals = set()` 之后添加：

```python
    suspect_signals = set()
```

然后在每个疑似事件（priority=0）的 append 之后，添加对应的 `suspect_signals.add(...)`。具体地：

- SC 疑似（约第 738 行附近）后添加 `suspect_signals.add("SC")`
- BC 疑似后添加 `suspect_signals.add("BC")`
- Spring 疑似后添加 `suspect_signals.add("Spring")`
- UT 疑似后添加 `suspect_signals.add("UT")`

- [ ] **Step 2: 修改 SOS 检测 — 引入弱前序匹配**

在 SOS 检测区域（约第 835-860 行），修改 `has_precursor` 逻辑：

```python
        # ====== SOS (Sign of Strength) ======
        if rv >= 1.5 and row["close"] > prev_high:
            breakout_pct = (row["close"] - prev_high) / (prev_high + 1e-12) * 100
            has_strong_precursor = bool(seen_signals & {"SC", "Spring", "StopVol_Bull"})
            has_weak_precursor = bool(suspect_signals & {"SC", "Spring", "StopVol_Bull"})
            breakout_ok = breakout_pct > 0.5

            if has_strong_precursor and breakout_ok:
                events.append({
                    "signal": "SOS", "date": date_str, "bias": "bullish",
                    "bar": bar,
                    "detail": f"放量(量比{rv:.1f})突破前高{prev_high:.0f}(+{breakout_pct:.1f}%), 有前序吸筹信号",
                    "ref_level": prev_high, "priority": 4,
                })
                seen_signals.add("SOS")
            elif has_weak_precursor and breakout_ok:
                events.append({
                    "signal": "SOS", "date": date_str, "bias": "bullish",
                    "bar": bar,
                    "detail": f"放量(量比{rv:.1f})突破前高{prev_high:.0f}(+{breakout_pct:.1f}%), 有疑似前序信号(弱确认)",
                    "ref_level": prev_high, "priority": 2,
                })
                seen_signals.add("SOS")
            else:
                reasons = []
                if not has_strong_precursor and not has_weak_precursor:
                    reasons.append("无前序SC/Spring信号")
                if not breakout_ok:
                    reasons.append(f"突破幅度仅{breakout_pct:.1f}%")
                events.append({
                    "signal": "SOS", "date": date_str, "bias": "bullish",
                    "bar": bar,
                    "detail": f"疑似SOS: 放量突破前高{prev_high:.0f}, 但{'; '.join(reasons)}",
                    "ref_level": prev_high, "priority": 0,
                })
```

- [ ] **Step 3: 修改 SOW 检测 — 对称地引入弱前序**

在 SOW 检测区域（约第 864-890 行），对称修改：

```python
        # ====== SOW (Sign of Weakness) ======
        if rv >= 1.5 and row["close"] < prev_low:
            breakdown_pct = (prev_low - row["close"]) / (prev_low + 1e-12) * 100
            has_strong_precursor = bool(seen_signals & {"BC", "UT", "StopVol_Bear"})
            has_weak_precursor = bool(suspect_signals & {"BC", "UT", "StopVol_Bear"})
            breakdown_ok = breakdown_pct > 0.5

            if has_strong_precursor and breakdown_ok:
                events.append({
                    "signal": "SOW", "date": date_str, "bias": "bearish",
                    "bar": bar,
                    "detail": f"放量(量比{rv:.1f})跌破前低{prev_low:.0f}(-{breakdown_pct:.1f}%), 有前序派发信号",
                    "ref_level": prev_low, "priority": 4,
                })
                seen_signals.add("SOW")
            elif has_weak_precursor and breakdown_ok:
                events.append({
                    "signal": "SOW", "date": date_str, "bias": "bearish",
                    "bar": bar,
                    "detail": f"放量(量比{rv:.1f})跌破前低{prev_low:.0f}(-{breakdown_pct:.1f}%), 有疑似前序信号(弱确认)",
                    "ref_level": prev_low, "priority": 2,
                })
                seen_signals.add("SOW")
            else:
                reasons = []
                if not has_strong_precursor and not has_weak_precursor:
                    reasons.append("无前序BC/UT信号")
                if not breakdown_ok:
                    reasons.append(f"跌破幅度仅{breakdown_pct:.1f}%")
                events.append({
                    "signal": "SOW", "date": date_str, "bias": "bearish",
                    "bar": bar,
                    "detail": f"疑似SOW: 放量跌破前低{prev_low:.0f}, 但{'; '.join(reasons)}",
                    "ref_level": prev_low, "priority": 0,
                })
```

- [ ] **Step 4: Commit**

```bash
git add scripts/wyckoff.py
git commit -m "feat: SOS/SOW accept suspect precursor events

Introduce suspect_signals set to track priority=0 events. SOS/SOW
now fire at priority=2 when only suspect precursors exist, instead
of falling to priority=0. This breaks the cascade failure where
a suspect Spring prevented SOS from triggering."
```

---

### Task 5: `assess_reversal_status` 新增 `signal_strength`

**Files:**
- Modify: `scripts/wyckoff.py:911-1044` (`assess_reversal_status` 函数)

- [ ] **Step 1: 添加 `signal_strength` 计算**

在 `scripts/wyckoff.py` 的 `assess_reversal_status` 函数中，修改返回字典，新增 `signal_strength` 字段。

在函数开头的 `_no_signal` 字典中添加 `"signal_strength": 0.0`。

然后在各个返回路径中计算 `signal_strength`：

**有新鲜入场信号的返回路径**（约第 971-997 行）：

```python
    if fresh_entry is not None:
        confidence = 0.7 if fresh_entry["signal"] in ("SOS", "SOW") else 0.55
        if prep_events:
            confidence = min(confidence + 0.15, 0.95)

        # signal_strength: 基于信号类型的 priority 和 confidence
        pri = fresh_entry.get("priority", 2)
        if pri >= 4:
            strength = 0.85
        elif pri >= 3:
            strength = 0.75
        elif pri >= 2:
            strength = 0.65
        else:
            strength = 0.5
        if prep_events:
            strength = min(strength + 0.1, 0.95)

        # ... 保持现有的 stage/next_exp 逻辑不变 ...

        return {
            "has_signal": True,
            "signal_strength": strength,
            # ... 其余字段不变 ...
        }
```

**有老入场信号的路径**（约第 1000-1014 行）：

```python
        result["signal_strength"] = 0.45  # 有信号但不新鲜
```

**有预备信号的路径**（约第 1016-1028 行）：

```python
        result["signal_strength"] = 0.35  # 有底/顶迹象
```

**无信号的路径**（约第 1030-1043 行）：

```python
        if suspect:
            result["signal_strength"] = 0.15  # 有疑似信号
        else:
            result["signal_strength"] = 0.0
```

- [ ] **Step 2: 保留 `has_signal` 的兼容性**

`has_signal` 的逻辑不变（仅当有新鲜入场信号时为 True）。不将 `has_signal` 改为基于 `signal_strength >= 0.5`，因为这会改变现有的入场条件语义。`signal_strength` 作为新增信息供排名和展示使用。

- [ ] **Step 3: Commit**

```bash
git add scripts/wyckoff.py
git commit -m "feat: add signal_strength continuous value to reversal assessment

New 0~1 field reflecting reversal signal quality: 0.0 for no signal,
0.15 for suspect events, 0.35 for prep signals, 0.65-0.85 for fresh
entry signals based on priority level. Existing has_signal unchanged."
```

---

### Task 6: `daily_workflow.py` 适配 — Phase 2 集成 + RRF + 报告

**Files:**
- Modify: `scripts/daily_workflow.py:366-495` (`phase_2_premarket` 函数)
- Modify: `scripts/daily_workflow.py:671-992` (`save_targets` 函数)

- [ ] **Step 1: 更新 `phase_2_premarket` 中的字段映射**

在 `scripts/daily_workflow.py` 的 `phase_2_premarket` 函数中，更新从 Phase 1 候选传递到 Phase 2 结果的字段：

1. 移除 `result["fund_tech_score"]` 和 `result["fund_fund_score"]`（Phase 1 不再有技术面分拆）
2. 保留 `result["fund_screen_score"] = cand.get("score", 0)` — 现在纯粹是基本面分
3. 新增 `result["signal_strength"] = result.get("reversal_status", {}).get("signal_strength", 0.0)`

```python
            if result:
                result["fund_range_pct"] = cand.get("range_pct", 0)
                result["fund_rsi"] = cand.get("rsi", 50)
                result["fund_inv_change"] = cand.get("inv_change_4wk")
                result["fund_inv_percentile"] = cand.get("inv_percentile")
                result["fund_receipt_change"] = cand.get("receipt_change")
                result["fund_oi_vs_price"] = cand.get("oi_vs_price")
                result["fund_seasonal"] = cand.get("seasonal_signal")
                result["fund_hog_profit"] = cand.get("hog_profit")
                result["fund_screen_score"] = cand.get("score", 0)
                result["fund_details"] = cand.get("fund_details", "")
                result["signal_strength"] = result.get("reversal_status", {}).get("signal_strength", 0.0)
```

（注意：删除 `result["fund_tech_score"]` 和 `result["fund_fund_score"]` 两行）

- [ ] **Step 2: 添加 RRF 排名融合**

在 `phase_2_premarket` 的排序区域（现有的 `actionable.sort(...)` 之前），添加 RRF 计算：

```python
    RRF_K = 10
    all_results = actionable + watchlist
    if all_results:
        p1_ranked = sorted(all_results, key=lambda x: abs(x.get("fund_screen_score", 0)), reverse=True)
        p2_ranked = sorted(all_results, key=lambda x: abs(x.get("score", 0)), reverse=True)
        p1_rank = {id(r): i + 1 for i, r in enumerate(p1_ranked)}
        p2_rank = {id(r): i + 1 for i, r in enumerate(p2_ranked)}
        for r in all_results:
            r1 = p1_rank[id(r)]
            r2 = p2_rank[id(r)]
            r["rrf_score"] = 1.0 / (RRF_K + r1) + 1.0 / (RRF_K + r2)
            r["rank_p1"] = r1
            r["rank_p2"] = r2

    actionable.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    actionable = actionable[:max_picks]
    watchlist.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
```

- [ ] **Step 3: 更新 Phase 2 汇总的控制台输出**

更新打印表头和内容，展示 RRF、P1/P2 排名、信号强度：

```python
    if actionable:
        print(f"\n  {'':2s}{'品种':8s} {'代码':6s} {'方向':4s} {'RRF':>7s} {'#P1':>4s} {'#P2':>4s}"
              f" {'基本面':>6s} {'技术面':>6s} {'信号':>6s} {'强度':>5s} {'入场':>8s} {'盈亏比':>6s}")
        print(f"  {'─'*88}")
        for a in actionable:
            rev = a.get("reversal_status", {})
            sig = rev.get("signal_type", "")
            sig_cn = {"Spring": "弹簧", "SOS": "SOS", "SC": "SC",
                      "UT": "UT", "SOW": "SOW", "BC": "BC",
                      "StopVol_Bull": "停止量", "StopVol_Bear": "停止量"}.get(sig, sig)
            dir_str = "做多" if a["direction"] == "long" else "做空"
            rrf = a.get("rrf_score", 0)
            r1 = a.get("rank_p1", 0)
            r2 = a.get("rank_p2", 0)
            p1 = a.get("fund_screen_score", 0)
            ss = a.get("signal_strength", 0)
            print(f"  🟢{a['name']:8s} {a['symbol']:6s} {dir_str:4s} "
                  f"{rrf:>6.4f} {r1:>4d} {r2:>4d} {p1:>+5.0f} {a['score']:>+5.0f} "
                  f"{sig_cn:6s} {ss:>4.2f} {a['entry']:>8.0f} {a['rr']:>5.1f}")
```

对 watchlist 做类似更新。

- [ ] **Step 4: 更新 `save_targets` 报告模板**

在 `scripts/daily_workflow.py` 的 `save_targets` 函数中：

1. 更新可操作品种表头：

```python
        lines.append("| 状态 | 合约 | 方向 | 当前价 | 入场信号 | 入场价 | 止损 | 止盈1 | 盈亏比 | RRF | #P1 | #P2 | 基本面 | 技术面 | 信号强度 | 基本面详情 |")
```

2. 更新观望品种表头，同样添加 RRF / 信号强度列

3. 在"评分分解"区域：
   - 移除 `fund_tech_score`/`fund_fund_score` 的引用
   - 将 `Phase 1 筛选` 的描述从 "(技术X/基本面Y)" 改为纯基本面分数

4. 更新"评分说明"部分：
   - Phase 1 筛选评分体系：删除技术面维度表格，只保留基本面维度
   - 新增 RRF 综合排名说明（与之前实现的一致）
   - 更新 Phase 2 深度分析评分体系，新增"价格位置"行

- [ ] **Step 5: 验证完整流程**

Run: `cd /Users/yimin.fu/PythonProjects/ctp && python scripts/daily_workflow.py --no-monitor --threshold 10 2>&1 | tail -40`

Expected: Phase 1 输出显示"基本面筛选"，通过品种列表中不再有 tech_score 列。Phase 2 输出包含 RRF 排名。报告文件生成成功。

- [ ] **Step 6: Commit**

```bash
git add scripts/daily_workflow.py
git commit -m "feat: integrate RRF fusion, signal_strength, and updated reports

Phase 2 now shows RRF ranking (P1 fundamental + P2 technical),
signal_strength in console and report output. Report template
updated to reflect pure-fundamental Phase 1 and enhanced Phase 2."
```

---

### Task 7: 端到端验证

**Files:**
- No new files

- [ ] **Step 1: 运行完整 Phase 0+1+2 流程**

```bash
cd /Users/yimin.fu/PythonProjects/ctp
python scripts/daily_workflow.py --no-monitor 2>&1 | tee /tmp/workflow_test.log
```

检查：
1. Phase 1 输出标题为"基本面筛选"
2. Phase 1 不再显示 tech_score
3. 阈值为 10（或通过 `--threshold` 指定）
4. 极端价格品种（range_pct < 5% 或 > 95%）即使基本面分低也出现在候选中
5. Phase 2 输出包含 RRF、#P1、#P2 列
6. 报告文件 `data/reports/YYYY-MM-DD_targets.md` 格式正确

- [ ] **Step 2: 检查报告中的多晶硅**

如果多晶硅(PS0)当前 range_pct < 5%，它应该通过安全网进入 Phase 2。检查它是否出现在报告中，以及其 Spring 信号是否被正确检测（带力弹簧路径）。

```bash
grep -A5 "多晶硅\|PS0" /Users/yimin.fu/PythonProjects/ctp/data/reports/$(date +%Y-%m-%d)_targets.md
```

- [ ] **Step 3: Commit 最终验证通过的状态**

如果有任何修复，提交最终修复：

```bash
git add -A
git commit -m "fix: address integration issues from end-to-end testing"
```
