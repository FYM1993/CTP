# 代码审计修复 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Phase 1/2 评分体系中的 7 个 Critical bug、7 个 Important 问题和 2 个设计清理项

**Architecture:** 分 4 个 Task 按模块修改：data_cache.py (防御性) → wyckoff.py (信号对齐) → pre_market.py (评分保护) → daily_workflow.py (约定清理+架构清理)。每个 Task 独立可验证。

**Tech Stack:** Python 3.12, pandas, numpy, akshare

**Spec:** `docs/superpowers/specs/2026-04-13-code-audit-fixes-design.md`

---

### Task 1: data_cache.py — 防御性编码 + 清理

**修复:** C5, C6, I5, I6, D2

**Files:**
- Modify: `scripts/data_cache.py:25` (import)
- Modify: `scripts/data_cache.py:270-278` (docstring)
- Modify: `scripts/data_cache.py:376-380` (GFEX 日期)
- Modify: `scripts/data_cache.py:388` (dead code)
- Modify: `scripts/data_cache.py:428-430` (hog 除零)
- Modify: `scripts/data_cache.py:483-484` (seasonality 除零)
- Modify: `scripts/data_cache.py:704` (dead code)

- [ ] **Step 1: 添加 timedelta import**

在 `scripts/data_cache.py:25`，把:
```python
from datetime import datetime
```
改为:
```python
from datetime import datetime, timedelta
```

- [ ] **Step 2: 修复 get_hog_fundamentals 除零 (C5)**

在 `scripts/data_cache.py:430`，把:
```python
        result["price_trend"] = (result["price"] / result["price_5d_ago"] - 1) * 100
```
改为:
```python
        p5 = result["price_5d_ago"]
        result["price_trend"] = (result["price"] / (p5 + 1e-12) - 1) * 100 if p5 > 0 else 0.0
```

- [ ] **Step 3: 修复 get_seasonality 除零 (C6)**

在 `scripts/data_cache.py:483-484`，把:
```python
        monthly = d.groupby(["year", "month"])["close"].agg(["first", "last"])
        monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
```
改为:
```python
        monthly = d.groupby(["year", "month"])["close"].agg(["first", "last"])
        monthly["first"] = monthly["first"].replace(0, np.nan)
        monthly["ret"] = (monthly["last"] / monthly["first"] - 1) * 100
        monthly = monthly.dropna(subset=["ret"])
```

注意: 需确认 `np` 已导入。检查文件顶部 imports — 如果没有 `import numpy as np`，需在 `import pandas as pd` 后添加。

- [ ] **Step 4: 修复 GFEX 非交易日容错 (I5)**

在 `scripts/data_cache.py:376-380`，把:
```python
        if _gfex_receipt_cache is None:
            try:
                _gfex_receipt_cache = aks.futures_gfex_warehouse_receipt(
                    date=datetime.now().strftime("%Y%m%d")
                )
            except Exception:
                _gfex_receipt_cache = {}
```
改为:
```python
        if _gfex_receipt_cache is None:
            for _offset in range(4):
                try:
                    _d = (datetime.now() - timedelta(days=_offset)).strftime("%Y%m%d")
                    _cache = aks.futures_gfex_warehouse_receipt(date=_d)
                    if _cache:
                        _gfex_receipt_cache = _cache
                        break
                except Exception:
                    continue
            if _gfex_receipt_cache is None:
                _gfex_receipt_cache = {}
```

- [ ] **Step 5: 修正 inv_percentile docstring (I6)**

在 `scripts/data_cache.py:278`，把:
```python
      inv_percentile: 库存在52周内的分位数 (0~100, 越高库存越高)
```
改为:
```python
      inv_percentile: 库存在52周高低区间的位置 (0~100, min-max归一化, 越高库存越高)
```

- [ ] **Step 6: 清理未使用变量 (D2)**

在 `scripts/data_cache.py:388`，删除:
```python
                yesterday_col = "昨日仓单量"
```

在 `scripts/data_cache.py:704`，删除:
```python
    has_today = api_calls - live_count
```

- [ ] **Step 7: 验证 Task 1**

```bash
cd /Users/yimin.fu/PythonProjects/ctp && python -c "
from scripts.data_cache import get_hog_fundamentals, get_seasonality, get_inventory
print('imports ok')
# 验证 get_inventory 仍正常
inv = get_inventory('CU0')
print(f'CU0 inventory: {inv is not None}')
print('Task 1 验证通过')
"
```
Expected: 无错误，打印 `Task 1 验证通过`。

---

### Task 2: wyckoff.py — 信号对齐

**修复:** C2, C4, I2, I7

**Files:**
- Modify: `scripts/wyckoff.py:696-697` (seen_signals → precursor_priority)
- Modify: `scripts/wyckoff.py:728-795` (precursor 记录)
- Modify: `scripts/wyckoff.py:862-936` (SOS/SOW 前序优先级)
- Modify: `scripts/wyckoff.py:959-973` (空数据保护)
- Modify: `scripts/wyckoff.py:1035-1036` (prep boost 限定)
- Modify: `scripts/wyckoff.py:1048-1112` (has_signal 由 strength 派生)

- [ ] **Step 1: assess_reversal_status 空数据保护 (C4)**

在 `scripts/wyckoff.py:968`（`events = _find_events_with_bars(...)` 之前），添加:
```python
    if len(df) < 21:
        return {
            "has_signal": False, "signal_strength": 0.0,
            "signal_type": "", "signal_date": "",
            "signal_bar": {}, "signal_detail": "",
            "current_stage": "数据不足", "next_expected": "",
            "confidence": 0.0,
            "all_events": [], "suspect_events": [],
        }
```

- [ ] **Step 2: 将 seen_signals 改为 precursor_priority (I2)**

在 `scripts/wyckoff.py:695-697`，把:
```python
    # 收集已出现的事件类型（用于 SOS/SOW 前序事件验证）
    seen_signals = set()
    suspect_signals = set()
```
改为:
```python
    # 收集已出现的事件类型及最高优先级（用于 SOS/SOW 前序事件验证）
    precursor_priority = {}  # {"SC": 3, "Spring": 2, ...}
    suspect_signals = set()
```

- [ ] **Step 3: 更新所有 seen_signals.add() 为 precursor_priority 更新**

在整个 `_find_events_with_bars` 函数中，把所有 `seen_signals.add("XXX")` 替换为 `precursor_priority["XXX"] = max(precursor_priority.get("XXX", 0), priority)`:

需要替换的行（共 12 处）:
- L734: `seen_signals.add("SC")` → `precursor_priority["SC"] = max(precursor_priority.get("SC", 0), priority)`
- L756: `seen_signals.add("BC")` → `precursor_priority["BC"] = max(precursor_priority.get("BC", 0), priority)`
- L785: `seen_signals.add("Spring")` → `precursor_priority["Spring"] = max(precursor_priority.get("Spring", 0), priority)`
- L795: `seen_signals.add("Spring")` → `precursor_priority["Spring"] = max(precursor_priority.get("Spring", 0), priority)`
- L833: `seen_signals.add("UT")` → `precursor_priority["UT"] = max(precursor_priority.get("UT", 0), priority)`
- L843: `seen_signals.add("UT")` → `precursor_priority["UT"] = max(precursor_priority.get("UT", 0), priority)`
- L878: `seen_signals.add("SOS")` → `precursor_priority["SOS"] = max(precursor_priority.get("SOS", 0), priority)`
- L886: `seen_signals.add("SOS")` → `precursor_priority["SOS"] = max(precursor_priority.get("SOS", 0), priority)`
- L916: `seen_signals.add("SOW")` → `precursor_priority["SOW"] = max(precursor_priority.get("SOW", 0), priority)`
- L924: `seen_signals.add("SOW")` → `precursor_priority["SOW"] = max(precursor_priority.get("SOW", 0), priority)`
- L947: `seen_signals.add("StopVol_Bull")` → `precursor_priority["StopVol_Bull"] = max(precursor_priority.get("StopVol_Bull", 0), priority)`
- L954: `seen_signals.add("StopVol_Bear")` → `precursor_priority["StopVol_Bear"] = max(precursor_priority.get("StopVol_Bear", 0), priority)`

- [ ] **Step 4: 更新 SOS 前序优先级判定**

在 `scripts/wyckoff.py:867-868`，把:
```python
            has_strong_precursor = bool(seen_signals & {"SC", "Spring", "StopVol_Bull"})
            has_weak_precursor = bool(suspect_signals & {"SC", "Spring", "StopVol_Bull"})
```
改为:
```python
            bullish_keys = {"SC", "Spring", "StopVol_Bull"}
            max_pre_pri = max((precursor_priority.get(k, -1) for k in bullish_keys), default=-1)
            has_strong_precursor = max_pre_pri >= 2
            has_weak_precursor = bool(suspect_signals & bullish_keys)
```

- [ ] **Step 5: 更新 SOW 前序优先级判定**

在 `scripts/wyckoff.py:905-906`，把:
```python
            has_strong_precursor = bool(seen_signals & {"BC", "UT", "StopVol_Bear"})
            has_weak_precursor = bool(suspect_signals & {"BC", "UT", "StopVol_Bear"})
```
改为:
```python
            bearish_keys = {"BC", "UT", "StopVol_Bear"}
            max_pre_pri = max((precursor_priority.get(k, -1) for k in bearish_keys), default=-1)
            has_strong_precursor = max_pre_pri >= 2
            has_weak_precursor = bool(suspect_signals & bearish_keys)
```

- [ ] **Step 6: prep boost 仅限 SOS/SOW (I7)**

在 `scripts/wyckoff.py:1035-1036`，把:
```python
        if prep_events:
            strength = min(strength + 0.1, 0.95)
```
改为:
```python
        if prep_events and fresh_entry["signal"] in ("SOS", "SOW"):
            strength = min(strength + 0.1, 0.95)
```

- [ ] **Step 7: has_signal 由 signal_strength 派生 (C2)**

在 `scripts/wyckoff.py` 的 `assess_reversal_status` 函数中，统一所有返回路径。

**路径 1 — fresh_entry（L1048-1060）:** 把:
```python
        return {
            "has_signal": True,
            "signal_strength": strength,
```
改为:
```python
        return {
            "has_signal": strength >= 0.5,
            "signal_strength": strength,
```

**路径 2 — stale entry（L1072-1078）:** 无需改动，`_no_signal` 模板中 `has_signal: False`，且 `signal_strength: 0.45` < 0.5，一致。

**路径 3 — prep only（L1089-1093）:** 无需改动，`signal_strength: 0.35` < 0.5，一致。

**路径 4 — suspect/none（L1105-1112）:** 无需改动，`signal_strength: 0.15/0.0` < 0.5，一致。

- [ ] **Step 8: 验证 Task 2**

```bash
cd /Users/yimin.fu/PythonProjects/ctp && python -c "
from scripts.wyckoff import assess_reversal_status
import pandas as pd
# 验证空 DataFrame 不崩溃
empty_df = pd.DataFrame(columns=['open','high','low','close','volume','oi'])
r = assess_reversal_status(empty_df, 'long')
assert r['has_signal'] == False
assert r['signal_strength'] == 0.0
print('空数据保护: OK')

# 验证短 DataFrame
short_df = pd.DataFrame({'open':[1]*10, 'high':[1]*10, 'low':[1]*10, 'close':[1]*10, 'volume':[1]*10, 'oi':[1]*10})
r2 = assess_reversal_status(short_df, 'long')
assert r2['has_signal'] == False
print('短数据保护: OK')
print('Task 2 验证通过')
"
```

---

### Task 3: pre_market.py — 评分保护 + 参数清理

**修复:** C1, C3, I3, I4

**Files:**
- Modify: `scripts/pre_market.py:174` (score_signals 签名)
- Modify: `scripts/pre_market.py:223-237` (MACD 长度保护)
- Modify: `scripts/pre_market.py:259-263` (动量长度保护)
- Modify: `scripts/pre_market.py:489` (调用方)
- Modify: `scripts/pre_market.py:631-637` (actionable 门控)

- [ ] **Step 1: score_signals 移除 direction 参数 (I3)**

在 `scripts/pre_market.py:174`，把:
```python
def score_signals(df: pd.DataFrame, direction: str, cfg: dict) -> dict:
```
改为:
```python
def score_signals(df: pd.DataFrame, cfg: dict) -> dict:
```

- [ ] **Step 2: 更新 score_signals 调用方**

在 `scripts/pre_market.py:489`，把:
```python
    scores = score_signals(df, direction, pre_cfg)
```
改为:
```python
    scores = score_signals(df, pre_cfg)
```

- [ ] **Step 3: MACD 长度保护 + NaN 安全 (C3, I4)**

在 `scripts/pre_market.py:223-237`，把:
```python
    # 2. MACD (10)
    mcfg = cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    if hist_now > 0 and hist_now > hist_prev:
        scores["MACD"] = 10
    elif hist_now > 0:
        scores["MACD"] = 5
    elif hist_now < 0 and hist_now < hist_prev:
        scores["MACD"] = -10
    elif hist_now < 0:
        scores["MACD"] = -5
    else:
        scores["MACD"] = 0
```
改为:
```python
    # 2. MACD (10)
    mcfg = cfg.get("macd", {})
    dif, dea, hist = calc_macd(close, mcfg.get("fast", 12), mcfg.get("slow", 26), mcfg.get("signal", 9))
    hist_now = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else 0.0
    hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 and pd.notna(hist.iloc[-2]) else 0.0
    if hist_now > 0 and hist_now > hist_prev:
        scores["MACD"] = 10
    elif hist_now > 0:
        scores["MACD"] = 5
    elif hist_now < 0 and hist_now < hist_prev:
        scores["MACD"] = -10
    elif hist_now < 0:
        scores["MACD"] = -5
    else:
        scores["MACD"] = 0
```

- [ ] **Step 4: 动量长度保护 (C3)**

在 `scripts/pre_market.py:259-263`，把:
```python
    # 5. 动量 (5)
    ret_5 = last / close.iloc[-6] - 1
    ret_20 = last / close.iloc[-21] - 1
    momentum = (ret_5 * 0.6 + ret_20 * 0.4) * 500
    scores["动量"] = np.clip(momentum, -5, 5)
```
改为:
```python
    # 5. 动量 (5)
    if len(close) >= 21:
        ret_5 = last / close.iloc[-6] - 1
        ret_20 = last / close.iloc[-21] - 1
        momentum = (ret_5 * 0.6 + ret_20 * 0.4) * 500
        scores["动量"] = float(np.clip(momentum, -5, 5))
    elif len(close) >= 6:
        ret_5 = last / close.iloc[-6] - 1
        scores["动量"] = float(np.clip(ret_5 * 500, -5, 5))
    else:
        scores["动量"] = 0
```

- [ ] **Step 5: 布林带 NaN 保护 (I4)**

在 `scripts/pre_market.py:256-257`，把:
```python
    bb_pos = (last - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-12)
    scores["布林带"] = np.clip((0.5 - bb_pos) * 20, -10, 10)
```
改为:
```python
    _upper = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else last
    _lower = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else last
    bb_pos = (last - _lower) / (_upper - _lower + 1e-12)
    scores["布林带"] = float(np.clip((0.5 - bb_pos) * 20, -10, 10))
```

- [ ] **Step 6: actionable 加入 signal_strength 门控 (C1)**

在 `scripts/pre_market.py:631-637`，把:
```python
        score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
        rr_ok = rr >= 1.0
        actionable = score_ok and rr_ok
        if not rr_ok:
            print(f"  ⚠️ 有入场信号但盈亏比({rr:.2f})不足1.0，谨慎入场")
        if not score_ok:
            print(f"  ⚠️ 有入场信号但评分({total:+.0f})未达标，谨慎入场")
```
改为:
```python
        score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
        rr_ok = rr >= 1.0
        strength_ok = reversal.get("signal_strength", 0) >= 0.5
        actionable = score_ok and rr_ok and strength_ok
        if not rr_ok:
            print(f"  ⚠️ 有入场信号但盈亏比({rr:.2f})不足1.0，谨慎入场")
        if not score_ok:
            print(f"  ⚠️ 有入场信号但评分({total:+.0f})未达标，谨慎入场")
        if not strength_ok:
            print(f"  ⚠️ 有入场信号但信号强度({reversal.get('signal_strength', 0):.2f})不足0.5，谨慎入场")
```

- [ ] **Step 7: 验证 Task 3**

```bash
cd /Users/yimin.fu/PythonProjects/ctp && python -c "
from scripts.pre_market import score_signals
import pandas as pd, numpy as np

# 构造最小测试数据(30行)
n = 30
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, n),
    'high': np.random.uniform(110, 120, n),
    'low': np.random.uniform(90, 100, n),
    'close': np.random.uniform(100, 110, n),
    'volume': np.random.uniform(1000, 2000, n),
    'oi': np.random.uniform(5000, 6000, n),
})
# 验证新签名(无 direction)
scores = score_signals(df, cfg={})
total = sum(scores.values())
assert np.isfinite(total), f'score is not finite: {total}'
print(f'score_signals(30行): total={total:.1f}, NaN检查通过')

# 验证短数据不崩溃
short = df.head(5)
scores2 = score_signals(short, cfg={})
total2 = sum(scores2.values())
assert np.isfinite(total2), f'short data score is not finite: {total2}'
print(f'score_signals(5行): total={total2:.1f}, 短数据保护通过')
print('Task 3 验证通过')
"
```

---

### Task 4: daily_workflow.py — 方向约定清理 + 架构清理

**修复:** C7, I1, D1

**Files:**
- Modify: `scripts/daily_workflow.py:188-198` (移除技术指标)
- Modify: `scripts/daily_workflow.py:206-297` (方向约定反转)
- Modify: `scripts/daily_workflow.py:299-313` (return dict 清理)
- Modify: `scripts/daily_workflow.py:364-365` (fund_rsi 移除)

- [ ] **Step 1: 移除 RSI/MA/returns 计算 (I1)**

在 `scripts/daily_workflow.py:188-198`，删除整个块:
```python
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rsi = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-12)))

        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma60 = float(close.rolling(min(60, len(close))).mean().iloc[-1])
        ma_trend = (ma5 / ma60 - 1) * 100

        ret_5d = (last / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0
        ret_20d = (last / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
```

- [ ] **Step 2: 反转库存变化评分符号**

在 `scripts/daily_workflow.py:219-224`，把:
```python
            if inv_4wk > 10: fund_score += 10
            elif inv_4wk > 3: fund_score += 5
            elif inv_4wk < -10: fund_score -= 10
            elif inv_4wk < -3: fund_score -= 5
            if cum >= 6: fund_score += 10
            elif cum <= 2: fund_score -= 10
```
改为:
```python
            if inv_4wk < -10: fund_score += 10
            elif inv_4wk < -3: fund_score += 5
            elif inv_4wk > 10: fund_score -= 10
            elif inv_4wk > 3: fund_score -= 5
            if cum <= 2: fund_score += 10
            elif cum >= 6: fund_score -= 10
```

- [ ] **Step 3: 反转库存分位评分符号**

在 `scripts/daily_workflow.py:230-239`，把:
```python
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
```
改为:
```python
            if inv_pct < 15:
                fund_score += 10
                fund_details.append(f"库存低位{inv_pct:.0f}%")
            elif inv_pct < 30:
                fund_score += 5
            elif inv_pct > 85:
                fund_score -= 10
                fund_details.append(f"库存高位{inv_pct:.0f}%")
            elif inv_pct > 70:
                fund_score -= 5
```

- [ ] **Step 4: 反转仓单评分符号**

在 `scripts/daily_workflow.py:247-260`，把:
```python
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
```
改为:
```python
            if rc < 0 and rt > 0:
                change_ratio = abs(rc) / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score += 15
                    fund_details.append(f"仓单大减{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score += 8
            elif rc > 0 and rt > 0:
                change_ratio = rc / (rt + 1e-12)
                if change_ratio > 0.05:
                    fund_score -= 15
                    fund_details.append(f"仓单大增{rc:+.0f}")
                elif change_ratio > 0.01:
                    fund_score -= 8
```

- [ ] **Step 5: 季节性保持不变（已是正=偏多）**

季节性块（L262-272）**不做任何改动**。当前 `fund_score += sig*10` 已经是正=偏多约定，在移除 `-fund_score` 取反后自然正确。

- [ ] **Step 6: 反转生猪评分符号**

在 `scripts/daily_workflow.py:280-295`，把:
```python
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
```
改为:
```python
                if pm < -15:
                    fund_score += 15
                    fund_details.append(f"养殖亏损{pm:.0f}%")
                elif pm < -5:
                    fund_score += 8
                elif pm > 20:
                    fund_score -= 10
                elif pm > 10:
                    fund_score -= 5

            if "price_trend" in hog:
                pt = hog["price_trend"]
                if pt < -3:
                    fund_score += 5
                elif pt > 3:
                    fund_score -= 5
```

- [ ] **Step 7: 移除 score 取反 + 清理 return dict**

在 `scripts/daily_workflow.py:297`，把:
```python
        score = -fund_score
```
改为:
```python
        score = fund_score
```

在 return dict（L299-313）中:
- 把 `"fund_score": -fund_score,` 改为 `"fund_score": fund_score,`
- 移除 `"rsi": rsi, "ma_trend": ma_trend,`
- 移除 `"ret_5d": ret_5d, "ret_20d": ret_20d,`
- 移除 `"wyckoff_phase": "",`

即把 return dict 改为:
```python
        return {
            "symbol": sym, "name": name, "exchange": exchange,
            "price": last, "range_pct": range_pct,
            "score": score,
            "fund_score": fund_score,
            "fund_details": ", ".join(fund_details) if fund_details else "",
            "inv_change_4wk": inv_4wk,
            "inv_percentile": inv_pct,
            "receipt_change": receipt_change,
            "seasonal_signal": seasonal_sig,
            "hog_profit": hog_profit,
        }
```

- [ ] **Step 8: 清理 Phase 2 中的 fund_rsi 引用**

在 `scripts/daily_workflow.py:365`，删除:
```python
                result["fund_rsi"] = cand.get("rsi", 50)
```

检查 `save_targets` 函数中是否有 `fund_rsi` 引用并删除。

- [ ] **Step 9: 验证 Task 4 — 方向正确性**

```bash
cd /Users/yimin.fu/PythonProjects/ctp && python -c "
from scripts.daily_workflow import _score_symbol
import pandas as pd, numpy as np

# 构造模拟数据
n = 100
df = pd.DataFrame({
    'open': np.linspace(100, 110, n),
    'high': np.linspace(105, 115, n),
    'low': np.linspace(95, 105, n),
    'close': np.linspace(100, 110, n),
    'volume': np.ones(n) * 1000,
    'oi': np.ones(n) * 5000,
})
r = _score_symbol('TEST', 'test', 'TEST', df)
print(f'score={r[\"score\"]}, fund_score={r[\"fund_score\"]}')
assert r['score'] == r['fund_score'], 'score should equal fund_score now'
assert 'rsi' not in r, 'rsi should be removed'
assert 'ma_trend' not in r, 'ma_trend should be removed'
print('字段清理验证通过')
print('Task 4 验证通过')
"
```

- [ ] **Step 10: 端到端集成验证**

```bash
cd /Users/yimin.fu/PythonProjects/ctp && python -c "
# 验证完整导入链
from scripts.daily_workflow import phase_1_screen, phase_2_premarket
from scripts.pre_market import score_signals, analyze_one
from scripts.wyckoff import assess_reversal_status, _find_events_with_bars
from scripts.data_cache import get_inventory, get_seasonality, get_hog_fundamentals
print('全部模块导入成功，无语法错误')
"
```
Expected: `全部模块导入成功，无语法错误`

---

## 验证清单

| 修复ID | 描述 | 验证方式 |
|--------|------|---------|
| C1 | actionable 加 signal_strength 门控 | Task 3 Step 6 |
| C2 | has_signal 由 strength 派生 | Task 2 Step 7 |
| C3 | 短数据 IndexError | Task 3 Steps 3-5 |
| C4 | 空 DataFrame 崩溃 | Task 2 Step 1 |
| C5 | 生猪除零 | Task 1 Step 2 |
| C6 | 季节性除零 | Task 1 Step 3 |
| C7 | 季节性方向反转 | Task 4 Step 5 (不改) + Step 7 (移除取反) |
| I1 | 移除 Phase 1 技术指标 | Task 4 Step 1 |
| I2 | SOS 前序优先级 | Task 2 Steps 2-5 |
| I3 | direction 参数移除 | Task 3 Steps 1-2 |
| I4 | NaN 保护 | Task 3 Steps 3-5 |
| I5 | GFEX 非交易日 | Task 1 Step 4 |
| I6 | inv_percentile docstring | Task 1 Step 5 |
| I7 | prep boost 限定 | Task 2 Step 6 |
| D1 | 方向约定清理 | Task 4 Steps 2-7 |
| D2 | 死代码清理 | Task 1 Step 6 |
