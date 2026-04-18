# Phase 2 双路径入场与 README 重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `Phase 2` 真正支持“反转入场 + 趋势延续入场”双路径并行，回测能识别两类交易，并把 README 重写为当前真实策略说明。

**Architecture:** 保留现有反转入场链路，直接在 `scripts/phase2/pre_market.py` 中新增独立的趋势延续评估和候选计划择优逻辑，避免把趋势语义再次塞回 `wyckoff.py`。回测层只消费新的 plan 字段，不重写策略；README 与 CLI 展示同步改成读取真实生产字段。

**Tech Stack:** Python 3、pandas、pytest、TqSdk、现有 `scripts/phase2` / `scripts/backtest` / `scripts/cli` 结构

---

## File Map

### Modify

- `scripts/phase2/pre_market.py`
  - 新增趋势延续候选计划生成
  - 统一反转/趋势候选计划结构
  - `build_trade_plan_from_daily_df()` 改为双路径择优
- `scripts/cli/daily_workflow.py`
  - 报告、终端输出、JSON 字段读取新 plan 字段
  - 删除对不存在的 `entry_mode` 的依赖
- `scripts/backtest/models.py`
  - 扩展 `TradePlan` / `TradeRecord` 元数据字段
- `scripts/backtest/cases.py`
  - 增加 `lh0_short` 内置 case，便于验证生猪趋势做空
- `scripts/backtest/phase23.py`
  - 回测记录 `entry_family`
  - 拆分反转单/趋势单诊断
- `scripts/backtest/rolling.py`
  - 聚合新增的双路径诊断字段
- `scripts/run_backtest.py`
  - CLI 打印双路径诊断
- `README.md`
  - 重写 `Phase 1 / Phase 2 / Phase 3 / 回测` 章节

### Test

- `tests/test_pre_market.py`
  - 新增趋势路径、双路径择优、字段桥接测试
- `tests/test_backtest_phase23.py`
  - 新增 `entry_family` 与双路径诊断测试
- `tests/test_backtest_rolling.py`
  - 新增双路径聚合诊断测试
- `tests/test_run_backtest.py`
  - 新增 CLI 输出双路径诊断测试

## Task 1: 为 Phase 2 双路径写失败测试

**Files:**
- Modify: `tests/test_pre_market.py`
- Test: `tests/test_pre_market.py`

- [ ] **Step 1: 写“趋势做多计划”失败测试**

在 `tests/test_pre_market.py` 追加下面的测试，先表达“即使没有反转信号，只要趋势延续路径成立，也应该生成可操作 plan”：

```python
def test_build_trade_plan_from_daily_df_returns_trend_long_plan_without_reversal_signal(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "上涨中",
            "next_expected": "等待回踩",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 28.0})
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [116.0, 120.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(
        pre_market,
        "_evaluate_trend_continuation_entry",
        lambda **kwargs: {
            "has_signal": True,
            "entry_family": "trend",
            "signal_type": "Pullback",
            "signal_detail": "上涨趋势中缩量回踩MA20后重新转强",
            "signal_bar": {"low": 110.0, "high": 114.0},
        },
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert bool(plan["actionable"]) is True
    assert plan["entry_family"] == "trend"
    assert plan["entry_signal_type"] == "Pullback"
    assert plan["entry_signal_detail"] == "上涨趋势中缩量回踩MA20后重新转强"
```

- [ ] **Step 2: 跑测试并确认它以缺少趋势路径而失败**

Run:

```bash
pytest tests/test_pre_market.py::test_build_trade_plan_from_daily_df_returns_trend_long_plan_without_reversal_signal -v
```

Expected:

- FAIL
- 失败点应表现为 `actionable` 仍然是 `False`，或者缺少 `entry_family`

- [ ] **Step 3: 写“趋势做空计划”失败测试**

继续追加对称的做空测试：

```python
def test_build_trade_plan_from_daily_df_returns_trend_short_plan_without_reversal_signal(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda *args, **kwargs: {
            "has_signal": False,
            "signal_type": "",
            "signal_date": "",
            "signal_bar": {"low": 0.0, "high": 0.0},
            "signal_detail": "",
            "current_stage": "下跌中",
            "next_expected": "等待反弹受阻",
            "confidence": 0.0,
            "all_events": [],
            "suspect_events": [],
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": -29.0})
    monkeypatch.setattr(pre_market, "calc_atr", lambda *args, **kwargs: pd.Series([2.0] * len(df), index=df.index))
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([104.0, 108.0], [118.0]))
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markdown"})())
    monkeypatch.setattr(
        pre_market,
        "_evaluate_trend_continuation_entry",
        lambda **kwargs: {
            "has_signal": True,
            "entry_family": "trend",
            "signal_type": "TrendBreak",
            "signal_detail": "下跌趋势中放量跌破20日低点",
            "signal_bar": {"low": 108.0, "high": 116.0},
        },
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="short",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert bool(plan["actionable"]) is True
    assert plan["entry_family"] == "trend"
    assert plan["entry_signal_type"] == "TrendBreak"
```

- [ ] **Step 4: 跑测试并确认它失败**

Run:

```bash
pytest tests/test_pre_market.py::test_build_trade_plan_from_daily_df_returns_trend_short_plan_without_reversal_signal -v
```

Expected:

- FAIL
- 当前代码仍会因为 `reversal["has_signal"] == False` 不出计划

- [ ] **Step 5: 写“双路径都成立时按盈亏比择优”失败测试**

继续追加择优测试：

```python
def test_build_trade_plan_from_daily_df_prefers_higher_rr_candidate(monkeypatch) -> None:
    df = _build_daily_frame()

    monkeypatch.setattr(
        pre_market,
        "_build_reversal_candidate",
        lambda **kwargs: {
            "entry_family": "reversal",
            "entry_signal_type": "SOS",
            "entry_signal_detail": "放量突破确认",
            "entry": 113.1,
            "stop": 109.0,
            "tp1": 116.0,
            "tp2": 120.0,
            "rr": 0.7,
            "actionable": False,
        },
    )
    monkeypatch.setattr(
        pre_market,
        "_build_trend_candidate",
        lambda **kwargs: {
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "缩量回踩后转强",
            "entry": 113.1,
            "stop": 111.0,
            "tp1": 118.0,
            "tp2": 121.0,
            "rr": 2.45,
            "actionable": True,
        },
    )
    monkeypatch.setattr(pre_market, "score_signals", lambda *args, **kwargs: {"dummy": 26.0})
    monkeypatch.setattr(pre_market, "wyckoff_phase", lambda *args, **kwargs: type("P", (), {"phase": "markup"})())
    monkeypatch.setattr(pre_market, "find_support_resistance", lambda *args, **kwargs: ([100.0], [116.0, 120.0]))

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=df,
        cfg={},
    )

    assert plan is not None
    assert plan["entry_family"] == "trend"
    assert plan["entry_signal_type"] == "Pullback"
    assert plan["rr"] == pytest.approx(2.45)
```

- [ ] **Step 6: 跑测试并确认失败**

Run:

```bash
pytest tests/test_pre_market.py::test_build_trade_plan_from_daily_df_prefers_higher_rr_candidate -v
```

Expected:

- FAIL
- 失败原因应表现为当前生产代码不存在 `_build_reversal_candidate` / `_build_trend_candidate` 之类的双路径选择

- [ ] **Step 7: 提交测试脚手架**

```bash
git add tests/test_pre_market.py
git commit -m "test: cover phase2 dual entry planning"
```

## Task 2: 在 Phase 2 落地双路径候选计划与择优逻辑

**Files:**
- Modify: `scripts/phase2/pre_market.py`
- Test: `tests/test_pre_market.py`

- [ ] **Step 1: 添加统一候选计划 helper**

在 `scripts/phase2/pre_market.py` 中，放在 `build_trade_plan_from_daily_df()` 之前，添加统一候选包装 helper：

```python
def _candidate_payload(
    *,
    entry_family: str,
    entry_signal_type: str,
    entry_signal_detail: str,
    entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    rr: float,
    actionable: bool,
) -> dict:
    return {
        "entry_family": entry_family,
        "entry_signal_type": entry_signal_type,
        "entry_signal_detail": entry_signal_detail,
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "actionable": bool(actionable),
    }
```

- [ ] **Step 2: 添加趋势延续评估函数**

在同一文件新增趋势信号评估函数，先用克制规则实现双边趋势：

```python
def _evaluate_trend_continuation_entry(*, df: pd.DataFrame, direction: str, cfg: dict) -> dict:
    close = df["close"]
    volume = df["volume"]
    ma20 = calc_ma(close, 20).iloc[-1]
    ma60 = calc_ma(close, 60).iloc[-1]
    last = float(close.iloc[-1])
    atr = float(calc_atr(df, cfg.get("atr_window", 14)).iloc[-1])
    recent = df.tail(20).copy()

    trend_up = ma20 > ma60 and recent["high"].iloc[-1] >= recent["high"].tail(5).max()
    trend_down = ma20 < ma60 and recent["low"].iloc[-1] <= recent["low"].tail(5).min()
    vol_ratio = float(volume.iloc[-1] / (volume.tail(20).mean() + 1e-12))

    if direction == "long" and trend_up:
        near_ma20 = abs(last - ma20) <= atr
        breakout = last >= recent["high"].iloc[:-1].max() and vol_ratio >= 1.3
        if near_ma20:
            return {
                "has_signal": True,
                "entry_family": "trend",
                "signal_type": "Pullback",
                "signal_detail": "上涨趋势中回踩MA20附近并保持强势",
                "signal_bar": {"low": float(df.iloc[-1]["low"]), "high": float(df.iloc[-1]["high"])},
            }
        if breakout:
            return {
                "has_signal": True,
                "entry_family": "trend",
                "signal_type": "TrendBreak",
                "signal_detail": "上涨趋势中放量突破20日高点",
                "signal_bar": {"low": float(df.iloc[-1]["low"]), "high": float(df.iloc[-1]["high"])},
            }

    if direction == "short" and trend_down:
        near_ma20 = abs(last - ma20) <= atr
        breakdown = last <= recent["low"].iloc[:-1].min() and vol_ratio >= 1.3
        if near_ma20:
            return {
                "has_signal": True,
                "entry_family": "trend",
                "signal_type": "Pullback",
                "signal_detail": "下跌趋势中反弹至MA20附近后重新转弱",
                "signal_bar": {"low": float(df.iloc[-1]["low"]), "high": float(df.iloc[-1]["high"])},
            }
        if breakdown:
            return {
                "has_signal": True,
                "entry_family": "trend",
                "signal_type": "TrendBreak",
                "signal_detail": "下跌趋势中放量跌破20日低点",
                "signal_bar": {"low": float(df.iloc[-1]["low"]), "high": float(df.iloc[-1]["high"])},
            }

    return {
        "has_signal": False,
        "entry_family": "trend",
        "signal_type": "",
        "signal_detail": "",
        "signal_bar": {"low": 0.0, "high": 0.0},
    }
```

- [ ] **Step 3: 把反转路径抽成候选计划 helper**

把现在 `if reversal["has_signal"]:` 里的计划计算，抽到独立 helper `_build_reversal_candidate(...)`，返回统一 payload：

```python
def _build_reversal_candidate(... ) -> dict | None:
    if not reversal["has_signal"]:
        return None
    ...
    return _candidate_payload(
        entry_family="reversal",
        entry_signal_type=str(reversal["signal_type"]),
        entry_signal_detail=str(reversal["signal_detail"]),
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        actionable=score_ok and rr_ok,
    )
```

- [ ] **Step 4: 新增趋势候选计划 helper**

使用趋势评估结果新增 `_build_trend_candidate(...)`：

```python
def _build_trend_candidate(... ) -> dict | None:
    trend = _evaluate_trend_continuation_entry(df=df, direction=direction, cfg=cfg)
    if not trend["has_signal"]:
        return None

    sig_bar = trend["signal_bar"]
    entry = last
    if direction == "long":
        stop = sig_bar["low"] - 0.5 * atr
        rr = (tp1 - entry) / (entry - stop + 1e-12)
    else:
        stop = sig_bar["high"] + 0.5 * atr
        rr = (entry - tp1) / (stop - entry + 1e-12)

    score_ok = (direction == "long" and total > 20) or (direction == "short" and total < -20)
    rr_ok = rr >= 1.0
    return _candidate_payload(
        entry_family="trend",
        entry_signal_type=str(trend["signal_type"]),
        entry_signal_detail=str(trend["signal_detail"]),
        entry=entry,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        rr=rr,
        actionable=score_ok and rr_ok,
    )
```

- [ ] **Step 5: 在主函数里做双路径择优**

把 `build_trade_plan_from_daily_df()` 的尾部改成：

```python
reversal_candidate = _build_reversal_candidate(...)
trend_candidate = _build_trend_candidate(...)

candidates = [
    c for c in [reversal_candidate, trend_candidate]
    if c is not None and c["actionable"]
]

selected = max(candidates, key=lambda c: c["rr"]) if candidates else None

if selected is None:
    entry_family = None
    entry_signal_type = ""
    entry_signal_detail = ""
    entry, stop, tp1, tp2, rr = last, 0.0, 0.0, 0.0, 0.0
    actionable = False
else:
    entry_family = selected["entry_family"]
    entry_signal_type = selected["entry_signal_type"]
    entry_signal_detail = selected["entry_signal_detail"]
    entry = selected["entry"]
    stop = selected["stop"]
    tp1 = selected["tp1"]
    tp2 = selected["tp2"]
    rr = selected["rr"]
    actionable = True
```

并在返回 dict 里加入：

```python
"entry_family": entry_family,
"entry_signal_type": entry_signal_type,
"entry_signal_detail": entry_signal_detail,
"reversal_status": reversal,
"trend_status": trend,
```

- [ ] **Step 6: 跑 `test_pre_market.py`，先看新增测试转绿，再看旧测试是否需要同步字段**

Run:

```bash
pytest tests/test_pre_market.py -v
```

Expected:

- 新增的三条测试 PASS
- 旧的 `reversal` 相关测试若因为字段变更失败，则在下一步一起修正

- [ ] **Step 7: 补齐旧测试期望并再次跑全量**

把旧测试中的断言补成新字段，比如：

```python
assert plan["entry_family"] == "reversal"
assert plan["entry_signal_type"] == "SOS"
assert plan["entry_signal_detail"] == "synthetic long signal"
```

Run:

```bash
pytest tests/test_pre_market.py -v
```

Expected:

- 全绿

- [ ] **Step 8: 提交 Phase 2 双路径实现**

```bash
git add scripts/phase2/pre_market.py tests/test_pre_market.py
git commit -m "feat: add dual phase2 entry paths"
```

## Task 3: 把新 plan 字段接到 CLI、报告和 JSON

**Files:**
- Modify: `scripts/cli/daily_workflow.py`
- Test: `tests/test_pre_market.py`

- [ ] **Step 1: 替换展示层对 `entry_mode` 的依赖**

在 `scripts/cli/daily_workflow.py` 中，把：

```python
mode_tag = "🔄" if rev.get("entry_mode") == "trend" else ""
```

替换成：

```python
entry_family = t.get("entry_family") or ("trend" if (t.get("trend_status") or {}).get("has_signal") else "reversal")
mode_tag = "🔄" if entry_family == "trend" else ""
```

同样替换详情页里的：

```python
mode_label = "[顺势]" if rev.get("entry_mode") == "trend" else "[反转]"
```

为：

```python
entry_family = t.get("entry_family") or "reversal"
mode_label = "[顺势]" if entry_family == "trend" else "[反转]"
```

- [ ] **Step 2: 报告里优先显示统一信号字段**

把详情页和列表页的信号展示改为优先读取：

```python
signal_type = t.get("entry_signal_type") or rev.get("signal_type", "")
signal_detail = t.get("entry_signal_detail") or rev.get("signal_detail", "")
```

并保留 `reversal_status` 作为反转细节补充，不再把它当作唯一来源。

- [ ] **Step 3: 运行最小定位测试**

Run:

```bash
pytest tests/test_pre_market.py::test_analyze_one_restores_phase2_section_logging -v
```

Expected:

- PASS
- 日志输出仍包含 `Phase 2` 关键信息，不会因新增字段崩掉

- [ ] **Step 4: 提交 CLI/报告接线**

```bash
git add scripts/cli/daily_workflow.py
git commit -m "fix: wire phase2 entry family into reports"
```

## Task 4: 让回测识别反转单和趋势单

**Files:**
- Modify: `scripts/backtest/models.py`
- Modify: `scripts/backtest/cases.py`
- Modify: `scripts/backtest/phase23.py`
- Modify: `scripts/backtest/rolling.py`
- Modify: `scripts/run_backtest.py`
- Modify: `tests/test_backtest_phase23.py`
- Modify: `tests/test_backtest_rolling.py`
- Modify: `tests/test_run_backtest.py`

- [ ] **Step 1: 先给回测补 `lh0_short` 内置 case 测试**

在 `tests/test_run_backtest.py` 追加一个最小 case 测试：

```python
def test_get_case_supports_lh0_short() -> None:
    from backtest.cases import get_case

    case = get_case("lh0_short")

    assert case.case_id == "lh0_short"
    assert case.symbol == "LH0"
    assert case.direction == "short"
```

- [ ] **Step 2: 跑测试并确认失败**

Run:

```bash
pytest tests/test_run_backtest.py::test_get_case_supports_lh0_short -v
```

Expected:

- FAIL
- 当前 `scripts/backtest/cases.py` 里还没有这个 case

- [ ] **Step 3: 添加 `lh0_short` case**

在 `scripts/backtest/cases.py` 中追加：

```python
"lh0_short": BacktestCase(
    case_id="lh0_short",
    symbol="LH0",
    name="生猪",
    direction="short",
    start_dt=date(2025, 1, 1),
    end_dt=date(2025, 12, 31),
    note="baseline short case",
),
```

- [ ] **Step 4: 跑 case 测试转绿**

Run:

```bash
pytest tests/test_run_backtest.py::test_get_case_supports_lh0_short -v
```

Expected:

- PASS

- [ ] **Step 5: 先写回测模型失败测试**

在 `tests/test_backtest_phase23.py` 增加一个映射测试：

```python
def test_make_trade_plan_from_phase2_maps_entry_family(monkeypatch):
    raw_plan = {
        "symbol": "LH0",
        "direction": "long",
        "score": 36.5,
        "actionable": True,
        "entry": 103.0,
        "stop": 99.0,
        "tp1": 106.0,
        "tp2": 109.0,
        "entry_family": "trend",
        "entry_signal_type": "Pullback",
        "entry_signal_detail": "上涨趋势回踩后转强",
        "reversal_status": {"signal_date": "2025-01-06", "signal_type": "SOS"},
    }
    ...
    assert plan.meta["entry_family"] == "trend"
    assert plan.signal_type == "Pullback"
```

- [ ] **Step 6: 跑失败测试**

Run:

```bash
pytest tests/test_backtest_phase23.py::test_make_trade_plan_from_phase2_maps_entry_family -v
```

Expected:

- FAIL
- 当前 `TradePlan` / `make_trade_plan_from_phase2()` 还没有保存这类字段

- [ ] **Step 7: 扩展 `TradePlan` / `TradeRecord` 元数据**

在 `scripts/backtest/models.py` 里保留 dataclass 结构不重构，只通过 `meta` 补充最小字段：

```python
TradePlan(
    ...,
    signal_type=str(raw.get("entry_signal_type") or reversal_status.get("signal_type") or ""),
    meta={
        "entry_family": raw.get("entry_family") or "reversal",
        "entry_signal_detail": raw.get("entry_signal_detail") or "",
    },
)
```

在 `_build_trade_record(...)` 中把 plan 的元数据桥接到成交记录：

```python
meta={
    "entry_family": plan.meta.get("entry_family", "reversal"),
    "entry_signal_detail": plan.meta.get("entry_signal_detail", ""),
}
```

- [ ] **Step 8: 拆分双路径诊断**

在 `scripts/backtest/phase23.py` 中新增诊断 key：

```python
PHASE2_ENTRY_FAMILY_KEYS = (
    "phase2_actionable_reversal_days",
    "phase2_actionable_trend_days",
    "trades_opened_reversal",
    "trades_opened_trend",
)
```

在计划生成与开仓时按 `entry_family` 计数：

```python
if plan is not None and plan.meta.get("entry_family") == "trend":
    diagnostics["phase2_actionable_trend_days"] += 1
elif plan is not None:
    diagnostics["phase2_actionable_reversal_days"] += 1

if opened_trade and plan.meta.get("entry_family") == "trend":
    diagnostics["trades_opened_trend"] += 1
elif opened_trade:
    diagnostics["trades_opened_reversal"] += 1
```

- [ ] **Step 9: 在 rolling 和 CLI 聚合/打印新字段**

在 `scripts/backtest/rolling.py` 的窗口诊断常量中加入：

```python
"phase2_actionable_reversal_days",
"phase2_actionable_trend_days",
"trades_opened_reversal",
"trades_opened_trend",
```

在 `scripts/run_backtest.py` 的打印逻辑里加入：

```python
print(f"diag_total_phase2_actionable_reversal_days={...}")
print(f"diag_total_phase2_actionable_trend_days={...}")
print(f"diag_total_trades_opened_reversal={...}")
print(f"diag_total_trades_opened_trend={...}")
```

- [ ] **Step 10: 跑回测相关测试**

Run:

```bash
pytest tests/test_backtest_phase23.py tests/test_backtest_rolling.py tests/test_run_backtest.py -v
```

Expected:

- 新增测试先转绿
- 如果旧测试因诊断字段精确比较失败，一并更新期望

- [ ] **Step 11: 跑全量回归**

Run:

```bash
pytest -q
```

Expected:

- 全绿

- [ ] **Step 12: 提交回测双路径支持**

```bash
git add scripts/backtest/cases.py scripts/backtest/models.py scripts/backtest/phase23.py scripts/backtest/rolling.py scripts/run_backtest.py tests/test_backtest_phase23.py tests/test_backtest_rolling.py tests/test_run_backtest.py
git commit -m "feat: track reversal and trend trades in backtests"
```

## Task 5: 重写 README 中的策略说明

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 重写 `Phase 1` 章节**

把 `README.md` 中 `Phase 1` 章节改成下面这个结构：

```md
### Phase 1：基本面发现机会

Phase 1 不直接输出最终做多/做空建议，只回答“这个品种值不值得进入 Phase 2 深挖”。

当前核心输出：

- 反转机会分
- 趋势机会分
- 关注优先级分
- 机会标签
- Phase1 摘要

当前主要指标：

- 库存变化与库存分位
- 仓单变化
- 季节性
- 生猪专项利润/周期逻辑
- 价格相对长期均衡的位置
```

- [ ] **Step 2: 重写 `Phase 2` 章节**

把 `README.md` 中 `Phase 2` 章节改成下面这个结构：

```md
### Phase 2：定方向与入场时机

Phase 2 分成两层：

1. 先解析方向：`long / short / watch`
2. 再在该方向上并行评估两条入场路径：
   - 反转入场
   - 趋势延续入场

方向层主要看：

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

反转入场：

- 做多：`SC / StopVol_Bull / Spring / SOS`
- 做空：`BC / StopVol_Bear / UT / SOW`

趋势延续入场：

- 做多：上涨趋势中的回踩企稳 / 放量突破
- 做空：下跌趋势中的反弹受阻 / 放量跌破

两条路径都要通过：

- 方向分数门槛
- 盈亏比门槛
```

- [ ] **Step 3: 重写 `Phase 3` 与回测章节**

把 `Phase 3` 和回测章节改成：

```md
### Phase 3：盘中监控

Phase 3 不负责日线选方向，而是在已入选目标上做盘中确认。

- 信号周期：`5m`
- 运行时钟：高频轮询当前进行中的 `5m` bar
- 终端重点打印：观望目标的新信号 / 持续信号 / 信号消失
- 详细分钟级信息写入日志

### 回测

当前最小回测只验证 `Phase 2 / Phase 3`：

- 用日线生成 `Phase 2` 计划
- 用 `1m` 推进时间
- 用当前进行中的 `5m` bar 判断盘中信号
- 支持单段回测和滚动回测

关键诊断字段：

- `diag_phase2_actionable_days`
- `diag_phase2_reject_no_signal_days`
- `diag_total_phase2_actionable_reversal_days`
- `diag_total_phase2_actionable_trend_days`
- `diag_total_trades_opened_reversal`
- `diag_total_trades_opened_trend`
```

- [ ] **Step 4: 人工检查 README 是否只描述真实逻辑**

检查点：

- 不再出现“顺势已经支持”但代码未接入的表述
- `Phase 2` 的双路径必须和实际代码一致
- 目录结构和当前仓库一致

- [ ] **Step 5: 提交 README**

```bash
git add README.md
git commit -m "docs: rewrite strategy overview for current phases"
```

## Task 6: 联网回测验证与结论记录

**Files:**
- Modify: `README.md`（仅当需要补回测示例）
- Test: 联网回测命令

- [ ] **Step 1: 先跑 `LH0` 反弹区间，验证反转做多链路**

Run:

```bash
python scripts/run_backtest.py --case lh0_long --start 2025-07-01 --end 2025-10-31 --rolling --train-days 90 --validation-days 30 --step-days 30
```

Expected:

- 命令成功返回
- 输出 `diag_total_phase2_actionable_reversal_days`
- 能判断反转做多是否开始产生计划

- [ ] **Step 2: 再跑明确下跌区间，验证趋势做空链路**

Run:

```bash
python scripts/run_backtest.py --case lh0_short --start 2025-04-01 --end 2025-06-30 --rolling --train-days 60 --validation-days 20 --step-days 20
```

Expected:

- 命令成功返回
- `diag_total_phase2_actionable_trend_days` 非零，或能明确说明趋势做空仍未触发的具体原因

- [ ] **Step 3: 记录结论**

把联网回测结论整理成这三类：

- 反转单有没有产出
- 趋势单有没有产出
- 哪类计划最终成交更多

如果需要，把一段最小示例命令补回 README 的回测章节。

- [ ] **Step 4: 最终验证**

Run:

```bash
pytest -q
python -m compileall scripts tests
```

Expected:

- `pytest` 全绿
- `compileall` 无报错

- [ ] **Step 5: 最终提交**

```bash
git add README.md scripts/phase2/pre_market.py scripts/cli/daily_workflow.py scripts/backtest/models.py scripts/backtest/phase23.py scripts/backtest/rolling.py scripts/run_backtest.py tests/test_pre_market.py tests/test_backtest_phase23.py tests/test_backtest_rolling.py tests/test_run_backtest.py
git commit -m "feat: add dual phase2 entry paths and docs"
```

## Self-Review

- Spec coverage:
  - `Phase 2` 双路径并行：Task 1-2
  - CLI/报告同步读取真实字段：Task 3
  - 回测识别反转单/趋势单：Task 4
  - README 完整重写：Task 5
  - `LH0` 反弹区间与趋势区间验证：Task 6
- Placeholder scan:
  - 无 `TODO / TBD / implement later`
  - 所有测试和命令都给了具体路径
- Type consistency:
  - 统一使用 `entry_family / entry_signal_type / entry_signal_detail`
  - 回测层通过 `meta` 桥接，不额外发明第二套命名
