# Phase 2/3 最小回测框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前项目新增一个基于 `TqBacktest` 的最小回测框架，先用 `LH0 long` 和 `PS0 short` 两个 case 验证 `Phase 2/3` 的执行层是否有统计优势。

**Architecture:** 回测框架分成四层：case 注册层、纯数据模型层、纯回测引擎层、`TqBacktest` 数据接入层。核心回测逻辑优先做成“吃历史 DataFrame 的纯函数/纯状态机”，这样单元测试不依赖网络和天勤环境；`TqBacktest` 只负责把历史 `1m` 和日线喂给引擎，保持实现贴近当前系统但仍可测试。

**Tech Stack:** Python 3.10+, `pytest`, `pandas`, `dataclasses`, `tqsdk.TqBacktest`, 现有 `scripts/pre_market.py`, `scripts/intraday.py`, `scripts/market_data_tq.py`

---

## File Map

- Create: `scripts/backtest_models.py`
  - 定义 `BacktestCase`、`TradePlan`、`TradeRecord`、`BacktestResult`
- Create: `scripts/backtest_cases.py`
  - 内置最小 case 注册表，首批覆盖 `LH0 long` / `PS0 short`
- Create: `scripts/backtest_metrics.py`
  - 汇总逐笔交易，生成胜率、总收益、回撤、`tp1/tp2/stop` 命中统计
- Create: `scripts/backtest_phase23.py`
  - 纯回测引擎、`1m -> 当前 5m` 聚合、持仓状态机、`TqBacktest` 数据装载
- Create: `scripts/run_backtest.py`
  - 命令行入口，运行单个或多个 case
- Modify: `scripts/pre_market.py`
  - 提取一个纯函数，接收日线 DataFrame，返回 `Phase 2` 交易计划，避免回测时重复抓远程数据和写大段日志
- Modify: `scripts/tqsdk_live.py`
  - 如有必要，抽一个 `period` 聚合辅助函数共用；若不需要则不动
- Create: `tests/test_backtest_models.py`
- Create: `tests/test_backtest_cases.py`
- Create: `tests/test_backtest_metrics.py`
- Create: `tests/test_backtest_phase23.py`
- Create: `tests/test_run_backtest.py`
- Modify: `README.md`
  - 增加最小回测命令和结果说明

## Task 1: 建立回测模型和 case 注册表

**Files:**
- Create: `scripts/backtest_models.py`
- Create: `scripts/backtest_cases.py`
- Test: `tests/test_backtest_models.py`
- Test: `tests/test_backtest_cases.py`

- [ ] **Step 1: Write the failing tests for models and registry**

```python
# tests/test_backtest_models.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_models import BacktestCase, TradePlan, TradeRecord  # noqa: E402


def test_backtest_case_keeps_symbol_direction_and_dates():
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 3, 31),
        note="baseline long case",
    )

    assert case.case_id == "lh0_long"
    assert case.symbol == "LH0"
    assert case.direction == "long"
    assert case.start_dt.isoformat() == "2025-01-01"
    assert case.end_dt.isoformat() == "2025-03-31"


def test_trade_plan_has_fixed_targets_once_created():
    plan = TradePlan(
        trade_id="lh0_long_2025-01-06",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=15000.0,
        stop=14500.0,
        tp1=15800.0,
        tp2=16300.0,
        phase2_score=34.5,
        signal_type="SOS",
    )

    assert plan.entry_ref == 15000.0
    assert plan.stop == 14500.0
    assert plan.tp1 == 15800.0
    assert plan.tp2 == 16300.0
    assert plan.signal_type == "SOS"


def test_trade_record_supports_exit_reason_and_tp1_flag():
    trade = TradeRecord(
        trade_id="lh0_long_2025-01-06",
        symbol="LH0",
        direction="long",
        entry_time="2025-01-06 10:31:00",
        entry_price=15020.0,
        exit_time="2025-01-08 09:01:00",
        exit_price=16280.0,
        exit_reason="tp2",
        bars_held=315,
        days_held=2,
        tp1_hit=True,
        pnl_ratio=0.0839,
    )

    assert trade.exit_reason == "tp2"
    assert trade.tp1_hit is True
    assert trade.days_held == 2
```

```python
# tests/test_backtest_cases.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_cases import BUILTIN_CASES, get_case  # noqa: E402


def test_builtin_cases_cover_long_and_short_baselines():
    assert "lh0_long" in BUILTIN_CASES
    assert "ps0_short" in BUILTIN_CASES
    assert BUILTIN_CASES["lh0_long"].direction == "long"
    assert BUILTIN_CASES["ps0_short"].direction == "short"


def test_get_case_returns_registered_case():
    case = get_case("lh0_long")

    assert case.symbol == "LH0"
    assert case.direction == "long"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_backtest_models.py tests/test_backtest_cases.py -v
```

Expected:

- `ModuleNotFoundError: No module named 'backtest_models'`
- `ModuleNotFoundError: No module named 'backtest_cases'`

- [ ] **Step 3: Write the minimal model and registry implementation**

```python
# scripts/backtest_models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class BacktestCase:
    case_id: str
    symbol: str
    name: str
    direction: str
    start_dt: date
    end_dt: date
    note: str = ""


@dataclass(frozen=True)
class TradePlan:
    trade_id: str
    symbol: str
    direction: str
    plan_date: str
    entry_ref: float
    stop: float
    tp1: float
    tp2: float
    phase2_score: float
    signal_type: str
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    bars_held: int
    days_held: int
    tp1_hit: bool
    pnl_ratio: float
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResult:
    case_id: str
    trades: list[TradeRecord]
    summary: dict[str, float | int]
```

```python
# scripts/backtest_cases.py
from __future__ import annotations

from datetime import date

from backtest_models import BacktestCase


BUILTIN_CASES: dict[str, BacktestCase] = {
    "lh0_long": BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        note="baseline long case for phase2/3",
    ),
    "ps0_short": BacktestCase(
        case_id="ps0_short",
        symbol="PS0",
        name="多晶硅",
        direction="short",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        note="baseline short case for phase2/3",
    ),
}


def get_case(case_id: str) -> BacktestCase:
    return BUILTIN_CASES[case_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_backtest_models.py tests/test_backtest_cases.py -v
```

Expected:

- `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_models.py scripts/backtest_cases.py tests/test_backtest_models.py tests/test_backtest_cases.py
git commit -m "feat: add phase23 backtest models and cases"
```

## Task 2: 提取可复用的 Phase 2 纯计划生成函数

**Files:**
- Modify: `scripts/pre_market.py`
- Test: `tests/test_pre_market.py`

- [ ] **Step 1: Write the failing tests for pure phase2 planning**

```python
# tests/test_pre_market.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_market  # noqa: E402


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=130, freq="D"),
            "open": [100 + i * 0.1 for i in range(130)],
            "high": [101 + i * 0.1 for i in range(130)],
            "low": [99 + i * 0.1 for i in range(130)],
            "close": [100 + i * 0.15 for i in range(130)],
            "volume": [1000 + i for i in range(130)],
            "oi": [5000 + i * 3 for i in range(130)],
        }
    )


def test_build_trade_plan_from_daily_df_returns_fixed_plan_fields(monkeypatch):
    monkeypatch.setattr(
        pre_market,
        "assess_reversal_status",
        lambda df, direction, lookback=60: {
            "has_signal": True,
            "signal_type": "SOS",
            "signal_date": "2025-05-10",
            "signal_detail": "mock signal",
            "signal_bar": {"low": 115.0, "high": 125.0},
            "confidence": 0.75,
            "current_stage": "上涨确认",
            "next_expected": "继续观察",
            "all_events": [],
            "suspect_events": [],
        },
    )

    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=_daily_frame(),
        cfg={"reason": "test reason"},
    )

    assert plan is not None
    assert plan["direction"] == "long"
    assert plan["entry"] > 0
    assert plan["stop"] > 0
    assert plan["tp1"] > plan["entry"]
    assert plan["tp2"] >= plan["tp1"]


def test_build_trade_plan_from_daily_df_returns_none_when_df_empty():
    plan = pre_market.build_trade_plan_from_daily_df(
        symbol="LH0",
        name="生猪",
        direction="long",
        df=pd.DataFrame(),
        cfg={},
    )

    assert plan is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_pre_market.py -v
```

Expected:

- `AttributeError: module 'pre_market' has no attribute 'build_trade_plan_from_daily_df'`

- [ ] **Step 3: Extract a pure planning helper from `pre_market.py`**

```python
# scripts/pre_market.py
def build_trade_plan_from_daily_df(
    *,
    symbol: str,
    name: str,
    direction: str,
    df: pd.DataFrame,
    cfg: dict,
) -> dict | None:
    if df is None or df.empty:
        return None

    close = df["close"]
    last = float(close.iloc[-1])
    atr = float(calc_atr(df, cfg.get("atr_window", 14)).iloc[-1])
    supports, resistances = find_support_resistance(df, lookback=cfg.get("sr_lookback", 120))
    fib_high = float(df.tail(120)["high"].max())
    fib_low = float(df.tail(120)["low"].min())
    fib_range = fib_high - fib_low
    scores = score_signals(df, direction, cfg)
    total = float(sum(scores.values()))
    reversal = assess_reversal_status(df, direction, lookback=60)

    if not reversal["has_signal"]:
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "score": total,
            "actionable": False,
            "price": last,
            "entry": last,
            "stop": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr": 0.0,
            "reversal_status": reversal,
        }

    sig_bar = reversal["signal_bar"]
    entry = last
    if direction == "long":
        stop = float(sig_bar["low"]) - 0.5 * atr
        tp1 = min(v for v in [next((r for r in resistances if r > entry + 0.5 * atr), None), fib_low + fib_range * 0.618] if v is not None)
        tp2 = min(v for v in [next((r for r in resistances if r > tp1 + 0.5 * atr), None), fib_low + fib_range * 1.0] if v is not None)
        rr = (tp1 - entry) / (entry - stop + 1e-12)
        score_ok = total > 20
    else:
        stop = float(sig_bar["high"]) + 0.5 * atr
        tp1 = max(v for v in [next((s for s in reversed(supports) if s < entry - 0.5 * atr), None), fib_high - fib_range * 0.618] if v is not None)
        tp2 = max(v for v in [next((s for s in reversed(supports) if s < tp1 - 0.5 * atr), None), fib_high - fib_range * 1.0] if v is not None)
        rr = (entry - tp1) / (stop - entry + 1e-12)
        score_ok = total < -20

    return {
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "score": total,
        "actionable": bool(score_ok and rr >= 1.0 and reversal["has_signal"]),
        "price": last,
        "entry": float(entry),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "reversal_status": reversal,
    }


def analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict | None:
    ...
    df = fetch_data(symbol, days=400)
    if df.empty:
        log.info("  ❌ 数据获取失败")
        return None
    result = build_trade_plan_from_daily_df(symbol=symbol, name=name, direction=direction, df=df, cfg=cfg)
    if result is None:
        return None
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_pre_market.py -v
```

Expected:

- `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/pre_market.py tests/test_pre_market.py
git commit -m "refactor: extract pure phase2 trade plan helper"
```

## Task 3: 实现纯回测引擎和 `1m -> 当前 5m` 聚合

**Files:**
- Create: `scripts/backtest_phase23.py`
- Test: `tests/test_backtest_phase23.py`

- [ ] **Step 1: Write the failing tests for intrabar aggregation and state transitions**

```python
# tests/test_backtest_phase23.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_models import BacktestCase, TradePlan  # noqa: E402
from backtest_phase23 import aggregate_partial_5m_bars, run_case_from_frames  # noqa: E402


def test_aggregate_partial_5m_bars_keeps_incomplete_current_bar():
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 10:01:00",
                    "2025-01-06 10:02:00",
                    "2025-01-06 10:03:00",
                ]
            ),
            "open": [100, 101, 102],
            "high": [101, 103, 104],
            "low": [99, 100, 101],
            "close": [100.5, 102.5, 103.5],
            "volume": [10, 20, 30],
        }
    )

    bars = aggregate_partial_5m_bars(minute_df)

    assert len(bars) == 1
    assert float(bars.iloc[-1]["open"]) == 100
    assert float(bars.iloc[-1]["high"]) == 104
    assert float(bars.iloc[-1]["low"]) == 99
    assert float(bars.iloc[-1]["close"]) == 103.5
    assert float(bars.iloc[-1]["volume"]) == 60


def test_run_case_from_frames_opens_and_closes_long_trade(monkeypatch):
    daily_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="D"),
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
            "volume": [1000, 1000, 1000, 1000],
            "oi": [5000, 5001, 5002, 5003],
        }
    )
    minute_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-06 09:31:00",
                    "2025-01-06 09:32:00",
                    "2025-01-06 09:33:00",
                    "2025-01-06 09:34:00",
                    "2025-01-06 09:35:00",
                    "2025-01-06 09:36:00",
                ]
            ),
            "open": [100, 101, 102, 103, 104, 110],
            "high": [101, 102, 103, 104, 110, 112],
            "low": [99, 100, 101, 102, 103, 109],
            "close": [100, 101, 102, 103, 110, 112],
            "volume": [10, 10, 10, 10, 10, 10],
        }
    )

    monkeypatch.setattr(
        "backtest_phase23.generate_signals",
        lambda df, direction, cfg: [{"type": "开多", "strength": "强", "reason": "mock", "entry": 110.0, "stop": 95.0, "target": 112.0}],
    )

    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=daily_df["date"].iloc[0].date(),
        end_dt=daily_df["date"].iloc[-1].date(),
    )
    plan = TradePlan(
        trade_id="lh0_long_2025-01-06",
        symbol="LH0",
        direction="long",
        plan_date="2025-01-06",
        entry_ref=103.0,
        stop=95.0,
        tp1=108.0,
        tp2=111.0,
        phase2_score=35.0,
        signal_type="SOS",
    )

    result = run_case_from_frames(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda *_args, **_kwargs: plan,
        pre_market_cfg={},
        signal_cfg={},
    )

    assert len(result.trades) == 1
    assert result.trades[0].direction == "long"
    assert result.trades[0].exit_reason == "tp2"
    assert result.trades[0].tp1_hit is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_backtest_phase23.py -v
```

Expected:

- `ModuleNotFoundError: No module named 'backtest_phase23'`

- [ ] **Step 3: Implement the pure engine**

```python
# scripts/backtest_phase23.py
from __future__ import annotations

from dataclasses import replace
import pandas as pd

from backtest_models import BacktestCase, BacktestResult, TradePlan, TradeRecord
from intraday import generate_signals
from pre_market import build_trade_plan_from_daily_df


def aggregate_partial_5m_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return minute_df.copy()
    df = minute_df.copy()
    df["bucket"] = pd.to_datetime(df["datetime"]).dt.floor("5min")
    grouped = df.groupby("bucket", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    grouped = grouped.rename(columns={"bucket": "datetime"})
    return grouped


def make_trade_plan_from_phase2(*, case: BacktestCase, daily_df: pd.DataFrame, pre_market_cfg: dict) -> TradePlan | None:
    raw = build_trade_plan_from_daily_df(
        symbol=case.symbol,
        name=case.name,
        direction=case.direction,
        df=daily_df,
        cfg=pre_market_cfg,
    )
    if raw is None or not raw.get("actionable"):
        return None
    return TradePlan(
        trade_id=f"{case.case_id}_{raw['reversal_status'].get('signal_date', raw['symbol'])}",
        symbol=case.symbol,
        direction=case.direction,
        plan_date=str(raw["reversal_status"].get("signal_date", "")),
        entry_ref=float(raw["entry"]),
        stop=float(raw["stop"]),
        tp1=float(raw["tp1"]),
        tp2=float(raw["tp2"]),
        phase2_score=float(raw["score"]),
        signal_type=str(raw["reversal_status"].get("signal_type", "")),
    )


def run_case_from_frames(
    *,
    case: BacktestCase,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    plan_factory,
    pre_market_cfg: dict,
    signal_cfg: dict,
) -> BacktestResult:
    trades: list[TradeRecord] = []
    active_plan: TradePlan | None = None
    position_open = False
    entry_time = None
    entry_price = None
    tp1_hit = False
    bars_held = 0
    days_held = 0

    df = minute_df.copy()
    df["trade_date"] = pd.to_datetime(df["datetime"]).dt.date
    daily_dates = pd.to_datetime(daily_df["date"]).dt.date

    for trade_date, day_minutes in df.groupby("trade_date", sort=True):
        visible_daily = daily_df.loc[daily_dates < trade_date].copy()

        if position_open:
            refreshed_plan = plan_factory(case=case, daily_df=visible_daily, pre_market_cfg=pre_market_cfg)
            if refreshed_plan is None:
                first_bar = day_minutes.iloc[0]
                exit_price = float(first_bar["open"])
                trades.append(
                    TradeRecord(
                        trade_id=active_plan.trade_id,
                        symbol=case.symbol,
                        direction=case.direction,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=str(first_bar["datetime"]),
                        exit_price=exit_price,
                        exit_reason="phase2_invalidated",
                        bars_held=bars_held,
                        days_held=days_held,
                        tp1_hit=tp1_hit,
                        pnl_ratio=((exit_price - entry_price) / entry_price) if case.direction == "long" else ((entry_price - exit_price) / entry_price),
                    )
                )
                return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": len(trades)})
        elif active_plan is None:
            active_plan = plan_factory(case=case, daily_df=visible_daily, pre_market_cfg=pre_market_cfg)
            if active_plan is None:
                continue

        for idx in range(len(day_minutes)):
            current_slice = day_minutes.iloc[: idx + 1].copy()
            partial_5m = aggregate_partial_5m_bars(current_slice)
            latest_1m = current_slice.iloc[-1]

            if not position_open:
                signals = generate_signals(partial_5m, case.direction, signal_cfg)
                open_type = "开多" if case.direction == "long" else "开空"
                if any(sig["type"] == open_type for sig in signals):
                    position_open = True
                    entry_time = str(latest_1m["datetime"])
                    entry_price = float(latest_1m["close"])
                    tp1_hit = False
                    bars_held = 0
                    days_held = 0
                continue

            bars_held += 1
            last_price = float(latest_1m["close"])
            if case.direction == "long":
                if last_price >= active_plan.tp1:
                    tp1_hit = True
                if last_price <= active_plan.stop or last_price >= active_plan.tp2:
                    exit_reason = "stop" if last_price <= active_plan.stop else "tp2"
                    trades.append(
                        TradeRecord(
                            trade_id=active_plan.trade_id,
                            symbol=case.symbol,
                            direction=case.direction,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            exit_time=str(latest_1m["datetime"]),
                            exit_price=last_price,
                            exit_reason=exit_reason,
                            bars_held=bars_held,
                            days_held=days_held,
                            tp1_hit=tp1_hit or last_price >= active_plan.tp1,
                            pnl_ratio=(last_price - entry_price) / entry_price,
                        )
                    )
                    return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": len(trades)})
            else:
                if last_price <= active_plan.tp1:
                    tp1_hit = True
                if last_price >= active_plan.stop or last_price <= active_plan.tp2:
                    exit_reason = "stop" if last_price >= active_plan.stop else "tp2"
                    trades.append(
                        TradeRecord(
                            trade_id=active_plan.trade_id,
                            symbol=case.symbol,
                            direction=case.direction,
                            entry_time=entry_time,
                            entry_price=entry_price,
                            exit_time=str(latest_1m["datetime"]),
                            exit_price=last_price,
                            exit_reason=exit_reason,
                            bars_held=bars_held,
                            days_held=days_held,
                            tp1_hit=tp1_hit or last_price <= active_plan.tp1,
                            pnl_ratio=(entry_price - last_price) / entry_price,
                        )
                    )
                    return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": len(trades)})

        if position_open:
            days_held += 1

    return BacktestResult(case_id=case.case_id, trades=trades, summary={"num_trades": len(trades)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_backtest_phase23.py -v
```

Expected:

- `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_phase23.py tests/test_backtest_phase23.py
git commit -m "feat: add pure phase23 backtest engine"
```

## Task 4: 添加指标统计和结果汇总

**Files:**
- Create: `scripts/backtest_metrics.py`
- Test: `tests/test_backtest_metrics.py`

- [ ] **Step 1: Write the failing metrics tests**

```python
# tests/test_backtest_metrics.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_metrics import summarize_trades  # noqa: E402
from backtest_models import TradeRecord  # noqa: E402


def test_summarize_trades_counts_win_rate_and_tp_hits():
    trades = [
        TradeRecord(
            trade_id="t1",
            symbol="LH0",
            direction="long",
            entry_time="2025-01-01 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-02 10:00:00",
            exit_price=110.0,
            exit_reason="tp2",
            bars_held=30,
            days_held=1,
            tp1_hit=True,
            pnl_ratio=0.10,
        ),
        TradeRecord(
            trade_id="t2",
            symbol="PS0",
            direction="short",
            entry_time="2025-01-03 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-04 10:00:00",
            exit_price=103.0,
            exit_reason="stop",
            bars_held=20,
            days_held=1,
            tp1_hit=False,
            pnl_ratio=-0.03,
        ),
    ]

    summary = summarize_trades(trades)

    assert summary["num_trades"] == 2
    assert summary["wins"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["tp1_hits"] == 1
    assert summary["tp2_hits"] == 1
    assert summary["stop_hits"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_backtest_metrics.py -v
```

Expected:

- `ModuleNotFoundError: No module named 'backtest_metrics'`

- [ ] **Step 3: Implement the metrics helper**

```python
# scripts/backtest_metrics.py
from __future__ import annotations

from backtest_models import TradeRecord


def summarize_trades(trades: list[TradeRecord]) -> dict[str, float | int]:
    if not trades:
        return {
            "num_trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "stop_hits": 0,
        }

    pnls = [float(t.pnl_ratio) for t in trades]
    wins = sum(1 for v in pnls if v > 0)
    return {
        "num_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "avg_pnl": sum(pnls) / len(pnls),
        "total_pnl": sum(pnls),
        "tp1_hits": sum(1 for t in trades if t.tp1_hit),
        "tp2_hits": sum(1 for t in trades if t.exit_reason == "tp2"),
        "stop_hits": sum(1 for t in trades if t.exit_reason == "stop"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_backtest_metrics.py -v
```

Expected:

- `1 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_metrics.py tests/test_backtest_metrics.py
git commit -m "feat: add phase23 backtest metrics"
```

## Task 5: 接入 `TqBacktest` 数据装载层

**Files:**
- Modify: `scripts/backtest_phase23.py`
- Test: `tests/test_backtest_phase23.py`

- [ ] **Step 1: Write the failing loader test**

```python
def test_load_case_frames_with_tqbacktest_uses_case_range(monkeypatch):
    from datetime import date
    from backtest_models import BacktestCase
    import backtest_phase23

    calls = {}

    class FakeApi:
        def __init__(self, *args, **kwargs):
            calls["api_kwargs"] = kwargs
            self.daily = None
            self.minute = None

        def get_kline_serial(self, symbol, duration_seconds, data_length=None):
            calls.setdefault("serial_calls", []).append((symbol, duration_seconds, data_length))
            import pandas as pd
            if duration_seconds == 86400:
                return pd.DataFrame(
                    {"datetime": pd.to_datetime(["2025-01-01"]), "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1], "close_oi": [1]}
                )
            return pd.DataFrame(
                {"datetime": pd.to_datetime(["2025-01-01 09:31:00"]), "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1], "close_oi": [1]}
            )

        def wait_update(self, *args, **kwargs):
            return False

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(backtest_phase23, "_create_backtest_api", lambda *args, **kwargs: FakeApi(*args, **kwargs))
    monkeypatch.setattr(backtest_phase23, "resolve_case_tq_symbol", lambda case, config: "KQ.m@DCE.lh")

    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
    )

    daily_df, minute_df = backtest_phase23.load_case_frames_with_tqbacktest(case=case, config={"tqsdk": {"account": "acct", "password": "pwd"}})

    assert not daily_df.empty
    assert not minute_df.empty
    assert calls["serial_calls"][0][0] == "KQ.m@DCE.lh"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_backtest_phase23.py::test_load_case_frames_with_tqbacktest_uses_case_range -v
```

Expected:

- `AttributeError` for missing loader functions

- [ ] **Step 3: Implement the loader with a thin TqBacktest seam**

```python
# scripts/backtest_phase23.py
from datetime import datetime
from tqsdk import TqApi, TqAuth, TqBacktest


def _create_backtest_api(*, account: str, password: str, start_dt, end_dt):
    return TqApi(
        backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt),
        auth=TqAuth(account, password),
    )


def resolve_case_tq_symbol(case: BacktestCase, config: dict) -> str:
    from data_cache import _exchange_for_symbol
    from market_data_tq import symbol_to_tq_main

    exchange = _exchange_for_symbol(case.symbol)
    if exchange is None:
        raise ValueError(f"未知交易所: {case.symbol}")
    return symbol_to_tq_main(case.symbol, exchange)


def load_case_frames_with_tqbacktest(*, case: BacktestCase, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tq_cfg = config.get("tqsdk") or {}
    api = _create_backtest_api(
        account=str(tq_cfg["account"]).strip(),
        password=str(tq_cfg["password"]).strip(),
        start_dt=case.start_dt,
        end_dt=case.end_dt,
    )
    try:
        tq_symbol = resolve_case_tq_symbol(case, config)
        daily_klines = api.get_kline_serial(tq_symbol, 86400, data_length=4000)
        minute_klines = api.get_kline_serial(tq_symbol, 60, data_length=10000)
        api.wait_update()
        from market_data_tq import klines_to_daily_frame
        daily_df = klines_to_daily_frame(daily_klines)
        minute_df = minute_klines.rename(columns={"close_oi": "oi"}).copy()
        minute_df["datetime"] = pd.to_datetime(minute_df["datetime"])
        return daily_df, minute_df[["datetime", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    finally:
        api.close()
```

- [ ] **Step 4: Run focused and full engine tests**

Run:

```bash
pytest tests/test_backtest_phase23.py -v
```

Expected:

- Existing aggregation/state tests still pass
- Loader test passes

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_phase23.py tests/test_backtest_phase23.py
git commit -m "feat: load phase23 backtest data from tqsdk"
```

## Task 6: 添加命令行入口与结果汇总

**Files:**
- Create: `scripts/run_backtest.py`
- Test: `tests/test_run_backtest.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CLI test**

```python
# tests/test_run_backtest.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_backtest  # noqa: E402
from backtest_models import BacktestCase, BacktestResult  # noqa: E402


def test_run_single_case_returns_summary(monkeypatch, capsys):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=run_backtest.date(2025, 1, 1),
        end_dt=run_backtest.date(2025, 1, 31),
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(run_backtest, "load_config", lambda: {"tqsdk": {"account": "acct", "password": "pwd"}})
    monkeypatch.setattr(run_backtest, "load_case_frames_with_tqbacktest", lambda case, config: ("daily", "minute"))
    monkeypatch.setattr(
        run_backtest,
        "run_case_from_frames",
        lambda case, daily_df, minute_df, plan_factory, cfg: BacktestResult(case_id=case.case_id, trades=[], summary={"num_trades": 0, "win_rate": 0.0}),
    )

    exit_code = run_backtest.main(["--case", "lh0_long"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "lh0_long" in captured.out
    assert "num_trades" in captured.out
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
pytest tests/test_run_backtest.py -v
```

Expected:

- `ModuleNotFoundError: No module named 'run_backtest'`

- [ ] **Step 3: Implement CLI and README note**

```python
# scripts/run_backtest.py
from __future__ import annotations

import argparse
from datetime import date

from backtest_cases import get_case
from backtest_metrics import summarize_trades
from backtest_phase23 import load_case_frames_with_tqbacktest, make_trade_plan_from_phase2, run_case_from_frames
from data_cache import _load_config as load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2/3 最小回测")
    parser.add_argument("--case", required=True, help="case id, e.g. lh0_long")
    args = parser.parse_args(argv)

    case = get_case(args.case)
    config = load_config()
    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
    result = run_case_from_frames(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=lambda case, daily_df, pre_market_cfg: make_trade_plan_from_phase2(
            case=case,
            daily_df=daily_df,
            pre_market_cfg=pre_market_cfg,
        ),
        pre_market_cfg=config.get("pre_market", {}),
        signal_cfg=config.get("intraday", {}),
    )
    summary = summarize_trades(result.trades)
    print(f"case={case.case_id}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```markdown
# README.md

## 最小回测

先验证 `Phase 2/3` 的执行层，不回测 `Phase 1` 选品：

```bash
python scripts/run_backtest.py --case lh0_long
python scripts/run_backtest.py --case ps0_short
```
```

- [ ] **Step 4: Run tests and a local smoke command**

Run:

```bash
pytest tests/test_run_backtest.py tests/test_backtest_metrics.py tests/test_backtest_phase23.py -v
python scripts/run_backtest.py --case lh0_long
```

Expected:

- CLI unit test passes
- focused backtest tests all pass
- smoke command prints `case=lh0_long` and summary keys

- [ ] **Step 5: Commit**

```bash
git add scripts/run_backtest.py README.md tests/test_run_backtest.py
git commit -m "feat: add minimal phase23 backtest cli"
```

## Task 7: 全量回归并收口

**Files:**
- Modify: `tests/test_backtest_phase23.py` (only if integration gaps remain)
- Modify: `tests/test_pre_market.py` (only if helper contract changed)

- [ ] **Step 1: Run full verification suite**

Run:

```bash
pytest -q
python -m compileall scripts tests
```

Expected:

- All tests pass
- `compileall` exits `0`

- [ ] **Step 2: Do a manual two-case smoke pass**

Run:

```bash
python scripts/run_backtest.py --case lh0_long
python scripts/run_backtest.py --case ps0_short
```

Expected:

- both commands exit `0`
- both commands print summary output
- no live trading account or下单动作发生

- [ ] **Step 3: Review against the spec checklist**

Check manually:

- `Phase 2` 计划是否来自纯 helper，而不是回测中直接调线上抓数逻辑
- `Phase 3` 是否保持 `5m` 信号、`1m` 推进
- `tp1` 是否只记录命中，不提前平仓
- 多空两条链路是否都已覆盖

- [ ] **Step 4: Final commit**

```bash
git add scripts tests README.md
git commit -m "feat: add minimal phase23 tq backtest framework"
```

## Self-Review

- Spec coverage:
  - 已覆盖 `Phase 2/3` 最小闭环、`1m` 推进、`5m` 信号、`long/short` 双 case、固定 `stop/tp1/tp2`、基础结果输出
  - 未超范围纳入 `Phase 1` 历史选品、参数优化、组合风控
- Placeholder scan:
  - 未使用 `TODO/TBD`
  - 每个代码步骤都给了具体代码块
- Type consistency:
  - `BacktestCase` / `TradePlan` / `TradeRecord` / `BacktestResult` 在后续任务中保持一致
  - `make_trade_plan_from_phase2` 与 `run_case_from_frames` 的 `plan_factory(case, daily_df, pre_market_cfg)` 签名一致
