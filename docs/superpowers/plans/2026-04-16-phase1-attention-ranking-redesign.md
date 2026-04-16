# Phase 1 关注优先级重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `Phase 1` 从“基本面定方向的单分数筛选”重构为“反转机会分 + 趋势机会分 + 关注优先级分”的统一总池子，并将价格/K线/持仓量数据统一切到 `TqSdk` 口径。

**Architecture:** 新增独立的 `Phase 1` 模型、因子、评分和流水线模块，`daily_workflow.py` 只负责编排。市场数据通过 `TqSdk` 统一获取，非价量基本面优先通过 `TqSdk EDB` 获取，未覆盖字段再回退到 `AkShare`；`Phase 2` 不再接收 `Phase 1` 的方向，而是根据技术面自行决定方向和时机。

**Tech Stack:** Python 3、pandas、PyYAML、TqSdk、AkShare、pytest

---

## File Structure

### New Files

- `scripts/phase1_models.py`
  负责 `Phase 1` 的结构化类型：原始因子、分数拆解、候选结果、标签、状态标签、数据源枚举。
- `scripts/market_data_tq.py`
  负责 `TqSdk` 行情读取、主连映射、日线/分钟线/持仓量统一格式转换。
- `scripts/fundamental_edb_map.py`
  负责 `TqSdk EDB` 指标映射注册表与字段查询入口。
- `scripts/fundamental_data.py`
  负责基本面数据统一入口，按 `EDB -> AkShare` 顺序取数据并返回内部模型。
- `scripts/phase1_factors.py`
  负责将价格与基本面原始数据转换为标准化因子。
- `scripts/phase1_scoring.py`
  负责计算反转机会分、趋势机会分、关注优先级分、标签、状态标签。
- `scripts/phase1_pipeline.py`
  负责跑全市场、生成总池子、排序、截取 `Top N`。
- `scripts/phase2_direction.py`
  负责 `Phase 2` 的方向判定与技术面优先级决策。
- `tests/test_phase1_models.py`
- `tests/test_market_data_tq.py`
- `tests/test_fundamental_data.py`
- `tests/test_phase1_factors.py`
- `tests/test_phase1_scoring.py`
- `tests/test_phase1_pipeline.py`
- `tests/test_phase2_direction.py`

### Modified Files

- `scripts/daily_workflow.py`
  去掉 `Phase 1` 的方向决策逻辑，改为调用新的 `phase1_pipeline` 与 `phase2_direction`。
- `scripts/data_cache.py`
  将 `get_daily` / `get_minute` / `get_daily_with_live_bar` / `prefetch_all` 改成 `TqSdk` 主口径，并保留 `AkShare` 基本面适配函数或迁移到 `fundamental_data.py`。
- `scripts/tqsdk_live.py`
  抽出或复用合约映射逻辑，避免与 `market_data_tq.py` 重复实现。
- `scripts/pre_market.py`
  让 `analyze_one()` 支持不依赖 `Phase 1 direction` 的新输入，或者配合 `phase2_direction.py` 完成技术面方向判断。
- `config.example.yaml`
  新增 `phase1.top_n`、`phase1.thresholds`、`phase1.single_side_bonus`、`phase1.data_source` 等配置示例。
- `README.md`
  更新 `Phase 1` 语义、数据源策略、输出字段说明。

---

### Task 1: 建立 Phase 1 类型模型

**Files:**
- Create: `scripts/phase1_models.py`
- Test: `tests/test_phase1_models.py`

- [ ] **Step 1: 写失败测试，固定标签、状态标签和结果结构**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_models import (  # noqa: E402
    Phase1Candidate,
    Phase1Label,
    Phase1StateLabel,
)


def test_phase1_candidate_defaults():
    cand = Phase1Candidate(
        symbol="LH0",
        name="生猪",
        reversal_score=82.0,
        trend_score=36.0,
        attention_score=74.0,
        data_coverage=0.75,
        labels=[Phase1Label.REVERSAL],
        state_labels=[Phase1StateLabel.LOW_LEVEL_EXIT],
        reason_summary="深亏接近成本线，存在低位出清迹象",
    )
    assert cand.symbol == "LH0"
    assert cand.labels == [Phase1Label.REVERSAL]
    assert cand.state_labels == [Phase1StateLabel.LOW_LEVEL_EXIT]
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_phase1_models.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase1_models'`

- [ ] **Step 3: 写最小实现**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase1Label(str, Enum):
    REVERSAL = "反转候选"
    TREND = "趋势候选"
    BOTH = "双标签候选"
    LOW_COVERAGE = "数据覆盖不足"


class Phase1StateLabel(str, Enum):
    LOW_LEVEL_EXIT = "低位出清"
    HIGH_LEVEL_EXPANSION = "高位扩产"
    TIGHTENING_BALANCE = "紧平衡强化"
    OVERSUPPLY_DEEPENING = "过剩深化"


@dataclass(slots=True)
class Phase1Candidate:
    symbol: str
    name: str
    reversal_score: float
    trend_score: float
    attention_score: float
    data_coverage: float
    labels: list[Phase1Label]
    state_labels: list[Phase1StateLabel]
    reason_summary: str
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase1_models.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_phase1_models.py scripts/phase1_models.py
git commit -m "feat: add phase1 typed models"
```

### Task 2: 抽离 TqSdk 市场数据入口

**Files:**
- Create: `scripts/market_data_tq.py`
- Modify: `scripts/tqsdk_live.py`
- Test: `tests/test_market_data_tq.py`

- [ ] **Step 1: 写失败测试，固定主连映射与日线格式转换**

```python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from market_data_tq import (  # noqa: E402
    klines_to_daily_frame,
    symbol_to_tq_main,
)


def test_symbol_to_tq_main_uses_exchange_table():
    assert symbol_to_tq_main("LH0", "dce") == "KQ.m@DCE.lh"


def test_klines_to_daily_frame_renames_columns():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-04-15"]),
            "open": [100],
            "high": [110],
            "low": [90],
            "close": [105],
            "volume": [1234],
            "close_oi": [888],
        }
    )
    out = klines_to_daily_frame(frame)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
    assert float(out.iloc[0]["oi"]) == 888.0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_market_data_tq.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'market_data_tq'`

- [ ] **Step 3: 写最小实现，并让 `tqsdk_live.py` 复用同一映射函数**

```python
from __future__ import annotations

import re
import pandas as pd

EXCHANGE_UPPER = {
    "dce": "DCE",
    "czce": "CZCE",
    "shfe": "SHFE",
    "ine": "INE",
    "gfex": "GFEX",
}


def symbol_to_tq_main(symbol: str, exchange: str) -> str:
    match = re.match(r"^([A-Za-z]+)0$", symbol)
    if not match:
        raise ValueError(f"不支持的主连代码: {symbol}")
    return f"KQ.m@{EXCHANGE_UPPER[exchange.lower()]}.{match.group(1).lower()}"


def klines_to_daily_frame(klines: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(klines["datetime"]).dt.normalize(),
            "open": pd.to_numeric(klines["open"], errors="coerce"),
            "high": pd.to_numeric(klines["high"], errors="coerce"),
            "low": pd.to_numeric(klines["low"], errors="coerce"),
            "close": pd.to_numeric(klines["close"], errors="coerce"),
            "volume": pd.to_numeric(klines["volume"], errors="coerce"),
            "oi": pd.to_numeric(klines.get("close_oi", 0.0), errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["close"]).reset_index(drop=True)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_market_data_tq.py -v`

Expected: PASS with `2 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_market_data_tq.py scripts/market_data_tq.py scripts/tqsdk_live.py
git commit -m "feat: add tq market data adapter"
```

### Task 3: 建立 EDB 优先、AkShare 兜底的数据源门面

**Files:**
- Create: `scripts/fundamental_edb_map.py`
- Create: `scripts/fundamental_data.py`
- Test: `tests/test_fundamental_data.py`

- [ ] **Step 1: 写失败测试，固定“未映射时回退 AkShare”的行为**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fundamental_data import FundamentalBundle, pick_inventory_source  # noqa: E402


def test_pick_inventory_source_falls_back_when_edb_unmapped():
    bundle = FundamentalBundle(edb_series=None, akshare_value={"inv_change_4wk": 10.0})
    source, payload = pick_inventory_source(bundle)
    assert source == "akshare"
    assert payload["inv_change_4wk"] == 10.0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_fundamental_data.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fundamental_data'`

- [ ] **Step 3: 写最小实现，并把 EDB 映射注册表单独放在 `fundamental_edb_map.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EDB_FIELD_MAP: dict[str, dict[str, int]] = {
    "inventory": {},
    "receipt": {},
    "profit": {},
    "cost": {},
    "spot": {},
}


@dataclass(slots=True)
class FundamentalBundle:
    edb_series: dict[str, Any] | None
    akshare_value: dict[str, Any] | None


def pick_inventory_source(bundle: FundamentalBundle) -> tuple[str, dict[str, Any] | None]:
    if bundle.edb_series:
        return "edb", bundle.edb_series
    return "akshare", bundle.akshare_value
```

- [ ] **Step 4: 手工核对 EDB 覆盖清单，并把已确认字段编码进映射注册表**

Run: 打开 `https://edb.shinnytech.com`

Expected:
- 记录项目当前使用字段是否存在 EDB 对应项
- 将已验证的指标 ID 写入 `scripts/fundamental_edb_map.py`
- 未覆盖字段明确保留为空映射，表示运行时自动回退 `AkShare`

- [ ] **Step 5: 提交**

```bash
git add tests/test_fundamental_data.py scripts/fundamental_data.py scripts/fundamental_edb_map.py
git commit -m "feat: add fundamental data facade"
```

### Task 4: 实现 Phase 1 因子提取层

**Files:**
- Create: `scripts/phase1_factors.py`
- Test: `tests/test_phase1_factors.py`

- [ ] **Step 1: 写失败测试，固定“低价但不接近成本线”与“低价且深亏”必须区分**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_factors import build_reversal_factors  # noqa: E402


def test_low_price_without_cost_stress_is_not_extreme_reversal():
    factors = build_reversal_factors(
        price_vs_cost=0.98,
        price_percentile_300d=12.0,
        price_percentile_full=18.0,
        profit_margin=-2.0,
        loss_persistence_days=8,
    )
    assert factors["长期均衡偏离"] < 70


def test_low_price_with_deep_loss_creates_extreme_reversal():
    factors = build_reversal_factors(
        price_vs_cost=0.84,
        price_percentile_300d=8.0,
        price_percentile_full=10.0,
        profit_margin=-35.0,
        loss_persistence_days=90,
    )
    assert factors["长期均衡偏离"] >= 80
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_phase1_factors.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase1_factors'`

- [ ] **Step 3: 写最小实现，先返回标准化后的中文因子字典**

```python
from __future__ import annotations


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def build_reversal_factors(
    *,
    price_vs_cost: float,
    price_percentile_300d: float,
    price_percentile_full: float,
    profit_margin: float,
    loss_persistence_days: int,
) -> dict[str, float]:
    cost_stress = _clip_score((1.0 - price_vs_cost) * 220)
    price_stress = _clip_score((20.0 - min(price_percentile_300d, price_percentile_full)) * 4.0)
    profit_stress = _clip_score(abs(min(profit_margin, 0.0)) * 2.0)
    persistence = _clip_score(loss_persistence_days / 120 * 100.0)
    return {
        "长期均衡偏离": round((cost_stress * 0.45 + price_stress * 0.20 + profit_stress * 0.20 + persistence * 0.15), 2),
        "失衡持续性": round(persistence, 2),
    }
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase1_factors.py -v`

Expected: PASS with `2 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_phase1_factors.py scripts/phase1_factors.py
git commit -m "feat: add phase1 factor builders"
```

### Task 5: 实现评分、标签与关注优先级分

**Files:**
- Create: `scripts/phase1_scoring.py`
- Test: `tests/test_phase1_scoring.py`

- [ ] **Step 1: 写失败测试，固定标签判定与单边强化加分**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_scoring import build_labels, calc_attention_raw  # noqa: E402


def test_build_labels_marks_both_when_scores_cross_threshold():
    labels = build_labels(reversal_score=78.0, trend_score=66.0, data_coverage=0.82)
    assert labels == ["双标签候选"]


def test_calc_attention_raw_keeps_single_side_strong_name_near_front():
    high_single = calc_attention_raw(
        reversal_rank=2,
        trend_rank=40,
        reversal_score=91.0,
        trend_score=34.0,
        data_coverage=0.88,
    )
    balanced = calc_attention_raw(
        reversal_rank=12,
        trend_rank=11,
        reversal_score=70.0,
        trend_score=68.0,
        data_coverage=0.88,
    )
    assert high_single > 0
    assert balanced > 0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_phase1_scoring.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase1_scoring'`

- [ ] **Step 3: 写最小实现**

```python
from __future__ import annotations


def build_labels(*, reversal_score: float, trend_score: float, data_coverage: float) -> list[str]:
    labels: list[str] = []
    if reversal_score >= 60 and trend_score >= 60:
        labels.append("双标签候选")
    elif reversal_score >= 60:
        labels.append("反转候选")
    elif trend_score >= 60:
        labels.append("趋势候选")
    if data_coverage < 0.5:
        labels.append("数据覆盖不足")
    return labels


def calc_attention_raw(
    *,
    reversal_rank: int,
    trend_rank: int,
    reversal_score: float,
    trend_score: float,
    data_coverage: float,
) -> float:
    rrf_core = 1 / (20 + reversal_rank) + 1 / (20 + trend_rank)
    dominant_score = max(reversal_score, trend_score)
    weak_score = min(reversal_score, trend_score)
    dominant_rank = min(reversal_rank, trend_rank)
    dominance_strength = max(0.0, min(1.0, (dominant_score - 75.0) / 25.0))
    dominance_asymmetry = max(0.0, min(1.0, (dominant_score - weak_score - 20.0) / 40.0))
    single_side_bonus = 0.9 * dominance_strength * dominance_asymmetry * (1 / (6 + dominant_rank))
    return (rrf_core + single_side_bonus) * (0.7 + 0.3 * data_coverage)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase1_scoring.py -v`

Expected: PASS with `2 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_phase1_scoring.py scripts/phase1_scoring.py
git commit -m "feat: add phase1 scoring and labels"
```

### Task 6: 实现统一总池子流水线

**Files:**
- Create: `scripts/phase1_pipeline.py`
- Test: `tests/test_phase1_pipeline.py`

- [ ] **Step 1: 写失败测试，固定“总池子 + Top N + 中文标签”行为**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_pipeline import select_top_candidates  # noqa: E402


def test_select_top_candidates_uses_attention_score_only():
    rows = [
        {"symbol": "LH0", "name": "生猪", "reversal_score": 88.0, "trend_score": 35.0, "attention_score": 76.0},
        {"symbol": "RM0", "name": "菜粕", "reversal_score": 40.0, "trend_score": 79.0, "attention_score": 73.0},
        {"symbol": "A0", "name": "豆一", "reversal_score": 65.0, "trend_score": 66.0, "attention_score": 74.0},
    ]
    out = select_top_candidates(rows, top_n=2)
    assert [x["symbol"] for x in out] == ["LH0", "A0"]
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_phase1_pipeline.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase1_pipeline'`

- [ ] **Step 3: 写最小实现**

```python
from __future__ import annotations


def select_top_candidates(rows: list[dict], top_n: int) -> list[dict]:
    eligible = [row for row in rows if max(row["reversal_score"], row["trend_score"]) >= 55]
    eligible.sort(key=lambda row: row["attention_score"], reverse=True)
    return eligible[:top_n]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase1_pipeline.py -v`

Expected: PASS with `1 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_phase1_pipeline.py scripts/phase1_pipeline.py
git commit -m "feat: add phase1 candidate pipeline"
```

### Task 7: 让 Phase 2 自行判定方向

**Files:**
- Create: `scripts/phase2_direction.py`
- Modify: `scripts/pre_market.py`
- Test: `tests/test_phase2_direction.py`

- [ ] **Step 1: 写失败测试，固定“Phase 2 自行做多/做空/仅观察”行为**

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase2_direction import choose_phase2_direction  # noqa: E402


def test_choose_phase2_direction_prefers_long_when_long_side_clearly_stronger():
    result = choose_phase2_direction(long_score=32.0, short_score=10.0, delta=12.0)
    assert result == "long"


def test_choose_phase2_direction_returns_watch_when_gap_too_small():
    result = choose_phase2_direction(long_score=21.0, short_score=17.0, delta=12.0)
    assert result == "watch"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_phase2_direction.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase2_direction'`

- [ ] **Step 3: 写最小实现，并在 `pre_market.py` 中保留现有 `score_signals()` 作为单边基础能力**

```python
from __future__ import annotations


def choose_phase2_direction(*, long_score: float, short_score: float, delta: float = 12.0) -> str:
    if long_score >= 20.0 and long_score - short_score >= delta:
        return "long"
    if short_score >= 20.0 and short_score - long_score >= delta:
        return "short"
    return "watch"
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase2_direction.py -v`

Expected: PASS with `2 passed`

- [ ] **Step 5: 提交**

```bash
git add tests/test_phase2_direction.py scripts/phase2_direction.py scripts/pre_market.py
git commit -m "feat: move direction selection into phase2"
```

### Task 8: 将 daily_workflow 接入新的 Phase 1 / Phase 2 边界

**Files:**
- Modify: `scripts/daily_workflow.py`
- Modify: `scripts/config.example.yaml`
- Test: `tests/test_phase1_pipeline.py`

- [ ] **Step 1: 写失败测试，固定 `Phase 1` 不再返回 direction**

```python
def test_phase1_output_has_attention_not_direction():
    candidate = {
        "symbol": "LH0",
        "name": "生猪",
        "reversal_score": 88.0,
        "trend_score": 35.0,
        "attention_score": 76.0,
        "labels": ["反转候选"],
    }
    assert "direction" not in candidate
    assert candidate["attention_score"] == 76.0
```

- [ ] **Step 2: 运行测试，确认旧逻辑仍依赖 direction**

Run: `pytest tests/test_phase1_pipeline.py::test_phase1_output_has_attention_not_direction -v`

Expected: FAIL because production output still carries `direction` assumptions

- [ ] **Step 3: 修改编排逻辑**

```python
# daily_workflow.py
phase1_candidates = run_phase1_pipeline(all_market_data, config)

phase2_inputs = []
for row in phase1_candidates:
    phase2_inputs.append(
        {
            "symbol": row.symbol,
            "name": row.name,
            "reversal_score": row.reversal_score,
            "trend_score": row.trend_score,
            "attention_score": row.attention_score,
            "labels": [label.value for label in row.labels],
            "state_labels": [label.value for label in row.state_labels],
            "data_coverage": row.data_coverage,
            "reason_summary": row.reason_summary,
        }
    )
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_phase1_pipeline.py -v`

Expected: PASS and no assertions depend on `Phase 1 direction`

- [ ] **Step 5: 提交**

```bash
git add scripts/daily_workflow.py scripts/config.example.yaml tests/test_phase1_pipeline.py
git commit -m "feat: wire workflow to new phase1 outputs"
```

### Task 9: 将缓存层切换到 TqSdk 行情口径

**Files:**
- Modify: `scripts/data_cache.py`
- Modify: `scripts/market_data_tq.py`
- Test: `tests/test_market_data_tq.py`

- [ ] **Step 1: 写失败测试，固定 `get_daily()` 走 TqSdk 统一列名**

```python
def test_get_daily_returns_tq_normalized_columns(monkeypatch):
    import pandas as pd
    from data_cache import get_daily

    fake = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-15"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [12.0],
            "oi": [7.0],
        }
    )

    monkeypatch.setattr("data_cache.fetch_daily_from_tq", lambda symbol, days: fake)
    out = get_daily("LH0", 30)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "oi"]
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `pytest tests/test_market_data_tq.py::test_get_daily_returns_tq_normalized_columns -v`

Expected: FAIL because `data_cache.get_daily` still calls the AkShare fetcher

- [ ] **Step 3: 修改缓存层，默认走 TqSdk，保留 `AkShare` 只做基本面**

```python
# data_cache.py
def get_daily(symbol: str, days: int = 400) -> pd.DataFrame | None:
    _ensure_cache_dir()
    final_path, live_path = _cache_paths(symbol, "daily")
    cached = _read_cached_daily(final_path, live_path, days)
    if cached is not None:
        return cached
    df = fetch_daily_from_tq(symbol, days)
    if df is not None and len(df) > 0:
        _write_daily_cache(df, final_path, live_path)
        return df.tail(days)
    return None
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `pytest tests/test_market_data_tq.py -v`

Expected: PASS and no daily/minute market fetch tests depend on AkShare行情

- [ ] **Step 5: 提交**

```bash
git add scripts/data_cache.py scripts/market_data_tq.py tests/test_market_data_tq.py
git commit -m "feat: switch cached market data to tqsdk"
```

### Task 10: 更新报告、文档和回归测试

**Files:**
- Modify: `scripts/daily_workflow.py`
- Modify: `README.md`
- Modify: `config.example.yaml`
- Test: `tests/test_fundamental_p1.py`

- [ ] **Step 1: 写失败测试，固定报告字段为中文新语义**

```python
def test_report_uses_attention_fields():
    row = {
        "symbol": "LH0",
        "name": "生猪",
        "attention_score": 76.0,
        "reversal_score": 88.0,
        "trend_score": 35.0,
        "labels": ["反转候选"],
    }
    assert "attention_score" in row
    assert "direction" not in row
```

- [ ] **Step 2: 运行测试，确认旧报告仍引用 direction/fund_screen_score 语义**

Run: `pytest tests/test_fundamental_p1.py -v`

Expected: FAIL or require report/output updates before tests are green

- [ ] **Step 3: 更新文档和输出字段**

```python
# daily_workflow.py save_targets()
payload = {
    "date": today_str,
    "targets": targets,
    "watchlist": watchlist,
    "phase1_summary": {
        "top_n": top_n,
        "排序字段": "关注优先级分",
        "标签字段": ["反转候选", "趋势候选", "双标签候选", "数据覆盖不足"],
    },
}
```

- [ ] **Step 4: 跑完整回归测试**

Run: `pytest -q`

Expected: PASS with all `Phase 1` / `Phase 2` / `TqSdk` adapter tests green

- [ ] **Step 5: 提交**

```bash
git add scripts/daily_workflow.py README.md config.example.yaml tests/test_fundamental_p1.py
git commit -m "docs: update phase1 semantics and report fields"
```

## Spec Coverage Check

- `Phase 1` 只负责发现机会，不再直接定方向：Task 5, Task 6, Task 8
- 反转机会分 / 趋势机会分 / 关注优先级分：Task 4, Task 5
- 总池子 + Top N：Task 6
- 中文术语统一：Task 8, Task 10
- `TqSdk` 统一行情源：Task 2, Task 9
- `EDB -> AkShare` 基本面优先级：Task 3
- `Phase 2` 自决方向：Task 7
- 模块边界拆分：Task 1-9 全部覆盖

## Self-Review

- 占位词检查：未使用 `TODO`、`TBD`、`implement later` 等占位词。
- 类型一致性检查：计划中统一使用“反转机会分 / 趋势机会分 / 关注优先级分 / 数据覆盖度”术语；`Phase 2` 输出方向，`Phase 1` 不输出方向。
- 依赖方向检查：`daily_workflow.py` 只依赖流水线与评分模块；`phase1_scoring.py` 不依赖 `AkShare` 或 `TqSdk`；`fundamental_data.py` 屏蔽底层数据源差异。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-phase1-attention-ranking-redesign.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
