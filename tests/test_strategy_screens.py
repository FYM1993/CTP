from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND  # noqa: E402
from strategy_reversal.screen import select_reversal_candidates  # noqa: E402
from strategy_trend import screen as trend_screen  # noqa: E402
from strategy_trend.screen import build_trend_universe, select_trend_candidates  # noqa: E402


def test_select_reversal_candidates_only_keeps_reversal_line() -> None:
    rows = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "labels": ["反转候选"],
            "reversal_score": 88.0,
            "trend_score": 22.0,
            "attention_score": 75.0,
        },
        {
            "symbol": "RM0",
            "name": "菜粕",
            "labels": ["趋势候选"],
            "reversal_score": 20.0,
            "trend_score": 78.0,
            "attention_score": 74.0,
        },
        {
            "symbol": "A0",
            "name": "豆一",
            "labels": ["双标签候选"],
            "reversal_score": 70.0,
            "trend_score": 67.0,
            "attention_score": 72.0,
        },
    ]

    selected = select_reversal_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["LH0", "A0"]
    assert {row["strategy_family"] for row in selected} == {STRATEGY_REVERSAL}


def test_select_trend_candidates_uses_trend_universe_only() -> None:
    rows = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "labels": [],
            "reversal_score": 88.0,
            "trend_score": 22.0,
            "attention_score": 75.0,
        },
        {
            "symbol": "RM0",
            "name": "菜粕",
            "labels": [],
            "reversal_score": 20.0,
            "trend_score": 78.0,
            "attention_score": 12.0,
        },
        {
            "symbol": "A0",
            "name": "豆一",
            "labels": [],
            "reversal_score": 62.0,
            "trend_score": 69.0,
            "attention_score": 11.0,
        },
    ]

    selected = select_trend_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["RM0", "A0"]
    assert {row["strategy_family"] for row in selected} == {STRATEGY_TREND}


def test_build_trend_universe_uses_price_trend_without_phase1(monkeypatch) -> None:
    import pandas as pd

    cu_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120, freq="D"),
            "open": [100 + i for i in range(120)],
            "high": [101 + i for i in range(120)],
            "low": [99 + i for i in range(120)],
            "close": [100 + i for i in range(120)],
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )
    ag_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120, freq="D"),
            "open": [200 - i for i in range(120)],
            "high": [201 - i for i in range(120)],
            "low": [199 - i for i in range(120)],
            "close": [200 - i for i in range(120)],
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        trend_screen.pre_market,
        "score_signals",
        lambda df, direction, cfg: {"trend": 58.0, "volume": 11.0, "wyckoff": -4.0}
        if float(df["close"].iloc[-1]) > 150
        else {"trend": -56.0, "volume": -10.0, "wyckoff": 3.0},
    )
    monkeypatch.setattr(
        trend_screen.pre_market,
        "resolve_phase2_direction",
        lambda *, long_score, short_score, delta=12.0: "long" if long_score >= short_score else "short",
    )

    selected = build_trend_universe(
        all_data={"CU0": cu_df, "AG0": ag_df},
        symbols=[
            {"symbol": "CU0", "name": "铜", "exchange": "shfe"},
            {"symbol": "AG0", "name": "白银", "exchange": "shfe"},
        ],
        config={"pre_market": {"min_history_bars": 60, "direction_delta": 12.0}},
    )

    assert [row["symbol"] for row in selected] == ["CU0", "AG0"]
    assert selected[0]["trend_direction"] == "long"
    assert selected[1]["trend_direction"] == "short"
    assert all(row["labels"] == ["趋势候选"] for row in selected)
