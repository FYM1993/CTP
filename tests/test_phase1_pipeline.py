from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_pipeline import select_top_candidates  # noqa: E402


def test_select_top_candidates_orders_by_attention_only() -> None:
    rows = [
        {
            "symbol": "LH0",
            "reversal_score": 60.0,
            "trend_score": 36.0,
            "attention_score": 76.0,
        },
        {
            "symbol": "RM0",
            "reversal_score": 82.0,
            "trend_score": 55.0,
            "attention_score": 73.0,
        },
        {
            "symbol": "A0",
            "reversal_score": 70.0,
            "trend_score": 57.0,
            "attention_score": 74.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=2)

    assert [row["symbol"] for row in selected] == ["LH0", "A0"]


def test_select_top_candidates_excludes_rows_below_threshold() -> None:
    rows = [
        {
            "symbol": "LOW0",
            "reversal_score": 54.0,
            "trend_score": 54.0,
            "attention_score": 99.0,
        },
        {
            "symbol": "KEEP0",
            "reversal_score": 55.0,
            "trend_score": 20.0,
            "attention_score": 10.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["KEEP0"]
