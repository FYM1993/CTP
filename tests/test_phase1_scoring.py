from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1_scoring import build_labels, calc_attention_raw  # noqa: E402


def test_build_labels_double_label_candidate() -> None:
    labels = build_labels(
        reversal_score=78.0,
        trend_score=66.0,
        data_coverage=0.82,
    )
    assert labels == ["双标签候选"]


def test_build_labels_low_coverage_adds_warning_label() -> None:
    labels = build_labels(
        reversal_score=78.0,
        trend_score=40.0,
        data_coverage=0.42,
    )
    assert labels == ["反转候选", "数据覆盖不足"]


def test_calc_attention_raw_prefers_strong_single_side_over_balanced() -> None:
    high_single = calc_attention_raw(
        reversal_rank=1,
        trend_rank=12,
        reversal_score=96.0,
        trend_score=40.0,
        data_coverage=0.82,
    )
    balanced = calc_attention_raw(
        reversal_rank=3,
        trend_rank=4,
        reversal_score=72.0,
        trend_score=70.0,
        data_coverage=0.82,
    )
    assert high_single > balanced


def test_calc_attention_raw_uses_dominant_side_rank() -> None:
    dominant_better_rank = calc_attention_raw(
        reversal_rank=1,
        trend_rank=12,
        reversal_score=96.0,
        trend_score=40.0,
        data_coverage=0.82,
    )
    dominant_worse_rank = calc_attention_raw(
        reversal_rank=12,
        trend_rank=1,
        reversal_score=96.0,
        trend_score=40.0,
        data_coverage=0.82,
    )
    assert dominant_worse_rank < dominant_better_rank


def test_calc_attention_raw_strictly_shrinks_with_lower_coverage() -> None:
    high_coverage = calc_attention_raw(
        reversal_rank=2,
        trend_rank=8,
        reversal_score=90.0,
        trend_score=45.0,
        data_coverage=0.9,
    )
    low_coverage = calc_attention_raw(
        reversal_rank=2,
        trend_rank=8,
        reversal_score=90.0,
        trend_score=45.0,
        data_coverage=0.3,
    )
    assert low_coverage < high_coverage
