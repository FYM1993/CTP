from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1.factors import build_reversal_factors  # noqa: E402


def test_low_price_without_cost_stress_is_not_extreme_reversal() -> None:
    factors = build_reversal_factors(
        price_vs_cost=0.98,
        price_percentile_300d=12.0,
        price_percentile_full=18.0,
        profit_margin=-2.0,
        loss_persistence_days=8,
    )
    assert factors["长期均衡偏离"] < 20
    assert factors["失衡持续性"] == 6.67


def test_low_price_with_deep_loss_creates_extreme_reversal() -> None:
    factors = build_reversal_factors(
        price_vs_cost=0.84,
        price_percentile_300d=8.0,
        price_percentile_full=10.0,
        profit_margin=-35.0,
        loss_persistence_days=90,
    )
    assert factors["长期均衡偏离"] >= 80
    assert factors["失衡持续性"] == 75.0


def test_mismatched_price_cycles_do_not_overinflate_score() -> None:
    consistent = build_reversal_factors(
        price_vs_cost=0.99,
        price_percentile_300d=10.0,
        price_percentile_full=10.0,
        profit_margin=-1.0,
        loss_persistence_days=5,
    )
    mismatched = build_reversal_factors(
        price_vs_cost=0.99,
        price_percentile_300d=1.0,
        price_percentile_full=19.0,
        profit_margin=-1.0,
        loss_persistence_days=5,
    )
    assert consistent["长期均衡偏离"] > mismatched["长期均衡偏离"]
    assert mismatched["长期均衡偏离"] < 20
