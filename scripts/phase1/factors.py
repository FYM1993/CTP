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
    average_price_percentile = (price_percentile_300d + price_percentile_full) / 2.0
    average_low = max(0.0, 20.0 - average_price_percentile)
    price_consistency = max(0.0, 1.0 - abs(price_percentile_300d - price_percentile_full) / 20.0)
    price_stress = _clip_score(average_low * (4.5 + 2.5 * price_consistency))
    profit_stress = _clip_score(abs(min(profit_margin, 0.0)) * 2.0)
    persistence = _clip_score(loss_persistence_days / 120 * 100.0)
    combined_boost = 0.0
    if (
        cost_stress >= 30.0
        and price_stress >= 70.0
        and profit_stress >= 60.0
        and persistence >= 60.0
    ):
        combined_boost = 25.0
    return {
        "长期均衡偏离": round(
            (
                cost_stress * 0.45
                + price_stress * 0.20
                + profit_stress * 0.20
                + persistence * 0.15
                + combined_boost
            ),
            2,
        ),
        "失衡持续性": round(persistence, 2),
    }
