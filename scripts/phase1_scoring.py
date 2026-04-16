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
    if reversal_score >= trend_score:
        dominant_score = reversal_score
        weak_score = trend_score
        dominant_rank = reversal_rank
    else:
        dominant_score = trend_score
        weak_score = reversal_score
        dominant_rank = trend_rank
    dominance_strength = max(0.0, min(1.0, (dominant_score - 75.0) / 25.0))
    dominance_asymmetry = max(0.0, min(1.0, (dominant_score - weak_score - 20.0) / 40.0))
    single_side_bonus = 0.9 * dominance_strength * dominance_asymmetry * (1 / (6 + dominant_rank))
    return (rrf_core + single_side_bonus) * (0.7 + 0.3 * data_coverage)
