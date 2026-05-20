from __future__ import annotations

STRATEGY_REVERSAL = "reversal_fundamental"
STRATEGY_TREND = "trend_following"


def strategy_sort_score(row: dict, strategy_family: str) -> float:
    if strategy_family == STRATEGY_REVERSAL:
        return float(row.get("reversal_score", 0.0))
    return float(row.get("trend_score", 0.0))
