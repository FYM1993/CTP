from __future__ import annotations

from shared.strategy import STRATEGY_REVERSAL, strategy_sort_score


def select_reversal_candidates(rows: list[dict], top_n: int) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        labels = set(row.get("labels") or [])
        if "反转候选" not in labels and "双标签候选" not in labels:
            continue
        candidate = dict(row)
        candidate["strategy_family"] = STRATEGY_REVERSAL
        candidate["strategy_score"] = strategy_sort_score(row, STRATEGY_REVERSAL)
        selected.append(candidate)
    return sorted(
        selected,
        key=lambda item: (item["strategy_score"], float(item.get("attention_score", 0.0))),
        reverse=True,
    )[:top_n]
