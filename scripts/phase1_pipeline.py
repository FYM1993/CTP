from __future__ import annotations


def select_top_candidates(rows: list[dict], top_n: int) -> list[dict]:
    eligible = [
        row
        for row in rows
        if max(row["reversal_score"], row["trend_score"]) >= 55
    ]
    eligible.sort(key=lambda row: row["attention_score"], reverse=True)
    return eligible[:top_n]
