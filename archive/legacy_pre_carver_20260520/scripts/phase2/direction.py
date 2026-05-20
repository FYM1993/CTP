from __future__ import annotations


def choose_phase2_direction(*, long_score: float, short_score: float, delta: float = 12.0) -> str:
    if long_score >= 20.0 and long_score - short_score >= delta:
        return "long"
    if short_score >= 20.0 and short_score - long_score >= delta:
        return "short"
    return "watch"
