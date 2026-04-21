from __future__ import annotations

from datetime import date

import pandas as pd

from phase2 import pre_market
from shared.strategy import STRATEGY_TREND, strategy_sort_score


def select_trend_candidates(rows: list[dict], top_n: int, *, min_trend_score: float = 60.0) -> list[dict]:
    selected: list[dict] = []
    for row in rows:
        trend_score = float(row.get("trend_score", 0.0))
        if trend_score < min_trend_score:
            continue
        candidate = dict(row)
        candidate["strategy_family"] = STRATEGY_TREND
        candidate["strategy_score"] = strategy_sort_score(row, STRATEGY_TREND)
        selected.append(candidate)
    selected.sort(key=lambda item: item["strategy_score"], reverse=True)
    return selected[:top_n]


def _slice_visible_frame(df: pd.DataFrame, *, as_of_date: date | None = None) -> pd.DataFrame:
    if as_of_date is None or "date" not in df.columns:
        return df.reset_index(drop=True)
    visible = df.copy()
    visible["date"] = pd.to_datetime(visible["date"])
    visible = visible.loc[visible["date"].dt.date <= as_of_date].copy()
    return visible.reset_index(drop=True)


def _trend_reason_summary(scores: dict[str, float]) -> str:
    ranked = sorted(
        ((str(key), float(value)) for key, value in scores.items()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    reasons = [f"{name}{value:+.0f}" for name, value in ranked[:3] if abs(value) >= 3.0]
    return "；".join(reasons) or "技术趋势证据有限"


def build_trend_universe(
    *,
    all_data: dict[str, pd.DataFrame],
    symbols: list[dict],
    config: dict,
    as_of_date: date | None = None,
    min_trend_score: float = 60.0,
) -> list[dict]:
    pre_market_cfg = config.get("pre_market") or {}
    min_history = max(int(pre_market_cfg.get("min_history_bars", 60)), 1)
    trend_rows: list[dict] = []

    for info in symbols:
        symbol = str(info["symbol"])
        frame = all_data.get(symbol)
        if frame is None or getattr(frame, "empty", False):
            continue
        df = _slice_visible_frame(frame, as_of_date=as_of_date)
        if df.empty or len(df) < min_history:
            continue

        phase2_scores = pre_market.score_signals(df, "long", pre_market_cfg)
        long_score = float(sum(value for value in phase2_scores.values() if value > 0))
        short_score = float(sum(-value for value in phase2_scores.values() if value < 0))
        resolved_direction = pre_market.resolve_phase2_direction(
            long_score=long_score,
            short_score=short_score,
            delta=float(pre_market_cfg.get("direction_delta", 12.0)),
        )
        trend_score = max(long_score, short_score)
        if trend_score < float(min_trend_score):
            continue
        if resolved_direction == "watch":
            continue

        direction = "long" if long_score >= short_score else "short"
        last = float(pd.to_numeric(df["close"], errors="coerce").dropna().iloc[-1])
        trend_rows.append(
            {
                "symbol": symbol,
                "name": str(info["name"]),
                "exchange": str(info["exchange"]),
                "price": last,
                "range_pct": 0.0,
                "score": trend_score,
                "fund_score": trend_score,
                "fund_details": _trend_reason_summary(phase2_scores),
                "reversal_score": 0.0,
                "trend_score": trend_score,
                "attention_score": trend_score,
                "trend_direction": direction,
                "labels": ["趋势候选"],
                "state_labels": [],
                "data_coverage": 1.0,
                "reason_summary": _trend_reason_summary(phase2_scores),
                "entry_pool_reason": f"趋势技术分达标（{trend_score:.0f}）",
            }
        )

    trend_rows.sort(key=lambda row: float(row.get("trend_score", 0.0)), reverse=True)
    return trend_rows
