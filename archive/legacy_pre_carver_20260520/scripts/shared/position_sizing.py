from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSizingResult:
    score_lots: int
    portfolio_lots: int
    risk_lots: int
    suggested_lots: int
    suggested_margin: float
    suggested_stop_risk: float
    sizing_limited_by: str
    sizing_zero_lot_reason: str


def positive_int_floor(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        return 0
    return max(int(math.floor(value)), 0)


def _limit_reason(
    *,
    score_lots: int,
    portfolio_lots: int,
    risk_lots: int,
    risk_budget: float,
) -> tuple[str, str]:
    caps = {
        "phase2_score_budget": int(score_lots),
        "portfolio_margin": int(portfolio_lots),
    }
    if risk_budget > 0:
        caps["stop_risk"] = int(risk_lots)
    min_lots = min(caps.values()) if caps else 0
    limited_by = "/".join(name for name, lots in caps.items() if lots == min_lots)
    if min_lots >= 1:
        return limited_by, ""
    if "stop_risk" in limited_by:
        return limited_by, "single_trade_stop_risk_below_one_lot"
    if "portfolio_margin" in limited_by:
        return limited_by, "portfolio_margin_below_one_lot"
    return limited_by, "phase2_score_budget_below_one_lot"


def calculate_position_sizing(
    *,
    margin_per_lot: float,
    risk_per_lot: float,
    score_margin_budget: float = 0.0,
    portfolio_margin_budget: float = 0.0,
    risk_budget: float = 0.0,
    score_lots: int | None = None,
    portfolio_lots: int | None = None,
    risk_lots: int | None = None,
) -> PositionSizingResult:
    resolved_score_lots = (
        int(score_lots)
        if score_lots is not None
        else positive_int_floor(score_margin_budget / margin_per_lot)
        if margin_per_lot > 0
        else 0
    )
    resolved_portfolio_lots = (
        int(portfolio_lots)
        if portfolio_lots is not None
        else positive_int_floor(portfolio_margin_budget / margin_per_lot)
        if margin_per_lot > 0
        else 0
    )
    resolved_risk_lots = (
        int(risk_lots)
        if risk_lots is not None
        else positive_int_floor(risk_budget / risk_per_lot)
        if risk_budget > 0 and risk_per_lot > 0
        else 0
    )

    caps = [resolved_score_lots, resolved_portfolio_lots]
    if risk_budget > 0:
        caps.append(resolved_risk_lots)
    suggested_lots = min(caps) if caps else 0
    sizing_limited_by, sizing_zero_lot_reason = _limit_reason(
        score_lots=resolved_score_lots,
        portfolio_lots=resolved_portfolio_lots,
        risk_lots=resolved_risk_lots,
        risk_budget=risk_budget,
    )
    return PositionSizingResult(
        score_lots=int(resolved_score_lots),
        portfolio_lots=int(resolved_portfolio_lots),
        risk_lots=int(resolved_risk_lots),
        suggested_lots=int(suggested_lots),
        suggested_margin=float(suggested_lots * max(margin_per_lot, 0.0)),
        suggested_stop_risk=float(suggested_lots * max(risk_per_lot, 0.0)),
        sizing_limited_by=sizing_limited_by,
        sizing_zero_lot_reason=sizing_zero_lot_reason,
    )
