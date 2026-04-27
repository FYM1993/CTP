from __future__ import annotations

import math


def analyze_holding_record(
    *,
    holding: dict,
    recommendation: dict | None,
    current_plan: dict | None,
    hold_eval: dict | None,
    current_price: float,
    minimum_tick: float,
    account_equity: float = 0.0,
    risk_per_trade_pct: float = 0.0,
    contract_multiplier: float = 0.0,
) -> dict:
    recommendation = recommendation or {}
    current_plan = current_plan or {}
    hold_eval = hold_eval or {}

    original_stop = float(recommendation.get("stop") or 0.0)
    original_first_stop = float(recommendation.get("first_stop") or original_stop)
    current_stop = float(current_plan.get("stop") or 0.0)
    current_first_stop = float(current_plan.get("first_stop") or current_stop)
    tp1 = float(recommendation.get("tp1") or current_plan.get("tp1") or 0.0)
    tp2 = float(recommendation.get("tp2") or current_plan.get("tp2") or 0.0)
    direction = str(holding.get("direction") or "")
    threshold = float(minimum_tick)
    story_family = _resolve_story_family(recommendation=recommendation, hold_eval=hold_eval)
    story_label = _story_label(story_family)
    story_transition = str(hold_eval.get("story_transition") or "")
    snapshot_missing = bool(hold_eval.get("snapshot_missing"))

    action = "继续持有"
    management_action = ""
    reason = "缺少原始交易剧本，仅按价格和止损保守管理" if snapshot_missing else f"原有{story_label}仍成立"
    stop_hit, active_stop = _stop_hit(
        direction=direction,
        current_price=current_price,
        original_stop=original_stop,
        current_stop=current_stop,
    )
    risk_control = _holding_risk_control(
        direction=direction,
        size=float(holding.get("size") or 0.0),
        current_price=float(current_price),
        stop=active_stop,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        contract_multiplier=contract_multiplier,
    )
    if stop_hit:
        action = "平仓"
        reason = f"当前价已触及止损{active_stop:.0f}，建议退出"
    elif _confirmation_level_invalidated(
        direction=direction,
        current_price=current_price,
        story_family=story_family,
        entry_signal_type=str(recommendation.get("entry_signal_type") or ""),
        confirmation_low=float(recommendation.get("confirmation_low") or 0.0),
        confirmation_high=float(recommendation.get("confirmation_high") or 0.0),
    ):
        action = "平仓"
        reason = _confirmation_break_reason(
            direction=direction,
            story_family=story_family,
            entry_signal_type=str(recommendation.get("entry_signal_type") or ""),
            confirmation_low=float(recommendation.get("confirmation_low") or 0.0),
            confirmation_high=float(recommendation.get("confirmation_high") or 0.0),
        )
    elif not hold_eval.get("hold_valid", True):
        action = "平仓"
        reason = f"原有{story_label}已失效，建议退出"
    elif _reached_target(direction, current_price, tp2):
        action = "平仓"
        reason = "原始止盈位已达到，旧计划已完成，建议退出"
    elif risk_control.get("over_cap"):
        max_safe_size = int(risk_control.get("max_safe_size") or 0)
        if max_safe_size < 1:
            action = "平仓"
            reason = (
                f"当前到止损的潜在亏损约{risk_control['current_risk_cny']:.0f}，"
                f"超过账户{risk_control['risk_pct']:.1%}预算{risk_control['risk_budget']:.0f}，"
                "无法通过减仓控制，建议退出或立即上移保护止损"
            )
        else:
            action = "减仓观察"
            management_action = "减仓"
            reason = (
                f"当前到止损的潜在亏损约{risk_control['current_risk_cny']:.0f}，"
                f"超过账户{risk_control['risk_pct']:.1%}预算{risk_control['risk_budget']:.0f}，"
                f"建议减至{max_safe_size}手或上移保护止损"
            )
    elif _reached_target(direction, current_price, tp1):
        action = "减仓观察"
        reason = "第一止盈位已达到或接近，建议管理仓位"
    elif not snapshot_missing and _stop_improved(direction, current_stop, original_stop, threshold):
        management_action = "上移止损"
        reason = _raise_stop_reason(
            story_label=story_label,
            original_stop=original_stop,
            current_stop=current_stop,
            story_transition=story_transition,
        )
    elif story_transition == "trend_followthrough":
        reason = "原有反转逻辑已进入顺势延续阶段，可继续持有"

    return {
        "record_id": holding.get("record_id"),
        "name": holding.get("name"),
        "direction": direction,
        "direction_cn": "做多" if direction == "long" else "做空",
        "size": float(holding.get("size") or 0.0),
        "entry_price": float(holding.get("entry_price") or 0.0),
        "current_price": float(current_price),
        "action": action,
        "management_action": management_action,
        "reason": reason,
        "risk_control": risk_control,
        "original_plan": {
            "entry": float(recommendation.get("entry") or 0.0),
            "first_stop": original_first_stop,
            "stop": original_stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr": float(recommendation.get("rr") or 0.0),
            "admission_rr": float(recommendation.get("admission_rr") or 0.0),
            "entry_signal_type": str(recommendation.get("entry_signal_type") or ""),
            "entry_signal_detail": str(recommendation.get("entry_signal_detail") or ""),
            "confirmation_low": float(recommendation.get("confirmation_low") or 0.0),
            "confirmation_high": float(recommendation.get("confirmation_high") or 0.0),
        },
        "current_plan": {
            "entry": float(current_plan.get("entry") or 0.0),
            "first_stop": current_first_stop,
            "stop": current_stop,
            "tp1": float(current_plan.get("tp1") or 0.0),
            "tp2": float(current_plan.get("tp2") or 0.0),
            "rr": float(current_plan.get("rr") or 0.0),
            "admission_rr": float(current_plan.get("admission_rr") or 0.0),
        },
    }


def _holding_risk_control(
    *,
    direction: str,
    size: float,
    current_price: float,
    stop: float,
    account_equity: float,
    risk_per_trade_pct: float,
    contract_multiplier: float,
) -> dict:
    if (
        direction not in {"long", "short"}
        or size <= 0
        or current_price <= 0
        or stop <= 0
        or account_equity <= 0
        or risk_per_trade_pct <= 0
        or contract_multiplier <= 0
    ):
        return {
            "enabled": False,
            "over_cap": False,
            "risk_budget": 0.0,
            "risk_pct": 0.0,
            "risk_per_lot": 0.0,
            "current_risk_cny": 0.0,
            "current_risk_pct": 0.0,
            "max_safe_size": 0,
        }

    if direction == "long":
        stop_distance = max(current_price - stop, 0.0)
    else:
        stop_distance = max(stop - current_price, 0.0)
    risk_per_lot = stop_distance * contract_multiplier
    risk_budget = account_equity * risk_per_trade_pct
    current_risk = risk_per_lot * size
    max_safe_size = math.floor(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0
    return {
        "enabled": True,
        "over_cap": bool(risk_per_lot > 0 and current_risk > risk_budget),
        "risk_budget": float(risk_budget),
        "risk_pct": float(risk_per_trade_pct),
        "risk_per_lot": float(risk_per_lot),
        "current_risk_cny": float(current_risk),
        "current_risk_pct": float(current_risk / account_equity),
        "max_safe_size": int(max_safe_size),
    }


def _reached_target(direction: str, current_price: float, target: float) -> bool:
    if target <= 0:
        return False
    if direction == "long":
        return current_price >= target
    return current_price <= target


def _active_stop(direction: str, original_stop: float, current_stop: float) -> float:
    stops = [stop for stop in [original_stop, current_stop] if stop > 0]
    if not stops:
        return 0.0
    if direction == "long":
        return max(stops)
    return min(stops)


def _stop_hit(
    *,
    direction: str,
    current_price: float,
    original_stop: float,
    current_stop: float,
) -> tuple[bool, float]:
    stop = _active_stop(direction, original_stop, current_stop)
    if stop <= 0:
        return False, 0.0
    if direction == "long":
        return current_price <= stop, stop
    return current_price >= stop, stop


def _stop_improved(direction: str, current_stop: float, original_stop: float, threshold: float) -> bool:
    if current_stop <= 0 or original_stop <= 0:
        return False
    if direction == "long":
        return current_stop - original_stop >= threshold
    return original_stop - current_stop >= threshold


def _resolve_story_family(*, recommendation: dict, hold_eval: dict) -> str:
    family = str(hold_eval.get("story_family") or recommendation.get("entry_family") or "")
    if family in {"trend", "reversal"}:
        return family
    return ""


def _confirmation_level_invalidated(
    *,
    direction: str,
    current_price: float,
    story_family: str,
    entry_signal_type: str,
    confirmation_low: float,
    confirmation_high: float,
) -> bool:
    if story_family == "trend" and entry_signal_type != "Pullback":
        return False
    if story_family not in {"trend", "reversal"}:
        return False
    if direction == "long" and confirmation_low > 0:
        return current_price <= confirmation_low
    if direction == "short" and confirmation_high > 0:
        return current_price >= confirmation_high
    return False


def _confirmation_break_reason(
    *,
    direction: str,
    story_family: str,
    entry_signal_type: str,
    confirmation_low: float,
    confirmation_high: float,
) -> str:
    level = confirmation_low if direction == "long" else confirmation_high
    verb = "已失守" if direction == "long" else "已突破"
    if story_family == "reversal":
        return f"原反转确认位{level:.0f}{verb}，建议退出"
    if story_family == "trend" and entry_signal_type == "Pullback":
        return f"原趋势回踩确认位{level:.0f}{verb}，建议退出"
    return f"原交易确认位{level:.0f}{verb}，建议退出"


def _story_label(story_family: str) -> str:
    if story_family == "trend":
        return "顺势逻辑"
    if story_family == "reversal":
        return "反转逻辑"
    return "交易逻辑"


def _raise_stop_reason(
    *,
    story_label: str,
    original_stop: float,
    current_stop: float,
    story_transition: str,
) -> str:
    if story_transition == "trend_followthrough":
        prefix = "原有反转逻辑已进入顺势延续阶段"
    else:
        prefix = f"原有{story_label}仍成立"
    return f"{prefix}，保护止损可从{original_stop:.0f}上移至{current_stop:.0f}"
