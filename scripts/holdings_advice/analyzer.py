from __future__ import annotations


def analyze_holding_record(
    *,
    holding: dict,
    recommendation: dict | None,
    current_plan: dict | None,
    hold_eval: dict | None,
    current_price: float,
    minimum_tick: float,
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

    action = "继续持有"
    management_action = ""
    reason = f"原有{story_label}仍成立"
    stop_hit, active_stop = _stop_hit(
        direction=direction,
        current_price=current_price,
        original_stop=original_stop,
        current_stop=current_stop,
    )
    if stop_hit:
        action = "平仓"
        reason = f"当前价已触及止损{active_stop:.0f}，建议退出"
    elif not hold_eval.get("hold_valid", True):
        action = "平仓"
        reason = f"原有{story_label}已失效，建议退出"
    elif _reached_target(direction, current_price, tp2):
        action = "平仓"
        reason = "原始止盈位已达到，旧计划已完成，建议退出"
    elif _reached_target(direction, current_price, tp1):
        action = "减仓观察"
        reason = "第一止盈位已达到或接近，建议管理仓位"
    elif _stop_improved(direction, current_stop, original_stop, threshold):
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
        "original_plan": {
            "entry": float(recommendation.get("entry") or 0.0),
            "first_stop": original_first_stop,
            "stop": original_stop,
            "tp1": tp1,
            "tp2": tp2,
            "rr": float(recommendation.get("rr") or 0.0),
            "admission_rr": float(recommendation.get("admission_rr") or 0.0),
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
