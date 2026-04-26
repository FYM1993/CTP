from __future__ import annotations


def holding_action_category(action: str) -> str:
    if action == "上移止损":
        return "继续持有"
    return action


def holding_primary_action(result: dict) -> str:
    action = str(result.get("action") or "")
    return holding_action_category(action)


def holding_management_action(result: dict) -> str:
    management_action = str(result.get("management_action") or "")
    if management_action:
        return management_action
    if str(result.get("action") or "") == "上移止损":
        return "上移止损"
    return ""


def format_holding_action_text(result: dict) -> str:
    action = holding_primary_action(result)
    if holding_management_action(result) == "上移止损":
        return f"{action}（保护止损上移）"
    return action


def format_holding_log_block(result: dict) -> str:
    size_text = _format_number(result.get("size"))
    entry_text = _format_number(result.get("entry_price"))
    current_text = _format_number(result.get("current_price"))
    lines = [
        f"{result['record_id']} {result['name']} {result['direction_cn']} {size_text}手 开仓均价{entry_text} 当前价{current_text}",
        str(result["original_plan_line"]),
        str(result["current_plan_line"]),
        f"持仓建议: {format_holding_action_text(result)}",
        f"当前状态: {format_holding_status_text(result)}",
        f"建议原因: {result['reason']}",
    ]
    return "\n".join(lines)


def format_plan_line(prefix: str, plan: dict) -> str:
    return (
        f"{prefix}: 入场{_format_number(plan.get('entry'))} "
        f"第一止损{_format_number(plan.get('first_stop'))} "
        f"止损{_format_number(plan.get('stop'))} "
        f"第一止盈{_format_number(plan.get('tp1'))} "
        f"止盈{_format_number(plan.get('tp2'))} "
        f"第一止盈RR {float(plan.get('rr') or 0.0):.2f} "
        f"准入RR {float(plan.get('admission_rr') or 0.0):.2f}"
    )


def format_holding_terminal_alert(result: dict) -> str:
    original_plan = result.get("original_plan") or {}
    current_plan = result.get("current_plan") or {}
    stop_text = _format_number(current_plan.get("stop"))
    tp1_text = _format_number(current_plan.get("tp1"))
    action_text = format_holding_action_text(result)
    reason_text = _terminal_reason_text(result)
    if holding_management_action(result) == "上移止损":
        if reason_text:
            action_text = f"{holding_primary_action(result)}（{reason_text}）"
        else:
            original_stop_text = _format_number(original_plan.get("stop"))
            action_text = f"{holding_primary_action(result)}（保护止损上移 {original_stop_text} -> {stop_text}）"
    elif reason_text:
        action_text = f"{action_text}（{reason_text}）"
    return (
        f"  🔔 {result['record_id']} {result['name']}: {action_text} | "
        f"当前价 {_format_number(result.get('current_price'))} 止损 {stop_text} 第一止盈 {tp1_text}"
    )


def should_emit_terminal_alert(result: dict, previous_actions: dict[str, str] | None) -> bool:
    previous_actions = previous_actions or {}
    previous = previous_actions.get(str(result.get("record_id")))
    current = holding_alert_state_key(result)
    if current and current != previous:
        return True
    return False


def build_holdings_summary_line(results: list[dict]) -> str:
    category_counts = {
        "继续持有": 0,
        "减仓观察": 0,
        "平仓": 0,
    }
    raise_stop_count = 0
    for result in results:
        category = holding_primary_action(result)
        if category in category_counts:
            category_counts[category] += 1
        if holding_management_action(result) == "上移止损":
            raise_stop_count += 1
    return (
        "持仓摘要: "
        f"继续持有{category_counts['继续持有']}（其中保护止损上移{raise_stop_count}） "
        f"减仓观察{category_counts['减仓观察']} "
        f"平仓{category_counts['平仓']}"
    )


def format_holding_status_text(result: dict) -> str:
    action = holding_primary_action(result)
    management_action = holding_management_action(result)
    current_plan = result.get("current_plan") or {}
    stop_text = _format_number(current_plan.get("stop"))
    tp1_text = _format_number(current_plan.get("tp1"))
    reason = str(result.get("reason") or "")

    if action == "继续持有" and management_action == "上移止损":
        prefix = reason.split("，保护止损", 1)[0].strip() or "原有交易故事仍成立"
        return f"{prefix}，保护止损已抬高到{stop_text}，第一止盈继续看{tp1_text}"
    if action == "继续持有":
        return f"原有交易故事仍成立，继续按原计划持有，第一止盈继续看{tp1_text}"
    if action == "减仓观察":
        return "已进入第一兑现区，优先执行减仓管理"
    if action == "平仓":
        if "已达到" in reason and "旧计划已完成" in reason:
            return "原计划已完成，优先执行退出"
        return "原有交易故事已失效，优先执行退出"
    return action


def _terminal_reason_text(result: dict) -> str:
    reason = str(result.get("reason") or "").strip()
    if not reason:
        return ""

    if holding_management_action(result) == "上移止损":
        original_plan = result.get("original_plan") or {}
        current_plan = result.get("current_plan") or {}
        original_stop_text = _format_number(original_plan.get("stop"))
        stop_text = _format_number(current_plan.get("stop"))
        prefix = reason.split("，保护止损", 1)[0].strip()
        if prefix:
            return f"{prefix}；保护止损上移 {original_stop_text} -> {stop_text}"
        return f"保护止损上移 {original_stop_text} -> {stop_text}"

    if "第一止盈位已达到或接近" in reason:
        return "第一止盈位已达到或接近"
    if reason.endswith("，建议退出"):
        return reason[: -len("，建议退出")]
    if reason.endswith("，建议管理仓位"):
        return reason[: -len("，建议管理仓位")]
    return reason


def holding_alert_state_key(result: dict) -> str:
    action = holding_primary_action(result)
    management_action = holding_management_action(result)
    if management_action == "上移止损":
        stop_text = _format_number((result.get("current_plan") or {}).get("stop"))
        return f"{action}|{management_action}|{stop_text}"
    if action in {"减仓观察", "平仓"}:
        return action
    return ""


def _format_number(value: object) -> str:
    number = float(value or 0.0)
    if number.is_integer():
        return f"{int(number)}"
    return f"{number:.2f}"
