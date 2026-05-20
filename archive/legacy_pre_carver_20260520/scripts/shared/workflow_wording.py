"""工作流用户可见文案的统一入口。"""


def minute_signal_section_title() -> str:
    """分钟面板里用于提示短线辅助判断的标题。"""
    return "分钟级执行观察"


def minute_signal_section_header() -> str:
    return f"│ ⚡ {minute_signal_section_title()}:"


def minute_signal_section_note() -> str:
    """分钟面板里说明其只承担执行辅助角色。"""
    return "│    仅作执行辅助观察，不单独改变日线级计划"


def minute_no_signal_line() -> str:
    """分钟面板未出现新增辅助信号时的固定文案。"""
    return "│ 🔇 分钟级无新增执行观察，日线计划不变"


def minute_signal_display_label(signal_type: str) -> str:
    """分钟级提示的用户可见标签，避免误读成独立主结论。"""
    mapping = {
        "开多": "执行辅助-偏多",
        "加多": "执行辅助-偏多增强",
        "开空": "执行辅助-偏空",
        "加空": "执行辅助-偏空增强",
        "平多": "执行辅助-保护",
        "平空": "执行辅助-保护",
    }
    return mapping.get(signal_type, f"执行辅助-{signal_type}")


def daily_entry_plan_label() -> str:
    """日线级原始/重评交易计划的统一标签。"""
    return "日线级入场计划"


def watchlist_update_header() -> str:
    """等待确认品种在盘中重评时的统一分节标题。"""
    return "盘中日线级计划更新"


def phase3_terminal_scope(log_path: str) -> str:
    """说明终端只打印哪些高优先级提醒，其余详见日志。"""
    return (
        "盘中监控：终端仅打印等待确认品种变化和持仓调整提醒；"
        f"其余见 {log_path}"
    )
