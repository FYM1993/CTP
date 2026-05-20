from __future__ import annotations

from dataclasses import replace
from datetime import date

from backtest.models import BacktestCase
from shared.strategy import STRATEGY_REVERSAL, STRATEGY_TREND

BUILTIN_CASES: dict[str, BacktestCase] = {
    "lh0_reversal_long": BacktestCase(
        case_id="lh0_reversal_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family=STRATEGY_REVERSAL,
        note="baseline reversal long case",
    ),
    "lh0_reversal_long_jul2025": BacktestCase(
        case_id="lh0_reversal_long_jul2025",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 7, 1),
        end_dt=date(2025, 7, 31),
        strategy_family=STRATEGY_REVERSAL,
        note="2025-07 低位反弹窗口：压栏、二育补栏导致短期供给收缩，适合回放反转做多入场确认",
    ),
    "lh0_mixed_long_jul2025": BacktestCase(
        case_id="lh0_mixed_long_jul2025",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 7, 1),
        end_dt=date(2025, 7, 31),
        note="2025-07 低位反弹窗口：反转与趋势延续混合评估",
    ),
    "lh0_trend_short": BacktestCase(
        case_id="lh0_trend_short",
        symbol="LH0",
        name="生猪",
        direction="short",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family=STRATEGY_TREND,
        note="baseline trend short case",
    ),
    "ps0_reversal_short": BacktestCase(
        case_id="ps0_reversal_short",
        symbol="PS0",
        name="多晶硅",
        direction="short",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family=STRATEGY_REVERSAL,
        note="baseline reversal short case",
    ),
    "ps0_reversal_long_jul2025": BacktestCase(
        case_id="ps0_reversal_long_jul2025",
        symbol="PS0",
        name="多晶硅",
        direction="long",
        start_dt=date(2025, 6, 25),
        end_dt=date(2025, 7, 25),
        strategy_family=STRATEGY_REVERSAL,
        note="2025-06/07 困境反转窗口：长期低于成本、低开工和小幅去库叠加反内卷预期",
    ),
    "ps0_mixed_long_jul2025": BacktestCase(
        case_id="ps0_mixed_long_jul2025",
        symbol="PS0",
        name="多晶硅",
        direction="long",
        start_dt=date(2025, 6, 25),
        end_dt=date(2025, 7, 25),
        note="2025-06/07 困境反转窗口：反转与趋势延续混合评估",
    ),
    "ps0_trend_long": BacktestCase(
        case_id="ps0_trend_long",
        symbol="PS0",
        name="多晶硅",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family=STRATEGY_TREND,
        note="baseline trend long case",
    ),
    "jm0_reversal_long_jul2025": BacktestCase(
        case_id="jm0_reversal_long_jul2025",
        symbol="JM0",
        name="焦煤",
        direction="long",
        start_dt=date(2025, 6, 4),
        end_dt=date(2025, 7, 24),
        strategy_family=STRATEGY_REVERSAL,
        note="2025-06/07 去库反转窗口：减产与补库共振，供需由宽松转向偏紧",
    ),
    "jm0_mixed_long_jul2025": BacktestCase(
        case_id="jm0_mixed_long_jul2025",
        symbol="JM0",
        name="焦煤",
        direction="long",
        start_dt=date(2025, 6, 4),
        end_dt=date(2025, 7, 24),
        note="2025-06/07 去库反转窗口：反转与趋势延续混合评估",
    ),
    "ao0_reversal_long_jul2024": BacktestCase(
        case_id="ao0_reversal_long_jul2024",
        symbol="AO0",
        name="氧化铝",
        direction="long",
        start_dt=date(2024, 7, 8),
        end_dt=date(2024, 7, 19),
        strategy_family=STRATEGY_REVERSAL,
        note="2024-07 强现实窗口：现货偏紧与低库存压过远端供给宽松预期",
    ),
    "ao0_mixed_long_jul2024": BacktestCase(
        case_id="ao0_mixed_long_jul2024",
        symbol="AO0",
        name="氧化铝",
        direction="long",
        start_dt=date(2024, 7, 8),
        end_dt=date(2024, 7, 19),
        note="2024-07 强现实窗口：反转与趋势延续混合评估",
    ),
}


_EXPANDED_TREND_WINDOW_SPECS: tuple[tuple[str, str, str, date, date, str], ...] = (
    ("jm0_q1_2025", "JM0", "焦煤", date(2025, 1, 1), date(2025, 3, 1), "黑色链一季度趋势窗口"),
    ("jm0_mar_apr_2025", "JM0", "焦煤", date(2025, 3, 2), date(2025, 4, 30), "黑色链春季趋势窗口"),
    ("au0_mar_may_2025", "AU0", "黄金", date(2025, 3, 28), date(2025, 5, 26), "贵金属趋势窗口"),
    ("ag0_aug_oct_2025", "AG0", "白银", date(2025, 8, 29), date(2025, 10, 27), "贵金属趋势窗口"),
    ("bu0_q1_2025", "BU0", "沥青", date(2025, 1, 1), date(2025, 3, 1), "能化趋势窗口"),
    ("fu0_q1_2025", "FU0", "燃料油", date(2025, 1, 1), date(2025, 3, 1), "能化趋势窗口"),
    ("sc0_q1_2025", "SC0", "原油", date(2025, 1, 1), date(2025, 3, 1), "能源趋势窗口"),
    ("ss0_jan_may_2025", "SS0", "不锈钢", date(2025, 1, 1), date(2025, 5, 1), "有色链趋势窗口"),
    ("ao0_may_jun_2025", "AO0", "氧化铝", date(2025, 5, 1), date(2025, 6, 29), "有色链趋势窗口"),
    ("si0_may_jun_2025", "SI0", "工业硅", date(2025, 5, 1), date(2025, 6, 29), "新能源材料趋势窗口"),
    ("sp0_may_jun_2025", "SP0", "纸浆", date(2025, 5, 1), date(2025, 6, 29), "轻工趋势窗口"),
    ("jd0_q1_2025", "JD0", "鸡蛋", date(2025, 1, 1), date(2025, 3, 1), "农产品趋势窗口"),
    ("ap0_aug_oct_2025", "AP0", "苹果", date(2025, 8, 29), date(2025, 10, 27), "农产品趋势窗口"),
    ("sa0_mar_apr_2025", "SA0", "纯碱", date(2025, 3, 2), date(2025, 4, 30), "建材化工趋势窗口"),
    ("cs0_mar_apr_2025", "CS0", "淀粉", date(2025, 3, 2), date(2025, 4, 30), "农产品加工趋势窗口"),
    ("sh0_q1_2025", "SH0", "烧碱", date(2025, 1, 1), date(2025, 3, 1), "化工趋势窗口"),
)


def _expanded_trend_cases() -> tuple[BacktestCase, ...]:
    cases: list[BacktestCase] = []
    for label, symbol, name, start_dt, end_dt, note in _EXPANDED_TREND_WINDOW_SPECS:
        for direction in ("long", "short"):
            cases.append(
                BacktestCase(
                    case_id=f"{symbol.lower()}_trend_{direction}_{label.removeprefix(symbol.lower() + '_')}",
                    symbol=symbol,
                    name=name,
                    direction=direction,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    strategy_family=STRATEGY_TREND,
                    note=f"{note}：{direction}",
                )
            )
    return tuple(cases)


EXPANDED_TREND_CASES = _expanded_trend_cases()
BUILTIN_CASES.update({case.case_id: case for case in EXPANDED_TREND_CASES})


_LONG_TREND_CORE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("JM0", "焦煤", "黑色链长历史趋势样本"),
    ("AU0", "黄金", "贵金属长历史趋势样本"),
    ("AG0", "白银", "贵金属长历史趋势样本"),
    ("BU0", "沥青", "能化长历史趋势样本"),
    ("FU0", "燃料油", "能化长历史趋势样本"),
    ("SC0", "原油", "能源长历史趋势样本"),
    ("SS0", "不锈钢", "有色链长历史趋势样本"),
    ("AP0", "苹果", "农产品长历史趋势样本"),
    ("SA0", "纯碱", "建材化工长历史趋势样本"),
    ("CS0", "淀粉", "农产品加工长历史趋势样本"),
)


def _long_trend_core_cases() -> tuple[BacktestCase, ...]:
    cases: list[BacktestCase] = []
    for symbol, name, note in _LONG_TREND_CORE_SPECS:
        for direction in ("long", "short"):
            cases.append(
                BacktestCase(
                    case_id=f"{symbol.lower()}_trend_{direction}_2022_2025",
                    symbol=symbol,
                    name=name,
                    direction=direction,
                    start_dt=date(2022, 1, 1),
                    end_dt=date(2025, 12, 31),
                    strategy_family=STRATEGY_TREND,
                    note=f"{note}：2022-2025 {direction}",
                )
            )
    return tuple(cases)


LONG_TREND_CORE_CASES = _long_trend_core_cases()
BUILTIN_CASES.update({case.case_id: case for case in LONG_TREND_CORE_CASES})

CASE_GROUPS: dict[str, tuple[str, ...]] = {
    "expanded_trend_cached": tuple(case.case_id for case in EXPANDED_TREND_CASES),
    "long_trend_core": tuple(case.case_id for case in LONG_TREND_CORE_CASES),
    "trend_cached": (
        "lh0_trend_short",
        "ps0_trend_long",
        *(case.case_id for case in EXPANDED_TREND_CASES),
    ),
}

CASE_ALIASES = {
    "lh0_long": "lh0_reversal_long",
    "lh0_short": "lh0_trend_short",
    "ps0_long": "ps0_trend_long",
    "ps0_short": "ps0_reversal_short",
}


def get_case(case_id: str) -> BacktestCase:
    resolved_case_id = CASE_ALIASES.get(case_id, case_id)
    case = BUILTIN_CASES[resolved_case_id]
    if resolved_case_id == case_id:
        return case
    return replace(case, case_id=case_id)


def get_case_group(group_id: str) -> list[BacktestCase]:
    return [get_case(case_id) for case_id in CASE_GROUPS[group_id]]
