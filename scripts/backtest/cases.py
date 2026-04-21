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
