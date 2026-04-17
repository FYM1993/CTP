from __future__ import annotations

from datetime import date

from backtest_models import BacktestCase

BUILTIN_CASES: dict[str, BacktestCase] = {
    "lh0_long": BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        note="baseline long case",
    ),
    "ps0_short": BacktestCase(
        case_id="ps0_short",
        symbol="PS0",
        name="多晶硅",
        direction="short",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
        note="baseline short case",
    ),
}


def get_case(case_id: str) -> BacktestCase:
    return BUILTIN_CASES[case_id]
