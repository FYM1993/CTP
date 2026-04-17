from __future__ import annotations

from backtest_models import BacktestCase

BACKTEST_CASES: dict[str, BacktestCase] = {
    "lh0_long": BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="baseline long case",
        direction="long",
        start_dt="",
        end_dt="",
        note="baseline long case",
    ),
    "ps0_short": BacktestCase(
        case_id="ps0_short",
        symbol="PS0",
        name="baseline short case",
        direction="short",
        start_dt="",
        end_dt="",
        note="baseline short case",
    ),
}


def get_case(case_id: str) -> BacktestCase:
    return BACKTEST_CASES[case_id]
