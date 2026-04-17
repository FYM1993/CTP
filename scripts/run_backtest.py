from __future__ import annotations

import argparse

from backtest_cases import get_case
from backtest_metrics import summarize_trades
from backtest_phase23 import (
    load_case_frames_with_tqbacktest,
    make_trade_plan_from_phase2,
    run_case_from_frames,
)
from data_cache import _load_config as load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal Phase 2/3 backtest case")
    parser.add_argument("--case", required=True, help="Built-in backtest case id, e.g. lh0_long")
    return parser


def _print_summary(case_id: str, summary: dict[str, float | int]) -> None:
    print(f"case={case_id}")
    for key, value in summary.items():
        print(f"{key}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    case = get_case(args.case)
    config = load_config()

    daily_df, minute_df = load_case_frames_with_tqbacktest(case=case, config=config)
    result = run_case_from_frames(
        case=case,
        daily_df=daily_df,
        minute_df=minute_df,
        plan_factory=make_trade_plan_from_phase2,
        pre_market_cfg=config.get("pre_market") or {},
        signal_cfg=config.get("intraday") or {},
    )
    summary = summarize_trades(result.trades)
    _print_summary(case.case_id, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
