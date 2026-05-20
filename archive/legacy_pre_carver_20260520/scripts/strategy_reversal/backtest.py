from __future__ import annotations

from backtest.phase23 import make_trade_plan_from_raw_builder
from phase2.pre_market import build_reversal_trade_plan_from_daily_df
from shared.strategy import STRATEGY_REVERSAL


def make_reversal_trade_plan(*, case, daily_df, pre_market_cfg):
    return make_trade_plan_from_raw_builder(
        case=case,
        daily_df=daily_df,
        pre_market_cfg=pre_market_cfg,
        raw_builder=build_reversal_trade_plan_from_daily_df,
        strategy_family=STRATEGY_REVERSAL,
    )


__all__ = ["make_reversal_trade_plan"]
