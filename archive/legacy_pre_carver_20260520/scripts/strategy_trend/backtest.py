from __future__ import annotations

from backtest.phase23 import make_trade_plan_from_raw_builder
from phase2.pre_market import build_trend_trade_plan_from_daily_df
from shared.strategy import STRATEGY_TREND


def make_trend_trade_plan(*, case, daily_df, pre_market_cfg):
    return make_trade_plan_from_raw_builder(
        case=case,
        daily_df=daily_df,
        pre_market_cfg=pre_market_cfg,
        raw_builder=build_trend_trade_plan_from_daily_df,
        strategy_family=STRATEGY_TREND,
    )


__all__ = ["make_trend_trade_plan"]
