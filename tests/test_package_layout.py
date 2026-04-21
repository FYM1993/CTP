from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_new_functional_packages_are_importable():
    from backtest.phase23 import load_case_frames_with_tqbacktest
    from cli.daily_workflow import phase_1_screen
    from market.tq import symbol_to_tq_main
    from phase1.pipeline import select_top_candidates
    from phase2.direction import choose_phase2_direction
    from phase2.pre_market import build_trade_plan_from_daily_df
    from phase3.intraday import print_dashboard
    from phase3.live import try_create_phase3_monitor
    from shared.ctp_log import get_logger
    from shared.strategy import STRATEGY_REVERSAL
    from strategy_reversal.backtest import make_reversal_trade_plan
    from strategy_reversal.intraday import print_dashboard as print_reversal_dashboard
    from strategy_reversal.pre_market import build_trade_plan_from_daily_df as build_reversal_trade_plan
    from strategy_reversal.screen import select_reversal_candidates
    from strategy_trend.backtest import make_trend_trade_plan
    from strategy_trend.intraday import print_dashboard as print_trend_dashboard
    from strategy_trend.pre_market import build_trade_plan_from_daily_df as build_trend_trade_plan
    from strategy_trend.screen import select_trend_candidates

    assert callable(load_case_frames_with_tqbacktest)
    assert callable(phase_1_screen)
    assert callable(symbol_to_tq_main)
    assert callable(select_top_candidates)
    assert callable(choose_phase2_direction)
    assert callable(build_trade_plan_from_daily_df)
    assert callable(print_dashboard)
    assert callable(try_create_phase3_monitor)
    assert callable(get_logger)
    assert callable(make_reversal_trade_plan)
    assert callable(print_reversal_dashboard)
    assert callable(build_reversal_trade_plan)
    assert callable(select_reversal_candidates)
    assert callable(make_trend_trade_plan)
    assert callable(print_trend_dashboard)
    assert callable(build_trend_trade_plan)
    assert callable(select_trend_candidates)
    assert STRATEGY_REVERSAL == "reversal_fundamental"


def test_scripts_root_does_not_keep_legacy_shims():
    legacy_shims = {
        "backtest_cases.py",
        "backtest_metrics.py",
        "backtest_models.py",
        "backtest_phase23.py",
        "ctp_log.py",
        "fundamental_data.py",
        "fundamental_edb_map.py",
        "intraday.py",
        "market_data_tq.py",
        "phase1_factors.py",
        "phase1_models.py",
        "phase1_pipeline.py",
        "phase1_scoring.py",
        "phase2_direction.py",
        "pre_market.py",
        "tqsdk_live.py",
    }

    root_files = {path.name for path in SCRIPTS.glob("*.py")}
    assert legacy_shims.isdisjoint(root_files)
