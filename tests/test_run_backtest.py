from __future__ import annotations

from datetime import date
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_backtest  # noqa: E402
from backtest.models import BacktestCase, BacktestResult, TradeRecord  # noqa: E402
from backtest.rolling import RollingBacktestResult, RollingWindow, RollingWindowResult  # noqa: E402


def test_get_case_supports_lh0_short() -> None:
    from backtest.cases import get_case

    case = get_case("lh0_short")

    assert case.case_id == "lh0_short"
    assert case.symbol == "LH0"
    assert case.direction == "short"
    assert case.strategy_family == "trend_following"


def test_main_runs_single_short_case(monkeypatch, capsys):
    from backtest.cases import get_case

    case = get_case("lh0_short")
    trade = TradeRecord(
        trade_id="t-short",
        symbol="LH0",
        direction="short",
        entry_time="2025-01-06 09:31:00",
        entry_price=100.0,
        exit_time="2025-01-06 10:00:00",
        exit_price=94.0,
        exit_reason="tp2",
        bars_held=5,
        days_held=0,
        tp1_hit=True,
        pnl_ratio=0.06,
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"x": 1}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(
            case_id="lh0_short",
            trades=[trade],
            summary={"num_trades": 1},
            diagnostics={
                "phase2_actionable_days": 2,
                "phase2_actionable_reversal_days": 0,
                "phase2_actionable_trend_days": 1,
                "trades_opened_reversal": 0,
                "trades_opened_trend": 1,
            },
        ),
    )
    monkeypatch.setattr(
        run_backtest,
        "summarize_trades",
        lambda trades: {"num_trades": len(trades), "wins": 1, "win_rate": 1.0},
    )

    exit_code = run_backtest.main(["--case", "lh0_short"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "case=lh0_short" in captured
    assert "diag_strategy_family=trend_following" in captured
    assert "num_trades=1" in captured
    assert "diag_phase2_actionable_trend_days=1" in captured
    assert "diag_trades_opened_trend=1" in captured


def test_main_prints_single_case_trade_debug(monkeypatch, capsys):
    from backtest.cases import get_case

    case = get_case("lh0_short")
    trade = TradeRecord(
        trade_id="t-short",
        symbol="LH0",
        direction="short",
        entry_time="2025-06-10 09:31:00",
        entry_price=14000.0,
        exit_time="2025-06-12 13:45:00",
        exit_price=14114.0,
        exit_reason="end_of_data",
        bars_held=20,
        days_held=2,
        tp1_hit=False,
        pnl_ratio=-0.0081,
        meta={
            "plan_date": "2025-06-10",
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "反弹受阻后转弱",
            "phase2_score": -31.0,
            "management_profile": "strategy_trend",
            "management_tp1_exit_fraction": 0.5,
            "management_move_stop_to_entry_after_tp1": True,
        },
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(
            case_id="lh0_short",
            trades=[trade],
            summary={"num_trades": 1},
            diagnostics={},
        ),
    )
    monkeypatch.setattr(run_backtest, "summarize_trades", lambda trades: {"num_trades": len(trades)})

    exit_code = run_backtest.main(["--case", "lh0_short", "--debug-trades"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "debug_trade_1_trade_id=t-short" in captured
    assert "debug_trade_1_entry_family=trend" in captured
    assert "debug_trade_1_entry_signal_type=Pullback" in captured
    assert "debug_trade_1_management_profile=strategy_trend" in captured
    assert "debug_trade_1_management_tp1_exit_fraction=0.5" in captured
    assert "debug_trade_1_management_move_stop_to_entry_after_tp1=True" in captured
    assert "debug_trade_1_exit_reason=end_of_data" in captured

def test_main_runs_single_case_and_prints_summary(monkeypatch, capsys):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
    )
    trade = TradeRecord(
        trade_id="t1",
        symbol="LH0",
        direction="long",
        entry_time="2025-01-06 09:31:00",
        entry_price=100.0,
        exit_time="2025-01-06 10:00:00",
        exit_price=110.0,
        exit_reason="tp2",
        bars_held=5,
        days_held=0,
        tp1_hit=True,
        pnl_ratio=0.1,
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case if case_id == "lh0_long" else None)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"x": 1}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "make_trade_plan_from_phase2",
        lambda *, case, daily_df, pre_market_cfg: object(),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(
            case_id="lh0_long",
            trades=[trade],
            summary={"num_trades": 1},
            diagnostics={
                "phase2_actionable_days": 3,
                "phase2_actionable_reversal_days": 1,
                "phase2_actionable_trend_days": 2,
                "phase3_entry_signal_hits": 1,
                "trades_opened_reversal": 0,
                "trades_opened_trend": 1,
            },
        ),
    )
    monkeypatch.setattr(
        run_backtest,
        "summarize_trades",
        lambda trades: {
            "num_trades": len(trades),
            "wins": 1,
            "win_rate": 1.0,
        },
    )

    exit_code = run_backtest.main(["--case", "lh0_long"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "case=lh0_long" in captured
    assert "num_trades=1" in captured
    assert "wins=1" in captured
    assert "win_rate=1.0" in captured
    assert "diag_phase2_actionable_days=3" in captured
    assert "diag_phase2_actionable_reversal_days=1" in captured
    assert "diag_phase2_actionable_trend_days=2" in captured
    assert "diag_phase3_entry_signal_hits=1" in captured
    assert "diag_trades_opened_reversal=0" in captured
    assert "diag_trades_opened_trend=1" in captured


def test_main_overrides_case_date_range(monkeypatch):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )

    def fake_load_case_frames_with_tqbacktest(*, case, config):
        seen["case"] = case
        return ("daily-frame", "minute-frame")

    monkeypatch.setattr(run_backtest, "load_case_frames_with_tqbacktest", fake_load_case_frames_with_tqbacktest)
    monkeypatch.setattr(
        run_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(case_id="lh0_long", trades=[], summary={"num_trades": 0}, diagnostics={}),
    )
    monkeypatch.setattr(run_backtest, "summarize_trades", lambda trades: {"num_trades": len(trades)})

    exit_code = run_backtest.main(["--case", "lh0_long", "--start", "2025-01-01", "--end", "2025-03-31"])

    assert exit_code == 0
    assert seen["case"].start_dt == date(2025, 1, 1)
    assert seen["case"].end_dt == date(2025, 3, 31)
    assert seen["case"].case_id == "lh0_long_2025-01-01_2025-03-31"


def test_main_defaults_to_strict_backtest_fundamental_mode(monkeypatch):
    case = BacktestCase(
        case_id="lh0_reversal_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
        strategy_family="reversal_fundamental",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )

    def fake_run_case_from_frames(**kwargs):
        seen["pre_market_cfg"] = kwargs["pre_market_cfg"]
        return BacktestResult(case_id="lh0_reversal_long", trades=[], summary={"num_trades": 0}, diagnostics={})

    monkeypatch.setattr(run_backtest, "run_case_from_frames", fake_run_case_from_frames)
    monkeypatch.setattr(run_backtest, "summarize_trades", lambda trades: {"num_trades": len(trades)})

    exit_code = run_backtest.main(["--case", "lh0_reversal_long"])

    assert exit_code == 0
    assert seen["pre_market_cfg"]["backtest_fundamental_mode"] == "strict"


def test_main_allows_proxy_backtest_fundamental_mode(monkeypatch):
    case = BacktestCase(
        case_id="lh0_reversal_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 1, 31),
        strategy_family="reversal_fundamental",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"backtest_fundamental_mode": "strict"}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )

    def fake_run_case_from_frames(**kwargs):
        seen["pre_market_cfg"] = kwargs["pre_market_cfg"]
        return BacktestResult(case_id="lh0_reversal_long", trades=[], summary={"num_trades": 0}, diagnostics={})

    monkeypatch.setattr(run_backtest, "run_case_from_frames", fake_run_case_from_frames)
    monkeypatch.setattr(run_backtest, "summarize_trades", lambda trades: {"num_trades": len(trades)})

    exit_code = run_backtest.main(["--case", "lh0_reversal_long", "--fundamental-mode", "proxy"])

    assert exit_code == 0
    assert seen["pre_market_cfg"]["backtest_fundamental_mode"] == "proxy"


def test_main_rejects_invalid_date_range(monkeypatch):
    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
    ))

    with pytest.raises(SystemExit):
        run_backtest.main(["--case", "lh0_long", "--start", "2025-04-01", "--end", "2025-03-31"])


def test_main_runs_rolling_backtest_and_prints_aggregate_summary(monkeypatch, capsys):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"min_history_bars": 60}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_walk_forward_from_frames",
        lambda **kwargs: RollingBacktestResult(
            base_case_id="lh0_long",
            effective_start=date(2025, 3, 1),
            windows=[],
            aggregate_summary={
                "num_windows": 3,
                "num_windows_with_trades": 1,
                "num_trades": 2,
                "wins": 1,
                "win_rate": 0.5,
                "avg_pnl": 0.01,
                "total_pnl": 0.02,
                "tp1_hits": 1,
                "tp2_hits": 0,
                "stop_hits": 1,
            },
            aggregate_diagnostics={
                "effective_start": "2025-03-01",
                "windows_skipped_no_history": 0,
                "total_phase2_actionable_days": 8,
                "total_phase2_actionable_reversal_days": 3,
                "total_phase2_actionable_trend_days": 5,
                "total_phase2_non_actionable_days": 12,
                "total_phase2_history_insufficient_days": 3,
                "total_phase2_reject_no_signal_days": 5,
                "total_phase2_reject_score_gate_days": 4,
                "total_phase2_reject_rr_gate_days": 2,
                "total_phase2_reject_duplicate_signal_days": 1,
                "total_phase3_signal_eval_bars": 21,
                "total_phase3_entry_signal_hits": 2,
                "total_trades_opened": 2,
                "total_trades_opened_reversal": 1,
                "total_trades_opened_trend": 1,
            },
        ),
    )

    exit_code = run_backtest.main(
        ["--case", "lh0_long", "--rolling", "--train-days", "180", "--validation-days", "60", "--step-days", "60"]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "case=lh0_long" in captured
    assert "num_windows=3" in captured
    assert "num_trades=2" in captured
    assert "diag_effective_start=2025-03-01" in captured
    assert "diag_total_phase2_actionable_days=8" in captured
    assert "diag_total_phase2_actionable_reversal_days=3" in captured
    assert "diag_total_phase2_actionable_trend_days=5" in captured
    assert "diag_total_phase2_reject_no_signal_days=5" in captured
    assert "diag_total_phase3_entry_signal_hits=2" in captured
    assert "diag_total_trades_opened_reversal=1" in captured
    assert "diag_total_trades_opened_trend=1" in captured


def test_main_prints_window_level_rolling_diagnostics(monkeypatch, capsys):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
    )

    window1 = RollingWindow(
        index=1,
        train_start=date(2024, 9, 1),
        train_end=date(2025, 2, 28),
        validation_start=date(2025, 3, 1),
        validation_end=date(2025, 4, 29),
    )
    window2 = RollingWindow(
        index=2,
        train_start=date(2024, 10, 31),
        train_end=date(2025, 4, 29),
        validation_start=date(2025, 4, 30),
        validation_end=date(2025, 6, 28),
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"min_history_bars": 60}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_walk_forward_from_frames",
        lambda **kwargs: RollingBacktestResult(
            base_case_id="lh0_long",
            effective_start=date(2025, 3, 1),
            windows=[
                RollingWindowResult(
                    window=window1,
                    result=BacktestResult(
                        case_id="lh0_long_wf01",
                        trades=[],
                        summary={"num_trades": 0},
                        diagnostics={
                            "phase2_actionable_days": 4,
                            "phase2_actionable_reversal_days": 1,
                            "phase2_actionable_trend_days": 3,
                            "phase2_non_actionable_days": 8,
                            "phase2_history_insufficient_days": 0,
                            "phase2_reject_no_signal_days": 6,
                            "phase2_reject_score_gate_days": 2,
                            "phase2_reject_rr_gate_days": 0,
                            "phase2_reject_duplicate_signal_days": 0,
                            "phase3_signal_eval_bars": 11,
                            "phase3_entry_signal_hits": 1,
                            "trades_opened": 1,
                            "trades_opened_reversal": 0,
                            "trades_opened_trend": 1,
                        },
                    ),
                ),
                RollingWindowResult(
                    window=window2,
                    result=BacktestResult(
                        case_id="lh0_long_wf02",
                        trades=[],
                        summary={"num_trades": 0},
                        diagnostics={
                            "phase2_actionable_days": 0,
                            "phase2_actionable_reversal_days": 0,
                            "phase2_actionable_trend_days": 0,
                            "phase2_non_actionable_days": 15,
                            "phase2_history_insufficient_days": 0,
                            "phase2_reject_no_signal_days": 12,
                            "phase2_reject_score_gate_days": 1,
                            "phase2_reject_rr_gate_days": 2,
                            "phase2_reject_duplicate_signal_days": 0,
                            "phase3_signal_eval_bars": 0,
                            "phase3_entry_signal_hits": 0,
                            "trades_opened": 0,
                            "trades_opened_reversal": 0,
                            "trades_opened_trend": 0,
                        },
                    ),
                ),
            ],
            aggregate_summary={
                "num_windows": 2,
                "num_windows_with_trades": 0,
                "num_trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "tp1_hits": 0,
                "tp2_hits": 0,
                "stop_hits": 0,
            },
            aggregate_diagnostics={
                "effective_start": "2025-03-01",
                "windows_skipped_no_history": 0,
                "total_phase2_actionable_days": 4,
                "total_phase2_actionable_reversal_days": 1,
                "total_phase2_actionable_trend_days": 3,
                "total_phase2_non_actionable_days": 23,
                "total_phase2_history_insufficient_days": 0,
                "total_phase2_reject_no_signal_days": 18,
                "total_phase2_reject_score_gate_days": 3,
                "total_phase2_reject_rr_gate_days": 2,
                "total_phase2_reject_duplicate_signal_days": 0,
                "total_phase3_signal_eval_bars": 11,
                "total_phase3_entry_signal_hits": 1,
                "total_trades_opened": 1,
                "total_trades_opened_reversal": 0,
                "total_trades_opened_trend": 1,
            },
        ),
    )

    exit_code = run_backtest.main(["--case", "lh0_long", "--rolling"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "diag_wf01_validation_start=2025-03-01" in captured
    assert "diag_wf01_validation_end=2025-04-29" in captured
    assert "diag_wf01_phase2_actionable_days=4" in captured
    assert "diag_wf01_phase2_actionable_reversal_days=1" in captured
    assert "diag_wf01_phase2_actionable_trend_days=3" in captured
    assert "diag_wf01_phase2_reject_no_signal_days=6" in captured
    assert "diag_wf01_phase3_entry_signal_hits=1" in captured
    assert "diag_wf02_phase2_non_actionable_days=15" in captured
    assert "diag_wf02_phase2_reject_rr_gate_days=2" in captured
    assert "diag_wf02_trades_opened=0" in captured
    assert "diag_wf01_trades_opened_reversal=0" in captured
    assert "diag_wf01_trades_opened_trend=1" in captured


def test_main_prints_selected_window_phase2_debug(monkeypatch, capsys):
    case = BacktestCase(
        case_id="lh0_long",
        symbol="LH0",
        name="生猪",
        direction="long",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 12, 31),
    )

    window1 = RollingWindow(
        index=1,
        train_start=date(2024, 9, 1),
        train_end=date(2025, 2, 28),
        validation_start=date(2025, 3, 1),
        validation_end=date(2025, 4, 29),
    )
    window2 = RollingWindow(
        index=2,
        train_start=date(2024, 10, 31),
        train_end=date(2025, 4, 29),
        validation_start=date(2025, 4, 30),
        validation_end=date(2025, 6, 28),
    )

    monkeypatch.setattr(run_backtest, "get_case", lambda case_id: case)
    monkeypatch.setattr(
        run_backtest,
        "load_config",
        lambda: {"pre_market": {"min_history_bars": 60}, "intraday": {}, "tqsdk": {"account": "a", "password": "b"}},
    )
    monkeypatch.setattr(
        run_backtest,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )
    monkeypatch.setattr(
        run_backtest,
        "run_walk_forward_from_frames",
        lambda **kwargs: RollingBacktestResult(
            base_case_id="lh0_long",
            effective_start=date(2025, 3, 1),
            windows=[
                RollingWindowResult(
                    window=window1,
                    result=BacktestResult(case_id="wf01", trades=[], summary={"num_trades": 0}, diagnostics={}, debug={}),
                ),
                RollingWindowResult(
                    window=window2,
                    result=BacktestResult(
                        case_id="wf02",
                        trades=[],
                        summary={"num_trades": 0},
                        diagnostics={},
                        debug={
                            "phase2_days": [
                                {
                                    "trade_date": "2025-05-06",
                                    "visible_daily_rows": 80,
                                    "phase2_state": "no_signal",
                                    "score": 12.0,
                                    "rr": 0.0,
                                    "entry_family": "",
                                    "entry_signal_type": "",
                                    "entry_signal_detail": "",
                                    "reversal_has_signal": False,
                                    "reversal_signal_type": "",
                                    "reversal_current_stage": "markdown",
                                    "reversal_next_expected": "等待Spring",
                                    "trend_has_signal": False,
                                    "trend_signal_type": "",
                                    "trend_phase": "markdown",
                                    "trend_phase_ok": False,
                                    "trend_slope_ok": False,
                                    "trend_indicator_ok": False,
                                }
                            ]
                        },
                    ),
                ),
            ],
            aggregate_summary={"num_windows": 2, "num_windows_with_trades": 0, "num_trades": 0, "wins": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0, "tp1_hits": 0, "tp2_hits": 0, "stop_hits": 0},
            aggregate_diagnostics={},
        ),
    )

    exit_code = run_backtest.main(["--case", "lh0_long", "--rolling", "--debug-phase2", "--window-index", "2"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "debug_wf02_phase2_day_1_trade_date=2025-05-06" in captured
    assert "debug_wf02_phase2_day_1_phase2_state=no_signal" in captured
    assert "debug_wf02_phase2_day_1_reversal_current_stage=markdown" in captured
    assert "debug_wf02_phase2_day_1_reversal_next_expected=等待Spring" in captured
