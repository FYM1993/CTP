from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_trend_backtest_params  # noqa: E402
from backtest.models import BacktestCase, BacktestResult, TradeRecord  # noqa: E402


def test_compare_trend_backtest_params_runs_parameter_grid(monkeypatch, tmp_path: Path) -> None:
    case = BacktestCase(
        case_id="jm0_trend_short_q1_2025",
        symbol="JM0",
        name="焦煤",
        direction="short",
        start_dt=date(2025, 1, 1),
        end_dt=date(2025, 3, 1),
        strategy_family="trend_following",
    )
    seen_cfgs: list[dict] = []

    monkeypatch.setattr(compare_trend_backtest_params, "get_case", lambda case_id: case)
    monkeypatch.setattr(compare_trend_backtest_params, "load_config", lambda: {"pre_market": {"base": True}, "intraday": {}})
    monkeypatch.setattr(
        compare_trend_backtest_params,
        "load_case_frames_with_tqbacktest",
        lambda *, case, config: ("daily-frame", "minute-frame"),
    )

    def fake_run_case_from_frames(**kwargs):
        seen_cfgs.append(dict(kwargs["pre_market_cfg"]))
        return BacktestResult(
            case_id="jm0_trend_short_q1_2025",
            trades=[
                TradeRecord(
                    trade_id="early-stop",
                    symbol="JM0",
                    direction="short",
                    entry_time="2025-01-03 09:00:00",
                    entry_price=100.0,
                    exit_time="2025-01-04 09:00:00",
                    exit_price=103.0,
                    exit_reason="stop",
                    bars_held=20,
                    days_held=1,
                    tp1_hit=False,
                    pnl_ratio=-0.03,
                )
            ],
            summary={},
            diagnostics={"phase3_entry_adverse_deviation_rejects": len(seen_cfgs)},
        )

    monkeypatch.setattr(compare_trend_backtest_params, "run_case_from_frames", fake_run_case_from_frames)
    monkeypatch.setattr(
        compare_trend_backtest_params,
        "summarize_trades",
        lambda trades: {"num_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "avg_pnl": 0.0},
    )
    out = tmp_path / "grid.csv"

    exit_code = compare_trend_backtest_params.main(
        [
            "--case",
            "jm0_trend_short_q1_2025",
            "--trend-min-stop-atr-multiples",
            "legacy,1.5",
            "--trend-max-entry-adverse-r-values",
            "legacy,1.0",
            "--trend-min-first-target-rr-values",
            "1.0",
            "--trend-entry-confirmation-bars-values",
            "2",
            "--trend-initial-stop-grace-bars-values",
            "3",
            "--output",
            str(out),
        ]
    )

    assert exit_code == 0
    assert len(seen_cfgs) == 4
    assert "trend_min_stop_atr_multiple" not in seen_cfgs[0]
    assert seen_cfgs[1]["trend_max_entry_adverse_deviation_r"] == 1.0
    assert seen_cfgs[0]["trend_min_first_target_rr"] == 1.0
    assert seen_cfgs[0]["trend_entry_confirmation_bars"] == 2
    assert seen_cfgs[0]["trend_initial_stop_grace_bars"] == 3
    assert seen_cfgs[2]["trend_min_stop_atr_multiple"] == 1.5
    text = out.read_text()
    assert "trend_min_first_target_rr" in text
    assert "trend_entry_confirmation_bars" in text
    assert "trend_initial_stop_grace_bars" in text
    assert "early_stop_trades" in text
    assert "early_stop_total_pnl" in text
    assert ",1,-0.03," in text
    assert "phase3_entry_adverse_deviation_rejects" in text
