from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_backtest  # noqa: E402
from backtest_models import BacktestCase, BacktestResult, TradeRecord  # noqa: E402


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
        lambda **kwargs: BacktestResult(case_id="lh0_long", trades=[trade], summary={"num_trades": 1}),
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
