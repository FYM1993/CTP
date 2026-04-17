from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_metrics import summarize_trades  # noqa: E402
from backtest_models import TradeRecord  # noqa: E402


def test_summarize_trades_counts_wins_losses_and_exit_reasons():
    trades = [
        TradeRecord(
            trade_id="t1",
            symbol="LH0",
            direction="long",
            entry_time="2025-01-01 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-02 10:00:00",
            exit_price=110.0,
            exit_reason="tp2",
            bars_held=30,
            days_held=1,
            tp1_hit=True,
            pnl_ratio=0.10,
        ),
        TradeRecord(
            trade_id="t2",
            symbol="PS0",
            direction="short",
            entry_time="2025-01-03 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-04 10:00:00",
            exit_price=103.0,
            exit_reason="stop",
            bars_held=20,
            days_held=1,
            tp1_hit=False,
            pnl_ratio=-0.03,
        ),
    ]

    summary = summarize_trades(trades)

    assert summary["num_trades"] == 2
    assert summary["wins"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["avg_pnl"] == 0.035
    assert summary["total_pnl"] == 0.07
    assert summary["tp1_hits"] == 1
    assert summary["tp2_hits"] == 1
    assert summary["stop_hits"] == 1


def test_summarize_trades_returns_zeroes_for_empty_input():
    summary = summarize_trades([])

    assert summary == {
        "num_trades": 0,
        "wins": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "total_pnl": 0.0,
        "tp1_hits": 0,
        "tp2_hits": 0,
        "stop_hits": 0,
    }
