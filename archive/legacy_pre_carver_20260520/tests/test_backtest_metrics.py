from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.metrics import summarize_trades  # noqa: E402
from backtest.models import TradeRecord  # noqa: E402


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
        TradeRecord(
            trade_id="t3",
            symbol="M0",
            direction="long",
            entry_time="2025-01-05 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-06 10:00:00",
            exit_price=99.0,
            exit_reason="stop",
            bars_held=10,
            days_held=1,
            tp1_hit=True,
            pnl_ratio=-0.01,
        ),
        TradeRecord(
            trade_id="t4",
            symbol="P0",
            direction="short",
            entry_time="2025-01-07 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-08 10:00:00",
            exit_price=98.0,
            exit_reason="phase2_invalidated",
            bars_held=12,
            days_held=1,
            tp1_hit=False,
            pnl_ratio=0.02,
        ),
    ]

    summary = summarize_trades(trades)

    assert summary["num_trades"] == 4
    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["avg_pnl"] == pytest.approx(0.02)
    assert summary["total_pnl"] == pytest.approx(0.08)
    assert summary["avg_win_pnl"] == pytest.approx(0.06)
    assert summary["avg_loss_pnl"] == pytest.approx(-0.02)
    assert summary["max_win_pnl"] == pytest.approx(0.10)
    assert summary["max_loss_pnl"] == pytest.approx(-0.03)
    assert summary["tp1_hits"] == 2
    assert summary["tp2_hits"] == 1
    assert summary["stop_hits"] == 2
    assert summary["loss_stop_hits"] == 2
    assert summary["protective_stop_hits"] == 0


def test_summarize_trades_splits_loss_stops_from_protective_stops():
    trades = [
        TradeRecord(
            trade_id="t1",
            symbol="PS0",
            direction="long",
            entry_time="2025-01-01 10:00:00",
            entry_price=100.0,
            exit_time="2025-01-02 10:00:00",
            exit_price=104.0,
            exit_reason="stop",
            bars_held=10,
            days_held=1,
            tp1_hit=True,
            pnl_ratio=0.04,
        )
    ]

    summary = summarize_trades(trades)

    assert summary["num_trades"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["stop_hits"] == 1
    assert summary["loss_stop_hits"] == 0
    assert summary["protective_stop_hits"] == 1


def test_summarize_trades_returns_zeroes_for_empty_input():
    summary = summarize_trades([])

    assert summary == {
        "num_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "total_pnl": 0.0,
        "avg_win_pnl": 0.0,
        "avg_loss_pnl": 0.0,
        "max_win_pnl": 0.0,
        "max_loss_pnl": 0.0,
        "tp1_hits": 0,
        "tp2_hits": 0,
        "stop_hits": 0,
        "loss_stop_hits": 0,
        "protective_stop_hits": 0,
    }
