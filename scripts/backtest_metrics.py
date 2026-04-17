from __future__ import annotations

from backtest_models import TradeRecord


def summarize_trades(trades: list[TradeRecord]) -> dict[str, float | int]:
    if not trades:
        return {
            "num_trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "stop_hits": 0,
        }

    pnls = [float(trade.pnl_ratio) for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0)
    total_pnl = sum(pnls)

    return {
        "num_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades),
        "avg_pnl": total_pnl / len(trades),
        "total_pnl": total_pnl,
        "tp1_hits": sum(1 for trade in trades if trade.tp1_hit),
        "tp2_hits": sum(1 for trade in trades if trade.exit_reason == "tp2"),
        "stop_hits": sum(1 for trade in trades if trade.exit_reason == "stop"),
    }
