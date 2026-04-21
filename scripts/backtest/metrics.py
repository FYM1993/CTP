from __future__ import annotations

from backtest.models import TradeRecord


def summarize_trades(trades: list[TradeRecord]) -> dict[str, float | int]:
    if not trades:
        return {
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

    pnls = [float(trade.pnl_ratio) for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0)
    losses = sum(1 for pnl in pnls if pnl < 0)
    total_pnl = sum(pnls)
    win_pnls = [pnl for pnl in pnls if pnl > 0]
    loss_pnls = [pnl for pnl in pnls if pnl < 0]
    stop_trades = [trade for trade in trades if trade.exit_reason == "stop"]
    loss_stop_hits = sum(1 for trade in stop_trades if float(trade.pnl_ratio) < 0)
    protective_stop_hits = sum(1 for trade in stop_trades if float(trade.pnl_ratio) >= 0)

    return {
        "num_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades),
        "avg_pnl": total_pnl / len(trades),
        "total_pnl": total_pnl,
        "avg_win_pnl": sum(win_pnls) / len(win_pnls) if win_pnls else 0.0,
        "avg_loss_pnl": sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0,
        "max_win_pnl": max(win_pnls) if win_pnls else 0.0,
        "max_loss_pnl": min(loss_pnls) if loss_pnls else 0.0,
        "tp1_hits": sum(1 for trade in trades if trade.tp1_hit),
        "tp2_hits": sum(1 for trade in trades if trade.exit_reason == "tp2"),
        "stop_hits": len(stop_trades),
        "loss_stop_hits": loss_stop_hits,
        "protective_stop_hits": protective_stop_hits,
    }
