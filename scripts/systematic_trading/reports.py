from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from systematic_trading.contracts import BacktestSummary


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def summarize_equity(equity: pd.DataFrame, initial_capital: float) -> BacktestSummary:
    if equity.empty:
        return BacktestSummary(
            start_date="",
            end_date="",
            initial_capital=float(initial_capital),
            final_equity=float(initial_capital),
            total_return=0.0,
            annual_return=0.0,
            annual_volatility=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            total_cost=0.0,
            turnover=0.0,
            avg_margin_to_equity=0.0,
            max_margin_to_equity=0.0,
        )

    values = equity["equity"].astype(float)
    returns = values.pct_change().fillna(0.0)
    total_return = round(float(values.iloc[-1] / initial_capital - 1.0), 12)
    years = max(len(values) / 252.0, 1 / 252.0)
    annual_return = float((values.iloc[-1] / initial_capital) ** (1 / years) - 1.0)
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    max_drawdown = _max_drawdown(values)
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    margin_to_equity = equity["margin"].astype(float) / values.replace(0.0, np.nan)
    gross_exposure = equity.get("gross_exposure", pd.Series(dtype=float)).astype(float)

    return BacktestSummary(
        start_date=str(equity["date"].iloc[0]),
        end_date=str(equity["date"].iloc[-1]),
        initial_capital=float(initial_capital),
        final_equity=float(values.iloc[-1]),
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=float(sharpe),
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        total_cost=float(equity["cost"].sum()),
        turnover=float(gross_exposure.sum()),
        avg_margin_to_equity=float(margin_to_equity.fillna(0.0).mean()),
        max_margin_to_equity=float(margin_to_equity.fillna(0.0).max()),
    )


def _format_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def write_markdown_report(
    *,
    path: Path,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    initial_capital: float,
) -> None:
    summary = summarize_equity(equity, initial_capital)
    path.parent.mkdir(parents=True, exist_ok=True)
    avg_positions = 0.0 if equity.empty else float(equity["num_positions"].astype(float).mean())
    max_positions = 0 if equity.empty else int(equity["num_positions"].max())

    lines = [
        "# 系统化期货组合回测报告",
        "",
        "## 账户级指标",
        "",
        f"- 起止日期: {summary.start_date} 至 {summary.end_date}",
        f"- 初始资金: {_format_float(summary.initial_capital)}",
        f"- 期末权益: {_format_float(summary.final_equity)}",
        f"- 总收益率: {summary.total_return:.2%}",
        f"- 年化收益率: {summary.annual_return:.2%}",
        f"- 年化波动率: {summary.annual_volatility:.2%}",
        f"- 夏普: {summary.sharpe:.2f}",
        f"- 最大回撤: {summary.max_drawdown:.2%}",
        f"- Calmar: {summary.calmar:.2f}",
        f"- 总成本: {_format_float(summary.total_cost)}",
        f"- 平均保证金占权益: {summary.avg_margin_to_equity:.2%}",
        f"- 最高保证金占权益: {summary.max_margin_to_equity:.2%}",
        "",
        "## 交易行为",
        "",
        f"- 调仓次数: {0 if trades.empty else len(trades)}",
        f"- 持仓记录数: {0 if positions.empty else len(positions)}",
        f"- 平均持仓品种数: {avg_positions:.2f}",
        f"- 最大持仓品种数: {max_positions}",
    ]
    path.write_text("\n".join(lines) + "\n")
