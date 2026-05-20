from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from systematic_trading.reports import summarize_equity, write_markdown_report  # noqa: E402


def test_summarize_equity_calculates_drawdown_and_return() -> None:
    equity = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "equity": [100.0, 110.0, 99.0],
            "pnl": [0.0, 10.0, -11.0],
            "cost": [0.0, 1.0, 1.0],
            "margin": [0.0, 10.0, 10.0],
            "gross_exposure": [0.0, 100.0, 100.0],
            "net_exposure": [0.0, 100.0, 100.0],
            "num_positions": [0, 1, 1],
        }
    )

    summary = summarize_equity(equity, initial_capital=100.0)

    assert summary.total_return == -0.01
    assert summary.max_drawdown < 0.0
    assert summary.total_cost == 2.0


def test_write_markdown_report_contains_portfolio_metrics(tmp_path: Path) -> None:
    equity = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "equity": [100.0, 101.0],
            "pnl": [0.0, 1.0],
            "cost": [0.0, 0.1],
            "margin": [0.0, 10.0],
            "gross_exposure": [0.0, 100.0],
            "net_exposure": [0.0, 100.0],
            "num_positions": [0, 1],
        }
    )
    path = tmp_path / "report.md"

    write_markdown_report(
        path=path,
        equity=equity,
        trades=pd.DataFrame(),
        positions=pd.DataFrame(),
        initial_capital=100.0,
    )

    text = path.read_text()
    assert "系统化期货组合回测报告" in text
    assert "账户级指标" in text
