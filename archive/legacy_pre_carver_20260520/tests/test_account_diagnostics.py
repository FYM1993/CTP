from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.account_diagnostics import (  # noqa: E402
    analyze_account_report,
    classify_trend_response,
    load_risk_sensitivity,
    render_markdown,
)


def _trade(
    *,
    symbol: str,
    name: str,
    score: float,
    net_pnl: float,
    exit_reason: str = "stop",
    tp1_hit: bool = False,
    entry_time: str = "2025-01-01 10:00:00",
    actual_exit_time: str = "2025-01-03 10:00:00",
    margin_used: float = 50_000.0,
    entry_equity: float = 1_000_000.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": name,
        "direction": "long",
        "entry_time": entry_time,
        "entry_price": 100.0,
        "actual_exit_time": actual_exit_time,
        "actual_exit_price": 105.0,
        "actual_exit_reason": exit_reason,
        "phase2_score": score,
        "phase2_abs_score": abs(score),
        "lots": 1,
        "risk_per_lot": 10_000.0,
        "margin_per_lot": margin_used,
        "entry_equity": entry_equity,
        "margin_used": margin_used,
        "initial_stop_risk": 10_000.0,
        "initial_stop_risk_pct_equity": 0.01,
        "tp1_hit": str(tp1_hit),
        "gross_pnl": net_pnl + 10.0,
        "fees": 10.0,
        "net_pnl": net_pnl,
    }


def test_analyze_account_report_keeps_symbol_as_diagnostic_slice_not_blacklist() -> None:
    analysis = analyze_account_report(
        [
            {
                **_trade(symbol="AG0", name="白银", score=32.0, net_pnl=20_000.0, exit_reason="tp2"),
                "target_margin_pct": 0.2,
                "margin_lots_cap": 6,
                "risk_lots_cap": 2,
            },
            _trade(symbol="AG0", name="白银", score=38.0, net_pnl=-5_000.0),
            _trade(symbol="CS0", name="淀粉", score=58.0, net_pnl=-8_000.0),
        ],
        risk_summary={
            "initial_equity": 1_000_000.0,
            "return_pct": 0.007,
            "max_drawdown_realized_pct": 0.02,
            "candidate_trades": 5,
            "accepted_trades": 3,
            "skipped_trades": 2,
            "skipped_reasons": {"single_trade_stop_risk_below_one_lot": 1},
        },
    )

    assert analysis["policy"]["symbol_is_diagnostic_only"] is True
    assert analysis["portfolio"]["candidate_trades"] == 5
    assert analysis["portfolio"]["accepted_trades"] == 3
    assert analysis["portfolio"]["return_drawdown_ratio"] == pytest.approx(0.35)
    assert analysis["by_symbol"][0]["symbol"] == "AG0"
    assert analysis["by_symbol"][0]["net_pnl"] == pytest.approx(15_000.0)
    assert analysis["by_symbol"][1]["symbol"] == "CS0"
    assert analysis["by_symbol"][1]["main_exit_reason"] == "stop"
    assert analysis["score_quality"]["is_monotonic_by_avg_net_pnl"] is False
    assert "score_net_pnl_correlation" in analysis["score_quality"]
    assert "score_r_multiple_correlation" in analysis["score_quality"]
    assert analysis["sizing_limits"][0]["limited_by"] == "stop_risk"
    assert analysis["sizing_limits"][0]["trades"] == 1
    assert "planned_entry_ref" in analysis["audit_gaps"]["missing_trade_columns"]
    assert "initial_tp2_price" in analysis["audit_gaps"]["missing_trade_columns"]
    assert "slippage" in analysis["audit_gaps"]["missing_trade_columns"]


def test_analyze_account_report_estimates_round_trip_slippage_cost() -> None:
    trade = _trade(symbol="AG0", name="白银", score=32.0, net_pnl=20_000.0)
    trade["notional"] = 100_000.0

    analysis = analyze_account_report([trade], slippage_bps_per_side=1.0)

    assert analysis["costs"]["fees"] == pytest.approx(10.0)
    assert analysis["costs"]["estimated_slippage"] == pytest.approx(20.0)
    assert analysis["costs"]["net_pnl_after_slippage"] == pytest.approx(19_980.0)


def test_render_markdown_explains_no_symbol_blacklist_policy() -> None:
    analysis = analyze_account_report(
        [_trade(symbol="CS0", name="淀粉", score=58.0, net_pnl=-8_000.0)],
        risk_summary={"initial_equity": 1_000_000.0, "max_drawdown_realized_pct": 0.02},
    )

    markdown = render_markdown(analysis)

    assert "不做品种黑名单" in markdown
    assert "趋势机会与策略响应" in markdown
    assert "得分分组" in markdown
    assert "持仓分组" in markdown
    assert "退出原因" in markdown
    assert "资金管理跳过" in markdown
    assert "成本侵蚀" in markdown
    assert "淀粉" in markdown


def test_classify_trend_response_attributes_failures_to_strategy_or_account_constraints() -> None:
    assert classify_trend_response(trend_available=False, trade_opened=True) == "false_positive_signal"
    assert classify_trend_response(trend_available=True, trade_opened=False, suggested_lots=0) == "account_constraint_zero_lot"
    assert classify_trend_response(trend_available=True, trade_opened=False, suggested_lots=2) == "missed_trend_signal"
    assert (
        classify_trend_response(
            trend_available=True,
            trade_opened=True,
            exit_reason="stop",
            trend_continued_after_exit=True,
        )
        == "entry_or_stop_problem"
    )


def test_analyze_account_report_flags_stopped_trade_when_trend_continues_after_exit() -> None:
    trade = _trade(
        symbol="AG0",
        name="白银",
        score=40.0,
        net_pnl=-1_000.0,
        exit_reason="stop",
        entry_time="2025-01-01 10:00:00",
        actual_exit_time="2025-01-02 10:00:00",
    )
    trade["entry_price"] = 100.0
    trade["actual_exit_price"] = 95.0
    trade["risk_per_lot"] = 5.0
    trade["notional_per_lot"] = 100.0
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-03", "2025-01-04"]),
            "high": [97.0, 101.0],
            "low": [94.0, 96.0],
            "close": [96.0, 100.0],
        }
    )

    analysis = analyze_account_report([trade], daily_bars_by_symbol={"AG0": daily}, trend_lookahead_days=5)

    assert analysis["trend_response"]["overall"][0]["response"] == "entry_or_stop_problem"
    assert analysis["trend_response"]["by_symbol"][0]["symbol"] == "AG0"
    assert analysis["trend_response"]["by_symbol"][0]["entry_or_stop_problem"] == 1


def test_load_risk_sensitivity_selects_requested_cap(tmp_path: Path) -> None:
    path = tmp_path / "risk.json"
    path.write_text(
        json.dumps(
            [
                {"risk_cap": 0.01, "return_pct": 0.1},
                {"risk_cap": 0.015, "return_pct": 0.2, "skipped_reasons": {"portfolio_margin_full": 3}},
            ]
        )
    )

    selected = load_risk_sensitivity(path, risk_cap=0.015)

    assert selected["return_pct"] == pytest.approx(0.2)
    assert selected["skipped_reasons"] == {"portfolio_margin_full": 3}


def test_analyze_account_backtest_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    import analyze_account_backtest  # noqa: E402

    trades_path = tmp_path / "trades.csv"
    trades_path.write_text(
        "\n".join(
            [
                "symbol,name,direction,entry_time,entry_price,actual_exit_time,actual_exit_price,actual_exit_reason,phase2_score,phase2_abs_score,lots,risk_per_lot,margin_per_lot,entry_equity,margin_used,initial_stop_risk,initial_stop_risk_pct_equity,tp1_hit,gross_pnl,fees,net_pnl",
                "AG0,白银,long,2025-01-01 10:00:00,100,2025-01-03 10:00:00,110,tp2,32,32,1,10000,50000,1000000,50000,10000,0.01,true,20010,10,20000",
            ]
        )
        + "\n"
    )
    risk_path = tmp_path / "risk.json"
    risk_path.write_text(json.dumps([{"risk_cap": 0.015, "initial_equity": 1_000_000.0, "return_pct": 0.02}]))
    json_out = tmp_path / "diagnostics.json"
    md_out = tmp_path / "diagnostics.md"

    exit_code = analyze_account_backtest.main(
        [
            "--trades",
            str(trades_path),
            "--risk-sensitivity",
            str(risk_path),
            "--risk-cap",
            "0.015",
            "--slippage-bps-per-side",
            "1.0",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_out.read_text())
    assert payload["portfolio"]["accepted_trades"] == 1
    assert payload["costs"]["slippage_bps_per_side"] == pytest.approx(1.0)
    assert "不做品种黑名单" in md_out.read_text()
