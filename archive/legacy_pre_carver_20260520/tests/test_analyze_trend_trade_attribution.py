from __future__ import annotations

from scripts.analyze_trend_trade_attribution import analyze_run_pair


def test_analyze_run_pair_reconciles_filter_and_redeployment_effects() -> None:
    legacy_trades = [
        {
            "symbol": "A0",
            "direction": "long",
            "entry_time": "2024-01-02 01:00:00",
            "actual_exit_time": "2024-01-05 01:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": "100",
            "lots": "1",
            "entry_rr": "0.8",
            "entry_admission_rr": "3.0",
        },
        {
            "symbol": "B0",
            "direction": "short",
            "entry_time": "2024-01-03 01:00:00",
            "actual_exit_time": "2024-01-03 13:00:00",
            "actual_exit_reason": "stop",
            "net_pnl": "-40",
            "lots": "1",
            "entry_rr": "0.7",
            "entry_admission_rr": "1.2",
        },
        {
            "symbol": "C0",
            "direction": "long",
            "entry_time": "2024-01-04 01:00:00",
            "actual_exit_time": "2024-01-10 01:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": "20",
            "lots": "1",
            "entry_rr": "2.0",
            "entry_admission_rr": "4.0",
        },
    ]
    strict_trades = [
        {
            "symbol": "C0",
            "direction": "long",
            "entry_time": "2024-01-04 01:00:00",
            "actual_exit_time": "2024-01-10 01:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": "35",
            "lots": "2",
            "entry_rr": "2.0",
            "entry_admission_rr": "4.0",
        },
        {
            "symbol": "D0",
            "direction": "long",
            "entry_time": "2024-01-06 01:00:00",
            "actual_exit_time": "2024-01-08 01:00:00",
            "actual_exit_reason": "stop",
            "net_pnl": "-10",
            "lots": "1",
            "entry_rr": "1.8",
            "entry_admission_rr": "2.5",
        },
    ]
    strict_decisions = [
        {
            "symbol": "A0",
            "direction": "long",
            "entry_time": "2024-01-02 01:00:00",
            "decision": "skipped",
            "skip_reason": "phase2_rr_gate",
        },
        {
            "symbol": "B0",
            "direction": "short",
            "entry_time": "2024-01-03 01:00:00",
            "decision": "skipped",
            "skip_reason": "phase2_rr_gate",
        },
    ]

    detail, summary = analyze_run_pair(
        year=2024,
        legacy_trades=legacy_trades,
        strict_trades=strict_trades,
        strict_decisions=strict_decisions,
    )

    assert summary["legacy_only_winner_trades"] == 1
    assert summary["legacy_only_winner_net_pnl"] == 100.0
    assert summary["legacy_only_loser_trades"] == 1
    assert summary["legacy_only_loser_net_pnl"] == -40.0
    assert summary["strict_only_trades"] == 1
    assert summary["strict_only_net_pnl"] == -10.0
    assert summary["common_net_pnl_delta"] == 15.0
    assert summary["strict_minus_legacy_net_pnl_reconciled"] == -55.0
    assert summary["low_first_high_second_trades"] == 1
    assert summary["low_first_high_second_winner_trades"] == 1
    assert {row["attribution"] for row in detail} >= {"低RR错杀趋势机会", "低RR过滤有效", "资金释放后新增亏损"}
