from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_account_backtest_runs  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reconcile_account_runs_classifies_trade_and_decision_differences() -> None:
    old_trades = [
        {
            "symbol": "AU0",
            "direction": "long",
            "entry_time": "2025-01-01 09:00:00",
            "lots": 1,
            "actual_exit_time": "2025-01-03 09:00:00",
            "actual_exit_reason": "stop",
            "net_pnl": -100.0,
        },
        {
            "symbol": "AG0",
            "direction": "long",
            "entry_time": "2025-01-02 09:00:00",
            "lots": 2,
            "actual_exit_time": "2025-01-05 09:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": 2000.0,
        },
        {
            "symbol": "AP0",
            "direction": "short",
            "entry_time": "2025-01-03 09:00:00",
            "lots": 1,
            "actual_exit_time": "2025-01-04 09:00:00",
            "actual_exit_reason": "stop",
            "net_pnl": -50.0,
        },
    ]
    new_trades = [
        {
            "symbol": "AU0",
            "direction": "long",
            "entry_time": "2025-01-01 09:00:00",
            "lots": 1,
            "actual_exit_time": "2025-01-06 09:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": 500.0,
        },
        {
            "symbol": "AG0",
            "direction": "long",
            "entry_time": "2025-01-02 09:00:00",
            "lots": 2,
            "actual_exit_time": "2025-01-05 09:00:00",
            "actual_exit_reason": "tp2",
            "net_pnl": 2000.0,
        },
        {
            "symbol": "SC0",
            "direction": "long",
            "entry_time": "2025-01-04 09:00:00",
            "lots": 1,
            "actual_exit_time": "2025-01-05 09:00:00",
            "actual_exit_reason": "stop",
            "net_pnl": -300.0,
        },
    ]
    new_decisions = [
        {"sequence": 1, "symbol": "AU0", "direction": "long", "entry_time": "2025-01-01 09:00:00", "decision": "accepted"},
        {"sequence": 2, "symbol": "AG0", "direction": "long", "entry_time": "2025-01-02 09:00:00", "decision": "accepted"},
        {
            "sequence": 3,
            "symbol": "AP0",
            "direction": "short",
            "entry_time": "2025-01-03 09:00:00",
            "decision": "skipped",
            "skip_reason": "portfolio_margin_full_lower_score",
            "replacement_block_reason": "no_lower_score_position",
        },
        {"sequence": 4, "symbol": "SC0", "direction": "long", "entry_time": "2025-01-04 09:00:00", "decision": "accepted"},
    ]

    rows, summary = compare_account_backtest_runs.reconcile_account_runs(
        old_trades=old_trades,
        new_trades=new_trades,
        new_decisions=new_decisions,
    )

    by_key = {row["trade_key"]: row for row in rows}
    assert summary["old_trades"] == 3
    assert summary["new_trades"] == 3
    assert summary["new_decisions"] == 4
    assert summary["difference_type_counts"]["matched"] == 1
    assert summary["difference_type_counts"]["exit_changed;pnl_changed"] == 1
    assert summary["difference_type_counts"]["old_only_new_skipped"] == 1
    assert summary["difference_type_counts"]["new_only"] == 1
    assert summary["first_difference"]["trade_key"] == "AU0|long|2025-01-01 09:00:00"
    assert by_key["AP0|short|2025-01-03 09:00:00"]["new_decision"] == "skipped"
    assert by_key["AP0|short|2025-01-03 09:00:00"]["new_skip_reason"] == "portfolio_margin_full_lower_score"


def test_compare_account_backtest_runs_cli_writes_csv_and_json(tmp_path: Path) -> None:
    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    decisions_path = tmp_path / "decisions.csv"
    out_csv = tmp_path / "diff.csv"
    out_json = tmp_path / "diff.json"
    _write_csv(
        old_path,
        [
            {
                "symbol": "AP0",
                "direction": "short",
                "entry_time": "2025-01-03 09:00:00",
                "lots": 1,
                "actual_exit_time": "2025-01-04 09:00:00",
                "actual_exit_reason": "stop",
                "net_pnl": -50.0,
            }
        ],
    )
    _write_csv(new_path, [])
    _write_csv(
        decisions_path,
        [
            {
                "sequence": 1,
                "symbol": "AP0",
                "direction": "short",
                "entry_time": "2025-01-03 09:00:00",
                "decision": "skipped",
                "skip_reason": "portfolio_margin_full_lower_score",
            }
        ],
    )

    exit_code = compare_account_backtest_runs.main(
        [
            "--old-trades",
            str(old_path),
            "--new-trades",
            str(new_path),
            "--new-decisions",
            str(decisions_path),
            "--output-csv",
            str(out_csv),
            "--output-json",
            str(out_json),
        ]
    )

    assert exit_code == 0
    assert "old_only_new_skipped" in out_csv.read_text()
    payload = json.loads(out_json.read_text())
    assert payload["summary"]["difference_type_counts"]["old_only_new_skipped"] == 1
