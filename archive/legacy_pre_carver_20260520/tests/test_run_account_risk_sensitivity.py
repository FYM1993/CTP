from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_account_risk_sensitivity  # noqa: E402
from backtest.account_runner import AccountCandidate  # noqa: E402


def test_run_account_risk_sensitivity_cli_writes_summary_by_symbol_and_detail_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate = AccountCandidate(
        symbol="AU0",
        name="黄金",
        direction="long",
        entry_time="2025-01-02 09:00:00",
        planned_exit_time="2025-01-03 09:00:00",
        entry_price=100.0,
        planned_exit_price=110.0,
        planned_exit_reason="tp2",
        phase2_score=65.0,
        risk_per_lot=2_000.0,
        margin_per_lot=10_000.0,
        notional_per_lot=1_000.0,
        multiplier=10.0,
        pnl_ratio_price=0.10,
        tp1_hit=True,
        commission_per_lot=2.0,
        trade_id="AU0-2025-01-02",
    )
    monkeypatch.setattr(run_account_risk_sensitivity, "load_config", lambda: {"pre_market": {}, "intraday": {}})
    monkeypatch.setattr(run_account_risk_sensitivity, "_resolved_fundamental_mode", lambda config, mode: None)
    monkeypatch.setattr(run_account_risk_sensitivity, "_config_with_fundamental_mode", lambda config, mode: config)
    monkeypatch.setattr(
        run_account_risk_sensitivity,
        "collect_account_candidates",
        lambda **_kwargs: ([candidate], {}),
    )
    output_prefix = tmp_path / "account"

    exit_code = run_account_risk_sensitivity.main(
        [
            "--year",
            "2025",
            "--initial-equity",
            "100000",
            "--risk-caps",
            "0.01,none",
            "--output-prefix",
            str(output_prefix),
            "--quiet",
        ]
    )

    assert exit_code == 0
    summary_rows = json.loads((tmp_path / "account_risk_sensitivity.json").read_text())
    assert [row["risk_cap"] for row in summary_rows] == [0.01, "none"]
    assert summary_rows[0]["accepted_trades"] == 0
    assert summary_rows[1]["accepted_trades"] == 1
    assert (tmp_path / "account_risk_sensitivity.csv").exists()
    assert "none,AU0" in (tmp_path / "account_risk_sensitivity_by_symbol.csv").read_text()
    assert (tmp_path / "account_risk_0p01_trades.csv").exists()
    assert (tmp_path / "account_risk_none_trades.csv").exists()
