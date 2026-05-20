from __future__ import annotations

from datetime import date
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_account_backtest  # noqa: E402
from backtest.models import BacktestCase, BacktestResult  # noqa: E402
from backtest.models import TradeRecord  # noqa: E402
from backtest.account_runner import AccountCandidate  # noqa: E402


def test_run_account_backtest_cli_writes_outputs_and_progress(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        run_account_backtest,
        "collect_account_candidates",
        lambda **_kwargs: [
            AccountCandidate(
                symbol="AU0",
                name="黄金",
                direction="long",
                entry_time="2025-01-02 09:00:00",
                planned_exit_time="2025-01-03 09:00:00",
                entry_price=100.0,
                planned_exit_price=110.0,
                planned_exit_reason="tp2",
                phase2_score=65.0,
                risk_per_lot=1_000.0,
                margin_per_lot=10_000.0,
                notional_per_lot=1_000.0,
                multiplier=10.0,
                pnl_ratio_price=0.10,
                tp1_hit=True,
                commission_per_lot=2.0,
                trade_id="AU0-2025-01-02",
                entry_rr=2.2,
                entry_admission_rr=3.4,
                medium_term_quality_score=70.0,
                entry_adverse_deviation_r=0.1,
            )
        ],
    )
    output_prefix = tmp_path / "account"

    exit_code = run_account_backtest.main(
        [
            "--case-group",
            "long_trend_core",
            "--year",
            "2025",
            "--risk-cap",
            "0.015",
            "--ordinary-risk-pct",
            "0.005",
            "--strong-risk-pct",
            "0.012",
            "--strong-first-target-rr",
            "2.0",
            "--strong-second-target-rr",
            "3.0",
            "--split-initial-fraction",
            "0.5",
            "--split-second-entry-trigger",
            "tp1",
            "--execution-policy",
            "conditional",
            "--position-budget-source",
            "phase1_story",
            "--output-prefix",
            str(output_prefix),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "collecting candidates: case_group=long_trend_core year=2025" in captured
    assert "collected candidates: 1" in captured
    assert "account result: accepted=1 skipped=0" in captured
    assert "wrote:" in captured
    summary = json.loads((tmp_path / "account_summary.json").read_text())
    assert summary["scope"] == "long_trend_core continuous 2025, long+short candidates, one active trade per symbol"
    assert summary["accepted_trades"] == 1
    assert summary["ordinary_trade_stop_risk_pct"] == 0.005
    assert summary["strong_trade_stop_risk_pct"] == 0.012
    assert summary["position_budget_source"] == "phase1_story"
    assert summary["execution_policy"] == "conditional"
    assert summary["execution_profile_counts"] == {"direct_full": 1}
    assert summary["split_initial_fraction"] == 0.5
    assert summary["split_second_entry_trigger"] == "tp1"
    assert (tmp_path / "account_decisions.csv").exists()
    decisions = (tmp_path / "account_decisions.csv").read_text()
    assert "strong" in decisions
    assert "0.012" in decisions


def test_run_account_backtest_cli_passes_trend_experiment_overrides(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, dict] = {}

    def fake_collect(**kwargs):
        seen["config"] = dict(kwargs["config"])
        return [], {}

    monkeypatch.setattr(run_account_backtest, "collect_account_candidates", fake_collect)
    output_prefix = tmp_path / "account"

    exit_code = run_account_backtest.main(
        [
            "--case-group",
            "long_trend_core",
            "--year",
            "2025",
            "--trend-min-first-target-rr",
            "1.2",
            "--trend-entry-confirmation-bars",
            "2",
            "--trend-initial-stop-grace-bars",
            "3",
            "--output-prefix",
            str(output_prefix),
            "--quiet",
        ]
    )

    assert exit_code == 0
    pre_market_cfg = seen["config"]["pre_market"]
    assert pre_market_cfg["trend_min_first_target_rr"] == 1.2
    assert pre_market_cfg["trend_entry_confirmation_bars"] == 2
    assert pre_market_cfg["trend_initial_stop_grace_bars"] == 3


def test_collect_account_candidates_replays_only_requested_year(monkeypatch) -> None:
    original_case = BacktestCase(
        case_id="jm0_trend_long_2022_2025",
        symbol="JM0",
        name="焦煤",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    seen: dict[str, BacktestCase] = {}
    daily = pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-01"])})
    minute = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02 09:00:00"]), "close": [100.0]})

    monkeypatch.setattr(run_account_backtest, "get_case_group", lambda group: [original_case])

    def fake_load(*, case, config):
        assert case is original_case
        return daily, minute

    def fake_run(**kwargs):
        seen["case"] = kwargs["case"]
        return BacktestResult(case_id=kwargs["case"].case_id, trades=[], summary={})

    monkeypatch.setattr(run_account_backtest, "load_case_frames_with_tqbacktest", fake_load)
    monkeypatch.setattr(run_account_backtest, "run_case_from_frames", fake_run)

    candidates, _minute_bars = run_account_backtest.collect_account_candidates(
        case_group="long_trend_core",
        year=2025,
        config={"pre_market": {}, "intraday": {}},
    )

    assert candidates == []
    assert seen["case"].case_id == "jm0_trend_long_2022_2025_2025"
    assert seen["case"].start_dt == date(2025, 1, 1)
    assert seen["case"].end_dt == date(2025, 12, 31)


def test_collect_account_candidates_applies_accounting_contract_specs(monkeypatch) -> None:
    case = BacktestCase(
        case_id="bu0_trend_long_2022_2025",
        symbol="BU0",
        name="沥青",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    daily = pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-01"])})
    minute = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02 09:00:00"]), "close": [3704.0]})
    trade = TradeRecord(
        trade_id="BU0-2025-01-02",
        symbol="BU0",
        direction="long",
        entry_time="2025-01-02 09:00:00",
        entry_price=3704.0,
        exit_time="2025-01-03 09:00:00",
        exit_price=3625.5857142857144,
        exit_reason="stop",
        bars_held=10,
        days_held=1,
        tp1_hit=False,
        pnl_ratio=-0.02117016352977473,
        meta={
            "phase2_score": 43.0,
            "contract_multiplier": 10.0,
            "execution_risk_per_lot": 784.142857142856,
            "execution_margin_per_lot": 4444.8,
        },
    )

    monkeypatch.setattr(run_account_backtest, "get_case_group", lambda group: [case])
    monkeypatch.setattr(run_account_backtest, "load_case_frames_with_tqbacktest", lambda **_kwargs: (daily, minute))
    monkeypatch.setattr(
        run_account_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(case_id=kwargs["case"].case_id, trades=[trade], summary={}),
    )

    candidates, _minute_bars = run_account_backtest.collect_account_candidates(
        case_group="long_trend_core",
        year=2025,
        config={"pre_market": {}, "intraday": {}},
    )

    assert len(candidates) == 1
    assert candidates[0].margin_per_lot == 3704.0 * 10.0 * 0.15
    assert candidates[0].commission_rate == 0.00005


def test_collect_account_candidates_enriches_medium_term_quality_from_visible_daily(monkeypatch) -> None:
    case = BacktestCase(
        case_id="au0_trend_long_2022_2025",
        symbol="AU0",
        name="黄金",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-08-25", periods=140, freq="D"),
            "open": [100.0 + idx * 0.5 for idx in range(140)],
            "high": [101.0 + idx * 0.5 for idx in range(140)],
            "low": [99.0 + idx * 0.5 for idx in range(140)],
            "close": [100.0 + idx * 0.5 for idx in range(140)],
            "volume": [1000 + idx for idx in range(140)],
            "oi": [500 + idx for idx in range(140)],
        }
    )
    minute = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-13 09:00:00"]), "close": [170.0]})
    trade = TradeRecord(
        trade_id="AU0-2025-01-13",
        symbol="AU0",
        direction="long",
        entry_time="2025-01-13 09:00:00",
        entry_price=170.0,
        exit_time="2025-01-20 09:00:00",
        exit_price=180.0,
        exit_reason="tp2",
        bars_held=20,
        days_held=7,
        tp1_hit=True,
        pnl_ratio=0.06,
        meta={
            "phase2_score": 65.0,
            "contract_multiplier": 10.0,
            "execution_risk_per_lot": 1_000.0,
            "execution_margin_per_lot": 10_000.0,
            "entry_rr": 1.6,
            "entry_admission_rr": 3.2,
            "trend_phase": "markup",
        },
    )

    monkeypatch.setattr(run_account_backtest, "get_case_group", lambda group: [case])
    monkeypatch.setattr(run_account_backtest, "load_case_frames_with_tqbacktest", lambda **_kwargs: (daily, minute))
    monkeypatch.setattr(
        run_account_backtest,
        "run_case_from_frames",
        lambda **kwargs: BacktestResult(case_id=kwargs["case"].case_id, trades=[trade], summary={}),
    )

    candidates, _minute_bars = run_account_backtest.collect_account_candidates(
        case_group="long_trend_core",
        year=2025,
        config={"pre_market": {}, "intraday": {}},
    )

    assert len(candidates) == 1
    assert candidates[0].medium_term_quality_score >= 60.0
    assert candidates[0].medium_term_entry_location_score > 0.0
