from __future__ import annotations

import csv
import importlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.models import BacktestCase, TradePlan  # noqa: E402


def test_forward_metrics_use_directional_returns() -> None:
    validation = importlib.import_module("run_phase2_score_validation")
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 15:00:00",
                    "2025-01-03 09:00:00",
                    "2025-01-03 15:00:00",
                ]
            ),
            "open": [100.0, 101.0, 95.0, 94.0],
            "high": [102.0, 103.0, 96.0, 97.0],
            "low": [99.0, 100.0, 90.0, 92.0],
            "close": [101.0, 102.0, 94.0, 95.0],
            "volume": [1, 1, 1, 1],
        }
    )

    long_metrics = validation._forward_metrics_by_date(minute, direction="long", horizons=(1,))
    short_metrics = validation._forward_metrics_by_date(minute, direction="short", horizons=(1,))

    assert round(long_metrics["2025-01-02"]["forward_1d_return"], 4) == -0.05
    assert round(short_metrics["2025-01-02"]["forward_1d_return"], 4) == 0.05
    assert round(long_metrics["2025-01-02"]["forward_1d_max_favorable_return"], 4) == 0.03
    assert round(short_metrics["2025-01-02"]["forward_1d_max_favorable_return"], 4) == 0.10


def test_summarize_buckets_reports_monotonicity() -> None:
    validation = importlib.import_module("run_phase2_score_validation")
    rows = [
        {
            "score": 22.0,
            "abs_score": 22.0,
            "score_bucket": "20_30",
            "phase2_state": "actionable",
            "phase2_score_gate_passed": True,
            "forward_1d_return": 0.01,
            "forward_1d_max_favorable_return": 0.02,
            "forward_1d_max_adverse_return": -0.01,
        },
        {
            "score": 35.0,
            "abs_score": 35.0,
            "score_bucket": "30_40",
            "phase2_state": "actionable",
            "phase2_score_gate_passed": True,
            "forward_1d_return": 0.02,
            "forward_1d_max_favorable_return": 0.03,
            "forward_1d_max_adverse_return": -0.01,
        },
        {
            "score": 55.0,
            "abs_score": 55.0,
            "score_bucket": "50_plus",
            "phase2_state": "actionable",
            "phase2_score_gate_passed": True,
            "forward_1d_return": 0.05,
            "forward_1d_max_favorable_return": 0.06,
            "forward_1d_max_adverse_return": -0.02,
        },
    ]

    bucket_rows, monotonic_rows = validation.summarize_buckets(rows, horizons=(1,), pools=("actionable",))

    assert [row["score_bucket"] for row in bucket_rows] == ["20_30", "30_40", "50_plus"]
    assert monotonic_rows[0]["adjacent_up_steps"] == 2
    assert monotonic_rows[0]["adjacent_pairs"] == 2
    assert monotonic_rows[0]["monotonic_bucket_pass"] is True
    assert monotonic_rows[0]["spearman_abs_score_forward_return"] == pytest.approx(1.0)


def test_medium_term_quality_scores_persistent_directional_trend() -> None:
    validation = importlib.import_module("run_phase2_score_validation")
    rising_daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-09-01", periods=130, freq="D"),
            "open": [100.0 + idx * 0.6 for idx in range(130)],
            "high": [101.0 + idx * 0.6 for idx in range(130)],
            "low": [99.0 + idx * 0.6 for idx in range(130)],
            "close": [100.0 + idx * 0.6 for idx in range(130)],
            "volume": [1000 + idx for idx in range(130)],
            "oi": [500 + idx for idx in range(130)],
        }
    )
    flat_daily = rising_daily.copy()
    flat_daily["close"] = [100.0 + (idx % 2) * 0.2 for idx in range(130)]
    flat_daily["open"] = flat_daily["close"]
    flat_daily["high"] = flat_daily["close"] + 1.0
    flat_daily["low"] = flat_daily["close"] - 1.0

    strong = validation._medium_term_quality_from_daily(rising_daily, direction="long", trend_phase="markup")
    weak = validation._medium_term_quality_from_daily(flat_daily, direction="long", trend_phase="neutral")

    assert strong["medium_term_quality_status"] == "candidate_v3"
    assert strong["medium_term_quality_score"] > weak["medium_term_quality_score"]
    assert strong["medium_term_quality_score"] >= 60.0
    assert strong["medium_term_momentum_score"] >= 60.0
    assert strong["medium_term_efficiency_score"] >= 60.0
    assert strong["medium_term_trend_state_score"] >= 50.0
    assert strong["medium_term_freshness_score"] >= 50.0
    assert strong["medium_term_regime_score"] >= 40.0
    assert weak["medium_term_quality_score"] <= 45.0


def test_medium_term_quality_penalizes_overextended_entry_location() -> None:
    validation = importlib.import_module("run_phase2_score_validation")
    orderly = pd.DataFrame(
        {
            "date": pd.date_range("2024-09-01", periods=160, freq="D"),
            "open": [100.0 + idx * 0.35 for idx in range(160)],
            "high": [101.0 + idx * 0.35 for idx in range(160)],
            "low": [99.0 + idx * 0.35 for idx in range(160)],
            "close": [100.0 + idx * 0.35 for idx in range(160)],
            "volume": [1000 + idx for idx in range(160)],
            "oi": [500 + idx for idx in range(160)],
        }
    )
    overextended = orderly.copy()
    overextended.loc[150:, "close"] = [float(orderly.iloc[149]["close"]) + 8.0 + idx * 2.5 for idx in range(10)]
    overextended.loc[150:, "open"] = overextended.loc[150:, "close"]
    overextended.loc[150:, "high"] = overextended.loc[150:, "close"] + 1.0
    overextended.loc[150:, "low"] = overextended.loc[150:, "close"] - 1.0

    orderly_score = validation._medium_term_quality_from_daily(orderly, direction="long", trend_phase="markup")
    overextended_score = validation._medium_term_quality_from_daily(
        overextended,
        direction="long",
        trend_phase="markup",
    )

    assert orderly_score["medium_term_quality_status"] == "candidate_v3"
    assert overextended_score["medium_term_quality_status"] == "candidate_v3"
    assert overextended_score["medium_term_quality_score"] < orderly_score["medium_term_quality_score"]
    assert overextended_score["medium_term_entry_location_score"] < orderly_score["medium_term_entry_location_score"]


def test_phase2_score_validation_cli_writes_reports(monkeypatch, tmp_path: Path) -> None:
    validation = importlib.import_module("run_phase2_score_validation")
    case = BacktestCase(
        case_id="au0_trend_long_2022_2025",
        symbol="AU0",
        name="gold",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
            "open": [99.0, 100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1, 1, 1, 1],
            "oi": [1, 1, 1, 1],
        }
    )
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 15:00:00",
                    "2025-01-03 09:00:00",
                    "2025-01-03 15:00:00",
                    "2025-01-04 09:00:00",
                    "2025-01-04 15:00:00",
                ]
            ),
            "open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
            "high": [101.0, 102.0, 102.0, 103.0, 103.0, 104.0],
            "low": [99.0, 100.0, 100.0, 101.0, 101.0, 102.0],
            "close": [100.5, 101.0, 101.5, 102.0, 102.5, 103.0],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )
    scores = iter([15.0, 25.0, 55.0])

    def fake_evaluate(**kwargs):
        score = next(scores)
        plan = None
        rejections = {
            "phase2_reject_no_signal_days": 0,
            "phase2_reject_score_gate_days": 0 if score >= 20.0 else 1,
            "phase2_reject_rr_gate_days": 0,
            "phase2_reject_duplicate_signal_days": 0,
            "phase2_reject_missing_fundamental_days": 0,
        }
        if score >= 20.0:
            plan = TradePlan(
                trade_id=f"AU0-{score}",
                symbol="AU0",
                direction="long",
                plan_date="2025-01-01",
                entry_ref=100.0,
                stop=98.0,
                tp1=103.0,
                tp2=106.0,
                phase2_score=score,
                signal_type="Pullback",
                meta={"entry_family": "trend", "strategy_family": "trend_following"},
            )
        snapshot = {
            "strategy_family": "trend_following",
            "score": score,
            "rr": 1.5,
            "admission_rr": 2.5,
            "risk_pct": 0.012,
            "phase2_score_gate_passed": score >= 20.0,
            "phase2_rr_gate_passed": True,
            "phase2_risk_gate_passed": True,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "",
            "reversal_has_signal": False,
            "trend_has_signal": True,
            "trend_phase": "markdown",
            "trend_phase_ok": True,
            "trend_slope_ok": True,
            "trend_indicator_ok": True,
            "scores": {
                "均线排列": 15.0,
                "MACD": 10.0,
                "RSI": 0.0,
                "布林带": 0.0,
                "动量": 5.0,
                "价格位置": 0.0,
                "Wyckoff阶段": 8.0,
                "量价关系": 10.0,
                "VSA信号": 6.0,
                "持仓信号": 15.0,
            },
        }
        return plan, rejections, snapshot

    monkeypatch.setattr(validation, "load_config", lambda: {"pre_market": {"min_history_bars": 1}, "intraday": {}})
    monkeypatch.setattr(validation, "_resolved_fundamental_mode", lambda _config, cli_mode: cli_mode or "strict")
    monkeypatch.setattr(validation, "_config_with_fundamental_mode", lambda config, _mode: config)
    monkeypatch.setattr(validation, "get_case_group", lambda _group: [case])
    monkeypatch.setattr(validation, "load_case_frames_with_tqbacktest", lambda **_kwargs: (daily, minute))
    monkeypatch.setattr(validation, "_plan_factory_for_case", lambda _case: "plan-factory")
    monkeypatch.setattr(validation, "_evaluate_phase2_plan", fake_evaluate)
    output_prefix = tmp_path / "phase2_validation"
    cache_dir = tmp_path / "phase2_score_cache"

    exit_code = validation.main(
        [
            "--case-group",
            "long_trend_core",
            "--year",
            "2025",
            "--horizons",
            "1",
            "--phase2-score-cache-dir",
            str(cache_dir),
            "--write-missing-phase2-score-cache",
            "--output-prefix",
            str(output_prefix),
            "--quiet",
        ]
    )

    assert exit_code == 0
    details_path = tmp_path / "phase2_validation_details.csv"
    buckets_path = tmp_path / "phase2_validation_buckets.csv"
    monotonicity_path = tmp_path / "phase2_validation_monotonicity.csv"
    components_path = tmp_path / "phase2_validation_components.csv"
    summary_path = tmp_path / "phase2_validation_summary.json"
    assert details_path.exists()
    assert buckets_path.exists()
    assert monotonicity_path.exists()
    assert components_path.exists()
    assert (tmp_path / "phase2_validation.md").exists()

    details = list(csv.DictReader(details_path.open()))
    assert len(details) == 3
    assert details[0]["score_bucket"] == "00_lt20"
    assert details[1]["phase2_state"] == "actionable"
    assert float(details[1]["short_term_price_volume_score"]) == 46.0
    assert float(details[1]["trend_structure_score"]) == 15.0
    assert float(details[1]["trend_phase_score"]) == 8.0
    assert details[1]["trend_structure_filter_passed"] == "True"
    assert float(details[1]["admission_reward_risk"]) == 2.5
    assert float(details[1]["stop_risk_pct"]) == 0.012
    assert details[1]["medium_term_quality_status"] == "candidate_v3"
    assert float(details[1]["medium_term_quality_score"]) >= 0.0
    assert float(details[1]["medium_term_momentum_score"]) >= 0.0
    assert float(details[1]["medium_term_efficiency_score"]) >= 0.0
    assert float(details[1]["medium_term_trend_state_score"]) >= 0.0
    assert float(details[1]["medium_term_freshness_score"]) >= 0.0
    assert float(details[1]["medium_term_regime_score"]) >= 0.0
    component_rows = list(csv.DictReader(components_path.open()))
    assert {row["component"] for row in component_rows} >= {
        "short_term_price_volume_score",
        "trend_structure_score",
        "trend_phase_score",
        "admission_reward_risk",
        "medium_term_quality_score",
    }
    summary = json.loads(summary_path.read_text())
    assert summary["daily_rows"] == 3
    assert summary["signal_rows"] == 3
    assert summary["actionable_rows"] == 2
    assert summary["phase2_score_cache_hits"] == 0
    assert summary["phase2_score_cache_misses"] == 1
    assert summary["medium_term_quality_status"] == "candidate_v3"
    assert summary["component_summary"]

    monkeypatch.setattr(
        validation,
        "_evaluate_phase2_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache should avoid Phase2 replay")),
    )
    cached_output_prefix = tmp_path / "phase2_validation_cached"
    cached_exit_code = validation.main(
        [
            "--case-group",
            "long_trend_core",
            "--year",
            "2025",
            "--horizons",
            "1",
            "--phase2-score-cache-dir",
            str(cache_dir),
            "--require-phase2-score-cache",
            "--output-prefix",
            str(cached_output_prefix),
            "--quiet",
        ]
    )

    assert cached_exit_code == 0
    cached_summary = json.loads((tmp_path / "phase2_validation_cached_summary.json").read_text())
    assert cached_summary["daily_rows"] == 3
    assert cached_summary["phase2_score_cache_hits"] == 1
    assert cached_summary["phase2_score_cache_misses"] == 0
    assert cached_summary["component_summary"]
