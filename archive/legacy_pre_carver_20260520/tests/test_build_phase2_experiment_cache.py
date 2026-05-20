from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_phase2_experiment_cache  # noqa: E402
from backtest.models import BacktestCase, TradePlan  # noqa: E402


def test_build_phase2_experiment_cache_cli_writes_cache_and_summary(monkeypatch, tmp_path: Path) -> None:
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
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1100],
            "oi": [500, 520],
        }
    )
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00"]),
            "open": [101.0],
            "high": [102.0],
            "low": [100.0],
            "close": [101.0],
            "volume": [10],
        }
    )

    def fake_plan_factory(*, case, daily_df, pre_market_cfg):
        entry = float(daily_df.iloc[-1]["close"])
        plan_date = str(pd.to_datetime(daily_df["date"]).max().date())
        return TradePlan(
            trade_id=f"{case.case_id}_{plan_date}",
            symbol=case.symbol,
            direction=case.direction,
            plan_date=plan_date,
            entry_ref=entry,
            stop=entry - 2.0,
            tp1=entry + 3.0,
            tp2=entry + 6.0,
            phase2_score=66.0,
            signal_type="TrendPullback",
            meta={"entry_family": "trend", "strategy_family": "trend_following"},
        )

    monkeypatch.setattr(build_phase2_experiment_cache, "load_config", lambda: {"pre_market": {"min_history_bars": 1}})
    monkeypatch.setattr(build_phase2_experiment_cache, "_resolved_fundamental_mode", lambda _config, cli_mode: cli_mode or "strict")
    monkeypatch.setattr(build_phase2_experiment_cache, "_config_with_fundamental_mode", lambda config, _mode: config)
    monkeypatch.setattr(build_phase2_experiment_cache, "get_case_group", lambda _group: [case])
    monkeypatch.setattr(build_phase2_experiment_cache, "load_case_frames_with_tqbacktest", lambda **_kwargs: (daily, minute))
    monkeypatch.setattr(build_phase2_experiment_cache, "_plan_factory_for_case", lambda _case: fake_plan_factory)

    exit_code = build_phase2_experiment_cache.main(
        [
            "--case-group",
            "long_trend_core",
            "--case-limit",
            "1",
            "--year",
            "2025",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    cache_path = tmp_path / "au0_trend_long_2022_2025_2025_phase2_cache.json"
    summary_path = tmp_path / "phase2_cache_summary.json"
    assert cache_path.exists()
    rows = json.loads(summary_path.read_text())
    assert rows == [
        {
            "case_id": "au0_trend_long_2022_2025_2025",
            "symbol": "AU0",
            "direction": "long",
            "entries": 1,
            "actionable_plans": 1,
            "mismatches": 0,
            "cache_path": str(cache_path),
        }
    ]
