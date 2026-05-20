from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.experiment_cache import (  # noqa: E402
    build_phase23_experiment_cache,
    phase23_experiment_cache_path,
    write_phase23_experiment_cache,
)
from backtest.models import BacktestCase, BacktestResult, TradeRecord  # noqa: E402


def test_trend_param_sweep_builds_margin_and_tiered_risk_grid() -> None:
    sweep = importlib.import_module("run_account_trend_param_sweep")
    args = sweep._build_parser().parse_args(
        [
            "--max-portfolio-margin-pct-values",
            "0.30,0.50",
            "--ordinary-risk-pct-values",
            "0.005,0.008",
            "--strong-risk-pct-values",
            "0.012",
            "--strong-first-target-rr-values",
            "2.0",
            "--strong-second-target-rr-values",
            "3.0",
            "--split-initial-fraction-values",
            "1.0,0.5",
            "--split-second-entry-trigger-values",
            "none,tp1",
            "--execution-policy-values",
            "fixed,conditional",
        ]
    )

    combos = sweep._combo_rows(args)
    labels = [sweep._combo_labels(combo) for combo in combos]

    assert len(combos) == 32
    assert {label["execution_policy"] for label in labels} == {"fixed", "conditional"}
    assert {label["max_portfolio_margin_pct"] for label in labels} == {"0.3", "0.5"}
    assert {label["ordinary_risk_per_trade_pct"] for label in labels} == {"0.005", "0.008"}
    assert {label["strong_risk_per_trade_pct"] for label in labels} == {"0.012"}
    assert {label["strong_first_target_rr"] for label in labels} == {"2.0"}
    assert {label["strong_second_target_rr"] for label in labels} == {"3.0"}
    assert {label["split_initial_fraction"] for label in labels} == {"1.0", "0.5"}
    assert {label["split_second_entry_trigger"] for label in labels} == {"none", "tp1"}


def test_trend_param_sweep_reuses_loaded_frames_and_writes_account_rows(monkeypatch, tmp_path: Path) -> None:
    sweep = importlib.import_module("run_account_trend_param_sweep")
    case = BacktestCase(
        case_id="au0_trend_long_2022_2025",
        symbol="AU0",
        name="黄金",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    skipped_case = BacktestCase(
        case_id="au0_trend_short_2022_2025",
        symbol="AU0",
        name="黄金",
        direction="short",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00", "2025-01-03 09:00:00"]),
            "close": [100.0, 90.0],
        }
    )
    loaded_cases: list[str] = []
    seen_pre_market_cfgs: list[dict] = []

    monkeypatch.setattr(sweep, "load_config", lambda: {"pre_market": {"base": True}, "intraday": {}})
    monkeypatch.setattr(sweep, "_resolved_fundamental_mode", lambda _config, cli_mode: cli_mode or "strict")
    monkeypatch.setattr(
        sweep,
        "_config_with_fundamental_mode",
        lambda config, mode: {
            **config,
            "pre_market": {**(config.get("pre_market") or {}), "backtest_fundamental_mode": mode},
        },
    )
    monkeypatch.setattr(sweep, "get_case_group", lambda group: [case, skipped_case])

    def fake_load(*, case, config):
        loaded_cases.append(case.case_id)
        return daily, minute

    def fake_run_case_from_frames(**kwargs):
        seen_pre_market_cfgs.append(dict(kwargs["pre_market_cfg"]))
        return BacktestResult(
            case_id=kwargs["case"].case_id,
            trades=[
                TradeRecord(
                    trade_id=f"{kwargs['case'].case_id}-early-stop",
                    symbol="AU0",
                    direction="long",
                    entry_time="2025-01-02 09:00:00",
                    entry_price=100.0,
                    exit_time="2025-01-03 09:00:00",
                    exit_price=90.0,
                    exit_reason="stop",
                    bars_held=10,
                    days_held=1,
                    tp1_hit=False,
                    pnl_ratio=-0.10,
                    meta={
                        "phase2_score": 65.0,
                        "contract_multiplier": 10.0,
                        "execution_risk_per_lot": 1_000.0,
                        "execution_margin_per_lot": 10_000.0,
                    },
                )
            ],
            summary={},
            diagnostics={"phase3_entry_confirmation_wait_bars": len(seen_pre_market_cfgs)},
        )

    monkeypatch.setattr(sweep, "load_case_frames_with_tqbacktest", fake_load)
    monkeypatch.setattr(sweep, "run_case_from_frames", fake_run_case_from_frames)
    monkeypatch.setattr(sweep, "_plan_factory_for_case", lambda _case: "plan-factory")
    monkeypatch.setattr(
        sweep,
        "_contract_spec",
        lambda _config, _symbol: {"commission_per_lot": 0.0, "commission_rate": 0.0, "margin_rate": 0.10},
    )
    output_prefix = tmp_path / "sweep"

    exit_code = sweep.main(
        [
            "--case-group",
            "long_trend_core",
            "--year",
            "2025",
            "--trend-min-stop-atr-multiples",
            "legacy",
            "--trend-max-entry-adverse-r-values",
            "legacy",
            "--trend-min-first-target-rr-values",
            "1.5",
            "--trend-second-target-rr-relax-threshold-values",
            "legacy,2.5",
            "--trend-relaxed-first-target-rr-values",
            "1.0",
            "--trend-entry-confirmation-bars-values",
            "legacy",
            "--trend-initial-stop-grace-bars-values",
            "legacy",
            "--output-prefix",
            str(output_prefix),
            "--case-limit",
            "1",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert loaded_cases == ["au0_trend_long_2022_2025"]
    assert len(seen_pre_market_cfgs) == 2
    assert seen_pre_market_cfgs[0]["trend_min_first_target_rr"] == 1.5
    assert "trend_second_target_rr_relax_threshold" not in seen_pre_market_cfgs[0]
    assert seen_pre_market_cfgs[1]["trend_min_first_target_rr"] == 1.5
    assert seen_pre_market_cfgs[1]["trend_second_target_rr_relax_threshold"] == 2.5
    assert seen_pre_market_cfgs[1]["trend_relaxed_first_target_rr"] == 1.0
    rows = json.loads((tmp_path / "sweep_summary.json").read_text())
    assert [row["trend_min_first_target_rr"] for row in rows] == ["1.5", "1.5"]
    assert [row["trend_second_target_rr_relax_threshold"] for row in rows] == ["legacy", "2.5"]
    assert [row["trend_relaxed_first_target_rr"] for row in rows] == ["1.0", "1.0"]
    assert rows[0]["candidate_trades"] == 1
    assert rows[0]["accepted_trades"] == 1
    assert rows[0]["split_target_lots"] == 15
    assert rows[0]["split_first_entry_lots"] == 15
    assert rows[0]["split_second_entry_filled_lots"] == 0
    assert rows[0]["split_second_entry_unfilled_lots"] == 0
    assert rows[0]["execution_profile_counts"] == {"fixed_full": 1}
    assert rows[0]["stop_trades"] == 1
    assert rows[0]["early_stop_trades"] == 1
    assert rows[0]["phase3_entry_confirmation_wait_bars"] == 1
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "sweep_by_symbol.csv").exists()


def test_trend_param_sweep_reads_phase23_cache_without_replaying(monkeypatch, tmp_path: Path) -> None:
    sweep = importlib.import_module("run_account_trend_param_sweep")
    case = BacktestCase(
        case_id="au0_trend_long_2022_2025",
        symbol="AU0",
        name="黄金",
        direction="long",
        start_dt=date(2022, 1, 1),
        end_dt=date(2025, 12, 31),
        strategy_family="trend_following",
    )
    run_case = sweep._case_for_year(case, 2025)
    daily = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
    minute = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02 09:00:00", "2025-01-03 09:00:00"]),
            "close": [100.0, 110.0],
        }
    )
    phase23_result = BacktestResult(
        case_id=run_case.case_id,
        trades=[
            TradeRecord(
                trade_id=f"{run_case.case_id}-cached",
                symbol="AU0",
                direction="long",
                entry_time="2025-01-02 09:00:00",
                entry_price=100.0,
                exit_time="2025-01-03 09:00:00",
                exit_price=110.0,
                exit_reason="tp2",
                bars_held=20,
                days_held=1,
                tp1_hit=True,
                pnl_ratio=0.10,
                meta={
                    "phase2_score": 65.0,
                    "contract_multiplier": 10.0,
                    "execution_risk_per_lot": 1_000.0,
                    "execution_margin_per_lot": 10_000.0,
                },
            )
        ],
        summary={"num_trades": 1},
        diagnostics={"phase2_actionable_days": 1, "trades_opened": 1},
    )
    base_config = {"pre_market": {"base": True}, "intraday": {}}
    cache_dir = tmp_path / "phase23_cache"
    cache_path = phase23_experiment_cache_path(
        cache_dir,
        case=run_case,
        pre_market_cfg=base_config["pre_market"],
        signal_cfg=base_config["intraday"],
    )
    write_phase23_experiment_cache(
        build_phase23_experiment_cache(
            case=run_case,
            result=phase23_result,
            pre_market_cfg=base_config["pre_market"],
            signal_cfg=base_config["intraday"],
        ),
        cache_path,
    )

    monkeypatch.setattr(sweep, "load_config", lambda: base_config)
    monkeypatch.setattr(sweep, "_resolved_fundamental_mode", lambda _config, cli_mode: cli_mode or "strict")
    monkeypatch.setattr(sweep, "_config_with_fundamental_mode", lambda config, _mode: config)
    monkeypatch.setattr(sweep, "get_case_group", lambda _group: [case])
    monkeypatch.setattr(sweep, "load_case_frames_with_tqbacktest", lambda **_kwargs: (daily, minute))
    monkeypatch.setattr(sweep, "_plan_factory_for_case", lambda _case: "plan-factory")
    monkeypatch.setattr(
        sweep,
        "run_case_from_frames",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("phase23 replay should not run")),
    )
    monkeypatch.setattr(
        sweep,
        "_contract_spec",
        lambda _config, _symbol: {"commission_per_lot": 0.0, "commission_rate": 0.0, "margin_rate": 0.10},
    )

    output_prefix = tmp_path / "sweep_cached"
    exit_code = sweep.main(
        [
            "--case-group",
            "long_trend_core",
            "--case-limit",
            "1",
            "--year",
            "2025",
            "--trend-min-stop-atr-multiples",
            "legacy",
            "--trend-max-entry-adverse-r-values",
            "legacy",
            "--trend-min-first-target-rr-values",
            "legacy",
            "--trend-entry-confirmation-bars-values",
            "legacy",
            "--trend-initial-stop-grace-bars-values",
            "legacy",
            "--phase23-cache-dir",
            str(cache_dir),
            "--require-phase23-cache",
            "--output-prefix",
            str(output_prefix),
            "--quiet",
        ]
    )

    assert exit_code == 0
    rows = json.loads((tmp_path / "sweep_cached_summary.json").read_text())
    assert rows[0]["candidate_trades"] == 1
    assert rows[0]["phase23_cache_hits"] == 1
    assert rows[0]["phase23_cache_misses"] == 0
