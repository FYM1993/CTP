from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1.pipeline import build_phase1_candidates, select_top_candidates  # noqa: E402
from phase1 import pipeline as phase1_pipeline  # noqa: E402
from cli import daily_workflow  # noqa: E402
import pandas as pd  # noqa: E402


def test_select_top_candidates_orders_by_attention_only() -> None:
    rows = [
        {
            "symbol": "LH0",
            "reversal_score": 60.0,
            "trend_score": 36.0,
            "attention_score": 76.0,
        },
        {
            "symbol": "RM0",
            "reversal_score": 82.0,
            "trend_score": 55.0,
            "attention_score": 73.0,
        },
        {
            "symbol": "A0",
            "reversal_score": 70.0,
            "trend_score": 57.0,
            "attention_score": 74.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=2)

    assert [row["symbol"] for row in selected] == ["LH0", "A0"]


def test_select_top_candidates_excludes_rows_below_threshold() -> None:
    rows = [
        {
            "symbol": "LOW0",
            "reversal_score": 54.0,
            "trend_score": 54.0,
            "attention_score": 99.0,
        },
        {
            "symbol": "KEEP0",
            "reversal_score": 55.0,
            "trend_score": 20.0,
            "attention_score": 10.0,
        },
    ]

    selected = select_top_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["KEEP0"]


def test_select_top_candidates_respects_symbol_specific_threshold() -> None:
    rows = [
        {
            "symbol": "LH0",
            "reversal_score": 62.0,
            "trend_score": 15.0,
            "attention_score": 62.0,
            "entry_threshold": 65.0,
            "data_coverage": 0.8,
        },
        {
            "symbol": "M0",
            "reversal_score": 59.0,
            "trend_score": 18.0,
            "attention_score": 59.0,
            "entry_threshold": 58.0,
            "data_coverage": 0.8,
        },
    ]

    selected = select_top_candidates(rows, top_n=5)

    assert [row["symbol"] for row in selected] == ["M0"]


def test_build_phase1_candidates_keeps_rows_below_threshold_for_diagnostics(monkeypatch) -> None:
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=120, freq="D")})
    raw_candidates = {
        "LH0": {
            "symbol": "LH0",
            "name": "生猪",
            "exchange": "dce",
            "reversal_score": 62.0,
            "trend_score": 15.0,
            "data_coverage": 0.8,
            "labels": ["反转候选"],
            "state_labels": ["低位出清"],
            "reason_summary": "养殖利润深亏",
            "reversal_direction": "long",
            "trend_direction": "short",
        },
        "M0": {
            "symbol": "M0",
            "name": "豆粕",
            "exchange": "dce",
            "reversal_score": 59.0,
            "trend_score": 18.0,
            "data_coverage": 0.8,
            "labels": ["反转候选"],
            "state_labels": ["低位出清"],
            "reason_summary": "库存偏紧",
            "reversal_direction": "long",
            "trend_direction": "short",
        },
    }

    monkeypatch.setattr(
        phase1_pipeline,
        "_build_candidate",
        lambda *, info, df, as_of_date=None: dict(raw_candidates[info["symbol"]]),
    )

    candidates = build_phase1_candidates(
        all_data={"LH0": df, "M0": df},
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ],
        threshold=10.0,
        config={
            "fundamental_screening": {
                "top_n": 5,
                "categories": {
                    "meal": {
                        "threshold": 3.0,
                        "symbols": ["M0"],
                    }
                },
            }
        },
    )

    assert [row["symbol"] for row in candidates] == ["LH0", "M0"]
    assert candidates[0]["entry_threshold"] == 65.0
    assert candidates[1]["entry_threshold"] == 58.0
    assert candidates[0]["entry_pool_reason"] == "反转机会分未达标（反转62 / 趋势15；门槛65）"
    assert candidates[1]["entry_pool_reason"] == "反转机会分达标（反转59 / 趋势18）"

    selected = phase1_pipeline.run_phase1_pipeline(
        all_data={"LH0": df, "M0": df},
        symbols=[
            {"symbol": "LH0", "name": "生猪", "exchange": "dce"},
            {"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        ],
        threshold=10.0,
        config={
            "fundamental_screening": {
                "top_n": 5,
                "categories": {
                    "meal": {
                        "threshold": 3.0,
                        "symbols": ["M0"],
                    }
                },
            }
        },
    )

    assert [row["symbol"] for row in selected] == ["M0"]


def test_phase1_output_has_attention_not_direction() -> None:
    candidate = {
        "symbol": "LH0",
        "name": "生猪",
        "reversal_score": 88.0,
        "trend_score": 35.0,
        "attention_score": 76.0,
        "labels": ["反转候选"],
    }

    assert "direction" not in candidate
    assert candidate["attention_score"] == 76.0


def test_phase_1_screen_adds_bridge_fields_without_direction(monkeypatch, capsys) -> None:
    seen_logs: list[str] = []

    monkeypatch.setattr(
        daily_workflow,
        "load_config",
        lambda: {"fundamental_screening": {"top_n": 10, "default_threshold": 10}},
    )
    monkeypatch.setattr(
        daily_workflow,
        "get_all_symbols",
        lambda: [{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_symbol_legacy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("旧 legacy 引擎不应再被调用")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_symbol_regime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("旧 regime 引擎不应再被调用")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_phase1_candidates",
        lambda *, all_data, symbols, threshold, config: [
            {
                "symbol": "LH0",
                "name": "生猪",
                "exchange": "dce",
                "price": 14000.0,
                "score": 66.0,
                "fund_score": 66.0,
                "fund_details": "库存下降",
                "range_pct": 3.0,
                "attention_score": 66.0,
                "reversal_score": 66.0,
                "trend_score": 31.0,
                "labels": ["反转候选"],
                "state_labels": ["低位出清"],
                "data_coverage": 0.75,
                "reason_summary": "库存下降",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "select_top_candidates",
        lambda rows, top_n: list(rows),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: seen_logs.append(msg % args if args else msg),
    )

    rows = daily_workflow.phase_1_screen(
        {"LH0": [None] * 60},
        threshold=10,
    )
    captured = capsys.readouterr()

    assert len(rows) == 1
    row = rows[0]
    assert captured.out == ""
    assert "direction" not in row
    assert row["attention_score"] == 66.0
    assert row["reversal_score"] == 66.0
    assert row["trend_score"] == 31.0
    assert row["labels"] == ["反转候选"]
    assert row["state_labels"] == ["低位出清"]
    assert row["data_coverage"] == 0.75
    assert row["reason_summary"] == "库存下降"
    assert any("Phase 1关注池（门槛通过）: 1 个品种，其中反转线1 个" in line for line in seen_logs)
    assert not any("进入 Phase 2 候选池" in line for line in seen_logs)


def test_phase_1_screen_logs_reversal_and_trend_story_details(monkeypatch) -> None:
    seen_logs: list[str] = []

    monkeypatch.setattr(
        daily_workflow,
        "load_config",
        lambda: {"fundamental_screening": {"top_n": 10, "default_threshold": 10}},
    )
    monkeypatch.setattr(
        daily_workflow,
        "get_all_symbols",
        lambda: [{"symbol": "UR0", "name": "尿素", "exchange": "czce"}],
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_phase1_candidates",
        lambda *, all_data, symbols, threshold, config: [
            {
                "symbol": "UR0",
                "name": "尿素",
                "exchange": "czce",
                "price": 2050.0,
                "score": 90.0,
                "fund_score": 90.0,
                "fund_details": "价格/OI共振上行",
                "range_pct": 100.0,
                "attention_score": 90.0,
                "reversal_score": 39.0,
                "trend_score": 90.0,
                "reversal_direction": "short",
                "trend_direction": "long",
                "labels": ["趋势候选"],
                "state_labels": ["紧平衡强化"],
                "data_coverage": 0.75,
                "reason_summary": "价格/OI共振上行",
                "entry_pool_reason": "趋势机会分达标",
                "phase1_score_details": {
                    "reversal": {
                        "direction": "short",
                        "score": 39.0,
                        "up_score": 0.0,
                        "down_score": 43.0,
                        "drivers": ["价格高位39", "高位持续15"],
                    },
                    "trend": {
                        "direction": "long",
                        "score": 90.0,
                        "up_score": 100.0,
                        "down_score": 0.0,
                        "drivers": ["价格/OI共振上行90", "均线多头扩散100"],
                    },
                    "coverage": {"data_coverage": 0.75, "shrink": 0.90},
                },
            }
        ],
        raising=False,
    )
    monkeypatch.setattr(daily_workflow, "select_top_candidates", lambda rows, top_n: list(rows), raising=False)
    monkeypatch.setattr(daily_workflow.log, "info", lambda msg, *args: seen_logs.append(msg % args if args else msg))

    daily_workflow.phase_1_screen({"UR0": [None] * 60}, threshold=10)

    assert any("反转故事线: 做空 39分" in line for line in seen_logs)
    assert any("价格高位39" in line and "高位持续15" in line for line in seen_logs)
    assert any("趋势故事线: 做多 90分" in line for line in seen_logs)
    assert any("价格/OI共振上行90" in line and "均线多头扩散100" in line for line in seen_logs)


def test_log_trend_bridge_diagnostics_explains_phase1_trend_rejection(monkeypatch) -> None:
    seen_logs: list[str] = []
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120, freq="D"),
            "open": [100 + i for i in range(120)],
            "high": [101 + i for i in range(120)],
            "low": [99 + i for i in range(120)],
            "close": [100 + i for i in range(120)],
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda frame, direction, cfg: {
            "均线排列": 30.0,
            "MACD": 20.0,
            "动量": 5.0,
            "量价关系": 0.0,
            "VSA信号": 0.0,
            "持仓信号": 0.0,
        },
    )
    monkeypatch.setattr(daily_workflow.log, "info", lambda msg, *args: seen_logs.append(msg % args if args else msg))

    daily_workflow._log_trend_bridge_diagnostics(
        phase1_candidates=[
            {
                "symbol": "UR0",
                "name": "尿素",
                "labels": ["趋势候选"],
                "trend_score": 90.0,
                "attention_score": 90.0,
            }
        ],
        trend_candidates=[],
        all_data={"UR0": df},
        config={"pre_market": {"min_history_bars": 60, "direction_delta": 12.0}},
    )

    assert any("趋势线衔接: Phase 1趋势标签1 个, 盘前趋势技术池0 个" in line for line in seen_logs)
    assert any("尿素 (UR0)" in line and "Phase1趋势90 -> 盘前趋势55" in line for line in seen_logs)
    assert any("未入池: 盘前趋势分55 < 60" in line for line in seen_logs)
    assert any("均线排列+30" in line and "MACD+20" in line for line in seen_logs)


def test_build_phase1_diagnostics_reports_factor_hits_and_blocked_rows() -> None:
    rows = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "attention_score": 70.0,
            "reversal_score": 62.0,
            "trend_score": 18.0,
            "entry_threshold": 58.0,
            "data_coverage": 0.75,
            "labels": ["反转候选"],
            "inv_change_4wk": 3.0,
            "receipt_change": None,
            "seasonal_signal": None,
            "hog_profit": -12.0,
            "reason_summary": "养殖利润深亏",
        },
        {
            "symbol": "M0",
            "name": "豆粕",
            "attention_score": 61.0,
            "reversal_score": 61.0,
            "trend_score": 12.0,
            "entry_threshold": 63.0,
            "data_coverage": 0.75,
            "labels": ["反转候选"],
            "inv_change_4wk": -18.0,
            "receipt_change": -120.0,
            "seasonal_signal": None,
            "hog_profit": None,
            "reason_summary": "库存偏紧",
        },
        {
            "symbol": "CF0",
            "name": "棉花",
            "attention_score": 58.0,
            "reversal_score": 22.0,
            "trend_score": 58.0,
            "entry_threshold": 63.0,
            "data_coverage": 0.5,
            "labels": [],
            "inv_change_4wk": None,
            "receipt_change": None,
            "seasonal_signal": 0.8,
            "hog_profit": None,
            "reason_summary": "季节性偏强",
        },
    ]

    diag = daily_workflow._build_phase1_diagnostics(rows, selected_rows=[rows[0]])

    assert diag["num_rows"] == 3
    assert diag["num_eligible"] == 1
    assert diag["num_selected"] == 1
    assert diag["factor_hits"] == {
        "inventory": 2,
        "receipt": 1,
        "seasonality": 1,
        "hog": 1,
    }
    assert diag["label_counts"] == {
        "reversal": 2,
        "trend": 0,
        "dual": 0,
        "low_coverage": 0,
    }
    assert diag["threshold_counts"] == {58.0: 1, 63.0: 2}
    assert diag["blocked_preview"][0]["symbol"] == "M0"
    assert diag["blocked_preview"][0]["dominant_score"] == 61.0
    assert diag["blocked_preview"][0]["entry_threshold"] == 63.0
    assert diag["blocked_reversal_labeled"] == 1


def test_build_candidate_includes_phase1_story_score_details(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=140, freq="D")
    closes = [100 + i for i in range(140)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0] * 140,
            "oi": [5000.0 + i * 20 for i in range(140)],
        }
    )

    monkeypatch.setattr(
        phase1_pipeline,
        "build_fundamental_snapshot",
        lambda **kwargs: {"raw_details": {}, "coverage_score": 0.0, "coverage_status": "missing"},
        raising=False,
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_oi_structure",
        lambda frame: {
            "oi_vs_price": "增仓上涨",
            "oi_20d_change": 8.0,
            "oi_percentile": 82.0,
        },
    )

    candidate = phase1_pipeline._build_candidate(
        info={"symbol": "UR0", "name": "尿素", "exchange": "czce"},
        df=df,
    )

    details = candidate["phase1_score_details"]
    assert set(details.keys()) == {"reversal", "trend", "coverage"}
    assert details["trend"]["direction"] == candidate["trend_direction"]
    assert details["trend"]["score"] == candidate["trend_score"]
    assert any("价格/OI共振上行" in item for item in details["trend"]["drivers"])
    assert details["coverage"]["data_coverage"] == candidate["data_coverage"]


def test_run_phase1_pipeline_builds_reversal_candidate_from_distress(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [20000 - i * 50 for i in range(119)] + [14100]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 100 for c in closes],
            "low": [c - 100 for c in closes],
            "close": closes,
            "volume": [1000] * 120,
            "oi": [20000] * 120,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_hog_fundamentals",
        lambda: {
            "price": 11.8,
            "price_5d_ago": 11.2,
            "price_trend": 5.4,
            "cost": 1750.0,
            "profit_margin": -22.0,
            "data_status": "full",
        },
    )

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"LH0": df},
        symbols=[{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5}},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] in (["反转候选"], ["双标签候选"])
    assert "低位出清" in row["state_labels"]
    assert row["reversal_direction"] == "long"
    assert row["attention_score"] >= 55.0


def test_build_candidate_keeps_hog_reversal_high_when_spot_is_still_low(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [12000.0] * 120
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        phase1_pipeline,
        "_price_stats",
        lambda frame: {
            "price": 11290.0,
            "range_pct": 40.38,
            "price_percentile_300d": 40.38,
            "price_percentile_full": 23.84,
            "low_persistence_days": 0,
            "high_persistence_days": 0,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "_historical_proxy_scores",
        lambda *, df, stats: (
            {
                "reversal_up": 0.0,
                "reversal_down": 0.0,
                "trend_up": 0.0,
                "trend_down": 0.0,
            },
            [],
        ),
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol, as_of_date=None: {
            "inv_now": 1210.0,
            "inv_change_4wk": -1.2,
            "inv_cumulating_weeks": 3,
            "inv_percentile": 96.8,
            "inv_trend": "持平",
        },
    )
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_hog_fundamentals",
        lambda as_of_date=None: {
            "price": 9.9,
            "price_5d_ago": 8.77,
            "price_trend": 12.88,
            "cost": 2396.0,
            "profit_margin": -50.4,
            "spot_price_percentile": 17.3,
            "data_status": "full",
        },
    )

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "LH0", "name": "生猪", "exchange": "dce"},
        df=df,
    )

    assert cand is not None
    assert cand["reversal_score"] >= 60.0
    assert "反转候选" in cand["labels"] or "双标签候选" in cand["labels"]


def test_build_candidate_keeps_structural_distress_reversal_high_after_small_rebound(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [42000.0] * 120
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        phase1_pipeline,
        "_price_stats",
        lambda frame: {
            "price": 42685.0,
            "range_pct": 38.96,
            "price_percentile_300d": 38.96,
            "price_percentile_full": 38.96,
            "low_persistence_days": 39,
            "high_persistence_days": 0,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "_historical_proxy_scores",
        lambda *, df, stats: (
            {
                "reversal_up": 0.0,
                "reversal_down": 0.0,
                "trend_up": 0.0,
                "trend_down": 0.0,
            },
            [],
        ),
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol, as_of_date=None: {
            "inv_now": 12580.0,
            "inv_change_4wk": 2.69,
            "inv_cumulating_weeks": 6,
            "inv_percentile": 100.0,
            "inv_trend": "累库",
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_warehouse_receipt",
        lambda symbol, as_of_date=None: {
            "receipt_total": 12580.0,
            "receipt_change": 40.0,
            "exchange": "GFEX",
        },
    )
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "PS0", "name": "多晶硅", "exchange": "gfex"},
        df=df,
    )

    assert cand is not None
    assert cand["reversal_score"] >= 60.0
    assert "反转候选" in cand["labels"] or "双标签候选" in cand["labels"]


def test_build_candidate_uses_supply_pressure_as_short_reversal_driver(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [100.0] * 120
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        phase1_pipeline,
        "_price_stats",
        lambda frame: {
            "price": 100.0,
            "range_pct": 100.0,
            "price_percentile_300d": 100.0,
            "price_percentile_full": 100.0,
            "low_persistence_days": 0,
            "high_persistence_days": 25,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "_historical_proxy_scores",
        lambda *, df, stats: (
            {
                "reversal_up": 0.0,
                "reversal_down": 0.0,
                "trend_up": 0.0,
                "trend_down": 0.0,
            },
            [],
        ),
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "build_fundamental_snapshot",
        lambda **kwargs: {
            "raw_details": {
                "inventory": {
                    "inv_now": 12580.0,
                    "inv_change_4wk": 18.0,
                    "inv_cumulating_weeks": 7,
                    "inv_percentile": 92.0,
                    "inv_trend": "累库",
                },
                "warehouse_receipt": {
                    "receipt_total": 2000.0,
                    "receipt_change": 160.0,
                    "exchange": "GFEX",
                },
            },
            "coverage_score": 0.75,
            "coverage_status": "partial",
        },
        raising=False,
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_oi_structure", lambda frame: None)

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "PS0", "name": "多晶硅", "exchange": "gfex"},
        df=df,
    )

    assert cand is not None
    assert cand["reversal_direction"] == "short"
    drivers = cand["phase1_score_details"]["reversal"]["drivers"]
    assert any("过剩/高库存" in item for item in drivers)
    assert not any("收紧/低库存" in item for item in drivers)
    assert phase1_pipeline.DOMAIN_INVENTORY_SUPPLY in cand["reversal_evidence_domains"]


def test_build_candidate_does_not_use_tight_supply_as_short_reversal_evidence(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [100.0] * 120
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )

    monkeypatch.setattr(
        phase1_pipeline,
        "_price_stats",
        lambda frame: {
            "price": 100.0,
            "range_pct": 100.0,
            "price_percentile_300d": 100.0,
            "price_percentile_full": 100.0,
            "low_persistence_days": 0,
            "high_persistence_days": 25,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "_historical_proxy_scores",
        lambda *, df, stats: (
            {
                "reversal_up": 0.0,
                "reversal_down": 0.0,
                "trend_up": 0.0,
                "trend_down": 0.0,
            },
            [],
        ),
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "build_fundamental_snapshot",
        lambda **kwargs: {
            "raw_details": {
                "inventory": {
                    "inv_now": 800.0,
                    "inv_change_4wk": -18.0,
                    "inv_cumulating_weeks": 1,
                    "inv_percentile": 12.0,
                    "inv_trend": "去库",
                },
                "warehouse_receipt": {
                    "receipt_total": 2000.0,
                    "receipt_change": -160.0,
                    "exchange": "GFEX",
                },
            },
            "coverage_score": 0.75,
            "coverage_status": "partial",
        },
        raising=False,
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_oi_structure", lambda frame: None)

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "PS0", "name": "多晶硅", "exchange": "gfex"},
        df=df,
    )

    assert cand is not None
    assert cand["reversal_direction"] == "short"
    drivers = cand["phase1_score_details"]["reversal"]["drivers"]
    assert not any("收紧/低库存" in item for item in drivers)
    assert phase1_pipeline.DOMAIN_INVENTORY_SUPPLY not in cand["reversal_evidence_domains"]


def test_build_candidate_uses_snapshot_confirmed_fundamental_reversal(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    closes = [42000.0] * 120
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )
    seen: list[tuple[str, object]] = []

    monkeypatch.setattr(
        phase1_pipeline,
        "_price_stats",
        lambda frame: {
            "price": 42685.0,
            "range_pct": 38.96,
            "price_percentile_300d": 38.96,
            "price_percentile_full": 38.96,
            "low_persistence_days": 39,
            "high_persistence_days": 0,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "_historical_proxy_scores",
        lambda *, df, stats: (
            {
                "reversal_up": 0.0,
                "reversal_down": 0.0,
                "trend_up": 0.0,
                "trend_down": 0.0,
            },
            [],
        ),
    )

    def fake_snapshot(**kwargs):
        seen.append((kwargs["symbol"], kwargs["as_of_date"]))
        return {
            "commodity_group": "new_energy_materials",
            "evidence_domains_present": ["inventory", "warehouse_receipt", "spot_basis"],
            "evidence_domains_missing": [],
            "coverage_score": 1.0,
            "coverage_status": "complete",
            "missing_domain_reasons": [],
            "extreme_state_direction": "long",
            "extreme_state_confirmed": True,
            "extreme_state_reasons": ["高库存"],
            "marginal_turn_direction": "long",
            "marginal_turn_confirmed": True,
            "marginal_turn_reasons": ["去库启动", "仓单回落"],
            "fundamental_reversal_confirmed": True,
            "only_proxy_evidence": False,
            "raw_details": {
                "inventory": {
                    "inv_now": 12580.0,
                    "inv_change_4wk": -4.0,
                    "inv_cumulating_weeks": 6,
                    "inv_percentile": 100.0,
                    "inv_trend": "去库",
                },
                "warehouse_receipt": {
                    "receipt_total": 12580.0,
                    "receipt_change": -120.0,
                    "exchange": "GFEX",
                },
                "spot_basis": {
                    "commodity_code": "PS",
                    "spot_price": 46666.67,
                    "dominant_contract_price": 43560.0,
                    "basis": -3106.67,
                    "basis_rate": -0.066571,
                    "data_date": "20250401",
                },
            },
        }

    monkeypatch.setattr(phase1_pipeline, "build_fundamental_snapshot", fake_snapshot, raising=False)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    as_of = pd.Timestamp("2025-04-01").date()
    cand = phase1_pipeline._build_candidate(
        info={"symbol": "PS0", "name": "多晶硅", "exchange": "gfex"},
        df=df,
        as_of_date=as_of,
    )

    assert seen == [("PS0", as_of)]
    assert cand is not None
    assert cand["fundamental_reversal_confirmed"] is True
    assert cand["fundamental_extreme_state_confirmed"] is True
    assert cand["fundamental_marginal_turn_confirmed"] is True
    assert cand["fundamental_missing_domain_reasons"] == []
    assert cand["fundamental_snapshot"]["raw_details"]["spot_basis"]["basis"] == -3106.67
    assert cand["inv_percentile"] == 100.0
    assert cand["receipt_change"] == -120.0
    assert cand["reversal_direction"] == "long"
    assert cand["reversal_score"] >= 60.0


def test_build_candidate_keeps_proxy_reversal_unconfirmed_when_snapshot_missing_domains(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    base = [22000 - i * 55 for i in range(170)]
    closes = base + [12550, 12480, 12420, 12380, 12360, 12340, 12320, 12300, 12280, 12260]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 80 for c in closes],
            "low": [c - 80 for c in closes],
            "close": closes,
            "volume": [1000.0] * 180,
            "oi": [15000 - i * 10 for i in range(180)],
        }
    )

    def fake_snapshot(**kwargs):
        return {
            "commodity_group": "livestock_perishables",
            "evidence_domains_present": [],
            "evidence_domains_missing": ["spot_basis", "supply"],
            "coverage_score": 0.0,
            "coverage_status": "missing",
            "missing_domain_reasons": ["缺少现货/基差数据", "缺少供给数据"],
            "extreme_state_direction": "",
            "extreme_state_confirmed": False,
            "extreme_state_reasons": [],
            "marginal_turn_direction": "",
            "marginal_turn_confirmed": False,
            "marginal_turn_reasons": [],
            "fundamental_reversal_confirmed": False,
            "only_proxy_evidence": True,
            "raw_details": {},
        }

    monkeypatch.setattr(phase1_pipeline, "build_fundamental_snapshot", fake_snapshot, raising=False)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "LH0", "name": "生猪", "exchange": "dce"},
        df=df,
        as_of_date=pd.Timestamp("2025-06-29").date(),
    )

    assert cand is not None
    assert cand["reversal_direction"] == "long"
    assert cand["reversal_score"] >= 55.0
    assert "历史代理" in cand["reason_summary"]
    assert cand["fundamental_reversal_confirmed"] is False
    assert cand["fundamental_only_proxy_evidence"] is True
    assert cand["fundamental_missing_domain_reasons"] == ["缺少现货/基差数据", "缺少供给数据"]


def test_run_phase1_pipeline_uses_price_trend_not_tightening_for_trend_score(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    closes = [7000 + i * 4 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 40 for c in closes],
            "low": [c - 40 for c in closes],
            "close": closes,
            "volume": [1000] * 180,
            "oi": [15000] * 180,
        }
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol: {
            "inv_now": 1200.0,
            "inv_change_4wk": -18.0,
            "inv_cumulating_weeks": 1,
            "inv_percentile": 18.0,
            "inv_trend": "去库",
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_warehouse_receipt",
        lambda symbol: {
            "receipt_total": 1000.0,
            "receipt_change": -120.0,
            "exchange": "SHFE",
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_seasonality",
        lambda frame: {
            "month": 4,
            "hist_avg_return": 3.8,
            "hist_up_pct": 68.0,
            "current_month_return": 1.2,
            "seasonal_signal": 0.8,
        },
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda: None)

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"M0": df},
        symbols=[{"symbol": "M0", "name": "豆粕", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trend_score"] >= 55.0
    assert row["trend_direction"] == "long"
    assert "历史代理" in row["reason_summary"]
    assert "库存" not in row["reason_summary"]
    assert row["attention_score"] >= 55.0


def test_build_candidate_exposes_oi_structure_and_evidence_domains(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    closes = [7000 + i * 3 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 30 for c in closes],
            "low": [c - 30 for c in closes],
            "close": closes,
            "volume": [1000.0] * 180,
            "oi": [15000.0] * 180,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_oi_structure",
        lambda frame: {
            "oi_now": 18200.0,
            "oi_percentile": 84.0,
            "oi_20d_change": 12.0,
            "oi_vs_price": "增仓上涨",
        },
    )

    cand = phase1_pipeline._build_candidate(
        info={"symbol": "M0", "name": "豆粕", "exchange": "dce"},
        df=df,
    )

    assert cand is not None
    assert cand["oi_vs_price"] == "增仓上涨"
    assert cand["oi_20d_change"] == 12.0
    assert cand["oi_percentile"] == 84.0
    assert "technical_trend" in cand["trend_evidence_domains"]
    assert "positioning_oi" in cand["trend_evidence_domains"]


def test_phase1_fundamentals_alone_do_not_create_trend_candidate(monkeypatch) -> None:
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 180,
            "high": [101.0] * 180,
            "low": [99.0] * 180,
            "close": [100.0] * 180,
            "volume": [1000.0] * 180,
            "oi": [5000.0] * 180,
        }
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol, as_of_date=None: {
            "inv_change_4wk": -20.0,
            "inv_percentile": 10.0,
        },
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_warehouse_receipt",
        lambda symbol, as_of_date=None: {"receipt_change": -500.0},
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_seasonality",
        lambda frame, as_of_date=None: {"seasonal_signal": 1.0},
    )
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)
    monkeypatch.setattr(
        phase1_pipeline,
        "get_oi_structure",
        lambda frame: {"oi_vs_price": "无明显信号", "oi_20d_change": 0.0, "oi_percentile": 50.0},
    )

    rows = phase1_pipeline.build_phase1_candidates(
        all_data={"M0": df},
        symbols=[{"symbol": "M0", "name": "豆粕", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
    )

    assert len(rows) == 1
    assert "趋势候选" not in rows[0]["labels"]
    assert rows[0]["trend_score"] < 55.0


def test_run_phase1_pipeline_passes_as_of_date_to_factor_fetchers(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 120,
            "high": [101.0] * 120,
            "low": [99.0] * 120,
            "close": [100.0] * 120,
            "volume": [1000.0] * 120,
            "oi": [5000.0] * 120,
        }
    )
    seen: list[tuple[str, object]] = []

    monkeypatch.setattr(
        phase1_pipeline,
        "get_inventory",
        lambda symbol, as_of_date=None: seen.append(("inventory", as_of_date)) or None,
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_warehouse_receipt",
        lambda symbol, as_of_date=None: seen.append(("receipt", as_of_date)) or None,
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_seasonality",
        lambda frame, as_of_date=None: seen.append(("seasonality", as_of_date)) or None,
    )
    monkeypatch.setattr(
        phase1_pipeline,
        "get_hog_fundamentals",
        lambda as_of_date=None: seen.append(("hog", as_of_date)) or None,
    )

    as_of_date = pd.Timestamp("2025-04-15").date()
    phase1_pipeline.run_phase1_pipeline(
        all_data={"LH0": df},
        symbols=[{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5}},
        as_of_date=as_of_date,
    )

    assert seen == [
        ("inventory", as_of_date),
        ("receipt", as_of_date),
        ("seasonality", as_of_date),
        ("hog", as_of_date),
    ]


def test_run_phase1_pipeline_historical_replay_uses_trend_proxy_when_external_factors_missing(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    closes = [7000 + i * 18 for i in range(180)]
    oi = [10000 + i * 25 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 40 for c in closes],
            "low": [c - 40 for c in closes],
            "close": closes,
            "volume": [1000] * 180,
            "oi": oi,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"M0": df},
        symbols=[{"symbol": "M0", "name": "豆粕", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
        as_of_date=pd.Timestamp("2025-06-29").date(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] == ["趋势候选"]
    assert row["trend_score"] >= 55.0
    assert row["trend_direction"] == "long"
    assert "历史代理" in row["reason_summary"]


def test_run_phase1_pipeline_live_uses_trend_proxy_when_external_factors_missing(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    closes = [7000 + i * 18 for i in range(180)]
    oi = [10000 + i * 25 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 40 for c in closes],
            "low": [c - 40 for c in closes],
            "close": closes,
            "volume": [1000] * 180,
            "oi": oi,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"M0": df},
        symbols=[{"symbol": "M0", "name": "豆粕", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] == ["趋势候选"]
    assert row["trend_score"] >= 55.0
    assert row["trend_direction"] == "long"
    assert "历史代理" in row["reason_summary"]


def test_run_phase1_pipeline_historical_replay_uses_reversal_proxy_when_external_factors_missing(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    base = [22000 - i * 55 for i in range(170)]
    closes = base + [12550, 12480, 12420, 12380, 12360, 12340, 12320, 12300, 12280, 12260]
    oi = [15000 - i * 10 for i in range(180)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c + 80 for c in closes],
            "low": [c - 80 for c in closes],
            "close": closes,
            "volume": [1000] * 180,
            "oi": oi,
        }
    )
    monkeypatch.setattr(phase1_pipeline, "get_inventory", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_warehouse_receipt", lambda symbol, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_seasonality", lambda frame, as_of_date=None: None)
    monkeypatch.setattr(phase1_pipeline, "get_hog_fundamentals", lambda as_of_date=None: None)

    rows = phase1_pipeline.run_phase1_pipeline(
        all_data={"LH0": df},
        symbols=[{"symbol": "LH0", "name": "生猪", "exchange": "dce"}],
        threshold=10.0,
        config={"fundamental_screening": {"top_n": 5, "default_threshold": 3}},
        as_of_date=pd.Timestamp("2025-06-29").date(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"] == ["反转候选"]
    assert row["reversal_score"] >= 55.0
    assert row["reversal_direction"] == "long"
    assert "历史代理" in row["reason_summary"]


def test_phase_2_premarket_uses_resolved_direction_before_analyze(monkeypatch) -> None:
    calls: list[dict] = []
    resolve_calls: list[dict] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"均线排列": -18.0, "量价关系": 6.0, "MACD": -8.0},
    )

    def fake_resolve_phase2_direction(**kwargs):
        resolve_calls.append(kwargs)
        return "long"

    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", fake_resolve_phase2_direction)

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        calls.append(
            {
                "symbol": symbol,
                "name": name,
                "direction": direction,
                "cfg": cfg,
            }
        )
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": -22.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 105.0,
            "tp1": 90.0,
            "rr": 1.2,
            "reversal_status": {
                "signal_strength": 0.4,
                "next_expected": "等待确认",
            },
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "score": 66.0,
                "attention_score": 66.0,
                "labels": ["趋势候选"],
                "entry_pool_reason": "配置桥接",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert resolve_calls == [{"long_score": 6.0, "short_score": 26.0, "delta": 12.0}]
    assert calls[0]["direction"] == "long"
    assert watchlist[0]["direction"] == "long"
    assert watchlist[0]["phase2_decision"] == "long"


def test_phase_2_premarket_falls_back_to_stronger_side_when_resolver_watches(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"均线排列": -18.0, "量价关系": 6.0, "MACD": -8.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "watch")

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        calls.append({"symbol": symbol, "direction": direction})
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": -22.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 105.0,
            "tp1": 90.0,
            "rr": 1.2,
            "reversal_status": {
                "signal_strength": 0.4,
                "next_expected": "等待确认",
            },
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "score": 66.0,
                "attention_score": 66.0,
                "labels": ["趋势候选"],
                "entry_pool_reason": "配置桥接",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert calls[0]["direction"] == "short"
    assert watchlist[0]["direction"] == "short"
    assert watchlist[0]["phase2_decision"] == "watch"


def test_phase_2_premarket_prefers_candidate_strategy_direction_over_mixed_scores(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: pd.DataFrame({"close": [100.0, 99.0, 98.0]}))
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {
            "RSI": 10.0,
            "布林带": 10.0,
            "价格位置": 5.0,
            "均线排列": 0.0,
            "MACD": 0.0,
            "动量": 0.0,
            "量价关系": 0.0,
            "VSA信号": 0.0,
            "持仓信号": 0.0,
        },
    )

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        calls.append({"direction": direction, "phase2_decision": cfg["phase2_decision"]})
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": -45.0,
            "actionable": False,
            "entry": 100.0,
            "stop": 105.0,
            "tp1": 90.0,
            "rr": 1.2,
            "reversal_status": {"next_expected": "等待确认"},
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "score": 66.0,
                "attention_score": 66.0,
                "labels": ["趋势候选"],
                "trend_score": 78.0,
                "reversal_score": 12.0,
                "trend_direction": "short",
                "entry_pool_reason": "趋势机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert calls == [{"direction": "short", "phase2_decision": "watch"}]
    assert watchlist[0]["direction"] == "short"


def test_run_dual_strategy_phase2_merges_results_with_strategy_family(monkeypatch) -> None:
    reversal_candidates = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "labels": ["反转候选"],
            "reversal_score": 88.0,
            "trend_score": 20.0,
            "attention_score": 76.0,
        },
    ]
    trend_candidates = [
        {
            "symbol": "RM0",
            "name": "菜粕",
            "labels": [],
            "reversal_score": 18.0,
            "trend_score": 78.0,
            "attention_score": 74.0,
        }
    ]

    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "LH0",
                    "name": "生猪",
                    "direction": "long",
                    "actionable": True,
                    "score": 32.0,
                    "rrf_score": 0.15,
                    "strategy_family": "reversal_fundamental",
                }
            ],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "RM0",
                    "name": "菜粕",
                    "direction": "short",
                    "actionable": True,
                    "score": -29.0,
                    "rrf_score": 0.14,
                    "strategy_family": "trend_following",
                }
            ],
            [],
        ),
        raising=False,
    )

    actionable, watchlist, grouped = daily_workflow.run_dual_strategy_phase2(
        reversal_candidates,
        trend_candidates,
        {"pre_market": {}},
        max_picks=4,
    )

    assert {row["strategy_family"] for row in actionable} == {
        "trend_following",
        "reversal_fundamental",
    }
    assert watchlist == []
    assert set(grouped.keys()) == {"reversal_fundamental", "trend_following"}


def test_run_dual_strategy_phase2_merges_same_direction_strategy_hits_into_one_playbook(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "CF0",
                    "name": "棉花",
                    "direction": "long",
                    "actionable": True,
                    "score": -34.0,
                    "rrf_score": 0.20,
                    "strategy_family": "reversal_fundamental",
                    "entry_family": "reversal",
                    "fundamental_extreme_state_confirmed": True,
                    "fundamental_marginal_turn_confirmed": True,
                    "fundamental_reversal_confirmed": True,
                    "evidence_domains": ["inventory_supply", "warehouse_receipt"],
                }
            ],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "CF0",
                    "name": "棉花",
                    "direction": "long",
                    "actionable": True,
                    "score": 41.0,
                    "rrf_score": 0.10,
                    "strategy_family": "trend_following",
                    "entry_family": "trend",
                    "entry_signal_type": "TrendBreak",
                    "evidence_domains": ["technical_trend", "positioning_oi"],
                }
            ],
            [],
        ),
        raising=False,
    )

    actionable, watchlist, grouped = daily_workflow.run_dual_strategy_phase2(
        [],
        [],
        {"pre_market": {}},
        max_picks=4,
    )

    assert watchlist == []
    assert len(actionable) == 1
    assert actionable[0]["symbol"] == "CF0"
    assert actionable[0]["strategy_family"] == "trend_following"
    assert actionable[0]["market_stage"] == "clear_trend"
    assert actionable[0]["strategy_resonance"] == "same_direction"
    assert actionable[0]["selected_playbook"] == "trend_continuation"
    assert actionable[0]["confluence_quality"] == "independent"
    assert actionable[0]["independent_evidence_count"] == 4
    assert "存在明确趋势" in actionable[0]["unselected_playbook_reason"]
    assert [hit["strategy_family"] for hit in actionable[0]["strategy_pool_hits"]] == [
        "trend_following",
        "reversal_fundamental",
    ]
    assert grouped["reversal_fundamental"]["actionable"][0]["strategy_family"] == "reversal_fundamental"


def test_run_dual_strategy_phase2_downgrades_opposite_direction_strategy_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "CF0",
                    "name": "棉花",
                    "direction": "short",
                    "score": -34.0,
                    "rrf_score": 0.20,
                    "strategy_family": "reversal_fundamental",
                    "entry_family": "reversal",
                    "fundamental_extreme_state_confirmed": True,
                    "fundamental_marginal_turn_confirmed": True,
                    "fundamental_reversal_confirmed": True,
                }
            ],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "CF0",
                    "name": "棉花",
                    "direction": "long",
                    "score": 41.0,
                    "rrf_score": 0.10,
                    "strategy_family": "trend_following",
                    "entry_family": "trend",
                    "entry_signal_type": "TrendBreak",
                }
            ],
            [],
        ),
        raising=False,
    )

    actionable, watchlist, _grouped = daily_workflow.run_dual_strategy_phase2(
        [],
        [],
        {"pre_market": {}},
        max_picks=4,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0]["symbol"] == "CF0"
    assert watchlist[0]["market_stage"] == "conflict"
    assert watchlist[0]["strategy_resonance"] == "conflict"
    assert watchlist[0]["confluence_quality"] == "conflict"
    assert watchlist[0]["selected_playbook"] == "conflict_watch"
    assert bool(watchlist[0]["actionable"]) is False
    assert "策略池方向冲突" in watchlist[0]["downgrade_reason"]


def test_run_dual_strategy_phase2_keeps_rebound_without_turn_as_observation(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: (
            [],
            [
                {
                    "symbol": "PS0",
                    "name": "多晶硅",
                    "direction": "long",
                    "score": 24.0,
                    "rrf_score": 0.18,
                    "strategy_family": "reversal_fundamental",
                    "entry_family": "reversal",
                    "strategy_role": "counter_fundamental_rebound_watch",
                    "fundamental_extreme_state_confirmed": True,
                    "fundamental_marginal_turn_confirmed": False,
                    "fundamental_reversal_confirmed": False,
                    "evidence_domains": ["technical_trend"],
                }
            ],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: ([], []),
        raising=False,
    )

    actionable, watchlist, _grouped = daily_workflow.run_dual_strategy_phase2(
        [],
        [],
        {"pre_market": {}},
        max_picks=4,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0]["market_stage"] == "technical_rebound_only"
    assert watchlist[0]["selected_playbook"] == "technical_rebound_watch"
    assert watchlist[0]["confluence_quality"] == "single_domain"
    assert "未见基本面拐点" in watchlist[0]["unselected_playbook_reason"]


def test_run_dual_strategy_phase2_keeps_trend_watch_as_observation_not_rebound(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: ([], []),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: (
            [],
            [
                {
                    "symbol": "CF0",
                    "name": "棉花",
                    "direction": "long",
                    "score": 18.0,
                    "rrf_score": 0.12,
                    "actionable": False,
                    "strategy_family": "trend_following",
                    "entry_family": "trend",
                    "entry_signal_type": "",
                    "evidence_domains": ["technical_trend"],
                }
            ],
        ),
        raising=False,
    )

    actionable, watchlist, _grouped = daily_workflow.run_dual_strategy_phase2(
        [],
        [],
        {"pre_market": {}},
        max_picks=4,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0]["market_stage"] == "observation"
    assert watchlist[0]["selected_playbook"] == "observation"
    assert "技术反弹" not in watchlist[0]["unselected_playbook_reason"]


def test_run_dual_strategy_phase2_treats_overlapping_price_hits_as_weak_confluence(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "run_reversal_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "JM0",
                    "name": "焦煤",
                    "direction": "long",
                    "actionable": True,
                    "score": 28.0,
                    "rrf_score": 0.16,
                    "strategy_family": "reversal_fundamental",
                    "entry_family": "reversal",
                    "fundamental_extreme_state_confirmed": False,
                    "fundamental_marginal_turn_confirmed": False,
                    "fundamental_reversal_confirmed": False,
                    "evidence_domains": ["technical_trend"],
                }
            ],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_trend_strategy_phase2",
        lambda candidates, config, max_picks: (
            [
                {
                    "symbol": "JM0",
                    "name": "焦煤",
                    "direction": "long",
                    "actionable": True,
                    "score": 35.0,
                    "rrf_score": 0.14,
                    "strategy_family": "trend_following",
                    "entry_family": "trend",
                    "entry_signal_type": "TrendBreak",
                    "evidence_domains": ["technical_trend"],
                }
            ],
            [],
        ),
        raising=False,
    )

    actionable, watchlist, _grouped = daily_workflow.run_dual_strategy_phase2(
        [],
        [],
        {"pre_market": {}},
        max_picks=4,
    )

    assert watchlist == []
    assert len(actionable) == 1
    assert actionable[0]["independent_evidence_count"] == 1
    assert actionable[0]["confluence_quality"] == "overlapping"


def test_merge_strategy_pool_does_not_promote_watch_hits_to_confirmed_confluence() -> None:
    actionable, watchlist = daily_workflow._merge_strategy_pool(
        trend_actionable=[
            {
                "symbol": "CF0",
                "name": "棉花",
                "direction": "long",
                "actionable": True,
                "score": 42.0,
                "rrf_score": 0.18,
                "strategy_family": "trend_following",
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "evidence_domains": ["technical_trend"],
            }
        ],
        trend_watchlist=[],
        reversal_actionable=[],
        reversal_watchlist=[
            {
                "symbol": "CF0",
                "name": "棉花",
                "direction": "long",
                "actionable": False,
                "score": 22.0,
                "rrf_score": 0.12,
                "strategy_family": "reversal_fundamental",
                "entry_family": "reversal",
                "fundamental_extreme_state_confirmed": True,
                "fundamental_marginal_turn_confirmed": False,
                "fundamental_reversal_confirmed": False,
                "evidence_domains": ["inventory_supply", "warehouse_receipt"],
            }
        ],
        max_picks=3,
    )

    assert watchlist == []
    assert len(actionable) == 1
    row = actionable[0]
    assert row["strategy_hit_count"] == 2
    assert row["confirmed_strategy_hit_count"] == 1
    assert row["strategy_resonance"] == "single_strategy"
    assert row["independent_evidence_count"] == 1
    assert row["independent_evidence_summary"] == ["技术趋势"]
    assert row["confluence_quality"] == "single_domain"


def test_merge_strategy_pool_keeps_actionable_plan_when_opposite_watch_exists() -> None:
    actionable, watchlist = daily_workflow._merge_strategy_pool(
        trend_actionable=[
            {
                "symbol": "CF0",
                "name": "棉花",
                "direction": "long",
                "actionable": True,
                "score": 42.0,
                "rrf_score": 0.18,
                "strategy_family": "trend_following",
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "evidence_domains": ["technical_trend"],
            }
        ],
        trend_watchlist=[],
        reversal_actionable=[],
        reversal_watchlist=[
            {
                "symbol": "CF0",
                "name": "棉花",
                "direction": "short",
                "actionable": False,
                "score": -24.0,
                "rrf_score": 0.12,
                "strategy_family": "reversal_fundamental",
                "entry_family": "reversal",
                "fundamental_extreme_state_confirmed": True,
                "fundamental_marginal_turn_confirmed": False,
                "fundamental_reversal_confirmed": False,
                "evidence_domains": ["inventory_supply"],
            }
        ],
        max_picks=3,
    )

    assert watchlist == []
    assert len(actionable) == 1
    row = actionable[0]
    assert row["actionable"] is True
    assert row["market_stage"] == "clear_trend"
    assert row["strategy_resonance"] == "single_strategy"
    assert row["confluence_quality"] == "single_domain"
    assert row["confirmed_strategy_hit_count"] == 1
    assert row["strategy_hit_count"] == 2
    assert "策略池方向冲突" not in str(row.get("downgrade_reason") or "")
    assert "反向观察" in row["unselected_playbook_reason"]


def test_merge_strategy_pool_treats_missing_actionable_as_observation() -> None:
    actionable, watchlist = daily_workflow._merge_strategy_pool(
        trend_actionable=[],
        trend_watchlist=[
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "direction": "short",
                "strategy_family": "trend_following",
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "score": -42.0,
                "rrf_score": 0.1,
                "evidence_domains": ["technical_trend"],
            }
        ],
        reversal_actionable=[],
        reversal_watchlist=[],
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0].get("actionable") is not True
    assert watchlist[0]["market_stage"] == "observation"


def test_format_watchlist_story_status_for_trend_ignores_reversal_next_expected() -> None:
    row = {
        "symbol": "CU0",
        "name": "铜",
        "direction": "long",
        "strategy_family": "trend_following",
        "entry_family": "trend",
        "entry_signal_type": "",
        "entry_signal_detail": "",
        "reversal_status": {"next_expected": "等待Spring反转"},
    }

    assert daily_workflow._format_watchlist_story_status(row) == "等待顺势确认"


def test_main_builds_reversal_and_trend_universes_separately(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(daily_workflow, "phase_0_prefetch", lambda: {"LH0": "lh", "RM0": "rm"})
    monkeypatch.setattr(
        daily_workflow,
        "phase_1_screen",
        lambda all_data, threshold=10: calls.append(("phase1", sorted(all_data.keys()))) or [
            {
                "symbol": "LH0",
                "name": "生猪",
                "labels": ["反转候选"],
                "reversal_score": 88.0,
                "trend_score": 18.0,
                "attention_score": 75.0,
            }
        ],
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_trend_universe",
        lambda all_data, config: calls.append(("trend_universe", sorted(all_data.keys()))) or [
            {
                "symbol": "RM0",
                "name": "菜粕",
                "labels": [],
                "reversal_score": 0.0,
                "trend_score": 78.0,
                "attention_score": 70.0,
            }
        ],
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_dual_strategy_phase2",
        lambda reversal_candidates, trend_candidates, config, max_picks=6: (
            calls.append(("phase2", [c["symbol"] for c in reversal_candidates], [c["symbol"] for c in trend_candidates]))
            or ([], [], {"reversal_fundamental": {"actionable": [], "watchlist": []}, "trend_following": {"actionable": [], "watchlist": []}})
        ),
    )
    monkeypatch.setattr(daily_workflow, "save_targets", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "build_phase1_summary", lambda **kwargs: {})
    monkeypatch.setattr(daily_workflow, "get_request_count", lambda: 0)
    monkeypatch.setattr(
        daily_workflow,
        "load_config",
        lambda: {"fundamental_screening": {"top_n": 5}, "phase1": {"top_n": 5}},
    )
    monkeypatch.setattr(
        daily_workflow.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "resume": False,
                "threshold": 10.0,
                "skip_screen": False,
                "max_picks": 3,
                "no_monitor": True,
                "period": "5",
                "interval": 60,
            },
        )(),
    )

    daily_workflow.main()

    assert calls == [
        ("phase1", ["LH0", "RM0"]),
        ("trend_universe", ["LH0", "RM0"]),
        ("phase2", ["LH0"], ["RM0"]),
    ]


def test_phase_0_prefetch_logs_structured_source_summary(monkeypatch) -> None:
    infos: list[str] = []

    monkeypatch.setattr(
        daily_workflow,
        "prefetch_all_with_stats",
        lambda symbols, completed_only=False: (
            {"LH0": "lh", "RM0": "rm"},
            type(
                "Stats",
                (),
                {
                    "cache_hits": 0,
                    "tq_fetches": 2,
                    "akshare_fetches": 0,
                    "latest_bar_not_today": 2,
                    "intraday_live": 0,
                    "missing": 0,
                },
            )(),
        ),
    )
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    out = daily_workflow.phase_0_prefetch()

    assert out == {"LH0": "lh", "RM0": "rm"}
    assert any("TqSdk拉取2" in message for message in infos)
    assert any("AkShare拉取0" in message for message in infos)
    assert any("最新bar非今日2" in message for message in infos)
    assert all("全部命中本地缓存" not in message for message in infos)


def test_phase_3_intraday_watchlist_signal_output_recomputes_trade_plan_and_highlights_actionable_upgrade(
    monkeypatch,
    capsys,
) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])
    live_plans = iter(
        [
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "strategy_family": "trend_following",
                "score": 42.2,
                "actionable": False,
                "price": 2034.0,
                "entry": 2034.0,
                "stop": 2006.0,
                "tp1": 2051.0,
                "tp2": 2058.0,
                "rr": 0.61,
                "admission_rr": 0.86,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
                "reversal_status": {},
                "trend_status": {
                    "has_signal": True,
                    "signal_type": "TrendBreak",
                    "signal_detail": "markup 阶段趋势突破",
                    "phase": "markup",
                    "phase_ok": True,
                    "slope_ok": True,
                    "trend_indicator_ok": True,
                },
            },
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "strategy_family": "trend_following",
                "score": 42.2,
                "actionable": True,
                "price": 2034.0,
                "entry": 2034.0,
                "stop": 2012.0,
                "tp1": 2062.0,
                "tp2": 2068.0,
                "rr": 1.27,
                "admission_rr": 1.55,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
                "reversal_status": {},
                "trend_status": {
                    "has_signal": True,
                    "signal_type": "TrendBreak",
                    "signal_detail": "markup 阶段趋势突破",
                    "phase": "markup",
                    "phase_ok": True,
                    "slope_ok": True,
                    "trend_indicator_ok": True,
                },
            },
        ]
    )

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_daily_with_live_bar",
        lambda symbol, period: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-22"]),
                "open": [1993.0],
                "high": [2018.0],
                "low": [1985.0],
                "close": [1999.0],
            }
        ),
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should use trend-specific builder")),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: next(live_plans),
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "entry": 2005.0,
                "stop": 1970.0,
                "tp1": 2026.0,
                "tp2": 2029.0,
                "rr": 0.61,
                "admission_rr": 0.68,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
                "entry_pool_reason": "顺势突破观察",
            }
        ],
    )

    out = capsys.readouterr().out

    assert "🔔🔔 尿素 出现顺势确认" in out
    assert "确认: 顺势突破 | 计划方向: 做多" in out
    assert "确认条件: 价格继续站稳入场2034上方，且不跌破止损2006" in out
    assert "🟢 尿素: 顺势突破确认持续" in out
    assert "交易计划: 入场2005 止损1970 止盈1 2026 止盈2 2029 第一止盈RR 0.61 准入RR 0.68" not in out
    assert "交易计划: 入场2034 止损2006 止盈1 2051 止盈2 2058 第一止盈RR 0.61 准入RR 0.86" in out
    assert "交易计划: 入场2034 止损2012 止盈1 2062 止盈2 2068 第一止盈RR 1.27 准入RR 1.55" in out
    assert "✅ 盘中重评后转为可执行：评分+42 准入RR 1.55" in out
    assert any("🔔🔔 尿素 出现顺势确认" in message for message in infos)
    assert any("🟢 尿素: 顺势突破确认持续 | 当日 O:1993 H:2018 L:1985 C:1999" in message for message in infos)
    assert any("确认条件: 价格继续站稳入场2034上方，且不跌破止损2006" in message for message in infos)
    assert any("交易计划: 入场2034 止损2012 止盈1 2062 止盈2 2068 第一止盈RR 1.27 准入RR 1.55" in message for message in infos)
    assert any("✅ 盘中重评后转为可执行：评分+42 准入RR 1.55" in message for message in infos)


def test_phase_3_intraday_watchlist_signal_fallback_reverts_to_waiting_confirmation(monkeypatch, capsys) -> None:
    infos: list[str] = []
    trading_states = iter([True, True, False])
    live_plans = iter(
        [
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "strategy_family": "trend_following",
                "score": 42.2,
                "actionable": False,
                "price": 2034.0,
                "entry": 2034.0,
                "stop": 2006.0,
                "tp1": 2051.0,
                "tp2": 2058.0,
                "rr": 0.61,
                "admission_rr": 0.86,
                "entry_family": "trend",
                "entry_signal_type": "TrendBreak",
                "entry_signal_detail": "markup 阶段趋势突破",
                "reversal_status": {},
                "trend_status": {
                    "has_signal": True,
                    "signal_type": "TrendBreak",
                },
            },
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "strategy_family": "trend_following",
                "score": 12.0,
                "actionable": False,
                "price": 2018.0,
                "entry": 2034.0,
                "stop": 2006.0,
                "tp1": 2051.0,
                "tp2": 2058.0,
                "rr": 0.61,
                "admission_rr": 0.86,
                "entry_family": "trend",
                "entry_signal_type": "",
                "entry_signal_detail": "",
                "reversal_status": {},
                "trend_status": {
                    "has_signal": False,
                    "signal_type": "",
                },
            },
        ]
    )

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_daily_with_live_bar",
        lambda symbol, period: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-22"]),
                "open": [1993.0],
                "high": [2018.0],
                "low": [1985.0],
                "close": [1999.0],
            }
        ),
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should use trend-specific builder")),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: next(live_plans),
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[
            {
                "symbol": "UR0",
                "name": "尿素",
                "direction": "long",
                "entry": 2005.0,
                "stop": 1970.0,
                "tp1": 2026.0,
                "tp2": 2029.0,
                "rr": 0.61,
                "admission_rr": 0.68,
                "entry_family": "trend",
                "entry_signal_type": "",
                "entry_signal_detail": "",
                "entry_pool_reason": "顺势突破观察",
            }
        ],
    )

    out = capsys.readouterr().out

    assert "确认已回退，重新等待顺势确认" in out
    assert "继续观望" not in out
    assert any("确认已回退，重新等待顺势确认" in message for message in infos)


def test_phase_3_intraday_preserves_watchlist_story_when_live_plan_has_no_signal(monkeypatch, capsys) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_daily_with_live_bar",
        lambda symbol, period: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-25"]),
                "open": [100.0],
                "high": [101.0],
                "low": [98.0],
                "close": [99.0],
            }
        ),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "symbol": "CU0",
            "name": "铜",
            "direction": "long",
            "strategy_family": "",
            "score": 8.0,
            "actionable": False,
            "entry": 0.0,
            "stop": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr": 0.0,
            "admission_rr": 0.0,
            "entry_family": "",
            "entry_signal_type": "",
            "entry_signal_detail": "",
            "reversal_status": {},
            "trend_status": {"has_signal": False},
        },
    )
    monkeypatch.setattr(daily_workflow, "run_phase_4_holdings", lambda *args, **kwargs: {}, raising=False)

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[
            {
                "symbol": "CU0",
                "name": "铜",
                "direction": "long",
                "strategy_family": "trend_following",
                "entry_family": "trend",
                "entry_signal_type": "",
                "entry_signal_detail": "",
                "entry": 100.0,
                "stop": 96.0,
                "tp1": 106.0,
                "tp2": 110.0,
                "rr": 1.5,
                "admission_rr": 2.5,
            }
        ],
    )

    out = capsys.readouterr().out
    joined = "\n".join(infos) + "\n" + out
    assert "等待顺势确认" in joined
    assert "等待交易故事确认" not in joined


def test_phase_3_intraday_trend_watchlist_ignores_reversal_stage_when_no_signal(monkeypatch, capsys) -> None:
    infos: list[str] = []
    trading_states = iter([True, False])

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "get_daily_with_live_bar",
        lambda symbol, period: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-25"]),
                "open": [100.0],
                "high": [101.0],
                "low": [98.0],
                "close": [99.0],
            }
        ),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "symbol": "CU0",
            "name": "铜",
            "direction": "long",
            "strategy_family": "trend_following",
            "score": 8.0,
            "actionable": False,
            "entry_family": "trend",
            "entry_signal_type": "",
            "entry_signal_detail": "",
            "reversal_status": {
                "current_stage": "等待Spring反转",
                "all_events": [
                    {"date": "2026-04-25", "signal": "Spring", "type": "Spring"},
                ],
            },
            "trend_status": {"has_signal": False},
        },
    )
    monkeypatch.setattr(daily_workflow, "run_phase_4_holdings", lambda *args, **kwargs: {}, raising=False)

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[
            {
                "symbol": "CU0",
                "name": "铜",
                "direction": "long",
                "strategy_family": "trend_following",
                "entry_family": "trend",
            }
        ],
    )

    out = capsys.readouterr().out
    joined = "\n".join(infos) + "\n" + out
    assert "等待顺势确认" in joined
    assert "等待Spring反转" not in joined
    assert "预备信号" not in joined


def test_phase_2_premarket_summary_uses_confirmation_wording(monkeypatch) -> None:
    messages: list[str] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"trend": 18.0, "volume": 6.0, "wyckoff": 8.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: messages.append(msg % args if args else msg),
    )

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
        return {
            "symbol": symbol,
            "name": name,
            "direction": direction,
            "price": 100.0,
            "score": 32.0,
            "actionable": True,
            "entry": 101.0,
            "stop": 95.0,
            "tp1": 110.0,
            "tp2": 118.0,
            "rr": 1.5,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 1,
            "fund_screen_score": 66.0,
            "signal_strength": 0.7,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "markup 阶段顺势回踩",
            "reversal_status": {
                "signal_strength": 0.7,
                "next_expected": "等待顺势确认",
            },
        }

    monkeypatch.setattr(daily_workflow, "analyze_one", fake_analyze_one)

    actionable, watchlist = daily_workflow.phase_2_premarket(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "score": 66.0,
                "attention_score": 66.0,
                "reversal_score": 55.0,
                "trend_score": 61.0,
                "labels": ["趋势候选"],
                "state_labels": ["高关注"],
                "data_coverage": 0.75,
                "reason_summary": "库存下降",
                "entry_pool_reason": "趋势机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    assert any("Phase 2 汇总: 1 个可执行, 0 个等待确认" in msg for msg in messages)
    assert any("确认" in msg and "执行价" in msg for msg in messages)
    assert not any("信号" in msg and "入场" in msg for msg in messages if "Phase 2 汇总" not in msg)


def test_run_trend_strategy_phase2_accepts_trend_universe_without_labels(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"trend": 14.0, "volume": 0.0, "wyckoff": -2.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        calls.append(
            {
                "symbol": kwargs["symbol"],
                "name": kwargs["name"],
                "direction": kwargs["direction"],
                "fund_screen_score": kwargs["cfg"]["fund_screen_score"],
                "reason": kwargs["cfg"]["reason"],
            }
        )
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 12.0,
            "actionable": True,
            "entry": 101.0,
            "stop": 96.0,
            "tp1": 110.0,
            "rr": 1.8,
            "reversal_status": {"signal_strength": 0.2, "next_expected": "继续观察"},
        }

    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_trend_strategy_phase2(
        [
            {
                "symbol": "RM0",
                "name": "菜粕",
                "labels": [],
                "reversal_score": 21.0,
                "trend_score": 78.0,
                "attention_score": 9.0,
                "entry_pool_reason": "趋势机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    assert calls == [
        {
            "symbol": "RM0",
            "name": "菜粕",
            "direction": "long",
            "fund_screen_score": 78.0,
            "reason": "趋势筛选+78, 趋势机会分达标",
        }
    ]


def test_run_reversal_strategy_phase2_prefers_candidate_reversal_direction(monkeypatch) -> None:
    df = pd.DataFrame({"close": [100.0, 99.0, 98.0], "date": pd.date_range("2025-01-01", periods=3)})
    calls: list[str] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: df)
    monkeypatch.setattr(daily_workflow, "score_signals", lambda df, direction, cfg: {"RSI": 30.0})
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        calls.append(kwargs["direction"])
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "strategy_family": "reversal_fundamental",
            "entry_family": "reversal",
            "actionable": False,
            "score": -25.0,
            "rrf_score": 0.1,
            "reversal_status": {},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 75.0,
                "trend_score": 10.0,
                "attention_score": 75.0,
                "reversal_direction": "short",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert calls == ["short"]
    assert watchlist[0]["direction"] == "short"


def test_run_strategy_phase2_passes_position_contract_specs_to_plan_builder(monkeypatch) -> None:
    seen_cfgs: list[dict] = []
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0], "date": pd.date_range("2025-01-01", periods=3)})

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: df)
    monkeypatch.setattr(daily_workflow, "score_signals", lambda df, direction, cfg: {"均线排列": 30.0})

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        seen_cfgs.append(kwargs["cfg"])
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "strategy_family": "trend_following",
            "entry_family": "trend",
            "actionable": True,
            "score": 30.0,
            "rrf_score": 0.1,
            "entry": 101.0,
            "stop": 96.0,
            "tp1": 110.0,
            "rr": 1.8,
            "reversal_status": {},
        }

    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_trend_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": [],
                "trend_score": 78.0,
                "reversal_score": 12.0,
                "attention_score": 78.0,
                "trend_direction": "long",
                "entry_pool_reason": "趋势机会分达标",
            }
        ],
        {
            "pre_market": {"direction_delta": 12.0},
            "positions": {
                "PS0": {
                    "name": "多晶硅",
                    "direction": "long",
                    "multiplier": 3,
                    "margin_rate": 0.12,
                }
            },
        },
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    assert seen_cfgs[0]["contract_specs"]["PS0"]["multiplier"] == 3
    assert seen_cfgs[0]["contract_specs"]["PS0"]["margin_rate"] == 0.12


def test_run_trend_strategy_phase2_downgrades_when_phase2_scores_reverse_candidate_direction(monkeypatch) -> None:
    df = pd.DataFrame({"close": [100.0, 99.0, 98.0], "date": pd.date_range("2025-01-01", periods=3)})
    calls: list[str] = []

    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: df)
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {
            "均线排列": -22.0,
            "MACD": -12.0,
            "动量": -8.0,
            "量价关系": 0.0,
            "VSA信号": 0.0,
            "持仓信号": 0.0,
        },
    )

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        calls.append(kwargs["direction"])
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "strategy_family": "trend_following",
            "entry_family": "trend",
            "actionable": True,
            "score": 31.0,
            "entry": 100.0,
            "stop": 96.0,
            "tp1": 108.0,
            "tp2": 112.0,
            "rr": 2.0,
            "admission_rr": 3.0,
            "reversal_status": {},
        }

    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_trend_strategy_phase2(
        [
            {
                "symbol": "CF0",
                "name": "棉花",
                "labels": ["趋势候选"],
                "trend_score": 78.0,
                "reversal_score": 12.0,
                "attention_score": 78.0,
                "trend_direction": "long",
                "entry_pool_reason": "趋势机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert calls == ["long"]
    assert actionable == []
    assert len(watchlist) == 1
    row = watchlist[0]
    assert row["direction"] == "long"
    assert row["actionable"] is False
    assert row["phase2_decision"] == "watch"
    assert row["phase2_score_direction"] == "short"
    assert row["phase2_candidate_direction"] == "long"
    assert "日线深度方向" in row["downgrade_reason"]


def test_run_reversal_strategy_phase2_uses_candidate_direction_without_false_conflict(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 30.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 30.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.65, "next_expected": "继续观察"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 65.0,
                "trend_score": 47.0,
                "attention_score": 57.8,
                "strategy_score": 65.0,
                "reversal_direction": "short",
                "reason_summary": "库存处于高位100%；最近8周有6周累库",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {
            "pre_market": {"direction_delta": 12.0},
            "positions": {"PS0": {"direction": "short"}},
        },
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0]["direction"] == "short"
    assert bool(watchlist[0]["actionable"]) is False
    assert watchlist[0]["fundamental_conflict"] is False
    assert "基本面方向" not in watchlist[0]["downgrade_reason"]


def test_run_reversal_strategy_phase2_downgrades_long_when_fundamentals_have_not_turned(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 30.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 30.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.75, "next_expected": "继续观察"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 65.0,
                "trend_score": 47.0,
                "attention_score": 57.8,
                "strategy_score": 65.0,
                "reversal_direction": "long",
                "inv_change_4wk": 17.0,
                "inv_percentile": 100.0,
                "receipt_change": 220.0,
                "reason_summary": "库存处于高位100%；最近8周有6周累库",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert bool(watchlist[0]["actionable"]) is False
    assert watchlist[0]["strategy_role"] == "counter_fundamental_rebound_watch"
    assert watchlist[0]["fundamental_extreme_state_confirmed"] is True
    assert watchlist[0]["fundamental_marginal_turn_confirmed"] is False
    assert "边际未转向" in watchlist[0]["downgrade_reason"]


def test_run_reversal_strategy_phase2_does_not_count_oi_as_fundamental_turn(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 99.0, 98.0], "oi": [1000.0, 930.0, 880.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 31.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 31.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.75, "next_expected": "继续观察"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 70.0,
                "trend_score": 32.0,
                "attention_score": 68.0,
                "strategy_score": 70.0,
                "reversal_direction": "long",
                "inv_change_4wk": 17.0,
                "inv_percentile": 100.0,
                "receipt_change": 220.0,
                "oi_vs_price": "减仓下跌",
                "oi_20d_change": -8.0,
                "fundamental_coverage_score": 1.0,
                "fundamental_coverage_status": "complete",
                "fundamental_domains_missing": [],
                "fundamental_missing_domain_reasons": [],
                "fundamental_extreme_state_confirmed": True,
                "fundamental_extreme_state_reasons": ["高库存", "仓单增加"],
                "fundamental_marginal_turn_confirmed": False,
                "fundamental_marginal_turn_reasons": [],
                "fundamental_reversal_confirmed": False,
                "reason_summary": "高库存仍在，持仓回落只是市场动作",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    row = watchlist[0]
    assert row["fundamental_extreme_state_confirmed"] is True
    assert row["fundamental_marginal_turn_confirmed"] is False
    assert row["fundamental_reversal_confirmed"] is False
    assert "持仓" not in "/".join(row["fundamental_marginal_turn_reasons"])
    assert "边际未转向" in row["downgrade_reason"]


def test_run_reversal_strategy_phase2_uses_snapshot_gate_and_exposes_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0], "oi": [1000.0, 980.0, 960.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 29.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 29.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.82, "next_expected": "等待确认"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 72.0,
                "trend_score": 36.0,
                "attention_score": 70.0,
                "strategy_score": 72.0,
                "reversal_direction": "long",
                "fundamental_coverage_score": 1.0,
                "fundamental_coverage_status": "complete",
                "fundamental_domains_present": ["inventory", "warehouse_receipt", "spot_basis"],
                "fundamental_domains_missing": [],
                "fundamental_missing_domain_reasons": [],
                "fundamental_extreme_state_confirmed": True,
                "fundamental_extreme_state_reasons": ["高库存"],
                "fundamental_marginal_turn_confirmed": True,
                "fundamental_marginal_turn_reasons": ["去库启动", "仓单回落"],
                "fundamental_reversal_confirmed": True,
                "reason_summary": "高库存后开始去库，仓单回落",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    row = actionable[0]
    assert row["fundamental_reversal_confirmed"] is True
    assert row["fundamental_coverage_score"] == 1.0
    assert row["fundamental_coverage_status"] == "complete"
    assert row["fundamental_domains_missing"] == []
    assert row["fundamental_missing_domain_reasons"] == []
    assert row["fundamental_extreme_state_reasons"] == ["高库存"]
    assert row["fundamental_marginal_turn_reasons"] == ["去库启动", "仓单回落"]


def test_run_reversal_strategy_phase2_confirms_turn_only_after_extreme_state(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0], "oi": [1000.0, 980.0, 940.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 24.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 24.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.75, "next_expected": "继续观察"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "RB0",
                "name": "螺纹钢",
                "labels": ["反转候选"],
                "reversal_score": 58.0,
                "trend_score": 33.0,
                "attention_score": 58.0,
                "strategy_score": 58.0,
                "reversal_direction": "long",
                "inv_change_4wk": -3.0,
                "inv_percentile": 54.0,
                "receipt_change": -20.0,
                "oi_vs_price": "减仓下跌",
                "oi_20d_change": -8.0,
                "reason_summary": "短线价格反弹但库存未进入极端",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    assert watchlist[0]["fundamental_extreme_state_confirmed"] is False
    assert watchlist[0]["fundamental_marginal_turn_confirmed"] is True
    assert "未进入极端失衡" in watchlist[0]["downgrade_reason"]


def test_run_reversal_strategy_phase2_accepts_turn_after_extreme_state(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0], "oi": [1000.0, 960.0, 900.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 27.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 27.0,
            "actionable": True,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "reversal_status": {"signal_strength": 0.82, "next_expected": "等待确认"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "PS0",
                "name": "多晶硅",
                "labels": ["反转候选"],
                "reversal_score": 68.0,
                "trend_score": 36.0,
                "attention_score": 64.0,
                "strategy_score": 68.0,
                "reversal_direction": "long",
                "inv_change_4wk": -4.0,
                "inv_percentile": 100.0,
                "receipt_change": -120.0,
                "oi_vs_price": "减仓下跌",
                "oi_20d_change": -10.0,
                "reason_summary": "高库存后开始去库，仓单回落",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    assert actionable[0]["fundamental_extreme_state_confirmed"] is True
    assert actionable[0]["fundamental_marginal_turn_confirmed"] is True
    assert actionable[0]["fundamental_reversal_confirmed"] is True


def test_run_reversal_strategy_phase2_accepts_profit_stress_repair_after_extreme_state(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_workflow,
        "get_daily",
        lambda symbol: pd.DataFrame({"close": [100.0, 101.0, 102.0], "oi": [1000.0, 1010.0, 1020.0]}),
    )
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"technical_long": 28.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_build_trade_plan_from_daily_df(**kwargs) -> dict:
        return {
            "symbol": kwargs["symbol"],
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "score": 28.0,
            "actionable": True,
            "entry": 11280.0,
            "stop": 10900.0,
            "tp1": 12000.0,
            "tp2": 12400.0,
            "rr": 1.89,
            "admission_rr": 2.95,
            "reversal_status": {"signal_strength": 0.78, "next_expected": "等待确认"},
        }

    monkeypatch.setattr(
        daily_workflow.reversal_pre_market,
        "build_trade_plan_from_daily_df",
        fake_build_trade_plan_from_daily_df,
    )

    actionable, watchlist = daily_workflow.run_reversal_strategy_phase2(
        [
            {
                "symbol": "LH0",
                "name": "生猪",
                "labels": ["反转候选"],
                "reversal_score": 72.0,
                "trend_score": 38.0,
                "attention_score": 68.0,
                "strategy_score": 72.0,
                "reversal_direction": "long",
                "hog_profit": -42.0,
                "hog_price_trend": 6.5,
                "reason_summary": "养殖利润深亏但现货开始走强",
                "entry_pool_reason": "反转机会分达标",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert len(actionable) == 1
    assert watchlist == []
    assert actionable[0]["fundamental_extreme_state_confirmed"] is True
    assert actionable[0]["fundamental_marginal_turn_confirmed"] is True
    assert "利润压力缓和" in actionable[0]["fundamental_marginal_turn_reasons"]


def test_main_no_monitor_runs_phase_4_holdings_even_without_new_candidates(monkeypatch) -> None:
    calls: list[tuple] = []

    monkeypatch.setattr(daily_workflow, "load_config", lambda: {"pre_market": {}, "intraday": {}})
    monkeypatch.setattr(daily_workflow, "phase_0_prefetch", lambda: {})
    monkeypatch.setattr(daily_workflow, "phase_1_screen", lambda all_data, threshold=10: [])
    monkeypatch.setattr(daily_workflow, "build_trend_universe", lambda all_data, config: [])
    monkeypatch.setattr(daily_workflow, "run_dual_strategy_phase2", lambda *args, **kwargs: ([], [], {}))
    monkeypatch.setattr(daily_workflow, "save_targets", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
        raising=False,
    )
    monkeypatch.setattr(
        daily_workflow.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            no_monitor=True,
            skip_screen=False,
            resume=False,
            threshold=10,
            max_picks=6,
            period="5",
            interval=60,
        ),
    )
    monkeypatch.setattr(daily_workflow.log, "info", lambda *args, **kwargs: None)

    daily_workflow.main()

    assert calls, "expected Phase 4 holdings advice to run even when there are no new candidates"
    assert calls[0][1]["period"] == "5"
    assert calls[0][1]["emit_terminal"] is False


def test_phase_3_intraday_emits_holdings_alert_only_on_action_change(monkeypatch, capsys) -> None:
    trading_states = iter([True, False])

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(daily_workflow.log, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: print("  🔔 H001 生猪: 继续持有（原有顺势逻辑仍成立；保护止损上移 11142 -> 11151） | 当前价 11540 止损 11151 第一止盈 12265") or {"H001": "上移止损"},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[],
    )

    out = capsys.readouterr().out
    assert "H001 生猪: 继续持有（原有顺势逻辑仍成立；保护止损上移 11142 -> 11151）" in out


def test_run_phase_4_holdings_reuses_supplied_live_frame(monkeypatch) -> None:
    infos: list[str] = []
    supplied_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-23"]),
            "open": [11475.0],
            "high": [11585.0],
            "low": [11385.0],
            "close": [11500.0],
        }
    )

    monkeypatch.setattr(daily_workflow, "find_daily_holdings_workbook", lambda **kwargs: Path("/tmp/holdings.xlsx"))
    monkeypatch.setattr(
        daily_workflow,
        "load_holding_contexts",
        lambda path: [
            {
                "record_id": "H001",
                "holding": {
                    "record_id": "H001",
                    "symbol": "LH0",
                    "name": "生猪",
                    "direction": "long",
                    "size": 1,
                    "entry_price": 11450.0,
                },
                "recommendation": {
                    "record_id": "H001",
                    "symbol": "LH0",
                    "name": "生猪",
                    "direction": "long",
                    "entry_family": "trend",
                    "entry": 11450.0,
                    "first_stop": 11128.0,
                    "stop": 11128.0,
                    "tp1": 12265.0,
                    "tp2": 12445.0,
                    "rr": 2.70,
                    "admission_rr": 3.29,
                },
            }
        ],
    )
    monkeypatch.setattr(
        daily_workflow,
        "get_daily_with_live_bar",
        lambda symbol, period: (_ for _ in ()).throw(AssertionError("should reuse supplied live frame")),
    )
    monkeypatch.setattr(
        daily_workflow,
        "assess_active_trend_hold_from_daily_df",
        lambda **kwargs: {"hold_valid": True},
    )
    monkeypatch.setattr(
        daily_workflow,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should use trend-specific builder")),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "entry": float(kwargs["df"].iloc[-1]["close"]),
            "first_stop": 11141.0,
            "stop": 11141.0,
            "tp1": 12265.0,
            "tp2": 12445.0,
            "rr": 2.13,
            "admission_rr": 2.63,
        },
    )
    monkeypatch.setattr(
        daily_workflow.log,
        "info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)

    actions = daily_workflow.run_phase_4_holdings(
        {"pre_market": {}},
        period="5",
        emit_terminal=False,
        live_frames={"LH0": supplied_df},
    )

    assert actions == {"H001": "继续持有|上移止损|11141"}
    assert any("当前重评: 入场11500" in message for message in infos)


def test_phase_3_intraday_passes_watchlist_live_frame_to_phase_4(monkeypatch) -> None:
    trading_states = iter([True, False])
    captured_live_frames: dict[str, pd.DataFrame] = {}

    class DummyMonitor:
        def describe_subscription(self) -> str:
            return "LH0 daily"

        def daily_df(self, symbol: str) -> pd.DataFrame:
            assert symbol == "LH0"
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-04-23"]),
                    "open": [11475.0],
                    "high": [11585.0],
                    "low": [11385.0],
                    "close": [11500.0],
                }
            )

        def wait_interval(self, seconds: float) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(daily_workflow, "get_log_path", lambda: "/tmp/test.log")
    monkeypatch.setattr(daily_workflow, "time_to_next_session", lambda: 0)
    monkeypatch.setattr(daily_workflow, "try_create_phase3_monitor", lambda *args, **kwargs: DummyMonitor())
    monkeypatch.setattr(daily_workflow, "is_trading_hours", lambda: next(trading_states))
    monkeypatch.setattr(daily_workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(daily_workflow.log, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(daily_workflow.log, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        daily_workflow,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should use trend-specific builder")),
    )
    monkeypatch.setattr(
        daily_workflow.trend_pre_market,
        "build_trade_plan_from_daily_df",
        lambda **kwargs: {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "strategy_family": "trend_following",
            "score": 42.0,
            "actionable": True,
            "price": 11500.0,
            "entry": 11500.0,
            "stop": 11141.0,
            "tp1": 12265.0,
            "tp2": 12445.0,
            "rr": 2.13,
            "admission_rr": 2.63,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "趋势回撤",
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Pullback",
                "signal_bar": {"low": 11385.0},
                "confidence": 0.72,
            },
            "trend_status": {
                "has_signal": True,
                "signal_type": "Pullback",
            },
        },
    )
    monkeypatch.setattr(
        daily_workflow,
        "run_phase_4_holdings",
        lambda *args, **kwargs: captured_live_frames.update(kwargs.get("live_frames") or {}) or {},
        raising=False,
    )

    daily_workflow.phase_3_intraday(
        [],
        {"intraday": {}, "pre_market": {}},
        period="5",
        interval=0,
        watchlist=[
            {
                "symbol": "LH0",
                "name": "生猪",
                "direction": "long",
                "entry": 11450.0,
                "stop": 11128.0,
                "tp1": 12265.0,
                "tp2": 12445.0,
                "rr": 2.70,
                "admission_rr": 3.29,
                "entry_family": "trend",
                "entry_signal_type": "Pullback",
                "entry_signal_detail": "趋势回撤",
            }
        ],
    )

    assert "LH0" in captured_live_frames
    assert float(captured_live_frames["LH0"].iloc[-1]["close"]) == 11500.0
