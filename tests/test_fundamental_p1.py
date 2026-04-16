"""
P1 基本面引擎单元测试（legacy + regime 纯逻辑，不依赖外网时可跑）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fundamental_regime import (  # noqa: E402
    classify_lh_regime,
    _lh_regime_hog_layers,
)
import daily_workflow  # noqa: E402


class TestClassifyLhRegime:
    def test_missing_margin_low_range_distress(self):
        assert classify_lh_regime(None, True, 10.0) == "distress"

    def test_missing_margin_mid_repair(self):
        assert classify_lh_regime(None, True, 30.0) == "repair"

    def test_pm_deep_distress(self):
        assert classify_lh_regime(-20.0, False, 50.0) == "distress"

    def test_pm_boom(self):
        assert classify_lh_regime(15.0, False, 50.0) == "boom"


class TestRegimeHogDistressVsBounce:
    """distress 下现货短期反弹(pt>3)不应像 legacy 那样扣 5 分。"""

    def test_distress_hot_spot_no_penalty(self):
        hog = {"profit_margin": -20.0, "price_trend": 5.0}
        score, details, _ = _lh_regime_hog_layers("distress", hog, range_pct=30.0)
        assert score > 0
        assert any("不扣分" in d for d in details)

    def test_legacy_style_would_penalize(self):
        """文档化：旧逻辑等价于 pt>3 → -5（此处用独立心算，不 import legacy hog）。"""
        # regime distress 对 pt>3 净效果应 >= 旧逻辑 (-5 来自猪) 的差值约 +5
        hog = {"profit_margin": -20.0, "price_trend": 5.0}
        s_reg, _, _ = _lh_regime_hog_layers("distress", hog, range_pct=30.0)
        # 若用 neutral 处理同一头猪利润 + 同一 pt
        s_neutral, _, _ = _lh_regime_hog_layers("neutral", hog, range_pct=30.0)
        assert s_reg >= s_neutral


def test_score_symbol_legacy_import():
    from fundamental_legacy import score_symbol_legacy

    assert callable(score_symbol_legacy)


def test_build_phase1_summary_returns_cn_metadata() -> None:
    summary = daily_workflow.build_phase1_summary(
        top_n=6,
        sort_field="attention_score",
        label_field="labels",
    )

    assert summary == {
        "阶段": "Phase 1 发现机会",
        "关注池上限": 6,
        "排序字段": "关注优先级分",
        "标签字段": ["反转候选", "趋势候选", "双标签候选", "数据覆盖不足"],
    }


def test_phase_2_premarket_bridges_phase1_fields(monkeypatch) -> None:
    monkeypatch.setattr(daily_workflow, "get_daily", lambda symbol: {"symbol": symbol})
    monkeypatch.setattr(
        daily_workflow,
        "score_signals",
        lambda df, direction, cfg: {"trend": -18.0, "volume": 6.0, "wyckoff": -8.0},
    )
    monkeypatch.setattr(daily_workflow, "resolve_phase2_direction", lambda **kwargs: "long")

    def fake_analyze_one(symbol: str, name: str, direction: str, cfg: dict) -> dict:
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
                "reversal_score": 55.0,
                "trend_score": 11.0,
                "labels": ["趋势候选"],
                "state_labels": ["高关注"],
                "data_coverage": 0.75,
                "reason_summary": "库存下降",
                "entry_pool_reason": "配置桥接",
            }
        ],
        {"pre_market": {"direction_delta": 12.0}},
        max_picks=3,
    )

    assert actionable == []
    assert len(watchlist) == 1
    row = watchlist[0]
    assert row["attention_score"] == 66.0
    assert row["reversal_score"] == 55.0
    assert row["trend_score"] == 11.0
    assert row["phase1_labels"] == ["趋势候选"]
    assert row["phase1_state_labels"] == ["高关注"]
    assert row["phase1_data_coverage"] == 0.75
    assert row["phase1_reason_summary"] == "库存下降"


def test_save_targets_writes_phase1_summary_to_payload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    targets = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 66.0,
            "price": 100.0,
            "entry": 101.0,
            "stop": 95.0,
            "tp1": 110.0,
            "rr": 1.5,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 2,
            "fund_screen_score": 66.0,
            "signal_strength": 0.7,
            "labels": ["趋势候选"],
            "state_labels": [],
            "data_coverage": 1.0,
            "reason_summary": "库存下降",
            "phase1_labels": ["趋势候选"],
            "phase1_state_labels": [],
            "phase1_data_coverage": 1.0,
            "phase1_reason_summary": "库存下降",
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, [], phase1_summary=summary)

    payload = json.loads((tmp_path / "2026-04-16_targets.json").read_text(encoding="utf-8"))
    assert payload["phase1_summary"] == summary
    assert payload["phase1_summary"]["排序字段"] == "关注优先级分"
    assert payload["phase1_summary"]["标签字段"] == [
        "反转候选",
        "趋势候选",
        "双标签候选",
        "数据覆盖不足",
    ]


def test_save_targets_keeps_actionable_risk_flags_visible(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    targets = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 66.0,
            "price": 100.0,
            "entry": 101.0,
            "stop": 95.0,
            "tp1": 110.0,
            "tp2": 118.0,
            "rr": 1.5,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 2,
            "fund_screen_score": 66.0,
            "attention_score": 72.0,
            "signal_strength": 0.7,
            "direction_conflict": True,
            "score_signs_support_direction": False,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "库存下降",
            "reason_summary": "库存下降",
            "entry_pool_reason": "基本面达标",
            "actionable": True,
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Spring",
                "signal_date": "2026-04-15",
                "entry_mode": "reversal",
                "signal_strength": 0.7,
                "current_stage": "accumulation",
                "confidence": 0.8,
                "signal_detail": "测试信号",
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, [], phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "⚠️P1P2异号/逆势" in md_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
