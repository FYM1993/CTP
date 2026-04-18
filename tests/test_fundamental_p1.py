"""
P1/P2 工作流回归测试。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cli import daily_workflow  # noqa: E402


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
    assert row["fund_screen_score"] == 66.0


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


def test_save_targets_does_not_emit_obsolete_phase1_direction_risk_flags(
    monkeypatch,
    tmp_path,
) -> None:
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
            "direction_conflict": False,
            "score_signs_support_direction": True,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "库存下降",
            "reason_summary": "库存下降",
            "entry_pool_reason": "反转机会分达标",
            "actionable": True,
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Spring",
                "signal_date": "2026-04-15",
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
    assert "⚠️P1P2异号/逆势" not in md_text


def test_save_targets_renders_trend_plan_from_entry_family(monkeypatch, tmp_path) -> None:
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
            "score": 31.0,
            "price": 100.0,
            "entry": 101.0,
            "stop": 97.0,
            "tp1": 109.0,
            "tp2": 113.0,
            "rr": 2.0,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 1,
            "fund_screen_score": 66.0,
            "attention_score": 72.0,
            "signal_strength": 0.7,
            "labels": ["趋势候选"],
            "phase1_labels": ["趋势候选"],
            "phase1_reason_summary": "库存下降",
            "reason_summary": "库存下降",
            "entry_pool_reason": "趋势机会分达标",
            "actionable": True,
            "entry_family": "trend",
            "entry_signal_type": "Pullback",
            "entry_signal_detail": "markup 阶段顺势回踩",
            "reversal_status": {
                "has_signal": False,
                "signal_type": "",
                "signal_date": "",
                "signal_strength": 0.0,
                "current_stage": "markup",
                "confidence": 0.0,
                "signal_detail": "",
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, [], phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "**入场信号**: [顺势] 顺势回撤(Pullback)" in md_text
    assert "- 信号详情: markup 阶段顺势回踩" in md_text
    assert "尚无有效入场信号" not in md_text


def test_load_targets_normalizes_legacy_trend_snapshot(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    payload = {
        "date": "2026-04-16",
        "targets": [
            {
                "symbol": "LH0",
                "name": "生猪",
                "direction": "long",
                "score": 28.0,
                "actionable": True,
                "price": 100.0,
                "entry": 101.0,
                "stop": 97.0,
                "tp1": 109.0,
                "tp2": 113.0,
                "rr": 2.0,
                "reversal_status": {
                    "has_signal": False,
                    "entry_mode": "trend",
                    "signal_type": "Pullback",
                    "signal_detail": "legacy trend snapshot",
                    "signal_date": "",
                    "current_stage": "markup",
                    "confidence": 0.0,
                },
            }
        ],
        "watchlist": [],
        "phase1_summary": {},
    }
    (tmp_path / "2026-04-16_targets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snapshot = daily_workflow.load_targets()

    assert snapshot is not None
    targets, watchlist = snapshot
    assert watchlist == []
    assert len(targets) == 1
    row = targets[0]
    assert row["entry_family"] == "trend"
    assert row["entry_signal_type"] == "Pullback"
    assert row["entry_signal_detail"] == "legacy trend snapshot"
    assert daily_workflow._resolve_entry_signal(row)["has_signal"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
