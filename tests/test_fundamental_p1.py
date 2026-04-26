"""
P1/P2 工作流回归测试。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
    assert "**交易故事确认**: [反转] 弹簧(Spring) (2026-04-15)" in md_text
    assert "- 确认详情: 测试信号" in md_text
    assert "- 入场依据: 当前价（确认于2026-04-15触发，新鲜可执行）" in md_text
    assert "⚠️P1P2异号/逆势" not in md_text


def test_save_targets_renders_downgrade_and_extended_target_notes(monkeypatch, tmp_path) -> None:
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
            "tp1": 104.0,
            "tp2": 118.0,
            "rr": 0.50,
            "admission_rr": 2.83,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 2,
            "fund_screen_score": 66.0,
            "attention_score": 72.0,
            "signal_strength": 0.75,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "库存下降",
            "reason_summary": "库存下降",
            "entry_pool_reason": "反转机会分达标",
            "actionable": True,
            "entry_family": "reversal",
            "entry_plan_type": "extended_target",
            "management_note": "第一止盈附近减仓或收紧保护止损，剩余仓位才看第二止盈",
            "phase2_rr_gate_passed": True,
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Spring",
                "signal_date": "2026-04-15",
                "signal_strength": 0.75,
                "current_stage": "accumulation",
                "confidence": 0.8,
                "signal_detail": "测试信号",
            },
        }
    ]
    watchlist = [
        {
            "symbol": "PS0",
            "name": "多晶硅",
            "direction": "long",
            "score": 30.0,
            "price": 43125.0,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "admission_rr": 4.55,
            "rrf_score": 0.1000,
            "rank_p1": 2,
            "rank_p2": 2,
            "fund_screen_score": 65.0,
            "attention_score": 57.8,
            "signal_strength": 0.65,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "库存处于高位100%",
            "reason_summary": "库存处于高位100%",
            "entry_pool_reason": "反转机会分达标",
            "actionable": False,
            "entry_family": "reversal",
            "downgrade_reason": "价格跌回突破位44350下方；基本面方向做空与交易方向做多冲突",
            "phase2_price_gate_passed": False,
            "phase2_risk_gate_passed": True,
            "phase2_rr_gate_passed": True,
            "reversal_status": {
                "has_signal": True,
                "signal_type": "SOS",
                "signal_date": "2026-04-15",
                "signal_strength": 0.65,
                "current_stage": "反转确认",
                "confidence": 0.7,
                "signal_detail": "测试弱确认",
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, watchlist, phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "- 计划类型: 依赖第二止盈的延展计划" in md_text
    assert "- 仓位管理: 第一止盈附近减仓或收紧保护止损，剩余仓位才看第二止盈" in md_text
    assert "价格跌回突破位44350下方；基本面方向做空与交易方向做多冲突" in md_text
    assert "确认于2026-04-15触发，但未达执行条件" in md_text
    assert "测试弱确认" in md_text
    downgraded_section = md_text.split("### 多晶硅(主力)", 1)[1]
    assert "新鲜可执行" not in downgraded_section


def test_save_targets_describes_strategy_pool_and_playbook_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets([], [], phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "系统采用策略池：各策略独立命中，多个同方向命中代表共振增强" in md_text
    assert "最终交易只能选择一个交易剧本执行" in md_text
    assert "系统以趋势跟随为核心策略" not in md_text
    assert "反转确认优先级高于顺势确认" not in md_text


def test_save_targets_renders_strategy_hit_matrix_for_symbol(monkeypatch, tmp_path) -> None:
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
            "symbol": "CF0",
            "name": "棉花",
            "direction": "long",
            "score": 41.0,
            "price": 15000.0,
            "entry": 15020.0,
            "stop": 14680.0,
            "tp1": 15600.0,
            "tp2": 16000.0,
            "rr": 1.70,
            "admission_rr": 2.88,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 1,
            "fund_screen_score": 78.0,
            "attention_score": 82.0,
            "signal_strength": 0.75,
            "labels": ["趋势候选"],
            "phase1_labels": ["趋势候选"],
            "phase1_reason_summary": "趋势技术分达标",
            "reason_summary": "趋势技术分达标",
            "entry_pool_reason": "趋势机会分达标",
            "actionable": True,
            "entry_family": "trend",
            "entry_signal_type": "TrendBreak",
            "entry_signal_detail": "趋势突破确认",
            "market_stage": "clear_trend",
            "strategy_resonance": "same_direction",
            "selected_playbook": "trend_continuation",
            "confluence_quality": "independent",
            "independent_evidence_count": 4,
            "independent_evidence_summary": ["技术趋势", "持仓资金", "库存供需", "仓单压力"],
            "unselected_playbook_reason": "存在明确趋势，基本面同向只作为共振增强，不单独切换到均值回归剧本",
            "strategy_pool_hits": [
                {
                    "strategy_family": "trend_following",
                    "direction": "long",
                    "actionable": True,
                    "score": 41.0,
                    "evidence_domains": ["technical_trend", "positioning_oi"],
                },
                {
                    "strategy_family": "reversal_fundamental",
                    "direction": "long",
                    "actionable": True,
                    "score": 28.0,
                    "evidence_domains": ["inventory_supply", "warehouse_receipt"],
                },
            ],
            "trend_status": {
                "has_signal": True,
                "signal_type": "TrendBreak",
                "signal_detail": "趋势突破确认",
            },
            "reversal_status": {
                "has_signal": False,
                "current_stage": "趋势延续",
                "confidence": 0.0,
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, [], phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "**策略池命中**: 同方向共振，执行剧本: 趋势延续" in md_text
    assert "**市场阶段**: 明确趋势" in md_text
    assert "**共振质量**: 独立共振（4个独立证据）" in md_text
    assert "**独立证据**: 技术趋势 / 持仓资金 / 库存供需 / 仓单压力" in md_text
    assert "**未选剧本原因**: 存在明确趋势，基本面同向只作为共振增强，不单独切换到均值回归剧本" in md_text
    assert "- 趋势跟随: 做多，可执行，评分+41" in md_text
    assert "- 基本面均值回归: 做多，可执行，评分+28" in md_text


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
    assert "## 可执行品种一览" in md_text
    assert "| ✅可执行 | 生猪(主力) |" in md_text
    assert "**交易故事确认**: [顺势] 顺势回撤(Pullback)" in md_text
    assert "- 确认详情: markup 阶段顺势回踩" in md_text
    assert "- 入场依据: 当前价（顺势回撤(Pullback)条件满足，按顺势计划执行）" in md_text
    assert "**入场信号**:" not in md_text
    assert "尚无有效入场信号" not in md_text


def test_save_targets_uses_trade_story_wording_for_watchlist(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    watchlist = [
        {
            "symbol": "LH0",
            "name": "生猪",
            "direction": "long",
            "score": 18.0,
            "price": 100.0,
            "entry": 101.0,
            "stop": 97.0,
            "tp1": 109.0,
            "tp2": 113.0,
            "rr": 0.8,
            "admission_rr": 0.8,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 2,
            "fund_screen_score": 66.0,
            "attention_score": 72.0,
            "signal_strength": 0.7,
            "labels": ["趋势候选"],
            "phase1_labels": ["趋势候选"],
            "phase1_reason_summary": "库存下降",
            "reason_summary": "库存下降",
            "entry_pool_reason": "趋势机会分达标",
            "actionable": False,
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
        },
        {
            "symbol": "RB0",
            "name": "螺纹钢",
            "direction": "long",
            "score": 12.0,
            "price": 3200.0,
            "entry": 0.0,
            "stop": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr": 0.0,
            "rrf_score": 0.0833,
            "rank_p1": 2,
            "rank_p2": 5,
            "fund_screen_score": 60.0,
            "attention_score": 68.0,
            "signal_strength": 0.2,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "基差偏强",
            "reason_summary": "基差偏强",
            "entry_pool_reason": "反转机会待确认",
            "actionable": False,
            "entry_family": "reversal",
            "entry_signal_type": "",
            "entry_signal_detail": "",
            "reversal_status": {
                "has_signal": False,
                "signal_type": "",
                "signal_date": "",
                "signal_strength": 0.0,
                "current_stage": "accumulation",
                "confidence": 0.0,
                "signal_detail": "",
                "next_expected": "",
            },
        },
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets([], watchlist, phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    assert "## 今日等待确认（尚无可执行品种）" in md_text
    assert "已出现顺势确认（顺势回撤），待执行条件改善" in md_text
    assert "**交易故事确认**: ⏳ 等待反转确认" in md_text
    assert "- ⚠️ 未达执行条件: 交易故事尚未确认" in md_text
    assert "- 反转确认: Spring(假跌破收回) 或 SOS(放量突破)" in md_text
    assert "- 顺势确认: 缩量回踩MA20企稳 或 放量突破前高" in md_text
    assert "### 螺纹钢(主力) — 做多 — 等待确认" in md_text
    assert "今日观望" not in md_text
    assert "待入场条件改善" not in md_text
    assert "无有效入场信号" not in md_text


def test_save_targets_renders_confirmed_fundamental_snapshot(monkeypatch, tmp_path) -> None:
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
            "symbol": "PS0",
            "name": "多晶硅",
            "direction": "long",
            "score": 29.0,
            "price": 43125.0,
            "entry": 43125.0,
            "stop": 40988.0,
            "tp1": 50175.0,
            "tp2": 52842.0,
            "rr": 3.30,
            "rrf_score": 0.1234,
            "rank_p1": 1,
            "rank_p2": 1,
            "fund_screen_score": 72.0,
            "attention_score": 70.0,
            "signal_strength": 0.82,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "高库存后开始去库，仓单回落",
            "reason_summary": "高库存后开始去库，仓单回落",
            "entry_pool_reason": "反转机会分达标",
            "actionable": True,
            "entry_family": "reversal",
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
            "reversal_status": {
                "has_signal": True,
                "signal_type": "Spring",
                "signal_date": "2026-04-15",
                "signal_strength": 0.82,
                "current_stage": "accumulation",
                "confidence": 0.8,
                "signal_detail": "反转确认",
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets(targets, [], phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    section = md_text.split("### 多晶硅(主力)", 1)[1].split("**Wyckoff阶段**", 1)[0]
    assert "**基本面反转**: 已确认" in section
    assert "- 基本面覆盖: 完整 (100%)" in section
    assert "- 已有产业证据: 库存 / 仓单 / 现货/基差" in section
    assert "- 缺失产业证据: 无" in section
    assert "- 极值状态: 高库存" in section
    assert "- 边际转向: 去库启动 / 仓单回落" in section


def test_save_targets_renders_missing_fundamental_snapshot_as_observation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    watchlist = [
        {
            "symbol": "RB0",
            "name": "螺纹钢",
            "direction": "long",
            "score": 24.0,
            "price": 3200.0,
            "entry": 3200.0,
            "stop": 3100.0,
            "tp1": 3500.0,
            "tp2": 3650.0,
            "rr": 3.0,
            "rrf_score": 0.1000,
            "rank_p1": 1,
            "rank_p2": 3,
            "fund_screen_score": 64.0,
            "attention_score": 66.0,
            "signal_strength": 0.65,
            "labels": ["反转候选"],
            "phase1_labels": ["反转候选"],
            "phase1_reason_summary": "库存处于高位，但产业链证据不足",
            "reason_summary": "库存处于高位，但产业链证据不足",
            "entry_pool_reason": "反转机会分达标",
            "actionable": False,
            "entry_family": "reversal",
            "downgrade_reason": "基本面数据不足：缺少现货/基差数据/缺少仓单数据，仅保留观察",
            "oi_vs_price": "减仓下跌",
            "oi_20d_change": -8.0,
            "fundamental_coverage_score": 0.33,
            "fundamental_coverage_status": "partial",
            "fundamental_domains_present": ["inventory"],
            "fundamental_domains_missing": ["spot_basis", "warehouse_receipt"],
            "fundamental_missing_domain_reasons": ["缺少现货/基差数据", "缺少仓单数据"],
            "fundamental_extreme_state_confirmed": True,
            "fundamental_extreme_state_reasons": ["高库存"],
            "fundamental_marginal_turn_confirmed": False,
            "fundamental_marginal_turn_reasons": [],
            "fundamental_reversal_confirmed": False,
            "reversal_status": {
                "has_signal": True,
                "signal_type": "SOS",
                "signal_date": "2026-04-15",
                "signal_strength": 0.65,
                "current_stage": "反转确认",
                "confidence": 0.7,
                "signal_detail": "市场动作开始改善",
            },
        }
    ]

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    daily_workflow.save_targets([], watchlist, phase1_summary=summary)

    md_text = (tmp_path / "2026-04-16_targets.md").read_text(encoding="utf-8")
    section = md_text.split("### 螺纹钢(主力)", 1)[1].split("**Wyckoff阶段**", 1)[0]
    assert "**基本面反转**: 未确认，仅观察" in section
    assert "- 基本面覆盖: 不完整 (33%)" in section
    assert "- 已有产业证据: 库存" in section
    assert "- 缺失产业证据: 现货/基差 / 仓单" in section
    assert "- 不建议开仓: 缺少现货/基差数据 / 缺少仓单数据" in section
    assert "持仓" not in section.split("**基本面数据**", 1)[1]


def test_save_targets_cleans_nested_numpy_in_strategy_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(daily_workflow, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(daily_workflow, "_today_json_path", lambda: tmp_path / "2026-04-16_targets.json")
    monkeypatch.setattr(daily_workflow, "_today_md_path", lambda: tmp_path / "2026-04-16_targets.md")

    class FixedDateTime:
        @classmethod
        def now(cls):
            return datetime(2026, 4, 16, 9, 30)

    monkeypatch.setattr(daily_workflow, "datetime", FixedDateTime)

    summary = daily_workflow.build_phase1_summary(6, "attention_score", "labels")
    strategy_results = {
        "reversal": {
            "actionable": [
                {
                    "symbol": "LH0",
                    "score": np.int64(31),
                    "rr": np.float64(2.0),
                }
            ]
        }
    }

    daily_workflow.save_targets([], [], phase1_summary=summary, grouped_results=strategy_results)

    payload = json.loads((tmp_path / "2026-04-16_targets.json").read_text(encoding="utf-8"))
    saved = payload["strategy_results"]["reversal"]["actionable"][0]
    assert saved["score"] == 31
    assert saved["rr"] == 2.0


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
