from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared.workflow_contract import build_workflow_contract_lines, build_workflow_contract_summary  # noqa: E402


def test_workflow_contract_summary_describes_all_four_phases_in_chinese() -> None:
    summary = build_workflow_contract_summary()

    assert "第一阶段选品种" in summary
    assert "第二阶段定计划" in summary
    assert "第三阶段只更新计划" in summary
    assert "第四阶段只管理持仓" in summary


def test_workflow_contract_lines_explain_phase_linkage_in_chinese() -> None:
    lines = build_workflow_contract_lines()

    assert any("盘中出现了更好的入场时机" in line for line in lines)
    assert any("同一份盘中快照" in line for line in lines)
    assert any("不负责改换策略" in line for line in lines)
