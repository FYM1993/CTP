from __future__ import annotations

WORKFLOW_CONTRACT_LINES = [
    "第一阶段负责发现值得继续跟踪的品种，不负责直接给出最终交易计划。",
    "第二阶段负责确定交易方向、策略类型、入场依据和原始交易计划。",
    "第三阶段负责在盘中更新第二阶段已经确定的计划；即使盘中出现了更好的入场时机，也必须继承原来的策略判断，不负责改换策略。",
    "第四阶段负责管理已经持有的仓位；它必须使用与第三阶段同一份盘中快照和相同的策略解释，再结合持仓成本给出持仓建议。",
]


def build_workflow_contract_summary() -> str:
    return "工作流原则：第一阶段选品种；第二阶段定计划；第三阶段只更新计划；第四阶段只管理持仓。"


def build_workflow_contract_lines() -> list[str]:
    return list(WORKFLOW_CONTRACT_LINES)
