from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase1Label(str, Enum):
    REVERSAL = "反转候选"
    TREND = "趋势候选"
    BOTH = "双标签候选"
    LOW_COVERAGE = "数据覆盖不足"


class Phase1StateLabel(str, Enum):
    LOW_LEVEL_EXIT = "低位出清"
    HIGH_LEVEL_EXPANSION = "高位扩产"
    TIGHTENING_BALANCE = "紧平衡强化"
    OVERSUPPLY_DEEPENING = "过剩深化"


@dataclass(slots=True)
class Phase1Candidate:
    symbol: str
    name: str
    reversal_score: float
    trend_score: float
    attention_score: float
    data_coverage: float
    labels: list[Phase1Label]
    state_labels: list[Phase1StateLabel]
    reason_summary: str
