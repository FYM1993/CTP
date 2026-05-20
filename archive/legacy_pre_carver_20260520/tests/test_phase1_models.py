from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase1.models import (  # noqa: E402
    Phase1Candidate,
    Phase1Label,
    Phase1StateLabel,
)


def test_phase1_candidate_preserves_fields():
    cand = Phase1Candidate(
        symbol="LH0",
        name="生猪",
        reversal_score=82.0,
        trend_score=36.0,
        attention_score=74.0,
        data_coverage=0.75,
        labels=[Phase1Label.REVERSAL],
        state_labels=[Phase1StateLabel.LOW_LEVEL_EXIT],
        reason_summary="深亏接近成本线，存在低位出清迹象",
    )
    assert cand.symbol == "LH0"
    assert cand.labels == [Phase1Label.REVERSAL]
    assert cand.labels[0].value == "反转候选"
    assert cand.state_labels == [Phase1StateLabel.LOW_LEVEL_EXIT]
    assert cand.state_labels[0].value == "低位出清"
