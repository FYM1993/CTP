from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase2_direction import choose_phase2_direction  # noqa: E402


def test_choose_phase2_direction_prefers_long_when_long_side_clearly_stronger() -> None:
    result = choose_phase2_direction(long_score=32.0, short_score=10.0, delta=12.0)
    assert result == "long"


def test_choose_phase2_direction_returns_watch_when_gap_too_small() -> None:
    result = choose_phase2_direction(long_score=21.0, short_score=17.0, delta=12.0)
    assert result == "watch"


def test_choose_phase2_direction_prefers_short_when_short_side_clearly_stronger() -> None:
    result = choose_phase2_direction(long_score=11.0, short_score=25.0, delta=12.0)
    assert result == "short"
