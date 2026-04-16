"""
P1 基本面引擎单元测试（legacy + regime 纯逻辑，不依赖外网时可跑）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fundamental_regime import (  # noqa: E402
    classify_lh_regime,
    _lh_regime_hog_layers,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
