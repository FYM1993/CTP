from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.cache_boundary import (  # noqa: E402
    BacktestCacheImportViolation,
    find_backtest_cache_import_violations,
    find_project_backtest_cache_import_violations,
)


def test_backtest_cache_boundary_flags_live_imports(tmp_path: Path) -> None:
    live_file = tmp_path / "scripts" / "phase2" / "pre_market.py"
    live_file.parent.mkdir(parents=True)
    live_file.write_text(
        "from backtest.experiment_cache import load_phase2_candidates\n",
        encoding="utf-8",
    )

    violations = find_backtest_cache_import_violations([live_file], root=tmp_path)

    assert violations == [
        BacktestCacheImportViolation(
            path=Path("scripts/phase2/pre_market.py"),
            line=1,
            module="backtest.experiment_cache",
        )
    ]


def test_project_runtime_paths_do_not_import_backtest_experiment_cache() -> None:
    assert find_project_backtest_cache_import_violations(ROOT) == []
