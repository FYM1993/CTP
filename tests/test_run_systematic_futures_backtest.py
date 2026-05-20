from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_systematic_futures_backtest_cli_help() -> None:
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_systematic_futures_backtest.py"), "--help"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "systematic futures" in result.stdout.lower()
