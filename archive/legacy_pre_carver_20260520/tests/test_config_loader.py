from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from shared.config_loader import load_yaml_config, resolve_config_path  # noqa: E402
from cli import daily_workflow  # noqa: E402
import data_cache  # noqa: E402


def test_resolve_config_path_falls_back_from_worktree_to_main_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "ctp"
    main_config = repo_root / "config.yaml"
    main_config.parent.mkdir(parents=True, exist_ok=True)
    main_config.write_text("alpha: 1\n", encoding="utf-8")

    worktree_file = repo_root / ".worktrees" / "branch-a" / "scripts" / "cli" / "daily_workflow.py"
    worktree_file.parent.mkdir(parents=True, exist_ok=True)
    worktree_file.write_text("# stub\n", encoding="utf-8")

    assert resolve_config_path(worktree_file) == main_config
    assert load_yaml_config(worktree_file)["alpha"] == 1


def test_resolve_config_path_prefers_local_worktree_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "ctp"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "config.yaml").write_text("alpha: 1\n", encoding="utf-8")

    worktree_root = repo_root / ".worktrees" / "branch-b"
    worktree_config = worktree_root / "config.yaml"
    worktree_config.parent.mkdir(parents=True, exist_ok=True)
    worktree_config.write_text("alpha: 2\n", encoding="utf-8")

    worktree_file = worktree_root / "scripts" / "cli" / "daily_workflow.py"
    worktree_file.parent.mkdir(parents=True, exist_ok=True)
    worktree_file.write_text("# stub\n", encoding="utf-8")

    assert resolve_config_path(worktree_file) == worktree_config
    assert load_yaml_config(worktree_file)["alpha"] == 2


def test_daily_workflow_load_config_uses_main_repo_config_when_worktree_has_none(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ctp"
    main_config = repo_root / "config.yaml"
    main_config.parent.mkdir(parents=True, exist_ok=True)
    main_config.write_text("intraday:\n  interval: 30\n", encoding="utf-8")

    fake_file = repo_root / ".worktrees" / "branch-c" / "scripts" / "cli" / "daily_workflow.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(daily_workflow, "__file__", str(fake_file))

    config = daily_workflow.load_config()

    assert config["intraday"]["interval"] == 30


def test_data_cache_load_config_keeps_empty_dict_when_no_config(monkeypatch, tmp_path: Path) -> None:
    fake_file = tmp_path / ".worktrees" / "branch-d" / "scripts" / "data_cache.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(data_cache, "__file__", str(fake_file))

    assert data_cache._load_config() == {}
