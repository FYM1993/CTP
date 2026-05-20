from __future__ import annotations

import os
from pathlib import Path

import yaml


def resolve_config_path(source_file: str | Path, *, env: dict[str, str] | None = None) -> Path:
    env_map = env or os.environ
    override = (env_map.get("CTP_CONFIG_PATH") or "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            return override_path
        raise FileNotFoundError(f"CTP_CONFIG_PATH 指向的配置文件不存在: {override_path}")

    source_path = Path(source_file).resolve()
    search_dir = source_path if source_path.is_dir() else source_path.parent
    worktree_root: Path | None = None

    for current in [search_dir, *search_dir.parents]:
        candidate = current / "config.yaml"
        if candidate.is_file():
            return candidate
        if current.name == ".worktrees":
            worktree_root = current.parent
            break

    if worktree_root is not None:
        fallback = worktree_root / "config.yaml"
        if fallback.is_file():
            return fallback

    raise FileNotFoundError(
        "未找到 config.yaml；如果你在 worktree 中运行，请在主目录放置 config.yaml，"
        "或设置环境变量 CTP_CONFIG_PATH 指向配置文件。"
    )


def load_yaml_config(
    source_file: str | Path,
    *,
    required: bool = True,
    env: dict[str, str] | None = None,
) -> dict:
    try:
        cfg_path = resolve_config_path(source_file, env=env)
    except FileNotFoundError:
        if required:
            raise
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
