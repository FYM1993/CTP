from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


BACKTEST_CACHE_MODULE_PREFIXES = (
    "backtest.experiment_cache",
)

LIVE_RUNTIME_PATHS = (
    Path("scripts/daily_workflow.py"),
    Path("scripts/cli"),
    Path("scripts/phase1"),
    Path("scripts/phase2"),
    Path("scripts/phase3"),
    Path("scripts/strategy_reversal/pre_market.py"),
    Path("scripts/strategy_reversal/intraday.py"),
    Path("scripts/strategy_reversal/screen.py"),
    Path("scripts/strategy_trend/pre_market.py"),
    Path("scripts/strategy_trend/intraday.py"),
    Path("scripts/strategy_trend/screen.py"),
    Path("scripts/holdings_advice"),
    Path("scripts/data_cache.py"),
)


@dataclass(frozen=True, slots=True)
class BacktestCacheImportViolation:
    path: Path
    line: int
    module: str


def _is_forbidden_module(module: str, forbidden_prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden_prefixes)


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((int(node.lineno), alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append((int(node.lineno), node.module))
    return modules


def find_backtest_cache_import_violations(
    paths: list[Path],
    *,
    root: Path,
    forbidden_prefixes: tuple[str, ...] = BACKTEST_CACHE_MODULE_PREFIXES,
) -> list[BacktestCacheImportViolation]:
    violations: list[BacktestCacheImportViolation] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            line = int(exc.lineno or 0)
            module = "<syntax-error>"
            violations.append(BacktestCacheImportViolation(path=path.relative_to(root), line=line, module=module))
            continue
        for line, module in _imported_modules(tree):
            if _is_forbidden_module(module, forbidden_prefixes):
                violations.append(
                    BacktestCacheImportViolation(
                        path=path.relative_to(root),
                        line=line,
                        module=module,
                    )
                )
    return violations


def _python_files_under(root: Path, relative_path: Path) -> list[Path]:
    path = root / relative_path
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    return sorted(child for child in path.rglob("*.py") if child.is_file())


def project_live_runtime_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for relative_path in LIVE_RUNTIME_PATHS:
        for path in _python_files_under(root, relative_path):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files


def find_project_backtest_cache_import_violations(root: Path) -> list[BacktestCacheImportViolation]:
    return find_backtest_cache_import_violations(project_live_runtime_python_files(root), root=root)
