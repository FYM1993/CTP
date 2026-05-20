from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scripts_using_runtime_union_annotations_postpone_annotation_evaluation():
    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ("| None" in source or "None |" in source) and "from __future__ import annotations" not in source:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
