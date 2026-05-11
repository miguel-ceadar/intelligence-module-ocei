"""Phase-1 §2.1: src layout, no sys.path hacks, all subpackages importable.

These tests guard against regression once the layout lands. They run even
before the refactor — most pass on the empty package skeleton, the grep
test will fail until the legacy ``oasis/`` tree is migrated and removed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

INTELLIGENCE_SUBPACKAGES = [
    "intelligence.adapters",
    "intelligence.api",
    "intelligence.config",
    "intelligence.contracts",
    "intelligence.tasks",
    "intelligence.trainers",
    "intelligence.utils",
]


def test_intelligence_package_imports():
    """The new home should import cleanly without any sys.path manipulation."""
    mod = importlib.import_module("intelligence")
    assert mod.__name__ == "intelligence"


@pytest.mark.parametrize("modname", INTELLIGENCE_SUBPACKAGES)
def test_subpackages_importable(modname: str):
    importlib.import_module(modname)


def test_no_sys_path_append_in_intelligence(src_root: Path):
    """``sys.path.append`` is the smell that the legacy tree relied on. Don't bring it forward."""
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "sys.path.append" in text or "sys.path.insert" in text:
            offenders.append(str(path.relative_to(src_root)))
    assert not offenders, f"sys.path manipulation found in: {offenders}"


def test_no_oasis_imports_inside_intelligence(src_root: Path):
    """The ``oasis/`` tree is legacy. New code must not depend on it."""
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Match `from oasis...` or `import oasis...` at line start (ignore comments / strings).
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("from oasis", "import oasis")):
                offenders.append(f"{path.relative_to(src_root)}: {stripped}")
                break
    assert not offenders, f"intelligence.* imports legacy oasis: {offenders}"
