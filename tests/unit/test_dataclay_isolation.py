"""Phase-1 §2.2: ``dataclay`` imports confined to one module.

Once this test passes, phase 2 can drop DataClay by deleting one file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ALLOWED_DATACLAY_IMPORTERS = {"intelligence/adapters/dataclay_client.py"}
DATACLAY_IMPORT_RE = re.compile(r"^\s*(from\s+dataclay\b|import\s+dataclay\b)", re.MULTILINE)


def test_dataclay_imports_confined_to_adapter(src_root: Path):
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        rel = path.relative_to(src_root.parent).as_posix()
        if rel in {"intelligence/" + p for p in ALLOWED_DATACLAY_IMPORTERS} or rel in ALLOWED_DATACLAY_IMPORTERS:
            continue
        text = path.read_text(encoding="utf-8")
        if DATACLAY_IMPORT_RE.search(text):
            offenders.append(rel)
    assert not offenders, (
        "dataclay imports outside the centralised adapter: "
        f"{offenders}. Move them into intelligence/adapters/dataclay_client.py."
    )


def test_importing_dataclay_client_does_not_load_dataclay():
    """The helper must import ``dataclay`` lazily so a Mac dev loop without
    the optional ``[legacy]`` extra installed can still import the module
    surface for typing/registration."""
    pytest.importorskip("intelligence.adapters.dataclay_client")
    # Tolerate dataclay already in sys.modules from a previous test that
    # actively used it; the contract is that *importing the helper alone*
    # does not pull dataclay.
    sys.modules.pop("intelligence.adapters.dataclay_client", None)
    sys.modules.pop("dataclay", None)
    __import__("intelligence.adapters.dataclay_client")
    assert "dataclay" not in sys.modules, (
        "intelligence.adapters.dataclay_client triggered a top-level "
        "`import dataclay`. Move it inside the function body."
    )
