"""Shared fixtures.

Tests import from ``intelligence.*`` (the new home). Where the implementation
hasn't landed yet, tests use ``pytest.importorskip`` so the suite reports
SKIPPED rather than blowing up at collection. As phase-1 chunks land, those
skips flip to passes. ``--strict-markers`` keeps marker typos honest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``intelligence.*`` importable without relying on the editable-install
# ``.pth`` file. On macOS the ``.venv/`` directory is marked hidden, the
# UF_HIDDEN flag propagates to the ``.pth`` files inside, and CPython 3.12+
# silently skips hidden ``.pth`` files (security check from CPython #113659).
# The net effect is that editable installs in dot-prefixed venvs randomly
# break with ``ModuleNotFoundError: No module named 'intelligence'``. This
# path injection makes the test suite robust to that quirk regardless of
# venv layout, OS, or build backend. Production (Docker / Helm) is
# unaffected — it ships non-editable wheels with no ``.pth`` in the loop.
# Upstream tracking: astral-sh/uv#16977, python/cpython#148121.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session", autouse=True)
def _isolate_bentoml_store(tmp_path_factory: pytest.TempPathFactory):
    """Point BentoML at a per-session temp dir so train tests don't pollute
    ``~/bentoml``. Set BEFORE any test imports bentoml."""
    bentoml_home = tmp_path_factory.mktemp("bentoml_home")
    os.environ["BENTOML_HOME"] = str(bentoml_home)
    yield


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def samples_dir(repo_root: Path) -> Path:
    return repo_root / "src" / "intelligence" / "data" / "samples"


@pytest.fixture(scope="session")
def sample_csv_univariate(samples_dir: Path) -> Path:
    """Single-feature CPU CSV — used for ARIMA and XGB happy-path training."""
    return samples_dir / "cpu_sample_dataset_orangepi.csv"


@pytest.fixture(scope="session")
def sample_csv_multivariate(samples_dir: Path) -> Path:
    """CPU+MEM CSV — used for the LSTM happy-path training."""
    return samples_dir / "node_3_utilisation_sample_dataset.csv"
