"""Shared fixtures.

Tests import from ``intelligence.*`` (the new home). Where the implementation
hasn't landed yet, tests use ``pytest.importorskip`` so the suite reports
SKIPPED rather than blowing up at collection. As phase-1 chunks land, those
skips flip to passes. ``--strict-markers`` keeps marker typos honest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


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
def src_root(repo_root: Path) -> Path:
    return repo_root / "src" / "intelligence"


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
