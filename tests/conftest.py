"""Shared fixtures.

Tests import from ``intelligence.*`` (the new home). Where the implementation
hasn't landed yet, tests use ``pytest.importorskip`` so the suite reports
SKIPPED rather than blowing up at collection. As phase-1 chunks land, those
skips flip to passes. ``--strict-markers`` keeps marker typos honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def src_root(repo_root: Path) -> Path:
    return repo_root / "src" / "intelligence"


@pytest.fixture(scope="session")
def oasis_root(repo_root: Path) -> Path:
    return repo_root / "oasis"


@pytest.fixture(scope="session")
def sample_csv_univariate(repo_root: Path) -> Path:
    """Single-feature CPU CSV — used for ARIMA and XGB happy-path training."""
    return repo_root / "oasis" / "dataset" / "cpu_sample_dataset_orangepi.csv"


@pytest.fixture(scope="session")
def sample_csv_multivariate(repo_root: Path) -> Path:
    """CPU+MEM CSV — used for the LSTM happy-path training."""
    return repo_root / "oasis" / "dataset" / "node_3_utilisation_sample_dataset.csv"
