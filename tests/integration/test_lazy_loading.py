"""Lazy model loading.

Importing the API module must not call ``bentoml.<framework>.get(...)``
or otherwise materialise saved Bentos — every task defers its artefact
load until first predict. That's the change that drops cold-start RAM
from ~40 GB to task-by-task.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest

from intelligence.tasks.base import Task

pytestmark = pytest.mark.integration


def test_importing_api_does_not_load_any_artefacts():
    """``intelligence.api.service`` import must not call
    ``get_artifact_by_tag``.
    """
    # Force re-execution so the patch below sees any module-body calls.
    # Only the service module — purging ``intelligence.api.schemas`` too
    # would replace ``PrometheusDataSource`` / ``StaticDataSource`` with
    # fresh classes, breaking ``isinstance`` checks in already-imported
    # code (e.g. ``intelligence.tasks.loaders``).
    sys.modules.pop("intelligence.api.service", None)

    calls: list[tuple] = []
    with mock.patch(
        "intelligence.tasks.base.get_artifact_by_tag",
        side_effect=lambda *a, **kw: calls.append(("get_artifact_by_tag", a, kw)),
    ):
        importlib.import_module("intelligence.api.service")

    assert not calls, (
        f"Importing intelligence.api.service triggered eager artefact loads: {calls}. "
        "Move them inside the per-task lazy initializer."
    )


def test_task_protocol_exposes_loaded_introspection():
    """The ``Task`` protocol exports ``is_loaded`` so the API layer can
    report a task's load state without poking at private cache fields."""
    assert hasattr(Task, "is_loaded")
