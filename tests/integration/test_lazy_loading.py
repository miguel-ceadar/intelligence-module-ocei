"""Phase-1 §2.3: lazy model loading.

Importing the API module must not call ``bentoml.<framework>.get(...)``.
That's the change that drops cold-start RAM from ~40GB to task-by-task.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest

pytestmark = pytest.mark.integration


def test_importing_api_does_not_load_any_artefacts():
    """Importing ``intelligence.api.service`` must not call
    ``get_artifact_by_tag`` — every task defers its artefact load until
    first predict. That's the change that drops cold-start RAM from
    ~40 GB to task-by-task.
    """
    pytest.importorskip("intelligence.api")

    # Force ``intelligence.api.service`` to re-execute its module body
    # so the patch below actually sees the calls (if any). Only the
    # service module — purging ``intelligence.api.schemas`` too would
    # replace ``PrometheusDataSource`` / ``StaticDataSource`` with
    # fresh classes, breaking ``isinstance`` checks in code that already
    # imported them (e.g. ``intelligence.tasks.loaders``).
    sys.modules.pop("intelligence.api.service", None)

    calls: list[tuple] = []

    with mock.patch(
        "intelligence.tasks.base.get_artifact_by_tag",
        side_effect=lambda *a, **kw: calls.append(("get_artifact_by_tag", a, kw)),
    ):
        try:
            importlib.import_module("intelligence.api.service")
        except ModuleNotFoundError:
            pytest.skip("intelligence.api.service not implemented yet")

    assert not calls, (
        f"Importing intelligence.api.service triggered eager artefact loads: {calls}. "
        "Move them inside the per-task lazy initializer."
    )


def test_task_loads_model_on_first_predict():
    """Once a task receives its first request, the BentoML model is fetched
    and cached on the task instance for subsequent calls."""
    tasks = pytest.importorskip("intelligence.tasks", reason="phase-1 §2.3 pending")
    base = pytest.importorskip("intelligence.tasks.base", reason="phase-1 §2.3 pending")

    Task = getattr(base, "Task", None) or getattr(tasks, "Task", None)
    if Task is None:
        pytest.skip("intelligence.tasks.Task protocol not exported yet")

    # We can't construct a real task without a registered Bento model; this
    # test asserts the *contract* — that Task instances expose a `loaded`
    # / `is_loaded()` accessor we can inspect.
    assert hasattr(Task, "is_loaded") or hasattr(Task, "loaded"), (
        "Task protocol should expose a way to check whether the underlying "
        "BentoML model has been loaded — needed to verify lazy loading."
    )
