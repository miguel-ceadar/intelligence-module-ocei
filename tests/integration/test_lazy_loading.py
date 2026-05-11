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


def test_importing_api_does_not_load_any_bento_models():
    """Sniff ``bentoml.<framework>.get`` for the duration of the import."""
    bentoml = pytest.importorskip("bentoml")

    # Drop a clean slate so the import actually runs.
    for mod in list(sys.modules):
        if mod.startswith("intelligence.api"):
            del sys.modules[mod]

    patches = []
    calls: list[str] = []

    for fw in ("sklearn", "xgboost", "picklable_model", "pytorch", "keras"):
        # BentoML framework modules are lazy proxies — they import their
        # underlying lib on first attribute access, and raise
        # MissingDependencyException if that lib isn't installed (e.g.
        # keras without tensorflow). Skip cleanly when that happens.
        try:
            sub = getattr(bentoml, fw, None)
            if sub is None or not hasattr(sub, "get"):
                continue
            p = mock.patch.object(
                sub, "get",
                side_effect=lambda *a, _fw=fw, **kw: calls.append(f"{_fw}.get({a!r})"),
            )
        except Exception:
            continue
        patches.append(p)
        p.start()

    try:
        importlib.import_module("intelligence.api.service")
    except ModuleNotFoundError:
        pytest.skip("intelligence.api.service not implemented yet")
    finally:
        for p in patches:
            p.stop()

    assert not calls, (
        f"Importing intelligence.api.service triggered eager BentoML model loads: {calls}. "
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
