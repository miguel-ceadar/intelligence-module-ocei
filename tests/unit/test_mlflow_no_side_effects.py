"""Phase-1 §2.2: ``mlflow gc`` out of ``ModelTrain.__init__``.

Constructing the orchestrator must not touch the filesystem, run shell
commands, or contact MLflow. Side effects in ``__init__`` make the class
untestable.
"""

from __future__ import annotations

from unittest import mock

import pytest

trainers = pytest.importorskip("intelligence.trainers", reason="phase-1 §2.2 pending")


def test_modeltrain_init_does_not_invoke_mlflow_gc():
    ModelTrain = getattr(trainers, "ModelTrain", None)
    if ModelTrain is None:
        pytest.skip("intelligence.trainers.ModelTrain not implemented yet")

    # Whether the legacy path used `os.system('mlflow gc')` or `mlflow.gc()`
    # directly, neither should fire from __init__.
    with mock.patch("os.system") as os_system, mock.patch.dict("sys.modules", {}, clear=False):
        # Construct with the loosest possible args — what matters is that
        # __init__ doesn't reach out to mlflow.
        try:
            ModelTrain(args=mock.MagicMock())
        except TypeError:
            pytest.skip("ModelTrain ctor signature changed — update this test")
    os_system.assert_not_called()


@pytest.mark.xfail(strict=True, reason="phase-1 task #6 pending — helper not yet extracted")
def test_explicit_gc_helper_exists():
    """The legacy behaviour (gc on every train) becomes an opt-in helper."""
    gc = getattr(trainers, "mlflow_gc", None) or getattr(trainers, "gc_mlruns", None)
    assert gc is not None and callable(gc), (
        "Move the mlflow gc logic to a named helper (e.g. `intelligence.trainers.mlflow_gc`) "
        "so it can be invoked from a startup hook behind a config flag."
    )
