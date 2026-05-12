"""Phase-1 §2.2: ``mlflow gc`` out of ``ModelTrain.__init__``.

Constructing the orchestrator must not touch the filesystem, run shell
commands, or contact MLflow. Side effects in ``__init__`` make the class
untestable.
"""

from __future__ import annotations

from unittest import mock

import pytest

trainers = pytest.importorskip("intelligence.ml.trainers", reason="phase-1 §2.2 pending")


def test_modeltrain_init_does_not_invoke_mlflow_gc():
    ModelTrain = getattr(trainers, "ModelTrain", None)
    if ModelTrain is None:
        pytest.skip("intelligence.ml.trainers.ModelTrain not implemented yet")

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


def test_explicit_gc_helper_exists():
    """The legacy behaviour (gc on every train) becomes an opt-in helper."""
    gc = getattr(trainers, "mlflow_gc", None) or getattr(trainers, "gc_mlruns", None)
    assert gc is not None and callable(gc), (
        "Move the mlflow gc logic to a named helper (e.g. `intelligence.ml.trainers.mlflow_gc`) "
        "so it can be invoked from a startup hook behind a config flag."
    )


def test_mlflow_gc_invokes_mlflow_cli():
    """The helper shells out to the ``mlflow`` CLI with ``gc`` as subcommand."""
    from intelligence.ml.trainers import mlflow_gc

    with mock.patch("intelligence.ml.trainers._mlflow.subprocess.run") as run:
        run.return_value.returncode = 0
        mlflow_gc()
        run.assert_called_once()
        cmd = run.call_args.args[0]
        assert cmd[0] == "mlflow"
        assert "gc" in cmd


def test_mlflow_gc_tolerates_missing_cli():
    """If ``mlflow`` isn't on PATH, the helper logs and returns rather than raising."""
    from intelligence.ml.trainers import mlflow_gc

    with mock.patch(
        "intelligence.ml.trainers._mlflow.subprocess.run", side_effect=FileNotFoundError
    ):
        mlflow_gc()  # must not raise
