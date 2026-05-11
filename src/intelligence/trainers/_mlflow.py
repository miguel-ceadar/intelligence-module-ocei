"""Helpers for managing MLflow tracking artefacts."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def mlflow_gc(*, log_errors: bool = True) -> None:
    """Invoke ``mlflow gc`` to permanently remove soft-deleted runs.

    Phase 1 extracts this from ``ModelTrain.__init__`` so the orchestrator
    can be constructed without subprocess side effects. Callers wire it
    in at the right moment — a startup hook, before a training round, etc.

    Idempotent: if the ``mlflow`` CLI isn't on PATH, logs and returns.
    """
    try:
        result = subprocess.run(
            ["mlflow", "gc"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and log_errors:
            logger.warning(
                "mlflow gc exited %s: %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
    except FileNotFoundError:
        if log_errors:
            logger.warning("mlflow CLI not found on PATH; skipping gc")
