"""Compatibility shim — the trainer lives in ``intelligence.trainers``.

The legacy class was ``ModelMetricsDataClay(DataClayObject)`` with
``@activemethod``-decorated training methods. After the phase-1 extraction
(see ``intelligence-utility-plan.v2.md`` §2.2) the methods sit on a plain
``ModelTrainer`` class. We re-export under the old name so the legacy
``oasis/api_train.py`` and the three ``oasis/models/*_compiler.py``
callers keep working unchanged until they're migrated.

This file goes away with ``oasis/`` at the end of phase 1.
"""

from intelligence.trainers import ModelTrainer as ModelMetricsDataClay  # noqa: F401

__all__ = ["ModelMetricsDataClay"]
