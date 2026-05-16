"""Drift-detection task builder."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from intelligence.ml.models.drift import DriftModel, make_drift_prepare
from intelligence.tasks.base import BaseTask
from intelligence.tasks.builders._common import build_input_spec
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import DriftTaskConfig, IntelligenceConfig


def build_drift_task(
    name: str,
    task_cfg: DriftTaskConfig,
    intelligence_cfg: IntelligenceConfig,
) -> BaseTask:
    # Eager-check the heavy optional dep at registry build time. Without
    # this probe a missing ``nannyml`` only surfaces deep in predict as
    # an opaque 500. ``find_spec`` avoids paying the import cost here;
    # the actual import happens lazily in ``DriftModel.load_artifacts``.
    if importlib.util.find_spec("nannyml") is None:
        raise ImportError(
            f"drift task {name!r} requires the optional `nannyml` package; "
            f"install with `pip install intelligence-module-ocei[drift]` "
            f"(or `uv sync --extra drift`)"
        )

    # ``chunk_size`` doubles as the InputSpec ``steps_back`` so the
    # contract layer enforces exact-length analysis windows. This dual
    # use is intentional (one NannyML chunk per request) — see the
    # maintainability pass §2 notes for the alternative.
    #
    # The loader's default ``min_points=30`` can sit below ``chunk_size``
    # for operators who pick a larger chunk; in that case NannyML would
    # fail mid-fit with an opaque message. Floor it at ``chunk_size * 2``
    # so the reference frame always covers at least two chunks.
    min_points = max(30, task_cfg.chunk_size * 2)
    return BaseTask(
        name=name,
        model=DriftModel(
            chunk_size=task_cfg.chunk_size,
            metric=task_cfg.metric,
            forecaster_task_name=task_cfg.forecaster,
        ),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_cols=[f.name for f in task_cfg.features],
            prepare=make_drift_prepare(value_col=None),
            queries=[f.query for f in task_cfg.features],
            min_points=min_points,
        ),
        input_spec=build_input_spec(
            features=task_cfg.features,
            steps_back=task_cfg.chunk_size,
        ),
        pinned_version=task_cfg.pinned_version,
    )
