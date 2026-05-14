"""Drift-detection task builder."""

from __future__ import annotations

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
    # ``chunk_size`` doubles as the InputSpec ``steps_back`` so the
    # contract layer enforces exact-length analysis windows. This dual
    # use is intentional (one NannyML chunk per request) — see the
    # maintainability pass §2 notes for the alternative.
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
        ),
        input_spec=build_input_spec(
            features=task_cfg.features,
            steps_back=task_cfg.chunk_size,
        ),
        pinned_version=task_cfg.pinned_version,
    )
