"""Drift-detection task builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.builders._common import build_input_spec
from intelligence.tasks.drift import DriftDetectionTask, make_drift_prepare
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import DriftTaskConfig, IntelligenceConfig


def build_drift_task(
    name: str,
    task_cfg: "DriftTaskConfig",
    intelligence_cfg: "IntelligenceConfig",
) -> DriftDetectionTask:
    # nannyml is the heavy dep here, but ``intelligence.tasks.drift``
    # already lazy-imports it inside ``DriftDetectionTask.train``. The
    # module itself is fine to load at the top.
    return DriftDetectionTask(
        name=name,
        forecaster_task_name=task_cfg.forecaster,
        model=None,
        data_loader=build_loader_for_task(
            intelligence_cfg, name,
            value_col=task_cfg.feature,
            prepare=make_drift_prepare(value_col=None),
            query=task_cfg.query,
        ),
        chunk_size=task_cfg.chunk_size,
        metric=task_cfg.metric,
        input_spec=build_input_spec(
            feature=task_cfg.feature,
            steps_back=task_cfg.chunk_size,
            value_range=task_cfg.value_range,
        ),
        pinned_version=task_cfg.pinned_version,
    )
