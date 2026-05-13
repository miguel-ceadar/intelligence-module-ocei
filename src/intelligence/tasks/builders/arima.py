"""ARIMA task builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.base import BaseTask
from intelligence.tasks.builders._common import build_input_spec
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import ArimaTaskConfig, IntelligenceConfig


def build_arima_task(
    name: str,
    task_cfg: ArimaTaskConfig,
    intelligence_cfg: IntelligenceConfig,
) -> BaseTask:
    # statsmodels lives inside ArimaModel — lazy-imported here so kinds
    # not configured for this task don't pull it.
    from intelligence.ml.models.arima import ArimaModel

    return BaseTask(
        name=name,
        model=ArimaModel(
            p=task_cfg.model_params.p,
            d=task_cfg.model_params.d,
            q=task_cfg.model_params.q,
        ),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_col=task_cfg.feature,
            query=task_cfg.query,
        ),
        input_spec=build_input_spec(
            feature=task_cfg.feature,
            steps_back=task_cfg.steps_back,
            value_range=task_cfg.value_range,
        ),
        pinned_version=task_cfg.pinned_version,
    )
