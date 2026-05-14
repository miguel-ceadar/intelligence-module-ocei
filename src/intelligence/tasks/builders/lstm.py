"""LSTM task builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.base import BaseTask
from intelligence.tasks.builders._common import build_input_spec
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig, LstmTaskConfig


def build_lstm_task(
    name: str,
    task_cfg: LstmTaskConfig,
    intelligence_cfg: IntelligenceConfig,
) -> BaseTask:
    # torch is the heaviest dep we ship — lazy-imported here so non-LSTM
    # deployments don't pay the cost on registry build.
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare

    # The network's I/O shape is dictated by the task config, not by
    # whatever default ``model_params`` carries: ``input_size`` is the
    # feature count (one channel per feature in the input tensor) and
    # ``output_size`` is the horizon (target-only multivariate output).
    # ``task_cfg.horizon`` also drives the request-time max_horizon clamp.
    model_params = {
        **task_cfg.model_params.model_dump(),
        "input_size": len(task_cfg.features),
        "output_size": task_cfg.horizon,
    }

    return BaseTask(
        name=name,
        model=LstmModel(**model_params),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_cols=[f.name for f in task_cfg.features],
            prepare=make_lstm_prepare(
                look_back=task_cfg.steps_back,
                num_variables=len(task_cfg.features),
                batch_size=task_cfg.batch_size,
                horizon=task_cfg.horizon,
            ),
            queries=[f.query for f in task_cfg.features],
        ),
        input_spec=build_input_spec(
            features=task_cfg.features,
            steps_back=task_cfg.steps_back,
            max_horizon=task_cfg.horizon,
        ),
        pinned_version=task_cfg.pinned_version,
    )
