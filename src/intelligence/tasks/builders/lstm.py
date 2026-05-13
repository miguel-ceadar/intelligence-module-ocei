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

    # ``task_cfg.horizon`` drives both the trained network's output_size
    # and the request-time max_horizon clamp.
    model_params = {**task_cfg.model_params.model_dump(), "output_size": task_cfg.horizon}

    return BaseTask(
        name=name,
        model=LstmModel(**model_params),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_col=task_cfg.feature,
            prepare=make_lstm_prepare(
                look_back=task_cfg.steps_back,
                num_variables=1,
                batch_size=task_cfg.batch_size,
                horizon=task_cfg.horizon,
            ),
            query=task_cfg.query,
        ),
        input_spec=build_input_spec(
            feature=task_cfg.feature,
            steps_back=task_cfg.steps_back,
            value_range=task_cfg.value_range,
            max_horizon=task_cfg.horizon,
        ),
        pinned_version=task_cfg.pinned_version,
    )
