"""XGBoost task builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.base import BaseTask
from intelligence.tasks.builders._common import build_input_spec
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig, XgbTaskConfig


def build_xgb_task(
    name: str,
    task_cfg: XgbTaskConfig,
    intelligence_cfg: IntelligenceConfig,
) -> BaseTask:
    # xgboost (+ libgomp at the OS level) lives inside XgbModel — lazy.
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    # ``model_dump()`` preserves both the named defaults and any
    # extra=allow fields (forward-compat with newer xgboost knobs).
    return BaseTask(
        name=name,
        model=XgbModel(**task_cfg.model_params.model_dump()),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_col=task_cfg.feature,
            prepare=make_xgb_prepare(look_back=task_cfg.steps_back, num_variables=1),
            query=task_cfg.query,
        ),
        input_spec=build_input_spec(
            feature=task_cfg.feature,
            steps_back=task_cfg.steps_back,
            value_range=task_cfg.value_range,
        ),
        pinned_version=task_cfg.pinned_version,
    )
