"""XGBoost task builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare
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
    # ``model_dump()`` preserves both the named defaults and any
    # extra=allow fields (forward-compat with newer xgboost knobs).
    feature_names = [f.name for f in task_cfg.features]
    # XGB is recursive (one-step ahead during training), so
    # ``supervised_window`` uses ``n_out=1``. The binding constraint is
    # ``len(test) >= look_back + 1`` after the 80/20 split — i.e.
    # ``5 * (look_back + 1)`` rows total. Below this, training crashed
    # deep in ``supervised_window`` with a less actionable message.
    min_points = max(30, 5 * (task_cfg.steps_back + 1))
    return BaseTask(
        name=name,
        model=XgbModel(**task_cfg.model_params.model_dump()),
        data_loader=build_loader_for_task(
            intelligence_cfg,
            name,
            value_cols=feature_names,
            prepare=make_xgb_prepare(
                look_back=task_cfg.steps_back,
                feature_names=feature_names,
            ),
            queries=[f.query for f in task_cfg.features],
            min_points=min_points,
        ),
        input_spec=build_input_spec(
            features=task_cfg.features,
            steps_back=task_cfg.steps_back,
        ),
        pinned_version=task_cfg.pinned_version,
    )
