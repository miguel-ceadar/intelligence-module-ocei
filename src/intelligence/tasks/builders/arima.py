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

    # ARIMA is univariate by construction. Multivariate ARIMA is VAR
    # (a different statsmodels API); ship that as its own `kind: var`
    # rather than silently picking one feature and dropping the rest.
    if len(task_cfg.features) > 1:
        names = [f.name for f in task_cfg.features]
        raise ValueError(
            f"ARIMA task {name!r} declares {len(task_cfg.features)} features ({names}); "
            f"ARIMA is univariate. Drop the extra features, or switch to a multivariate "
            f"kind once `kind: var` ships."
        )

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
            value_cols=[f.name for f in task_cfg.features],
            queries=[f.query for f in task_cfg.features],
        ),
        input_spec=build_input_spec(
            features=task_cfg.features,
            steps_back=task_cfg.steps_back,
        ),
        pinned_version=task_cfg.pinned_version,
    )
