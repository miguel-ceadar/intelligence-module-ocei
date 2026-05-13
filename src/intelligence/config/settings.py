"""Typed config layer.

Loads from a YAML file (path via ``INTELLIGENCE_CONFIG`` env var or
explicit ``load_config(path)``). Env vars override file values via
``INTELLIGENCE_<SECTION>__<FIELD>`` (double underscore separates
nested sections).

Tasks are declared as a dict keyed by name; each value is a
discriminated-union ``TaskInstanceConfig`` (chosen by ``kind``).
Cross-task references (e.g. a drift task's ``forecaster:`` pointing at
another task's name) are validated at load time so misconfigured
deployments fail loudly at startup, not at first request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRepoConfig(BaseModel):
    """Hugging Face model push/pull settings.

    The HF token is intentionally not in this schema — it's read from
    the ``HF_TOKEN`` environment variable at request time so secrets
    don't end up in YAML config or ConfigMaps.
    """

    hf_enabled: bool = False
    repo_id: str | None = None


class AuthConfig(BaseModel):
    """Optional bearer-token auth on the HTTP API.

    Set ``token_env`` to the name of an env var holding the expected
    token; every protected request must then carry
    ``Authorization: Bearer <token>`` matching that env var's value.
    Probes (``/healthz``, ``/readyz``, ``/metrics``) and the auto-
    generated docs (``/docs``, ``/redoc``, ``/openapi.json``) stay
    open regardless — k8s probes and API discovery don't need a
    credential.

    Default off — local dev and the smoke stack stay frictionless.
    """

    token_env: str | None = None


class PrometheusConfig(BaseModel):
    """Connection + auth for the deployment-wide Prometheus.

    Per-task PromQL lives on each task config block (``query:``); this
    config carries only what's shared across tasks — endpoint, auth,
    TLS, timeout.

    Auth: ``token_env`` reads a bearer token from an environment variable
    at call time; ``token_file`` reads it from a file path. Pick one,
    not both. TLS verify can be skipped for inside-cluster traffic.
    """

    endpoint: str
    token_env: str | None = None
    token_file: str | None = None
    tls_skip_verify: bool = False
    timeout: float = 30.0

    @model_validator(mode="after")
    def _one_auth_method(self) -> PrometheusConfig:
        if self.token_env and self.token_file:
            raise ValueError("prometheus auth: set token_env or token_file, not both")
        return self


class TelemetryConfig(BaseModel):
    """Where the data comes from. ``static`` reads CSVs from the bundled
    samples directory; ``prometheus`` queries PromQL via
    ``PrometheusConfig``.

    ``allow_endpoint_override`` lets a train request supply a per-call
    Prometheus URL via ``data_source.endpoint``. Off by default: an
    authenticated POST /train can otherwise redirect outbound traffic
    anywhere (SSRF probe surface). Flip on per deployment when you
    genuinely need to flip between Prometheus instances at request
    time. The configured auth (``token_env`` / ``tls_skip_verify``)
    carries through to the override — per-request auth overrides are
    not in scope.
    """

    source: Literal["static", "prometheus"] = "static"
    prometheus: PrometheusConfig | None = None
    allow_endpoint_override: bool = False

    @model_validator(mode="after")
    def _prometheus_block_required_when_selected(self) -> TelemetryConfig:
        if self.source == "prometheus" and self.prometheus is None:
            raise ValueError(
                "telemetry.source='prometheus' requires the telemetry.prometheus block"
            )
        return self


class BootstrapConfig(BaseModel):
    """Per-task auto-train on startup.

    When enabled, the service spawns a background coroutine on startup
    that calls ``task.train(...)`` against the configured data source.
    ``/readyz`` reports the task as not-ready until bootstrap completes.

    The ``window`` / ``step`` fields apply to ``telemetry.source =
    prometheus``; ``dataset_name`` applies to ``static``. Default is
    off — operators opt in explicitly so a bad PromQL query doesn't
    silently block startup.
    """

    auto_train_on_startup: bool = False
    dataset_name: str | None = None  # static mode
    window: str | None = None  # prometheus mode
    step: str | None = None  # prometheus mode


# --- Per-kind task instance configs ----------------------------------------
#
# Each entry under ``intelligence.tasks`` is a discriminated-union config
# block keyed by ``kind``. The kind picks the algorithm (and a per-task
# instance config schema); the surrounding fields declare what to train on
# and how to validate request inputs.


class ArimaModelParams(BaseModel):
    """ARIMA order. Defaults match ``ArimaModel.__init__``."""

    p: int = 5
    d: int = 1
    q: int = 0


class XgbModelParams(BaseModel):
    """XGBoost regressor params. Extra fields tolerated for forward-compat
    with newer xgboost versions; the model passes them through.
    """

    model_config = SettingsConfigDict(extra="allow")
    n_estimators: int = 100
    max_depth: int = 3
    eta: float = 0.1


class LstmModelParams(BaseModel):
    """LSTM network shape. ``num_epochs`` deliberately small by default so
    the demo trains fast; real deployments override.
    """

    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 4
    num_epochs: int = 3


class _BaseTaskConfig(BaseModel):
    """Fields shared by every kind.

    ``feature`` is both the InputSpec ``feature_names[0]`` and the key
    clients use in ``input_series`` at predict time. The loader renames
    the source column to this name (see ``PrometheusLoader`` /
    ``StaticCsvLoader``), so feature naming is consistent regardless of
    what the upstream telemetry source labels its value column.

    ``query`` is the PromQL string used when ``telemetry.source ==
    "prometheus"``. In static mode the train request body supplies the
    CSV filename; ``query`` is unused and may be left ``None``.

    ``pinned_version`` locks predict requests to a specific Bento
    version (e.g. for rollback or staged rollouts). ``None`` means the
    task uses ``:latest`` unless a request overrides via
    ``PredictRequest.model_version``.
    """

    feature: str
    value_range: tuple[float, float] | None = None
    query: str | None = None
    pinned_version: str | None = None
    bootstrap: BootstrapConfig = BootstrapConfig()


class ArimaTaskConfig(_BaseTaskConfig):
    """``kind: arima`` — single-observation lookback, statsmodels ARIMA."""

    kind: Literal["arima"]
    steps_back: int = 1
    model_params: ArimaModelParams = ArimaModelParams()


class XgbTaskConfig(_BaseTaskConfig):
    """``kind: xgb`` — sliding-window XGBoost regressor."""

    kind: Literal["xgb"]
    steps_back: int = 6
    model_params: XgbModelParams = XgbModelParams()


class LstmTaskConfig(_BaseTaskConfig):
    """``kind: lstm`` — PyTorch LSTM. ``batch_size`` flows into the
    LSTM-specific prepare; ``steps_back`` is the input window length.

    LSTM is direct multi-output: ``horizon`` becomes both the trained
    network's ``output_size`` and the ``InputSpec.max_horizon`` clamp.
    Predict requests with horizon above this are refused at the API
    boundary — retrain with a larger ``horizon:`` to extend the forecast
    window.
    """

    kind: Literal["lstm"]
    steps_back: int = 6
    batch_size: int = 16
    horizon: int = 1
    model_params: LstmModelParams = LstmModelParams()


class DriftTaskConfig(_BaseTaskConfig):
    """``kind: drift`` — NannyML univariate drift detection.

    ``forecaster`` references another task's name; it carries the
    semantic link to the forecaster this drift detector pairs with.
    No per-algorithm model — the calculator itself is the artifact.
    """

    kind: Literal["drift"]
    forecaster: str
    chunk_size: int = 12
    metric: str = "jensen_shannon"


# Discriminated union — pydantic dispatches on ``kind`` and produces a
# clear validation error when an unknown kind appears.
TaskInstanceConfig = Annotated[
    ArimaTaskConfig | XgbTaskConfig | LstmTaskConfig | DriftTaskConfig,
    Field(discriminator="kind"),
]


class IntelligenceConfig(BaseSettings):
    """The ``intelligence:`` section of the config file.

    Tasks are declared as a dict keyed by task name; each value is a
    discriminated-union ``TaskInstanceConfig`` (picked by ``kind``).
    Every entry under ``tasks:`` is registered at startup — there's no
    separate enabled list. To disable a task, comment its block.

    Env-var overrides: ``INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT=...``
    overrides ``intelligence.telemetry.prometheus.endpoint``. Env wins
    over init args (so a file-loaded value can be overridden from the
    environment without editing the file).
    """

    model_config = SettingsConfigDict(
        env_prefix="INTELLIGENCE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    model_repo: ModelRepoConfig = ModelRepoConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    auth: AuthConfig = AuthConfig()
    tasks: dict[str, TaskInstanceConfig] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Env vars take precedence over init args (i.e. over YAML file values).
        return env_settings, init_settings, dotenv_settings, file_secret_settings


class AppConfig(BaseModel):
    intelligence: IntelligenceConfig = IntelligenceConfig()

    def validate_against_registry(self) -> None:
        """Cross-reference checks that pydantic can't express on its own.

        Pydantic's discriminated union already rejects unknown ``kind``
        values at parse time, so what's left here is the *reference*
        from a drift task's ``forecaster:`` field to another task that
        must exist in the same config.
        """
        from intelligence.config.settings import DriftTaskConfig

        task_names = set(self.intelligence.tasks)
        for name, task_cfg in self.intelligence.tasks.items():
            if isinstance(task_cfg, DriftTaskConfig) and task_cfg.forecaster not in task_names:
                raise ValueError(
                    f"drift task {name!r} references forecaster "
                    f"{task_cfg.forecaster!r} which isn't defined under "
                    f"`tasks:`. Define the forecaster task or fix the reference."
                )


def load_config(path: Path | str | None = None, *, validate: bool = True) -> AppConfig:
    """Load ``AppConfig`` from a YAML file (with env-var overrides).

    Args:
        path: YAML file. ``None`` returns defaults + env overrides only.
        validate: if ``True`` (default), run cross-reference checks
            (e.g. drift tasks point at defined forecasters). Set
            ``False`` for unit tests that exercise the schema in isolation.
    """
    if path is None:
        cfg = AppConfig()
    else:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        intelligence_data = data.get("intelligence", {})
        cfg = AppConfig(intelligence=IntelligenceConfig(**intelligence_data))

    if validate:
        cfg.validate_against_registry()
    return cfg
