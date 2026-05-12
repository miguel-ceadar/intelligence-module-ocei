"""Typed config layer (phase-1 minimum).

Loads from a YAML file (path via ``INTELLIGENCE_CONFIG`` env var or
explicit ``load_config(path)``). Env vars override file values via
``INTELLIGENCE_<SECTION>__<FIELD>`` (double underscore separates
nested sections).

Validates ``enabled_tasks`` against the registered builtin catalog at
load time so misconfigured deployments fail loudly at startup, not at
first request.

Phase-2 will add real fields under ``telemetry`` (Prometheus endpoint,
auth, query templates) and ``bootstrap`` (auto-train-on-startup per
task). The keys are reserved here so the phase-2 additions are
additive and don't break existing config files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    auto_gc: bool = False


class ModelRepoConfig(BaseModel):
    """Hugging Face model push/pull settings.

    The HF token is intentionally not in this schema — it's read from
    the ``HF_TOKEN`` environment variable at request time so secrets
    don't end up in YAML config or ConfigMaps.
    """

    hf_enabled: bool = False
    repo_id: str | None = None


class PrometheusConfig(BaseModel):
    """Connection + auth + per-task PromQL queries.

    The query for each task lives here (not in the request body) — the
    operator wires the data source once, callers just say
    ``kind: "prometheus"`` with a window/step.

    Auth: ``token_env`` reads a bearer token from an environment variable
    at call time; ``token_file`` reads it from a file path. Pick one,
    not both. TLS verify can be skipped for inside-cluster traffic.
    """

    endpoint: str
    token_env: str | None = None
    token_file: str | None = None
    tls_skip_verify: bool = False
    timeout: float = 30.0
    queries: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_auth_method(self) -> "PrometheusConfig":
        if self.token_env and self.token_file:
            raise ValueError(
                "prometheus auth: set token_env or token_file, not both"
            )
        return self


class TelemetryConfig(BaseModel):
    """Where the data comes from. ``static`` reads CSVs from the bundled
    samples directory; ``prometheus`` queries PromQL via
    ``PrometheusConfig``.
    """

    source: Literal["static", "prometheus"] = "static"
    prometheus: PrometheusConfig | None = None

    @model_validator(mode="after")
    def _prometheus_block_required_when_selected(self) -> "TelemetryConfig":
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
    dataset_name: str | None = None    # static mode
    window: str | None = None          # prometheus mode
    step: str | None = None            # prometheus mode


class TaskConfig(BaseModel):
    """Per-task config — currently just the bootstrap block. Reserved as
    the home for future per-task knobs (e.g. ``allow_unverified_models``).
    """

    bootstrap: BootstrapConfig = BootstrapConfig()


class IntelligenceConfig(BaseSettings):
    """The ``intelligence:`` section of the config file.

    Env-var overrides: ``INTELLIGENCE_MLFLOW__TRACKING_URI=...`` overrides
    ``intelligence.mlflow.tracking_uri``. Env wins over init args (so a
    file-loaded value can be overridden from the environment without
    editing the file).
    """

    model_config = SettingsConfigDict(
        env_prefix="INTELLIGENCE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    enabled_tasks: list[str] = ["cpu_forecast_arima"]
    mlflow: MlflowConfig = MlflowConfig()
    model_repo: ModelRepoConfig = ModelRepoConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    tasks: dict[str, TaskConfig] = Field(default_factory=dict)

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
        """Check that every name in ``enabled_tasks`` has a registered factory.

        Importing ``intelligence.tasks.catalog`` populates the catalog;
        we check membership against ``_BUILTIN_FACTORIES``.
        """
        # Local imports — keeps this module importable without side effects
        # for callers who only want to read the schema.
        import intelligence.tasks.catalog  # noqa: F401  populates _BUILTIN_FACTORIES
        from intelligence.tasks.base import _BUILTIN_FACTORIES

        unknown = [
            t for t in self.intelligence.enabled_tasks if t not in _BUILTIN_FACTORIES
        ]
        if unknown:
            raise ValueError(
                f"unknown task(s) in enabled_tasks: {unknown}. "
                f"Available builtin tasks: {sorted(_BUILTIN_FACTORIES)}"
            )


def load_config(path: Path | str | None = None, *, validate: bool = True) -> AppConfig:
    """Load ``AppConfig`` from a YAML file (with env-var overrides).

    Args:
        path: YAML file. ``None`` returns defaults + env overrides only.
        validate: if ``True`` (default), check ``enabled_tasks`` against
            the registered task catalog. Set ``False`` for unit tests
            that exercise the config schema in isolation.
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
