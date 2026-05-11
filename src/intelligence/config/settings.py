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

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    auto_gc: bool = False


class DataClayConfig(BaseModel):
    """Phase-2 deletion target. Kept so the legacy ``oasis/`` path stays
    configurable during the transition."""

    enabled: bool = False


class ModelRepoConfig(BaseModel):
    """Hugging Face model push/pull settings. Endpoint not wired in
    phase 1 — placeholder so phase-2 add is additive."""

    hf_enabled: bool = False


class TelemetryConfig(BaseModel):
    """Phase-2 placeholder. Only ``source: static`` is meaningful today;
    phase 2 adds ``prometheus`` / ``thanos`` / ``otel`` plus their
    auth/endpoint/query fields."""

    source: str = "static"


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
    dataclay: DataClayConfig = DataClayConfig()
    model_repo: ModelRepoConfig = ModelRepoConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

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
