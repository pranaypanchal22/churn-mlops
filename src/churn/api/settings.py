from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Runtime configuration, read from CHURN_* environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    registry_dir: Path = Path("models/registry")
    # Serve this exact version instead of the promoted one (e.g. to pin during a rollback).
    model_version: str | None = None
    schema_path: Path = Path("configs/schema.yaml")
    monitoring_config_path: Path = Path("configs/monitoring.yaml")
    max_batch_size: int = Field(default=100, ge=1, le=1000)
    log_level: str = "INFO"
