from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class RoboflowSettings(BaseModel):
    project: str
    version: int
    confidence: float = Field(0.01, ge=0.0, le=1.0)
    overlap: float = Field(0.3, ge=0.0, le=1.0)
    per_class_thresholds: dict[str, float] = Field(default_factory=dict)
    api_url: str = "https://serverless.roboflow.com"
    api_key_env: str = "ROBOFLOW_API_KEY"

    @validator("per_class_thresholds", pre=True)
    def _default_per_class(cls, value: Any) -> dict[str, float]:  # noqa: D401
        if value is None:
            return {}
        return value

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


class PreprocessSettings(BaseModel):
    dpi: int = Field(400, ge=72, le=1200)


class GeometryPreprocessSettings(BaseModel):
    enabled: bool = False
    command: list[str] | str | None = None
    args: list[str] | str | None = None
    env: dict[str, str] | None = None
    emit_stats: bool = True
    emit_wexbim: bool = False
    capture_logs: bool = True
    timeout_seconds: float | None = 600.0


class GeometrySettings(BaseModel):
    preprocess: GeometryPreprocessSettings = Field(default_factory=GeometryPreprocessSettings)


class IfcSettings(BaseModel):
    schema: str = "IFC4"
    unit: str = "mm"
    storey_zero_name: str = "EG"
    wall_thickness_standards_mm: list[float] = Field(default_factory=lambda: [115.0, 240.0, 300.0, 400.0, 500.0])

    @validator("wall_thickness_standards_mm", pre=True)
    def _normalize_standards(cls, value: Any) -> list[float]:  # noqa: D401
        if value is None:
            return [115.0, 240.0, 300.0, 400.0, 500.0]
        if isinstance(value, (int, float)):
            value = [float(value)]
        if not isinstance(value, (list, tuple)):
            raise ValueError("wall_thickness_standards_mm must be a list of numbers")
        filtered: list[float] = []
        for item in value:
            try:
                number = abs(float(item))
            except (TypeError, ValueError) as exc:
                raise ValueError("wall_thickness_standards_mm entries must be numeric") from exc
            if number > 0.0:
                filtered.append(number)
        return filtered or [115.0, 240.0, 300.0, 400.0, 500.0]


class StorageSettings(BaseModel):
    bucket: str
    region: str | None = None


class DatabaseSettings(BaseModel):
    url_env: str = "DATABASE_URL"

    @property
    def url(self) -> str:
        value = os.getenv(self.url_env)
        if not value:
            raise RuntimeError(f"Environment variable '{self.url_env}' is required for database connectivity")
        return value


class QueueSettings(BaseModel):
    broker_url_env: str = "CELERY_BROKER_URL"
    result_backend_env: str | None = "CELERY_RESULT_BACKEND"

    @property
    def broker_url(self) -> str:
        value = os.getenv(self.broker_url_env)
        if not value:
            raise RuntimeError("Celery broker URL is not configured")
        return value

    @property
    def result_backend(self) -> str | None:
        if not self.result_backend_env:
            return None
        return os.getenv(self.result_backend_env)


class Settings(BaseModel):
    roboflow: RoboflowSettings
    preprocess: PreprocessSettings
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    ifc: IfcSettings
    storage: StorageSettings
    database: DatabaseSettings
    queue: QueueSettings

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        config_path = path or Path(os.getenv("BIMIFY_CONFIG", "config/default.yaml"))
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
        return cls(**payload)


@lru_cache(maxsize=1)
def get_settings(path: str | None = None) -> Settings:
    return Settings.load(Path(path) if path else None)


__all__ = [
    "Settings",
    "RoboflowSettings",
    "PreprocessSettings",
    "GeometrySettings",
    "GeometryPreprocessSettings",
    "IfcSettings",
    "StorageSettings",
    "DatabaseSettings",
    "QueueSettings",
    "get_settings",
]

