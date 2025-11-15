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


class RoboflowModelConfig(BaseModel):
    project: str
    version: int
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap: float | None = Field(default=None, ge=0.0, le=1.0)
    per_class_thresholds: dict[str, float] | None = None

    @validator("per_class_thresholds", pre=True)
    def _default_per_class(cls, value: Any) -> dict[str, float] | None:  # noqa: D401
        if value is None:
            return None
        return value


class RoboflowSettings(BaseModel):
    project: str
    version: int
    confidence: float = Field(0.01, ge=0.0, le=1.0)
    overlap: float = Field(0.3, ge=0.0, le=1.0)
    per_class_thresholds: dict[str, float] = Field(default_factory=dict)
    api_url: str = "https://serverless.roboflow.com"
    api_key_env: str = "ROBOFLOW_API_KEY"
    models: dict[str, RoboflowModelConfig] = Field(default_factory=dict)

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


class RepairLevel1Settings(BaseModel):
    # Tolerances (mm / deg)
    posTol_mm: float = Field(20.0, ge=0.0)
    angleTol_deg: float = Field(5.0, ge=0.0, le=45.0)
    simplifyTol_mm: float = Field(20.0, ge=0.0)
    areaThreshold_m2: float = Field(0.05, ge=0.0)
    branchMin_mm: float = Field(150.0, ge=0.0)
    minOverlap_mm: float = Field(50.0, ge=0.0)

    # Edge detector
    canny_low: int = Field(50, ge=0, le=255)
    canny_high: int = Field(150, ge=0, le=255)
    hough_threshold: int = Field(80, ge=1, le=500)
    hough_min_line_length_mm: float = Field(200.0, ge=0.0)
    hough_max_line_gap_mm: float = Field(20.0, ge=0.0)


class GeometrySettings(BaseModel):
    preprocess: GeometryPreprocessSettings = Field(default_factory=GeometryPreprocessSettings)
    repair_level1: RepairLevel1Settings | None = None

class IfcSettings(BaseModel):
    schema: str = "IFC4"
    unit: str = "mm"
    storey_zero_name: str = "EG"
    wall_thickness_standards_mm: list[float] = Field(default_factory=lambda: [115.0, 240.0, 300.0, 400.0, 500.0])
    # Owner/Application metadata for IFC owner history
    owner_org_name: str = "BIMMATRIX"
    owner_org_identification: str | None = None
    app_identifier: str = "BIMMATRIX"
    app_full_name: str = "BIMMATRIX IFC Exporter"
    app_version: str = "1.0"
    person_identification: str | None = None
    person_given_name: str = "BIMMATRIX"
    person_family_name: str = "User"

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
        """Load settings from YAML configuration file.
        
        Args:
            path: Optional path to configuration file. If not provided, uses
                BIMIFY_CONFIG environment variable or defaults to config/default.yaml.
        
        Returns:
            Settings instance with loaded configuration.
        
        Raises:
            FileNotFoundError: If configuration file does not exist.
            ValueError: If configuration is invalid.
        """
        config_path = path or Path(os.getenv("BIMIFY_CONFIG", "config/default.yaml"))
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
        try:
            return cls(**payload)
        except Exception as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings(path: str | None = None) -> Settings:
    return Settings.load(Path(path) if path else None)


__all__ = [
    "Settings",
    "RoboflowModelConfig",
    "RoboflowSettings",
    "PreprocessSettings",
    "GeometrySettings",
    "GeometryPreprocessSettings",
    "RepairLevel1Settings",
    "IfcSettings",
    "StorageSettings",
    "DatabaseSettings",
    "QueueSettings",
    "get_settings",
]

