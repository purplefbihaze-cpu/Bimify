"""Shared utilities for API routes."""

from __future__ import annotations

import os
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import urlopen
from collections.abc import Mapping
from typing import Any
from uuid import UUID
import tempfile

from fastapi import HTTPException
from core.settings import RoboflowModelConfig, get_settings
from services.api.jobs_store import FileJobStore
from core.exceptions import JobNotFoundError, FileNotFoundError as BimifyFileNotFoundError


DATA_ROOT = Path("data")
JOB_ROOT = DATA_ROOT / "jobs"
EXPORT_ROOT = DATA_ROOT / "exports"
PREVIEW_ROOT = DATA_ROOT / "previews"
SETTINGS_PATH = DATA_ROOT / "settings.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _project_id_only(value: str) -> str:
    """Extract project ID from full project path."""
    parts = [segment for segment in (value or "").split("/") if segment]
    return parts[-1] if parts else value


def _load_server_settings() -> dict[str, Any]:
    """Load server settings from JSON file."""
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _resolve_model(model_kind: str | None = None) -> RoboflowModelConfig:
    """Resolve Roboflow model configuration for the given kind."""
    settings = get_settings()
    stored = _load_server_settings()

    kind = (model_kind or "geometry").strip().lower()
    models = settings.roboflow.models or {}
    geometry_model = models.get("geometry")
    requested_model = models.get(kind)

    def _first_non_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    # Allow overriding via persisted server settings (data/settings.json)
    specific_project_override = stored.get(f"roboflow_project_{kind}")
    if specific_project_override is not None:
        project_override = specific_project_override
    elif kind == "geometry":
        project_override = stored.get("roboflow_project")
    else:
        project_override = None

    specific_version_override = stored.get(f"roboflow_version_{kind}")
    if specific_version_override is not None:
        version_override = specific_version_override
    elif kind == "geometry":
        version_override = stored.get("roboflow_version")
    else:
        version_override = None

    project_raw = _first_non_none(
        project_override,
        getattr(requested_model, "project", None),
        getattr(geometry_model, "project", None),
        settings.roboflow.project,
    )
    version_raw = _first_non_none(
        version_override,
        getattr(requested_model, "version", None),
        getattr(geometry_model, "version", None),
        settings.roboflow.version,
    )

    confidence_raw = _first_non_none(
        getattr(requested_model, "confidence", None),
        getattr(geometry_model, "confidence", None),
        settings.roboflow.confidence,
    )
    overlap_raw = _first_non_none(
        getattr(requested_model, "overlap", None),
        getattr(geometry_model, "overlap", None),
        settings.roboflow.overlap,
    )

    per_class_raw = _first_non_none(
        getattr(requested_model, "per_class_thresholds", None),
        getattr(geometry_model, "per_class_thresholds", None),
        settings.roboflow.per_class_thresholds,
    ) or {}

    project = _project_id_only(str(project_raw))

    return RoboflowModelConfig(
        project=str(project),
        version=int(version_raw),
        confidence=float(confidence_raw) if confidence_raw is not None else None,
        overlap=float(overlap_raw) if overlap_raw is not None else None,
        per_class_thresholds=dict(per_class_raw) if isinstance(per_class_raw, dict) else None,
    )


def _normalize_relative_path(path_str: str) -> Path:
    """Normalize a relative path string to a Path object."""
    cleaned = (path_str or "").replace("\\", "/")
    pure = PurePosixPath(cleaned)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if parts and parts[0] == "/":
        parts = parts[1:]
    return Path(*parts)


def _resolve_ifc_from_url(ifc_url: str) -> tuple[Path, bool]:
    """Resolve IFC file path from URL.
    
    Returns:
        Tuple of (path, should_cleanup) where cleanup indicates if file is temporary.
    """
    parsed = urlparse(ifc_url)
    scheme = parsed.scheme.lower()
    if scheme and len(scheme) == 1 and not parsed.netloc:
        return Path(ifc_url), False
    if scheme in {"http", "https"}:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
        cleanup = True
        with urlopen(ifc_url) as resp:
            tmp.write(resp.read())
        tmp.flush()
        tmp.close()
        return Path(tmp.name), cleanup

    path_candidate: Path | None = None
    path_part = parsed.path or ifc_url

    if path_part.startswith("/files/"):
        path_candidate = EXPORT_ROOT / PurePosixPath(path_part).name
    else:
        normalized = _normalize_relative_path(path_part)
        candidate = Path(path_part)
        if candidate.is_absolute():
            path_candidate = candidate
        else:
            path_candidate = Path.cwd() / normalized

    return path_candidate, False


def _job_relative_path(job_id: UUID, relative: str) -> Path:
    """Get a path relative to a job directory, with security checks."""
    job_root = (JOB_ROOT / str(job_id)).resolve()
    normalized = _normalize_relative_path(relative)
    candidate = (job_root / normalized).resolve()
    if job_root not in candidate.parents and candidate != job_root:
        raise HTTPException(status_code=400, detail="Ungültiger Job-Dateipfad")
    return candidate


def _resolve_ifc_from_job(job_id: str) -> Path:
    """Resolve IFC file path from job ID."""
    try:
        job_uuid = UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Ungültige Job-ID: {job_id}") from exc

    store = FileJobStore(JOB_ROOT)
    job = store.load(job_uuid)
    if not job:
        raise JobNotFoundError(f"Job {job_id} nicht gefunden", {"job_id": job_id})

    job_meta = job.meta or {}
    ifc_meta = job_meta.get("ifc") or {}

    candidate_rel = ifc_meta.get("improved")
    if candidate_rel:
        candidate = _job_relative_path(job_uuid, candidate_rel)
        if candidate.exists():
            return candidate

    job_dir = JOB_ROOT / str(job_uuid) / "ifc"
    fallback = job_dir / "model_preproc.ifc"
    if fallback.exists():
        return fallback

    raise BimifyFileNotFoundError("Keine verbesserte (xBIM) IFC-Datei im Job gefunden", {"job_id": job_id})


def _resolve_ifc_source(ifc_url: str | None, job_id: str | None) -> tuple[Path, bool]:
    """Resolve IFC source from URL or job ID.
    
    Returns:
        Tuple of (path, should_cleanup) where cleanup indicates if file is temporary.
    """
    if ifc_url:
        path, cleanup = _resolve_ifc_from_url(ifc_url)
        if not path.exists():
            raise BimifyFileNotFoundError("IFC-Datei nicht gefunden", {"ifc_url": ifc_url})
        return path, cleanup
    if job_id:
        path = _resolve_ifc_from_job(job_id)
        return path, False
    raise HTTPException(status_code=400, detail="Entweder ifc_url oder job_id muss angegeben werden")


def _bool_from(value: Any, default: bool) -> bool:
    """Convert various types to boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return default
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return default


def _float_from(value: Any, default: float) -> float:
    """Convert value to float with default fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_from(value: Any, default: str) -> str:
    """Convert value to string with default fallback."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


MATRIX_DEFAULTS = {
    "matrix_enabled": True,
    "matrix_speed": 1.0,
    "matrix_density": 1.0,
    "matrix_opacity": 0.35,
    "matrix_color": "#00ff41",
}


def _resolve_matrix_settings(stored: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve matrix settings from stored settings or environment."""
    matrix_enabled_raw = stored.get("matrix_enabled")
    if matrix_enabled_raw is None:
        matrix_enabled_raw = os.getenv("MATRIX_ENABLED")
    matrix_speed_raw = stored.get("matrix_speed")
    if matrix_speed_raw is None:
        matrix_speed_raw = os.getenv("MATRIX_SPEED")
    matrix_density_raw = stored.get("matrix_density")
    if matrix_density_raw is None:
        matrix_density_raw = os.getenv("MATRIX_DENSITY")
    matrix_opacity_raw = stored.get("matrix_opacity")
    if matrix_opacity_raw is None:
        matrix_opacity_raw = os.getenv("MATRIX_OPACITY")
    matrix_color_raw = stored.get("matrix_color")
    if matrix_color_raw is None:
        matrix_color_raw = os.getenv("MATRIX_COLOR")
    return {
        "matrix_enabled": _bool_from(matrix_enabled_raw, MATRIX_DEFAULTS["matrix_enabled"]),
        "matrix_speed": _float_from(matrix_speed_raw, MATRIX_DEFAULTS["matrix_speed"]),
        "matrix_density": _float_from(matrix_density_raw, MATRIX_DEFAULTS["matrix_density"]),
        "matrix_opacity": _float_from(matrix_opacity_raw, MATRIX_DEFAULTS["matrix_opacity"]),
        "matrix_color": _str_from(matrix_color_raw, MATRIX_DEFAULTS["matrix_color"]),
    }


def _persist_env_var(name: str, value: str) -> None:
    """Persist environment variable to .env file."""
    os.environ[name] = value
    env_path = Path(".env")
    try:
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
            out: list[str] = []
            found = False
            for line in lines:
                if line.strip().startswith(f"{name}="):
                    out.append(f"{name}={value}")
                    found = True
                else:
                    out.append(line)
            if not found:
                out.append(f"{name}={value}")
            env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        else:
            env_path.write_text(f"{name}={value}\n", encoding="utf-8")
    except Exception:
        pass

