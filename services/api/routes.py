from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlparse
from urllib.request import urlopen
from collections.abc import Mapping
from typing import Annotated, Any
from uuid import UUID

import cv2
import numpy as np
import supervision as sv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from shapely.geometry import Point, Polygon

from core.ifc.hottcad_simulate import simulate_hottcad
from core.ifc.hottcad_validator import validate_hottcad
from core.ml.postprocess_floorplan import (
    NormalizedDet,
    estimate_wall_axes_and_thickness,
    normalize_predictions,
)
from core.ml.roboflow_client import RFPred, RFOptions, infer_floorplan_with_raw
from core.reconstruct.openings import snap_openings_to_walls
from core.reconstruct.spaces import polygonize_spaces_from_walls
from core.settings import get_settings
from core.validate.reconstruction_validation import generate_validation_report, write_validation_report
from core.vector.ifc_topview import build_topview_geojson
from core.vector.pdf_export import PDFExportOptions, generate_vector_pdf
from pydantic import ValidationError

from services.api.ifc_exporter import run_ifc_export, prediction_to_rfpred
from services.api.ifc_repair import RepairError, run_ifc_repair
from services.api.ifc_job_runner import process_ifc_job_async
from services.api.jobs_store import FileJobStore
from services.api.schemas import (
    AnalyzeOptions,
    AnalyzeResponse,
    ExportIFCJobResponse,
    ExportIFCRequest,
    ExportIFCResponse,
    ExportPDFRequest,
    ExportPDFResponse,
    JobStatusResponse,
    IFCTopViewRequest,
    IFCTopViewResponse,
    HottCADCheckOut,
    HottCADCompletenessOut,
    HottCADConnectionOut,
    HottCADFileInfo,
    HottCADMaterialSuggestionOut,
    HottCADMetricsOut,
    HottCADSimulateRequest,
    HottCADSimulationProposedOut,
    HottCADSimulationResponse,
    HottCADValidateRequest,
    HottCADValidationResponse,
    HottCADHighlightOut,
    HottCADSpaceBoundaryOut,
    LineCount,
    PredictionOut,
    SettingsPayload,
    ZoneCount,
    IFCRepairRequest,
    IFCRepairResponse,
)


router = APIRouter(prefix="/v1")


DATA_ROOT = Path("data")
JOB_ROOT = DATA_ROOT / "jobs"
EXPORT_ROOT = DATA_ROOT / "exports"
SETTINGS_PATH = DATA_ROOT / "settings.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _dispatch_ifc_job(job_id: UUID) -> bool:
    broker_url = os.getenv("CELERY_BROKER_URL")
    if not broker_url:
        return False
    try:
        from services.worker.tasks.export_ifc_payload import export_ifc_payload

        export_ifc_payload.delay(str(job_id))
        logger.info("[export-ifc-async] Job %s an Celery übergeben", job_id)
        return True
    except Exception as exc:  # pragma: no cover - celery misconfiguration
        logger.warning("[export-ifc-async] Celery-Dispatch fehlgeschlagen: %s", exc)
        return False


MATRIX_DEFAULTS = {
    "matrix_enabled": True,
    "matrix_speed": 1.0,
    "matrix_density": 1.0,
    "matrix_opacity": 0.35,
    "matrix_color": "#00ff41",
}


def _bool_from(value: Any, default: bool) -> bool:
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
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_from(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _resolve_matrix_settings(stored: Mapping[str, Any]) -> dict[str, Any]:
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


def _load_server_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _project_id_only(value: str) -> str:
    parts = [segment for segment in (value or "").split("/") if segment]
    return parts[-1] if parts else value


def _resolve_model() -> tuple[str, int]:
    settings = get_settings()
    stored = _load_server_settings()
    project_raw = stored.get("roboflow_project") or settings.roboflow.project
    project = _project_id_only(project_raw)
    version = stored.get("roboflow_version") or settings.roboflow.version
    return str(project), int(version)


def _normalize_relative_path(path_str: str) -> Path:
    cleaned = (path_str or "").replace("\\", "/")
    pure = PurePosixPath(cleaned)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if parts and parts[0] == "/":
        parts = parts[1:]
    return Path(*parts)


def _resolve_ifc_from_url(ifc_url: str) -> tuple[Path, bool]:
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
    job_root = (JOB_ROOT / str(job_id)).resolve()
    normalized = _normalize_relative_path(relative)
    candidate = (job_root / normalized).resolve()
    if job_root not in candidate.parents and candidate != job_root:
        raise HTTPException(status_code=400, detail="Ungültiger Job-Dateipfad")
    return candidate


def _resolve_ifc_from_job(job_id: str) -> Path:
    try:
        job_uuid = UUID(job_id)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Ungültige Job-ID: {job_id}") from exc

    store = FileJobStore(JOB_ROOT)
    job = store.load(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

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

    raise HTTPException(status_code=404, detail="Keine verbesserte (xBIM) IFC-Datei im Job gefunden")


def _resolve_ifc_source(ifc_url: str | None, job_id: str | None) -> tuple[Path, bool]:
    if ifc_url:
        path, cleanup = _resolve_ifc_from_url(ifc_url)
        if not path.exists():
            raise HTTPException(status_code=404, detail="IFC-Datei nicht gefunden")
        return path, cleanup
    if job_id:
        path = _resolve_ifc_from_job(job_id)
        return path, False
    raise HTTPException(status_code=400, detail="Entweder ifc_url oder job_id muss angegeben werden")


@router.post("/hottcad/validate", response_model=HottCADValidationResponse, tags=["hottcad"])
async def hottcad_validate(payload: HottCADValidateRequest) -> HottCADValidationResponse:
    path, cleanup = _resolve_ifc_source(payload.ifc_url, payload.job_id)
    cleanup_path = path if cleanup else None
    try:
        try:
            result = validate_hottcad(path, tolerance_mm=payload.tolerance_mm)
        except HTTPException:
            raise
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - validation failure
            raise HTTPException(status_code=500, detail=f"HottCAD-Validierung fehlgeschlagen: {exc}") from exc

        file_info = HottCADFileInfo(
            schema=result.file_info.get("schema") or result.schema,
            path=result.file_info.get("path"),
            sizeBytes=result.file_info.get("sizeBytes"),
            isPlainIFC=result.file_info.get("isPlainIFC"),
        )
        checks_out = [
            HottCADCheckOut(
                id=check.id,
                title=check.title,
                status=check.status,  # type: ignore[arg-type]
                details=list(check.details),
                affected={key: list(values) for key, values in check.affected.items()},
            )
            for check in result.checks
        ]
        metrics_out = HottCADMetricsOut(**vars(result.metrics))
        highlight_out = [
            HottCADHighlightOut(
                id=hs.id,
                label=hs.label,
                guids=list(hs.guids),
                productIds=list(hs.product_ids),
            )
            for hs in result.highlight_sets
        ]

        return HottCADValidationResponse(
            schema=result.schema,
            file_info=file_info,
            checks=checks_out,
            metrics=metrics_out,
            score=result.score,
            highlightSets=highlight_out,
        )
    finally:
        if cleanup_path:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass


@router.post("/hottcad/simulate", response_model=HottCADSimulationResponse, tags=["hottcad"])
async def hottcad_simulate(payload: HottCADSimulateRequest) -> HottCADSimulationResponse:
    path, cleanup = _resolve_ifc_source(payload.ifc_url, payload.job_id)
    cleanup_path = path if cleanup else None
    try:
        try:
            result = simulate_hottcad(path, tolerance_mm=payload.tolerance_mm)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - simulation failure
            raise HTTPException(status_code=500, detail=f"HottCAD-Simulation fehlgeschlagen: {exc}") from exc

        connects = [
            HottCADConnectionOut(
                walls=list(prop.walls),
                distanceMm=prop.distance_mm,
                contactType=prop.contact_type,  # type: ignore[arg-type]
                notes=list(prop.notes),
            )
            for prop in result.proposed.get("connects", [])
        ]
        space_boundaries = [
            HottCADSpaceBoundaryOut(
                walls=list(entry.get("walls", [])),
                spaces=list(entry.get("spaces", [])),
                note=entry.get("note"),
            )
            for entry in result.proposed.get("spaceBoundaries", [])
        ]
        materials = [
            HottCADMaterialSuggestionOut(
                wall=item.wall,
                thicknessMm=item.thickness_mm,
                note=item.note,
            )
            for item in result.proposed.get("materials", [])
        ]

        proposed_out = HottCADSimulationProposedOut(
            connects=connects,
            spaceBoundaries=space_boundaries,
            materials=materials,
        )

        completeness_out = HottCADCompletenessOut(**result.completeness)
        highlight_out = [
            HottCADHighlightOut(
                id=hs.id,
                label=hs.label,
                guids=list(hs.guids),
                productIds=list(hs.product_ids),
            )
            for hs in result.highlight_sets
        ]

        return HottCADSimulationResponse(
            proposed=proposed_out,
            completeness=completeness_out,
            highlightSets=highlight_out,
        )
    finally:
        if cleanup_path:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_image(
    file: Annotated[UploadFile, File(description="Bilddatei (PNG, JPG, JPEG, WEBP, BMP, TIFF)")],
    options: Annotated[str | None, Form(description="JSON mit Analyseoptionen")] = None,
) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dateiname fehlt")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unterstützt werden nur Bilddateien (PNG/JPG/JPEG/WebP/BMP/TIFF)")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Ungültiger MIME-Type für Bilddatei")

    try:
        payload = await file.read()
    except Exception as exc:  # pragma: no cover - IO failure
        raise HTTPException(status_code=400, detail=f"Upload konnte nicht gelesen werden: {exc}") from exc
    if not payload:
        raise HTTPException(status_code=400, detail="Die hochgeladene Datei ist leer")

    analyze_opts = AnalyzeOptions()
    if options:
        try:
            analyze_opts = AnalyzeOptions.model_validate_json(options)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Analyseoptionen ungültig: {exc}") from exc

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Bild konnte nicht verarbeitet werden")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".png") as tmp:
        tmp.write(payload)
        temp_path = Path(tmp.name)

    project, version = _resolve_model()
    rf_opts = RFOptions(
        project=project,
        version=version,
        confidence=float(analyze_opts.confidence),
        overlap=float(analyze_opts.overlap),
        per_class=analyze_opts.per_class_thresholds or None,
    )

    raw: dict[str, Any] = {}
    try:
        try:
            _, raw = await infer_floorplan_with_raw(temp_path, opts=rf_opts)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        detections = sv.Detections.from_inference(raw)
    except Exception:
        detections = sv.Detections.empty()
    labels = [str(item.get("class") or "") for item in raw.get("predictions", [])]
    annotated_b64: str | None = None
    try:
        annotated_scene = frame.copy()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        try:
            annotated_scene = mask_annotator.annotate(scene=annotated_scene, detections=detections)
        except Exception:
            box_annotator = sv.BoxAnnotator()
            annotated_scene = box_annotator.annotate(scene=annotated_scene, detections=detections, labels=labels)
        try:
            annotated_scene = label_annotator.annotate(scene=annotated_scene, detections=detections, labels=labels)
        except Exception:
            pass
        success, buffer = cv2.imencode(".png", annotated_scene)
        if success:
            annotated_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        annotated_b64 = None

    predictions_out = []
    per_class: dict[str, int] = {}
    centers: list[dict[str, float | str | None]] = []
    for pred in raw.get("predictions", []):
        label = str(pred.get("class") or "").strip() or "unknown"
        per_class[label] = per_class.get(label, 0) + 1
        x_val = float(pred.get("x")) if pred.get("x") is not None else None
        y_val = float(pred.get("y")) if pred.get("y") is not None else None
        width = float(pred.get("width")) if pred.get("width") is not None else None
        height = float(pred.get("height")) if pred.get("height") is not None else None
        centers.append({"label": label, "x": x_val, "y": y_val})
        pts_raw = pred.get("points") or pred.get("polygon") or None
        norm_points: list[list[float]] | None = None
        if isinstance(pts_raw, list):
            norm_points = []
            for pt in pts_raw:
                if isinstance(pt, dict):
                    norm_points.append([
                        float(pt.get("x", 0.0)),
                        float(pt.get("y", 0.0)),
                    ])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    norm_points.append([float(pt[0]), float(pt[1])])

        predictions_out.append({
            "id": str(pred.get("prediction_id") or pred.get("id") or pred.get("uuid") or ""),
            "class": label,
            "confidence": float(pred.get("confidence", 0.0)),
            "x": x_val,
            "y": y_val,
            "width": width,
            "height": height,
            "points": norm_points,
            "raw": pred,
        })

    zones_result: list[ZoneCount] = []
    if analyze_opts.zones:
        for zone in analyze_opts.zones:
            if not zone.points or len(zone.points) < 3:
                continue
            try:
                polygon = Polygon(zone.points)
            except Exception:
                continue
            zone_total = 0
            zone_per_class: dict[str, int] = {}
            for info in centers:
                cx = info.get("x")
                cy = info.get("y")
                label = str(info.get("label") or "unknown")
                if cx is None or cy is None:
                    continue
                if polygon.contains(Point(cx, cy)):
                    zone_total += 1
                    zone_per_class[label] = zone_per_class.get(label, 0) + 1
            zones_result.append(ZoneCount(name=zone.name, total=zone_total, per_class=zone_per_class))

    lines_result: list[LineCount] = []
    if analyze_opts.lines:
        for line in analyze_opts.lines:
            if not line.start or not line.end or len(line.start) < 2 or len(line.end) < 2:
                continue
            x1, y1 = float(line.start[0]), float(line.start[1])
            x2, y2 = float(line.end[0]), float(line.end[1])
            dx = x2 - x1
            dy = y2 - y1
            counts_map = {"positive": 0, "negative": 0, "on_line": 0}
            per_class_map: dict[str, dict[str, int]] = {}
            for info in centers:
                cx = info.get("x")
                cy = info.get("y")
                label = str(info.get("label") or "unknown")
                if cx is None or cy is None:
                    continue
                value = (cx - x1) * dy - (cy - y1) * dx
                side = "on_line"
                if abs(value) > 1e-6:
                    side = "positive" if value > 0 else "negative"
                counts_map[side] += 1
                per_class_map.setdefault(label, {"positive": 0, "negative": 0, "on_line": 0})
                per_class_map[label][side] += 1
            lines_result.append(LineCount(name=line.name, counts=counts_map, per_class=per_class_map))

    return AnalyzeResponse(
        model_id=f"{project}/{version}",
        confidence=rf_opts.confidence,
        overlap=rf_opts.overlap,
        total=len(predictions_out),
        per_class=per_class,
        predictions=predictions_out,
        zones=zones_result or None,
        lines=lines_result or None,
        annotated_image=annotated_b64,
        image=raw.get("image"),
        raw=raw,
    )


@router.get("/settings", response_model=SettingsPayload, tags=["settings"])
async def get_settings_api() -> SettingsPayload:
    settings = get_settings()
    stored = _load_server_settings()
    key = stored.get("roboflow_api_key", "")
    project = stored.get("roboflow_project") or settings.roboflow.project
    version = stored.get("roboflow_version") or settings.roboflow.version
    matrix_settings = _resolve_matrix_settings(stored)
    return SettingsPayload(
        has_roboflow_api_key=bool(key),
        roboflow_api_key=None,
        roboflow_project=project,
        roboflow_version=version,
        matrix_enabled=matrix_settings["matrix_enabled"],
        matrix_speed=matrix_settings["matrix_speed"],
        matrix_density=matrix_settings["matrix_density"],
        matrix_opacity=matrix_settings["matrix_opacity"],
        matrix_color=matrix_settings["matrix_color"],
    )


def _persist_env_var(name: str, value: str) -> None:
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


@router.put("/settings", response_model=SettingsPayload, tags=["settings"])
async def put_settings_api(payload: SettingsPayload) -> SettingsPayload:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    data = _load_server_settings()
    key = (payload.roboflow_api_key or "").strip()
    if payload.roboflow_api_key is not None:
        data["roboflow_api_key"] = key or None
    if payload.roboflow_project is not None:
        data["roboflow_project"] = payload.roboflow_project.strip() or None
    if payload.roboflow_version is not None:
        data["roboflow_version"] = int(payload.roboflow_version)
    if payload.matrix_enabled is not None:
        enabled_value = bool(payload.matrix_enabled)
        data["matrix_enabled"] = enabled_value
        _persist_env_var("MATRIX_ENABLED", "true" if enabled_value else "false")
    if payload.matrix_speed is not None:
        speed_value = float(payload.matrix_speed)
        data["matrix_speed"] = speed_value
        _persist_env_var("MATRIX_SPEED", f"{speed_value}")
    if payload.matrix_density is not None:
        density_value = float(payload.matrix_density)
        data["matrix_density"] = density_value
        _persist_env_var("MATRIX_DENSITY", f"{density_value}")
    if payload.matrix_opacity is not None:
        opacity_value = float(payload.matrix_opacity)
        data["matrix_opacity"] = opacity_value
        _persist_env_var("MATRIX_OPACITY", f"{opacity_value}")
    if payload.matrix_color is not None:
        color_value = (payload.matrix_color or "").strip() or None
        data["matrix_color"] = color_value
        if color_value:
            _persist_env_var("MATRIX_COLOR", color_value)
    SETTINGS_PATH.write_text(json.dumps(data), encoding="utf-8")
    if key:
        _persist_env_var("ROBOFLOW_API_KEY", key)
    matrix_settings = _resolve_matrix_settings(data)
    return SettingsPayload(
        has_roboflow_api_key=bool(data.get("roboflow_api_key")),
        roboflow_api_key=None,
        roboflow_project=data.get("roboflow_project") or get_settings().roboflow.project,
        roboflow_version=data.get("roboflow_version") or get_settings().roboflow.version,
        matrix_enabled=matrix_settings["matrix_enabled"],
        matrix_speed=matrix_settings["matrix_speed"],
        matrix_density=matrix_settings["matrix_density"],
        matrix_opacity=matrix_settings["matrix_opacity"],
        matrix_color=matrix_settings["matrix_color"],
    )
@router.post("/export-ifc", response_model=ExportIFCResponse, tags=["ifc"])
async def export_ifc(payload: ExportIFCRequest) -> ExportIFCResponse:
    logger.info(
        "[export-ifc] received payload: preds=%d, px_per_mm=%s, image=%s",
        len(payload.predictions or []),
        getattr(payload, "px_per_mm", None),
        "yes" if bool(payload.image) else "no",
    )
    if not payload.predictions:
        raise HTTPException(status_code=400, detail="Keine Vorhersagen für den IFC-Export vorhanden")

    try:
        return await run_ifc_export(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unerwarteter Fehler beim IFC-Export")
        raise HTTPException(
            status_code=500,
            detail=f"Unerwarteter Fehler beim IFC-Export: {type(exc).__name__}: {exc}",
        ) from exc


@router.post("/export-ifc/async", response_model=ExportIFCJobResponse, tags=["ifc"])
async def export_ifc_async(payload: ExportIFCRequest) -> ExportIFCJobResponse:
    if not payload.predictions:
        raise HTTPException(status_code=400, detail="Keine Vorhersagen für den IFC-Export vorhanden")

    jobs = FileJobStore(JOB_ROOT)
    payload_dict = payload.model_dump(mode="json")
    job = jobs.create(
        meta={
            "type": "export_ifc",
            "export_ifc": {
                "payload": payload_dict,
            },
        }
    )
    logger.info("[export-ifc-async] Job %s erstellt (%d predictions)", job.id, len(payload.predictions))

    if not _dispatch_ifc_job(job.id):
        async def _run_local() -> None:
            try:
                await process_ifc_job_async(job.id)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("[export-ifc-async] Fallback-Job %s fehlgeschlagen: %s", job.id, exc)

        asyncio.create_task(_run_local())

    return ExportIFCJobResponse(job_id=str(job.id))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: UUID) -> JobStatusResponse:
    jobs = FileJobStore(JOB_ROOT)
    job = jobs.load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

    meta = job.meta or {}
    export_meta = meta.get("export_ifc") or {}
    result_payload = export_meta.get("result")
    result_model: ExportIFCResponse | None = None
    if isinstance(result_payload, Mapping):
        try:
            result_model = ExportIFCResponse.model_validate(result_payload)
        except ValidationError as exc:
            logger.warning("[export-ifc-async] Ergebnis konnte nicht geladen werden (%s): %s", job_id, exc)

    error_value = meta.get("error")
    if isinstance(error_value, (dict, list)):
        error_text = json.dumps(error_value)
    elif error_value is not None:
        error_text = str(error_value)
    else:
        error_text = None

    return JobStatusResponse(
        id=str(job.id),
        status=job.status,
        progress=int(job.progress or 0),
        result=result_model,
        error=error_text,
    )


@router.post("/ifc/topview", response_model=IFCTopViewResponse, tags=["ifc"])
async def ifc_topview(payload: IFCTopViewRequest) -> IFCTopViewResponse:
    file_name = payload.file_name
    if not file_name and payload.ifc_url:
        parsed = urlparse(payload.ifc_url)
        path_part = parsed.path or payload.ifc_url
        file_name = PurePosixPath(path_part).name

    if not file_name:
        raise HTTPException(status_code=400, detail="file_name oder ifc_url muss gesetzt sein")

    try:
        ifc_path = (EXPORT_ROOT / file_name).resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"IFC-Datei {file_name} nicht gefunden") from exc

    export_root = EXPORT_ROOT.resolve()
    if export_root not in ifc_path.parents:
        raise HTTPException(status_code=400, detail="Ungültiger IFC-Pfad")

    if ifc_path.suffix.lower() != ".ifc":
        raise HTTPException(status_code=400, detail="Ziel ist keine IFC-Datei")

    out_path = ifc_path.with_name(f"{ifc_path.stem}_topview.geojson")

    needs_refresh = True
    if out_path.exists():
        try:
            needs_refresh = out_path.stat().st_mtime < ifc_path.stat().st_mtime
        except OSError:
            needs_refresh = True

    if needs_refresh or not out_path.exists():
        try:
            build_topview_geojson(ifc_path, out_path, section_elevation_mm=payload.section_elevation_mm)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"IFC-Datei {ifc_path.name} nicht gefunden")
        except Exception as exc:  # pragma: no cover - shapely/ifcopenshell failures
            logger.exception("TopView-Generierung fehlgeschlagen für %s", ifc_path)
            raise HTTPException(
                status_code=500,
                detail=f"TopView konnte nicht erzeugt werden: {type(exc).__name__}: {exc}",
            ) from exc

    return IFCTopViewResponse(topview_url=f"/files/{out_path.name}", file_name=out_path.name)


@router.post("/ifc/repair", response_model=IFCRepairResponse, tags=["ifc"])
async def ifc_repair(payload: IFCRepairRequest) -> IFCRepairResponse:
    file_name = payload.file_name
    if not file_name and payload.ifc_url:
        parsed = urlparse(payload.ifc_url)
        path_part = parsed.path or payload.ifc_url
        file_name = PurePosixPath(path_part).name

    if not file_name:
        raise HTTPException(status_code=400, detail="file_name oder ifc_url muss gesetzt sein")

    try:
        source_path = (EXPORT_ROOT / file_name).resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"IFC-Datei {file_name} nicht gefunden") from exc

    try:
        export_root = EXPORT_ROOT.resolve()
    except FileNotFoundError:
        export_root = EXPORT_ROOT

    if export_root not in source_path.parents:
        raise HTTPException(status_code=400, detail="Ungültiger IFC-Pfad")

    try:
        repaired_path, warnings = run_ifc_repair(source_path, level=payload.level, export_root=export_root)
    except RepairError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("IFC-Reparatur fehlgeschlagen")
        raise HTTPException(
            status_code=500,
            detail=f"Reparatur fehlgeschlagen: {type(exc).__name__}: {exc}",
        ) from exc

    topview_path = repaired_path.with_name(f"{repaired_path.stem}_topview.geojson")
    try:
        build_topview_geojson(repaired_path, topview_path, section_elevation_mm=None)
        topview_url = f"/files/{topview_path.name}"
    except Exception:
        logger.warning("TopView für reparierte IFC konnte nicht erstellt werden", exc_info=True)
        topview_url = None

    return IFCRepairResponse(
        file_name=repaired_path.name,
        ifc_url=f"/files/{repaired_path.name}",
        level=payload.level,
        topview_url=topview_url,
        warnings=warnings or None,
    )


@router.post("/export-pdf", response_model=ExportPDFResponse, tags=["export"])
async def export_pdf(payload: ExportPDFRequest) -> ExportPDFResponse:
    if not payload.predictions:
        raise HTTPException(status_code=400, detail="Keine Vorhersagen für den PDF-Export vorhanden")

    px_per_mm = payload.px_per_mm or 1.0
    rf_preds = [prediction_to_rfpred(pred) for pred in payload.predictions]
    normalized: list[NormalizedDet] = normalize_predictions(
        rf_preds,
        px_per_mm,
        per_class_thresholds=None,
        global_threshold=0.0,
    )
    if not normalized:
        raise HTTPException(status_code=400, detail="Keine gültigen Geometrien für den PDF-Export gefunden")

    pdf_options = PDFExportOptions.from_schema(payload.options)

    try:
        out_path, warnings = generate_vector_pdf(
            normalized=normalized,
            image_meta=payload.image or {},
            px_per_mm=px_per_mm,
            options=pdf_options,
            output_dir=DATA_ROOT / "exports",
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - CairoSVG or geometry failure
        raise HTTPException(status_code=500, detail=f"PDF Export fehlgeschlagen: {type(exc).__name__}: {exc}") from exc

    file_name = out_path.name
    return ExportPDFResponse(
        pdf_url=f"/files/{file_name}",
        file_name=file_name,
        warnings=warnings or None,
    )

