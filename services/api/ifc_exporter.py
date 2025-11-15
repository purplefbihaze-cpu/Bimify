from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any
from uuid import uuid4
import os
import json

from loguru import logger
from pydantic import BaseModel

from core.ifc.build_ifc43_model import collect_wall_polygons, write_ifc_with_spaces
from core.ml.postprocess_floorplan import (
    RASTER_PX_PER_MM,
    NormalizedDet,
    estimate_wall_axes_and_thickness,
    normalize_predictions,
)
from core.reconstruct.openings import snap_openings_to_walls
from core.reconstruct.spaces import polygonize_spaces_from_walls
from core.settings import get_settings
from core.validate.reconstruction_validation import generate_validation_report, write_validation_report
from core.validate.ifc_compliance import validate_ifc_compliance
from core.vector.ifc_topview import build_topview_geojson
import json
from services.api.schemas import ExportIFCRequest, ExportIFCResponse
from core.ml.roboflow_client import RFPred
from core.exceptions import (
    IFCExportError,
    GeometryError,
    TopViewError,
    IFCValidationError,
)


EXPORT_ROOT = Path("data/exports")

# Top-level import to avoid UnboundLocalError
try:
    import ifcopenshell  # type: ignore
except ImportError:
    ifcopenshell = None  # type: ignore


def prediction_to_rfpred(pred: Any) -> RFPred:
    data: dict[str, Any]
    if isinstance(pred, BaseModel):
        data = pred.model_dump(by_alias=True)
    elif isinstance(pred, Mapping):
        data = dict(pred)
    else:
        try:
            data = dict(pred)
        except Exception as exc:
            raise ValueError(f"Ungültige Prediction-Daten: {exc}") from exc

    klass = str(data.get("class") or data.get("label") or "").strip()
    raw_value = data.get("raw")
    raw_payload = raw_value if isinstance(raw_value, dict) else None
    if not klass and isinstance(raw_payload, dict):
        klass = str(raw_payload.get("class", "")).strip()

    polygon = None
    points_value = data.get("points")
    if isinstance(points_value, list):
        pts: list[tuple[float, float]] = []
        for pt in points_value:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                continue
            try:
                pts.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                continue
        if pts:
            polygon = pts

    bbox = None
    x = data.get("x")
    y = data.get("y")
    width = data.get("width")
    height = data.get("height")
    if all(v is not None for v in (x, y, width, height)):
        try:
            x_val = float(x) - float(width) / 2.0
            y_val = float(y) - float(height) / 2.0
            bbox = (
                x_val,
                y_val,
                float(width),
                float(height),
            )
        except (TypeError, ValueError):
            bbox = None

    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return RFPred(
        doc=0,
        page=0,
        klass=klass,
        confidence=confidence,
        polygon=polygon,
        bbox=bbox,
    )


async def run_ifc_export(payload: ExportIFCRequest, *, export_root: Path | None = None) -> ExportIFCResponse:
    global ifcopenshell  # Explicitly declare as global to avoid UnboundLocalError
    if ifcopenshell is None:
        raise ImportError("ifcopenshell is not available")
    
    # Store in local variable to avoid scope issues in nested functions and threads
    ifc_io = ifcopenshell
    
    settings = get_settings()
    calibration_payload = payload.calibration
    calibration_dict: dict | None = None
    if calibration_payload is not None:
        if isinstance(calibration_payload, BaseModel):
            calibration_dict = calibration_payload.model_dump()
        elif isinstance(calibration_payload, Mapping):
            calibration_dict = dict(calibration_payload)
        elif hasattr(calibration_payload, "__dict__"):
            calibration_dict = dict(calibration_payload.__dict__)

    px_per_mm = payload.px_per_mm
    if px_per_mm is None and calibration_payload is not None:
        if isinstance(calibration_payload, BaseModel):
            px_per_mm = calibration_payload.px_per_mm
        elif isinstance(calibration_dict, dict):
            px_per_mm = calibration_dict.get("px_per_mm")
    if px_per_mm is None:
        px_per_mm = 1.0

    warnings: list[str] = []
    if payload.px_per_mm is None:
        warnings.append("px_per_mm nicht angegeben – Standardwert 1.0 verwendet")
    if calibration_payload and payload.px_per_mm is None:
        warnings.append("Kalibrierung übernommen – px_per_mm aus Pixel-to-Meta ermittelt")

    flip_y_flag = bool(payload.flip_y) if payload.flip_y is not None else False
    if not flip_y_flag and isinstance(calibration_dict, dict):
        flip_y_flag = bool(calibration_dict.get("flip_y"))

    image_height_px: float | None = payload.image_height_px
    image_meta = payload.image if isinstance(payload.image, Mapping) else None
    if image_height_px is None and image_meta:
        for key in ("height", "Height", "image_height", "imageHeight"):
            value = image_meta.get(key)
            if value is None:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                image_height_px = candidate
                break
        if image_height_px is None:
            meta_section = image_meta.get("meta")
            if isinstance(meta_section, Mapping):
                for key in ("height", "Height"):
                    value = meta_section.get(key)
                    if value is None:
                        continue
                    try:
                        candidate = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(candidate) and candidate > 0.0:
                        image_height_px = candidate
                        break

    if flip_y_flag and image_height_px is None:
        warnings.append("flip_y aktiviert, aber Bildhöhe unbekannt – Spiegelung deaktiviert")
        flip_y_flag = False

    if isinstance(calibration_dict, dict):
        calibration_dict.setdefault("flip_y", flip_y_flag)
        if image_height_px is not None:
            calibration_dict.setdefault("image_height_px", image_height_px)

    window_height_mm = payload.window_height_mm if payload.window_height_mm is not None else 1000.0
    if window_height_mm >= payload.window_head_elevation_mm:
        window_height_mm = max(payload.window_head_elevation_mm - 100.0, 100.0)
        warnings.append("Fensterhöhe musste angepasst werden, da sie oberhalb des Sturzes lag")

    rf_preds = [prediction_to_rfpred(pred) for pred in payload.predictions]
    normalized: list[NormalizedDet] = normalize_predictions(
        rf_preds,
        px_per_mm,
        per_class_thresholds=None,
        global_threshold=0.0,
        flip_y=flip_y_flag,
        image_height_px=image_height_px,
    )

    if not normalized:
        raise GeometryError("Keine gültigen Geometrien für den IFC-Export gefunden")

    spaces = polygonize_spaces_from_walls(normalized)
    effective_raster_px_per_mm = RASTER_PX_PER_MM
    if px_per_mm and px_per_mm > 0.0:
        effective_raster_px_per_mm = max(px_per_mm, 1e-3)
    wall_axes = estimate_wall_axes_and_thickness(
        normalized,
        raster_px_per_mm=effective_raster_px_per_mm,
    )
    wall_polygons = collect_wall_polygons(normalized)

    wall_count = sum(1 for det in normalized if det.type == "WALL")
    opening_dets = [det for det in normalized if det.type in ("DOOR", "WINDOW")]
    if wall_count == 0:
        warnings.append("Keine Wände erkannt – IFC enthält nur Grundelemente")
    unmatched_openings = 0
    if opening_dets:
        try:
            assignments, _ = snap_openings_to_walls(normalized, wall_axes=wall_axes)
            unmatched_openings = sum(1 for assignment in assignments if assignment.wall_index is None)
        except Exception:
            unmatched_openings = 0
    if unmatched_openings:
        warnings.append(
            f"{unmatched_openings} Öffnung(en) konnten keiner Wand zugeordnet werden – Platzierung erfolgt ohne Wand"
        )

    project_name = (payload.project_name or "Bimify Project").strip() or "Bimify Project"
    storey_name = (payload.storey_name or "EG").strip() or "EG"

    export_dir = export_root or EXPORT_ROOT
    export_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{uuid4().hex}.ifc"
    out_path = export_dir / file_name

    improved_ifc_url: str | None = None
    improved_stats_url: str | None = None
    improved_wexbim_url: str | None = None
    validation_report_url: str | None = None

    validation_path = out_path.with_name(f"{out_path.stem}_validation.json")

    try:
        def _write_sync() -> None:
            ifc_settings = getattr(settings, "ifc", None)
            write_ifc_with_spaces(
                normalized=normalized,
                spaces=spaces,
                out_path=out_path,
                project_name=project_name,
                storey_name=storey_name,
                storey_elevation=0.0,
                wall_axes=wall_axes,
                wall_polygons=wall_polygons,
                storey_height_mm=payload.storey_height_mm,
                door_height_mm=payload.door_height_mm,
                window_height_mm=window_height_mm,
                window_head_elevation_mm=payload.window_head_elevation_mm,
                px_per_mm=px_per_mm,
                calibration=calibration_dict,
                schema_version=getattr(ifc_settings, "schema", "IFC4"),
                wall_thickness_standards_mm=getattr(ifc_settings, "wall_thickness_standards_mm", None),
                owner_org_name=getattr(ifc_settings, "owner_org_name", None),
                owner_org_identification=getattr(ifc_settings, "owner_org_identification", None),
                app_identifier=getattr(ifc_settings, "app_identifier", None),
                app_full_name=getattr(ifc_settings, "app_full_name", None),
                app_version=getattr(ifc_settings, "app_version", None),
                person_identification=getattr(ifc_settings, "person_identification", None),
                person_given_name=getattr(ifc_settings, "person_given_name", None),
                person_family_name=getattr(ifc_settings, "person_family_name", None),
            )

        logger.info("[export-ifc] writing IFC to %s ...", out_path)
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(_write_sync)
        except Exception as write_exc:
            # Check if this is an owner history error - be more specific
            error_msg = str(write_exc).lower()
            error_type = type(write_exc).__name__
            is_owner_error = (
                "owner" in error_msg
                or "identification" in error_msg
                or "ifcorganization" in error_msg
                or "ifcperson" in error_msg
                or "ifcownerhistory" in error_msg
                or "please create a user" in error_msg
                or "doesn't have the following attributes" in error_msg
            )
            if is_owner_error:
                logger.warning(
                    "[export-ifc] Owner history error detected (type=%s, msg=%s), "
                    "this should be fixed by schema-safe setup. Not retrying to avoid loop.",
                    error_type,
                    write_exc,
                )
                # Don't retry - the error should be fixed by schema-safe setup
                # If it still fails, it's a different issue that needs investigation
                raise RuntimeError(f"IFC owner history setup failed: {write_exc}") from write_exc
            else:
                # Re-raise if it's not an owner history error
                raise
        dt = time.perf_counter() - t0
        logger.info("[export-ifc] IFC written in %.2fs", dt)

        # Validate IFC parseability; if it fails, rewrite without openings as safe fallback
        try:
            if ifc_io is None:
                raise ImportError("ifcopenshell is not available")
            _ = await asyncio.to_thread(ifc_io.open, str(out_path))
        except Exception as parse_exc:
            logger.warning("IFC parse failed (%s). Rewriting without openings as fallback.", parse_exc)
            # This is expected behavior - continue with fallback
            def _write_safe_sync() -> None:
                ifc_settings = getattr(settings, "ifc", None)
                write_ifc_with_spaces(
                    normalized=[d for d in normalized if d.type not in ("DOOR", "WINDOW")],
                    spaces=spaces,
                    out_path=out_path,
                    project_name=project_name,
                    storey_name=storey_name,
                    storey_elevation=0.0,
                    wall_axes=wall_axes,
                    wall_polygons=wall_polygons,
                    storey_height_mm=payload.storey_height_mm,
                    door_height_mm=payload.door_height_mm,
                    window_height_mm=window_height_mm,
                    window_head_elevation_mm=payload.window_head_elevation_mm,
                    px_per_mm=px_per_mm,
                    calibration=calibration_dict,
                    schema_version=getattr(ifc_settings, "schema", "IFC4"),
                    wall_thickness_standards_mm=getattr(ifc_settings, "wall_thickness_standards_mm", None),
                    owner_org_name=getattr(ifc_settings, "owner_org_name", None),
                    owner_org_identification=getattr(ifc_settings, "owner_org_identification", None),
                    app_identifier=getattr(ifc_settings, "app_identifier", None),
                    app_full_name=getattr(ifc_settings, "app_full_name", None),
                    app_version=getattr(ifc_settings, "app_version", None),
                    person_identification=getattr(ifc_settings, "person_identification", None),
                    person_given_name=getattr(ifc_settings, "person_given_name", None),
                    person_family_name=getattr(ifc_settings, "person_family_name", None),
                )
            await asyncio.to_thread(_write_safe_sync)
            warnings.append("IFC wurde ohne Öffnungen neu geschrieben, da die ursprüngliche Datei nicht lesbar war.")
        
        # Run IFC compliance validation
        compliance_warnings = []
        try:
            # Check if ifcopenshell is available before calling validate_ifc_compliance
            if ifc_io is None:
                logger.warning("ifcopenshell is not available - skipping IFC compliance validation")
                compliance_warnings.append("IFC compliance validation skipped (ifcopenshell not available)")
            else:
                def _validate_compliance_sync() -> None:
                    nonlocal compliance_warnings
                    # validate_ifc_compliance has its own ifcopenshell import, so we don't need global here
                    compliance_report = validate_ifc_compliance(out_path)
                    if not compliance_report.is_compliant:
                        error_count = sum(1 for issue in compliance_report.issues if issue.severity == "ERROR")
                        warning_count = sum(1 for issue in compliance_report.issues if issue.severity == "WARNING")
                        if error_count > 0:
                            compliance_warnings.append(f"IFC Compliance: {error_count} error(s) found")
                        if warning_count > 0:
                            compliance_warnings.append(f"IFC Compliance: {warning_count} warning(s) found")
                        logger.info("IFC compliance check: %d errors, %d warnings", error_count, warning_count)
                    else:
                        logger.info("IFC file passed compliance validation")
                
                await asyncio.to_thread(_validate_compliance_sync)
        except IFCValidationError as compliance_exc:
            logger.warning("IFC compliance validation error: %s", compliance_exc)
            compliance_warnings.append(f"IFC compliance validation error: {compliance_exc}")
        except Exception as compliance_exc:
            logger.warning("IFC compliance validation failed: %s", compliance_exc)
            compliance_warnings.append("IFC compliance validation could not be completed")
        
        # Add compliance warnings
        warnings.extend(compliance_warnings)

        # Build/refresh topview automatically
        try:
            # Use fixed cut height of 1000mm for floor plan view to ensure all walls are captured
            cut_height = 1000.0
            topview_path = out_path.with_name(f"{out_path.stem}_topview.geojson")
            await asyncio.to_thread(build_topview_geojson, out_path, topview_path, section_elevation_mm=cut_height)
            topview_url = f"/files/{topview_path.name}"
        except TopViewError as tv_exc:
            logger.warning("TopView-Erstellung fehlgeschlagen: %s", tv_exc)
            topview_url = None
        except Exception as tv_exc:
            logger.warning("Unerwarteter Fehler bei TopView-Erstellung: %s", tv_exc)
            topview_url = None

        # Optional: debug overlay for openings
        try:
            if os.getenv("DEBUG_OPENINGS", "").lower() in {"1", "true", "yes"}:
                from shapely.geometry import mapping
                debug = {"type": "FeatureCollection", "features": []}
                for det in normalized:
                    if det.type not in {"WINDOW", "DOOR"}:
                        continue
                    try:
                        debug["features"].append({
                            "type": "Feature",
                            "properties": {"type": det.type, "source": det.attrs.get("geometry_source") if isinstance(det.attrs, dict) else None, "iou": det.attrs.get("iou") if isinstance(det.attrs, dict) else None},
                            "geometry": mapping(det.geom),
                        })
                    except Exception:
                        continue
                debug_path = out_path.with_name(f"{out_path.stem}_openings.geojson")
                await asyncio.to_thread(debug_path.write_text, json.dumps(debug), "utf-8")
        except Exception:
            pass

        async def _run_validation_bg() -> None:
            try:
                def _validate_sync() -> None:
                    report = generate_validation_report(normalized, wall_axes, out_path)
                    write_validation_report(report, validation_path)

                await asyncio.to_thread(_validate_sync)
            except Exception as validation_exc:
                logger.warning("Validierungsreport konnte nicht erzeugt werden: %s", validation_exc)

        try:
            asyncio.create_task(_run_validation_bg())
            validation_report_url = f"/files/{validation_path.name}"
        except Exception:
            pass
    except (IFCExportError, GeometryError) as exc:
        logger.exception("IFC-Export fehlgeschlagen beim Schreiben von %s", out_path)
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as exc:  # pragma: no cover - IfcOpenShell failure path
        logger.exception("Unerwarteter Fehler beim IFC-Export von %s", out_path)
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise IFCExportError(f"IFC-Export fehlgeschlagen ({out_path.name}): {type(exc).__name__}: {exc}", {"out_path": str(out_path)}) from exc

    preprocess_settings = getattr(getattr(settings, "geometry", None), "preprocess", None)
    if preprocess_settings and getattr(preprocess_settings, "enabled", False):
        logger.info("xBIM-Vorverarbeitung ist deaktiviert – IFC wird ohne Nachbearbeitung bereitgestellt.")

    return ExportIFCResponse(
        ifc_url=f"/files/{file_name}",
        improved_ifc_url=improved_ifc_url,
        improved_ifc_stats_url=improved_stats_url,
        improved_wexbim_url=improved_wexbim_url,
        file_name=file_name,
        topview_url=locals().get("topview_url"),
        validation_report_url=validation_report_url,
        storey_height_mm=payload.storey_height_mm,
        door_height_mm=payload.door_height_mm,
        window_height_mm=window_height_mm,
        window_head_elevation_mm=payload.window_head_elevation_mm,
        px_per_mm=px_per_mm,
        warnings=warnings or None,
    )



