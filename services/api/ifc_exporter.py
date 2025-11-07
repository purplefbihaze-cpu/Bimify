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
from core.vector.ifc_topview import build_topview_geojson
import json
from services.api.schemas import ExportIFCRequest, ExportIFCResponse
from core.ml.roboflow_client import RFPred


EXPORT_ROOT = Path("data/exports")


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
        raise ValueError("Keine gültigen Geometrien für den IFC-Export gefunden")

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
                schema_version=getattr(getattr(settings, "ifc", None), "schema", "IFC4"),
                wall_thickness_standards_mm=getattr(getattr(settings, "ifc", None), "wall_thickness_standards_mm", None),
            )

        logger.info("[export-ifc] writing IFC to %s ...", out_path)
        t0 = time.perf_counter()
        await asyncio.to_thread(_write_sync)
        dt = time.perf_counter() - t0
        logger.info("[export-ifc] IFC written in %.2fs", dt)

        # Build/refresh topview automatically
        try:
            cut_height = 1000.0
            if payload.window_head_elevation_mm and window_height_mm:
                sill = max(payload.window_head_elevation_mm - window_height_mm, 0.0)
                cut_height = max(min(payload.window_head_elevation_mm - (window_height_mm / 2.0), payload.storey_height_mm or 3000.0), sill + 1.0)
            topview_path = out_path.with_name(f"{out_path.stem}_topview.geojson")
            await asyncio.to_thread(build_topview_geojson, out_path, topview_path, section_elevation_mm=cut_height)
            topview_url = f"/files/{topview_path.name}"
        except Exception as tv_exc:
            logger.warning("TopView konnte nicht erstellt werden: %s", tv_exc)
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
    except Exception as exc:  # pragma: no cover - IfcOpenShell failure path
        logger.exception("IFC-Export fehlgeschlagen beim Schreiben von %s", out_path)
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"IFC-Export fehlgeschlagen ({out_path.name}): {type(exc).__name__}: {exc}") from exc

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


