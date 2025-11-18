"""
IFC Export V2

Post-processing pipeline to clean pixel noise from Roboflow predictions,
followed by normalization and IFC export.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
import math
import re
from pathlib import Path
from typing import Any
from uuid import uuid4
import json

from loguru import logger
from pydantic import BaseModel

from services.api.schemas import ExportIFCV2Request, ExportIFCV2Response, GeometryFidelityLevel
from core.exceptions import (
    IFCExportError,
    GeometryError,
    GeometryValidationError,
    InsufficientConfidenceError,
    WallReconstructionError,
)
from core.ml.postprocess_floorplan import (
    NormalizedDet,
    normalize_predictions,
    estimate_wall_axes_and_thickness,
    RASTER_PX_PER_MM,
)
from core.ml.roboflow_client import RFPred, RoboflowPrediction
from core.ml.pipeline_config import PipelineConfig, GapClosureMode
from core.ml.postprocess_v2 import process_roboflow_predictions_v2
from core.reconstruct.spaces import polygonize_spaces_from_walls, SpaceConfig
from core.reconstruct.walls import (
    repair_wall_detections,
    collapse_wall_nodes,
    resolve_t_junctions,
    merge_overlapping_walls,
)
from core.reconstruct.spaces_graph import polygonize_spaces_graph
from core.ifc.build_ifc43_model_v2 import write_ifc_v2
from core.validate.fallback_generation import ensure_minimum_geometry
from core.ifc.viewer_v2 import generate_predictions_viewer_html
from core.settings import get_settings
from core.vector.ifc_topview import build_topview_geojson
from core.validate.reconstruction_validation import generate_validation_report, write_validation_report
from core.validate.input_validation import validate_raw_predictions
from core.validate.geometry_validation import validate_reconstruction
from core.ifc.template_analyzer import compare_ifc_to_template
from core.metrics.pipeline_metrics import PipelineMetrics


EXPORT_ROOT = Path("data/exports")

# Top-level import to avoid UnboundLocalError
try:
    import ifcopenshell  # type: ignore
except ImportError:
    ifcopenshell = None  # type: ignore


def prediction_to_rfpred_v2(pred: RoboflowPrediction | dict[str, Any] | Any) -> RFPred:
    """
    Convert prediction (RoboflowPrediction, dict, or other) to RFPred format.
    
    Args:
        pred: RoboflowPrediction, dict, or other object with prediction data
        
    Returns:
        RFPred object for normalization
    """
    # If already a RoboflowPrediction, use it directly
    if isinstance(pred, RoboflowPrediction):
        data = pred.model_dump(by_alias=True)
    else:
        # Try to parse as RoboflowPrediction for validation
        try:
            if isinstance(pred, BaseModel):
                data = pred.model_dump(by_alias=True)
            elif isinstance(pred, Mapping):
                data = dict(pred)
            else:
                data = dict(pred)
            # Validate by creating RoboflowPrediction (will raise if invalid)
            _ = RoboflowPrediction(**data)
        except Exception as exc:
            # Fallback to old behavior for backward compatibility
            if isinstance(pred, BaseModel):
                data = pred.model_dump(by_alias=True)
            elif isinstance(pred, Mapping):
                data = dict(pred)
            else:
                try:
                    data = dict(pred)
                except Exception as dict_exc:
                    raise ValueError(f"Ungültige Prediction-Daten: {dict_exc}") from dict_exc

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


async def run_ifc_export_v2(
    payload: ExportIFCV2Request, *, export_root: Path | None = None
) -> ExportIFCV2Response:
    """
    Main IFC export function with post-processing pipeline.
    
    Pipeline:
    1. Post-processing to clean pixel noise from Roboflow predictions
    2. Generate viewer for cleaned predictions
    3. Normalize cleaned predictions
    4. Generate spaces and wall axes
    5. Export to IFC with validation and reports
    
    Args:
        payload: Export request with predictions
        export_root: Optional export directory (defaults to EXPORT_ROOT)
        
    Returns:
        Export response with IFC file URL, viewer URL, and metadata
    """
    # Re-import ifcopenshell if needed (hot reload support)
    global ifcopenshell
    if ifcopenshell is None:
        try:
            import ifcopenshell  # type: ignore
        except ImportError:
            raise ImportError("ifcopenshell is not available")
    
    logger.info("[export-ifc-v2] Starting IFC export")
    logger.info("[export-ifc-v2] Received %d predictions", len(payload.predictions or []))
    
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
    window_head_elevation_mm = payload.window_head_elevation_mm if payload.window_head_elevation_mm is not None else 2000.0
    if window_height_mm >= window_head_elevation_mm:
        window_height_mm = max(window_head_elevation_mm - 100.0, 100.0)
        warnings.append("Fensterhöhe musste angepasst werden, da sie oberhalb des Sturzes lag")

    # Initialize metrics and config
    metrics = PipelineMetrics()
    pipeline_config = PipelineConfig.default()  # Can be extended to accept config from payload
    
    # Set default geometry preservation settings
    pipeline_config.preserve_exact_geometry = True  # Erhalte Original-Polygone
    pipeline_config.gap_closure_mode = GapClosureMode.PROPOSE_ONLY  # Warnen, nicht reparieren
    pipeline_config.enable_input_validation = True  # Filtere nur Rauschen
    
    # Override with payload values if provided
    if hasattr(payload, 'preserve_exact_geometry') and payload.preserve_exact_geometry is not None:
        pipeline_config.preserve_exact_geometry = payload.preserve_exact_geometry
    
    # Apply geometry fidelity override if specified
    if payload.geometry_fidelity is not None:
        pipeline_config = pipeline_config.with_fidelity(payload.geometry_fidelity)
        logger.info("[export-ifc-v2] Applied geometry fidelity level: %s", payload.geometry_fidelity.value)
    else:
        # Legacy mode: use existing config flags (backward compatibility)
        logger.debug("[export-ifc-v2] Using legacy config flags (geometry_fidelity not specified)")
    
    # Use consistent variable name throughout
    config = pipeline_config
    
    # Step 1: Use RoboflowPrediction objects directly (already validated by Pydantic)
    # Convert to dict format only for post-processing pipeline
    raw_predictions: list[RoboflowPrediction | dict[str, Any]] = list(payload.predictions)
    
    metrics.total_input_predictions = len(raw_predictions)
    
    # Step 1.5: Input validation (RoboflowPrediction objects are already validated by Pydantic,
    # but we run additional geometry/dimension checks here)
    t_validation_start = time.perf_counter()
    if config.enable_input_validation:
        validation_result = validate_raw_predictions(raw_predictions, config=config, px_per_mm=px_per_mm)
        # Convert validated predictions to dict format for post-processing
        raw_predictions = [
            pred.model_dump(by_alias=True) if isinstance(pred, RoboflowPrediction) else pred
            for pred in validation_result.valid_predictions
        ]
        metrics.valid_input_predictions = len(raw_predictions)
        metrics.invalid_input_predictions = len(validation_result.invalid_predictions)
        metrics.filtered_by_confidence = validation_result.statistics.get("filtered_by_confidence", 0)
        metrics.filtered_by_points = validation_result.statistics.get("filtered_by_points", 0)
        metrics.filtered_by_geometry = validation_result.statistics.get("filtered_by_geometry", 0)
        metrics.filtered_by_dimensions = validation_result.statistics.get("filtered_by_dimensions", 0)
        for warning in validation_result.warnings:
            metrics.add_warning(warning, "input_validation")
        if not validation_result.is_valid:
            # Check if failure is due to confidence issues
            confidence_filtered = validation_result.statistics.get("filtered_by_confidence", 0)
            total_invalid = len(validation_result.invalid_predictions)
            # Use 80% threshold instead of exact match for more lenient confidence filtering
            # This prevents false positives when some predictions fail for other reasons
            if confidence_filtered > 0 and confidence_filtered >= total_invalid * 0.8:
                raise InsufficientConfidenceError(
                    f"Most predictions filtered by confidence threshold: {confidence_filtered}/{total_invalid} predictions below threshold"
                )
            raise GeometryValidationError(
                f"Input validation failed: {total_invalid} invalid predictions (confidence: {confidence_filtered}, geometry: {validation_result.statistics.get('filtered_by_geometry', 0)})"
            )
    metrics.time_input_validation = time.perf_counter() - t_validation_start
    
    # Step 2: Apply post-processing pipeline
    logger.info("[export-ifc-v2] Applying post-processing pipeline to %d predictions", len(raw_predictions))
    t_postprocess_start = time.perf_counter()
    cleaned_predictions = process_roboflow_predictions_v2(
        raw_predictions,
        px_per_mm=px_per_mm,
        config=config,
    )
    metrics.total_processed = len(cleaned_predictions)
    t_postprocess = time.perf_counter() - t_postprocess_start
    metrics.time_post_processing = t_postprocess
    logger.info("[export-ifc-v2] Post-processing completed in %.3fs (%d predictions)", t_postprocess, len(cleaned_predictions))
    
    # Additional check: ensure we have predictions after post-processing
    if not cleaned_predictions:
        raise GeometryError("Keine gültigen Predictions nach Post-Processing gefunden")
    
    # Step 3: Generate viewer for cleaned predictions (in memory, write only on success)
    # Sanitize project_name to prevent path traversal attacks
    raw_project_name = (payload.project_name or "Bimify Project").strip() or "Bimify Project"
    # Remove path traversal characters and sanitize: only allow alphanumeric, spaces, hyphens, underscores
    project_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', raw_project_name)
    # Remove leading/trailing spaces and collapse multiple spaces
    project_name = re.sub(r'\s+', ' ', project_name).strip()
    # Fallback if sanitization removed everything
    if not project_name:
        project_name = "Bimify Project"
    # Limit length to prevent filesystem issues
    if len(project_name) > 100:
        project_name = project_name[:100]
    
    export_dir = export_root or EXPORT_ROOT
    export_dir.mkdir(parents=True, exist_ok=True)
    
    viewer_html: str | None = None
    viewer_url: str | None = None
    try:
        # Generate viewer HTML in memory
        viewer_html = generate_predictions_viewer_html(
            cleaned_predictions,
            output_path=None,  # Don't write yet
            title=f"Cleaned Predictions Viewer - {project_name}",
            return_string=True,
        )
        logger.info("[export-ifc-v2] Viewer HTML generated in memory")
    except (ValueError, TypeError, KeyError) as viewer_exc:
        # Specific exceptions for viewer generation - log but don't fail export
        logger.warning("[export-ifc-v2] Viewer generation failed (non-critical): %s", viewer_exc)
    except Exception as viewer_exc:
        # Catch-all for unexpected errors - log but continue
        logger.warning("[export-ifc-v2] Viewer generation failed with unexpected error: %s", viewer_exc, exc_info=True)
    
    # Step 4: Convert cleaned predictions to RFPred format
    rf_preds = [prediction_to_rfpred_v2(pred) for pred in cleaned_predictions]
    
    # Step 5: Normalize predictions
    t_normalize_start = time.perf_counter()
    normalized: list[NormalizedDet] = normalize_predictions(
        rf_preds,
        px_per_mm,
        per_class_thresholds=None,
        global_threshold=0.0,
        flip_y=flip_y_flag,
        image_height_px=image_height_px,
    )
    metrics.time_normalization = time.perf_counter() - t_normalize_start

    if not normalized:
        raise GeometryError("Keine gültigen Geometrien für den IFC-Export gefunden (nach Post-Processing)")

    # Step 5.2: Wall Graph Repair (snap, prune, merge, orient, rectangle fallback)
    try:
        normalized = list(repair_wall_detections(
            list(normalized),
            snap_endpoints_mm=8.0,
            min_segment_len_mm=400.0,
            min_area_mm2=max(200_000.0, float(getattr(config, "min_room_area_m2", 1.0)) * 200_000.0),
            angle_tol_deg=float(getattr(config, "angle_tolerance", 5.0)),
        ))
    except Exception as repair_exc:
        logger.warning("Wall graph repair failed (continuing with original): %s", repair_exc)

    # Step 5.3: Node Collapse (merge nearby endpoints) on polygons
    try:
        normalized = list(collapse_wall_nodes(list(normalized)))
    except Exception as exc:
        logger.warning("Node collapse failed (non-critical): %s", exc)

    # Step 5.4: T-Junction Resolver and Overlap Merge (polygons)
    try:
        normalized = list(resolve_t_junctions(list(normalized)))
    except Exception as exc:
        logger.warning("T-junction resolve failed (non-critical): %s", exc)
    try:
        normalized = list(merge_overlapping_walls(list(normalized)))
    except Exception as exc:
        logger.warning("Overlap merge failed (non-critical): %s", exc)

    # Step 5.5: Reconstruction
    t_reconstruction_start = time.perf_counter()
    # Deterministic spaces (graph-like wrapper), then fallback to Two-Pass heuristics
    spaces = []
    try:
        spaces = polygonize_spaces_graph(normalized, config=None)
    except Exception as exc:
        logger.warning("Deterministic space polygonization failed (will use Two-Pass): %s", exc)
    # Use pipeline config thresholds for space extraction; retry with permissive fallback if needed
    try:
        base_space_cfg = SpaceConfig(
            min_room_area_m2=float(getattr(config, "min_room_area_m2", 1.0)),
            gap_threshold_mm=float(getattr(config, "max_gap_close", 50.0)),
            gap_buffer_mm=25.0,
            min_space_area_mm2=max(10000.0, float(getattr(config, "min_room_area_m2", 1.0)) * 1_000_000.0),
            treat_open_areas_as_rooms=False,
        )
    except Exception:
        base_space_cfg = SpaceConfig()
    if not spaces:
        spaces = polygonize_spaces_from_walls(normalized, config=base_space_cfg)
    if not spaces:
        fallback_space_cfg = SpaceConfig(
            min_room_area_m2=min(0.5, float(getattr(config, "min_room_area_m2", 1.0))),
            gap_threshold_mm=100.0,
            gap_buffer_mm=50.0,
            min_space_area_mm2=10000.0,
            treat_open_areas_as_rooms=True,
        )
        spaces = polygonize_spaces_from_walls(normalized, config=fallback_space_cfg)
    # Step 5.6: Validation Fallback (Never-Abort): ensure at least one wall and one space
    try:
        doc = int(getattr(normalized[0], "doc", 0)) if normalized else 0
        page = int(getattr(normalized[0], "page", 0)) if normalized else 0
    except Exception:
        doc = 0
        page = 0
    normalized, spaces, fb_notes = ensure_minimum_geometry(normalized, spaces, doc=doc, page=page)
    
    # Only compute wall axes if not preserving exact geometry (saves CPU time)
    if not config.preserve_exact_geometry:
        effective_raster_px_per_mm = RASTER_PX_PER_MM
        if px_per_mm and px_per_mm > 0.0:
            effective_raster_px_per_mm = max(px_per_mm, 1e-3)
        try:
            wall_axes = estimate_wall_axes_and_thickness(
                normalized,
                raster_px_per_mm=effective_raster_px_per_mm,
                max_dimension=config.skeletonization_max_dimension,
                target_dpi=config.skeletonization_target_dpi,
                enable_cache=config.enable_skeletonization_cache,
            )
        except Exception as e:
            logger.error(f"Wall reconstruction (skeletonization) failed: {e}")
            raise WallReconstructionError(f"Failed to reconstruct wall axes: {e}") from e
    else:
        # Skip expensive wall axis reconstruction when preserving exact geometry
        wall_axes = []
        logger.debug("[export-ifc-v2] Skipping wall axis reconstruction (preserve_exact_geometry=True)")
    
    metrics.time_reconstruction = time.perf_counter() - t_reconstruction_start
    
    # Update metrics
    metrics.total_walls = len([d for d in normalized if d.type == "WALL"])
    metrics.total_doors = len([d for d in normalized if d.type == "DOOR"])
    metrics.total_windows = len([d for d in normalized if d.type == "WINDOW"])
    metrics.total_spaces = len(spaces)
    metrics.total_wall_axes = len(wall_axes)
    
    # Step 5.6: Geometry validation
    if config.enable_geometry_validation:
        geometry_validation = validate_reconstruction(normalized, wall_axes, spaces, config=config)
        metrics.closed_spaces = geometry_validation.statistics.get("closed_spaces", 0)
        metrics.open_spaces = geometry_validation.statistics.get("open_spaces", 0)
        metrics.overlapping_spaces = geometry_validation.statistics.get("overlapping_spaces", 0)
        metrics.spaces_below_min_area = geometry_validation.statistics.get("spaces_below_min_area", 0)
        metrics.parallel_axes = geometry_validation.statistics.get("parallel_axes", 0)
        metrics.non_parallel_axes = geometry_validation.statistics.get("non_parallel_axes", 0)
        if geometry_validation.statistics.get("wall_axes_quality_scores"):
            metrics.skeletonization_quality_scores = geometry_validation.statistics["wall_axes_quality_scores"]
        for warning in geometry_validation.warnings:
            metrics.add_warning(warning, "geometry_validation")
        for error in geometry_validation.errors:
            metrics.add_error(error)
        if geometry_validation.has_critical_issues():
            logger.warning("[export-ifc-v2] Geometry validation found critical issues")
            warnings.extend(geometry_validation.errors[:5])  # Add first 5 errors to warnings

    if not any(det.type == "WALL" for det in normalized):
        warnings.append("Keine Wände erkannt – IFC enthält nur Grundelemente")

    # Step 5.7: Pre-export validation with gap repair (if enabled)
    repaired_wall_axes = wall_axes
    if config.enable_geometry_validation:
        from core.validate.reconstruction_validation import validate_before_ifc_export
        is_valid, pre_warnings, repaired_axes_result, action_items, gap_proposals = validate_before_ifc_export(
            normalized, wall_axes, auto_repair=True, config=config
        )
        if repaired_axes_result is not None:
            repaired_wall_axes = repaired_axes_result
            logger.info("[export-ifc-v2] Using gap-repaired wall axes for IFC export")
        for warning in pre_warnings:
            warnings.append(warning)
            metrics.add_warning(warning, "pre_export_validation")
        if gap_proposals:
            logger.info("[export-ifc-v2] Generated %d gap repair proposals (mode=PROPOSE)", len(gap_proposals))

    storey_name = (payload.storey_name or "EG").strip() or "EG"

    file_name = f"{uuid4().hex}.ifc"
    out_path = export_dir / file_name

    validation_report_url: str | None = None
    comparison_report_url: str | None = None

    validation_path = out_path.with_name(f"{out_path.stem}_validation.json")

    try:
        # Capture config in closure to avoid UnboundLocalError
        config_for_export = config
        
        def _write_sync() -> None:
            # Create IFCExportConfig from payload and config
            from core.ifc.build_ifc43_model_v2 import IFCExportConfig
            
            export_config = IFCExportConfig(
                schema="IFC4",
                geometry_fidelity=payload.geometry_fidelity.value if payload.geometry_fidelity else None,
                gap_closure_mode=config_for_export.gap_closure_mode.value,
                validate_schema=True,
                validate_topology=True,
            )
            
            write_ifc_v2(
                normalized=normalized,
                spaces=spaces,
                out_path=out_path,
                storey_height_mm=payload.storey_height_mm,
                door_height_mm=payload.door_height_mm,
                window_height_mm=window_height_mm,
                floor_thickness_mm=payload.floor_thickness_mm,
                px_per_mm=px_per_mm,
                # Let builder compute axes if preserve_exact_geometry=True (for robust axis rep)
                wall_axes=(repaired_wall_axes if not config_for_export.preserve_exact_geometry else None),
                config=export_config,
            )

        logger.info("[export-ifc-v2] writing IFC to %s ...", out_path)
        t0 = time.perf_counter()
        metrics.time_ifc_export = 0.0
        try:
            # Add timeout to prevent server hang (5 minutes max)
            await asyncio.wait_for(asyncio.to_thread(_write_sync), timeout=300.0)
            metrics.time_ifc_export = time.perf_counter() - t0
        except asyncio.TimeoutError:
            metrics.time_ifc_export = time.perf_counter() - t0
            logger.error("[export-ifc-v2] IFC export timed out after 5 minutes")
            raise IFCExportError("IFC export timed out after 5 minutes - file may be too large or complex") from None
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
                    "[export-ifc-v2] Owner history error detected (type=%s, msg=%s), "
                    "this should be fixed by schema-safe setup. Not retrying to avoid loop.",
                    error_type,
                    write_exc,
                )
                raise RuntimeError(f"IFC owner history setup failed: {write_exc}") from write_exc
            else:
                raise
        dt = time.perf_counter() - t0
        logger.info("[export-ifc-v2] IFC written in %.2fs", dt)

        # Validate IFC parseability (non-blocking, with timeout)
        try:
            def _test_parse() -> None:
                try:
                    test_model = ifc_io.open(str(out_path))
                    close_fn = getattr(test_model, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as e:
                    logger.warning("[export-ifc-v2] IFC file could not be parsed: %s", e)
                    warnings.append(f"IFC-Datei konnte nicht geparst werden: {e}")
            
            # Add timeout to prevent hang (30 seconds max)
            await asyncio.wait_for(asyncio.to_thread(_test_parse), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("[export-ifc-v2] IFC parseability check timed out (non-critical)")
            warnings.append("IFC-Datei Parse-Check timeout (Datei wurde trotzdem erstellt)")
        except Exception as parse_exc:
            logger.warning("[export-ifc-v2] Error during IFC parseability check: %s", parse_exc)
            warnings.append(f"IFC-Datei Parse-Check Fehler: {parse_exc}")

        # Generate validation report (non-blocking, with timeout)
        t_validation_report_start = time.perf_counter()
        try:
            def _generate_validation() -> None:
                try:
                    validation_report = generate_validation_report(normalized, wall_axes, out_path, auto_repair=True, config=config)
                    write_validation_report(validation_report, validation_path)
                except Exception as e:
                    logger.warning("[export-ifc-v2] Validation report generation failed: %s", e)
            
            # Add timeout to prevent hang (60 seconds max)
            await asyncio.wait_for(asyncio.to_thread(_generate_validation), timeout=60.0)
            validation_report_url = f"/files/{validation_path.name}"
            metrics.time_validation = time.perf_counter() - t_validation_report_start
        except asyncio.TimeoutError:
            logger.warning("[export-ifc-v2] Validation report generation timed out (non-critical)")
            metrics.time_validation = time.perf_counter() - t_validation_report_start
        except Exception as validation_exc:
            logger.warning("[export-ifc-v2] Validation report generation failed: %s", validation_exc)
            metrics.time_validation = time.perf_counter() - t_validation_report_start

        # Generate topview (only after successful IFC export, non-blocking, with timeout)
        topview_url: str | None = None
        try:
            def _generate_topview() -> None:
                try:
                    topview_path = out_path.with_name(f"{out_path.stem}_topview.geojson")
                    build_topview_geojson(out_path, topview_path, section_elevation_mm=None)
                    return topview_path
                except Exception as e:
                    logger.warning("[export-ifc-v2] TopView generation failed: %s", e)
                    return None
            
            # Add timeout to prevent hang (30 seconds max)
            topview_path_result = await asyncio.wait_for(asyncio.to_thread(_generate_topview), timeout=30.0)
            if topview_path_result:
                topview_url = f"/files/{topview_path_result.name}"
                logger.info("[export-ifc-v2] TopView generated at %s", topview_path_result)
        except asyncio.TimeoutError:
            logger.warning("[export-ifc-v2] TopView generation timed out (non-critical)")
        except Exception as topview_exc:
            logger.warning("[export-ifc-v2] TopView generation failed: %s", topview_exc)
        
        # Write viewer HTML only after successful IFC export
        if viewer_html is not None:
            try:
                viewer_path = export_dir / f"{out_path.stem}_viewer.html"
                viewer_path.write_text(viewer_html, encoding="utf-8")
                viewer_url = f"/files/{viewer_path.name}"
                logger.info("[export-ifc-v2] Viewer written to %s", viewer_path)
            except Exception as viewer_write_exc:
                logger.warning("[export-ifc-v2] Viewer file write failed: %s", viewer_write_exc)

        try:
            # Try to find template IFC file for comparison (optional feature)
            # Only proceed if template exists - don't fail if it doesn't
            template_candidates = [
                Path("example ifc.ifc"),
                Path("examples") / "example ifc.ifc",
                Path("examples/example ifc.ifc"),  # Alternative path format
            ]
            template_path = next((p for p in template_candidates if p.exists() and p.is_file()), None)
            if template_path is not None:
                comparison = compare_ifc_to_template(template_path, out_path)
                comparison_path = out_path.with_name(f"{out_path.stem}_comparison.json")
                comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
                comparison_report_url = f"/files/{comparison_path.name}"
                if comparison.get("missing_property_sets") or comparison.get("extra_property_sets"):
                    warnings.append("Template-Vergleich weist Unterschiede bei Property Sets auf – siehe Comparison Report")
        except Exception as comparison_exc:
            logger.warning("[export-ifc-v2] Template comparison failed: %s", comparison_exc)

    except Exception as exc:
        logger.exception("[export-ifc-v2] Error during IFC export")
        metrics.add_error(str(exc))
        metrics.time_total = time.perf_counter() - t_postprocess_start
        raise IFCExportError(f"IFC Export V2 fehlgeschlagen: {exc}") from exc
    
    # Calculate total time
    metrics.time_total = time.perf_counter() - t_postprocess_start
    
    # Log metrics summary
    summary = metrics.get_summary()
    logger.info(
        "[export-ifc-v2] Pipeline completed: quality=%.2f, time=%.2fs, warnings=%d, errors=%d",
        summary["quality_score"],
        summary["total_time_seconds"],
        summary["total_warnings"],
        summary["total_errors"],
    )

    return ExportIFCV2Response(
        ifc_url=f"/files/{file_name}",
        file_name=file_name,
        viewer_url=viewer_url,
        topview_url=topview_url,
        validation_report_url=validation_report_url,
        storey_height_mm=payload.storey_height_mm,
        door_height_mm=payload.door_height_mm,
        window_height_mm=window_height_mm,
        window_head_elevation_mm=window_head_elevation_mm,
        px_per_mm=px_per_mm,
        comparison_report_url=comparison_report_url,
        warnings=warnings if warnings else None,
    )

