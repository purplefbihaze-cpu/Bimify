from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, shape, mapping

from core.ml.postprocess_floorplan import NormalizedDet, estimate_wall_axes_and_thickness
from core.reconstruct.spaces import polygonize_spaces_from_walls
from core.settings import get_settings
from core.vector.ifc_topview import build_topview_geojson
from services.api.ifc_exporter import EXPORT_ROOT
from core.ifc.build_ifc43_model import write_ifc_with_spaces, collect_wall_polygons, snap_thickness_mm
from core.reconstruct.openings import snap_openings_to_walls, reproject_openings_to_snapped_axes
from core.vector.snap import cluster_orientations, snap_axis_orientation, build_line_graph, extend_trim_to_intersections, merge_colinear, closed_outer_hull
from core.validate.reconstruction_validation import generate_validation_report, write_validation_report
from core.exceptions import (
    RepairError,
    GeometryExtractionError,
    GeometryProcessingError,
    TopViewError,
    IFCExportError,
)


logger = logging.getLogger(__name__)


def build_preview_axes(
    source_path: Path,
    *,
    px_per_mm: float | None = None,
    image_bgr: "numpy.ndarray | None" = None,  # type: ignore[name-defined]
    rf_norm: Sequence[NormalizedDet] | None = None,
) -> tuple[list[NormalizedDet], list]:
    """
    Extract normalized geometry and estimate axes; optionally refine against image edges and RF shapes.
    """
    try:
        normalized = _extract_normalized_geometry(source_path)
    except (GeometryExtractionError, TopViewError) as exc:
        logger.error("Preview: _extract_normalized_geometry fehlgeschlagen für %s: %s", source_path, exc)
        raise
    except Exception as exc:
        logger.error("Preview: Unerwarteter Fehler bei _extract_normalized_geometry für %s: %s", source_path, exc)
        raise GeometryExtractionError(f"Geometrie-Extraktion fehlgeschlagen: {exc}", {"source_path": str(source_path)}) from exc
    if not normalized:
        raise RepairError(f"Keine Geometrie aus IFC-Datei extrahiert: {source_path.name}")
    
    try:
        axes = estimate_wall_axes_and_thickness(normalized)
    except Exception as exc:
        logger.error("Preview: estimate_wall_axes_and_thickness fehlgeschlagen: %s", exc)
        raise GeometryProcessingError(f"Achsen-Schätzung fehlgeschlagen: {exc}") from exc
    
    try:
        if image_bgr is not None:
            from core.vision.edge_detection import detect_edges_and_lines
            from core.reconstruct.final_fit import refine_axes_using_edges_and_rf

            _, image_lines = detect_edges_and_lines(image_bgr, px_per_mm=px_per_mm)
            axes = refine_axes_using_edges_and_rf(axes, image_lines=image_lines, rf_norm=rf_norm or [])
    except Exception:
        # Best effort refinement; ignore failures in preview
        logger.debug("Preview: Bild-Verfeinerung übersprungen (optional)")
        pass
    return normalized, axes

def run_ifc_repair(
    source_path: Path,
    *,
    level: int = 1,
    export_root: Path | None = None,
) -> Tuple[Path, List[str]]:
    """Repair IFC file by extracting geometry and regenerating with improved topology.
    
    Args:
        source_path: Path to source IFC file.
        level: Repair level (currently only level 1 is supported).
        export_root: Optional root directory for exports. Defaults to data/exports.
    
    Returns:
        Tuple of (repaired_path, warnings_list).
    
    Raises:
        RepairError: If repair level is not supported.
        FileNotFoundError: If source file does not exist.
        GeometryExtractionError: If geometry extraction fails.
    """

    if level != 1:
        raise RepairError(f"Reparatur-Level {level} wird noch nicht unterstützt")

    if not source_path.exists():
        raise FileNotFoundError(f"IFC-Datei nicht gefunden: {source_path}")

    repair_dir = (export_root or EXPORT_ROOT).resolve()
    repair_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = repair_dir / f"{source_path.stem}_repair_L{level}.ifc"

    settings = get_settings()
    wall_thickness_standards = getattr(getattr(settings, "ifc", None), "wall_thickness_standards_mm", None)

    warnings: List[str] = []

    try:
        normalized = _extract_normalized_geometry(source_path)
    except (GeometryExtractionError, TopViewError) as exc:
        logger.exception("Konnte TopView-Geometrie nicht extrahieren")
        warnings.append(f"TopView-Extraktion fehlgeschlagen: {type(exc).__name__}: {exc}")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unerwarteter Fehler bei TopView-Extraktion")
        warnings.append(f"TopView-Extraktion fehlgeschlagen: {type(exc).__name__}: {exc}")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    walls = [det for det in normalized if det.type == "WALL"]
    if not walls:
        warnings.append("Keine Wände gefunden – Originaldatei wird kopiert")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    # Spaces (initial – will be recomputed implicitly from result as well)
    spaces = polygonize_spaces_from_walls(walls)
    # Filter Mini-Inseln
    try:
        spaces = [sp for sp in spaces if float(getattr(sp, "area_m2", 0.0)) >= 0.05]
    except Exception:
        pass

    # Estimate wall axes and thickness
    axes = estimate_wall_axes_and_thickness(normalized)
    # Snap thickness to standards for stability
    try:
        for ax in axes:
            try:
                is_external = getattr(getattr(ax, "detection", None), "is_external", None)
            except Exception:
                is_external = None
            ax.width_mm = float(
                snap_thickness_mm(
                    getattr(ax, "width_mm", None),
                    standards=tuple(wall_thickness_standards or ()) if wall_thickness_standards else (),
                    is_external=is_external,
                )
            )
    except Exception:
        pass

    # Optional: lightweight orientation snapping of axes (±6°) to stabilize topology
    try:
        base_angles = cluster_orientations([ax.axis for ax in axes if getattr(ax, "axis", None) is not None])
        snapped_axes = []
        for ax in axes:
            geom = getattr(ax, "axis", None)
            if geom is None:
                snapped_axes.append(None)
                continue
            snapped = snap_axis_orientation(geom, base_angles, angle_tolerance_deg=6.0)
            try:
                ax.axis = snapped  # mutate in place
            except Exception:
                pass
            snapped_axes.append(snapped)

        # Build and gently repair linework (collapse tiny endpoint gaps)
        graph = build_line_graph([ln for ln in snapped_axes if ln is not None], merge_tolerance_mm=10.0)
        graph = extend_trim_to_intersections(graph, snap_dist_mm=10.0)
        graph = merge_colinear(graph)

        # Compute robust outer hull for diagnostics and potential clipping
        wall_polygons = collect_wall_polygons(normalized)
        hull = closed_outer_hull(list(wall_polygons.values()), epsilon_mm=10.0)
    except Exception:
        wall_polygons = collect_wall_polygons(normalized)
        hull = None

    # Openings: snap to axes and reproject so depth == wall thickness (bündig)
    try:
        assignments, wall_polys = snap_openings_to_walls(normalized, wall_axes=axes, wall_polygons_override=list(wall_polygons.values()))
        reproject_openings_to_snapped_axes(
            normalized, 
            assignments, 
            wall_axes=axes, 
            depth_equals_wall_thickness=True,
            wall_polygons=wall_polys
        )
    except Exception as exc:
        logger.warning("Konnte Öffnungen nicht neu ausrichten: %s", exc)

    try:
        write_ifc_with_spaces(
            normalized=normalized,
            spaces=spaces,
            out_path=repaired_path,
            project_name="BIMify Repair",
            storey_name="EG",
            storey_elevation=0.0,
            wall_axes=axes,
            storey_height_mm=3000.0,
            door_height_mm=2100.0,
            window_height_mm=1000.0,
            window_head_elevation_mm=2000.0,
            px_per_mm=1.0,
            calibration=None,
            schema_version=getattr(getattr(settings, "ifc", None), "schema", "IFC4"),
            wall_thickness_standards_mm=wall_thickness_standards,
        )

        # Debug layers
        import os
        debug_snap = str(os.getenv("DEBUG_SNAP", "")).lower() in {"1", "true", "yes"}
        if debug_snap:
            try:
                debug_dir = repaired_path.parent
                # Hull
                if hull is not None and not hull.is_empty:
                    (debug_dir / f"{repaired_path.stem}_snap_hull.wkt").write_text(hull.wkt, encoding="utf-8")
                # Axes
                axes_geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"type": "WALL_AXIS", "width_mm": getattr(ax, "width_mm", None)},
                            "geometry": None if getattr(ax, "axis", None) is None else mapping(ax.axis),
                        }
                        for ax in axes
                    ],
                }
                (debug_dir / f"{repaired_path.stem}_snap_axes.geojson").write_text(json.dumps(axes_geojson), encoding="utf-8")
            except Exception:
                pass
        # Validation report
        try:
            validation_report = generate_validation_report(normalized, axes, repaired_path)
            report_path = repaired_path.with_name(f"{repaired_path.stem}_validation.json")
            write_validation_report(validation_report, report_path)
        except Exception:
            pass
    except (IFCExportError, GeometryProcessingError) as exc:
        logger.exception("Konnte reparierte IFC nicht schreiben")
        warnings.append(f"Reparatur fehlgeschlagen – Originaldatei kopiert ({type(exc).__name__}: {exc})")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unerwarteter Fehler beim Schreiben der reparierten IFC")
        warnings.append(f"Reparatur fehlgeschlagen – Originaldatei kopiert ({type(exc).__name__}: {exc})")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    return repaired_path, warnings


def _extract_normalized_geometry(source_path: Path) -> List[NormalizedDet]:
    # Check if topview already exists and is up-to-date
    existing_topview = source_path.with_name(f"{source_path.stem}_topview.geojson")
    use_existing = False
    if existing_topview.exists():
        try:
            # Check if topview is newer than source file
            if existing_topview.stat().st_mtime >= source_path.stat().st_mtime:
                use_existing = True
                logger.debug("Preview: Verwende vorhandene TopView-Datei: %s", existing_topview.name)
        except OSError:
            pass
    
    if use_existing:
        temp_geojson_path = existing_topview
    else:
        with tempfile.NamedTemporaryFile("w+", suffix="_repair.geojson", delete=False, dir=source_path.parent) as tmp:
            temp_geojson_path = Path(tmp.name)
        try:
            logger.info("Preview: Erstelle TopView für %s (kann bei großen Dateien länger dauern)", source_path.name)
            build_topview_geojson(source_path, temp_geojson_path)
        except Exception as exc:
            temp_geojson_path.unlink(missing_ok=True)
            raise TopViewError(f"TopView-Erstellung fehlgeschlagen: {exc}", {"source_path": str(source_path)}) from exc
    
    try:
        data = json.loads(temp_geojson_path.read_text(encoding="utf-8"))
    finally:
        # Only delete if it was a temporary file
        if not use_existing:
            temp_geojson_path.unlink(missing_ok=True)

    features = data.get("features", [])
    normalized: List[NormalizedDet] = []

    for feature in features:
        props = feature.get("properties", {}) or {}
        label = str(props.get("type", "")).strip().upper() or "WALL"
        geometry = feature.get("geometry") or {}

        try:
            geom = shape(geometry)
        except Exception:
            continue

        geom = _coerce_geometry(geom)
        if geom is None or geom.is_empty:
            continue

        normalized.append(
            NormalizedDet(
                doc=0,
                page=0,
                type=label,
                is_external=None,
                geom=geom,
                attrs={
                    "source": "repair",
                    "productId": props.get("productId"),
                    "guid": props.get("guid"),
                },
            )
        )

    if not normalized:
        raise GeometryExtractionError("Keine verwertbare Geometrie gefunden", {"source_path": str(source_path)})

    return normalized


def _coerce_geometry(geom) -> Polygon | LineString | None:
    if geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom.buffer(0)

    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area, default=None)
        return largest.buffer(0) if largest is not None else None

    if isinstance(geom, LineString):
        return geom

    if isinstance(geom, MultiLineString):
        longest = max(geom.geoms, key=lambda g: g.length, default=None)
        return longest

    if isinstance(geom, GeometryCollection):
        candidates: Iterable = [
            _coerce_geometry(item)
            for item in geom.geoms
            if isinstance(item, (Polygon, MultiPolygon, LineString, MultiLineString))
        ]
        first_valid = next((item for item in candidates if item is not None), None)
        return first_valid

    return None
