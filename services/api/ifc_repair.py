from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, shape

from core.ml.postprocess_floorplan import NormalizedDet, estimate_wall_axes_and_thickness
from core.reconstruct.spaces import polygonize_spaces_from_walls
from core.settings import get_settings
from core.vector.ifc_topview import build_topview_geojson
from services.api.ifc_exporter import EXPORT_ROOT
from core.ifc.build_ifc43_model import write_ifc_with_spaces


logger = logging.getLogger(__name__)


class RepairError(RuntimeError):
    """Raised when the repair pipeline cannot proceed."""


def run_ifc_repair(
    source_path: Path,
    *,
    level: int = 1,
    export_root: Path | None = None,
) -> Tuple[Path, List[str]]:
    """Render a repaired IFC using geometry extracted from the source file."""

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
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Konnte TopView-Geometrie nicht extrahieren")
        warnings.append(f"TopView-Extraktion fehlgeschlagen: {type(exc).__name__}: {exc}")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    walls = [det for det in normalized if det.type == "WALL"]
    if not walls:
        warnings.append("Keine Wände gefunden – Originaldatei wird kopiert")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    spaces = polygonize_spaces_from_walls(walls)
    axes = estimate_wall_axes_and_thickness(normalized)

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
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Konnte reparierte IFC nicht schreiben")
        warnings.append(f"Reparatur fehlgeschlagen – Originaldatei kopiert ({type(exc).__name__}: {exc})")
        shutil.copyfile(source_path, repaired_path)
        return repaired_path, warnings

    return repaired_path, warnings


def _extract_normalized_geometry(source_path: Path) -> List[NormalizedDet]:
    with tempfile.NamedTemporaryFile("w+", suffix="_repair.geojson", delete=False, dir=source_path.parent) as tmp:
        temp_geojson_path = Path(tmp.name)

    try:
        build_topview_geojson(source_path, temp_geojson_path)
        data = json.loads(temp_geojson_path.read_text(encoding="utf-8"))
    finally:
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
        raise RepairError("Keine verwertbare Geometrie gefunden")

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
