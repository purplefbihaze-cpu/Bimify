from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence

import ifcopenshell
import ifcopenshell.util.placement as placement_util
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


logger = logging.getLogger(__name__)


TARGET_TYPES = {
    "IfcWall": "WALL",
    "IfcDoor": "DOOR",
    "IfcWindow": "WINDOW",
}


def build_topview_geojson(
    ifc_path: Path,
    output_path: Path,
    *,
    section_elevation_mm: float | None = None,
) -> Path:
    """Generate a GeoJSON top view (plan) for walls, doors and windows."""

    if not ifc_path.exists():
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")

    model = ifcopenshell.open(ifc_path)
    features = []

    for ifc_class, label in TARGET_TYPES.items():
        try:
            elements = model.by_type(ifc_class)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Fehler beim Lesen von IFC-Elementen (%s)", ifc_class)
            continue

        for element in elements:
            geom = _extract_element_geometry(element, section_elevation_mm=section_elevation_mm)
            if geom is None or geom.is_empty:
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    continue

            if geom.geom_type == "MultiPolygon":
                parts = list(geom.geoms)
            else:
                parts = [geom]

            merged = unary_union(parts)
            if merged.geom_type == "GeometryCollection":
                polys = [g for g in merged.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
                merged = unary_union(polys) if polys else Polygon()
            if not merged.is_valid:
                merged = merged.buffer(0)
            if merged.is_empty:
                continue

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": label,
                        "productId": int(element.id()),
                        "guid": str(getattr(element, "GlobalId", "")),
                    },
                    "geometry": _to_geojson_geometry(merged),
                }
            )

    collection = {"type": "FeatureCollection", "features": features}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    return output_path


def _extract_element_geometry(element: ifcopenshell.entity_instance, *, section_elevation_mm: float | None) -> BaseGeometry | None:
    placement = getattr(element, "ObjectPlacement", None)
    try:
        base_matrix = placement_util.get_local_placement(placement) if placement else np.identity(4)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Konnte Placement nicht für %s berechnen", element)
        base_matrix = np.identity(4)

    shape = getattr(element, "Representation", None)
    if not shape:
        return None

    solids: List[BaseGeometry] = []

    representations = getattr(shape, "Representations", []) or []
    for rep in representations:
        items = getattr(rep, "Items", []) or []
        for item in items:
            geom = None
            if item.is_a("IfcExtrudedAreaSolid"):
                geom = _geometry_from_extruded(item, base_matrix, section_elevation_mm=section_elevation_mm)
            if geom is not None and not geom.is_empty:
                solids.append(geom)

    if not solids:
        return None

    merged: BaseGeometry = unary_union(solids)
    return merged


def _geometry_from_extruded(extruded: ifcopenshell.entity_instance, base_matrix: np.ndarray, *, section_elevation_mm: float | None = None) -> BaseGeometry | None:
    profile = getattr(extruded, "SweptArea", None)
    if profile is None:
        return None

    local_points = _profile_points(profile)
    if not local_points:
        return None

    try:
        profile_matrix = placement_util.get_axis2placement(extruded.Position)
    except Exception:  # pragma: no cover - defensive
        profile_matrix = np.identity(4)

    transform = base_matrix @ profile_matrix

    # Section filter: include if cut elevation intersects the extrusion range
    # Changed from < to <= to include elements that intersect the cut plane at the upper edge
    if section_elevation_mm is not None:
        try:
            origin_z = _apply_matrix((0.0, 0.0, 0.0), transform)[2]
            depth = float(getattr(extruded, "Depth", 0.0))
            if not (origin_z <= section_elevation_mm <= origin_z + depth):
                return None
        except Exception:
            pass

    transformed = [_apply_matrix(point, transform) for point in local_points]

    coords_2d = [(x, y) for x, y, _ in transformed]
    polygon = Polygon(coords_2d)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _profile_points(profile: ifcopenshell.entity_instance) -> List[Sequence[float]]:
    if profile.is_a("IfcRectangleProfileDef"):
        x_dim = float(getattr(profile, "XDim", 0.0))
        y_dim = float(getattr(profile, "YDim", 0.0))
        if x_dim <= 0 or y_dim <= 0:
            return []
        hx = x_dim / 2.0
        hy = y_dim / 2.0
        return [
            (-hx, -hy, 0.0),
            (hx, -hy, 0.0),
            (hx, hy, 0.0),
            (-hx, hy, 0.0),
            (-hx, -hy, 0.0),
        ]

    if profile.is_a("IfcArbitraryClosedProfileDef"):
        curve = getattr(profile, "OuterCurve", None)
        if curve and curve.is_a("IfcPolyline"):
            points = []
            for pt in curve.Points or []:
                coords = list(pt.Coordinates)
                while len(coords) < 3:
                    coords.append(0.0)
                points.append((float(coords[0]), float(coords[1]), float(coords[2])))
            if len(points) >= 2 and points[0] != points[-1]:
                points.append(points[0])
            return points
    return []


def _apply_matrix(point: Sequence[float], matrix: np.ndarray) -> tuple[float, float, float]:
    if len(point) == 2:
        x, y = point
        z = 0.0
    else:
        x, y, z = point[:3]
    vec = np.array([x, y, z, 1.0])
    result = matrix @ vec
    return float(result[0]), float(result[1]), float(result[2])


def _to_geojson_geometry(geometry: BaseGeometry) -> dict:
    if geometry.geom_type == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [[(float(x), float(y)) for x, y in geometry.exterior.coords]],
        }
    if geometry.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[(float(x), float(y)) for x, y in polygon.exterior.coords]] for polygon in geometry.geoms
            ],
        }
    # Fallback to empty geometry
    return {"type": "GeometryCollection", "geometries": []}


__all__ = ["build_topview_geojson"]

