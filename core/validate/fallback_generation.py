from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from core.geometry.contract import (
    FALLBACK_WALL_THICKNESS,
    MIN_ROOM_AREA_FALLBACK,
    MIN_POLYGON_AREA,
    MIN_OPENING_DEPTH,
    mm,
)
from core.ml.postprocess_floorplan import NormalizedDet
from core.reconstruct.spaces import SpacePoly


@dataclass
class FallbackNotes:
    created_walls: int = 0
    created_spaces: int = 0
    used_default_extent: bool = False


def _cw(poly: Polygon) -> Polygon:
    try:
        return orient(poly, sign=-1.0)
    except Exception:
        return poly


def _default_bounds_mm() -> Tuple[float, float, float, float]:
    # 10m x 8m at origin (in mm)
    return (0.0, 0.0, 10000.0, 8000.0)


def _bounds_from_norm(normalized: Sequence[NormalizedDet]) -> Tuple[float, float, float, float] | None:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    found = False
    for det in normalized:
        geom = getattr(det, "geom", None)
        if geom is None or getattr(geom, "is_empty", True):
            continue
        try:
            bx0, by0, bx1, by1 = geom.bounds
            minx = min(minx, bx0)
            miny = min(miny, by0)
            maxx = max(maxx, bx1)
            maxy = max(maxy, by1)
            found = True
        except Exception:
            continue
    if not found:
        return None
    # sanity clamp
    if maxx - minx < 100.0 or maxy - miny < 100.0:
        return None
    return (float(minx), float(miny), float(maxx), float(maxy))


def _make_strip(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return _cw(Polygon(coords))


def _synthesize_perimeter_walls(
    bounds: Tuple[float, float, float, float],
    thickness_m: float,
    *,
    doc: int = 0,
    page: int = 0,
) -> List[NormalizedDet]:
    minx, miny, maxx, maxy = bounds
    t = mm(thickness_m)
    width = max(1.0, maxx - minx)
    height = max(1.0, maxy - miny)
    t = max(50.0, min(t, width * 0.25, height * 0.25))  # clamp thickness

    # Four strips along the rectangle edges (inside the bounds)
    bottom = _make_strip(minx, miny, maxx, miny + t)
    top = _make_strip(minx, maxy - t, maxx, maxy)
    left = _make_strip(minx, miny + t, minx + t, maxy - t)
    right = _make_strip(maxx - t, miny + t, maxx, maxy - t)

    walls: List[NormalizedDet] = []
    for idx, poly in enumerate((bottom, top, left, right), start=1):
        walls.append(
            NormalizedDet(
                doc=doc,
                page=page,
                type="WALL",
                is_external=True,
                geom=poly,
                attrs={
                    "confidence": 1.0,
                    "geometry_source": "fallback",
                    "fallback_perimeter": True,
                    "index": idx,
                },
            )
        )
    return walls


def _synthesize_single_space(
    bounds: Tuple[float, float, float, float],
    thickness_m: float,
) -> SpacePoly | None:
    minx, miny, maxx, maxy = bounds
    t = mm(thickness_m)
    inner = (
        minx + t,
        miny + t,
        maxx - t,
        maxy - t,
    )
    if inner[2] - inner[0] <= 100.0 or inner[3] - inner[1] <= 100.0:
        # too small, use full bounds as space as last resort
        inner = bounds
    poly = _cw(Polygon([(inner[0], inner[1]), (inner[2], inner[1]), (inner[2], inner[3]), (inner[0], inner[3])]))
    if poly.is_empty or (hasattr(poly, "area") and poly.area / 1_000_000.0 < MIN_POLYGON_AREA):
        return None
    return SpacePoly(polygon=poly, area_m2=float(poly.area / 1_000_000.0))


def ensure_minimum_geometry(
    normalized: Sequence[NormalizedDet],
    spaces: Sequence[SpacePoly],
    *,
    doc: int = 0,
    page: int = 0,
) -> Tuple[List[NormalizedDet], List[SpacePoly], FallbackNotes]:
    """Ensure there is at least one wall and one space by synthesizing fallbacks.

    Returns updated lists and notes about what was created.
    """
    notes = FallbackNotes()

    # Determine bounds
    bounds = _bounds_from_norm(normalized)
    if bounds is None:
        bounds = _default_bounds_mm()
        notes.used_default_extent = True

    out_norm = list(normalized)
    out_spaces = list(spaces)

    # Create walls if none
    if not any(det.type == "WALL" for det in out_norm):
        walls = _synthesize_perimeter_walls(bounds, FALLBACK_WALL_THICKNESS, doc=doc, page=page)
        out_norm.extend(walls)
        notes.created_walls = len(walls)

    # Create a space if none
    if not out_spaces:
        space = _synthesize_single_space(bounds, FALLBACK_WALL_THICKNESS)
        if space is not None:
            out_spaces.append(space)
            notes.created_spaces = 1

    return out_norm, out_spaces, notes
