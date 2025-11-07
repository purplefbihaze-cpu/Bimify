from __future__ import annotations

from typing import Iterable, List

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import snap, unary_union


def merge_polygons(
    polygons: Iterable[Polygon],
    *,
    tolerance: float,
    snap_tolerance: float,
) -> List[Polygon]:
    """Merge polygons with optional smoothing and snapping.

    Parameters
    ----------
    polygons:
        Iterable of polygon geometries to merge.
    tolerance:
        Simplification tolerance in millimetres. Values <= 0 disable simplification.
    snap_tolerance:
        Distance tolerance for snapping vertices after merging. Values <= 0 disable snapping.
    """

    cleaned: List[Polygon] = []
    for poly in polygons:
        if poly.is_empty:
            continue
        candidate = poly
        if tolerance > 0:
            try:
                candidate = candidate.simplify(tolerance, preserve_topology=True)
            except Exception:
                # If simplify fails keep original polygon
                candidate = poly
        candidate = candidate.buffer(0)
        if isinstance(candidate, Polygon):
            cleaned.append(candidate)
        elif isinstance(candidate, MultiPolygon):
            cleaned.extend(list(candidate.geoms))

    if not cleaned:
        return []

    merged = unary_union(cleaned)
    if snap_tolerance > 0:
        try:
            merged = snap(merged, merged, snap_tolerance)
        except Exception:
            pass

    if isinstance(merged, Polygon):
        return [merged]
    if isinstance(merged, MultiPolygon):
        return [poly for poly in merged.geoms]

    return []

