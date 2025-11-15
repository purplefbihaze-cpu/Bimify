from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, linemerge, nearest_points
from shapely.geometry.base import BaseGeometry
from shapely.geometry import MultiPolygon


@dataclass
class LineGraph:
    """Lightweight container for snapped linework."""
    lines: List[LineString]


def _angle_deg_of(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx, dy = float(x2 - x1), float(y2 - y1)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    return ang


def cluster_orientations(axes: Sequence[LineString]) -> List[float]:
    """
    Return dominant base angles (degrees) for the given axes.
    Minimal, robust heuristic: prefer 0 and 90 when present;
    include 45 when many segments are near 45.
    """
    if not axes:
        return [0.0, 90.0]
    counts = {0: 0, 45: 0, 90: 0}
    for line in axes:
        a = _angle_deg_of(line)
        da0 = min(abs(a - 0.0), abs(a - 180.0))
        da90 = abs(a - 90.0)
        da45 = min(abs(a - 45.0), abs(a - 135.0))
        if da0 <= da90 and da0 <= da45:
            counts[0] += 1
        elif da90 <= da45:
            counts[90] += 1
        else:
            counts[45] += 1
    bases: List[float] = []
    if counts[0] > 0:
        bases.append(0.0)
    if counts[90] > 0:
        bases.append(90.0)
    if counts[45] > max(0, len(axes) // 6):  # only add 45° if it’s not noise
        bases.append(45.0)
    if not bases:
        bases = [0.0, 90.0]
    return bases


def _unit_from_angle(angle_deg: float) -> Tuple[float, float]:
    a = math.radians(angle_deg % 180.0)
    return math.cos(a), math.sin(a)


def snap_axis_orientation(
    line: LineString,
    bases_deg: Sequence[float],
    angle_tolerance_deg: float,
) -> LineString:
    """
    If line orientation is within tolerance to any base direction,
    rotate the segment to align while preserving center and length.
    """
    if len(list(line.coords)) < 2:
        return line
    ang = _angle_deg_of(line)
    best: Tuple[float, float] | None = None  # (delta, base)
    for base in bases_deg:
        delta = min(abs(ang - base), abs(ang - (base + 180.0)))
        if best is None or delta < best[0]:
            best = (delta, base)
    if best is None or best[0] > angle_tolerance_deg:
        return line

    base = best[1]
    ux, uy = _unit_from_angle(base)
    coords = list(line.coords)
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    length = math.hypot(x2 - x1, y2 - y1)
    half = length / 2.0
    snapped = LineString([(cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)])
    return snapped


def build_line_graph(axes: Sequence[LineString], merge_tolerance_mm: float) -> LineGraph:
    """
    Build a simple line graph; for now, just merge nearly collinear overlaps
    using unary_union + linemerge to reduce duplicates.
    """
    if not axes:
        return LineGraph(lines=[])
    merged = linemerge(unary_union(list(axes)))
    if isinstance(merged, LineString):
        lines = [merged]
    else:
        lines = list(merged.geoms)
    return LineGraph(lines=lines)


def snap_endpoints_to_envelope(graph: LineGraph, envelope: Polygon, dist_tol_mm: float) -> LineGraph:
    """
    Move endpoints that are within dist_tol_mm to the nearest point on the
    envelope boundary. Keeps line directions and lengths where possible.
    """
    boundary = envelope.boundary
    snapped: List[LineString] = []
    for line in graph.lines:
        coords = list(line.coords)
        if len(coords) < 2:
            snapped.append(line)
            continue
        p1 = Point(coords[0]); p2 = Point(coords[-1])
        np1a, np1b = nearest_points(p1, boundary)
        np2a, np2b = nearest_points(p2, boundary)
        c1 = p1 if p1.distance(np1b) > dist_tol_mm else np1b
        c2 = p2 if p2.distance(np2b) > dist_tol_mm else np2b
        snapped.append(LineString([(c1.x, c1.y), (c2.x, c2.y)]))
    return LineGraph(lines=snapped)


def merge_colinear(graph: LineGraph) -> LineGraph:
    """Merge overlapping colinear lines via unary_union + linemerge."""
    if not graph.lines:
        return graph
    merged = linemerge(unary_union(graph.lines))
    if isinstance(merged, LineString):
        return LineGraph([merged])
    return LineGraph(list(merged.geoms))


def closed_outer_hull(wall_polygons: Sequence[Polygon | MultiPolygon], epsilon_mm: float = 10.0) -> Polygon:
    """
    Compute a robust outer hull around wall polygons using a morphological closing:
    buffer(+eps) then buffer(-eps). Returns a Polygon (outer shell).
    """
    polys: list[Polygon] = []
    for geom in wall_polygons:
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(list(geom.geoms))
    if not polys:
        return Polygon()
    unioned = unary_union(polys)
    closed = unioned.buffer(max(epsilon_mm, 0.0)).buffer(-max(epsilon_mm, 0.0))
    if isinstance(closed, Polygon):
        return closed
    # Fallback to envelope when topology becomes multi-part
    try:
        envelope = unioned.envelope
        if isinstance(envelope, Polygon):
            return envelope
    except Exception:
        pass
    # return the largest polygon
    try:
        return max(list(closed.geoms), key=lambda g: g.area)  # type: ignore[attr-defined]
    except Exception:
        return Polygon()


def extend_trim_to_intersections(graph: LineGraph, snap_dist_mm: float = 10.0) -> LineGraph:
    """
    Very lightweight trim/extend: endpoints that are within snap_dist_mm are
    collapsed to their midpoint to close tiny gaps; returns updated lines.
    """
    if not graph.lines:
        return graph
    # Collect endpoints
    endpoints: List[Point] = []
    for ln in graph.lines:
        coords = list(ln.coords)
        if len(coords) >= 2:
            endpoints.append(Point(coords[0]))
            endpoints.append(Point(coords[-1]))
    # Compute clusters by simple O(n^2) since n is small
    clusters: List[List[int]] = []
    taken = set()
    for i, pi in enumerate(endpoints):
        if i in taken:
            continue
        cluster = [i]
        taken.add(i)
        for j in range(i + 1, len(endpoints)):
            if j in taken:
                continue
            if pi.distance(endpoints[j]) <= snap_dist_mm:
                cluster.append(j)
                taken.add(j)
        clusters.append(cluster)
    # Map old endpoints to cluster midpoint
    replacement: dict[int, Point] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            idx = cluster[0]
            replacement[idx] = endpoints[idx]
        else:
            xs = [endpoints[k].x for k in cluster]
            ys = [endpoints[k].y for k in cluster]
            mid = Point(sum(xs) / len(xs), sum(ys) / len(ys))
            for k in cluster:
                replacement[k] = mid
    # Rebuild lines with snapped endpoints
    result: List[LineString] = []
    k = 0
    for ln in graph.lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            result.append(ln)
            continue
        p_start = replacement.get(k, Point(coords[0])); k += 1
        p_end = replacement.get(k, Point(coords[-1])); k += 1
        result.append(LineString([(p_start.x, p_start.y), (p_end.x, p_end.y)]))
    return LineGraph(result)


