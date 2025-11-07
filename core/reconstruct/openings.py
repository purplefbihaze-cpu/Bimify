from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis


@dataclass
class OpeningAssignment:
    opening: NormalizedDet
    wall_index: Optional[int]
    distance_mm: float
    axis_index: Optional[int] = None
    axis_distance_mm: Optional[float] = None
    angle_delta_deg: Optional[float] = None


def _line_orientation(line: LineString) -> float | None:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    angle = angle % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def _polygon_orientation(polygon: Polygon) -> float | None:
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 2:
        return None
    best_len = -1.0
    best_dir: Tuple[float, float] | None = None
    for idx in range(len(coords) - 1):
        x1, y1 = coords[idx]
        x2, y2 = coords[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_dir = (dx, dy)
    if not best_dir:
        return None
    return _line_orientation(LineString([(0.0, 0.0), best_dir]))


def _geometry_orientation(geom) -> float | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return _line_orientation(geom)
    if isinstance(geom, Polygon):
        return _polygon_orientation(geom)
    if isinstance(geom, MultiPolygon):
        try:
            largest = max(geom.geoms, key=lambda g: g.area)
        except ValueError:
            return None
        return _polygon_orientation(largest)
    return None


def _angle_difference_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def snap_openings_to_walls(
    dets: List[NormalizedDet],
    wall_axes: Sequence[WallAxis] | None = None,
    *,
    wall_polygons_override: Sequence[Polygon | MultiPolygon] | None = None,
    max_axis_distance_mm: float = 500.0,
    max_axis_angle_delta_deg: float = 30.0,
) -> Tuple[List[OpeningAssignment], List[Polygon]]:
    walls = [d for d in dets if d.type == "WALL"]
    openings = [d for d in dets if d.type in ("DOOR", "WINDOW")]
    wall_polys: List[Polygon] = []
    if wall_polygons_override is not None:
        for idx, wall in enumerate(walls):
            override_geom = None
            if idx < len(wall_polygons_override):
                override_geom = wall_polygons_override[idx]
            if override_geom is None:
                wall_polys.append(_ensure_polygon(wall.geom))
            else:
                wall_polys.append(_ensure_polygon(override_geom))
    else:
        wall_polys = [_ensure_polygon(w.geom) for w in walls]
    axes_by_wall: Dict[int, List[Tuple[int, WallAxis]]] = _organize_axes_by_wall(wall_axes)

    assignments: List[OpeningAssignment] = []
    for opening in openings:
        best_choice = _select_wall_for_opening(
            opening,
            wall_polys,
            axes_by_wall,
            max_axis_distance_mm=max_axis_distance_mm,
            max_axis_angle_deg=max_axis_angle_delta_deg,
        )
        assignments.append(best_choice)
    return assignments, wall_polys


def _organize_axes_by_wall(wall_axes: Sequence[WallAxis] | None) -> Dict[int, List[Tuple[int, WallAxis]]]:
    axes_by_wall: Dict[int, List[Tuple[int, WallAxis]]] = {}
    if not wall_axes:
        return axes_by_wall
    for axis_index, axis in enumerate(wall_axes):
        source_idx = getattr(axis, "source_index", None)
        if source_idx is None:
            continue
        axes_by_wall.setdefault(int(source_idx), []).append((axis_index, axis))
    return axes_by_wall


def _select_wall_for_opening(
    opening: NormalizedDet,
    wall_polys: Sequence[Polygon],
    axes_by_wall: Dict[int, List[Tuple[int, WallAxis]]],
    *,
    max_axis_distance_mm: float,
    max_axis_angle_deg: float,
) -> OpeningAssignment:
    best_assignment: OpeningAssignment | None = None
    best_score: Tuple[float, float, float] | None = None
    opening_geom = opening.geom
    opening_direction = _opening_direction_deg(opening_geom)

    for wall_index, wall_poly in enumerate(wall_polys):
        distance_mm = float(opening_geom.distance(wall_poly))

        axis_choice = _best_axis_candidate(
            opening_geom,
            axes_by_wall.get(wall_index, []),
            opening_direction,
            max_axis_distance_mm=max_axis_distance_mm,
            max_axis_angle_deg=max_axis_angle_deg,
        )

        if axis_choice is not None:
            axis_distance, angle_delta, axis_index = axis_choice
            score = (axis_distance, distance_mm, angle_delta or 90.0)
        else:
            score = (distance_mm + 1000.0, distance_mm, 90.0)
            axis_index = None
            axis_distance = None
            angle_delta = None

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = OpeningAssignment(
                opening=opening,
                wall_index=wall_index,
                distance_mm=distance_mm,
                axis_index=axis_index,
                axis_distance_mm=axis_distance,
                angle_delta_deg=angle_delta,
            )

    if best_assignment is None:
        return OpeningAssignment(opening=opening, wall_index=None, distance_mm=float("inf"))

    return best_assignment


def _best_axis_candidate(
    opening_geom: BaseGeometry,
    axes: Iterable[Tuple[int, WallAxis]],
    opening_direction: float | None,
    *,
    max_axis_distance_mm: float,
    max_axis_angle_deg: float,
) -> Tuple[float, float | None, int] | None:
    if not axes:
        return None

    centroid = _geometry_centroid(opening_geom)
    best: Tuple[float, float | None, int] | None = None

    for axis_index, axis in axes:
        axis_geom = axis.axis
        if not isinstance(axis_geom, LineString) or axis_geom.length <= 0.0:
            continue

        axis_distance = float(axis_geom.distance(centroid))
        if axis_distance > max_axis_distance_mm:
            continue

        axis_direction = _axis_direction_deg(axis_geom)
        angle_delta: float | None = None
        if opening_direction is not None and axis_direction is not None:
            angle_delta = _angular_delta_deg(opening_direction, axis_direction)
            if angle_delta > max_axis_angle_deg:
                continue

        candidate = (axis_distance, angle_delta, axis_index)
        if best is None or _compare_axis_candidates(candidate, best):
            best = candidate

    return best


def _compare_axis_candidates(a: Tuple[float, float | None, int], b: Tuple[float, float | None, int]) -> bool:
    a_angle = a[1] if a[1] is not None else 90.0
    b_angle = b[1] if b[1] is not None else 90.0
    return (a[0], a_angle) < (b[0], b_angle)


def _angular_delta_deg(a_deg: float, b_deg: float) -> float:
    diff = abs(a_deg - b_deg) % 180.0
    return diff if diff <= 90.0 else 180.0 - diff


def _geometry_centroid(geom: BaseGeometry) -> Point:
    if geom.is_empty:
        return Point(0.0, 0.0)
    centroid = geom.centroid
    if centroid.is_empty:
        return Point(geom.representative_point())
    return centroid


def _opening_direction_deg(geom: BaseGeometry) -> float | None:
    if geom.is_empty:
        return None
    if isinstance(geom, LineString):
        coords = list(geom.coords)
    elif isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        try:
            largest = max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            return None
        coords = list(largest.exterior.coords)
    else:
        return None

    if len(coords) < 2:
        return None

    longest_vec = _longest_edge_vector(coords)
    if longest_vec is None:
        return None
    dx, dy = longest_vec
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    angle_norm = angle % 180.0
    return angle_norm


def _axis_direction_deg(axis: LineString) -> float | None:
    coords = list(axis.coords)
    if len(coords) < 2:
        return None
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 180.0


def _longest_edge_vector(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float] | None:
    best_vec: Tuple[float, float] | None = None
    best_length_sq = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq > best_length_sq:
            best_length_sq = length_sq
            best_vec = (dx, dy)
    return best_vec


def _ensure_polygon(geom: BaseGeometry) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            return Polygon()
    buffered = geom.buffer(1.0)
    if isinstance(buffered, Polygon):
        return buffered
    if isinstance(buffered, MultiPolygon):
        try:
            return max(list(buffered.geoms), key=lambda g: g.area)
        except ValueError:
            return Polygon()
    return Polygon()


