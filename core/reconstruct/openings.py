from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis

logger = logging.getLogger(__name__)


@dataclass
class OpeningAssignment:
    opening: NormalizedDet
    wall_index: Optional[int]
    distance_mm: float
    axis_index: Optional[int] = None
    axis_distance_mm: Optional[float] = None
    angle_delta_deg: Optional[float] = None


def _rectangle_polygon(
    center_xy: tuple[float, float],
    width_mm: float,
    depth_mm: float,
    axis_vec_2d: tuple[float, float],
) -> Polygon:
    """Create an oriented rectangle polygon centered at center_xy."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    wx = max(float(width_mm) / 2.0, 0.5)
    dz = max(float(depth_mm) / 2.0, 0.5)
    ux, uy = axis_vec_2d
    norm = math.hypot(ux, uy) or 1.0
    ux /= norm
    uy /= norm
    px, py = -uy, ux
    p1 = (cx - ux * wx - px * dz, cy - uy * wx - py * dz)
    p2 = (cx + ux * wx - px * dz, cy + uy * wx - py * dz)
    p3 = (cx + ux * wx + px * dz, cy + uy * wx + py * dz)
    p4 = (cx - ux * wx + px * dz, cy - uy * wx + py * dz)
    return Polygon([p1, p2, p3, p4])


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


def reproject_openings_to_snapped_axes(
    dets: List[NormalizedDet],
    assignments: List[OpeningAssignment],
    wall_axes: Sequence[WallAxis] | None = None,
    *,
    depth_equals_wall_thickness: bool = True,
    wall_polygons: Sequence[Polygon | MultiPolygon] | None = None,
) -> None:
    """
    Mutate opening detections to rectangles centered and aligned on their assigned axes.
    - Width: taken from RF geometry span along the chosen axis.
    - Depth: wall thickness when requested (bündig), otherwise preserved.
    - Validates: opening fully within wall polygon, on axis, depth clamped to wall thickness.
    """
    if not assignments:
        return
    axes: List[WallAxis] = list(wall_axes or [])
    wall_polys_list = list(wall_polygons or [])

    # Precompute mapping axis_index -> (unit vector, length, origin, thickness)
    axis_info: dict[int, tuple[tuple[float, float], float, tuple[float, float], float]] = {}
    for idx, axis in enumerate(axes):
        if getattr(axis, "axis", None) is None:
            continue
        coords = list(axis.axis.coords)
        if len(coords) < 2:
            continue
        (ox, oy), (ex, ey) = coords[0], coords[-1]
        dx, dy = float(ex - ox), float(ey - oy)
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        thickness = float(getattr(axis, "width_mm", None) or 0.0)
        axis_info[idx] = ((ux, uy), length, (ox, oy), max(thickness, 0.0))

    # Helper to measure RF span along a direction
    def rf_span_and_center(det: NormalizedDet, dir_vec: tuple[float, float]) -> tuple[float, tuple[float, float]]:
        ux, uy = dir_vec
        geom = det.geom
        coords: List[tuple[float, float]] = []
        if isinstance(geom, Polygon):
            coords = [(float(x), float(y)) for x, y in list(geom.exterior.coords)]
        elif isinstance(geom, LineString):
            coords = [(float(x), float(y)) for x, y in list(geom.coords)]
        else:
            try:
                rect = geom.minimum_rotated_rectangle
                coords = [(float(x), float(y)) for x, y in list(rect.exterior.coords)]
            except Exception:
                pass
        if not coords:
            c = _geometry_centroid(geom)
            return (900.0 if det.type == "DOOR" else 1200.0), (float(c.x), float(c.y))
        projs = [x * ux + y * uy for (x, y) in coords]
        width = float(max(projs) - min(projs))
        c = _geometry_centroid(geom)
        return max(width, 100.0), (float(c.x), float(c.y))

    for ass in assignments:
        if ass.wall_index is None or ass.axis_index is None:
            continue
        opening = ass.opening
        info = axis_info.get(int(ass.axis_index))
        if not info:
            continue
        (ux, uy), seg_len, (ox, oy), axis_thickness = info
        width, centroid_xy = rf_span_and_center(opening, (ux, uy))

        # Clamp projected center to segment extents (keep half-width inside)
        vx, vy = centroid_xy[0] - ox, centroid_xy[1] - oy
        t = vx * ux + vy * uy
        half = max(width / 2.0, 1.0)
        if seg_len < half * 2.0:
            t = seg_len / 2.0
        else:
            t = min(max(t, half), seg_len - half)
        center_xy = (ox + ux * t, oy + uy * t)

        # Clamp depth to wall thickness if opening depth > wall thickness
        depth = float(axis_thickness) if depth_equals_wall_thickness and axis_thickness > 0.0 else max(40.0, axis_thickness)
        if depth > axis_thickness and axis_thickness > 0.0:
            depth = float(axis_thickness)
            logger.debug(
                "Opening depth clamped to wall thickness: %.1fmm (was %.1fmm) for opening on axis %d",
                depth, float(axis_thickness) if depth_equals_wall_thickness else max(40.0, axis_thickness),
                int(ass.axis_index)
            )
        
        # Ensure depth is valid (fallback if axis_thickness is 0)
        if depth <= 0.0:
            depth = max(40.0, axis_thickness if axis_thickness > 0.0 else 115.0)
            logger.debug(
                "Opening depth set to fallback value: %.1fmm (axis thickness was %.1fmm) for opening on axis %d",
                depth, axis_thickness, int(ass.axis_index)
            )
        
        rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
        
        # Enhanced validation: Check if opening is fully within wall polygon (5mm tolerance for BIM compliance)
        if ass.wall_index is not None and ass.wall_index < len(wall_polys_list):
            wall_poly = wall_polys_list[ass.wall_index]
            if isinstance(wall_poly, Polygon):
                # First check: validation with 5mm tolerance (BIM-compliant)
                # Use buffer(-5mm) to check if opening fits with tolerance
                wall_with_tolerance = wall_poly.buffer(-5.0) if wall_poly.area > 0 else wall_poly
                opening_fits = wall_with_tolerance.contains(rect) if not wall_with_tolerance.is_empty else wall_poly.contains(rect)
                if not opening_fits:
                    # Opening extends beyond wall - need to adjust
                    # Strategy 1: Try to move opening center to be within wall
                    center_in_wall = wall_poly.contains(Point(center_xy))
                    
                    if center_in_wall:
                        # Center is in wall, but opening extends beyond - intelligent size adjustment
                        # Strategy: Try to preserve original size, only reduce if absolutely necessary
                        # First, try with 5mm tolerance (BIM-compliant)
                        wall_with_tolerance = wall_poly.buffer(-5.0) if wall_poly.area > 0 else wall_poly
                        if not wall_with_tolerance.is_empty and wall_with_tolerance.contains(rect):
                            # Opening fits with 5mm tolerance - keep original size
                            logger.debug(
                                "Opening fits within wall %d with 5mm tolerance - preserving original size %.1fmm x %.1fmm",
                                ass.wall_index, width, depth
                            )
                        else:
                            # Opening doesn't fit even with tolerance - reduce size intelligently
                            # Find maximum width that fits (iterative binary search for better accuracy)
                            max_width = width
                            min_width = 100.0  # Minimum opening width
                            test_rect = None
                            
                            # Binary search for maximum width that fits with tolerance
                            for iteration in range(10):  # Max 10 iterations
                                test_width = (max_width + min_width) / 2.0
                                test_rect = _rectangle_polygon(center_xy, test_width, depth, (ux, uy))
                                if not wall_with_tolerance.is_empty and wall_with_tolerance.contains(test_rect):
                                    min_width = test_width
                                    max_width = max(max_width, test_width)
                                else:
                                    max_width = test_width
                                
                                if abs(max_width - min_width) < 1.0:  # Convergence
                                    break
                            
                            # Also check if depth needs reduction (only if width reduction wasn't enough)
                            max_depth = depth
                            if test_rect and (wall_with_tolerance.is_empty or not wall_with_tolerance.contains(test_rect)):
                                # Try reducing depth as well, but prefer keeping original depth
                                for test_depth in [depth, depth * 0.95, depth * 0.9, depth * 0.8, depth * 0.7, axis_thickness]:
                                    if test_depth <= 0.0:
                                        continue
                                    test_rect_depth = _rectangle_polygon(center_xy, max_width, test_depth, (ux, uy))
                                    if not wall_with_tolerance.is_empty and wall_with_tolerance.contains(test_rect_depth):
                                        max_depth = test_depth
                                        break
                            
                            # Only reduce if necessary (intelligent adjustment)
                            if max_width < width * 0.95 or max_depth < depth * 0.95:  # Only reduce if >5% reduction needed
                                logger.debug(
                                    "Opening size reduced from %.1fmm x %.1fmm to %.1fmm x %.1fmm to fit within wall %d (with 5mm tolerance)",
                                    width, depth, max_width, max_depth, ass.wall_index
                                )
                                width = max(max_width, min_width)
                                depth = max(max_depth, 40.0)
                                rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                            else:
                                # Reduction <5% - keep original size
                                logger.debug(
                                    "Opening size adjustment <5%% - preserving original size %.1fmm x %.1fmm",
                                    width, depth
                                )
                    else:
                        # Center is outside wall - project to nearest point on wall boundary or axis
                        # Strategy 1: Try to project onto axis first (preferred)
                        if ass.axis_index is not None and ass.axis_index < len(axes):
                            axis_obj = axes[ass.axis_index]
                            if hasattr(axis_obj, "axis") and axis_obj.axis:
                                axis_line = axis_obj.axis
                                # Project center onto axis
                                coords = list(axis_line.coords)
                                if len(coords) >= 2:
                                    min_dist = float('inf')
                                    closest_point_on_axis = center_xy
                                    for i in range(len(coords) - 1):
                                        p1 = Point(coords[i])
                                        p2 = Point(coords[i + 1])
                                        dx = p2.x - p1.x
                                        dy = p2.y - p1.y
                                        length_sq = dx * dx + dy * dy
                                        if length_sq > 1e-6:
                                            vx = center_xy[0] - p1.x
                                            vy = center_xy[1] - p1.y
                                            t = max(0.0, min(1.0, (vx * dx + vy * dy) / length_sq))
                                            proj_x = p1.x + t * dx
                                            proj_y = p1.y + t * dy
                                            proj_point = Point(proj_x, proj_y)
                                            dist = Point(center_xy).distance(proj_point)
                                            if dist < min_dist:
                                                min_dist = dist
                                                closest_point_on_axis = (proj_x, proj_y)
                                    
                                    # Check if point on axis is within wall
                                    if wall_poly.contains(Point(closest_point_on_axis)):
                                        center_xy = closest_point_on_axis
                                        rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                                        logger.debug(
                                            "Opening center projected onto axis %d and adjusted to be within wall %d",
                                            int(ass.axis_index), ass.wall_index
                                        )
                                    else:
                                        # Fallback: project to nearest point on wall boundary
                                        nearest_point = wall_poly.boundary.interpolate(wall_poly.boundary.project(Point(center_xy)))
                                        center_xy = (float(nearest_point.x), float(nearest_point.y))
                                        rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                                        logger.debug(
                                            "Opening center projected to wall %d boundary (fallback)",
                                            ass.wall_index
                                        )
                                        # Verify opening fits after projection
                                        if not wall_poly.contains(rect):
                                            # Still doesn't fit - reduce size
                                            for test_width in [width * 0.9, width * 0.8, width * 0.7, width * 0.5]:
                                                test_rect = _rectangle_polygon(center_xy, test_width, depth, (ux, uy))
                                                if wall_poly.contains(test_rect):
                                                    width = test_width
                                                    rect = test_rect
                                                    logger.debug(
                                                        "Opening width reduced to %.1fmm after boundary projection",
                                                        width
                                                    )
                                                    break
                        else:
                            # No axis available - project to wall boundary
                            nearest_point = wall_poly.boundary.interpolate(wall_poly.boundary.project(Point(center_xy)))
                            center_xy = (float(nearest_point.x), float(nearest_point.y))
                            rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                            logger.debug(
                                "Opening center projected to wall %d boundary (no axis available)",
                                ass.wall_index
                            )
                    
                    # Final verification: opening must be within wall with 5mm tolerance (BIM-compliant)
                    wall_with_tolerance = wall_poly.buffer(-5.0) if wall_poly.area > 0 else wall_poly
                    final_check = wall_with_tolerance.contains(rect) if not wall_with_tolerance.is_empty else wall_poly.contains(rect)
                    if not final_check:
                        # Last resort: reduce size until it fits (intelligent: preserve as much as possible)
                        for scale in [0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
                            test_width = width * scale
                            test_depth = depth * scale
                            if test_width < 100.0 or test_depth < 40.0:
                                break
                            test_rect = _rectangle_polygon(center_xy, test_width, test_depth, (ux, uy))
                            check_result = wall_with_tolerance.contains(test_rect) if not wall_with_tolerance.is_empty else wall_poly.contains(test_rect)
                            if check_result:
                                width = test_width
                                depth = test_depth
                                rect = test_rect
                                logger.debug(
                                    "Opening size reduced to %.1fmm x %.1fmm to guarantee fit within wall %d (with 5mm tolerance)",
                                    width, depth, ass.wall_index
                                )
                                break
        
        # Enhanced validation: Check if opening center is on wall axis (10mm tolerance for BIM compliance)
        if ass.axis_index is not None and ass.axis_index < len(axes):
            axis_obj = axes[ass.axis_index]
            if hasattr(axis_obj, "axis") and axis_obj.axis:
                axis_line = axis_obj.axis
                center_point = Point(center_xy)
                distance_to_axis = axis_line.distance(center_point)
                max_axis_distance = 10.0  # Reduced to 10mm for better BIM compliance (was 20mm)
                if distance_to_axis > max_axis_distance:
                    # Project center onto axis (GUARANTEED)
                    coords = list(axis_line.coords)
                    if len(coords) >= 2:
                        # Find closest point on axis line
                        min_dist = float('inf')
                        closest_point = center_xy
                        for i in range(len(coords) - 1):
                            p1 = Point(coords[i])
                            p2 = Point(coords[i + 1])
                            # Project center onto line segment
                            dx = p2.x - p1.x
                            dy = p2.y - p1.y
                            length_sq = dx * dx + dy * dy
                            if length_sq > 1e-6:
                                vx = center_point.x - p1.x
                                vy = center_point.y - p1.y
                                t = max(0.0, min(1.0, (vx * dx + vy * dy) / length_sq))
                                proj_x = p1.x + t * dx
                                proj_y = p1.y + t * dy
                                proj_point = Point(proj_x, proj_y)
                                dist = center_point.distance(proj_point)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_point = (proj_x, proj_y)
                        
                        # Guaranteed projection: always project if distance > threshold
                        if min_dist < distance_to_axis:
                            center_xy = closest_point
                            # Verify opening still fits within wall after projection (with 5mm tolerance)
                            if ass.wall_index is not None and ass.wall_index < len(wall_polys_list):
                                wall_poly = wall_polys_list[ass.wall_index]
                                if isinstance(wall_poly, Polygon):
                                    rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                                    wall_with_tolerance = wall_poly.buffer(-5.0) if wall_poly.area > 0 else wall_poly
                                    opening_fits = wall_with_tolerance.contains(rect) if not wall_with_tolerance.is_empty else wall_poly.contains(rect)
                                    if not opening_fits:
                                        # Opening doesn't fit after projection - reduce size intelligently
                                        for scale in [0.98, 0.95, 0.9, 0.85, 0.8, 0.7]:
                                            test_width = width * scale
                                            test_rect = _rectangle_polygon(center_xy, test_width, depth, (ux, uy))
                                            check_result = wall_with_tolerance.contains(test_rect) if not wall_with_tolerance.is_empty else wall_poly.contains(test_rect)
                                            if check_result:
                                                width = test_width
                                                rect = test_rect
                                                logger.debug(
                                                    "Opening width reduced to %.1fmm after axis projection (with 5mm tolerance)",
                                                    width
                                                )
                                                break
                            else:
                                rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                            
                            logger.debug(
                                "Opening center projected onto axis %d (distance reduced from %.1fmm to %.1fmm) - GUARANTEED",
                                int(ass.axis_index), distance_to_axis, min_dist
                            )
                        else:
                            # Projection failed - use closest point anyway
                            center_xy = closest_point
                            rect = _rectangle_polygon(center_xy, width, depth, (ux, uy))
                            logger.debug(
                                "Opening center projected onto axis %d (fallback projection)",
                                int(ass.axis_index)
                            )
        
        # Validate and repair opening geometry before assignment
        if rect is not None and not rect.is_empty:
            # Validate polygon is valid
            if not rect.is_valid:
                try:
                    repaired = rect.buffer(0)
                    if not repaired.is_empty and repaired.is_valid:
                        if isinstance(repaired, Polygon):
                            rect = repaired
                        elif isinstance(repaired, MultiPolygon):
                            # Take largest valid polygon
                            valid_polys = [p for p in repaired.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                            if valid_polys:
                                rect = max(valid_polys, key=lambda p: p.area)
                except Exception:
                    pass
            
            # Check if closed
            if rect.is_valid:
                coords = list(rect.exterior.coords)
                if len(coords) >= 3:
                    first = coords[0]
                    last = coords[-1]
                    dist = math.hypot(first[0] - last[0], first[1] - last[1])
                    if dist > 1.0:  # Not closed
                        try:
                            coords.append(coords[0])
                            rect = Polygon(coords)
                        except Exception:
                            pass
            
            # Validate coordinates are finite
            if rect.is_valid:
                coords = list(rect.exterior.coords)
                for coord in coords:
                    if not all(math.isfinite(c) for c in coord):
                        logger.warning("Opening geometry has non-finite coordinates - skipping assignment")
                        rect = None
                        break
        
        try:
            if rect is not None and not rect.is_empty and rect.is_valid:
                opening.geom = rect
        except Exception:
            # keep original geometry on failure
            pass

