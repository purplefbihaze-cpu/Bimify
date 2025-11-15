"""Geometry utility functions for IFC model building."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from core.ml.postprocess_floorplan import NormalizedDet
from core.vector.geometry import merge_polygons as merge_polygons_helper


logger = logging.getLogger(__name__)


STANDARD_WALL_THICKNESSES_MM: Sequence[float] = (115.0, 240.0, 300.0, 400.0, 500.0)
DEFAULT_WALL_THICKNESS_STANDARDS_MM: Tuple[float, ...] = (115.0, 240.0, 300.0, 400.0, 500.0)


@dataclass
class OpeningPlacement:
    """Placement information for an opening (door/window) in a wall."""
    width_mm: float
    center_xy: Tuple[float, float]
    axis_vec: Tuple[float, float]
    depth_mm: float


def collect_wall_polygons(
    normalized: Sequence[NormalizedDet],
    *,
    smooth_tolerance_mm: float = 1.0,
    snap_tolerance_mm: float = 5.0,
) -> Dict[int, Polygon | MultiPolygon]:
    """Return cleaned wall polygons keyed by their source index within wall detections.
    
    Args:
        normalized: Sequence of normalized detections.
        smooth_tolerance_mm: Tolerance for polygon smoothing.
        snap_tolerance_mm: Tolerance for snapping vertices.
    
    Returns:
        Dictionary mapping source index to cleaned polygon.
    """
    wall_polygons: Dict[int, Polygon | MultiPolygon] = {}
    walls = [nd for nd in normalized if nd.type == "WALL"]

    for source_index, det in enumerate(walls):
        geom = det.geom
        polygons: list[Polygon] = []
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))

        if not polygons:
            continue

        merged = merge_polygons_helper(
            polygons,
            tolerance=max(smooth_tolerance_mm, 0.0),
            snap_tolerance=max(snap_tolerance_mm, 0.0),
        )
        if not merged:
            continue

        if len(merged) == 1:
            wall_polygons[source_index] = merged[0]
        else:
            wall_polygons[source_index] = unary_union(merged)

    return wall_polygons


def snap_thickness_mm(
    value_mm: float | None,
    *,
    standards: Sequence[float] = STANDARD_WALL_THICKNESSES_MM,
    is_external: bool | None = None,
    default_external_mm: float = 240.0,
    default_internal_mm: float = 115.0,
) -> float:
    """Snap measured wall thickness to the closest standard size.
    
    BIM-compliant: Only snaps if deviation > 10mm from standard.
    Warns if deviation > 5mm from standard (BIM requirement: ±5mm accuracy).
    
    Args:
        value_mm: Measured wall thickness in millimeters.
        standards: Sequence of standard thickness values.
        is_external: Whether the wall is external (affects default).
        default_external_mm: Default thickness for external walls.
        default_internal_mm: Default thickness for internal walls.
    
    Returns:
        Snapped or original thickness value.
    """
    candidates = [float(v) for v in standards if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0]
    if not candidates:
        candidates = [default_external_mm, default_internal_mm]
    candidates = [c for c in candidates if c > 0.0]
    if not candidates:
        result = max(float(value_mm or 0.0), 40.0)
        if result < 40.0:
            logger.warning("Wall thickness: Value %.1fmm is below minimum (40mm) - using fallback 40mm", result)
        return result

    fallback = default_external_mm if is_external else default_internal_mm
    if fallback <= 0.0:
        fallback = candidates[0]

    if value_mm is None or not math.isfinite(value_mm) or value_mm <= 0.0:
        logger.debug("Wall thickness: Invalid value (None/infinite/<=0), using fallback %.1fmm", fallback)
        return fallback

    # Validate thin walls (< 40mm) - issue warning
    if value_mm < 40.0:
        logger.warning("Wall thickness: Value %.1fmm is below minimum standard (40mm) - using fallback %.1fmm", 
                      value_mm, fallback)
        return max(fallback, 40.0)

    # Find closest standard
    closest = min(candidates, key=lambda option: abs(option - value_mm))
    deviation = abs(closest - value_mm)
    
    # BIM-compliant: Warn if deviation > 5mm (BIM requirement: ±5mm accuracy)
    if deviation > 5.0:
        wall_type = "external" if is_external else "internal"
        logger.warning("Wall thickness: Deviation %.1fmm from standard %.1fmm exceeds BIM tolerance (±5mm) for %s wall (detected: %.1fmm)", 
                      deviation, closest, wall_type, value_mm)
    
    # Only snap if deviation > 10mm (reduced from 20mm for better BIM compliance)
    if deviation > 10.0:
        logger.debug("Wall thickness: Snapping %.1fmm to standard %.1fmm (deviation: %.1fmm)", 
                    value_mm, closest, deviation)
        return closest
    else:
        # Preserve original value if close to standard (within 10mm tolerance)
        logger.debug("Wall thickness: Preserving original value %.1fmm (deviation from standard %.1fmm: %.1fmm <= 10mm)", 
                    value_mm, closest, deviation)
        return value_mm


def prepare_thickness_standards(standards: Sequence[float] | None) -> Tuple[float, ...]:
    """Prepare and validate wall thickness standards.
    
    Args:
        standards: Optional sequence of standard thickness values.
    
    Returns:
        Tuple of validated standard thickness values.
    """
    values: Sequence[float] = standards if standards is not None else DEFAULT_WALL_THICKNESS_STANDARDS_MM
    filtered = sorted({float(abs(s)) for s in values if isinstance(s, (int, float)) and abs(float(s)) > 0.0})
    return tuple(filtered) if filtered else DEFAULT_WALL_THICKNESS_STANDARDS_MM


def snap_wall_thickness(value: float, *, is_external: bool | None, standards: Sequence[float]) -> float:
    """Snap wall thickness to nearest standard (simplified version).
    
    Args:
        value: Wall thickness value.
        is_external: Whether wall is external.
        standards: Sequence of standard thickness values.
    
    Returns:
        Snapped thickness value.
    """
    if not standards:
        return max(value, 1.0)
    reference = float(value) if value and value > 0 else (240.0 if is_external else 115.0)
    snapped = min(standards, key=lambda candidate: abs(candidate - reference))
    return float(snapped)


def planar_rectangle_polygon(
    center: Tuple[float, float],
    width: float,
    depth: float,
    axis_vec: Tuple[float, float],
) -> Polygon | None:
    """Create a planar rectangle polygon from center, dimensions, and axis vector.
    
    Args:
        center: Center point (x, y).
        width: Width of rectangle.
        depth: Depth of rectangle.
        axis_vec: Axis direction vector (ux, uy).
    
    Returns:
        Polygon representing the rectangle, or None if invalid.
    """
    ux, uy = axis_vec
    length = math.hypot(ux, uy)
    if length <= 1e-6:
        return None
    ux /= length
    uy /= length
    px = -uy
    py = ux
    half_w = max(width / 2.0, 1.0)
    half_d = max(depth / 2.0, 1.0)

    cx, cy = center
    corners = [
        (cx - ux * half_w - px * half_d, cy - uy * half_w - py * half_d),
        (cx + ux * half_w - px * half_d, cy + uy * half_w + py * half_d),
        (cx + ux * half_w + px * half_d, cy + uy * half_w + py * half_d),
        (cx - ux * half_w + px * half_d, cy - uy * half_w + py * half_d),
    ]
    return Polygon(corners)


def largest_polygon(geom: Polygon | MultiPolygon | LineString | None) -> Polygon | None:
    """Extract the largest polygon from geometry.
    
    Args:
        geom: Input geometry (Polygon, MultiPolygon, or LineString).
    
    Returns:
        Largest polygon, or None if no valid polygon found.
    """
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            return None
    return None


def iou(a: Polygon | None, b: Polygon | None) -> float:
    """Calculate Intersection over Union (IoU) of two polygons.
    
    Args:
        a: First polygon.
        b: Second polygon.
        
    Returns:
        IoU value between 0.0 and 1.0.
    """
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return float(inter / union) if union > 1e-6 else 0.0


def find_nearest_wall(
    target: Polygon,
    wall_records: List[Tuple[Any, Polygon, bool, float]],
    wall_index_tree: STRtree | None,
    wall_geoms: List[Polygon],
) -> Tuple[Any, Polygon, bool, float] | None:
    """Find wall that exactly intersects with target polygon using exact geometric checks.
    
    This function is shared between door and window processing to avoid code duplication.
    
    Args:
        target: Target polygon (door or window) to find nearest wall for
        wall_records: List of (wall_entity, wall_geom, is_external, thickness) tuples
        wall_index_tree: STRtree spatial index for walls (can be None)
        wall_geoms: List of wall geometry polygons (for index lookup)
        
    Returns:
        Tuple of (wall_entity, wall_geom, is_external, thickness) or None if no wall found
    """
    if not wall_records or wall_index_tree is None:
        return None
    
    # Use exact intersection check: search for wall that contains centroid or intersects polygon
    candidates = wall_index_tree.query(target)
    if not candidates:
        # Try with small buffer for near-misses
        buffered = target.buffer(0.01)
        candidates = wall_index_tree.query(buffered)
    
    # Check candidates for exact intersection
    for candidate_geom in candidates:
        try:
            candidate_idx = wall_geoms.index(candidate_geom)
            if candidate_idx < len(wall_records):
                wall, geom, _is_external, thickness = wall_records[candidate_idx]
                # Exact check: wall contains target centroid OR wall intersects target polygon
                if geom.contains(target.centroid) or geom.intersects(target):
                    return (wall, geom, _is_external, thickness)
        except (ValueError, IndexError):
            # Skip invalid indices
            continue
    
    # No wall found - return None (no placeholder)
    return None


def compute_opening_placement(
    opening_det: NormalizedDet,
    axis: LineString | None,
    default_width: float,
    wall_thickness: float,
) -> OpeningPlacement:
    """Compute placement for an opening (door/window) in a wall.
    
    Args:
        opening_det: Normalized detection for the opening.
        axis: Wall axis line string.
        default_width: Default width for the opening.
        wall_thickness: Thickness of the wall.
    
    Returns:
        OpeningPlacement with computed dimensions and position.
    """
    geom = opening_det.geom
    coords: List[tuple[float, float]] = []
    if isinstance(geom, Polygon):
        coords = [(float(x), float(y)) for x, y in list(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        try:
            largest = max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            largest = None
        if largest is not None:
            coords = [(float(x), float(y)) for x, y in list(largest.exterior.coords)]
    elif isinstance(geom, LineString):
        coords = [(float(x), float(y)) for x, y in list(geom.coords)]

    default_center = (float(geom.centroid.x), float(geom.centroid.y)) if geom is not None else (0.0, 0.0)
    width = float(default_width)
    depth = float(wall_thickness)
    axis_vec = (1.0, 0.0)

    # Ensure coords is checked properly (avoid numpy array boolean ambiguity)
    # Use len() check instead of direct boolean evaluation
    has_coords = len(coords) > 0
    if axis is not None and axis.length > 1e-3 and has_coords:
        ax_coords = list(axis.coords)
        origin_x, origin_y = float(ax_coords[0][0]), float(ax_coords[0][1])
        end_x, end_y = float(ax_coords[-1][0]), float(ax_coords[-1][1])
        dx = end_x - origin_x
        dy = end_y - origin_y
        length = math.hypot(dx, dy)
        if length > 1e-6:
            ux = dx / length
            uy = dy / length
            # Snap axis to global orthogonal to avoid slight skews (viewer-friendly)
            angle = (math.degrees(math.atan2(uy, ux)) + 360.0) % 180.0
            if abs(angle - 90.0) < 45.0 and abs(angle - 90.0) <= abs(angle - 0.0):
                axis_vec = (0.0, 1.0)
            else:
                axis_vec = (1.0, 0.0)
            px = -uy
            py = ux

            axis_projections: List[float] = []
            perp_projections: List[float] = []
            for x, y in coords:
                vx = x - origin_x
                vy = y - origin_y
                axis_projections.append(vx * ux + vy * uy)
                perp_projections.append(vx * px + vy * py)

            # Compute center from centroid projected onto axis (reduces offsets)
            try:
                c = opening_det.geom.centroid
                cx, cy = float(c.x), float(c.y)
            except Exception:
                cx = float(sum(x for x, _ in coords) / len(coords))
                cy = float(sum(y for _, y in coords) / len(coords))
            vcx = cx - origin_x
            vcy = cy - origin_y
            proj = vcx * ux + vcy * uy
            # clamp to axis extent
            proj = max(0.0, min(length, proj))
            default_center = (origin_x + ux * proj, origin_y + uy * proj)

            if axis_projections:
                min_proj = min(axis_projections)
                max_proj = max(axis_projections)
                span = max_proj - min_proj
                if span > 1e-3:
                    width = float(span)

            if perp_projections:
                perp_min = min(perp_projections)
                perp_max = max(perp_projections)
                perp_span = perp_max - perp_min
                if perp_span > 1e-3:
                    depth = float(min(wall_thickness, max(perp_span, min(wall_thickness, 80.0))))
    elif has_coords:
        # Fallback: orient by longest edge (with orthogonal snap tolerance)
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        if xs and ys:
            width = max(max(xs) - min(xs), max(ys) - min(ys), default_width * 0.5)
            default_center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
            # determine direction
            best_len_sq = -1.0
            best_vec = (1.0, 0.0)
            for i in range(len(coords) - 1):
                vx = float(coords[i + 1][0] - coords[i][0])
                vy = float(coords[i + 1][1] - coords[i][1])
                cand = vx * vx + vy * vy
                if cand > best_len_sq:
                    best_len_sq = cand
                    best_vec = (vx, vy)
            ang = math.degrees(math.atan2(best_vec[1], best_vec[0])) % 180.0
            # snap to 0/90 if close
            if min(ang, abs(180.0 - ang)) <= 10.0:
                axis_vec = (1.0, 0.0)
            elif abs(ang - 90.0) <= 10.0:
                axis_vec = (0.0, 1.0)
            else:
                norm = math.hypot(best_vec[0], best_vec[1]) or 1.0
                axis_vec = (best_vec[0] / norm, best_vec[1] / norm)
            depth = min(wall_thickness, max(width * 0.4, min(wall_thickness, 80.0)))

    width = max(width, default_width * 0.5)
    depth = max(depth, min(wall_thickness, 40.0))
    # Persist axis-aligned rectangle into detection for downstream consumers
    oriented_poly = planar_rectangle_polygon(default_center, width, min(depth, wall_thickness), axis_vec)
    if oriented_poly is not None and not oriented_poly.is_empty:
        try:
            opening_det.geom = oriented_poly
            if isinstance(opening_det.attrs, dict):
                opening_det.attrs["geometry_source"] = "axis_aligned"
        except Exception:
            pass

    return OpeningPlacement(width_mm=width, center_xy=default_center, axis_vec=axis_vec, depth_mm=depth)


def fit_opening_to_axis(
    opening_det: NormalizedDet,
    axis: LineString,
    wall_thickness: float,
) -> tuple[OpeningPlacement, Polygon, dict]:
    """Fit an opening to a wall axis with improved geometry.
    
    Args:
        opening_det: Normalized detection for the opening.
        axis: Wall axis line string.
        wall_thickness: Thickness of the wall.
    
    Returns:
        Tuple of (OpeningPlacement, fitted Polygon, metadata dict with IoU).
    """
    rf_poly = largest_polygon(opening_det.geom)
    coords = []
    if isinstance(rf_poly, Polygon):
        coords = [(float(x), float(y)) for x, y in list(rf_poly.exterior.coords)]
    # Ensure coords is checked properly (avoid numpy array boolean ambiguity)
    # Use len() check instead of direct boolean evaluation
    has_coords_fit = len(coords) > 0
    if not has_coords_fit:
        # Fallback: use current placement
        placement = compute_opening_placement(opening_det, axis, 900.0 if opening_det.type == "DOOR" else 1200.0, wall_thickness)
        rect = planar_rectangle_polygon(placement.center_xy, placement.width_mm, min(placement.depth_mm, wall_thickness), placement.axis_vec)
        return placement, rect if rect is not None else Polygon(), {"iou": 0.0}

    # Axis basis
    ax = list(axis.coords)
    ox, oy = float(ax[0][0]), float(ax[0][1])
    ex, ey = float(ax[-1][0]), float(ax[-1][1])
    dx = ex - ox
    dy = ey - oy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        length = 1.0
    ux = dx / length
    uy = dy / length
    # Snap to 0/90 deg
    ang = (math.degrees(math.atan2(uy, ux)) + 360.0) % 180.0
    if abs(ang - 90.0) < 45.0 and abs(ang - 90.0) <= abs(ang - 0.0):
        base = (0.0, 1.0)
    else:
        base = (1.0, 0.0)
    ux, uy = base
    px, py = -uy, ux

    # Projections of RF polygon to derive spans
    along_vals = []
    perp_vals = []
    for x, y in coords:
        vx = x - ox
        vy = y - oy
        along_vals.append(vx * ux + vy * uy)
        perp_vals.append(vx * px + vy * py)
    min_a, max_a = min(along_vals), max(along_vals)
    min_p, max_p = min(perp_vals), max(perp_vals)
    span = max(1.0, max_a - min_a)
    perp_span = max(1.0, max_p - min_p)
    width = float(span)
    depth = float(min(wall_thickness, max(perp_span, min(wall_thickness, 80.0))))

    # Candidate centers
    mid_center = (ox + ux * ((min_a + max_a) / 2.0), oy + uy * ((min_a + max_a) / 2.0))
    try:
        c = rf_poly.centroid
        cproj = ((float(c.x) - ox) * ux + (float(c.y) - oy) * uy)
        centroid_center = (ox + ux * cproj, oy + uy * cproj)
    except Exception:
        centroid_center = mid_center

    # Slide search ±150mm in 15 steps
    search_radius = 150.0
    steps = 15
    step = (2.0 * search_radius) / steps
    candidates = []
    for base_center in (mid_center, centroid_center):
        for i in range(steps + 1):
            delta = -search_radius + i * step
            cx = base_center[0] + ux * delta
            cy = base_center[1] + uy * delta
            rect = planar_rectangle_polygon((cx, cy), width, depth, (ux, uy))
            candidates.append(((cx, cy), rect))

    best_iou = -1.0
    best_center = mid_center
    best_rect = None
    for (cx, cy), rect in candidates:
        iou_val = iou(rect, rf_poly)
        if iou_val > best_iou:
            best_iou = iou_val
            best_center = (cx, cy)
            best_rect = rect

    # Fallback if the IoU is too low or geometry invalid
    if best_rect is None or best_iou < 0.05 or width <= 1e-3 or depth <= 1e-3:
        default_width = 900.0 if opening_det.type == "DOOR" else 1200.0
        fallback = compute_opening_placement(opening_det, axis, default_width, wall_thickness)
        rect = planar_rectangle_polygon(fallback.center_xy, max(fallback.width_mm, 100.0), min(max(fallback.depth_mm, 40.0), wall_thickness), fallback.axis_vec)
        return fallback, (rect if rect is not None else Polygon()), {"iou": 0.0, "fallback": True}

    placement = OpeningPlacement(width_mm=width, center_xy=best_center, axis_vec=(ux, uy), depth_mm=depth)
    metrics = {"iou": float(max(0.0, min(1.0, best_iou))), "span_mm": width, "perp_span_mm": perp_span}
    return placement, (best_rect if best_rect is not None else Polygon()), metrics


__all__ = [
    "OpeningPlacement",
    "collect_wall_polygons",
    "snap_thickness_mm",
    "prepare_thickness_standards",
    "snap_wall_thickness",
    "planar_rectangle_polygon",
    "largest_polygon",
    "iou",
    "compute_opening_placement",
    "fit_opening_to_axis",
    "find_nearest_wall",
    "STANDARD_WALL_THICKNESSES_MM",
    "DEFAULT_WALL_THICKNESS_STANDARDS_MM",
]

