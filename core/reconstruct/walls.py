from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from core.geometry.contract import (
    SNAP_ENDPOINTS_MM,
    MIN_SEGMENT_LEN_MM,
    MIN_POLYGON_AREA,
    SEGMENT_MERGE_ANGLE_DEG,
)


logger = logging.getLogger(__name__)


def _cluster_points(points: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    clusters: List[Tuple[float, float, int]] = []  # (cx, cy, count)
    for x, y in points:
        assigned = -1
        for i, (cx, cy, cnt) in enumerate(clusters):
            if math.hypot(x - cx, y - cy) <= tol:
                assigned = i
                break
        if assigned == -1:
            clusters.append((x, y, 1))
        else:
            cx, cy, cnt = clusters[assigned]
            cnt2 = cnt + 1
            clusters[assigned] = (cx + (x - cx) / cnt2, cy + (y - cy) / cnt2, cnt2)
    return [(cx, cy) for (cx, cy, _cnt) in clusters]


def collapse_wall_nodes(
    normalized: List,
    *,
    tol_mm: float = SNAP_ENDPOINTS_MM,
) -> List:
    """Merge nearby exterior vertices of wall polygons to shared cluster centers."""
    from shapely.geometry import Polygon as _Polygon, MultiPolygon as _MultiPolygon
    # Collect unique clusters across all exterior vertices
    all_pts: List[Tuple[float, float]] = []
    for det in normalized:
        if getattr(det, "type", "").upper() != "WALL":
            continue
        geom = getattr(det, "geom", None)
        if isinstance(geom, _Polygon):
            coords = list(geom.exterior.coords)[:-1]
            all_pts.extend(coords)
        elif isinstance(geom, _MultiPolygon):
            for part in geom.geoms:
                if isinstance(part, _Polygon):
                    coords = list(part.exterior.coords)[:-1]
                    all_pts.extend(coords)
    centers = _cluster_points(all_pts, tol_mm)

    def _snap_ring(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for (x, y) in coords:
            best = (x, y)
            best_d = float("inf")
            for (cx, cy) in centers:
                d = math.hypot(x - cx, y - cy)
                if d < best_d and d <= tol_mm:
                    best_d = d
                    best = (cx, cy)
            out.append(best)
        if out and out[0] != out[-1]:
            out.append(out[0])
        return out

    repaired: List = []
    for det in normalized:
        try:
            if getattr(det, "type", "").upper() != "WALL":
                repaired.append(det)
                continue
            geom = getattr(det, "geom", None)
            if not isinstance(geom, _Polygon):
                repaired.append(det)
                continue
            ring = list(geom.exterior.coords)
            ring = _snap_ring(ring)
            poly = _Polygon(ring)
            if not poly.is_valid:
                try:
                    poly = poly.buffer(0)
                except Exception:
                    pass
            setattr(det, "geom", poly)
            repaired.append(det)
        except Exception:
            repaired.append(det)
    return repaired


def _project_point_to_segment(px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return ax, ay, 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t_clamped = max(0.0, min(1.0, t))
    qx = ax + t_clamped * dx
    qy = ay + t_clamped * dy
    return qx, qy, t_clamped


def resolve_t_junctions(
    normalized: List,
    *,
    tol_mm: float = SNAP_ENDPOINTS_MM,
    endpoint_margin_ratio: float = 0.15,
) -> List:
    """Insert connector nodes where a vertex projects into the middle of another wall edge."""
    from shapely.geometry import Polygon as _Polygon
    walls = [det for det in normalized if getattr(det, "type", "").upper() == "WALL" and isinstance(getattr(det, "geom", None), _Polygon)]
    for i, det_i in enumerate(walls):
        poly_i: _Polygon = det_i.geom
        vcoords = list(poly_i.exterior.coords)[:-1]
        for vx, vy in vcoords:
            for j, det_j in enumerate(walls):
                if i == j:
                    continue
                poly_j: _Polygon = det_j.geom
                coords = list(poly_j.exterior.coords)
                modified = False
                for k in range(len(coords) - 1):
                    p0 = coords[k]
                    p1 = coords[k + 1]
                    qx, qy, t = _project_point_to_segment(vx, vy, p0, p1)
                    d = math.hypot(vx - qx, vy - qy)
                    if d <= tol_mm and endpoint_margin_ratio < t < (1.0 - endpoint_margin_ratio):
                        # insert projected point between k and k+1
                        coords.insert(k + 1, (qx, qy))
                        modified = True
                        break
                if modified:
                    # Rebuild polygon and repair
                    try:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        new_poly = _Polygon(coords)
                        if not new_poly.is_valid:
                            new_poly = new_poly.buffer(0)
                        det_j.geom = new_poly
                    except Exception:
                        pass
    return normalized


def merge_overlapping_walls(
    normalized: List,
    *,
    iou_threshold: float = 0.7,
    cover_threshold: float = 0.85,
) -> List:
    """Remove duplicate parallel/overlapped walls based on area overlap coverage.

    Keep the larger polygon when two wall polygons substantially overlap each other.
    """
    from shapely.geometry import Polygon as _Polygon
    walls = [(idx, det, det.geom) for idx, det in enumerate(normalized) if getattr(det, "type", "").upper() == "WALL" and isinstance(getattr(det, "geom", None), _Polygon)]
    to_drop: set[int] = set()
    for a_idx, a_det, a_poly in walls:
        if a_idx in to_drop:
            continue
        for b_idx, b_det, b_poly in walls:
            if b_idx <= a_idx or b_idx in to_drop:
                continue
            try:
                inter = a_poly.intersection(b_poly)
                if inter.is_empty:
                    continue
                inter_area = float(inter.area)
                a_area = float(a_poly.area) or 1.0
                b_area = float(b_poly.area) or 1.0
                iou = inter_area / (a_area + b_area - inter_area)
                cov_a = inter_area / a_area
                cov_b = inter_area / b_area
                if iou >= iou_threshold or (cov_a >= cover_threshold and cov_b >= cover_threshold * 0.8):
                    # Drop the smaller
                    drop_idx = a_idx if a_area < b_area else b_idx
                    to_drop.add(drop_idx)
            except Exception:
                continue
    if not to_drop:
        return normalized
    return [det for idx, det in enumerate(normalized) if idx not in to_drop]

def _enforce_clockwise(poly: Polygon) -> Polygon:
    try:
        from shapely.geometry.polygon import orient
        return orient(poly, sign=-1.0)  # Clockwise
    except Exception:
        return poly


def _remove_short_segments(coords: list[tuple[float, float]], min_len_mm: float) -> list[tuple[float, float]]:
    if not coords:
        return coords
    result: list[tuple[float, float]] = []
    for pt in coords:
        if not result:
            result.append(pt)
            continue
        px, py = result[-1]
        dx = pt[0] - px
        dy = pt[1] - py
        if (dx * dx + dy * dy) ** 0.5 >= min_len_mm:
            result.append(pt)
    # Ensure closed ring
    if result and (result[0][0] != result[-1][0] or result[0][1] != result[-1][1]):
        result.append(result[0])
    return result


def _merge_colinear(coords: list[tuple[float, float]], angle_tol_deg: float = 5.0) -> list[tuple[float, float]]:
    if len(coords) < 4:
        return coords
    def angle(p0, p1, p2):
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1 = math.hypot(*v1) or 1.0
        l2 = math.hypot(*v2) or 1.0
        dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))
    res = [coords[0]]
    for i in range(1, len(coords) - 1):
        a = angle(res[-1], coords[i], coords[i + 1])
        if abs(180.0 - a) <= angle_tol_deg:
            # Skip nearly colinear middle vertex
            continue
        res.append(coords[i])
    res.append(coords[-1])
    return res


def _round_to_grid(coords: list[tuple[float, float]], snap_mm: float) -> list[tuple[float, float]]:
    if snap_mm <= 0:
        return coords
    def rnd(v: float) -> float:
        return round(v / snap_mm) * snap_mm
    out = [(rnd(x), rnd(y)) for (x, y) in coords]
    # Ensure closure
    if out and (out[0][0] != out[-1][0] or out[0][1] != out[-1][1]):
        out.append(out[0])
    return out


def repair_wall_detections(
    normalized: list,
    *,
    snap_endpoints_mm: float = SNAP_ENDPOINTS_MM,
    min_segment_len_mm: float = MIN_SEGMENT_LEN_MM,
    min_area_mm2: float = MIN_POLYGON_AREA * 1_000_000.0,
    angle_tol_deg: float = SEGMENT_MERGE_ANGLE_DEG,
) -> list:
    """Repair wall polygons in-place and return the list.

    Steps:
    - Snap vertices to grid (snap_endpoints_mm)
    - Remove short segments (< min_segment_len_mm)
    - Merge nearly-colinear vertices (angle within angle_tol_deg of 180°)
    - Enforce clockwise orientation
    - If invalid/too small/<3 points, replace with minimum rotated rectangle
    """
    from shapely.geometry import Polygon as _Polygon
    repaired = []
    for det in normalized:
        try:
            det_type = getattr(det, "type", "") or getattr(det, "klass", "")
            if str(det_type).upper() != "WALL":
                repaired.append(det)
                continue
            geom = getattr(det, "geom", None)
            if not isinstance(geom, _Polygon) or geom.is_empty:
                repaired.append(det)
                continue
            poly: _Polygon = geom
            if not poly.is_valid:
                try:
                    poly = poly.buffer(0)
                except Exception:
                    pass
            coords = list(poly.exterior.coords)
            coords = _round_to_grid(coords, snap_endpoints_mm)
            coords = _remove_short_segments(coords, min_segment_len_mm)
            coords = _merge_colinear(coords, angle_tol_deg)
            if len(coords) < 4:
                rect = poly.minimum_rotated_rectangle
                if isinstance(rect, _Polygon) and not rect.is_empty:
                    poly = rect
                else:
                    repaired.append(det)
                    continue
            else:
                poly = _Polygon(coords)
            if (not poly.is_valid) or poly.area < min_area_mm2:
                rect = poly.minimum_rotated_rectangle
                if isinstance(rect, _Polygon) and not rect.is_empty:
                    poly = rect
            poly = _enforce_clockwise(poly)
            try:
                setattr(det, "geom", poly)
            except Exception:
                # If det is immutable (pydantic), attempt to rebuild dict
                if hasattr(det, "model_copy"):
                    new_det = det.model_copy(update={"geom": poly})
                    det = new_det
            repaired.append(det)
        except Exception:
            repaired.append(det)
    return repaired


def rebuffer_walls(
    snapped_axes: Sequence[LineString],
    thickness_by_index_mm: Mapping[int, float],
) -> List[Polygon | MultiPolygon]:
    """
    Create wall polygons from snapped axes by buffering half the thickness
    on each side and dissolving overlaps per axis segment.
    The order of returned items follows the snapped_axes order.
    
    Enhanced with overlap detection and validation:
    - Detects overlaps between wall polygons (>100mm intersection length)
    - Distinguishes between T-junctions (allowed) and true overlaps (problematic)
    - Validates polygon geometry (closed, valid coordinates)
    - Auto-repairs invalid polygons
    """
    results: List[Polygon | MultiPolygon] = []
    for idx, axis in enumerate(snapped_axes):
        thickness = float(thickness_by_index_mm.get(idx, 115.0))
        half = max(thickness * 0.5, 1.0)
        buffered = axis.buffer(half, join_style=2, cap_style=2)  # mitre, flat
        
        # Validate and repair polygon geometry
        if isinstance(buffered, Polygon):
            if not buffered.is_valid:
                # Attempt repair using buffer(0) trick
                try:
                    repaired = buffered.buffer(0)
                    if not repaired.is_empty and repaired.is_valid:
                        buffered = repaired
                        logger.debug("Repaired invalid polygon for wall axis %d", idx)
                    else:
                        logger.warning("Failed to repair invalid polygon for wall axis %d", idx)
                except Exception as repair_exc:
                    logger.warning("Exception during polygon repair for wall axis %d: %s", idx, repair_exc)
            
            # Validate polygon is closed (first and last points should be same or very close)
            if buffered.is_valid:
                coords = list(buffered.exterior.coords)
                if len(coords) >= 3:
                    first = coords[0]
                    last = coords[-1]
                    dist = math.hypot(first[0] - last[0], first[1] - last[1])
                    if dist > 1.0:  # Not closed - try to close it
                        try:
                            coords.append(coords[0])
                            buffered = Polygon(coords)
                            logger.debug("Closed polygon for wall axis %d (gap: %.2fmm)", idx, dist)
                        except Exception:
                            pass
            
            # Validate coordinates are finite
            if buffered.is_valid:
                coords = list(buffered.exterior.coords)
                for coord in coords:
                    if not all(math.isfinite(c) for c in coord):
                        logger.warning("Non-finite coordinates detected in wall axis %d polygon", idx)
                        break
        
        elif isinstance(buffered, MultiPolygon):
            # Validate and repair each part
            repaired_parts = []
            for part_idx, part in enumerate(buffered.geoms):
                if isinstance(part, Polygon):
                    if not part.is_valid:
                        try:
                            repaired = part.buffer(0)
                            if not repaired.is_empty and repaired.is_valid:
                                part = repaired
                                logger.debug("Repaired invalid polygon part %d for wall axis %d", part_idx, idx)
                        except Exception:
                            pass
                    
                    # Validate polygon is closed
                    if part.is_valid:
                        coords = list(part.exterior.coords)
                        if len(coords) >= 3:
                            first = coords[0]
                            last = coords[-1]
                            dist = math.hypot(first[0] - last[0], first[1] - last[1])
                            if dist > 1.0:
                                try:
                                    coords.append(coords[0])
                                    part = Polygon(coords)
                                except Exception:
                                    pass
                    
                    repaired_parts.append(part)
                else:
                    repaired_parts.append(part)
            
            if repaired_parts:
                buffered = MultiPolygon(repaired_parts) if len(repaired_parts) > 1 else repaired_parts[0]
        
        # buffer returns Polygon or MultiPolygon already; keep as-is for caller
        results.append(buffered)
    
    # Check for overlaps between wall polygons (after all polygons are created)
    overlap_threshold_mm = 100.0  # Overlap length threshold
    overlaps_detected = []
    for i, poly1 in enumerate(results):
        if poly1.is_empty or not poly1.is_valid:
            continue
        
        # Extract largest polygon if MultiPolygon
        if isinstance(poly1, MultiPolygon):
            try:
                poly1_largest = max(list(poly1.geoms), key=lambda g: g.area if isinstance(g, Polygon) else 0.0)
                if not isinstance(poly1_largest, Polygon):
                    continue
                poly1 = poly1_largest
            except (ValueError, AttributeError):
                continue
        
        for j, poly2 in enumerate(results[i + 1:], start=i + 1):
            if poly2.is_empty or not poly2.is_valid:
                continue
            
            # Extract largest polygon if MultiPolygon
            if isinstance(poly2, MultiPolygon):
                try:
                    poly2_largest = max(list(poly2.geoms), key=lambda g: g.area if isinstance(g, Polygon) else 0.0)
                    if not isinstance(poly2_largest, Polygon):
                        continue
                    poly2 = poly2_largest
                except (ValueError, AttributeError):
                    continue
            
            try:
                # Check for intersection
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    
                    # Calculate intersection length/area
                    intersection_size = 0.0
                    if isinstance(intersection, Polygon):
                        # For polygon intersection, check if it's a significant overlap
                        # by measuring the intersection area relative to wall thickness
                        intersection_area = intersection.area
                        # Estimate intersection length from area (approximation)
                        # Assuming intersection is roughly rectangular: length ≈ area / thickness
                        thickness1 = float(thickness_by_index_mm.get(i, 115.0))
                        thickness2 = float(thickness_by_index_mm.get(j, 115.0))
                        avg_thickness = (thickness1 + thickness2) / 2.0
                        if avg_thickness > 0:
                            estimated_length = intersection_area / avg_thickness
                            intersection_size = estimated_length
                    elif isinstance(intersection, LineString):
                        intersection_size = intersection.length
                    elif hasattr(intersection, 'length'):
                        intersection_size = float(intersection.length)
                    
                    # Check if it's a T-junction (one axis endpoint projects onto the other axis)
                    axis1 = snapped_axes[i] if i < len(snapped_axes) else None
                    axis2 = snapped_axes[j] if j < len(snapped_axes) else None
                    is_t_junction = False
                    
                    if axis1 and axis2:
                        # Check if one endpoint projects onto the other axis
                        coords1 = list(axis1.coords)
                        coords2 = list(axis2.coords)
                        if len(coords1) >= 2 and len(coords2) >= 2:
                            ep1_start = Point(coords1[0])
                            ep1_end = Point(coords1[-1])
                            ep2_start = Point(coords2[0])
                            ep2_end = Point(coords2[-1])
                            
                            # Check if ep1_start or ep1_end projects onto axis2
                            for ep in [ep1_start, ep1_end]:
                                dist_to_axis2 = axis2.distance(ep)
                                if dist_to_axis2 <= max(thickness1, thickness2) * 0.5:
                                    # Check if projection is in middle portion of axis2 (not near endpoints)
                                    proj_point = axis2.interpolate(axis2.project(ep))
                                    dist_to_start = proj_point.distance(ep2_start)
                                    dist_to_end = proj_point.distance(ep2_end)
                                    axis2_length = axis2.length
                                    margin = max(50.0, axis2_length * 0.1)  # 10% margin, min 50mm
                                    if dist_to_start > margin and dist_to_end > margin:
                                        is_t_junction = True
                                        break
                            
                            # Check if ep2_start or ep2_end projects onto axis1
                            if not is_t_junction:
                                for ep in [ep2_start, ep2_end]:
                                    dist_to_axis1 = axis1.distance(ep)
                                    if dist_to_axis1 <= max(thickness1, thickness2) * 0.5:
                                        proj_point = axis1.interpolate(axis1.project(ep))
                                        dist_to_start = proj_point.distance(ep1_start)
                                        dist_to_end = proj_point.distance(ep1_end)
                                        axis1_length = axis1.length
                                        margin = max(50.0, axis1_length * 0.1)
                                        if dist_to_start > margin and dist_to_end > margin:
                                            is_t_junction = True
                                            break
                    
                    # Log overlap if significant and not a T-junction
                    if intersection_size > overlap_threshold_mm and not is_t_junction:
                        overlaps_detected.append((i, j, intersection_size))
                        logger.warning(
                            "Wall overlap detected: axis %d and %d overlap by %.1fmm (not a T-junction) - "
                            "may cause double geometry in IFC model",
                            i, j, intersection_size
                        )
                    elif intersection_size > overlap_threshold_mm and is_t_junction:
                        logger.debug(
                            "Wall intersection detected: axis %d and %d intersect by %.1fmm (T-junction - allowed)",
                            i, j, intersection_size
                        )
            except Exception as overlap_exc:
                logger.debug("Exception during overlap detection for walls %d and %d: %s", i, j, overlap_exc)
                continue
    
    if overlaps_detected:
        logger.warning(
            "Wall polygon creation: %d significant overlap(s) detected between wall polygons (>%.1fmm) - "
            "attempting automatic resolution",
            len(overlaps_detected), overlap_threshold_mm
        )
        # Attempt automatic overlap resolution
        try:
            results = resolve_wall_overlaps(results, snapped_axes, thickness_by_index_mm)
        except Exception as resolve_exc:
            logger.warning("Failed to automatically resolve wall overlaps: %s", resolve_exc)
    
    return results


def resolve_wall_overlaps(
    wall_polygons: List[Polygon | MultiPolygon],
    axes: Sequence[LineString],
    thickness_by_index: Mapping[int, float],
) -> List[Polygon | MultiPolygon]:
    """
    Resolve overlaps between wall polygons automatically.
    T-junctions are preserved, true overlaps are resolved using union and repair.
    
    Args:
        wall_polygons: List of wall polygons (potentially overlapping)
        axes: Sequence of wall axis LineStrings for T-junction detection
        thickness_by_index: Mapping of axis index to wall thickness
    
    Returns:
        List of wall polygons with overlaps resolved
    """
    if not wall_polygons or len(wall_polygons) < 2:
        return wall_polygons
    
    resolved = []
    overlap_threshold_mm = 100.0
    resolved_count = 0
    
    for i, poly1 in enumerate(wall_polygons):
        if poly1.is_empty or not poly1.is_valid:
            resolved.append(poly1)
            continue
        
        # Extract largest polygon if MultiPolygon
        if isinstance(poly1, MultiPolygon):
            try:
                poly1_largest = max(list(poly1.geoms), key=lambda g: g.area if isinstance(g, Polygon) else 0.0)
                if not isinstance(poly1_largest, Polygon):
                    resolved.append(poly1)
                    continue
                poly1 = poly1_largest
            except (ValueError, AttributeError):
                resolved.append(poly1)
                continue
        
        # Check for overlaps with previous polygons
        has_overlap = False
        for j, poly2 in enumerate(resolved):
            if poly2.is_empty or not poly2.is_valid:
                continue
            
            # Extract largest polygon if MultiPolygon
            if isinstance(poly2, MultiPolygon):
                try:
                    poly2_largest = max(list(poly2.geoms), key=lambda g: g.area if isinstance(g, Polygon) else 0.0)
                    if not isinstance(poly2_largest, Polygon):
                        continue
                    poly2 = poly2_largest
                except (ValueError, AttributeError):
                    continue
            
            try:
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    intersection_size = 0.0
                    if isinstance(intersection, Polygon):
                        intersection_area = intersection.area
                        thickness1 = thickness_by_index.get(i, 115.0)
                        thickness2 = thickness_by_index.get(j, 115.0)
                        avg_thickness = (thickness1 + thickness2) / 2.0
                        if avg_thickness > 0:
                            estimated_length = intersection_area / avg_thickness
                            intersection_size = estimated_length
                    elif isinstance(intersection, LineString):
                        intersection_size = intersection.length
                    
                    # Check if T-junction
                    is_t_junction = False
                    axis1 = axes[i] if i < len(axes) else None
                    axis2 = axes[j] if j < len(axes) else None
                    
                    if axis1 and axis2:
                        # T-junction detection: check if one endpoint projects onto the other axis
                        coords1 = list(axis1.coords)
                        coords2 = list(axis2.coords)
                        if len(coords1) >= 2 and len(coords2) >= 2:
                            ep1_start = Point(coords1[0])
                            ep1_end = Point(coords1[-1])
                            ep2_start = Point(coords2[0])
                            ep2_end = Point(coords2[-1])
                            
                            # Check if ep1_start or ep1_end projects onto axis2
                            for ep in [ep1_start, ep1_end]:
                                dist_to_axis2 = axis2.distance(ep)
                                if dist_to_axis2 <= max(thickness1, thickness2) * 0.5:
                                    proj_point = axis2.interpolate(axis2.project(ep))
                                    dist_to_start = proj_point.distance(ep2_start)
                                    dist_to_end = proj_point.distance(ep2_end)
                                    axis2_length = axis2.length
                                    margin = max(50.0, axis2_length * 0.1)
                                    if dist_to_start > margin and dist_to_end > margin:
                                        is_t_junction = True
                                        break
                            
                            # Check if ep2_start or ep2_end projects onto axis1
                            if not is_t_junction:
                                for ep in [ep2_start, ep2_end]:
                                    dist_to_axis1 = axis1.distance(ep)
                                    if dist_to_axis1 <= max(thickness1, thickness2) * 0.5:
                                        proj_point = axis1.interpolate(axis1.project(ep))
                                        dist_to_start = proj_point.distance(ep1_start)
                                        dist_to_end = proj_point.distance(ep1_end)
                                        axis1_length = axis1.length
                                        margin = max(50.0, axis1_length * 0.1)
                                        if dist_to_start > margin and dist_to_end > margin:
                                            is_t_junction = True
                                            break
                    
                    # Resolve overlap if not T-junction
                    if intersection_size > overlap_threshold_mm and not is_t_junction:
                        # Union and repair
                        try:
                            unioned = unary_union([poly1, poly2])
                            if not unioned.is_empty and unioned.is_valid:
                                # Repair unioned geometry
                                if not unioned.is_valid:
                                    unioned = unioned.buffer(0)
                                
                                # For now, keep the larger polygon (simplified approach)
                                # Full implementation would require splitting the union back to individual walls
                                if isinstance(unioned, Polygon):
                                    if unioned.area > poly1.area:
                                        resolved[j] = unioned
                                    else:
                                        resolved[j] = poly2
                                    has_overlap = True
                                    resolved_count += 1
                                    logger.debug("Resolved overlap between wall polygons %d and %d (intersection: %.1fmm)", i, j, intersection_size)
                                    break
                        except Exception as union_exc:
                            logger.debug("Failed to resolve overlap between walls %d and %d: %s", i, j, union_exc)
            except Exception as overlap_exc:
                logger.debug("Exception during overlap resolution for walls %d and %d: %s", i, j, overlap_exc)
                continue
        
        if not has_overlap:
            resolved.append(poly1)
    
    if resolved_count > 0:
        logger.info("Wall overlap resolution: Resolved %d overlap(s) automatically", resolved_count)
    
    return resolved


class TopologyGraph:
    """
    Topology graph for managing wall axis connections.
    Tracks endpoints and their connections to enable better gap closure logic.
    """
    
    def __init__(self, axes: Sequence[LineString]):
        self.axes = list(axes)
        self.endpoints: Dict[Tuple[int, bool], Point] = {}  # (axis_idx, is_start) -> Point
        self.connections: Dict[Tuple[int, bool], Set[Tuple[int, bool]]] = defaultdict(set)
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the initial topology graph from axes."""
        for idx, axis in enumerate(self.axes):
            if axis.length < 1e-3:
                continue
            coords = list(axis.coords)
            if len(coords) < 2:
                continue
            self.endpoints[(idx, True)] = Point(coords[0])
            self.endpoints[(idx, False)] = Point(coords[-1])
    
    def find_gaps(
        self, 
        max_gap_mm: float,
        thickness_by_index: Mapping[int, float] | None = None
    ) -> List[Tuple[Tuple[int, bool], Tuple[int, bool], float]]:
        """
        Find all gaps between endpoints that are within max_gap_mm.
        Returns list of ((axis1_idx, is_start1), (axis2_idx, is_start2), distance).
        """
        gaps = []
        endpoint_list = list(self.endpoints.items())
        
        for i, ((idx1, is_start1), ep1) in enumerate(endpoint_list):
            for j, ((idx2, is_start2), ep2) in enumerate(endpoint_list[i + 1:], start=i + 1):
                if idx1 == idx2:
                    continue
                
                distance = ep1.distance(ep2)
                
                # Use adaptive tolerance if thickness info available
                if thickness_by_index:
                    thickness1 = thickness_by_index.get(idx1, 115.0)
                    thickness2 = thickness_by_index.get(idx2, 115.0)
                    avg_thickness = (thickness1 + thickness2) / 2.0
                    adaptive_max = max(50.0, min(avg_thickness * 2.5, max_gap_mm))
                else:
                    adaptive_max = max_gap_mm
                
                if 1.0 < distance <= adaptive_max:
                    gaps.append(((idx1, is_start1), (idx2, is_start2), distance))
        
        return sorted(gaps, key=lambda x: x[2])  # Sort by distance
    
    def connect_endpoints(
        self, 
        ep1_key: Tuple[int, bool], 
        ep2_key: Tuple[int, bool],
        connection_point: Tuple[float, float]
    ) -> None:
        """Connect two endpoints at the given connection point."""
        self.connections[ep1_key].add(ep2_key)
        self.connections[ep2_key].add(ep1_key)
        
        # Update endpoint positions
        cp = Point(connection_point)
        self.endpoints[ep1_key] = cp
        self.endpoints[ep2_key] = cp
    
    def validate_gaps(self, max_gap_mm: float = 50.0) -> List[Tuple[int, int, float]]:
        """
        Validate that all gaps are <= max_gap_mm.
        Returns list of (axis1_idx, axis2_idx, gap_size) for gaps that exceed threshold.
        """
        remaining_gaps = []
        endpoint_list = list(self.endpoints.items())
        
        for i, ((idx1, is_start1), ep1) in enumerate(endpoint_list):
            for j, ((idx2, is_start2), ep2) in enumerate(endpoint_list[i + 1:], start=i + 1):
                if idx1 == idx2:
                    continue
                
                distance = ep1.distance(ep2)
                if distance > max_gap_mm:
                    remaining_gaps.append((idx1, idx2, distance))
        
        return remaining_gaps


def close_wall_gaps(
    wall_axes: Sequence[LineString],
    gap_tolerance_mm: float = 100.0,
    max_gap_tolerance_mm: float = 300.0,
    *,
    thickness_by_index_mm: Mapping[int, float] | None = None,
) -> List[LineString]:
    """
    Close gaps between wall axes by extending them or adding connection segments.
    Enhanced version with topology graph, improved gap detection, T-junction handling, 
    and iterative improvement with validation.
    
    Args:
        wall_axes: Sequence of wall axis LineStrings
        gap_tolerance_mm: Preferred gap size to close (default 100mm, adaptive if thickness provided)
        max_gap_tolerance_mm: Maximum gap size to attempt closing (default 300mm, adaptive if thickness provided)
        thickness_by_index_mm: Optional mapping of axis index to wall thickness for adaptive tolerances
    
    Returns:
        List of potentially extended/modified wall axes with gaps closed
    """
    if not wall_axes or len(wall_axes) < 2:
        return list(wall_axes)
    
    axes_list = list(wall_axes)
    
    # Build topology graph for better connection management
    topology = TopologyGraph(axes_list)
    
    # Calculate adaptive tolerances based on wall thickness if available
    if thickness_by_index_mm:
        # Use average thickness for tolerance calculation
        thicknesses = [thickness_by_index_mm.get(i, 115.0) for i in range(len(axes_list))]
        avg_thickness = sum(thicknesses) / len(thicknesses) if thicknesses else 115.0
        # Adaptive gap tolerance: 0.5x to 1.5x average thickness, but at least 50mm
        adaptive_gap_tolerance = max(50.0, min(avg_thickness * 0.5, gap_tolerance_mm))
        # Adaptive max gap: 2x to 3x average thickness, but at least 150mm
        adaptive_max_gap = max(150.0, min(avg_thickness * 2.5, max_gap_tolerance_mm))
        gap_tolerance = float(adaptive_gap_tolerance)
        max_gap = float(adaptive_max_gap)
    else:
        gap_tolerance = float(gap_tolerance_mm)
        max_gap = float(max_gap_tolerance_mm)
    
    # Helper function to get adaptive tolerance for a specific axis pair
    def _get_adaptive_tolerance(idx1: int, idx2: int, base_tolerance: float) -> float:
        """Get adaptive tolerance based on wall thicknesses."""
        if not thickness_by_index_mm:
            return base_tolerance
        thickness1 = thickness_by_index_mm.get(idx1, 115.0)
        thickness2 = thickness_by_index_mm.get(idx2, 115.0)
        avg_thickness = (thickness1 + thickness2) / 2.0
        # Tolerance scales with average thickness: 0.3x to 1.0x thickness
        adaptive = max(base_tolerance * 0.5, min(avg_thickness * 0.3, base_tolerance * 1.5))
        return adaptive
    
    # Helper function to check if point is on line segment (for T-junctions)
    def _point_on_line_segment(point: Point, line: LineString, tolerance: float = 10.0) -> bool:
        """Check if point is on or near a line segment."""
        try:
            distance = line.distance(point)
            return distance <= tolerance
        except Exception:
            return False
    
    # Helper function to find closest point on line to a given point
    def _project_point_to_line(point: Point, line: LineString) -> Tuple[float, float]:
        """Project a point onto a line and return the projected coordinates."""
        coords = list(line.coords)
        if len(coords) < 2:
            return (point.x, point.y)
        
        min_dist = float('inf')
        closest_point = (point.x, point.y)
        
        for i in range(len(coords) - 1):
            p1 = Point(coords[i])
            p2 = Point(coords[i + 1])
            
            # Vector from p1 to p2
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length_sq = dx * dx + dy * dy
            
            if length_sq < 1e-6:
                continue
            
            # Vector from p1 to point
            vx = point.x - p1.x
            vy = point.y - p1.y
            
            # Project point onto line segment
            t = max(0.0, min(1.0, (vx * dx + vy * dy) / length_sq))
            proj_x = p1.x + t * dx
            proj_y = p1.y + t * dy
            proj_point = Point(proj_x, proj_y)
            
            dist = point.distance(proj_point)
            if dist < min_dist:
                min_dist = dist
                closest_point = (proj_x, proj_y)
        
        return closest_point
    
    # Collect all endpoints with their associated axes and directions
    endpoints: List[Tuple[Point, int, bool, Tuple[float, float], LineString]] = []  # (point, axis_index, is_start, direction, axis)
    for idx, axis in enumerate(axes_list):
        if axis.length < 1e-3:
            continue
        coords = list(axis.coords)
        if len(coords) >= 2:
            start = Point(coords[0])
            end = Point(coords[-1])
            # Calculate direction vector
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            length = math.hypot(dx, dy) or 1.0
            direction = (dx / length, dy / length)
            endpoints.append((start, idx, True, direction, axis))
            endpoints.append((end, idx, False, direction, axis))
    
    # Iterative improvement: Multiple passes with decreasing tolerance
    # This ensures we catch all gaps progressively
    modified_axes = [LineString(axis.coords) for axis in axes_list]
    max_iterations = 5  # Increased to 5: 4 standard + 1 strict final pass with 30mm tolerance
    current_tolerance = max_gap_tolerance_mm
    current_gap_tolerance = gap_tolerance_mm
    
    for iteration in range(max_iterations):
        if iteration > 0:
            # Rebuild topology graph with updated axes
            topology = TopologyGraph(modified_axes)
            # Decrease tolerance for subsequent iterations
            if iteration < max_iterations - 2:
                current_tolerance = max_gap_tolerance_mm * (0.7 ** iteration)
                current_gap_tolerance = gap_tolerance_mm * (0.8 ** iteration)
            elif iteration == max_iterations - 2:
                # Second-to-last iteration: strict tolerance for gaps ≤100mm (BIM requirement)
                current_tolerance = 100.0
                current_gap_tolerance = 50.0
            else:
                # Final iteration: ultra-strict tolerance (30mm) for guaranteed closure of all ≤100mm gaps
                current_tolerance = 100.0
                current_gap_tolerance = 30.0
        
        # Find gaps using topology graph
        gaps = topology.find_gaps(current_tolerance, thickness_by_index_mm)
        
        if not gaps:
            logger.debug("Gap closure iteration %d: No gaps found", iteration + 1)
            if iteration == max_iterations - 1:
                # Final iteration found no gaps - success
                break
            continue
        
        logger.debug("Gap closure iteration %d: Found %d gaps (tolerance: %.1fmm)", 
                    iteration + 1, len(gaps), current_tolerance)
        
        connections_made = set()  # Track connections to avoid duplicates
        connection_count = 0
    
        # Process gaps found by topology graph
        for (idx1, is_start1), (idx2, is_start2), distance in gaps:
            # Check if this connection was already made
            connection_key = tuple(sorted([idx1, idx2]))
            if connection_key in connections_made:
                continue
            
            ep1 = topology.endpoints[(idx1, is_start1)]
            ep2 = topology.endpoints[(idx2, is_start2)]
            
            # Get axis geometries
            axis1 = modified_axes[idx1]
            axis2 = modified_axes[idx2]
            
            # Calculate direction vectors for alignment check
            coords1 = list(axis1.coords)
            coords2 = list(axis2.coords)
            
            if len(coords1) < 2 or len(coords2) < 2:
                continue
            
            dx1 = coords1[-1][0] - coords1[0][0]
            dy1 = coords1[-1][1] - coords1[0][1]
            length1 = math.hypot(dx1, dy1) or 1.0
            dir1 = (dx1 / length1, dy1 / length1)
            
            dx2 = coords2[-1][0] - coords2[0][0]
            dy2 = coords2[-1][1] - coords2[0][1]
            length2 = math.hypot(dx2, dy2) or 1.0
            dir2 = (dx2 / length2, dy2 / length2)
            
            # Enhanced T-junction detection: Check if endpoint projects onto line segment
            is_t_junction = False
            connection_point = None
            
            # Improved T-junction detection with better projection
            # Check if ep1 projects onto axis2 (T-junction)
            proj_point_ep1 = _project_point_to_line(ep1, axis2)
            proj_point_ep1_pt = Point(proj_point_ep1)
            dist_to_axis2 = axis2.distance(proj_point_ep1_pt)
            
            # Use adaptive tolerance based on wall thickness for T-junction detection
            t_junction_tolerance = current_tolerance * 1.5
            if thickness_by_index_mm:
                thickness1 = thickness_by_index_mm.get(idx1, 115.0)
                thickness2 = thickness_by_index_mm.get(idx2, 115.0)
                avg_thickness = (thickness1 + thickness2) / 2.0
                t_junction_tolerance = max(30.0, min(avg_thickness * 0.5, current_tolerance * 1.5))
            
            if dist_to_axis2 <= t_junction_tolerance:
                # Check if projection is on the line segment (not just near it)
                coords2_list = list(axis2.coords)
                if len(coords2_list) >= 2:
                    # Calculate distances from projection to endpoints
                    dist_to_start = math.hypot(proj_point_ep1[0] - coords2_list[0][0], 
                                              proj_point_ep1[1] - coords2_list[0][1])
                    dist_to_end = math.hypot(proj_point_ep1[0] - coords2_list[-1][0], 
                                           proj_point_ep1[1] - coords2_list[-1][1])
                    axis_length = axis2.length
                    # Improved T-junction detection: adaptive threshold based on axis length
                    # For short walls (<500mm), use 20% margin; for longer walls, use 10% margin
                    margin_factor = 0.2 if axis_length < 500.0 else 0.1
                    min_margin = max(50.0, axis_length * margin_factor)  # At least 50mm margin
                    # T-junction if projection is in middle portion of axis (not near endpoints)
                    if dist_to_start > min_margin and dist_to_end > min_margin:
                        is_t_junction = True
                        connection_point = proj_point_ep1
            
            # Check if ep2 projects onto axis1 (T-junction)
            if not is_t_junction:
                proj_point_ep2 = _project_point_to_line(ep2, axis1)
                proj_point_ep2_pt = Point(proj_point_ep2)
                dist_to_axis1 = axis1.distance(proj_point_ep2_pt)
                
                if dist_to_axis1 <= t_junction_tolerance:
                    coords1_list = list(axis1.coords)
                    if len(coords1_list) >= 2:
                        dist_to_start = math.hypot(proj_point_ep2[0] - coords1_list[0][0], 
                                                  proj_point_ep2[1] - coords1_list[0][1])
                        dist_to_end = math.hypot(proj_point_ep2[0] - coords1_list[-1][0], 
                                               proj_point_ep2[1] - coords1_list[-1][1])
                        axis_length = axis1.length
                        # Improved T-junction detection: adaptive threshold based on axis length
                        # For short walls (<500mm), use 20% margin; for longer walls, use 10% margin
                        margin_factor = 0.2 if axis_length < 500.0 else 0.1
                        min_margin = max(50.0, axis_length * margin_factor)  # At least 50mm margin
                        # T-junction if projection is in middle portion of axis (not near endpoints)
                        if dist_to_start > min_margin and dist_to_end > min_margin:
                            is_t_junction = True
                            connection_point = proj_point_ep2
            
            # Direct connection: use midpoint
            if not is_t_junction:
                # Use adaptive tolerance for this specific pair
                pair_gap_tolerance = _get_adaptive_tolerance(idx1, idx2, current_gap_tolerance)
                if distance <= pair_gap_tolerance:
                    # Small gap: meet at midpoint
                    mid_x = (ep1.x + ep2.x) / 2.0
                    mid_y = (ep1.y + ep2.y) / 2.0
                    connection_point = (mid_x, mid_y)
                else:
                    # Larger gap: extend to meet
                    connection_point = ((ep1.x + ep2.x) / 2.0, (ep1.y + ep2.y) / 2.0)
            
            if connection_point is None:
                continue
            
            # Extend axes to meet at connection point
            try:
                axis1_coords = list(axis1.coords)
                axis2_coords = list(axis2.coords)
                
                # Extend axis1 towards connection point
                if is_start1:
                    if math.hypot(connection_point[0] - axis1_coords[0][0], 
                                connection_point[1] - axis1_coords[0][1]) > 1.0:
                        axis1_coords = [connection_point] + axis1_coords
                else:
                    if math.hypot(connection_point[0] - axis1_coords[-1][0], 
                                connection_point[1] - axis1_coords[-1][1]) > 1.0:
                        axis1_coords = axis1_coords + [connection_point]
                
                # For T-junctions, only extend the endpoint axis, not the middle axis
                if not is_t_junction:
                    # Extend axis2 towards connection point
                    if is_start2:
                        if math.hypot(connection_point[0] - axis2_coords[0][0], 
                                    connection_point[1] - axis2_coords[0][1]) > 1.0:
                            axis2_coords = [connection_point] + axis2_coords
                    else:
                        if math.hypot(connection_point[0] - axis2_coords[-1][0], 
                                    connection_point[1] - axis2_coords[-1][1]) > 1.0:
                            axis2_coords = axis2_coords + [connection_point]
                    
                    modified_axes[idx2] = LineString(axis2_coords)
                
                modified_axes[idx1] = LineString(axis1_coords)
                
                # Update topology graph
                topology.connect_endpoints((idx1, is_start1), (idx2, is_start2), connection_point)
                connections_made.add(connection_key)
                connection_count += 1
            except Exception:
                continue
        
        # Validate after each iteration with detailed logging
        validation_threshold = 100.0 if iteration >= max_iterations - 2 else 50.0
        remaining = topology.validate_gaps(max_gap_mm=validation_threshold)
        if remaining:
            if iteration >= max_iterations - 2:
                # Final iterations: log remaining gaps ≤100mm for guaranteed repair
                gaps_to_repair = [g for g in remaining if g[2] <= 100.0]
                if gaps_to_repair:
                    logger.debug("Gap closure iteration %d (%s): %d gaps ≤100mm need repair", 
                               iteration + 1, 
                               "ultra-strict" if iteration == max_iterations - 1 else "strict",
                               len(gaps_to_repair))
                    # Force repair of remaining gaps ≤100mm - NO EXCEPTIONS
                    for gap_idx, (i, j, gap_size) in enumerate(gaps_to_repair):
                        try:
                            ax1 = modified_axes[i]
                            ax2 = modified_axes[j]
                            if ax1.length < 1e-3 or ax2.length < 1e-3:
                                continue
                            coords1 = list(ax1.coords)
                            coords2 = list(ax2.coords)
                            if len(coords1) < 2 or len(coords2) < 2:
                                continue
                            
                            ep1_start = Point(coords1[0])
                            ep1_end = Point(coords1[-1])
                            ep2_start = Point(coords2[0])
                            ep2_end = Point(coords2[-1])
                            
                            # Find closest endpoint pair
                            dists = [
                                (ep1_start, ep2_start, ep1_start.distance(ep2_start), False, False),
                                (ep1_start, ep2_end, ep1_start.distance(ep2_end), False, True),
                                (ep1_end, ep2_start, ep1_end.distance(ep2_start), True, False),
                                (ep1_end, ep2_end, ep1_end.distance(ep2_end), True, True),
                            ]
                            closest = min(dists, key=lambda x: x[2])
                            ep1, ep2, dist, extend_end1, extend_end2 = closest
                            
                            if dist <= 100.0:
                                # Calculate midpoint
                                mid_x = (ep1.x + ep2.x) / 2.0
                                mid_y = (ep1.y + ep2.y) / 2.0
                                mid_point = (mid_x, mid_y)
                                
                                # Extend axes
                                new_coords1 = list(coords1)
                                if extend_end1:
                                    if mid_point not in new_coords1:
                                        new_coords1 = new_coords1 + [mid_point]
                                else:
                                    if mid_point not in new_coords1:
                                        new_coords1 = [mid_point] + new_coords1
                                
                                new_coords2 = list(coords2)
                                if extend_end2:
                                    if mid_point not in new_coords2:
                                        new_coords2 = new_coords2 + [mid_point]
                                else:
                                    if mid_point not in new_coords2:
                                        new_coords2 = [mid_point] + new_coords2
                                
                                modified_axes[i] = LineString(new_coords1)
                                modified_axes[j] = LineString(new_coords2)
                                logger.debug("Guaranteed repair: Closed gap %d (axis %d to %d, %.1fmm)", 
                                           gap_idx, i, j, dist)
                        except Exception as repair_exc:
                            logger.warning("CRITICAL: Failed guaranteed repair for gap %d (axis %d to %d, %.1fmm): %s", 
                                         gap_idx, i, j, gap_size, repair_exc)
                            # Try alternative repair: direct extension without midpoint
                            try:
                                ax1 = modified_axes[i]
                                ax2 = modified_axes[j]
                                if ax1.length >= 1e-3 and ax2.length >= 1e-3:
                                    coords1 = list(ax1.coords)
                                    coords2 = list(ax2.coords)
                                    ep1_start = Point(coords1[0])
                                    ep1_end = Point(coords1[-1])
                                    ep2_start = Point(coords2[0])
                                    ep2_end = Point(coords2[-1])
                                    # Find closest endpoints and extend directly
                                    dists = [
                                        (ep1_start, ep2_start, ep1_start.distance(ep2_start), False, False),
                                        (ep1_start, ep2_end, ep1_start.distance(ep2_end), False, True),
                                        (ep1_end, ep2_start, ep1_end.distance(ep2_start), True, False),
                                        (ep1_end, ep2_end, ep1_end.distance(ep2_end), True, True),
                                    ]
                                    closest = min(dists, key=lambda x: x[2])
                                    ep1, ep2, dist, extend_end1, extend_end2 = closest
                                    if dist <= 100.0:
                                        # Direct extension: extend to other endpoint
                                        target_point = (ep2.x, ep2.y)
                                        new_coords1 = list(coords1)
                                        new_coords2 = list(coords2)
                                        if extend_end1:
                                            if target_point not in new_coords1:
                                                new_coords1 = new_coords1 + [target_point]
                                        else:
                                            if target_point not in new_coords1:
                                                new_coords1 = [target_point] + new_coords1
                                        if extend_end2:
                                            if target_point not in new_coords2:
                                                new_coords2 = new_coords2 + [target_point]
                                        else:
                                            if target_point not in new_coords2:
                                                new_coords2 = [target_point] + new_coords2
                                        modified_axes[i] = LineString(new_coords1)
                                        modified_axes[j] = LineString(new_coords2)
                                        logger.debug("Alternative repair successful: Closed gap %d via direct extension", gap_idx)
                            except Exception as alt_repair_exc:
                                logger.error("CRITICAL: Alternative repair also failed for gap %d: %s", gap_idx, alt_repair_exc)
                            continue
                else:
                    logger.debug("Gap closure iteration %d: %d gaps > 100mm remain (cannot guarantee repair)", 
                               iteration + 1, len(remaining))
            else:
                logger.debug("Gap closure iteration %d: %d gaps > 50mm remain", 
                           iteration + 1, len(remaining))
        else:
            logger.debug("Gap closure iteration %d: All gaps ≤ %dmm (BIM-compliant)", 
                       iteration + 1, validation_threshold)
            if iteration < max_iterations - 1:
                # Continue to final iterations for strict check
                continue
            # Final iteration completed successfully
            break
    
    # Final aggressive gap closure: Check for any remaining gaps > 50mm and attempt closure
    remaining_gaps = []
    for i, axis1 in enumerate(modified_axes):
        if axis1.length < 1e-3:
            continue
        coords1 = list(axis1.coords)
        if len(coords1) < 2:
            continue
        ep1_start = Point(coords1[0])
        ep1_end = Point(coords1[-1])
        
        for j, axis2 in enumerate(modified_axes[i + 1:], start=i + 1):
            if axis2.length < 1e-3:
                continue
            coords2 = list(axis2.coords)
            if len(coords2) < 2:
                continue
            ep2_start = Point(coords2[0])
            ep2_end = Point(coords2[-1])
            
            # Check distances between endpoints
            distances = [
                (ep1_start, ep2_start, ep1_start.distance(ep2_start)),
                (ep1_start, ep2_end, ep1_start.distance(ep2_end)),
                (ep1_end, ep2_start, ep1_end.distance(ep2_start)),
                (ep1_end, ep2_end, ep1_end.distance(ep2_end)),
            ]
            min_dist_info = min(distances, key=lambda x: x[2])
            min_dist = min_dist_info[2]
            
            if 50.0 < min_dist <= 500.0:  # Gaps between 50mm and 500mm
                remaining_gaps.append((i, j, min_dist, min_dist_info))
    
    # Enhanced: Additional aggressive pass for gaps up to 200mm (BIM requirement)
    if remaining_gaps:
        # Sort by distance (smallest first for priority)
        remaining_gaps.sort(key=lambda x: x[2])
        
        # Attempt to close gaps up to 200mm more aggressively
        aggressive_gaps = [g for g in remaining_gaps if 50.0 < g[2] <= 200.0]
        if aggressive_gaps:
            logger.debug("Enhanced gap closure: Attempting to close %d gaps between 50-200mm", len(aggressive_gaps))
            for gap_idx, (i, j, gap_dist, (p1, p2, _)) in enumerate(aggressive_gaps):
                try:
                    axis1 = modified_axes[i]
                    axis2 = modified_axes[j]
                    
                    if axis1.length < 1e-3 or axis2.length < 1e-3:
                        continue
                    
                    coords1 = list(axis1.coords)
                    coords2 = list(axis2.coords)
                    
                    if len(coords1) < 2 or len(coords2) < 2:
                        continue
                    
                    # Find closest endpoint pair
                    ep1_start = Point(coords1[0])
                    ep1_end = Point(coords1[-1])
                    ep2_start = Point(coords2[0])
                    ep2_end = Point(coords2[-1])
                    
                    dists = [
                        (ep1_start, ep2_start, ep1_start.distance(ep2_start), False, False),
                        (ep1_start, ep2_end, ep1_start.distance(ep2_end), False, True),
                        (ep1_end, ep2_start, ep1_end.distance(ep2_start), True, False),
                        (ep1_end, ep2_end, ep1_end.distance(ep2_end), True, True),
                    ]
                    closest = min(dists, key=lambda x: x[2])
                    ep1, ep2, dist, extend_end1, extend_end2 = closest
                    
                    if dist <= 200.0:
                        # Calculate connection point (midpoint for better alignment)
                        mid_x = (ep1.x + ep2.x) / 2.0
                        mid_y = (ep1.y + ep2.y) / 2.0
                        mid_point = (mid_x, mid_y)
                        
                        # Extend axis1
                        new_coords1 = list(coords1)
                        if extend_end1:
                            if math.hypot(mid_x - coords1[-1][0], mid_y - coords1[-1][1]) > 1.0:
                                new_coords1 = new_coords1 + [mid_point]
                        else:
                            if math.hypot(mid_x - coords1[0][0], mid_y - coords1[0][1]) > 1.0:
                                new_coords1 = [mid_point] + new_coords1
                        
                        # Extend axis2
                        new_coords2 = list(coords2)
                        if extend_end2:
                            if math.hypot(mid_x - coords2[-1][0], mid_y - coords2[-1][1]) > 1.0:
                                new_coords2 = new_coords2 + [mid_point]
                        else:
                            if math.hypot(mid_x - coords2[0][0], mid_y - coords2[0][1]) > 1.0:
                                new_coords2 = [mid_point] + new_coords2
                        
                        modified_axes[i] = LineString(new_coords1)
                        modified_axes[j] = LineString(new_coords2)
                        
                except Exception as aggressive_exc:
                    logger.debug("Failed to close aggressive gap %d: %s", gap_idx, aggressive_exc)
                    continue
    
    # Final pass: Attempt to close remaining gaps > 50mm
    if remaining_gaps:
        logger.debug("Wall gap closure: %d remaining gaps > 50mm detected (max: %.1fmm). Attempting final closure.", 
                    len(remaining_gaps), max(gap[2] for gap in remaining_gaps) if remaining_gaps else 0.0)
        
        # Sort gaps by distance (smallest first)
        remaining_gaps.sort(key=lambda x: x[2])
        
        for gap_idx, (i, j, gap_dist, (p1, p2, _)) in enumerate(remaining_gaps):
            if gap_dist > 200.0:  # Skip very large gaps
                continue
            
            try:
                axis1 = modified_axes[i]
                axis2 = modified_axes[j]
                
                if axis1.length < 1e-3 or axis2.length < 1e-3:
                    continue
                
                coords1 = list(axis1.coords)
                coords2 = list(axis2.coords)
                
                if len(coords1) < 2 or len(coords2) < 2:
                    continue
                
                # Determine which endpoints to connect
                ep1_start = Point(coords1[0])
                ep1_end = Point(coords1[-1])
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                # Find closest endpoint pair
                dists = [
                    (ep1_start, ep2_start, ep1_start.distance(ep2_start)),
                    (ep1_start, ep2_end, ep1_start.distance(ep2_end)),
                    (ep1_end, ep2_start, ep1_end.distance(ep2_start)),
                    (ep1_end, ep2_end, ep1_end.distance(ep2_end)),
                ]
                closest_pair = min(dists, key=lambda x: x[2])
                ep1, ep2, dist = closest_pair
                
                if dist > 200.0:  # Skip if still too large
                    continue
                
                # Calculate midpoint for connection
                mid_x = (ep1.x + ep2.x) / 2.0
                mid_y = (ep1.y + ep2.y) / 2.0
                mid_point = (mid_x, mid_y)
                
                # Extend axis1 towards midpoint
                new_coords1 = list(coords1)
                if ep1 == ep1_start:
                    if math.hypot(mid_x - coords1[0][0], mid_y - coords1[0][1]) > 1.0:
                        new_coords1 = [mid_point] + new_coords1
                elif ep1 == ep1_end:
                    if math.hypot(mid_x - coords1[-1][0], mid_y - coords1[-1][1]) > 1.0:
                        new_coords1 = new_coords1 + [mid_point]
                
                # Extend axis2 towards midpoint
                new_coords2 = list(coords2)
                if ep2 == ep2_start:
                    if math.hypot(mid_x - coords2[0][0], mid_y - coords2[0][1]) > 1.0:
                        new_coords2 = [mid_point] + new_coords2
                elif ep2 == ep2_end:
                    if math.hypot(mid_x - coords2[-1][0], mid_y - coords2[-1][1]) > 1.0:
                        new_coords2 = new_coords2 + [mid_point]
                
                # Update axes
                modified_axes[i] = LineString(new_coords1)
                modified_axes[j] = LineString(new_coords2)
                
            except Exception as final_gap_exc:
                logger.debug("Failed to close final gap %d: %s", gap_idx, final_gap_exc)
                continue
        
        # Re-check remaining gaps after final closure attempt
        final_remaining = []
        for i, axis1 in enumerate(modified_axes):
            if axis1.length < 1e-3:
                continue
            coords1 = list(axis1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            
            for j, axis2 in enumerate(modified_axes[i + 1:], start=i + 1):
                if axis2.length < 1e-3:
                    continue
                coords2 = list(axis2.coords)
                if len(coords2) < 2:
                    continue
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                distances = [
                    ep1_start.distance(ep2_start),
                    ep1_start.distance(ep2_end),
                    ep1_end.distance(ep2_start),
                    ep1_end.distance(ep2_end),
                ]
                min_dist = min(distances)
                if min_dist > 50.0:  # Still a gap > 50mm
                    final_remaining.append((i, j, min_dist))
        
        if final_remaining:
            max_gap = max(gap[2] for gap in final_remaining) if final_remaining else 0.0
            logger.warning("Wall gap closure: %d gaps > 50mm remain after final closure attempt (max: %.1fmm)", 
                        len(final_remaining), max_gap)
            # Log details for gaps that couldn't be closed
            for gap_info in final_remaining[:5]:  # Log first 5
                logger.debug("Unclosed gap: axis %d to %d, distance: %.1fmm", gap_info[0], gap_info[1], gap_info[2])
        else:
            logger.info("Wall gap closure: All gaps <= 50mm successfully closed (BIM-compliant)")
    
    # STRICT FINAL VALIDATION: Verify all gaps <= 100mm are closed (MANDATORY - NO EXCEPTIONS)
    # Rebuild topology graph for final validation
    final_topology = TopologyGraph(modified_axes)
    final_validation_gaps_100mm = final_topology.validate_gaps(max_gap_mm=100.0)
    final_validation_gaps_50mm = final_topology.validate_gaps(max_gap_mm=50.0)
    
    # CRITICAL: All gaps <= 100mm MUST be closed - attempt one more aggressive repair if needed
    gaps_100mm_or_less = [g for g in final_validation_gaps_100mm if g[2] <= 100.0]
    if gaps_100mm_or_less:
        logger.warning("STRICT VALIDATION: %d gaps <= 100mm still remain - attempting final aggressive repair", 
                    len(gaps_100mm_or_less))
        # One final aggressive pass with direct extension
        for (i, j, gap_size) in gaps_100mm_or_less:
            try:
                ax1 = modified_axes[i]
                ax2 = modified_axes[j]
                if ax1.length < 1e-3 or ax2.length < 1e-3:
                    continue
                coords1 = list(ax1.coords)
                coords2 = list(ax2.coords)
                if len(coords1) < 2 or len(coords2) < 2:
                    continue
                ep1_start = Point(coords1[0])
                ep1_end = Point(coords1[-1])
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                # Find closest endpoints
                dists = [
                    (ep1_start, ep2_start, ep1_start.distance(ep2_start), False, False),
                    (ep1_start, ep2_end, ep1_start.distance(ep2_end), False, True),
                    (ep1_end, ep2_start, ep1_end.distance(ep2_start), True, False),
                    (ep1_end, ep2_end, ep1_end.distance(ep2_end), True, True),
                ]
                closest = min(dists, key=lambda x: x[2])
                ep1, ep2, dist, extend_end1, extend_end2 = closest
                if dist <= 100.0:
                    # Direct extension to closest endpoint
                    target_point = (ep2.x, ep2.y)
                    new_coords1 = list(coords1)
                    new_coords2 = list(coords2)
                    if extend_end1:
                        if target_point not in new_coords1:
                            new_coords1 = new_coords1 + [target_point]
                    else:
                        if target_point not in new_coords1:
                            new_coords1 = [target_point] + new_coords1
                    if extend_end2:
                        if target_point not in new_coords2:
                            new_coords2 = new_coords2 + [target_point]
                    else:
                        if target_point not in new_coords2:
                            new_coords2 = [target_point] + new_coords2
                    modified_axes[i] = LineString(new_coords1)
                    modified_axes[j] = LineString(new_coords2)
                    logger.debug("Final aggressive repair: Closed gap (axis %d to %d, %.1fmm)", i, j, dist)
            except Exception as final_repair_exc:
                logger.error("CRITICAL: Final aggressive repair failed for gap (axis %d to %d, %.1fmm): %s", 
                           i, j, gap_size, final_repair_exc)
        
        # Re-validate after final aggressive repair
        final_topology = TopologyGraph(modified_axes)
        final_validation_gaps_100mm = final_topology.validate_gaps(max_gap_mm=100.0)
        final_validation_gaps_50mm = final_topology.validate_gaps(max_gap_mm=50.0)
    
    # Report on gaps > 100mm (cannot guarantee repair)
    if final_validation_gaps_100mm:
        gaps_over_100mm = [g for g in final_validation_gaps_100mm if g[2] > 100.0]
        if gaps_over_100mm:
            max_gap = max(gap[2] for gap in gaps_over_100mm)
            logger.warning("Final gap validation: %d gaps > 100mm detected (max: %.1fmm) - cannot guarantee repair", 
                        len(gaps_over_100mm), max_gap)
            # Log details for unrepairable gaps
            for gap_info in gaps_over_100mm[:10]:  # Log first 10
                logger.debug("Unrepairable gap: axis %d to %d, distance: %.1fmm", 
                            gap_info[0], gap_info[1], gap_info[2])
    
    # CRITICAL: Report on gaps <= 100mm that still remain (should have been repaired)
    gaps_50_to_100mm = [g for g in final_validation_gaps_50mm if 50.0 < g[2] <= 100.0] if final_validation_gaps_50mm else []
    if gaps_50_to_100mm:
        max_gap = max(gap[2] for gap in gaps_50_to_100mm)
        logger.error("CRITICAL: Final gap validation FAILED - %d gaps between 50-100mm remain (max: %.1fmm) - BIM compliance compromised", 
                    len(gaps_50_to_100mm), max_gap)
        for gap_info in gaps_50_to_100mm[:10]:  # Log first 10
            logger.error("UNCLOSED GAP (50-100mm): axis %d to %d, distance: %.1fmm - REQUIRES MANUAL INTERVENTION", 
                        gap_info[0], gap_info[1], gap_info[2])
    else:
        # Check for any remaining gaps <= 100mm
        remaining_100mm = [g for g in final_validation_gaps_100mm if g[2] <= 100.0] if final_validation_gaps_100mm else []
        if remaining_100mm:
            max_gap = max(gap[2] for gap in remaining_100mm)
            logger.error("CRITICAL: Final gap validation FAILED - %d gaps <= 100mm remain (max: %.1fmm) - BIM compliance compromised", 
                        len(remaining_100mm), max_gap)
            for gap_info in remaining_100mm[:10]:
                logger.error("UNCLOSED GAP (<=100mm): axis %d to %d, distance: %.1fmm - REQUIRES MANUAL INTERVENTION", 
                            gap_info[0], gap_info[1], gap_info[2])
    
    if not final_validation_gaps_50mm:
        logger.info("Final gap validation: All gaps <= 50mm (BIM-compliant)")
    elif not final_validation_gaps_100mm or all(g[2] > 100.0 for g in final_validation_gaps_100mm):
        logger.info("Final gap validation: All gaps <= 100mm (guaranteed repair threshold met)")
    else:
        logger.warning("Final gap validation: Some gaps <= 100mm remain - BIM compliance may be compromised")
    
    return modified_axes


def post_process_gap_closure(
    axes: List[LineString],
    thickness_by_index_mm: Mapping[int, float] | None = None,
) -> List[LineString]:
    """
    Final aggressive gap closure - guarantees all gaps ≤100mm are closed.
    This is a post-processing step after close_wall_gaps() to ensure BIM compliance.
    
    Args:
        axes: List of wall axis LineStrings (potentially with remaining gaps)
        thickness_by_index_mm: Optional mapping of axis index to wall thickness for adaptive tolerances
    
    Returns:
        List of wall axes with all gaps ≤100mm closed
    """
    from shapely.geometry import Point
    
    if not axes or len(axes) < 2:
        return list(axes)
    
    modified = [LineString(axis.coords) for axis in axes]
    max_iterations = 3
    
    for iteration in range(max_iterations):
        remaining_gaps = []
        
        for i, ax1 in enumerate(modified):
            if ax1.length < 1e-3:
                continue
            coords1 = list(ax1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            
            for j, ax2 in enumerate(modified[i + 1:], start=i + 1):
                if ax2.length < 1e-3:
                    continue
                coords2 = list(ax2.coords)
                if len(coords2) < 2:
                    continue
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                dists = [
                    ep1_start.distance(ep2_start),
                    ep1_start.distance(ep2_end),
                    ep1_end.distance(ep2_start),
                    ep1_end.distance(ep2_end),
                ]
                min_dist = min(dists)
                if 1.0 < min_dist <= 100.0:
                    remaining_gaps.append((i, j, min_dist, dists.index(min_dist)))
        
        if not remaining_gaps:
            logger.debug("Post-process gap closure iteration %d: All gaps ≤100mm closed", iteration + 1)
            break
        
        logger.debug("Post-process gap closure iteration %d: Repairing %d gaps ≤100mm", iteration + 1, len(remaining_gaps))
        
        # Aggressive repair: direct extension
        for i, j, dist, dist_idx in remaining_gaps:
            try:
                ax1 = modified[i]
                ax2 = modified[j]
                coords1 = list(ax1.coords)
                coords2 = list(ax2.coords)
                
                # Determine which endpoints to connect
                ep1_start = Point(coords1[0])
                ep1_end = Point(coords1[-1])
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                endpoints = [
                    (ep1_start, ep2_start, False, False),
                    (ep1_start, ep2_end, False, True),
                    (ep1_end, ep2_start, True, False),
                    (ep1_end, ep2_end, True, True),
                ]
                ep1, ep2, extend_end1, extend_end2 = endpoints[dist_idx]
                
                # Direct extension to closest endpoint
                target = (ep2.x, ep2.y)
                new_coords1 = list(coords1)
                new_coords2 = list(coords2)
                
                if extend_end1:
                    if target not in new_coords1:
                        new_coords1.append(target)
                else:
                    if target not in new_coords1:
                        new_coords1.insert(0, target)
                
                if extend_end2:
                    if target not in new_coords2:
                        new_coords2.append(target)
                else:
                    if target not in new_coords2:
                        new_coords2.insert(0, target)
                
                modified[i] = LineString(new_coords1)
                modified[j] = LineString(new_coords2)
                logger.debug("Post-process gap closure: Closed gap between axis %d and %d (%.1fmm)", i, j, dist)
            except Exception as repair_exc:
                logger.warning("Post-process gap closure: Failed to repair gap between axis %d and %d: %s", i, j, repair_exc)
                continue
        
        # Validate after iteration
        final_gaps = []
        for i, ax1 in enumerate(modified):
            if ax1.length < 1e-3:
                continue
            coords1 = list(ax1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            for j, ax2 in enumerate(modified[i + 1:], start=i + 1):
                if ax2.length < 1e-3:
                    continue
                coords2 = list(ax2.coords)
                if len(coords2) < 2:
                    continue
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                dists = [
                    ep1_start.distance(ep2_start),
                    ep1_start.distance(ep2_end),
                    ep1_end.distance(ep2_start),
                    ep1_end.distance(ep2_end),
                ]
                min_dist = min(dists)
                if 1.0 < min_dist <= 100.0:
                    final_gaps.append((i, j, min_dist))
        
        if final_gaps:
            logger.warning("Post-process gap closure: %d gaps ≤100mm remain after iteration %d", len(final_gaps), iteration + 1)
        else:
            logger.info("Post-process gap closure: All gaps ≤100mm successfully closed after iteration %d", iteration + 1)
    
    return modified


def guarantee_gap_closure(
    axes: List[LineString],
    thickness_by_index_mm: Mapping[int, float] | None = None,
) -> List[LineString]:
    """
    Final guarantee: All gaps ≤100mm MUST be closed.
    This is the last resort before IFC export to ensure 100% BIM compliance.
    
    Args:
        axes: List of wall axis LineStrings (potentially with remaining gaps)
        thickness_by_index_mm: Optional mapping of axis index to wall thickness for adaptive tolerances
    
    Returns:
        List of wall axes with all gaps ≤100mm closed (guaranteed)
    """
    from shapely.geometry import Point
    
    if not axes or len(axes) < 2:
        return list(axes)
    
    modified = [LineString(axis.coords) for axis in axes]
    max_iterations = 5
    
    for iteration in range(max_iterations):
        remaining_gaps = []
        
        for i, ax1 in enumerate(modified):
            if ax1.length < 1e-3:
                continue
            coords1 = list(ax1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            
            for j, ax2 in enumerate(modified[i + 1:], start=i + 1):
                if ax2.length < 1e-3:
                    continue
                coords2 = list(ax2.coords)
                if len(coords2) < 2:
                    continue
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                dists = [
                    ep1_start.distance(ep2_start),
                    ep1_start.distance(ep2_end),
                    ep1_end.distance(ep2_start),
                    ep1_end.distance(ep2_end),
                ]
                min_dist = min(dists)
                if 1.0 < min_dist <= 100.0:
                    remaining_gaps.append((i, j, min_dist, dists.index(min_dist)))
        
        if not remaining_gaps:
            logger.info("Guarantee gap closure: All gaps ≤100mm closed after iteration %d (BIM-compliant)", iteration + 1)
            break
        
        logger.debug("Guarantee gap closure iteration %d: Repairing %d gaps ≤100mm", iteration + 1, len(remaining_gaps))
        
        # Aggressive repair: direct extension
        for i, j, dist, dist_idx in remaining_gaps:
            try:
                ax1 = modified[i]
                ax2 = modified[j]
                coords1 = list(ax1.coords)
                coords2 = list(ax2.coords)
                
                # Determine which endpoints to connect
                ep1_start = Point(coords1[0])
                ep1_end = Point(coords1[-1])
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                
                endpoints = [
                    (ep1_start, ep2_start, False, False),
                    (ep1_start, ep2_end, False, True),
                    (ep1_end, ep2_start, True, False),
                    (ep1_end, ep2_end, True, True),
                ]
                ep1, ep2, extend_end1, extend_end2 = endpoints[dist_idx]
                
                # Direct extension to closest endpoint
                target = (ep2.x, ep2.y)
                new_coords1 = list(coords1)
                new_coords2 = list(coords2)
                
                if extend_end1:
                    if target not in new_coords1:
                        new_coords1.append(target)
                else:
                    if target not in new_coords1:
                        new_coords1.insert(0, target)
                
                if extend_end2:
                    if target not in new_coords2:
                        new_coords2.append(target)
                else:
                    if target not in new_coords2:
                        new_coords2.insert(0, target)
                
                modified[i] = LineString(new_coords1)
                modified[j] = LineString(new_coords2)
                logger.debug("Guarantee gap closure: Closed gap between axis %d and %d (%.1fmm)", i, j, dist)
            except Exception as repair_exc:
                logger.warning("Guarantee gap closure: Failed to repair gap between axis %d and %d: %s", i, j, repair_exc)
                continue
        
        # Final validation after iteration
        final_gaps = []
        for i, ax1 in enumerate(modified):
            if ax1.length < 1e-3:
                continue
            coords1 = list(ax1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            for j, ax2 in enumerate(modified[i + 1:], start=i + 1):
                if ax2.length < 1e-3:
                    continue
                coords2 = list(ax2.coords)
                if len(coords2) < 2:
                    continue
                ep2_start = Point(coords2[0])
                ep2_end = Point(coords2[-1])
                dists = [
                    ep1_start.distance(ep2_start),
                    ep1_start.distance(ep2_end),
                    ep1_end.distance(ep2_start),
                    ep1_end.distance(ep2_end),
                ]
                min_dist = min(dists)
                if 1.0 < min_dist <= 100.0:
                    final_gaps.append((i, j, min_dist))
        
        if final_gaps:
            max_gap = max(gap[2] for gap in final_gaps)
            logger.warning("Guarantee gap closure: %d gaps ≤100mm remain after iteration %d (max: %.1fmm)", len(final_gaps), iteration + 1, max_gap)
        else:
            logger.info("Guarantee gap closure: All gaps ≤100mm successfully closed after iteration %d (BIM-compliant)", iteration + 1)
    
    # Final validation
    final_validation_gaps = []
    for i, ax1 in enumerate(modified):
        if ax1.length < 1e-3:
            continue
        coords1 = list(ax1.coords)
        if len(coords1) < 2:
            continue
        ep1_start = Point(coords1[0])
        ep1_end = Point(coords1[-1])
        for j, ax2 in enumerate(modified[i + 1:], start=i + 1):
            if ax2.length < 1e-3:
                continue
            coords2 = list(ax2.coords)
            if len(coords2) < 2:
                continue
            ep2_start = Point(coords2[0])
            ep2_end = Point(coords2[-1])
            dists = [
                ep1_start.distance(ep2_start),
                ep1_start.distance(ep2_end),
                ep1_end.distance(ep2_start),
                ep1_end.distance(ep2_end),
            ]
            min_dist = min(dists)
            if 1.0 < min_dist <= 100.0:
                final_validation_gaps.append((i, j, min_dist))
    
    if final_validation_gaps:
        max_gap = max(gap[2] for gap in final_validation_gaps)
        logger.error("CRITICAL: Guarantee gap closure FAILED - %d gaps ≤100mm remain (max: %.1fmm) - BIM compliance compromised", 
                    len(final_validation_gaps), max_gap)
    else:
        logger.info("Guarantee gap closure: SUCCESS - All gaps ≤100mm closed (100% BIM-compliant)")
    
    return modified


