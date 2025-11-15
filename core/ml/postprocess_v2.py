"""
IFC Export V2 - Post-Processing Pipeline for Pixel Noise Reduction

This module implements a 3-step post-processing pipeline to clean up
Roboflow polygon predictions:
1. Polygon simplification (Douglas-Peucker)
2. Snap-to-grid and enforce 90° angles
3. Context-based correction (rule-based fixes)
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from loguru import logger
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from core.ml.pipeline_config import PipelineConfig
from core.ml.geometry_processor import GeometryProcessor


def get_thickness(prediction: dict[str, Any]) -> float:
    """
    Estimate wall thickness from prediction polygon.
    
    Uses minimum distance from centroid to boundary as thickness estimate.
    For rectangular walls, this approximates the wall thickness.
    
    Args:
        prediction: Prediction dict with 'points' field
        
    Returns:
        Estimated thickness in mm, or 0.0 if cannot be calculated
    """
    points = prediction.get("points") or []
    if not points or len(points) < 3:
        return 0.0
    
    try:
        # Convert points to polygon
        polygon_points = [(float(pt[0]), float(pt[1])) for pt in points if len(pt) >= 2]
        if len(polygon_points) < 3:
            return 0.0
        
        # Ensure closed polygon
        if polygon_points[0] != polygon_points[-1]:
            polygon_points.append(polygon_points[0])
        
        poly = Polygon(polygon_points)
        if poly.is_empty or not poly.is_valid:
            return 0.0
        
        # Estimate thickness as minimum distance from centroid to boundary
        centroid = poly.centroid
        boundary = poly.boundary
        min_dist = boundary.distance(centroid)
        
        # For walls, thickness is approximately 2 * min_dist (centroid to edge)
        thickness = float(min_dist * 2.0)
        return max(thickness, 0.0)
    except Exception:
        return 0.0


def filter_noise(predictions: list[dict[str, Any]], min_thickness_mm: float = 50.0, min_confidence: float = 0.7) -> list[dict[str, Any]]:
    """
    Filtere nur Rauschen, nicht Geometrie.
    
    Filters predictions based on confidence and minimum thickness to remove noise
    while preserving actual geometry.
    
    Args:
        predictions: List of prediction dicts with 'confidence' and 'points' fields
        min_thickness_mm: Minimum wall thickness in mm (default: 50.0)
        min_confidence: Minimum confidence score (default: 0.7)
        
    Returns:
        Filtered list of predictions that meet the criteria
    """
    filtered = []
    for p in predictions:
        confidence = float(p.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        
        thickness = get_thickness(p)
        if thickness < min_thickness_mm:
            continue
        
        filtered.append(p)
    
    return filtered


# Legacy default values (for backward compatibility)
SIMPLIFY_TOLERANCE_WALLS = 2.0  # mm
SIMPLIFY_TOLERANCE_DOORS_WINDOWS = 0.5  # mm
GRID_SIZE = 50.0  # mm (5cm)
ANGLE_TOLERANCE = 5.0  # degrees
MAX_GAP_CLOSE = 10.0  # mm
DOOR_WALL_MAX_DISTANCE = 100.0  # mm


def simplify_polygon(
    polygon: Polygon,
    tolerance: float,
    preserve_topology: bool = True,
) -> Polygon:
    """
    Simplify polygon using Douglas-Peucker algorithm.
    
    Args:
        polygon: Input polygon to simplify
        tolerance: Simplification tolerance in mm
        preserve_topology: Whether to preserve topology during simplification
        
    Returns:
        Simplified polygon (validated and repaired if needed)
    """
    if polygon.is_empty or not polygon.is_valid:
        # Try to repair first
        try:
            polygon = polygon.buffer(0)
            if isinstance(polygon, Polygon) and polygon.is_valid:
                pass
            else:
                return polygon
        except Exception:
            return polygon
    
    try:
        simplified = polygon.simplify(tolerance, preserve_topology=preserve_topology)
        
        # Validate simplified geometry
        if simplified.is_empty:
            logger.warning("Simplification resulted in empty polygon, using original")
            return polygon
        
        if not simplified.is_valid:
            # Try to repair
            try:
                repaired = simplified.buffer(0)
                if isinstance(repaired, Polygon) and repaired.is_valid and not repaired.is_empty:
                    return repaired
                else:
                    logger.warning("Simplified polygon could not be repaired, using original")
                    return polygon
            except Exception:
                logger.warning("Error repairing simplified polygon, using original")
                return polygon
        
        return simplified
    except Exception as exc:
        logger.warning("Error during polygon simplification: %s, using original", exc)
        return polygon


def snap_to_grid(
    point: tuple[float, float],
    grid_size: float = GRID_SIZE,
) -> tuple[float, float]:
    """
    Snap a point to the nearest grid position.
    
    Args:
        point: (x, y) coordinates
        grid_size: Grid size in mm (default: 50mm = 5cm)
        
    Returns:
        Snapped (x, y) coordinates
    """
    x, y = point
    snapped_x = round(x / grid_size) * grid_size
    snapped_y = round(y / grid_size) * grid_size
    return (snapped_x, snapped_y)


def snap_polygon_to_grid(
    polygon: Polygon,
    grid_size: float | None = None,
    config: PipelineConfig | None = None,
    class_name: str = "",
) -> Polygon:
    """
    Snap all vertices of a polygon to the grid.
    
    Args:
        polygon: Input polygon
        grid_size: Grid size in mm (if None, uses config or default)
        config: Pipeline configuration (for adaptive grid)
        class_name: Object class name (for adaptive grid)
        
    Returns:
        Polygon with snapped vertices
    """
    if polygon.is_empty:
        return polygon
    
    # Determine grid size
    if grid_size is None:
        if config is not None and config.enable_adaptive_grid:
            grid_size = config.get_grid_size(class_name)
        else:
            grid_size = GRID_SIZE  # Legacy default
    
    try:
        # Get exterior coordinates
        exterior_coords = list(polygon.exterior.coords)
        # Remove duplicate last point if polygon is closed
        if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]
        
        # Snap exterior points
        snapped_exterior = [snap_to_grid(coord, grid_size) for coord in exterior_coords]
        # Ensure closed
        if snapped_exterior[0] != snapped_exterior[-1]:
            snapped_exterior.append(snapped_exterior[0])
        
        # Snap interior holes
        snapped_interiors = []
        for interior in polygon.interiors:
            interior_coords = list(interior.coords)
            if len(interior_coords) > 1 and interior_coords[0] == interior_coords[-1]:
                interior_coords = interior_coords[:-1]
            snapped_interior = [snap_to_grid(coord, grid_size) for coord in interior_coords]
            if snapped_interior[0] != snapped_interior[-1]:
                snapped_interior.append(snapped_interior[0])
            snapped_interiors.append(snapped_interior)
        
        # Create new polygon
        new_poly = Polygon(snapped_exterior, snapped_interiors)
        
        # Validate
        if not new_poly.is_valid:
            try:
                new_poly = new_poly.buffer(0)
                if isinstance(new_poly, Polygon) and new_poly.is_valid:
                    return new_poly
            except Exception:
                pass
            logger.warning("Snapped polygon is invalid, using original")
            return polygon
        
        return new_poly
    except Exception as exc:
        logger.warning("Error snapping polygon to grid: %s, using original", exc)
        return polygon


def _calculate_angle(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    """Calculate angle at p2 between p1-p2 and p2-p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Calculate angle using dot product
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0
    
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def _snap_angle_to_90(angle_deg: float, tolerance_deg: float = ANGLE_TOLERANCE) -> float | None:
    """
    Snap angle to nearest 90° if within tolerance.
    
    Args:
        angle_deg: Angle in degrees
        tolerance_deg: Tolerance in degrees
        
    Returns:
        Snapped angle (0, 90, 180, 270) or None if not within tolerance
    """
    # Normalize to 0-180 range
    angle_norm = angle_deg % 180.0
    if angle_norm > 90.0:
        angle_norm = 180.0 - angle_norm
    
    # Check if close to 0° or 90°
    if angle_norm <= tolerance_deg:
        return 0.0
    elif abs(angle_norm - 90.0) <= tolerance_deg:
        return 90.0
    
    return None


def enforce_right_angles(
    polygon: Polygon,
    tolerance_deg: float = ANGLE_TOLERANCE,
) -> Polygon:
    """
    Enforce 90° angles in polygon by detecting and correcting near-orthogonal segments.
    
    Uses RANSAC-like approach to detect straight line segments and snap them to
    horizontal/vertical directions.
    
    Args:
        polygon: Input polygon
        tolerance_deg: Angle tolerance in degrees (default: 5°)
        
    Returns:
        Polygon with enforced right angles
    """
    if polygon.is_empty or len(polygon.exterior.coords) < 3:
        return polygon
    
    try:
        coords = list(polygon.exterior.coords)
        # Remove duplicate last point if closed
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        
        if len(coords) < 3:
            return polygon
        
        # Detect line segments and their directions
        segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            
            if length < 1e-6:
                continue
            
            # Calculate angle of segment
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            # Normalize to 0-180 range
            angle_norm = angle_deg % 180.0
            if angle_norm > 90.0:
                angle_norm = 180.0 - angle_norm
            
            segments.append((p1, p2, angle_norm))
        
        # Snap segments to horizontal/vertical
        corrected_coords: list[tuple[float, float]] = []
        
        for i, (p1, p2, angle_norm) in enumerate(segments):
            # Check if segment should be horizontal or vertical
            if angle_norm <= tolerance_deg:
                # Horizontal
                corrected_coords.append(p1)
                if i == len(segments) - 1:
                    # Last segment, also add p2
                    corrected_coords.append((p2[0], p1[1]))  # Keep x from p2, y from p1
            elif abs(angle_norm - 90.0) <= tolerance_deg:
                # Vertical
                corrected_coords.append(p1)
                if i == len(segments) - 1:
                    corrected_coords.append((p1[0], p2[1]))  # Keep x from p1, y from p2
            else:
                # Not orthogonal, keep original
                corrected_coords.append(p1)
                if i == len(segments) - 1:
                    corrected_coords.append(p2)
        
        # Remove duplicates and ensure closed
        if len(corrected_coords) > 0:
            if corrected_coords[0] != corrected_coords[-1]:
                corrected_coords.append(corrected_coords[0])
        
        if len(corrected_coords) < 3:
            logger.warning("Enforce right angles resulted in too few points, using original")
            return polygon
        
        new_poly = Polygon(corrected_coords)
        
        # Validate
        if not new_poly.is_valid:
            try:
                new_poly = new_poly.buffer(0)
                if isinstance(new_poly, Polygon) and new_poly.is_valid:
                    return new_poly
            except Exception:
                pass
            logger.warning("Enforced polygon is invalid, using original")
            return polygon
        
        return new_poly
    except Exception as exc:
        logger.warning("Error enforcing right angles: %s, using original", exc)
        return polygon


def fix_door_position(
    door_polygon: Polygon,
    wall_polygons: list[Polygon],
    max_distance: float = DOOR_WALL_MAX_DISTANCE,
) -> Polygon:
    """
    Fix door position to ensure it touches a wall.
    
    Args:
        door_polygon: Door polygon
        wall_polygons: List of wall polygons
        max_distance: Maximum distance to search for walls (mm)
        
    Returns:
        Corrected door polygon
    """
    if not wall_polygons or door_polygon.is_empty:
        return door_polygon
    
    try:
        door_centroid = door_polygon.centroid
        
        # Find nearest wall
        nearest_wall: Polygon | None = None
        min_distance = float('inf')
        
        for wall in wall_polygons:
            if wall.is_empty:
                continue
            distance = door_centroid.distance(wall)
            if distance < min_distance:
                min_distance = distance
                nearest_wall = wall
        
        if nearest_wall is None or min_distance > max_distance:
            logger.debug("No wall found within %f mm of door", max_distance)
            return door_polygon
        
        # Check if door already intersects wall
        if door_polygon.intersects(nearest_wall):
            # Door already touches wall, but ensure it's properly clipped
            intersection = door_polygon.intersection(nearest_wall)
            if not intersection.is_empty:
                # Use intersection if it's a valid polygon
                if isinstance(intersection, Polygon) and intersection.is_valid:
                    return intersection
                elif hasattr(intersection, 'geoms'):
                    # MultiPolygon - take largest
                    valid_polys = [p for p in intersection.geoms if isinstance(p, Polygon) and p.is_valid]
                    if valid_polys:
                        return max(valid_polys, key=lambda p: p.area)
        
        # Move door to nearest point on wall boundary
        wall_boundary = nearest_wall.boundary
        nearest_point = wall_boundary.interpolate(wall_boundary.project(door_centroid))
        
        # Calculate translation
        dx = nearest_point.x - door_centroid.x
        dy = nearest_point.y - door_centroid.y
        
        # Translate door
        translated = door_polygon.translate(dx, dy)
        
        # Clip to wall
        clipped = translated.intersection(nearest_wall)
        if isinstance(clipped, Polygon) and clipped.is_valid and not clipped.is_empty:
            return clipped
        
        return door_polygon
    except Exception as exc:
        logger.warning("Error fixing door position: %s, using original", exc)
        return door_polygon


def close_wall_gaps(
    wall_polygons: list[Polygon],
    max_gap: float = MAX_GAP_CLOSE,
) -> list[Polygon]:
    """
    Close small gaps between wall segments.
    
    Args:
        wall_polygons: List of wall polygons
        max_gap: Maximum gap size to close (mm)
        
    Returns:
        List of corrected wall polygons
    """
    if len(wall_polygons) < 2:
        return wall_polygons
    
    try:
        corrected_walls: list[Polygon] = []
        processed_indices: set[int] = set()
        
        for i, wall1 in enumerate(wall_polygons):
            if i in processed_indices or wall1.is_empty:
                corrected_walls.append(wall1)
                continue
            
            # Find nearby walls
            wall1_boundary = wall1.boundary
            wall1_endpoints: list[Point] = []
            
            # Get endpoints (simplified: use first and last point of boundary)
            if isinstance(wall1_boundary, LineString) and len(wall1_boundary.coords) >= 2:
                coords = list(wall1_boundary.coords)
                wall1_endpoints = [Point(coords[0]), Point(coords[-1])]
            
            merged_wall = wall1
            merged = False
            
            for j, wall2 in enumerate(wall_polygons):
                if i == j or j in processed_indices or wall2.is_empty:
                    continue
                
                wall2_boundary = wall2.boundary
                if isinstance(wall2_boundary, LineString) and len(wall2_boundary.coords) >= 2:
                    coords = list(wall2_boundary.coords)
                    wall2_endpoints = [Point(coords[0]), Point(coords[-1])]
                    
                    # Check for nearby endpoints
                    for ep1 in wall1_endpoints:
                        for ep2 in wall2_endpoints:
                            distance = ep1.distance(ep2)
                            if distance <= max_gap:
                                # Try to merge walls
                                try:
                                    merged_geom = unary_union([merged_wall, wall2])
                                    if isinstance(merged_geom, Polygon) and merged_geom.is_valid:
                                        merged_wall = merged_geom
                                        processed_indices.add(j)
                                        merged = True
                                        logger.debug("Merged walls %d and %d (gap: %.2f mm)", i, j, distance)
                                        break
                                except Exception:
                                    pass
                        if merged:
                            break
                    if merged:
                        break
            
            corrected_walls.append(merged_wall)
            processed_indices.add(i)
        
        # Add any unprocessed walls
        for i, wall in enumerate(wall_polygons):
            if i not in processed_indices:
                corrected_walls.append(wall)
        
        return corrected_walls
    except Exception as exc:
        logger.warning("Error closing wall gaps: %s, using original walls", exc)
        return wall_polygons


def clip_window_to_wall(
    window_polygon: Polygon,
    wall_polygon: Polygon,
) -> Polygon:
    """
    Clip window polygon to ensure it's completely inside the wall.
    
    Args:
        window_polygon: Window polygon
        wall_polygon: Wall polygon
        
    Returns:
        Clipped window polygon (intersection with wall)
    """
    if window_polygon.is_empty or wall_polygon.is_empty:
        return window_polygon
    
    try:
        intersection = window_polygon.intersection(wall_polygon)
        
        if intersection.is_empty:
            logger.debug("Window does not intersect wall, returning original")
            return window_polygon
        
        if isinstance(intersection, Polygon) and intersection.is_valid:
            return intersection
        elif hasattr(intersection, 'geoms'):
            # MultiPolygon - take largest valid polygon
            valid_polys = [p for p in intersection.geoms if isinstance(p, Polygon) and p.is_valid]
            if valid_polys:
                return max(valid_polys, key=lambda p: p.area)
        
        return window_polygon
    except Exception as exc:
        logger.warning("Error clipping window to wall: %s, using original", exc)
        return window_polygon


def process_roboflow_predictions_v2(
    raw_predictions: list[dict[str, Any]],
    px_per_mm: float | None = None,
    *,
    config: PipelineConfig | None = None,
    simplify_tolerance_walls: float | None = None,
    simplify_tolerance_doors_windows: float | None = None,
    grid_size: float | None = None,  # Deprecated: use config
    angle_tolerance: float | None = None,
    max_gap_close: float | None = None,
    door_wall_max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """
    Main post-processing pipeline for Roboflow predictions.
    
    Processes each prediction through 3 steps:
    1. Polygon simplification (Douglas-Peucker)
    2. Snap-to-grid and enforce 90° angles
    3. Context-based correction
    
    Args:
        raw_predictions: List of Roboflow prediction dicts with 'points', 'class', 'confidence'
        px_per_mm: Pixels per millimeter conversion factor (if None, assumes points are already in mm)
        config: Pipeline configuration (uses defaults if None)
        simplify_tolerance_walls: Simplification tolerance for walls (mm) - deprecated, use config
        simplify_tolerance_doors_windows: Simplification tolerance for doors/windows (mm) - deprecated, use config
        grid_size: Grid size for snapping (mm) - deprecated, use config
        angle_tolerance: Angle tolerance for enforcing right angles (degrees) - deprecated, use config
        max_gap_close: Maximum gap to close between walls (mm) - deprecated, use config
        door_wall_max_distance: Maximum distance to search for walls when fixing doors (mm) - deprecated, use config
        
    Returns:
        List of cleaned prediction dicts with processed polygons
    """
    if not raw_predictions:
        return []
    
    # Use config or create default
    if config is None:
        config = PipelineConfig.default()
    
    # Override config with legacy parameters if provided (for backward compatibility)
    if simplify_tolerance_walls is not None:
        config.simplify_tolerance_walls = simplify_tolerance_walls
    if simplify_tolerance_doors_windows is not None:
        config.simplify_tolerance_doors_windows = simplify_tolerance_doors_windows
    if angle_tolerance is not None:
        config.angle_tolerance = angle_tolerance
    if max_gap_close is not None:
        config.max_gap_close = max_gap_close
    if door_wall_max_distance is not None:
        config.door_wall_max_distance = door_wall_max_distance
    
    # Initialize geometry processor chain
    geometry_processor = GeometryProcessor(config)
    
    cleaned_objects: list[dict[str, Any]] = []
    
    # Separate objects by class for context correction
    walls: list[tuple[int, Polygon]] = []
    doors: list[tuple[int, Polygon]] = []
    windows: list[tuple[int, Polygon]] = []
    
    # Step 1 & 2: Process each polygon through geometry processor chain
    for idx, obj in enumerate(raw_predictions):
        class_name = str(obj.get("class", "")).lower().strip()
        points = obj.get("points") or []
        confidence = float(obj.get("confidence", 0.0))
        
        if not points:
            # Skip objects without polygon points
            continue
        
        # Convert points to list of tuples
        polygon_points: list[tuple[float, float]] = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                # Convert from pixels to mm if px_per_mm is provided
                if px_per_mm is not None:
                    x = x / px_per_mm
                    y = y / px_per_mm
                polygon_points.append((x, y))
        
        if len(polygon_points) < 3:
            continue
        
        # Ensure closed polygon
        if polygon_points[0] != polygon_points[-1]:
            polygon_points.append(polygon_points[0])
        
        try:
            raw_poly = Polygon(polygon_points)
            
            if raw_poly.is_empty or not raw_poly.is_valid:
                # Try to repair
                try:
                    raw_poly = raw_poly.buffer(0)
                    if not isinstance(raw_poly, Polygon) or raw_poly.is_empty:
                        continue
                except Exception:
                    continue
            
            # Process through geometry processor chain
            processed_poly = geometry_processor.process(obj, raw_poly)
            
            # Store for context correction
            if "wall" in class_name:
                walls.append((idx, processed_poly))
            elif "door" in class_name:
                doors.append((idx, processed_poly))
            elif "window" in class_name:
                windows.append((idx, processed_poly))
            
            # Convert back to points format
            exterior_coords = list(processed_poly.exterior.coords)
            # Remove duplicate last point
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            
            # Convert back to pixels if needed
            if px_per_mm is not None:
                processed_points = [[x * px_per_mm, y * px_per_mm] for x, y in exterior_coords]
            else:
                processed_points = [[x, y] for x, y in exterior_coords]
            
            cleaned_obj = {
                "class": obj.get("class", ""),
                "confidence": confidence,
                "points": processed_points,
                **{k: v for k, v in obj.items() if k not in ("class", "confidence", "points")},
            }
            cleaned_objects.append(cleaned_obj)
            
        except Exception as exc:
            logger.warning("Error processing prediction %d (%s): %s", idx, class_name, exc)
            continue
    
    # Step 3: Context-based correction
    wall_polygons = [wall for _, wall in walls]
    
    # Fix door positions
    for door_idx, door_poly in doors:
        if door_idx < len(cleaned_objects):
            fixed_door = fix_door_position(door_poly, wall_polygons, config.door_wall_max_distance)
            # Update points in cleaned_objects
            exterior_coords = list(fixed_door.exterior.coords)
            if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            if px_per_mm is not None:
                processed_points = [[x * px_per_mm, y * px_per_mm] for x, y in exterior_coords]
            else:
                processed_points = [[x, y] for x, y in exterior_coords]
            cleaned_objects[door_idx]["points"] = processed_points
    
    # Clip windows to walls
    for window_idx, window_poly in windows:
        if window_idx < len(cleaned_objects):
            # Find nearest wall
            nearest_wall: Polygon | None = None
            min_distance = float('inf')
            window_centroid = window_poly.centroid
            
            for wall in wall_polygons:
                distance = window_centroid.distance(wall)
                if distance < min_distance:
                    min_distance = distance
                    nearest_wall = wall
            
            if nearest_wall:
                clipped_window = clip_window_to_wall(window_poly, nearest_wall)
                exterior_coords = list(clipped_window.exterior.coords)
                if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                    exterior_coords = exterior_coords[:-1]
                if px_per_mm is not None:
                    processed_points = [[x * px_per_mm, y * px_per_mm] for x, y in exterior_coords]
                else:
                    processed_points = [[x, y] for x, y in exterior_coords]
                cleaned_objects[window_idx]["points"] = processed_points
    
    # Close wall gaps (update wall objects)
    if len(wall_polygons) > 1:
        corrected_walls = close_wall_gaps(wall_polygons, max_gap_close)
        # Update cleaned_objects for walls
        for i, (wall_idx, _) in enumerate(walls):
            if wall_idx < len(cleaned_objects) and i < len(corrected_walls):
                corrected_wall = corrected_walls[i]
                exterior_coords = list(corrected_wall.exterior.coords)
                if len(exterior_coords) > 1 and exterior_coords[0] == exterior_coords[-1]:
                    exterior_coords = exterior_coords[:-1]
                if px_per_mm is not None:
                    processed_points = [[x * px_per_mm, y * px_per_mm] for x, y in exterior_coords]
                else:
                    processed_points = [[x, y] for x, y in exterior_coords]
                cleaned_objects[wall_idx]["points"] = processed_points
    
    return cleaned_objects

