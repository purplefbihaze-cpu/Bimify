from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet


logger = logging.getLogger(__name__)


@dataclass
class SpaceConfig:
    """Configuration for space recognition from walls."""
    min_room_area_m2: float = 5.0
    max_wall_gap_mm: float = 500.0
    treat_open_areas_as_rooms: bool = False
    gap_threshold_mm: float = 50.0
    gap_buffer_mm: float = 25.0
    min_space_area_mm2: float = 5000.0
    min_space_dimension_mm: float = 200.0


@dataclass
class SpacePoly:
    polygon: Polygon
    area_m2: float


def _repair_polygon(poly: Polygon) -> Polygon:
    """Repair invalid polygon using buffer(0) trick."""
    if poly is None or poly.is_empty:
        return poly
    if not poly.is_valid:
        try:
            repaired = poly.buffer(0)
            if isinstance(repaired, Polygon) and not repaired.is_empty and repaired.is_valid:
                return repaired
            elif isinstance(repaired, MultiPolygon):
                # Take largest valid polygon
                valid_polys = [p for p in repaired.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                if valid_polys:
                    return max(valid_polys, key=lambda p: p.area)
        except Exception:
            pass
    return poly


def polygonize_spaces_from_walls(
    dets: List[NormalizedDet],
    config: SpaceConfig | None = None,
) -> List[SpacePoly]:
    """Extract space polygons from wall detections.
    
    Args:
        dets: List of normalized detections
        config: Space recognition configuration (uses defaults if None)
        
    Returns:
        List of space polygons with area in m²
    """
    if config is None:
        config = SpaceConfig()
    
    walls = [d for d in dets if d.type == "WALL"]
    if not walls:
        return []
    
    # Repair wall geometries before union
    repaired_wall_geoms = []
    for wall in walls:
        geom = wall.geom
        if isinstance(geom, Polygon):
            geom = _repair_polygon(geom)
        elif isinstance(geom, MultiPolygon):
            # Repair each part
            repaired_parts = []
            for part in geom.geoms:
                if isinstance(part, Polygon):
                    repaired = _repair_polygon(part)
                    if not repaired.is_empty:
                        repaired_parts.append(repaired)
            if repaired_parts:
                geom = MultiPolygon(repaired_parts) if len(repaired_parts) > 1 else repaired_parts[0]
        if geom is not None and not getattr(geom, "is_empty", False):
            repaired_wall_geoms.append(geom)
    
    if not repaired_wall_geoms:
        return []
    
    # Initialize variables that may be used outside try block
    all_wall_points = []
    has_significant_gaps = False
    polys: List[Polygon] = []
    
    try:
        # Pre-union validation: Check for large gaps that would prevent space polygonization
        from shapely.geometry import Point
        from shapely.ops import unary_union
        
        # Gap detection: Check if walls form a closed boundary
        # Calculate bounding box of all walls
        if not repaired_wall_geoms:
            return []
        
        # Stage 1: Pre-union gap detection
        for geom in repaired_wall_geoms:
            if isinstance(geom, Polygon):
                all_wall_points.extend([Point(c) for c in geom.exterior.coords])
            elif isinstance(geom, MultiPolygon):
                for part in geom.geoms:
                    if isinstance(part, Polygon):
                        all_wall_points.extend([Point(c) for c in part.exterior.coords])
        
        # Stage 2: Union with aggressive repair
        wall_union = unary_union(repaired_wall_geoms)
        
        # Stage 3: Post-union validation and aggressive repair
        if wall_union.is_empty:
            return []
        
        if not wall_union.is_valid:
            # Aggressive repair: multiple attempts
            repair_attempts = [
                lambda g: g.buffer(0),
                lambda g: g.buffer(1.0).buffer(-1.0),
                lambda g: g.simplify(2.0, preserve_topology=True).buffer(0),
            ]
            
            for repair_func in repair_attempts:
                try:
                    repaired = repair_func(wall_union)
                    if not repaired.is_empty and repaired.is_valid:
                        wall_union = repaired
                        break
                except Exception:
                    continue
            
            # If still invalid, try to extract valid parts
            if not wall_union.is_valid:
                try:
                    if isinstance(wall_union, MultiPolygon):
                        valid_parts = [p for p in wall_union.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                        if valid_parts:
                            if len(valid_parts) == 1:
                                wall_union = valid_parts[0]
                            else:
                                wall_union = MultiPolygon(valid_parts)
                    else:
                        wall_union = wall_union.buffer(0)
                except Exception:
                    return []
        
        # Stage 4: Enhanced gap detection - Check actual endpoint distances (more accurate than coverage ratio)
        from shapely.geometry import Point
        
        # Improved gap detection: Check actual endpoint distances between wall segments
        try:
            # Collect all wall endpoints
            wall_endpoints = []
            for geom in repaired_wall_geoms:
                if isinstance(geom, Polygon):
                    coords = list(geom.exterior.coords)
                    if len(coords) >= 2:
                        wall_endpoints.append(Point(coords[0]))
                        wall_endpoints.append(Point(coords[-1]))
                elif isinstance(geom, MultiPolygon):
                    for part in geom.geoms:
                        if isinstance(part, Polygon):
                            coords = list(part.exterior.coords)
                            if len(coords) >= 2:
                                wall_endpoints.append(Point(coords[0]))
                                wall_endpoints.append(Point(coords[-1]))
            
            # Check for gaps: find endpoints that are far from other endpoints
            gap_threshold = config.gap_threshold_mm
            gaps_found = 0
            for i, ep1 in enumerate(wall_endpoints):
                min_dist = float('inf')
                for j, ep2 in enumerate(wall_endpoints):
                    if i != j:
                        dist = ep1.distance(ep2)
                        if dist < min_dist:
                            min_dist = dist
                # If endpoint is >50mm from nearest endpoint, it's a gap
                if min_dist > gap_threshold:
                    gaps_found += 1
            
            # If more than 10% of endpoints have gaps, consider it significant
            if len(wall_endpoints) > 0 and gaps_found > len(wall_endpoints) * 0.1:
                has_significant_gaps = True
                logger.debug("Space polygonization: Detected %d gaps (%.1f%% of endpoints) using endpoint distance method",
                           gaps_found, (gaps_found / len(wall_endpoints)) * 100.0)
        except Exception as gap_detection_exc:
            logger.debug("Endpoint-based gap detection failed, falling back to coverage ratio: %s", gap_detection_exc)
            # Fallback to coverage ratio if endpoint detection fails
            envelope = wall_union.envelope
            union_area = wall_union.area if hasattr(wall_union, 'area') else 0.0
            envelope_area = envelope.area if hasattr(envelope, 'area') else 0.0
            
            if envelope_area > 0 and union_area > 0:
                coverage_ratio = union_area / envelope_area
                if coverage_ratio < 0.1:  # Less than 10% coverage suggests unclosed walls
                    has_significant_gaps = True
        
        # If significant gaps detected, try to repair or use fallback
        if has_significant_gaps:
            # Try to close gaps by buffering walls slightly
            try:
                buffered_union = wall_union.buffer(config.gap_buffer_mm)
                if not buffered_union.is_empty and buffered_union.is_valid:
                    # Use buffered union for space extraction (will be larger but closed)
                    wall_union = buffered_union
                    logger.debug("Space polygonization: Applied 25mm buffer to close gaps in wall union")
            except Exception:
                pass
        
        # Stage 5: Extract spaces from wall union
        # bounding hull from walls
        hull = wall_union.envelope.buffer(0.0)
        if hull.is_empty:
            return []
        
        free = hull.difference(wall_union)
        if free.is_empty:
            return []
        
        # Stage 6: Validate and repair free space polygons
        if isinstance(free, Polygon):
            repaired = _repair_polygon(free)
            if repaired and not repaired.is_empty and repaired.is_valid:
                polys.append(repaired)
        elif isinstance(free, MultiPolygon):
            for p in free.geoms:
                if isinstance(p, Polygon):
                    repaired = _repair_polygon(p)
                    if repaired and not repaired.is_empty and repaired.is_valid:
                        polys.append(repaired)
    except Exception as exc:
        logger.warning("Space polygonization: Exception during space extraction: %s", exc)
        return []
    
    # Fallback: Only use convex hull if absolutely necessary (no spaces found and significant gaps)
    if not polys and has_significant_gaps:
        logger.debug("Space polygonization: No spaces found and significant gaps detected - using convex hull fallback")
        try:
            from shapely.geometry import MultiPoint
            if all_wall_points:
                # Create convex hull from wall points as last resort fallback
                multi_point = MultiPoint(all_wall_points)
                convex_hull = multi_point.convex_hull
                if isinstance(convex_hull, Polygon) and not convex_hull.is_empty and convex_hull.is_valid:
                    # Extract interior spaces from convex hull (simplified approach)
                    # This is a fallback when walls don't form closed boundaries
                    hull_interior = convex_hull.buffer(-100.0)  # Inset by 100mm to approximate interior
                    if isinstance(hull_interior, Polygon) and not hull_interior.is_empty and hull_interior.is_valid:
                        polys.append(hull_interior)
                        logger.debug("Space polygonization: Created space from convex hull fallback")
                    elif isinstance(hull_interior, MultiPolygon):
                        for part in hull_interior.geoms:
                            if isinstance(part, Polygon) and not part.is_empty and part.is_valid:
                                polys.append(part)
                                logger.debug("Space polygonization: Created space from convex hull fallback (multi-part)")
        except Exception as hull_exc:
            logger.debug("Convex hull fallback failed: %s", hull_exc)
    elif not polys:
        logger.warning("Space polygonization: No spaces found but no significant gaps detected - this may indicate an issue")
    
    # Enhanced: Validate resulting spaces
    validated_polys = []
    small_spaces_count = 0
    invalid_spaces_count = 0
    min_space_area_m2 = config.min_room_area_m2
    
    for poly_idx, poly in enumerate(polys):
        try:
            # Validate polygon is not empty
            if poly.is_empty:
                invalid_spaces_count += 1
                logger.warning("Space polygonization: Space %d is empty - skipping", poly_idx)
                continue
            
            # Validate polygon is valid
            if not poly.is_valid:
                # Attempt repair
                try:
                    repaired = poly.buffer(0)
                    if not repaired.is_empty and repaired.is_valid:
                        poly = repaired
                        logger.debug("Space polygonization: Repaired invalid space %d", poly_idx)
                    else:
                        invalid_spaces_count += 1
                        logger.warning("Space polygonization: Space %d is invalid and could not be repaired - skipping", poly_idx)
                        continue
                except Exception as repair_exc:
                    invalid_spaces_count += 1
                    logger.warning("Space polygonization: Failed to repair invalid space %d: %s - skipping", poly_idx, repair_exc)
                    continue
            
            # Validate polygon is closed
            if isinstance(poly, Polygon):
                coords = list(poly.exterior.coords)
                if len(coords) >= 3:
                    first = coords[0]
                    last = coords[-1]
                    dist = math.hypot(first[0] - last[0], first[1] - last[1])
                    if dist > 1.0:  # Not closed
                        try:
                            coords.append(coords[0])
                            poly = Polygon(coords)
                            logger.debug("Space polygonization: Closed space %d (gap: %.2fmm)", poly_idx, dist)
                        except Exception:
                            invalid_spaces_count += 1
                            logger.warning("Space polygonization: Failed to close space %d - skipping", poly_idx)
                            continue
            
            # Validate coordinates are finite
            if isinstance(poly, Polygon):
                coords = list(poly.exterior.coords)
                for coord in coords:
                    if not all(math.isfinite(c) for c in coord):
                        invalid_spaces_count += 1
                        logger.warning("Space polygonization: Space %d has non-finite coordinates - skipping", poly_idx)
                        break
                else:
                    # Check space area
                    area_m2 = poly.area / 1_000_000.0  # Convert mm² to m²
                    if area_m2 < min_space_area_m2:
                        small_spaces_count += 1
                        logger.warning(
                            "Space polygonization: Space %d is very small (%.2f m², minimum: %.2f m²) - may be an artifact",
                            poly_idx, area_m2, min_space_area_m2
                        )
                    validated_polys.append(poly)
            else:
                validated_polys.append(poly)
        except Exception as validation_exc:
            invalid_spaces_count += 1
            logger.warning("Space polygonization: Exception during validation of space %d: %s - skipping", poly_idx, validation_exc)
            continue
    
    # Summary logging
    if invalid_spaces_count > 0:
        logger.warning(
            "Space polygonization: %d invalid space(s) detected and removed (total spaces: %d)",
            invalid_spaces_count, len(polys)
        )
    if small_spaces_count > 0:
        logger.warning(
            "Space polygonization: %d very small space(s) detected (<%.2f m²) - may be artifacts",
            small_spaces_count, min_space_area_m2
        )
    if len(validated_polys) < len(polys):
        logger.info(
            "Space polygonization: Validated %d/%d spaces (removed %d invalid/small spaces)",
            len(validated_polys), len(polys), len(polys) - len(validated_polys)
        )
    
    # Use validated polygons
    polys = validated_polys
    
    # Post-Processing: Filter and validate spaces (additional filtering)
    valid_spaces = []
    min_area_mm2 = config.min_space_area_mm2
    
    # Calculate building envelope for filtering
    try:
        building_envelope = hull.buffer(100.0)  # 100mm buffer for tolerance
    except Exception:
        building_envelope = None
    
    for p in polys:
        if p is None or p.is_empty:
            continue
        
        # Additional validation: Space must be a closed, valid polygon (double-check)
        if not p.is_valid:
            p = _repair_polygon(p)
            if p is None or p.is_empty or not p.is_valid:
                continue
        
        # Check if polygon is closed (exterior ring is closed)
        if hasattr(p, 'exterior') and p.exterior:
            coords = list(p.exterior.coords)
            if len(coords) >= 2 and coords[0] != coords[-1]:
                # Not closed - try to close it
                try:
                    closed_coords = list(coords) + [coords[0]]
                    p = Polygon(closed_coords)
                except Exception:
                    continue
        
        # Filter: Remove spaces that are too small
        if p.area < min_area_mm2:
            continue
        
        # Filter: Remove spaces that are outside building envelope (with tolerance)
        if building_envelope and not building_envelope.contains(p) and not building_envelope.intersects(p.buffer(50.0)):
            continue
        
        # Additional validation: Check if space has reasonable dimensions
        # Reject spaces that are too narrow (width or height < 200mm)
        try:
            rect = p.minimum_rotated_rectangle
            if not rect.is_empty:
                coords = list(rect.exterior.coords)
                if len(coords) >= 4:
                    edges = []
                    for i in range(4):
                        x1, y1 = coords[i]
                        x2, y2 = coords[(i + 1) % 4]
                        edge_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                        edges.append(edge_len)
                    min_dimension = min(edges) if edges else 0.0
                    if min_dimension < config.min_space_dimension_mm:
                        continue
        except Exception:
            pass  # If dimension check fails, still include the space
        
        valid_spaces.append(SpacePoly(polygon=p, area_m2=float(p.area / 1_000_000.0)))
    
    return valid_spaces


