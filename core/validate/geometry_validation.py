"""
Geometry Validation for Reconstruction

Validates reconstructed geometry (wall axes, spaces) before IFC export
to ensure quality and catch issues early.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from loguru import logger
from shapely.geometry import LineString, Polygon

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.ml.pipeline_config import PipelineConfig
from core.reconstruct.spaces import SpacePoly


@dataclass
class GeometryValidationResult:
    """Result of geometry validation."""
    
    is_valid: bool
    warnings: list[str]
    errors: list[str]
    statistics: dict[str, Any]
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues that should block export."""
        return len(self.errors) > 0


def validate_reconstruction(
    normalized: list[NormalizedDet],
    wall_axes: list[WallAxis],
    spaces: list[SpacePoly],
    config: PipelineConfig | None = None,
) -> GeometryValidationResult:
    """
    Validate reconstructed geometry before IFC export.
    
    Checks:
    - Wall axes parallelism to source polygons
    - Space polygon closure (except for 1A)
    - Overlapping spaces (topology check)
    - Minimum room area
    - Wall axes quality (skeletonization score)
    
    Args:
        normalized: Normalized detections
        wall_axes: Wall axes from skeletonization
        spaces: Space polygons
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        GeometryValidationResult with validation status
    """
    if config is None:
        config = PipelineConfig.default()
    
    warnings: list[str] = []
    errors: list[str] = []
    stats = {
        "total_walls": 0,
        "valid_wall_axes": 0,
        "parallel_axes": 0,
        "non_parallel_axes": 0,
        "total_spaces": len(spaces),
        "closed_spaces": 0,
        "open_spaces": 0,
        "overlapping_spaces": 0,
        "spaces_below_min_area": 0,
        "wall_axes_quality_scores": [],
    }
    
    # Validate wall axes
    wall_detections = [det for det in normalized if det.type == "WALL"]
    stats["total_walls"] = len(wall_detections)
    
    for det_idx, det in enumerate(wall_detections):
        if det.geom is None:
            continue
        
        # Extract polygon from detection
        try:
            from shapely.geometry import mapping
            if hasattr(det.geom, 'geoms'):
                # MultiPolygon - use largest
                polygons = [p for p in det.geom.geoms if isinstance(p, Polygon)]
                if not polygons:
                    continue
                source_poly = max(polygons, key=lambda p: p.area)
            elif isinstance(det.geom, Polygon):
                source_poly = det.geom
            else:
                continue
            
            if source_poly.is_empty or not source_poly.is_valid:
                warnings.append(f"Wall {det_idx}: Invalid source polygon")
                continue
            
            # Find corresponding wall axes
            axes_for_wall = [ax for ax in wall_axes if ax.source_index == det_idx]
            
            if not axes_for_wall:
                warnings.append(f"Wall {det_idx}: No axes found")
                continue
            
            stats["valid_wall_axes"] += len(axes_for_wall)
            
            # Check parallelism for each axis
            for axis in axes_for_wall:
                if axis.line is None or axis.line.is_empty:
                    continue
                
                # Calculate angle of axis line
                coords = list(axis.line.coords)
                if len(coords) < 2:
                    continue
                
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                axis_length = math.hypot(dx, dy)
                
                if axis_length < 1e-6:
                    continue
                
                axis_angle = math.degrees(math.atan2(dy, dx))
                
                # Calculate angle of source polygon's longest edge
                source_bounds = source_poly.bounds
                source_width = source_bounds[2] - source_bounds[0]
                source_height = source_bounds[3] - source_bounds[1]
                
                # Determine dominant direction of source polygon
                if source_width > source_height:
                    source_angle = 0.0  # Horizontal
                else:
                    source_angle = 90.0  # Vertical
                
                # Check angle difference (normalize to 0-90 range)
                angle_diff = abs(axis_angle - source_angle) % 180.0
                if angle_diff > 90.0:
                    angle_diff = 180.0 - angle_diff
                
                # Tolerance: 10 degrees
                if angle_diff <= 10.0:
                    stats["parallel_axes"] += 1
                else:
                    stats["non_parallel_axes"] += 1
                    if angle_diff > 30.0:
                        warnings.append(
                            f"Wall {det_idx} axis: Large angle deviation ({angle_diff:.1f}°) "
                            f"from source polygon"
                        )
                
                # Calculate quality score (based on IoU if available)
                quality_score = 1.0
                if hasattr(axis, 'metadata') and isinstance(axis.metadata, dict):
                    iou = axis.metadata.get('iou', 1.0)
                    quality_score = float(iou)
                
                stats["wall_axes_quality_scores"].append(quality_score)
                
        except Exception as exc:
            logger.warning("Error validating wall %d: %s", det_idx, exc)
            warnings.append(f"Wall {det_idx}: Validation error: {exc}")
            continue
    
    # Validate spaces
    for space_idx, space in enumerate(spaces):
        if space.polygon is None:
            warnings.append(f"Space {space_idx}: No polygon")
            continue
        
        poly = space.polygon
        
        # Check 1: Closure (except for 1A - single-sided spaces)
        if poly.is_empty:
            errors.append(f"Space {space_idx}: Empty polygon")
            stats["open_spaces"] += 1
            continue
        
        if not poly.is_valid:
            # Try to repair
            try:
                repaired = poly.buffer(0)
                if isinstance(repaired, Polygon) and repaired.is_valid and not repaired.is_empty:
                    poly = repaired
                    warnings.append(f"Space {space_idx}: Repaired invalid polygon")
                else:
                    errors.append(f"Space {space_idx}: Invalid polygon (cannot repair)")
                    stats["open_spaces"] += 1
                    continue
            except Exception:
                errors.append(f"Space {space_idx}: Invalid polygon (repair failed)")
                stats["open_spaces"] += 1
                continue
        
        # Check if polygon is closed (exterior should form a closed ring)
        if len(poly.exterior.coords) < 4:  # At least 4 points (including duplicate last)
            errors.append(f"Space {space_idx}: Polygon has too few points")
            stats["open_spaces"] += 1
            continue
        
        # Check if first and last points are the same (closed)
        exterior_coords = list(poly.exterior.coords)
        if len(exterior_coords) > 1 and exterior_coords[0] != exterior_coords[-1]:
            # Not explicitly closed, but might still be valid
            warnings.append(f"Space {space_idx}: Polygon not explicitly closed")
            # Don't count as error - Shapely handles this
        
        stats["closed_spaces"] += 1
        
        # Check 2: Minimum area
        area_m2 = poly.area / 1_000_000.0  # Convert mm² to m²
        if area_m2 < config.min_room_area_m2:
            warnings.append(
                f"Space {space_idx}: Area {area_m2:.2f} m² below minimum "
                f"({config.min_room_area_m2} m²)"
            )
            stats["spaces_below_min_area"] += 1
        
        # Check 3: Overlapping spaces (topology check)
        for other_idx, other_space in enumerate(spaces):
            if other_idx == space_idx or other_space.polygon is None:
                continue
            
            other_poly = other_space.polygon
            if other_poly.is_empty:
                continue
            
            # Check for significant overlap (not just touching)
            if poly.intersects(other_poly):
                intersection = poly.intersection(other_poly)
                if isinstance(intersection, Polygon) and not intersection.is_empty:
                    intersection_area = intersection.area
                    # Consider overlap if > 1% of either space
                    overlap_ratio_self = intersection_area / poly.area if poly.area > 0 else 0
                    overlap_ratio_other = intersection_area / other_poly.area if other_poly.area > 0 else 0
                    
                    if overlap_ratio_self > 0.01 or overlap_ratio_other > 0.01:
                        warnings.append(
                            f"Space {space_idx} overlaps with space {other_idx}: "
                            f"{overlap_ratio_self*100:.1f}% / {overlap_ratio_other*100:.1f}%"
                        )
                        stats["overlapping_spaces"] += 1
    
    # Calculate overall statistics
    if stats["total_walls"] > 0:
        parallel_ratio = stats["parallel_axes"] / max(stats["valid_wall_axes"], 1)
        stats["parallel_ratio"] = parallel_ratio
        if parallel_ratio < 0.7:
            warnings.append(
                f"Low parallelism: Only {parallel_ratio*100:.1f}% of axes are parallel to source polygons"
            )
    
    if stats["wall_axes_quality_scores"]:
        avg_quality = sum(stats["wall_axes_quality_scores"]) / len(stats["wall_axes_quality_scores"])
        stats["average_quality_score"] = avg_quality
        if avg_quality < 0.7:
            warnings.append(f"Low wall axes quality: Average score {avg_quality:.2f}")
    
    if stats["total_spaces"] > 0:
        closed_ratio = stats["closed_spaces"] / stats["total_spaces"]
        stats["closed_ratio"] = closed_ratio
        if closed_ratio < 0.9:
            warnings.append(
                f"Low space closure: Only {closed_ratio*100:.1f}% of spaces are closed"
            )
    
    # Determine overall validity
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.warning("Geometry validation found %d errors", len(errors))
    
    if warnings:
        logger.info("Geometry validation found %d warnings", len(warnings))
    
    return GeometryValidationResult(
        is_valid=is_valid,
        warnings=warnings,
        errors=errors,
        statistics=stats,
    )

