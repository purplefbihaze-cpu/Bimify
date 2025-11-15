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
    Enhanced with adaptive tolerance, validation, and fallback mechanisms.

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
        
        # Calculate adaptive tolerance based on polygon size (smaller for thin walls)
        # Estimate wall thickness from polygon's minimum rotated rectangle
        adaptive_tolerance = tolerance
        if tolerance > 0:
            try:
                rect = poly.minimum_rotated_rectangle
                if not rect.is_empty:
                    coords = list(rect.exterior.coords)
                    if len(coords) >= 4:
                        # Calculate edge lengths
                        edges = []
                        for i in range(4):
                            x1, y1 = coords[i]
                            x2, y2 = coords[(i + 1) % 4]
                            edge_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                            edges.append(edge_len)
                        # Shortest edge approximates wall thickness
                        min_edge = min(edges) if edges else 0.0
                        # Use smaller tolerance for thin walls (thickness < 300mm)
                        if 0.0 < min_edge < 300.0:
                            adaptive_tolerance = min(tolerance, min_edge * 0.1)
            except Exception:
                adaptive_tolerance = tolerance
        
        candidate = poly
        if adaptive_tolerance > 0:
            try:
                simplified = candidate.simplify(adaptive_tolerance, preserve_topology=True)
                # Validate: check if simplified geometry is valid
                if not simplified.is_empty and simplified.is_valid:
                    candidate = simplified
                else:
                    # Fallback: use original with buffer(0) repair
                    candidate = poly.buffer(0)
            except Exception:
                # Fallback: if simplify fails, use original with buffer(0) repair
                try:
                    candidate = poly.buffer(0)
                except Exception:
                    candidate = poly
        
        # Validate and repair geometry
        if not candidate.is_valid:
            try:
                candidate = candidate.buffer(0)
            except Exception:
                pass
        
        # Final validation: ensure geometry is valid and not empty
        if candidate.is_empty or not candidate.is_valid:
            # Try buffer(0) as last resort
            try:
                repaired = poly.buffer(0)
                if not repaired.is_empty and repaired.is_valid:
                    candidate = repaired
                else:
                    continue  # Skip invalid polygons
            except Exception:
                continue  # Skip polygons that can't be repaired
        
        if isinstance(candidate, Polygon):
            cleaned.append(candidate)
        elif isinstance(candidate, MultiPolygon):
            cleaned.extend([p for p in candidate.geoms if not p.is_empty and p.is_valid])

    if not cleaned:
        return []

    # Multi-stage validation: Before merge, after merge, after snap
    # Stage 1: Pre-merge validation - ensure all polygons are valid
    pre_merge_valid = []
    for poly in cleaned:
        if poly.is_valid and not poly.is_empty:
            pre_merge_valid.append(poly)
        else:
            # Aggressive repair: try buffer(0) and simplify
            try:
                repaired = poly.buffer(0)
                if isinstance(repaired, Polygon) and repaired.is_valid and not repaired.is_empty:
                    pre_merge_valid.append(repaired)
                elif isinstance(repaired, MultiPolygon):
                    for part in repaired.geoms:
                        if isinstance(part, Polygon) and part.is_valid and not part.is_empty:
                            pre_merge_valid.append(part)
            except Exception:
                # Last resort: try simplify with very small tolerance
                try:
                    simplified = poly.simplify(0.1, preserve_topology=True)
                    if simplified.is_valid and not simplified.is_empty:
                        pre_merge_valid.append(simplified)
                except Exception:
                    pass
    
    if not pre_merge_valid:
        return []
    
        # Stage 2: Merge with validation
    try:
        merged = unary_union(pre_merge_valid)
        
        # Stage 3: Post-merge validation (ENHANCED - validate after merge)
        if merged.is_empty:
            return []
        
        # Enhanced validation: Check geometry validity and log issues
        if not merged.is_valid:
            # Aggressive repair: multiple attempts with validation after each
            repair_attempts = [
                lambda g: g.buffer(0),  # Standard repair
                lambda g: g.buffer(0.1).buffer(-0.1),  # Buffer in/out
                lambda g: g.simplify(1.0, preserve_topology=True).buffer(0),  # Simplify + repair
            ]
            
            for repair_idx, repair_func in enumerate(repair_attempts):
                try:
                    repaired = repair_func(merged)
                    # Validate after each repair attempt
                    if not repaired.is_empty and repaired.is_valid:
                        merged = repaired
                        # Additional validation: check polygon is closed and has valid coordinates
                        if isinstance(merged, Polygon):
                            coords = list(merged.exterior.coords)
                            if len(coords) >= 3:
                                # Check if polygon is closed (first and last points should be same or very close)
                                if len(coords) >= 2:
                                    first = coords[0]
                                    last = coords[-1]
                                    dist = ((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2) ** 0.5
                                    if dist > 1.0:  # Not closed - try to close it
                                        coords.append(coords[0])
                                        try:
                                            merged = Polygon(coords)
                                        except Exception:
                                            pass
                        break
                except Exception:
                    continue
        
        # Stage 4: Snap vertices if requested (with enhanced validation)
        if snap_tolerance > 0:
            try:
                snapped = snap(merged, merged, snap_tolerance)
                # Enhanced validation: Check snapped geometry validity and structure
                if not snapped.is_empty and snapped.is_valid:
                    # Additional validation: verify polygon structure after snap
                    if isinstance(snapped, Polygon):
                        coords = list(snapped.exterior.coords)
                        if len(coords) >= 3:
                            # Check if polygon is still closed after snap
                            if len(coords) >= 2:
                                first = coords[0]
                                last = coords[-1]
                                dist = ((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2) ** 0.5
                                if dist <= snap_tolerance * 2.0:  # Allow tolerance for snap
                                    merged = snapped
                                else:
                                    # Not closed - try to close it
                                    try:
                                        coords.append(coords[0])
                                        merged = Polygon(coords)
                                    except Exception:
                                        # Snap may have broken geometry - try repair
                                        merged = merged.buffer(0)
                            else:
                                merged = snapped
                        else:
                            merged = snapped
                    else:
                        merged = snapped
                elif not merged.is_valid:
                    # Snap may have broken geometry - try repair
                    try:
                        repaired = merged.buffer(0)
                        if not repaired.is_empty and repaired.is_valid:
                            merged = repaired
                    except Exception:
                        pass
            except Exception:
                # Snap failed - continue with merged geometry
                pass
        
        # Stage 5: Final validation and aggressive repair
        if merged.is_empty:
            return []
        
        if not merged.is_valid:
            # Final aggressive repair attempts
            final_repair_attempts = [
                lambda g: g.buffer(0),
                lambda g: g.buffer(0.5).buffer(-0.5),
                lambda g: g.simplify(2.0, preserve_topology=True),
            ]
            
            for repair_func in final_repair_attempts:
                try:
                    repaired = repair_func(merged)
                    if not repaired.is_empty and repaired.is_valid:
                        merged = repaired
                        break
                except Exception:
                    continue
            
            # If still invalid, try to extract valid parts
            if not merged.is_valid:
                try:
                    if isinstance(merged, MultiPolygon):
                        valid_parts = [p for p in merged.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                        if valid_parts:
                            if len(valid_parts) == 1:
                                merged = valid_parts[0]
                            else:
                                merged = MultiPolygon(valid_parts)
                    else:
                        # Try buffer(0) one more time
                        merged = merged.buffer(0)
                except Exception:
                    return []
    except Exception:
        return []

    if isinstance(merged, Polygon):
        return [merged] if merged.is_valid and not merged.is_empty else []
    if isinstance(merged, MultiPolygon):
        return [poly for poly in merged.geoms if poly.is_valid and not poly.is_empty]

    return []

