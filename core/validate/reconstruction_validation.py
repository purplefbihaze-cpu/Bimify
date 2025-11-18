from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import ifcopenshell
import ifcopenshell.api
from ifcopenshell.util import element as ifc_element_utils
from shapely.geometry import LineString, MultiPolygon, Polygon

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.ml.pipeline_config import GapClosureMode, PipelineConfig
from core.exceptions import IFCExportBlockedError

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Overall quality metrics for IFC export validation."""
    overall_score: float  # 0.0-1.0
    geometric_accuracy: float  # 0.0-1.0
    completeness: float  # 0.0-1.0 (detected / expected elements)
    bim_compliance: float  # 0.0-1.0
    
    def is_acceptable(self, threshold: float = 0.75) -> bool:
        """Check if overall score meets acceptance threshold."""
        return self.overall_score >= threshold
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": round(self.overall_score, 4),
            "geometric_accuracy": round(self.geometric_accuracy, 4),
            "completeness": round(self.completeness, 4),
            "bim_compliance": round(self.bim_compliance, 4),
            "is_acceptable": self.is_acceptable(),
        }


@dataclass
class GapRepairProposal:
    """Proposal for gap repair without modifying geometry."""
    gap_id: str
    proposed_geometry: LineString
    confidence: float  # How confident is the repair? (0.0-1.0)
    width_mm: float  # Gap width in millimeters
    manual_review_required: bool  # True if gap > 50mm
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gap_id": self.gap_id,
            "proposed_geometry": {
                "type": "LineString",
                "coordinates": list(self.proposed_geometry.coords)
            },
            "confidence": round(self.confidence, 4),
            "width_mm": round(self.width_mm, 3),
            "manual_review_required": self.manual_review_required,
        }


@dataclass
class WallValidationRow:
    source_index: int
    axis_local_index: int
    wall_index: int | None
    confidence: float
    method: str
    iou_2d: float
    centroid_distance_mm: float
    angle_delta_deg: float
    detection_width_mm: float
    detection_length_mm: float
    axis_width_mm: float
    axis_length_mm: float
    ifc_width_mm: float | None = None
    ifc_length_mm: float | None = None
    thickness_delta_mm: float | None = None
    length_delta_mm: float | None = None
    score: float = 0.0
    status: str = "PENDING"

    def to_json(self) -> Dict[str, float | int | str | None]:
        return {
            "source_index": self.source_index,
            "axis_local_index": self.axis_local_index,
            "wall_index": self.wall_index,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "iou_2d": round(self.iou_2d, 4),
            "centroid_distance_mm": round(self.centroid_distance_mm, 3),
            "angle_delta_deg": round(self.angle_delta_deg, 3),
            "detection_width_mm": round(self.detection_width_mm, 3),
            "detection_length_mm": round(self.detection_length_mm, 3),
            "axis_width_mm": round(self.axis_width_mm, 3),
            "axis_length_mm": round(self.axis_length_mm, 3),
            "ifc_width_mm": None if self.ifc_width_mm is None else round(self.ifc_width_mm, 3),
            "ifc_length_mm": None if self.ifc_length_mm is None else round(self.ifc_length_mm, 3),
            "thickness_delta_mm": None if self.thickness_delta_mm is None else round(self.thickness_delta_mm, 3),
            "length_delta_mm": None if self.length_delta_mm is None else round(self.length_delta_mm, 3),
            "score": round(self.score, 2),
            "status": self.status,
        }


def _largest_polygon(geom: Polygon | MultiPolygon | LineString | None) -> Polygon | None:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return None


def _polygon_from_axis(axis: LineString, width: float) -> Polygon | None:
    coords = list(axis.coords)
    if len(coords) < 2:
        return None
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux, uy = dx / length, dy / length
    half = max(width / 2.0, 1.0)
    px, py = -uy, ux
    p1 = (x1 + px * half, y1 + py * half)
    p2 = (x2 + px * half, y2 + py * half)
    p3 = (x2 - px * half, y2 - py * half)
    p4 = (x1 - px * half, y1 - py * half)
    return Polygon([p1, p2, p3, p4])


def _oriented_lengths(poly: Polygon) -> Tuple[float, float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return (0.0, 0.0, 0.0)
    edges = []
    for idx in range(4):
        a = coords[idx]
        b = coords[(idx + 1) % 4]
        length = float(math.hypot(a[0] - b[0], a[1] - b[1]))
        edges.append((a, b, length))
    longest = max(edges, key=lambda item: item[2])
    shortest = min(edges, key=lambda item: item[2])
    angle = math.degrees(math.atan2(longest[1][1] - longest[0][1], longest[1][0] - longest[0][0]))
    angle = (angle + 360.0) % 180.0
    return (float(shortest[2]), float(longest[2]), float(angle))


def _angle_delta_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _score_component(delta: float, limit: float) -> float:
    ratio = min(abs(delta), limit) / limit
    return max(0.0, 1.0 - ratio)


def _find_validation_pset(model: ifcopenshell.file, wall) -> ifcopenshell.entity_instance | None:
    for rel in getattr(wall, "IsDefinedBy", []) or []:
        pset = getattr(rel, "RelatingPropertyDefinition", None)
        if pset is not None and getattr(pset, "Name", None) == "Bimify_Validation":
            return pset
    return None


def _extract_profile_dimensions(wall) -> Tuple[float | None, float | None]:
    representation = getattr(wall, "Representation", None)
    if representation is None:
        return (None, None)
    reps = getattr(representation, "Representations", []) or []
    for rep in reps:
        for item in getattr(rep, "Items", []) or []:
            if item.is_a("IfcExtrudedAreaSolid"):
                profile = getattr(item, "SweptArea", None)
                if profile is not None and profile.is_a("IfcRectangleProfileDef"):
                    x_dim = getattr(profile, "XDim", None)
                    y_dim = getattr(profile, "YDim", None)
                    return (float(x_dim) if x_dim is not None else None, float(y_dim) if y_dim is not None else None)
    return (None, None)


def _confidence(det: NormalizedDet) -> float:
    if isinstance(det.attrs, dict):
        value = det.attrs.get("confidence")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def auto_repair_gaps(
    wall_axes: Sequence[WallAxis],
    *,
    max_gap_mm: float = 50.0,
    mode: GapClosureMode = GapClosureMode.PROPOSE,
) -> Sequence[WallAxis] | List[GapRepairProposal] | None:
    """
    Automatically repair gaps > max_gap_mm in wall axes or return proposals.
    
    Args:
        wall_axes: Sequence of wall axes to check for gaps
        max_gap_mm: Maximum gap size to consider for repair
        mode: Gap closure mode (PROPOSE, REPAIR_AND_MARK, SILENT_REPAIR)
    
    Returns:
        - If mode=PROPOSE: List[GapRepairProposal] (no geometry changes)
        - If mode=REPAIR_AND_MARK or SILENT_REPAIR: Sequence[WallAxis] with repaired axes, or None if repair failed
    """
    from core.reconstruct.walls import close_wall_gaps, post_process_gap_closure
    from shapely.geometry import LineString, Point
    
    try:
        axis_lines = [axis_info.axis for axis_info in wall_axes if axis_info.axis]
        if len(axis_lines) < 2:
            return None if mode != GapClosureMode.PROPOSE else []
        
        # Build thickness mapping for adaptive gap closure
        thickness_by_index: Dict[int, float] = {}
        for idx, axis_info in enumerate(wall_axes):
            if axis_info.width_mm is not None and axis_info.width_mm > 0:
                thickness_by_index[idx] = float(axis_info.width_mm)
        
        # Detect gaps
        gaps_found = []
        for i, axis1 in enumerate(axis_lines):
            if axis1.length < 1e-3:
                continue
            coords1 = list(axis1.coords)
            if len(coords1) < 2:
                continue
            ep1_start = Point(coords1[0])
            ep1_end = Point(coords1[-1])
            
            for j, axis2 in enumerate(axis_lines[i + 1:], start=i + 1):
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
                if min_dist > 1.0 and min_dist <= max_gap_mm * 2.0:  # Only consider gaps up to 2x max_gap_mm
                    # Determine which endpoints are closest
                    min_idx = distances.index(min_dist)
                    if min_idx == 0:  # ep1_start to ep2_start
                        proposed_line = LineString([ep1_start, ep2_start])
                    elif min_idx == 1:  # ep1_start to ep2_end
                        proposed_line = LineString([ep1_start, ep2_end])
                    elif min_idx == 2:  # ep1_end to ep2_start
                        proposed_line = LineString([ep1_end, ep2_start])
                    else:  # ep1_end to ep2_end
                        proposed_line = LineString([ep1_end, ep2_end])
                    
                    gaps_found.append({
                        "axis1_idx": i,
                        "axis2_idx": j,
                        "distance": min_dist,
                        "proposed_line": proposed_line,
                    })
        
        # If mode is PROPOSE, return proposals without modifying geometry
        if mode == GapClosureMode.PROPOSE:
            proposals = []
            for gap in gaps_found:
                gap_id = f"gap_{gap['axis1_idx']}_{gap['axis2_idx']}"
                confidence = 1.0 - (gap['distance'] / (max_gap_mm * 2.0))  # Higher confidence for smaller gaps
                confidence = max(0.0, min(1.0, confidence))
                proposals.append(GapRepairProposal(
                    gap_id=gap_id,
                    proposed_geometry=gap['proposed_line'],
                    confidence=confidence,
                    width_mm=gap['distance'],
                    manual_review_required=gap['distance'] > 50.0,
                ))
            return proposals
        
        # Otherwise, perform actual repair
        # Attempt gap closure
        closed_axes = close_wall_gaps(
            axis_lines,
            gap_tolerance_mm=max_gap_mm * 0.5,
            max_gap_tolerance_mm=max_gap_mm * 2.0,
            thickness_by_index_mm=thickness_by_index if thickness_by_index else None,
        )
        
        # Post-process gap closure: Aggressive repair to guarantee all gaps ≤100mm are closed
        if len(closed_axes) > 1:
            closed_axes = post_process_gap_closure(
                closed_axes,
                thickness_by_index_mm=thickness_by_index if thickness_by_index else None,
            )
        
        # Update axes with closed versions and track gap repairs
        repaired_axes = []
        gap_repair_tracking: Dict[int, Dict[str, any]] = {}  # Track which axes were gap-repaired
        
        # Build gap repair tracking from gaps_found
        for gap in gaps_found:
            axis1_idx = gap['axis1_idx']
            axis2_idx = gap['axis2_idx']
            gap_width = gap['distance']
            # Mark both axes as gap-repaired
            gap_repair_tracking[axis1_idx] = {
                "original_gap_width_mm": gap_width,
                "closure_confidence": 1.0 - (gap_width / (max_gap_mm * 2.0)),
                "manual_review_required": gap_width > 50.0,
            }
            gap_repair_tracking[axis2_idx] = {
                "original_gap_width_mm": gap_width,
                "closure_confidence": 1.0 - (gap_width / (max_gap_mm * 2.0)),
                "manual_review_required": gap_width > 50.0,
            }
        
        for idx, axis_info in enumerate(wall_axes):
            if idx < len(closed_axes):
                axis_info.axis = closed_axes[idx]
                # Add gap repair metadata if this axis was gap-repaired
                if idx in gap_repair_tracking and mode == GapClosureMode.REPAIR_AND_MARK:
                    if axis_info.metadata is None:
                        axis_info.metadata = {}
                    axis_info.metadata["gap_repair_info"] = gap_repair_tracking[idx]
            repaired_axes.append(axis_info)
        
        return repaired_axes
    except Exception as e:
        logger.warning(f"Gap repair failed: {e}")
        return None


def validate_before_ifc_export(
    normalized: Sequence[NormalizedDet],
    wall_axes: Sequence[WallAxis],
    *,
    auto_repair: bool = True,  # Default to True (standard behavior)
    config: PipelineConfig | None = None,
) -> Tuple[bool, List[str], Sequence[WallAxis] | None, Dict[str, List[str]], List[GapRepairProposal] | None]:
    """
    Validate critical conditions before IFC export.
    Checks: Gaps, Thickness, Openings, Geometry validity.
    Returns (is_valid, warnings, repaired_axes, action_items, gap_proposals).
    """
    if config is None:
        config = PipelineConfig.default()
    
    warnings: List[str] = []
    action_items: Dict[str, List[str]] = {
        "repaired": [],
        "needs_attention": [],
        "critical": [],
    }
    repaired_axes: Sequence[WallAxis] | None = None
    gap_proposals: List[GapRepairProposal] | None = None
    
    # Check 1: Wall gaps (CRITICAL - ENHANCED)
    from shapely.geometry import Point, LineString
    
    gaps_found = []
    gaps_50_to_100mm = []
    gaps_over_100mm = []
    axis_lines = [axis_info.axis for axis_info in wall_axes if axis_info.axis]
    for i, axis1 in enumerate(axis_lines):
        if axis1.length < 1e-3:
            continue
        coords1 = list(axis1.coords)
        if len(coords1) < 2:
            continue
        ep1_start = Point(coords1[0])
        ep1_end = Point(coords1[-1])
        
        for j, axis2 in enumerate(axis_lines[i + 1:], start=i + 1):
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
            if min_dist > 50.0:
                gaps_found.append((i, j, min_dist))
                if 50.0 < min_dist <= 100.0:
                    gaps_50_to_100mm.append((i, j, min_dist))
                elif min_dist > 100.0:
                    gaps_over_100mm.append((i, j, min_dist))
    
    if gaps_found:
        max_gap = max(gap[2] for gap in gaps_found)
        gap_count = len(gaps_found)
        gap_50_100_count = len(gaps_50_to_100mm)
        gap_over_100_count = len(gaps_over_100mm)
        
        # Enhanced reporting: categorize gaps by severity
        if gap_50_100_count > 0:
            warnings.append(f"Found {gap_50_100_count} wall gaps between 50-100mm (BIM compliance issue, max: {max(gap[2] for gap in gaps_50_to_100mm):.1f}mm)")
        if gap_over_100_count > 0:
            warnings.append(f"Found {gap_over_100_count} wall gaps > 100mm (cannot guarantee repair, max: {max(gap[2] for gap in gaps_over_100mm):.1f}mm)")
        
        # Determine gap closure mode from config
        gap_mode = config.gap_closure_mode if config else GapClosureMode.PROPOSE
        
        if auto_repair and gap_mode != GapClosureMode.PROPOSE:
            # Enhanced: Repair gaps up to 100mm (BIM requirement)
            repaired_axes = auto_repair_gaps(wall_axes, max_gap_mm=100.0, mode=gap_mode)
            if repaired_axes:
                # Re-validate after repair
                remaining_gaps = []
                repaired_axis_lines = [axis_info.axis for axis_info in repaired_axes if axis_info.axis]
                for i, axis1 in enumerate(repaired_axis_lines):
                    if axis1.length < 1e-3:
                        continue
                    coords1 = list(axis1.coords)
                    if len(coords1) < 2:
                        continue
                    ep1_start = Point(coords1[0])
                    ep1_end = Point(coords1[-1])
                    for j, axis2 in enumerate(repaired_axis_lines[i + 1:], start=i + 1):
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
                        if min_dist > 50.0:
                            remaining_gaps.append((i, j, min_dist))
                
                if remaining_gaps:
                    remaining_50_100 = [g for g in remaining_gaps if 50.0 < g[2] <= 100.0]
                    if remaining_50_100:
                        action_items["critical"].append(f"{len(remaining_50_100)} gaps 50-100mm remain after repair - BIM compliance compromised")
                        warnings.append(f"CRITICAL: {len(remaining_50_100)} gaps 50-100mm remain after repair")
                    else:
                        action_items["repaired"].append(f"Automatically repaired {gap_count} wall gaps (max: {max_gap:.1f}mm)")
                        warnings.append(f"Automatically repaired {gap_count} wall gaps")
                else:
                    action_items["repaired"].append(f"Automatically repaired {gap_count} wall gaps (max: {max_gap:.1f}mm)")
                    warnings.append(f"Automatically repaired {gap_count} wall gaps - all gaps <= 50mm (BIM-compliant)")
            else:
                action_items["critical"].append(f"Failed to repair {gap_count} wall gaps - manual intervention required")
                warnings.append("Automatic gap repair failed - manual intervention required")
        elif gap_mode == GapClosureMode.PROPOSE:
            # Generate proposals without modifying geometry
            gap_proposals = auto_repair_gaps(wall_axes, max_gap_mm=100.0, mode=GapClosureMode.PROPOSE)
            if isinstance(gap_proposals, list):
                proposal_count = len(gap_proposals)
                manual_review_count = sum(1 for p in gap_proposals if p.manual_review_required)
                warnings.append(f"Found {proposal_count} wall gaps requiring review ({manual_review_count} require manual review)")
                action_items["needs_attention"].append(f"{proposal_count} gap repair proposals generated (no geometry modified)")
        else:
            if gap_50_100_count > 0:
                action_items["critical"].append(f"{gap_50_100_count} wall gaps 50-100mm need repair (BIM compliance issue, max: {max(gap[2] for gap in gaps_50_to_100mm):.1f}mm)")
            if gap_over_100_count > 0:
                action_items["critical"].append(f"{gap_over_100_count} wall gaps > 100mm need repair (max: {max(gap[2] for gap in gaps_over_100mm):.1f}mm)")
    
    # Check 2: Wall count (CRITICAL)
    wall_count = sum(1 for det in normalized if det.type == "WALL")
    if wall_count == 0:
        action_items["critical"].append("No walls detected - IFC model will be incomplete")
        warnings.append("No walls detected - IFC model will be incomplete")
    
    # Check 2b: Wall axes count (ENHANCED)
    axis_count = len([ax for ax in wall_axes if ax.axis and ax.axis.length >= 1e-3])
    if axis_count == 0 and wall_count > 0:
        if not (config and getattr(config, "preserve_exact_geometry", False)):
            action_items["critical"].append("No valid wall axes extracted - IFC model will be incomplete")
            warnings.append("No valid wall axes extracted - IFC model will be incomplete")
    elif axis_count < wall_count * 0.5:
        action_items["needs_attention"].append(f"Only {axis_count}/{wall_count} walls have valid axes - some walls may be missing")
        warnings.append(f"Only {axis_count}/{wall_count} walls have valid axes")
    
    # Check 3: Opening assignments (IMPORTANT)
    from core.reconstruct.openings import snap_openings_to_walls
    try:
        assignments, _ = snap_openings_to_walls(list(normalized), wall_axes=wall_axes if repaired_axes is None else repaired_axes)
        unmatched = sum(1 for ass in assignments if ass.wall_index is None)
        if unmatched > 0:
            action_items["needs_attention"].append(f"{unmatched} opening(s) could not be assigned to walls")
            warnings.append(f"{unmatched} opening(s) could not be assigned to walls")
    except Exception as opening_exc:
        action_items["needs_attention"].append(f"Could not validate opening assignments: {opening_exc}")
        warnings.append("Could not validate opening assignments")
    
    # Check 4: Wall thickness validation (ENHANCED - BIM compliance)
    invalid_thickness_count = 0
    thin_walls_count = 0
    thick_walls_count = 0
    for axis_info in (repaired_axes if repaired_axes is not None else wall_axes):
        if axis_info.width_mm is None or axis_info.width_mm <= 0.0:
            invalid_thickness_count += 1
        elif axis_info.width_mm < 40.0:
            invalid_thickness_count += 1
            thin_walls_count += 1
        elif axis_info.width_mm > 1000.0:
            invalid_thickness_count += 1
            thick_walls_count += 1
        elif axis_info.width_mm < 80.0:  # Warn for thin walls (40-80mm)
            thin_walls_count += 1
    
    if invalid_thickness_count > 0:
        action_items["needs_attention"].append(f"{invalid_thickness_count} wall(s) have invalid thickness values")
        warnings.append(f"{invalid_thickness_count} wall(s) have invalid thickness values")
    if thin_walls_count > 0:
        action_items["needs_attention"].append(f"{thin_walls_count} wall(s) have thickness < 80mm (may be below standard)")
        warnings.append(f"{thin_walls_count} wall(s) have thickness < 80mm")
    if thick_walls_count > 0:
        action_items["needs_attention"].append(f"{thick_walls_count} wall(s) have thickness > 1000mm (may be incorrect)")
        warnings.append(f"{thick_walls_count} wall(s) have thickness > 1000mm")
    
    # Check 5: Geometry validity (ENHANCED)
    invalid_geometry_count = 0
    short_axes_count = 0
    for axis_info in (repaired_axes if repaired_axes is not None else wall_axes):
        if axis_info.axis is None:
            invalid_geometry_count += 1
        elif axis_info.axis.length < 1e-3:
            invalid_geometry_count += 1
        elif axis_info.axis.length < 100.0:  # Warn for very short axes (< 100mm)
            short_axes_count += 1
    
    if invalid_geometry_count > 0:
        action_items["needs_attention"].append(f"{invalid_geometry_count} wall axis/axes have invalid geometry")
        warnings.append(f"{invalid_geometry_count} wall axis/axes have invalid geometry")
    if short_axes_count > 0:
        action_items["needs_attention"].append(f"{short_axes_count} wall axis/axes are very short (< 100mm) - may be artifacts")
        warnings.append(f"{short_axes_count} wall axis/axes are very short (< 100mm)")
    
    # Check 6: Opening validation (ENHANCED)
    opening_dets = [det for det in normalized if det.type in ("DOOR", "WINDOW")]
    if opening_dets:
        try:
            assignments, _ = snap_openings_to_walls(
                list(normalized), 
                wall_axes=wall_axes if repaired_axes is None else repaired_axes
            )
            unmatched = sum(1 for ass in assignments if ass.wall_index is None)
            if unmatched > 0:
                action_items["needs_attention"].append(f"{unmatched} opening(s) could not be assigned to walls")
                warnings.append(f"{unmatched} opening(s) could not be assigned to walls")
            
            # Check opening distances from walls
            far_openings = []
            for ass in assignments:
                if ass.wall_index is not None and ass.distance_mm > 500.0:
                    far_openings.append(ass)
            if far_openings:
                action_items["needs_attention"].append(f"{len(far_openings)} opening(s) are far from assigned walls (> 500mm)")
                warnings.append(f"{len(far_openings)} opening(s) are far from assigned walls")
        except Exception as opening_exc:
            action_items["needs_attention"].append(f"Could not validate opening assignments: {opening_exc}")
            warnings.append("Could not validate opening assignments")
    
    # Determine overall validity: critical issues block export
    critical_issues = len(action_items["critical"])
    is_valid = critical_issues == 0
    
    # Block export if critical gaps >50mm exist (BIM compliance requirement)
    gap_mode = config.gap_closure_mode if config else GapClosureMode.PROPOSE
    critical_gaps = [g for g in gaps_found if g[2] > 50.0]  # Gaps >50mm
    
    if critical_gaps and gap_mode in (GapClosureMode.REPAIR_AND_MARK, GapClosureMode.PROPOSE):
        gap_details = [f"Gap {g[0]}-{g[1]}: {g[2]:.1f}mm" for g in critical_gaps[:5]]
        error_msg = (
            f"Export blockiert: {len(critical_gaps)} kritische Gaps >50mm gefunden. "
            f"Manuelle Review erforderlich. Details: {', '.join(gap_details)}"
        )
        if len(critical_gaps) > 5:
            error_msg += f" (und {len(critical_gaps) - 5} weitere)"
        raise IFCExportBlockedError(error_msg)
    
    return is_valid, warnings, repaired_axes, action_items, gap_proposals


def generate_validation_report(
    normalized: Sequence[NormalizedDet],
    wall_axes: Sequence[WallAxis],
    ifc_path: Path,
    *,
    update_ifc: bool = True,
    auto_repair: bool = True,  # Default to True (standard behavior)
    config: PipelineConfig | None = None,
) -> Dict[str, object]:
    # Pre-validation: Check critical conditions and auto-repair (standard behavior)
    is_valid, pre_warnings, repaired_axes, action_items, gap_proposals = validate_before_ifc_export(
        normalized, wall_axes, auto_repair=auto_repair, config=config
    )
    
    # Use repaired axes if available
    effective_axes = repaired_axes if repaired_axes is not None else wall_axes
    
    rows: Dict[Tuple[int, int], WallValidationRow] = {}
    simple_mode = len(effective_axes) > 5000

    for axis_info in effective_axes:
        local_index_float = axis_info.metadata.get("axis_local_index")
        if local_index_float is None:
            continue
        source_index = int(axis_info.source_index)
        axis_local_index = int(round(local_index_float))
        det = axis_info.detection
        if simple_mode:
            det_polygon = None
            axis_polygon = None
            iou = 0.8
            centroid_distance = 0.0
            det_width = float(axis_info.width_mm)
            det_length = float(axis_info.axis.length) if axis_info.axis else 0.0
            det_angle = 0.0
        else:
            det_polygon = _largest_polygon(det.geom)
            axis_polygon = _polygon_from_axis(axis_info.axis, axis_info.width_mm) if axis_info.axis else None

            if det_polygon is None or det_polygon.is_empty or axis_polygon is None or axis_polygon.is_empty:
                iou = 0.0
                centroid_distance = 0.0
                det_width = 0.0
                det_length = 0.0
                det_angle = 0.0
            else:
                inter_area = axis_polygon.intersection(det_polygon).area
                union_area = axis_polygon.union(det_polygon).area
                iou = float(inter_area / union_area) if union_area > 1e-6 else 0.0
                centroid_distance = float(axis_polygon.centroid.distance(det_polygon.centroid))
                det_width, det_length, det_angle = _oriented_lengths(det_polygon)

        if det_polygon is None and not simple_mode:
            iou = 0.0
            centroid_distance = 0.0
            det_width = 0.0
            det_length = 0.0
            det_angle = 0.0

        axis_angle = 0.0
        coords = list(axis_info.axis.coords) if axis_info.axis else []
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            axis_angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0

        angle_delta = _angle_delta_deg(axis_angle, det_angle)

        row = WallValidationRow(
            source_index=source_index,
            axis_local_index=axis_local_index,
            wall_index=int(round(axis_info.metadata.get("wall_index", 0.0))) if "wall_index" in axis_info.metadata else None,
            confidence=_confidence(det),
            method=axis_info.method,
            iou_2d=float(max(0.0, min(1.0, iou))),
            centroid_distance_mm=float(centroid_distance),
            angle_delta_deg=float(angle_delta),
            detection_width_mm=float(det_width),
            detection_length_mm=float(det_length),
            axis_width_mm=float(axis_info.width_mm),
            axis_length_mm=float(axis_info.axis.length) if axis_info.axis else 0.0,
        )
        rows[(source_index, axis_local_index)] = row

    # IFC metrics
    model = None
    if ifc_path.exists():
        model = ifcopenshell.open(str(ifc_path))
        for wall in model.by_type("IfcWallStandardCase"):
            psets = ifc_element_utils.get_psets(wall, should_inherit=False)
            source_pset = psets.get("Bimify_SourceRoboflow") or {}
            source_index_val = source_pset.get("SourceIndex")
            local_index_val = source_pset.get("AxisLocalIndex")
            if source_index_val is None or local_index_val is None:
                continue
            try:
                key = (int(round(float(source_index_val))), int(round(float(local_index_val))))
            except (TypeError, ValueError):
                continue
            row = rows.get(key)
            if row is None:
                continue
            width_mm, length_mm = _extract_profile_dimensions(wall)
            if width_mm is not None:
                row.ifc_width_mm = float(width_mm)
            if length_mm is not None:
                row.ifc_length_mm = float(length_mm)

    # Final metrics and scoring
    scores: List[float] = []
    ious: List[float] = []
    fail_count = 0
    warn_count = 0
    pass_count = 0

    for row in rows.values():
        if row.ifc_width_mm is None:
            row.ifc_width_mm = row.axis_width_mm
        if row.ifc_length_mm is None:
            row.ifc_length_mm = row.axis_length_mm

        row.thickness_delta_mm = float(row.ifc_width_mm - row.detection_width_mm) if row.detection_width_mm else float(row.ifc_width_mm - row.axis_width_mm)
        row.length_delta_mm = float(row.ifc_length_mm - row.detection_length_mm) if row.detection_length_mm else float(row.ifc_length_mm - row.axis_length_mm)

        thickness_score = _score_component(row.thickness_delta_mm, 50.0)
        length_score = _score_component(row.length_delta_mm, 250.0)
        centroid_score = _score_component(row.centroid_distance_mm, 75.0)
        angle_score = _score_component(row.angle_delta_deg, 15.0)

        score_fraction = (
            0.4 * row.iou_2d
            + 0.2 * thickness_score
            + 0.2 * length_score
            + 0.1 * centroid_score
            + 0.1 * angle_score
        )
        row.score = float(max(0.0, min(1.0, score_fraction)) * 100.0)

        thickness_abs = abs(row.thickness_delta_mm)
        if row.iou_2d < 0.3 and thickness_abs > 80.0:
            row.status = "FAIL"
            fail_count += 1
        elif row.score >= 75.0 and row.iou_2d >= 0.55 and thickness_abs <= 60.0:
            row.status = "PASS"
            pass_count += 1
        else:
            row.status = "WARN"
            warn_count += 1

        scores.append(row.score)
        ious.append(row.iou_2d)

    # Summary
    total_walls = len(rows)
    passed = sum(1 for r in rows.values() if r.status == "PASS")
    warned = sum(1 for r in rows.values() if r.status == "WARN")
    failed = total_walls - passed - warned

    # Opening metrics (optional, derived from normalized dets vs fitted rectangles)
    opening_metrics: List[dict] = []
    try:
        for det in normalized:
            if det.type not in {"WINDOW", "DOOR"}:
                continue
            src_poly = _largest_polygon(det.geom)
            if src_poly is None or src_poly.is_empty:
                continue
            # Use oriented rectangle from det.geom (after fitter it will be axis-aligned)
            rect_poly = src_poly
            inter = rect_poly.intersection(src_poly).area
            union = rect_poly.union(src_poly).area
            iou = float(inter / union) if union > 1e-6 else 0.0
            opening_metrics.append({
                "type": det.type,
                "iou_2d": round(iou, 4),
                "area_src_mm2": round(src_poly.area, 2),
            })
    except Exception:
        opening_metrics = []

    # Calculate quality metrics
    # Geometric accuracy: based on average IoU and centroid distances
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    # Normalize centroid distances (assume max reasonable distance is 500mm)
    avg_centroid_dist = sum(abs(row.centroid_distance_mm) for row in rows.values()) / len(rows) if rows else 0.0
    centroid_score = max(0.0, 1.0 - (avg_centroid_dist / 500.0))  # 500mm = 0 score
    geometric_accuracy = (avg_iou * 0.7 + centroid_score * 0.3)  # Weight IoU more
    
    # Completeness: ratio of passed walls to total (with some tolerance for warnings)
    completeness = (passed + warned * 0.5) / total_walls if total_walls > 0 else 0.0
    
    # BIM compliance: based on validation status and IFC export success
    # Higher score if fewer critical issues and more passed validations
    critical_ratio = len(action_items["critical"]) / total_walls if total_walls > 0 else 1.0
    bim_compliance = max(0.0, 1.0 - critical_ratio * 2.0)  # Critical issues heavily penalize
    if passed > 0:
        bim_compliance = min(1.0, bim_compliance + (passed / total_walls) * 0.3)  # Bonus for passed
    
    # Overall score: weighted average
    overall_score = (
        geometric_accuracy * 0.4 +
        completeness * 0.3 +
        bim_compliance * 0.3
    )
    
    quality_report = QualityReport(
        overall_score=overall_score,
        geometric_accuracy=geometric_accuracy,
        completeness=completeness,
        bim_compliance=bim_compliance,
    )
    
    # Enhanced summary with action items and detailed metrics
    summary = {
        "total_walls": total_walls,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "score": int(round(sum(scores) / len(scores))) if scores else 0,
        "openings": opening_metrics,
        "pre_validation": {
            "is_valid": is_valid,
            "warnings": pre_warnings,
            "auto_repair_applied": repaired_axes is not None,
        },
        "action_items": {
            "repaired": action_items["repaired"],
            "needs_attention": action_items["needs_attention"],
            "critical": action_items["critical"],
        },
        "metrics": {
            "average_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "min_score": round(min(scores), 2) if scores else 0.0,
            "max_score": round(max(scores), 2) if scores else 0.0,
            "pass_rate": round(passed / total_walls * 100.0, 2) if total_walls > 0 else 0.0,
        },
        "quality": quality_report.to_dict(),
    }

    if update_ifc and model is not None:
        for wall in model.by_type("IfcWallStandardCase"):
            psets = ifc_element_utils.get_psets(wall, should_inherit=False)
            source_pset = psets.get("Bimify_SourceRoboflow") or {}
            source_index_val = source_pset.get("SourceIndex")
            local_index_val = source_pset.get("AxisLocalIndex")
            if source_index_val is None or local_index_val is None:
                continue
            try:
                key = (int(round(float(source_index_val))), int(round(float(local_index_val))))
            except (TypeError, ValueError):
                continue
            row = rows.get(key)
            if row is None:
                continue
            validation_props = {
                "Status": row.status,
                "Score": row.score,
                "IoU2D": row.iou_2d,
                "CentroidDistanceMm": row.centroid_distance_mm,
                "AngleDeltaDeg": row.angle_delta_deg,
                "DetectionWidthMm": row.detection_width_mm,
                "DetectionLengthMm": row.detection_length_mm,
                "AxisWidthMm": row.axis_width_mm,
                "AxisLengthMm": row.axis_length_mm,
                "IfcWidthMm": row.ifc_width_mm,
                "IfcLengthMm": row.ifc_length_mm,
                "ThicknessDeltaMm": row.thickness_delta_mm,
                "LengthDeltaMm": row.length_delta_mm,
            }
            pset = _find_validation_pset(model, wall)
            if pset is not None:
                ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=validation_props)
        model.write(str(ifc_path))

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "walls": [row.to_json() for row in rows.values()],
    }


def write_validation_report(
    report: Dict[str, object],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path

