"""Quality validators for merged plan."""

from typing import List, Dict, Any
from dataclasses import dataclass
import math

from .schema import CanonicalPlan, Wall, Opening, Room


@dataclass
class ValidationReport:
    """Quality validation report."""
    connectivity_issues: List[str]
    overlap_issues: List[str]
    room_issues: List[str]
    opening_issues: List[str]
    metrics: Dict[str, Any]


def validate_plan(plan: CanonicalPlan, tolerance_m: float = 0.01) -> ValidationReport:
    """Validate merged plan quality."""
    connectivity_issues = []
    overlap_issues = []
    room_issues = []
    opening_issues = []
    metrics = {
        "wall_count": len(plan.walls),
        "opening_count": len(plan.openings),
        "room_count": len(plan.rooms),
        "connected_walls": 0,
        "disconnected_walls": 0,
        "rooms_with_boundaries": 0,
        "openings_in_walls": 0,
    }
    
    # Build wall connectivity graph
    wall_map = {w.id: w for w in plan.walls}
    wall_endpoints: Dict[str, List[tuple]] = {}
    
    for wall in plan.walls:
        if len(wall.polyline) < 2:
            connectivity_issues.append(f"Wall {wall.id} has < 2 points")
            continue
        
        start = (wall.polyline[0].x, wall.polyline[0].y)
        end = (wall.polyline[-1].x, wall.polyline[-1].y)
        wall_endpoints[wall.id] = [start, end]
        
        # Check connections
        if wall.connections:
            metrics["connected_walls"] += 1
            for conn_id in wall.connections:
                if conn_id not in wall_map:
                    connectivity_issues.append(f"Wall {wall.id} references non-existent connection {conn_id}")
        else:
            metrics["disconnected_walls"] += 1
    
    # Check wall overlaps
    for i, wall1 in enumerate(plan.walls):
        if len(wall1.polyline) < 2:
            continue
        
        for wall2 in plan.walls[i+1:]:
            if len(wall2.polyline) < 2:
                continue
            
            # Check if walls are parallel and close
            start1 = wall1.polyline[0]
            end1 = wall1.polyline[-1]
            start2 = wall2.polyline[0]
            end2 = wall2.polyline[-1]
            
            dx1 = end1.x - start1.x
            dy1 = end1.y - start1.y
            dx2 = end2.x - start2.x
            dy2 = end2.y - start2.y
            
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            
            # Check angle
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, math.pi - angle_diff)
            
            if angle_diff < 0.1:  # Parallel
                # Check distance
                mid1 = ((start1.x + end1.x) / 2, (start1.y + end1.y) / 2)
                mid2 = ((start2.x + end2.x) / 2, (start2.y + end2.y) / 2)
                dist = math.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
                
                if dist < tolerance_m * 2:  # Very close parallel walls
                    overlap_issues.append(f"Walls {wall1.id} and {wall2.id} are parallel and very close ({dist:.3f}m)")
    
    # Validate openings
    for opening in plan.openings:
        host_wall = wall_map.get(opening.hostWallId)
        if not host_wall:
            opening_issues.append(f"Opening {opening.id} references non-existent wall {opening.hostWallId}")
            continue
        
        metrics["openings_in_walls"] += 1
        
        # Check opening position
        if opening.s < 0 or opening.s > 1:
            opening_issues.append(f"Opening {opening.id} has invalid position s={opening.s}")
        if opening.overallHeight is not None and opening.overallHeight <= 0:
            opening_issues.append(f"Opening {opening.id} has non-positive overallHeight {opening.overallHeight}")
        if opening.sillHeight is not None and opening.sillHeight < 0:
            opening_issues.append(f"Opening {opening.id} has negative sillHeight {opening.sillHeight}")
        if (
            opening.headHeight is not None
            and opening.sillHeight is not None
            and opening.headHeight < opening.sillHeight
        ):
            opening_issues.append(
                f"Opening {opening.id} headHeight {opening.headHeight} below sillHeight {opening.sillHeight}"
            )
        
        # Check opening is within wall bounds
        if len(host_wall.polyline) >= 2:
            start = host_wall.polyline[0]
            end = host_wall.polyline[-1]
            dx = end.x - start.x
            dy = end.y - start.y
            length = math.sqrt(dx * dx + dy * dy)
            
            opening_center_dist = opening.s * length
            half_width = opening.width / 2
            
            if opening_center_dist - half_width < -tolerance_m or \
               opening_center_dist + half_width > length + tolerance_m:
                opening_issues.append(f"Opening {opening.id} extends beyond wall {opening.hostWallId} bounds")
    
    # Validate rooms
    for room in plan.rooms:
        if len(room.polygon) < 3:
            room_issues.append(f"Room {room.id} has < 3 vertices")
            continue
        
        # Check polygon is closed
        first = room.polygon[0]
        last = room.polygon[-1]
        dist = math.sqrt((first.x - last.x) ** 2 + (first.y - last.y) ** 2)
        if dist > tolerance_m:
            room_issues.append(f"Room {room.id} polygon is not closed (gap: {dist:.3f}m)")
        
        # Check area
        if room.area <= 0:
            room_issues.append(f"Room {room.id} has non-positive area: {room.area}")
        
        # Check boundary walls
        if room.boundaryWallIds:
            metrics["rooms_with_boundaries"] += 1
            for wall_id in room.boundaryWallIds:
                if wall_id not in wall_map:
                    room_issues.append(f"Room {room.id} references non-existent boundary wall {wall_id}")
    
    return ValidationReport(
        connectivity_issues=connectivity_issues,
        overlap_issues=overlap_issues,
        room_issues=room_issues,
        opening_issues=opening_issues,
        metrics=metrics,
    )


def generate_qa_report(plan: CanonicalPlan, validation: ValidationReport) -> Dict[str, Any]:
    """Generate QA report JSON."""
    total_issues = (
        len(validation.connectivity_issues) +
        len(validation.overlap_issues) +
        len(validation.room_issues) +
        len(validation.opening_issues)
    )
    
    return {
        "summary": {
            "total_issues": total_issues,
            "connectivity_issues": len(validation.connectivity_issues),
            "overlap_issues": len(validation.overlap_issues),
            "room_issues": len(validation.room_issues),
            "opening_issues": len(validation.opening_issues),
        },
        "metrics": validation.metrics,
        "issues": {
            "connectivity": validation.connectivity_issues,
            "overlaps": validation.overlap_issues,
            "rooms": validation.room_issues,
            "openings": validation.opening_issues,
        },
        "plan_stats": {
            "walls": len(plan.walls),
            "openings": len(plan.openings),
            "rooms": len(plan.rooms),
            "scale": plan.metadata.scale,
            "image_size": f"{plan.metadata.imageWidth}x{plan.metadata.imageHeight}",
        },
    }

