"""Extract clean straight wall lines from Model 3."""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import math

from .parsers import ParsedWall


@dataclass
class CleanWallLine:
    """Clean straight wall line from Model 3."""
    id: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    angle: float  # radians
    length: float
    confidence: float
    isExternal: bool  # Will be inferred


def fit_line_to_points(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Fit line to points using least squares.
    
    Returns: (start_point, end_point, angle_rad)
    """
    if len(points) < 2:
        return points[0] if points else (0.0, 0.0), (0.0, 0.0), 0.0
    
    # Simple approach: use first and last point, or fit line
    if len(points) == 2:
        start, end = points[0], points[1]
    else:
        # Use endpoints of bounding box aligned with principal direction
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # Compute centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # Compute principal direction via covariance
        cov_xx = sum((x - cx) ** 2 for x in xs) / len(xs)
        cov_yy = sum((y - cy) ** 2 for y in ys) / len(ys)
        cov_xy = sum((xs[i] - cx) * (ys[i] - cy) for i in range(len(xs))) / len(xs)
        
        # Eigenvalue decomposition for 2x2
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        lambda1 = trace / 2 + math.sqrt(trace * trace / 4 - det)
        
        # Principal direction
        if abs(cov_xy) < 1e-6:
            angle = 0.0 if cov_xx > cov_yy else math.pi / 2
        else:
            angle = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
        
        # Project points onto line and find extent
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        projections = [(cos_a * (x - cx) + sin_a * (y - cy)) for x, y in points]
        min_proj = min(projections)
        max_proj = max(projections)
        
        start = (cx + cos_a * min_proj, cy + sin_a * min_proj)
        end = (cx + cos_a * max_proj, cy + sin_a * max_proj)
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    return start, end, angle


def extract_clean_walls(walls: List[ParsedWall], angle_tolerance: float = 0.1) -> List[CleanWallLine]:
    """Extract and cluster clean wall lines from Model 3."""
    clean_walls: List[CleanWallLine] = []
    
    for wall in walls:
        if len(wall.points) < 2:
            continue
        
        start, end, angle = fit_line_to_points(wall.points)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 1.0:  # Skip very short walls
            continue
        
        clean_walls.append(CleanWallLine(
            id=wall.id,
            start=start,
            end=end,
            angle=angle,
            length=length,
            confidence=wall.confidence,
            isExternal=wall.isExternal
        ))
    
    # Deduplicate overlapping walls
    # Simple approach: if two walls are very close and parallel, keep the one with higher confidence
    deduplicated: List[CleanWallLine] = []
    used = set()
    
    for i, wall1 in enumerate(clean_walls):
        if i in used:
            continue
        
        best_wall = wall1
        best_conf = wall1.confidence
        
        for j, wall2 in enumerate(clean_walls[i+1:], start=i+1):
            if j in used:
                continue
            
            # Check if parallel and close
            angle_diff = abs(wall1.angle - wall2.angle)
            angle_diff = min(angle_diff, math.pi - angle_diff)  # Handle wrap-around
            
            if angle_diff > angle_tolerance:
                continue
            
            # Check distance between lines
            # Simple: distance from midpoint of wall1 to line of wall2
            mid1 = ((wall1.start[0] + wall1.end[0]) / 2, (wall1.start[1] + wall1.end[1]) / 2)
            
            # Distance from point to line
            dx2 = wall2.end[0] - wall2.start[0]
            dy2 = wall2.end[1] - wall2.start[1]
            len2_sq = dx2 * dx2 + dy2 * dy2
            if len2_sq < 1e-6:
                continue
            
            # Project mid1 onto wall2 line
            t = ((mid1[0] - wall2.start[0]) * dx2 + (mid1[1] - wall2.start[1]) * dy2) / len2_sq
            proj = (wall2.start[0] + t * dx2, wall2.start[1] + t * dy2)
            dist = math.sqrt((mid1[0] - proj[0]) ** 2 + (mid1[1] - proj[1]) ** 2)
            
            if dist < 20.0:  # Within 20 pixels
                # They overlap, keep the one with higher confidence
                if wall2.confidence > best_conf:
                    best_wall = wall2
                    best_conf = wall2.confidence
                    used.add(i)
                    used.add(j)
                else:
                    used.add(j)
        
        if i not in used:
            deduplicated.append(best_wall)
    
    return deduplicated

