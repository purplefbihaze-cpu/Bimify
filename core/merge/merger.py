"""Merge Model 1, 2, 3 outputs into canonical JSON."""

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import math
import logging

from .parsers import parse_model1, parse_model2, parse_model3, ParsedWall, ParsedOpening, ParsedRoom
from .wall_graph import WallGraph
from .wall_cleaner import extract_clean_walls, CleanWallLine
from .schema import CanonicalPlan, Wall, Opening, Room, Point2D, Metadata
from .alignment import check_coregistration
from config.ifc_standards import STANDARDS

logger = logging.getLogger(__name__)


def pixel_to_meter(points: List[Tuple[float, float]], scale: float) -> List[Point2D]:
    """Convert pixel coordinates to meters."""
    return [Point2D(x=p[0] * scale, y=p[1] * scale) for p in points]


def compute_wall_thickness(wall: ParsedWall, default_internal: float = 0.115, default_external: float = 0.24) -> float:
    """Estimate wall thickness."""
    # For now, use defaults based on isExternal
    # TODO: Could analyze double-line patterns in Model 3
    return default_external if wall.isExternal else default_internal


def match_m3_to_m1_edge(
    m1_edge_id: str,
    m1_start: Tuple[float, float],
    m1_end: Tuple[float, float],
    m3_walls: List[CleanWallLine],
    angle_tolerance: float = 0.2,
    distance_tolerance: float = 30.0
) -> Optional[CleanWallLine]:
    """Find best matching M3 wall for M1 edge."""
    m1_dx = m1_end[0] - m1_start[0]
    m1_dy = m1_end[1] - m1_start[1]
    m1_angle = math.atan2(m1_dy, m1_dx)
    m1_mid = ((m1_start[0] + m1_end[0]) / 2, (m1_start[1] + m1_end[1]) / 2)
    m1_len = math.sqrt(m1_dx * m1_dx + m1_dy * m1_dy)
    
    best_match = None
    best_score = -1.0
    
    for m3_wall in m3_walls:
        # Check angle similarity
        angle_diff = abs(m1_angle - m3_wall.angle)
        angle_diff = min(angle_diff, math.pi - angle_diff)
        if angle_diff > angle_tolerance:
            continue
        
        # Check distance from M1 midpoint to M3 line
        m3_dx = m3_wall.end[0] - m3_wall.start[0]
        m3_dy = m3_wall.end[1] - m3_wall.start[1]
        m3_len_sq = m3_dx * m3_dx + m3_dy * m3_dy
        if m3_len_sq < 1e-6:
            continue
        
        # Project M1 mid onto M3 line
        t = ((m1_mid[0] - m3_wall.start[0]) * m3_dx + (m1_mid[1] - m3_wall.start[1]) * m3_dy) / m3_len_sq
        proj = (m3_wall.start[0] + t * m3_dx, m3_wall.start[1] + t * m3_dy)
        dist = math.sqrt((m1_mid[0] - proj[0]) ** 2 + (m1_mid[1] - proj[1]) ** 2)
        
        if dist > distance_tolerance:
            continue
        
        # Check length overlap
        len_ratio = min(m1_len, m3_wall.length) / max(m1_len, m3_wall.length)
        
        # Composite score
        score = m3_wall.confidence * len_ratio * (1.0 - dist / distance_tolerance) * (1.0 - angle_diff / angle_tolerance)
        
        if score > best_score:
            best_score = score
            best_match = m3_wall
    
    return best_match


def project_opening_to_wall(
    opening: ParsedOpening,
    wall_start: Tuple[float, float],
    wall_end: Tuple[float, float],
    wall_length: float
) -> Optional[float]:
    """Project opening bbox center onto wall axis, return s in [0,1]."""
    # Opening center
    ox = opening.bbox[0] + opening.bbox[2] / 2
    oy = opening.bbox[1] + opening.bbox[3] / 2
    
    # Wall vector
    wx = wall_end[0] - wall_start[0]
    wy = wall_end[1] - wall_start[1]
    
    if wall_length < 1e-6:
        return None
    
    # Project opening center onto wall
    t = ((ox - wall_start[0]) * wx + (oy - wall_start[1]) * wy) / (wall_length * wall_length)
    
    # Clamp to [0, 1]
    s = max(0.0, min(1.0, t))
    
    return s


def _opening_vertical_defaults(opening_type: str, wall_thickness_m: float) -> Dict[str, float]:
    """Return default sill/head/height/depth values for an opening."""
    base_depth = wall_thickness_m if wall_thickness_m and wall_thickness_m > 0 else STANDARDS.get("WALL_INTERNAL_THICKNESS", 0.115)
    if opening_type == "door":
        sill = STANDARDS.get("DOOR_SILL_HEIGHT", 0.0)
        overall = STANDARDS.get("DOOR_HEIGHT", 2.0)
        head = STANDARDS.get("DOOR_HEAD_HEIGHT", sill + overall)
    else:
        sill = STANDARDS.get("WINDOW_SILL_HEIGHT", 0.9)
        overall = STANDARDS.get("WINDOW_OVERALL_HEIGHT", STANDARDS.get("WINDOW_HEIGHT", 1.2))
        head = STANDARDS.get("WINDOW_HEAD_HEIGHT", sill + overall)
    head = max(head, sill + overall)
    return {
        "sill": sill,
        "overall": overall,
        "head": head,
        "depth": base_depth,
    }


def _create_opening_model(
    opening_id: str,
    opening_type: str,
    host_wall: Wall,
    s_param: float,
    width_m: float,
    height_m: float,
    confidence: float,
    swing_direction: Optional[str] = None,
) -> Opening:
    """Instantiate Opening with vertical defaults applied."""
    defaults = _opening_vertical_defaults(opening_type, host_wall.thickness)
    overall_height = defaults["overall"]
    sill_height = defaults["sill"]
    head_height = defaults["head"]
    depth = defaults["depth"]
    if head_height < sill_height + overall_height:
        head_height = sill_height + overall_height
    return Opening(
        id=opening_id,
        type=opening_type,
        hostWallId=host_wall.id,
        s=max(0.0, min(1.0, s_param)),
        width=width_m,
        height=height_m,
        confidence=confidence,
        swingDirection=swing_direction,
        sillHeight=sill_height,
        headHeight=head_height,
        overallHeight=overall_height,
        depth=depth,
    )


def merge_models(
    model1_path: Path,
    model2_path: Path,
    model3_path: Path,
    px_to_meter: float = 0.001,  # Default: 1mm per pixel
    snap_tolerance_px: float = 5.0
) -> CanonicalPlan:
    """Merge all three models into canonical plan."""
    
    # Parse all models
    m1_walls, m1_openings, m1_meta = parse_model1(model1_path)
    m2_rooms, m2_meta = parse_model2(model2_path)
    m3_walls, m3_openings, m3_meta = parse_model3(model3_path)
    
    # Check co-registration
    is_aligned, transform_info = check_coregistration(m1_meta, m2_meta, m3_meta)
    if not is_aligned:
        logger.warning(f"Models not perfectly aligned: {transform_info}")
        # For now, we'll proceed but could apply transform if needed
    
    width = m1_meta.get("width", 0)
    height = m1_meta.get("height", 0)
    if abs(width - m2_meta.get("width", 0)) > 5 or abs(height - m2_meta.get("height", 0)) > 5:
        raise ValueError(f"Model 1 and 2 image dimensions don't match: {width}x{height} vs {m2_meta.get('width')}x{m2_meta.get('height')}")
    if abs(width - m3_meta.get("width", 0)) > 5 or abs(height - m3_meta.get("height", 0)) > 5:
        raise ValueError(f"Model 1 and 3 image dimensions don't match: {width}x{height} vs {m3_meta.get('width')}x{m3_meta.get('height')}")
    
    # Build M1 wall graph
    graph = WallGraph(snap_tolerance=snap_tolerance_px)
    for wall in m1_walls:
        graph.add_wall(wall)
    
    # Extract clean M3 walls (only for reference, not for geometry)
    clean_m3_walls = extract_clean_walls(m3_walls)
    logger.info(f"M1 walls: {len(graph.edges)}, M3 clean walls: {len(clean_m3_walls)}")
    
    # Merge walls: M1 is PRIMARY - use M1 topology and geometry
    # M3 is only used for validation/reference, NOT for geometry
    merged_walls: List[Wall] = []
    
    for edge_id, edge in graph.edges.items():
        # ALWAYS use M1 geometry - it has the correct topology and snapping
        centerline = graph.get_centerline(edge_id)
        
        if not centerline or len(centerline) < 2:
            logger.warning(f"Edge {edge_id} has invalid centerline, skipping")
            continue
        
        m1_start = (graph.nodes[edge.start_node].x, graph.nodes[edge.start_node].y)
        m1_end = (graph.nodes[edge.end_node].x, graph.nodes[edge.end_node].y)
        
        # Check if M3 has a matching wall (for validation only)
        m3_match = match_m3_to_m1_edge(edge_id, m1_start, m1_end, clean_m3_walls)
        
        if m3_match:
            logger.debug(f"Edge {edge_id} has M3 match (confidence: {m3_match.confidence:.2f}), but using M1 geometry")
            # We could use M3's isExternal flag if it's more reliable
            is_ext = m3_match.isExternal if hasattr(m3_match, 'isExternal') else edge.isExternal
        else:
            logger.debug(f"Edge {edge_id} has no M3 match, using M1 only")
            is_ext = edge.isExternal
        
        # Convert to meters
        centerline_m = pixel_to_meter(centerline, px_to_meter)
        
        # Get thickness from M1 (M3 is only reference)
        thickness = compute_wall_thickness(edge, default_internal=0.115, default_external=0.24)
        
        # Get connections from M1 graph
        connections = graph.get_connections(edge_id)
        
        merged_walls.append(Wall(
            id=edge_id,
            polyline=centerline_m,
            thickness=thickness,
            isExternal=is_ext,
            connections=connections,
            confidence=edge.confidence  # Use M1 confidence as primary
        ))
    
    logger.info(f"Merged {len(merged_walls)} walls from M1 topology")
    
    # Place openings: M3 is PRIMARY for openings (better detection)
    # But they must be projected onto M1 walls
    merged_openings: List[Opening] = []
    used_openings = set()
    
    logger.info(f"Processing {len(m3_openings)} M3 openings and {len(m1_openings)} M1 openings")
    
    # First, process M3 openings (better detection quality)
    for m3_opening in m3_openings:
        if m3_opening.id in used_openings:
            continue
        
        # Find nearest wall
        best_wall = None
        best_s = None
        best_dist = float('inf')
        
        ox = m3_opening.bbox[0] + m3_opening.bbox[2] / 2
        oy = m3_opening.bbox[1] + m3_opening.bbox[3] / 2
        
        for wall in merged_walls:
            if len(wall.polyline) < 2:
                continue
            
            start = (wall.polyline[0].x / px_to_meter, wall.polyline[0].y / px_to_meter)
            end = (wall.polyline[-1].x / px_to_meter, wall.polyline[-1].y / px_to_meter)
            
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            
            s = project_opening_to_wall(m3_opening, start, end, length)
            if s is None:
                continue
            
            # Distance from opening center to wall line
            proj_x = start[0] + s * dx
            proj_y = start[1] + s * dy
            dist = math.sqrt((ox - proj_x) ** 2 + (oy - proj_y) ** 2)
            
            # Increase tolerance for opening placement (100 pixels = ~10cm at 0.001 scale)
            if dist < best_dist and dist < 100.0:  # Within 100 pixels
                best_dist = dist
                best_wall = wall
                best_s = s
        
        # Topology-Awareness: If M3 doesn't fit well (>20px = ~2cm), try M1 alternative
        if best_dist > 20.0 and best_wall:  # M3 passt nicht gut
            # Suche M1-Alternative für dieselbe Wand
            m1_alternative = None
            m1_best_dist = float('inf')
            m1_best_s = None
            
            for m1_opening in m1_openings:
                if m1_opening.id in used_openings:
                    continue
                
                # Prüfe ob M1 besser auf Wand passt
                m1_ox = m1_opening.bbox[0] + m1_opening.bbox[2] / 2
                m1_oy = m1_opening.bbox[1] + m1_opening.bbox[3] / 2
                
                start = (best_wall.polyline[0].x / px_to_meter, best_wall.polyline[0].y / px_to_meter)
                end = (best_wall.polyline[-1].x / px_to_meter, best_wall.polyline[-1].y / px_to_meter)
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx * dx + dy * dy)
                
                m1_s = project_opening_to_wall(m1_opening, start, end, length)
                if m1_s is None:
                    continue
                
                m1_proj_x = start[0] + m1_s * dx
                m1_proj_y = start[1] + m1_s * dy
                m1_dist = math.sqrt((m1_ox - m1_proj_x) ** 2 + (m1_oy - m1_proj_y) ** 2)
                
                if m1_dist < 20.0 and m1_opening.confidence > 0.85 and m1_dist < m1_best_dist:
                    m1_alternative = m1_opening
                    m1_best_dist = m1_dist
                    m1_best_s = m1_s
            
            if m1_alternative:
                logger.info(f"Using M1 opening {m1_alternative.id} (better fit, dist={m1_best_dist:.1f}px) instead of M3 {m3_opening.id} (dist={best_dist:.1f}px)")
                width_m = m1_alternative.bbox[2] * px_to_meter
                height_m = m1_alternative.bbox[3] * px_to_meter
                
                merged_openings.append(Opening(
                    id=m1_alternative.id,
                    type=m1_alternative.type,
                    hostWallId=best_wall.id,
                    s=m1_best_s,
                    width=width_m,
                    height=height_m,
                    confidence=m1_alternative.confidence
                ))
                used_openings.add(m1_alternative.id)
                continue  # Skip M3, use M1 instead
        
        if best_wall:
            width_m = m3_opening.bbox[2] * px_to_meter
            height_m = m3_opening.bbox[3] * px_to_meter
            
            merged_openings.append(
                _create_opening_model(
                    opening_id=m3_opening.id,
                    opening_type=m3_opening.type,
                    host_wall=best_wall,
                    s_param=best_s,
                    width_m=width_m,
                    height_m=height_m,
                    confidence=m3_opening.confidence,
                )
            )
            used_openings.add(m3_opening.id)
            logger.debug(f"M3 opening {m3_opening.id} ({m3_opening.type}) placed on wall {best_wall.id} at s={best_s:.3f}")
        else:
            logger.warning(f"M3 opening {m3_opening.id} ({m3_opening.type}) could not be placed on any M1 wall (best_dist={best_dist:.1f}px)")
    
    # Fallback: M1 openings not matched in M3 (verwaiste M1-Öffnungen)
    # Prüfe ob M1-Öffnung bereits durch M3 abgedeckt ist (IOU > 0.5)
    def calculate_iou(bbox1, bbox2):
        """Calculate Intersection over Union for two bounding boxes."""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    for m1_opening in m1_openings:
        if m1_opening.id in used_openings:
            continue
        
        # Prüfe ob M1-Öffnung bereits durch M3 abgedeckt ist
        is_covered_by_m3 = False
        for m3_opening in m3_openings:
            iou = calculate_iou(m1_opening.bbox, m3_opening.bbox)
            if iou > 0.5:  # M3 deckt M1 ab
                is_covered_by_m3 = True
                break
        
        if is_covered_by_m3:
            continue  # Überspringe, da bereits durch M3 abgedeckt
        
        # Same matching logic
        best_wall = None
        best_s = None
        best_dist = float('inf')
        
        ox = m1_opening.bbox[0] + m1_opening.bbox[2] / 2
        oy = m1_opening.bbox[1] + m1_opening.bbox[3] / 2
        
        for wall in merged_walls:
            if len(wall.polyline) < 2:
                continue
            
            start = (wall.polyline[0].x / px_to_meter, wall.polyline[0].y / px_to_meter)
            end = (wall.polyline[-1].x / px_to_meter, wall.polyline[-1].y / px_to_meter)
            
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            
            s = project_opening_to_wall(m1_opening, start, end, length)
            if s is None:
                continue
            
            proj_x = start[0] + s * dx
            proj_y = start[1] + s * dy
            dist = math.sqrt((ox - proj_x) ** 2 + (oy - proj_y) ** 2)
            
            if dist < best_dist and dist < 50.0:
                best_dist = dist
                best_wall = wall
                best_s = s
        
        if best_wall:
            width_m = m1_opening.bbox[2] * px_to_meter
            height_m = m1_opening.bbox[3] * px_to_meter
            
            merged_openings.append(
                _create_opening_model(
                    opening_id=m1_opening.id,
                    opening_type=m1_opening.type,
                    host_wall=best_wall,
                    s_param=best_s,
                    width_m=width_m,
                    height_m=height_m,
                    confidence=m1_opening.confidence,
                )
            )
            used_openings.add(m1_opening.id)
            logger.debug(f"M1 opening {m1_opening.id} ({m1_opening.type}) placed on wall {best_wall.id} at s={best_s:.3f}")
    
    # Count openings by source (check if ID contains model indicator or check source)
    m3_count = 0
    m1_count = 0
    for o in merged_openings:
        # M3 openings typically have detection_id from model 3, M1 from model 1
        # For now, count by order: first processed are M3
        if len([x for x in merged_openings if merged_openings.index(x) < merged_openings.index(o)]) < len(m3_openings):
            m3_count += 1
        else:
            m1_count += 1
    
    logger.info(f"Placed {len(merged_openings)} openings ({m3_count} from M3, {m1_count} from M1)")
    
    # Integrate rooms from M2 (M2 is PRIMARY for rooms)
    merged_rooms: List[Room] = []
    
    logger.info(f"Processing {len(m2_rooms)} rooms from M2")
    
    for room in m2_rooms:
        polygon_m = pixel_to_meter(room.polygon, px_to_meter)
        
        # Compute area (shoelace formula)
        area = 0.0
        n = len(polygon_m)
        for i in range(n):
            j = (i + 1) % n
            area += polygon_m[i].x * polygon_m[j].y
            area -= polygon_m[j].x * polygon_m[i].y
        area = abs(area) / 2.0
        
        # Find boundary walls: walls that are close to room polygon edges
        boundary_wall_ids = []
        from shapely.geometry import Polygon as ShapelyPolygon, LineString, Point
        
        try:
            room_poly = ShapelyPolygon([(p.x, p.y) for p in polygon_m])
            if not room_poly.is_valid:
                room_poly = room_poly.buffer(0)  # Fix invalid polygon
            
            # Buffer room polygon slightly to find nearby walls
            room_buffer = room_poly.buffer(0.5)  # 0.5m buffer
            
            for wall in merged_walls:
                if len(wall.polyline) < 2:
                    continue
                
                # Create wall line
                wall_line = LineString([(p.x, p.y) for p in wall.polyline])
                
                # Check if wall intersects or is close to room boundary
                if wall_line.intersects(room_buffer) or wall_line.distance(room_poly) < 0.5:
                    # Check if wall is actually on the boundary (not just crossing)
                    wall_mid = Point(wall_line.interpolate(0.5, normalized=True))
                    if room_poly.buffer(0.1).contains(wall_mid) or room_poly.boundary.distance(wall_mid) < 0.3:
                        boundary_wall_ids.append(wall.id)
            
            logger.debug(f"Room {room.id} has {len(boundary_wall_ids)} boundary walls from M1")
        except Exception as e:
            logger.warning(f"Could not compute room boundaries for {room.id}: {e}")
        
        merged_rooms.append(Room(
            id=room.id,
            polygon=polygon_m,
            area=area,
            boundaryWallIds=boundary_wall_ids,
            confidence=room.confidence
        ))
    
    logger.info(f"Merge complete: {len(merged_walls)} walls, {len(merged_openings)} openings, {len(merged_rooms)} rooms")
    logger.info(f"  - Walls: {len(merged_walls)} from M1 topology (PRIMARY)")
    logger.info(f"  - Openings: {len([o for o in merged_openings if any(m3.id == o.id for m3 in m3_openings)])} from M3 (PRIMARY), {len([o for o in merged_openings if any(m1.id == o.id for m1 in m1_openings)])} from M1 (fallback)")
    logger.info(f"  - Rooms: {len(merged_rooms)} from M2 (PRIMARY)")
    
    return CanonicalPlan(
        metadata=Metadata(
            units="m",
            scale=px_to_meter,
            imageWidth=width,
            imageHeight=height
        ),
        walls=merged_walls,
        openings=merged_openings,
        rooms=merged_rooms
    )

