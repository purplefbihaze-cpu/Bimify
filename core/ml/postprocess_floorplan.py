from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING
import unicodedata

import cv2
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, MultiLineString
from shapely.ops import linemerge, unary_union

# Avoid importing heavy inference client at import time; we only need the name for typing
# RFPred is referenced in type annotations only (annotations are postponed by __future__ above)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from core.ml.roboflow_client import RFPred
else:  # pragma: no cover - fallback for runtime
    RFPred = object



@dataclass
class NormalizedDet:
    doc: int
    page: int
    type: str
    is_external: bool | None
    geom: Polygon | LineString
    attrs: Dict


_DOOR_KEYWORDS: Tuple[str, ...] = (
    "door",
    "doorway",
    "doorframe",
    "glassdoor",
    "tuer",
    "tur",
)

_WINDOW_KEYWORDS: Tuple[str, ...] = (
    "window",
    "windowframe",
    "fenster",
    "fenst",
)

_WINDOW_ABBREVIATIONS = {"win", "wnd"}

_WALL_KEYWORDS: Tuple[str, ...] = (
    "wall",
    "wand",
)


def _validate_and_repair_polygon(poly: Polygon) -> Polygon:
    """Validate and repair polygon geometry.
    
    Ensures polygon is valid, closed, and has finite coordinates.
    Uses buffer(0) trick for repair and closes unclosed polygons.
    
    Args:
        poly: Input polygon to validate and repair
    
    Returns:
        Validated and repaired polygon
    """
    if poly.is_empty:
        return poly
    
    # Repair invalid polygons
    if not poly.is_valid:
        try:
            repaired = poly.buffer(0)
            if not repaired.is_empty and repaired.is_valid:
                if isinstance(repaired, Polygon):
                    poly = repaired
                elif isinstance(repaired, MultiPolygon):
                    # Take largest valid polygon
                    valid_polys = [p for p in repaired.geoms if isinstance(p, Polygon) and p.is_valid and not p.is_empty]
                    if valid_polys:
                        poly = max(valid_polys, key=lambda p: p.area)
                    else:
                        return poly
        except Exception:
            pass
    
    # Check if closed
    if poly.is_valid:
        coords = list(poly.exterior.coords)
        if len(coords) >= 3:
            first = coords[0]
            last = coords[-1]
            dist = math.hypot(first[0] - last[0], first[1] - last[1])
            if dist > 1.0:  # Not closed
                try:
                    coords.append(coords[0])
                    poly = Polygon(coords)
                except Exception:
                    pass
    
    # Validate coordinates are finite
    if poly.is_valid:
        coords = list(poly.exterior.coords)
        for coord in coords:
            if not all(math.isfinite(c) for c in coord):
                # Try to repair by removing invalid coordinates
                try:
                    valid_coords = [c for c in coords if all(math.isfinite(x) for x in c)]
                    if len(valid_coords) >= 3:
                        # Ensure closed
                        if valid_coords[0] != valid_coords[-1]:
                            valid_coords.append(valid_coords[0])
                        poly = Polygon(valid_coords)
                except Exception:
                    pass
                break
    
    return poly


def _normalize_class_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_value.lower().replace(" ", "").replace("-", "").replace("_", "")


@dataclass
class WallMask:
    detection: NormalizedDet
    source_index: int
    polygon: Polygon
    mask: np.ndarray
    origin_mm: Tuple[float, float]
    px_per_mm: float
    bounds_mm: Tuple[float, float, float, float]


@dataclass
class WallAxis:
    detection: NormalizedDet
    source_index: int
    axis: LineString
    width_mm: float
    length_mm: float
    centroid_mm: Tuple[float, float]
    method: str
    metadata: Dict[str, float]


RASTER_PX_PER_MM = 2.0
MASK_MARGIN_MM = 50.0
MIN_COMPONENT_PIXELS = 8
TRIM_PERCENT = 0.15
MIN_AXIS_LENGTH_MM = 60.0
MIN_WALL_WIDTH_MM = 40.0
ORTHOGONAL_TOLERANCE_DEG = 10.0
ENDPOINT_SNAP_TOLERANCE_MM = 15.0
SHORT_FRAGMENT_THRESHOLD_MM = 150.0
NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def normalize_predictions(
    preds: List[RFPred],
    px_to_mm: float,
    per_class_thresholds: Dict[str, float] | None = None,
    global_threshold: float = 0.01,
    *,
    flip_y: bool = False,
    image_height_px: float | None = None,
) -> List[NormalizedDet]:
    from loguru import logger
    from core.exceptions import CoordinatePrecisionError
    
    per_cls = {k.lower(): v for k, v in (per_class_thresholds or {}).items()}
    out: List[NormalizedDet] = []
    flip_height = float(image_height_px) if flip_y and image_height_px is not None else None

    def _to_mm(x_px: float, y_px: float) -> Tuple[float, float]:
        y_value = y_px
        if flip_height is not None:
            y_value = flip_height - y_px
        return (x_px / px_to_mm, y_value / px_to_mm)
    
    def _to_px(x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Rückkonvertierung mm → px für Präzisions-Validierung."""
        y_px = y_mm * px_to_mm
        if flip_height is not None:
            y_px = flip_height - y_px
        return (x_mm * px_to_mm, y_px)

    for p in preds:
        t_raw = (p.klass or "").lower().strip()
        # threshold lookup using raw class names
        thr = per_cls.get(t_raw, global_threshold)
        if p.confidence < thr:
            continue
        normalized_label = _normalize_class_token(t_raw)
        t = t_raw
        is_external = None
        if any(keyword in normalized_label for keyword in _WALL_KEYWORDS):
            t = "WALL"
            if "extern" in normalized_label:
                is_external = True
            elif "intern" in normalized_label:
                is_external = False
        elif any(keyword in normalized_label for keyword in _DOOR_KEYWORDS):
            t = "DOOR"
        elif normalized_label in _WINDOW_ABBREVIATIONS or any(
            keyword in normalized_label for keyword in _WINDOW_KEYWORDS
        ):
            t = "WINDOW"
        elif "stair" in normalized_label:
            t = "STAIR"
        pts = p.polygon
        geometry_source = "polygon"
        if pts:
            polygon_mm = Polygon([_to_mm(x, y) for (x, y) in pts])
            # Präzisions-Validierung: Rückkonvertiere und prüfe Fehler
            max_error = 0.0
            mm_coords = list(polygon_mm.exterior.coords)
            for (px_x, px_y), mm_point in zip(pts, mm_coords):
                back_px = _to_px(mm_point[0], mm_point[1])
                error = math.hypot(px_x - back_px[0], px_y - back_px[1])
                max_error = max(max_error, error)
            
            if max_error > 0.5:
                raise CoordinatePrecisionError(
                    f"Präzisionsverlust {max_error:.2f}px > 0.5px. "
                    f"px_per_mm={px_to_mm} ist ungenau! "
                    f"Prediction: {p.klass} (confidence={p.confidence:.2f})"
                )
            elif max_error > 0.1:
                logger.warning(
                    f"Präzisionsverlust {max_error:.2f}px (tolerierbar aber hoch) "
                    f"für {p.klass} mit px_per_mm={px_to_mm}"
                )
            
            # Validate and repair polygon geometry
            poly = _validate_and_repair_polygon(polygon_mm)
            geom = poly
        elif p.bbox:
            x, y, w, h = p.bbox
            top_left = _to_mm(x, y)
            top_right = _to_mm(x + w, y)
            bottom_right = _to_mm(x + w, y + h)
            bottom_left = _to_mm(x, y + h)
            polygon_mm = Polygon(
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ]
            )
            # Präzisions-Validierung für BBox
            bbox_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            mm_points = [top_left, top_right, bottom_right, bottom_left]
            max_error = 0.0
            for (px_x, px_y), mm_point in zip(bbox_points, mm_points):
                back_px = _to_px(mm_point[0], mm_point[1])
                error = math.hypot(px_x - back_px[0], px_y - back_px[1])
                max_error = max(max_error, error)
            
            if max_error > 0.5:
                raise CoordinatePrecisionError(
                    f"Präzisionsverlust {max_error:.2f}px > 0.5px (BBox). "
                    f"px_per_mm={px_to_mm} ist ungenau! "
                    f"Prediction: {p.klass} (confidence={p.confidence:.2f})"
                )
            elif max_error > 0.1:
                logger.warning(
                    f"Präzisionsverlust {max_error:.2f}px (tolerierbar aber hoch, BBox) "
                    f"für {p.klass} mit px_per_mm={px_to_mm}"
                )
            
            # Validate and repair polygon geometry
            poly = _validate_and_repair_polygon(polygon_mm)
            geom = poly
            geometry_source = "bbox"
        else:
            continue
        out.append(
            NormalizedDet(
                doc=p.doc,
                page=p.page,
                type=t,
                is_external=is_external,
                geom=geom,
                attrs={
                    "confidence": p.confidence,
                    "geometry_source": geometry_source,
                    "flip_y": bool(flip_height is not None),
                },
            )
        )
    return out


def _iter_wall_polygons(geom: Polygon | MultiPolygon | LineString) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        if not geom.is_empty and geom.area > 1e-6:
            yield geom
        return
    if isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if part.is_empty or part.area <= 1e-6:
                continue
            yield part


def _clip_point(px: int, py: int, width: int, height: int) -> Tuple[int, int]:
    px = min(max(px, 0), width - 1)
    py = min(max(py, 0), height - 1)
    return px, py


def _coords_to_pixels(
    coords: Sequence[Tuple[float, float]],
    origin_x: float,
    origin_y: float,
    scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    for x, y in coords:
        px = int(round((x - origin_x) * scale))
        py = int(round((y - origin_y) * scale))
        px, py = _clip_point(px, py, width, height)
        pts.append((px, py))
    return np.asarray(pts, dtype=np.int32)


def _rasterize_wall_polygon(
    det: NormalizedDet,
    source_index: int,
    polygon: Polygon,
    *,
    raster_px_per_mm: float,
    margin_mm: float,
) -> WallMask | None:
    if polygon.is_empty or polygon.area <= 1e-6:
        return None

    minx, miny, maxx, maxy = polygon.bounds
    width_mm = max((maxx - minx) + 2.0 * margin_mm, 1.0)
    height_mm = max((maxy - miny) + 2.0 * margin_mm, 1.0)
    width_px = max(int(math.ceil(width_mm * raster_px_per_mm)) + 1, 4)
    height_px = max(int(math.ceil(height_mm * raster_px_per_mm)) + 1, 4)

    origin_x = minx - margin_mm
    origin_y = miny - margin_mm

    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    exterior = _coords_to_pixels(
        polygon.exterior.coords,
        origin_x,
        origin_y,
        raster_px_per_mm,
        width_px,
        height_px,
    )
    if len(exterior) >= 3:
        cv2.fillPoly(mask, [exterior], 1)

    for interior in polygon.interiors:
        pts = _coords_to_pixels(
            interior.coords,
            origin_x,
            origin_y,
            raster_px_per_mm,
            width_px,
            height_px,
        )
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 0)

    if not np.any(mask):
        return None

    return WallMask(
        detection=det,
        source_index=source_index,
        polygon=polygon,
        mask=mask,
        origin_mm=(origin_x, origin_y),
        px_per_mm=raster_px_per_mm,
        bounds_mm=(minx, miny, maxx, maxy),
    )


def build_wall_masks(
    norm: Sequence[NormalizedDet],
    *,
    raster_px_per_mm: float = RASTER_PX_PER_MM,
    margin_mm: float = MASK_MARGIN_MM,
) -> List[WallMask]:
    masks: List[WallMask] = []
    wall_index = -1
    for det in norm:
        if det.type != "WALL":
            continue
        wall_index += 1
        for poly in _iter_wall_polygons(det.geom):
            mask = _rasterize_wall_polygon(
                det,
                wall_index,
                poly,
                raster_px_per_mm=raster_px_per_mm,
                margin_mm=margin_mm,
            )
            if mask is not None:
                masks.append(mask)
    return masks


def _pixel_to_mm(px: int, py: int, mask: WallMask) -> Tuple[float, float]:
    x_mm = (px / mask.px_per_mm) + mask.origin_mm[0]
    y_mm = (py / mask.px_per_mm) + mask.origin_mm[1]
    return (float(x_mm), float(y_mm))


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    img = mask.copy().astype(np.uint8)
    img[img > 0] = 1
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _build_adjacency(nodes: Iterable[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    node_set = set(nodes)
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r, c in node_set:
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_set:
                neighbors.append((nr, nc))
        adjacency[(r, c)] = neighbors
    return adjacency


def _connected_components(nodes: Iterable[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    node_set = set(nodes)
    adjacency = _build_adjacency(node_set)
    visited: set[Tuple[int, int]] = set()
    components: List[List[Tuple[int, int]]] = []

    for node in node_set:
        if node in visited:
            continue
        stack = [node]
        current: List[Tuple[int, int]] = []
        while stack:
            item = stack.pop()
            if item in visited:
                continue
            visited.add(item)
            current.append(item)
            for neighbor in adjacency.get(item, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        if current:
            components.append(current)
    return components


def _edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _extract_segments(
    component: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    key_nodes = [node for node in component if len(adjacency.get(node, [])) != 2]
    visited_edges: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    segments: List[List[Tuple[int, int]]] = []

    def trace(start: Tuple[int, int], next_node: Tuple[int, int]) -> None:
        path = [start]
        prev = start
        current = next_node
        visited_edges.add(_edge_key(start, next_node))
        while True:
            path.append(current)
            neighbors = adjacency.get(current, [])
            if len(neighbors) != 2:
                break
            candidate = neighbors[0] if neighbors[0] != prev else neighbors[1]
            edge = _edge_key(current, candidate)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            prev, current = current, candidate
        if len(path) >= 2:
            segments.append(path)

    if key_nodes:
        for node in key_nodes:
            for neighbor in adjacency.get(node, []):
                edge = _edge_key(node, neighbor)
                if edge in visited_edges:
                    continue
                trace(node, neighbor)
    return segments


def _segment_to_axis(
    segment_nodes: Sequence[Tuple[int, int]],
    wall_mask: WallMask,
    distance_map: np.ndarray,
    *,
    component_index: int,
    component_size: int,
    segment_index: int,
) -> WallAxis | None:
    if len(segment_nodes) < 2:
        return None

    start_node = segment_nodes[0]
    end_node = segment_nodes[-1]
    start_mm = _pixel_to_mm(start_node[1], start_node[0], wall_mask)
    end_mm = _pixel_to_mm(end_node[1], end_node[0], wall_mask)
    dx = start_mm[0] - end_mm[0]
    dy = start_mm[1] - end_mm[1]
    if math.hypot(dx, dy) < 1e-3:
        return None

    axis = LineString([start_mm, end_mm])
    length_mm = float(axis.length)
    if length_mm < MIN_AXIS_LENGTH_MM:
        return None

    width_samples: List[float] = []
    for row, col in segment_nodes:
        dist_px = float(distance_map[row, col])
        if dist_px <= 0.0:
            continue
        width_samples.append(dist_px / wall_mask.px_per_mm)

    if not width_samples:
        return None

    if len(width_samples) >= 5:
        lower = np.percentile(width_samples, TRIM_PERCENT * 100.0)
        upper = np.percentile(width_samples, (1.0 - TRIM_PERCENT) * 100.0)
        trimmed = [v for v in width_samples if lower <= v <= upper]
        if trimmed:
            width_samples = trimmed

    # Consistent use of median for robust thickness estimation (robust against outliers)
    # Try median first, then trimmed mean, then mean as last resort
    half_thickness_mm = 0.0
    if width_samples:
        # Primary: Use median (most robust)
        half_thickness_mm = float(np.median(width_samples))
        
        # If median is invalid, try trimmed mean (remove outliers)
        if half_thickness_mm <= 0.0 and len(width_samples) >= 3:
            sorted_samples = sorted(width_samples)
            # Remove top and bottom 10% for trimmed mean
            trim_count = max(1, len(sorted_samples) // 10)
            trimmed = sorted_samples[trim_count:-trim_count] if len(sorted_samples) > trim_count * 2 else sorted_samples
            if trimmed:
                half_thickness_mm = float(np.mean(trimmed))
        
        # Last resort: use mean if median and trimmed mean both fail
        if half_thickness_mm <= 0.0:
            half_thickness_mm = float(np.mean(width_samples))
    
    width_mm = max(half_thickness_mm * 2.0, MIN_WALL_WIDTH_MM)

    centroid_point = axis.interpolate(0.5, normalized=True)
    centroid_mm = (float(centroid_point.x), float(centroid_point.y))

    metadata = {
        "component_index": float(component_index),
        "component_pixels": float(component_size),
        "segment_index": float(segment_index),
        "segment_pixels": float(len(segment_nodes)),
        "width_samples": float(len(width_samples)),
        "width_std": float(np.std(width_samples)) if len(width_samples) >= 2 else 0.0,
        "source_index": float(wall_mask.source_index),
    }

    return WallAxis(
        detection=wall_mask.detection,
        source_index=wall_mask.source_index,
        axis=axis,
        width_mm=width_mm,
        length_mm=length_mm,
        centroid_mm=centroid_mm,
        method="skeleton",
        metadata=metadata,
    )


def _apply_orth_snap(axes: List[WallAxis], tolerance_deg: float) -> None:
    """
    Apply orthogonal snapping to wall axes.
    Enhanced with validation and logging for non-orthogonal walls.
    
    Args:
        axes: List of wall axes to snap
        tolerance_deg: Angle tolerance in degrees (default ±6°)
    """
    if not axes:
        return
    
    import logging
    logger = logging.getLogger(__name__)
    
    non_orthogonal_count = 0
    snapped_count = 0
    failed_snap_count = 0
    
    for axis_info in axes:
        axis = axis_info.axis
        if axis.length <= 0.0:
            continue
        x1, y1 = axis.coords[0]
        x2, y2 = axis.coords[-1]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_norm = abs(((angle_deg + 180.0) % 180.0))
        if angle_norm > 90.0:
            angle_norm = 180.0 - angle_norm

        target_dir: Tuple[float, float] | None = None
        if angle_norm <= tolerance_deg:
            target_dir = (1.0 if dx >= 0 else -1.0, 0.0)
        elif abs(angle_norm - 90.0) <= tolerance_deg:
            target_dir = (0.0, 1.0 if dy >= 0 else -1.0)

        if target_dir is None:
            # Wall is not orthogonal (outside tolerance)
            non_orthogonal_count += 1
            deviation_from_horizontal = min(angle_norm, abs(angle_norm - 90.0))
            deviation_from_vertical = min(abs(angle_norm - 90.0), abs(angle_norm - 0.0))
            min_deviation = min(deviation_from_horizontal, deviation_from_vertical)
            
            logger.debug(
                "Non-orthogonal wall detected: angle=%.1f° (normalized=%.1f°), deviation from orthogonal=%.1f° "
                "(tolerance=±%.1f°) - wall will not be snapped",
                angle_deg, angle_norm, min_deviation, tolerance_deg
            )
            continue

        # Apply snapping
        length_mm = axis.length
        midpoint = axis.interpolate(0.5, normalized=True)
        half = length_mm / 2.0
        new_start = (float(midpoint.x - target_dir[0] * half), float(midpoint.y - target_dir[1] * half))
        new_end = (float(midpoint.x + target_dir[0] * half), float(midpoint.y + target_dir[1] * half))
        
        try:
            new_axis = LineString([new_start, new_end])
            
            # Validate snapped axis
            if new_axis.length < 1e-3:
                failed_snap_count += 1
                logger.warning(
                    "Orthogonal snapping failed: resulting axis length too short (%.3fmm) - keeping original axis",
                    new_axis.length
                )
                continue
            
            # Validate coordinates are finite
            coords = list(new_axis.coords)
            if not all(all(math.isfinite(c) for c in coord) for coord in coords):
                failed_snap_count += 1
                logger.warning(
                    "Orthogonal snapping failed: non-finite coordinates detected - keeping original axis"
                )
                continue
            
            # Update axis
            axis_info.axis = new_axis
            axis_info.length_mm = float(new_axis.length)
            axis_info.centroid_mm = (float(midpoint.x), float(midpoint.y))
            axis_info.metadata["orth_snapped"] = axis_info.metadata.get("orth_snapped", 0.0) + 1.0
            axis_info.metadata["angle_before"] = float(angle_deg)
            axis_info.metadata["angle_after"] = float(math.degrees(math.atan2(target_dir[1], target_dir[0])))
            
            snapped_count += 1
            logger.debug(
                "Orthogonal snapping successful: angle changed from %.1f° to %.1f° (axis length: %.1fmm)",
                angle_deg, axis_info.metadata["angle_after"], new_axis.length
            )
        except Exception as snap_exc:
            failed_snap_count += 1
            logger.warning(
                "Orthogonal snapping failed with exception: %s - keeping original axis",
                snap_exc
            )
            continue
    
    # Summary logging
    total_axes = len([ax for ax in axes if ax.axis and ax.axis.length > 1e-3])
    if non_orthogonal_count > 0:
        logger.info(
            "Orthogonal snapping: %d/%d wall(s) are non-orthogonal (outside ±%.1f° tolerance) - not snapped",
            non_orthogonal_count, total_axes, tolerance_deg
        )
    if snapped_count > 0:
        logger.debug(
            "Orthogonal snapping: Successfully snapped %d/%d wall(s) to orthogonal directions",
            snapped_count, total_axes
        )
    if failed_snap_count > 0:
        logger.warning(
            "Orthogonal snapping: Failed to snap %d wall(s) - original axes preserved",
            failed_snap_count
        )


def _snap_axis_endpoints(
    axes: List[WallAxis],
    tolerance_mm: float,
) -> tuple[List[dict], Dict[Tuple[int, int], int]]:
    clusters: List[dict] = []
    endpoint_cluster: Dict[Tuple[int, int], int] = {}

    for idx, axis_info in enumerate(axes):
        coords = list(axis_info.axis.coords)
        for endpoint_idx, point in enumerate((coords[0], coords[-1])):
            assigned = None
            for cluster_idx, cluster in enumerate(clusters):
                cx, cy = cluster["center"]
                if math.hypot(point[0] - cx, point[1] - cy) <= tolerance_mm:
                    assigned = cluster_idx
                    break
            if assigned is None:
                clusters.append({"center": np.array(point, dtype=float), "points": []})
                assigned = len(clusters) - 1
            cluster = clusters[assigned]
            cluster["points"].append((idx, endpoint_idx, point))
            # incremental average
            count = len(cluster["points"])
            center = cluster["center"]
            center += (np.array(point, dtype=float) - center) / count
            endpoint_cluster[(idx, endpoint_idx)] = assigned

    for cluster_idx, cluster in enumerate(clusters):
        center = tuple(cluster["center"].tolist())
        for axis_index, endpoint_idx, _original in cluster["points"]:
            axis_info = axes[axis_index]
            coords = list(axis_info.axis.coords)
            if endpoint_idx == 0:
                coords[0] = center
            else:
                coords[-1] = center
            new_axis = LineString(coords)
            axis_info.axis = new_axis
            axis_info.length_mm = float(new_axis.length)
            midpoint = new_axis.interpolate(0.5, normalized=True)
            axis_info.centroid_mm = (float(midpoint.x), float(midpoint.y))
            axis_info.metadata["endpoint_snapped"] = axis_info.metadata.get("endpoint_snapped", 0.0) + 1.0
            axis_info.metadata["cluster_%d" % endpoint_idx] = float(cluster_idx)

    return clusters, endpoint_cluster


def _filter_short_axes(
    axes: List[WallAxis],
    endpoint_cluster: Dict[Tuple[int, int], int],
    clusters: List[dict],
    *,
    min_length_mm: float,
) -> List[WallAxis]:
    if not axes:
        return []

    cluster_sizes = [float(len(cluster["points"])) for cluster in clusters]
    kept: List[WallAxis] = []
    for idx, axis_info in enumerate(axes):
        if axis_info.length_mm >= min_length_mm:
            kept.append(axis_info)
            continue
        start_cluster_idx = endpoint_cluster.get((idx, 0))
        end_cluster_idx = endpoint_cluster.get((idx, 1))
        start_neighbors = cluster_sizes[start_cluster_idx] if start_cluster_idx is not None else 0.0
        end_neighbors = cluster_sizes[end_cluster_idx] if end_cluster_idx is not None else 0.0
        if start_neighbors > 1.0 and end_neighbors > 1.0:
            continue
        kept.append(axis_info)
    return kept


def _merge_colinear_wall_axes(axes: List[WallAxis]) -> List[WallAxis]:
    grouped: Dict[int, List[WallAxis]] = defaultdict(list)
    passthrough: List[WallAxis] = []

    for axis in axes:
        src_idx = getattr(axis, "source_index", None)
        geom = getattr(axis, "axis", None)
        if src_idx is None or not isinstance(geom, LineString) or geom.length <= 1e-6:
            passthrough.append(axis)
            continue
        grouped[int(src_idx)].append(axis)

    merged_axes: List[WallAxis] = []

    def _weighted_width(line: LineString, candidates: Sequence[WallAxis]) -> float:
        contributions: List[Tuple[float, float]] = []
        for candidate in candidates:
            width_value = float(candidate.width_mm or MIN_WALL_WIDTH_MM)
            geom = getattr(candidate, "axis", None)
            if not isinstance(geom, LineString):
                continue
            try:
                overlap = geom.intersection(line)
                weight = float(overlap.length)
            except Exception:
                weight = 0.0
            if weight > 0.0:
                contributions.append((width_value, weight))
        if contributions:
            total = sum(weight for _, weight in contributions)
            if total > 0.0:
                return sum(width * weight for width, weight in contributions) / total
        widths = [float(candidate.width_mm) for candidate in candidates if candidate.width_mm]
        if widths:
            return sum(widths) / len(widths)
        return float(MIN_WALL_WIDTH_MM)

    merged_axes.extend(passthrough)

    for source_index, axis_list in grouped.items():
        if len(axis_list) == 1:
            merged_axes.extend(axis_list)
            continue

        try:
            merged_geom = linemerge(unary_union([axis.axis for axis in axis_list if isinstance(axis.axis, LineString)]))
        except Exception:
            merged_axes.extend(axis_list)
            continue

        if isinstance(merged_geom, LineString):
            merged_lines = [merged_geom]
        elif isinstance(merged_geom, MultiLineString):
            merged_lines = [line for line in merged_geom.geoms if line.length > 1e-6]
        else:
            merged_axes.extend(axis_list)
            continue

        base_detection = axis_list[0].detection
        total_segments = float(sum(axis.axis.length for axis in axis_list if isinstance(axis.axis, LineString)))

        for idx, line in enumerate(merged_lines):
            length_mm = float(line.length)
            if length_mm < MIN_AXIS_LENGTH_MM:
                continue
            centroid = line.interpolate(0.5, normalized=True)
            width_mm = _weighted_width(line, axis_list)
            metadata = dict(axis_list[0].metadata or {})
            metadata["merged_count"] = float(len(axis_list))
            metadata["merged_total_length_mm"] = total_segments
            metadata["merged_segment_index"] = float(idx)
            metadata["merged_segment_count"] = float(len(merged_lines))

            merged_axes.append(
                WallAxis(
                    detection=base_detection,
                    source_index=source_index,
                    axis=line,
                    width_mm=width_mm,
                    length_mm=length_mm,
                    centroid_mm=(float(centroid.x), float(centroid.y)),
                    method="merged",
                    metadata=metadata,
                )
            )

    return merged_axes


def _downsample_mask_if_needed(
    mask: np.ndarray,
    max_dimension: int = 1000,
    target_dpi: float = 100.0,
) -> tuple[np.ndarray, float]:
    """
    Downsample mask if it exceeds max_dimension.
    
    Args:
        mask: Binary mask array
        max_dimension: Maximum dimension before downsampling
        target_dpi: Target DPI for downsampled mask
        
    Returns:
        Tuple of (downsampled_mask, scale_factor)
    """
    height, width = mask.shape
    max_dim = max(height, width)
    
    if max_dim <= max_dimension:
        return mask, 1.0
    
    # Calculate scale factor to bring max dimension to target
    scale_factor = max_dimension / max_dim
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Downsample using area interpolation (better for binary masks)
    downsampled = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return downsampled, scale_factor


def _upscale_coords(
    coords: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Upscale coordinates back to original scale."""
    if scale_factor == 1.0:
        return coords
    return (coords / scale_factor).astype(np.int32)


def _polygon_hash(polygon: Polygon) -> str:
    """Create hash for polygon for caching."""
    coords = list(polygon.exterior.coords)
    coords_str = ",".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return hashlib.md5(coords_str.encode()).hexdigest()


def skeletonize_and_estimate_width(
    wall_mask: WallMask,
    *,
    max_dimension: int = 1000,
    target_dpi: float = 100.0,
    enable_cache: bool = True,
) -> List[WallAxis]:
    binary = wall_mask.mask.astype(np.uint8)
    binary[binary > 0] = 1
    if not np.any(binary):
        return []

    # Downsample if needed
    scale_factor = 1.0
    if max_dimension > 0:
        binary, scale_factor = _downsample_mask_if_needed(binary, max_dimension, target_dpi)

    skeleton = _morphological_skeleton(binary)
    coords = np.argwhere(skeleton > 0)
    if coords.shape[0] < MIN_COMPONENT_PIXELS:
        return []

    # Upscale coordinates if we downsampled
    if scale_factor != 1.0:
        coords = _upscale_coords(coords, scale_factor)
        # Recalculate distance map at original scale if needed
        # For now, we'll work with downsampled distance map and scale results
        original_binary = wall_mask.mask.astype(np.uint8)
        original_binary[original_binary > 0] = 1
        distance_map = cv2.distanceTransform(original_binary, cv2.DIST_L2, 5)
    else:
        distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    components = _connected_components([(int(r), int(c)) for (r, c) in coords])
    axes: List[WallAxis] = []

    for component_index, component in enumerate(components):
        component_size = len(component)
        if component_size < MIN_COMPONENT_PIXELS:
            continue
        adjacency = _build_adjacency(component)
        segments = _extract_segments(component, adjacency)
        if not segments:
            axis_info = _segment_to_axis(
                component,
                wall_mask,
                distance_map,
                component_index=component_index,
                component_size=component_size,
                segment_index=0,
            )
            if axis_info is not None:
                axes.append(axis_info)
            continue

        for segment_index, segment_nodes in enumerate(segments):
            axis_info = _segment_to_axis(
                segment_nodes,
                wall_mask,
                distance_map,
                component_index=component_index,
                component_size=component_size,
                segment_index=segment_index,
            )
            if axis_info is not None:
                axes.append(axis_info)

    if not axes:
        return []

    _apply_orth_snap(axes, ORTHOGONAL_TOLERANCE_DEG)
    clusters, endpoint_cluster = _snap_axis_endpoints(axes, ENDPOINT_SNAP_TOLERANCE_MM)
    axes = _filter_short_axes(axes, endpoint_cluster, clusters, min_length_mm=SHORT_FRAGMENT_THRESHOLD_MM)

    return axes


def _is_valid_axis(axis: LineString, width_mm: float) -> bool:
    """
    Validate that an axis is acceptable for wall reconstruction.
    
    Args:
        axis: LineString axis to validate
        width_mm: Wall width in millimeters
        
    Returns:
        True if axis is valid, False otherwise
    """
    if axis is None or axis.is_empty:
        return False
    
    # Check axis length
    length_mm = float(axis.length)
    if length_mm < MIN_AXIS_LENGTH_MM:
        return False
    
    # Check width is within reasonable bounds
    if width_mm < MIN_WALL_WIDTH_MM or width_mm > 1000.0:  # Max 1m wall thickness
        return False
    
    # Check axis is reasonably straight (not fragmented)
    coords = list(axis.coords)
    if len(coords) < 2:
        return False
    
    # For multi-point lines, check that deviation from straight line is reasonable
    if len(coords) > 2:
        start = coords[0]
        end = coords[-1]
        # Calculate maximum deviation from straight line
        max_deviation = 0.0
        for coord in coords[1:-1]:
            # Distance from point to line segment
            line_length = math.hypot(end[0] - start[0], end[1] - start[1])
            if line_length < 1e-6:
                continue
            # Vector from start to point
            vx = coord[0] - start[0]
            vy = coord[1] - start[1]
            # Vector from start to end
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            # Project point onto line
            t = max(0.0, min(1.0, (vx * dx + vy * dy) / (line_length * line_length)))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            deviation = math.hypot(coord[0] - proj_x, coord[1] - proj_y)
            max_deviation = max(max_deviation, deviation)
        
        # Allow deviation up to 10% of length or 50mm, whichever is smaller
        max_allowed_deviation = min(length_mm * 0.1, 50.0)
        if max_deviation > max_allowed_deviation:
            return False
    
    return True


def _estimate_by_bounding_box(polygon: Polygon, det: NormalizedDet, source_index: int) -> WallAxis | None:
    """
    Estimate wall axis using minimum rotated rectangle (bounding box method).
    Faster and more robust than skeletonization for thin walls.
    
    Args:
        polygon: Wall polygon
        det: Normalized detection
        source_index: Source index
        
    Returns:
        WallAxis or None if estimation failed
    """
    if polygon.is_empty or polygon.area <= 1e-6:
        return None
    
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return None
    
    # Find longest edge (axis) and shortest edge (width)
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    for i in range(4):
        a = coords[i]
        b = coords[(i + 1) % 4]
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        length = float(math.hypot(dx, dy))
        edges.append((a, b, length))
    
    longest = max(edges, key=lambda item: item[2])
    shortest = min(edges, key=lambda item: item[2])
    length_mm = longest[2]
    width_mm = max(shortest[2], MIN_WALL_WIDTH_MM)
    
    # Create axis from longest edge
    axis = LineString([longest[0], longest[1]])
    
    # Validate axis
    if not _is_valid_axis(axis, width_mm):
        return None
    
    centroid = (float(rect.centroid.x), float(rect.centroid.y))
    
    return WallAxis(
        detection=det,
        source_index=source_index,
        axis=axis,
        width_mm=width_mm,
        length_mm=length_mm,
        centroid_mm=centroid,
        method="bounding_box",
        metadata={"source_index": float(source_index)},
    )


def _estimate_by_centerline_heuristic(polygon: Polygon, det: NormalizedDet, source_index: int) -> WallAxis | None:
    """
    Estimate wall axis using centerline heuristic for orthogonal plans.
    Uses polygon centerline (interior parallel offset) for better accuracy.
    
    Args:
        polygon: Wall polygon
        det: Normalized detection
        source_index: Source index
        
    Returns:
        WallAxis or None if estimation failed
    """
    if polygon.is_empty or polygon.area <= 1e-6:
        return None
    
    try:
        # Calculate approximate thickness from polygon
        # Use minimum distance from centroid to boundary as half-thickness estimate
        centroid = polygon.centroid
        boundary = polygon.boundary
        min_dist = boundary.distance(centroid)
        estimated_half_thickness = float(min_dist)
        
        # Create centerline by offsetting boundary inward
        # Use a small buffer inward to get centerline
        if estimated_half_thickness > 1.0:
            centerline_poly = polygon.buffer(-estimated_half_thickness * 0.5)
            if centerline_poly.is_empty:
                # Fallback: use centroid as single point
                centerline = LineString([centroid, centroid])
            elif isinstance(centerline_poly, Polygon):
                # Use longest edge of interior polygon as axis
                coords = list(centerline_poly.exterior.coords)
                if len(coords) >= 2:
                    # Find longest edge
                    max_length = 0.0
                    best_start = coords[0]
                    best_end = coords[1]
                    for i in range(len(coords) - 1):
                        a = coords[i]
                        b = coords[i + 1]
                        length = math.hypot(b[0] - a[0], b[1] - a[1])
                        if length > max_length:
                            max_length = length
                            best_start = a
                            best_end = b
                    centerline = LineString([best_start, best_end])
                else:
                    centerline = LineString([centroid, centroid])
            else:
                # MultiPolygon - use largest component
                if isinstance(centerline_poly, MultiPolygon):
                    largest = max(centerline_poly.geoms, key=lambda g: g.area if hasattr(g, 'area') else 0.0)
                    if isinstance(largest, Polygon):
                        coords = list(largest.exterior.coords)
                        if len(coords) >= 2:
                            centerline = LineString([coords[0], coords[1]])
                        else:
                            centerline = LineString([centroid, centroid])
                    else:
                        centerline = LineString([centroid, centroid])
                else:
                    centerline = LineString([centroid, centroid])
        else:
            # Very thin wall, use centroid as axis
            centerline = LineString([centroid, centroid])
        
        length_mm = float(centerline.length)
        width_mm = max(estimated_half_thickness * 2.0, MIN_WALL_WIDTH_MM)
        
        # Validate axis
        if not _is_valid_axis(centerline, width_mm):
            return None
        
        centroid_mm = (float(centroid.x), float(centroid.y))
        
        return WallAxis(
            detection=det,
            source_index=source_index,
            axis=centerline,
            width_mm=width_mm,
            length_mm=length_mm,
            centroid_mm=centroid_mm,
            method="centerline_heuristic",
            metadata={"source_index": float(source_index)},
        )
    except Exception:
        return None


def _fallback_axis(det: NormalizedDet, source_index: int, polygon: Polygon) -> WallAxis | None:
    if polygon.is_empty or polygon.area <= 1e-6:
        return None
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return None

    edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    for i in range(4):
        a = coords[i]
        b = coords[(i + 1) % 4]
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        length = float(math.hypot(dx, dy))
        edges.append((a, b, length))

    longest = max(edges, key=lambda item: item[2])
    shortest = min(edges, key=lambda item: item[2])
    length_mm = longest[2]
    width_mm = max(shortest[2], MIN_WALL_WIDTH_MM)

    if length_mm < MIN_AXIS_LENGTH_MM:
        return None

    dx = longest[1][0] - longest[0][0]
    dy = longest[1][1] - longest[0][1]
    mag = math.hypot(dx, dy) or 1.0
    ux = dx / mag
    uy = dy / mag
    cx, cy = rect.centroid.x, rect.centroid.y
    half = length_mm / 2.0
    start = (cx - ux * half, cy - uy * half)
    end = (cx + ux * half, cy + uy * half)
    axis = LineString([start, end])
    centroid = (cx, cy)

    return WallAxis(
        detection=det,
        source_index=source_index,
        axis=axis,
        width_mm=width_mm,
        length_mm=length_mm,
        centroid_mm=centroid,
        method="fallback",
        metadata={"strategy": 1.0, "source_index": float(source_index)},
    )


def _legacy_axes(norm: Sequence[NormalizedDet]) -> List[WallAxis]:
    axes: List[WallAxis] = []
    wall_index = -1
    for det in norm:
        if det.type != "WALL" or not isinstance(det.geom, (Polygon, MultiPolygon)):
            continue
        wall_index += 1
        polygons = list(_iter_wall_polygons(det.geom))
        if not polygons:
            continue
        target_poly = max(polygons, key=lambda p: p.area)
        fallback = _fallback_axis(det, wall_index, target_poly)
        if fallback is not None:
            axes.append(fallback)
    return axes


def estimate_wall_axes_and_thickness(
    norm: Sequence[NormalizedDet],
    *,
    raster_px_per_mm: float = RASTER_PX_PER_MM,
    margin_mm: float = MASK_MARGIN_MM,
    max_dimension: int = 1000,
    target_dpi: float = 100.0,
    enable_cache: bool = True,
) -> List[WallAxis]:
    """
    Estimate wall axes and thickness using method hierarchy:
    1. Skeletonization (for thick walls, most accurate)
    2. Bounding box (faster, more robust for thin walls)
    3. Centerline heuristic (for orthogonal plans)
    
    Each method validates its result before returning.
    """
    from loguru import logger
    
    wall_masks = build_wall_masks(norm, raster_px_per_mm=raster_px_per_mm, margin_mm=margin_mm)
    axes: List[WallAxis] = []
    
    for mask in wall_masks:
        axis_result = None
        method_used = None
        
        # Method 1: Try skeletonization (for thick walls)
        try:
            skeleton_axes = skeletonize_and_estimate_width(
                mask,
                max_dimension=max_dimension,
                target_dpi=target_dpi,
                enable_cache=enable_cache,
            )
            if skeleton_axes:
                # Validate ALL skeleton axes - only use if at least one is valid
                valid_skeleton_axes = [ax for ax in skeleton_axes if _is_valid_axis(ax.axis, ax.width_mm)]
                if valid_skeleton_axes:
                    axis_result = valid_skeleton_axes
                    method_used = "skeletonization"
                    if len(valid_skeleton_axes) < len(skeleton_axes):
                        logger.warning(
                            f"Skeletonization produced {len(skeleton_axes)} axes for wall {mask.source_index}, "
                            f"but only {len(valid_skeleton_axes)} are valid. Using valid axes only."
                        )
                else:
                    logger.warning(
                        f"Skeletonization produced invalid axes for wall {mask.source_index} "
                        f"(all {len(skeleton_axes)} axes failed validation), using fallback"
                    )
        except Exception as e:
            logger.debug(f"Skeletonization failed for wall {mask.source_index}: {e}")
        
        # Method 2: Try bounding box (faster, more robust)
        if axis_result is None:
            try:
                bbox_axis = _estimate_by_bounding_box(mask.polygon, mask.detection, mask.source_index)
                if bbox_axis is not None and _is_valid_axis(bbox_axis.axis, bbox_axis.width_mm):
                    axis_result = [bbox_axis]
                    method_used = "bounding_box"
            except Exception as e:
                logger.debug(f"Bounding box method failed for wall {mask.source_index}: {e}")
        
        # Method 3: Try centerline heuristic (for orthogonal plans)
        if axis_result is None:
            try:
                centerline_axis = _estimate_by_centerline_heuristic(mask.polygon, mask.detection, mask.source_index)
                if centerline_axis is not None and _is_valid_axis(centerline_axis.axis, centerline_axis.width_mm):
                    axis_result = [centerline_axis]
                    method_used = "centerline_heuristic"
            except Exception as e:
                logger.debug(f"Centerline heuristic failed for wall {mask.source_index}: {e}")
        
        # Fallback to legacy method if all fail
        if axis_result is None:
            fallback = _fallback_axis(mask.detection, mask.source_index, mask.polygon)
            if fallback is not None and _is_valid_axis(fallback.axis, fallback.width_mm):
                axis_result = [fallback]
                method_used = "fallback"
        
        if axis_result:
            axes.extend(axis_result)
            if method_used != "skeletonization":
                logger.debug(f"Used {method_used} method for wall {mask.source_index} (skeletonization unavailable)")

    # Final fallback: legacy axes if nothing worked
    if not axes:
        legacy_axes = _legacy_axes(norm)
        for ax in legacy_axes:
            if _is_valid_axis(ax.axis, ax.width_mm):
                axes.append(ax)

    if axes:
        axes = _merge_colinear_wall_axes(axes)

    return axes


