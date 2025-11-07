from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
    per_cls = {k.lower(): v for k, v in (per_class_thresholds or {}).items()}
    out: List[NormalizedDet] = []
    flip_height = float(image_height_px) if flip_y and image_height_px is not None else None

    def _to_mm(x_px: float, y_px: float) -> Tuple[float, float]:
        y_value = y_px
        if flip_height is not None:
            y_value = flip_height - y_px
        return (x_px / px_to_mm, y_value / px_to_mm)

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
            poly = Polygon([_to_mm(x, y) for (x, y) in pts])
            geom = poly
        elif p.bbox:
            x, y, w, h = p.bbox
            top_left = _to_mm(x, y)
            top_right = _to_mm(x + w, y)
            bottom_right = _to_mm(x + w, y + h)
            bottom_left = _to_mm(x, y + h)
            geom = Polygon(
                [
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ]
            )
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
    if not axes:
        return
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
            continue

        length_mm = axis.length
        midpoint = axis.interpolate(0.5, normalized=True)
        half = length_mm / 2.0
        new_start = (float(midpoint.x - target_dir[0] * half), float(midpoint.y - target_dir[1] * half))
        new_end = (float(midpoint.x + target_dir[0] * half), float(midpoint.y + target_dir[1] * half))
        new_axis = LineString([new_start, new_end])
        axis_info.axis = new_axis
        axis_info.length_mm = float(new_axis.length)
        axis_info.centroid_mm = (float(midpoint.x), float(midpoint.y))
        axis_info.metadata["orth_snapped"] = axis_info.metadata.get("orth_snapped", 0.0) + 1.0
        axis_info.metadata["angle_before"] = float(angle_deg)


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


def skeletonize_and_estimate_width(wall_mask: WallMask) -> List[WallAxis]:
    binary = wall_mask.mask.astype(np.uint8)
    binary[binary > 0] = 1
    if not np.any(binary):
        return []

    skeleton = _morphological_skeleton(binary)
    coords = np.argwhere(skeleton > 0)
    if coords.shape[0] < MIN_COMPONENT_PIXELS:
        return []

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
) -> List[WallAxis]:
    wall_masks = build_wall_masks(norm, raster_px_per_mm=raster_px_per_mm, margin_mm=margin_mm)
    axes: List[WallAxis] = []
    for mask in wall_masks:
        skeleton_axes = skeletonize_and_estimate_width(mask)
        if skeleton_axes:
            axes.extend(skeleton_axes)
        else:
            fallback = _fallback_axis(mask.detection, mask.source_index, mask.polygon)
            if fallback is not None:
                axes.append(fallback)

    if not axes:
        axes.extend(_legacy_axes(norm))

    if axes:
        axes = _merge_colinear_wall_axes(axes)

    return axes


