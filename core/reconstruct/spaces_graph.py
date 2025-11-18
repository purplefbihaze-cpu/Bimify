from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from shapely.geometry import LinearRing, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet
from core.reconstruct.spaces import SpacePoly, SpaceConfig
from core.geometry.contract import (
    MIN_POLYGON_AREA,
    MIN_SPACE_DIMENSION_MM,
    NODE_MERGE_DIST,
    mm,
)


logger = logging.getLogger(__name__)


@dataclass
class _Node:
    key: Tuple[int, int]
    point: Tuple[float, float]
    out_edges: List["_HalfEdge"] = field(default_factory=list)


@dataclass
class _HalfEdge:
    origin: _Node
    dest: _Node
    angle: float
    twin: Optional["_HalfEdge"] = None
    visited: bool = False
    sort_index: int = -1


_SIMPLIFY_MM = 5.0  # micro-simplification threshold (mm)
_EPS_AREA = 1e-2


def find_space_cycles_graph(
    dets: List[NormalizedDet],
    config: SpaceConfig | None = None,
) -> List[SpacePoly]:
    if not dets:
        return []
    if config is None:
        config = SpaceConfig()

    wall_polys: List[Polygon] = []
    for det in dets:
        if getattr(det, "type", "").upper() != "WALL":
            continue
        geom = getattr(det, "geom", None)
        if isinstance(geom, Polygon) and not geom.is_empty:
            wall_polys.append(geom)
    if not wall_polys:
        return []

    try:
        wall_union = unary_union(wall_polys)
    except Exception as exc:
        logger.warning("Failed to union walls for space cycles: %s", exc)
        return []

    if wall_union.is_empty:
        return []

    hull = wall_union.envelope.buffer(0.0)
    hull_area = float(hull.area)

    tol_mm = max(mm(NODE_MERGE_DIST), 1.0)
    segments = _collect_segments(wall_polys, tol_mm)
    if not segments:
        return []

    nodes, half_edges = _build_planar_graph(segments, tol_mm)
    if not half_edges:
        return []

    cycles = _traverse_faces(half_edges)
    if not cycles:
        return []

    candidate_polys = _cycles_to_polygons(
        cycles,
        wall_union,
        hull,
        hull_area,
        tol_mm,
        config,
    )

    if not candidate_polys:
        return []

    final_polys = _assign_holes(candidate_polys, tol_mm)

    if not final_polys:
        return []

    spaces: List[SpacePoly] = []
    seen: set[bytes] = set()

    for poly in final_polys:
        if not poly or poly.is_empty:
            continue
        try:
            poly = orient(poly, sign=-1.0)
            poly = poly.simplify(_SIMPLIFY_MM, preserve_topology=True)
        except Exception:
            pass

        if not _is_valid_space(poly, config.min_room_area_m2, config.min_space_dimension_mm):
            continue

        try:
            key = poly.wkb
        except Exception:
            key = None
        if key and key in seen:
            continue
        if key:
            seen.add(key)

        area_m2 = float(poly.area) / 1_000_000.0
        spaces.append(SpacePoly(polygon=poly, area_m2=area_m2))

    spaces.sort(key=_space_sort_key)
    return spaces


def _space_sort_key(space: SpacePoly) -> Tuple[float, float, float]:
    try:
        centroid = space.polygon.centroid
        return (-space.area_m2, float(centroid.x), float(centroid.y))
    except Exception:
        return (-space.area_m2, 0.0, 0.0)


def _is_valid_space(poly: Polygon, min_area_m2: float, min_dim_mm: float) -> bool:
    if poly is None or poly.is_empty or not isinstance(poly, Polygon):
        return False

    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return False
        if not poly.is_valid or poly.is_empty:
            return False

    area_m2 = float(poly.area) / 1_000_000.0
    if area_m2 < max(min_area_m2, MIN_POLYGON_AREA):
        return False

    try:
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        edges = []
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if edges and min(edges) < float(min_dim_mm):
            return False
    except Exception:
        pass

    return True


def _collect_segments(polys: List[Polygon], tol_mm: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    def _add_ring(ring: LinearRing) -> None:
        coords = list(ring.coords)
        if len(coords) < 4:
            return
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            if _segment_length_sq(p1, p2) <= (tol_mm * tol_mm) * 1e-6:
                continue
            segments.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))

    for poly in polys:
        if poly.is_empty:
            continue
        try:
            _add_ring(LinearRing(poly.exterior.coords))
            for interior in poly.interiors:
                _add_ring(LinearRing(interior.coords))
        except Exception:
            continue

    return segments


def _segment_length_sq(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _build_planar_graph(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tol_mm: float,
) -> Tuple[Dict[Tuple[int, int], _Node], List[_HalfEdge]]:
    node_map: Dict[Tuple[int, int], _Node] = {}
    half_edges: List[_HalfEdge] = []
    segment_keys: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    def _quantize(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(pt[0] / tol_mm)), int(round(pt[1] / tol_mm)))

    def _node_for(pt: Tuple[float, float]) -> _Node:
        key = _quantize(pt)
        node = node_map.get(key)
        if node is None:
            coord = (key[0] * tol_mm, key[1] * tol_mm)
            node = _Node(key=key, point=coord)
            node_map[key] = node
        return node

    for p1, p2 in segments:
        key_u = _quantize(p1)
        key_v = _quantize(p2)
        if key_u == key_v:
            continue
        seg_key = tuple(sorted((key_u, key_v), key=lambda x: (x[0], x[1])))
        if seg_key in segment_keys:
            continue
        segment_keys.add(seg_key)

        node_u = _node_for(p1)
        node_v = _node_for(p2)

        angle_uv = _angle(node_u.point, node_v.point)
        angle_vu = _normalize_angle(angle_uv + math.pi)

        forward = _HalfEdge(origin=node_u, dest=node_v, angle=angle_uv)
        backward = _HalfEdge(origin=node_v, dest=node_u, angle=angle_vu)
        forward.twin = backward
        backward.twin = forward

        node_u.out_edges.append(forward)
        node_v.out_edges.append(backward)

        half_edges.append(forward)
        half_edges.append(backward)

    for node in node_map.values():
        if not node.out_edges:
            continue
        node.out_edges.sort(key=lambda e: e.angle)
        for idx, edge in enumerate(node.out_edges):
            edge.sort_index = idx

    return node_map, half_edges


def _angle(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    return _normalize_angle(ang)


def _normalize_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    ang = angle % two_pi
    if ang < 0.0:
        ang += two_pi
    return ang


def _next_edge(edge: _HalfEdge) -> Optional[_HalfEdge]:
    twin = edge.twin
    if twin is None:
        return None
    origin = twin.origin
    if not origin.out_edges:
        return None
    idx = twin.sort_index
    if idx < 0:
        try:
            idx = origin.out_edges.index(twin)
        except ValueError:
            return None
    next_idx = (idx + 1) % len(origin.out_edges)
    return origin.out_edges[next_idx]


def _traverse_faces(half_edges: List[_HalfEdge]) -> List[List[Tuple[float, float]]]:
    cycles: List[List[Tuple[float, float]]] = []
    max_steps = len(half_edges) + 5

    for edge in half_edges:
        if edge.visited:
            continue
        cycle: List[Tuple[float, float]] = []
        current = edge
        steps = 0
        while True:
            steps += 1
            edge.visited = True
            cycle.append(edge.origin.point)
            nxt = _next_edge(edge)
            if nxt is None:
                cycle = []
                break
            edge = nxt
            if edge is current:
                cycle.append(edge.origin.point)
                break
            if steps > max_steps:
                logger.debug("Cycle traversal aborted due to step limit")
                cycle = []
                break

        if cycle and len(cycle) >= 4:
            cycles.append(cycle)

    return cycles


def _cycles_to_polygons(
    cycles: List[List[Tuple[float, float]]],
    wall_union: Polygon | MultiPolygon,
    hull: Polygon,
    hull_area: float,
    tol_mm: float,
    config: SpaceConfig,
) -> List[Polygon]:
    polys: List[Polygon] = []
    keep_threshold = hull_area * 0.999

    for coords in cycles:
        if len(coords) < 4:
            continue
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if poly.is_empty or not poly.is_valid:
            try:
                poly = poly.buffer(0)
            except Exception:
                continue
        if poly.is_empty or not isinstance(poly, Polygon):
            continue

        if float(poly.area) <= _EPS_AREA:
            continue

        if float(poly.area) >= keep_threshold:
            # Likely the exterior face
            continue

        try:
            if not poly.within(hull.buffer(tol_mm)):
                continue
        except Exception:
            continue

        # Exclude polygons overlapping with walls (they represent wall interiors)
        try:
            if poly.intersection(wall_union).area >= poly.area * 0.6:
                continue
        except Exception:
            pass

        try:
            poly = orient(poly, sign=-1.0)
            poly = poly.simplify(_SIMPLIFY_MM, preserve_topology=True)
        except Exception:
            pass

        polys.append(poly)

    return polys


def _assign_holes(polys: List[Polygon], tol_mm: float) -> List[Polygon]:
    if not polys:
        return []

    items: List[Dict[str, object]] = []
    for idx, poly in enumerate(polys):
        items.append(
            {
                "index": idx,
                "poly": poly,
                "area": float(abs(poly.area)),
                "parent": -1,
                "depth": 0,
                "children": [],
            }
        )

    items.sort(key=lambda item: item["area"], reverse=True)

    for i, item in enumerate(items):
        poly: Polygon = item["poly"]  # type: ignore[assignment]
        parent_idx = -1
        best_area = math.inf
        for j in range(i):
            parent_poly: Polygon = items[j]["poly"]  # type: ignore[index]
            try:
                if poly.within(parent_poly.buffer(tol_mm)):
                    parent_area = items[j]["area"]  # type: ignore[index]
                    if parent_area < best_area:
                        parent_idx = j
                        best_area = parent_area
            except Exception:
                continue

        item["parent"] = parent_idx
        if parent_idx != -1:
            item["depth"] = items[parent_idx]["depth"] + 1  # type: ignore[index]
            items[parent_idx]["children"].append(i)  # type: ignore[index]

    final_polys: List[Polygon] = []

    for i, item in enumerate(items):
        depth = item["depth"]  # type: ignore[index]
        if depth % 2 != 0:
            continue  # holes handled with parent shell

        shell_poly: Polygon = item["poly"]  # type: ignore[index]
        hole_indices: List[int] = [
            child_idx
            for child_idx in item["children"]  # type: ignore[index]
            if items[child_idx]["depth"] == depth + 1
        ]

        holes_coords: List[List[Tuple[float, float]]] = []
        for hole_idx in hole_indices:
            hole_poly: Polygon = items[hole_idx]["poly"]  # type: ignore[index]
            try:
                hole_poly = orient(hole_poly, sign=1.0)
                holes_coords.append(list(hole_poly.exterior.coords))
            except Exception:
                continue

        try:
            shell_poly = orient(shell_poly, sign=-1.0)
            final_poly = Polygon(shell_poly.exterior.coords, holes=holes_coords)
            if not final_poly.is_valid:
                final_poly = final_poly.buffer(0)
            if isinstance(final_poly, Polygon) and not final_poly.is_empty:
                final_polys.append(final_poly)
        except Exception:
            continue

    return final_polys


# Backwards compatibility alias
def polygonize_spaces_graph(
    dets: List[NormalizedDet],
    config: SpaceConfig | None = None,
) -> List[SpacePoly]:
    return find_space_cycles_graph(dets, config)
