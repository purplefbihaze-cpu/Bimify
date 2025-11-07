from __future__ import annotations

from dataclasses import dataclass
from typing import List

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet


@dataclass
class SpacePoly:
    polygon: Polygon
    area_m2: float


def polygonize_spaces_from_walls(dets: List[NormalizedDet]) -> List[SpacePoly]:
    walls = [d for d in dets if d.type == "WALL"]
    if not walls:
        return []
    wall_union = unary_union([w.geom for w in walls])
    # bounding hull from walls
    hull = wall_union.envelope.buffer(0.0)
    free = hull.difference(wall_union)
    polys: List[Polygon] = []
    if isinstance(free, Polygon):
        polys = [free]
    elif isinstance(free, MultiPolygon):
        polys = list(free.geoms)
    spaces = [SpacePoly(polygon=p, area_m2=float(p.area / 1_000_000.0)) for p in polys]
    return spaces


