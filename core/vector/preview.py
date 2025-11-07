from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def save_geojson_thumbnail(geojson: dict[str, Any], out_path: Path, size: tuple[int, int] = (512, 512)) -> Path:
    features = geojson.get("features", [])
    if not features:
        return out_path

    min_x, min_y, max_x, max_y = _bounds(features)
    if min_x == max_x or min_y == max_y:
        max_x += 1
        max_y += 1

    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (9, 13, 18)

    scale_x = (width * 0.8) / (max_x - min_x)
    scale_y = (height * 0.8) / (max_y - min_y)
    scale = min(scale_x, scale_y)
    offset_x = (width - (max_x - min_x) * scale) / 2
    offset_y = (height - (max_y - min_y) * scale) / 2

    for feature in features:
        geom = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        color = _color_for(props.get("type"))
        if not geom:
            continue
        kind = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")
        if kind in {"polygon", "multipolygon"}:
            for ring in _iterate_rings(coords, kind):
                pts = _transform_points(ring, min_x, min_y, scale, offset_x, offset_y, height)
                if pts.size == 0:
                    continue
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.fillPoly(canvas, [pts], color=_fill_color(color), lineType=cv2.LINE_AA)
        elif kind in {"linestring", "multilinestring"}:
            for line in _iterate_lines(coords, kind):
                pts = _transform_points(line, min_x, min_y, scale, offset_x, offset_y, height)
                if pts.size == 0:
                    continue
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def _bounds(features: Iterable[dict[str, Any]]) -> tuple[float, float, float, float]:
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    def update(pt: tuple[float, float]) -> None:
        nonlocal min_x, min_y, max_x, max_y
        x, y = pt
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    for feature in features:
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        kind = (geom.get("type") or "").lower()
        if kind in {"polygon", "multipolygon"}:
            for ring in _iterate_rings(coords, kind):
                for pt in ring:
                    update(pt)
        elif kind in {"linestring", "multilinestring"}:
            for line in _iterate_lines(coords, kind):
                for pt in line:
                    update(pt)
        elif kind == "point":
            update(tuple(coords))
        elif kind == "multipoint":
            for pt in coords:
                update(tuple(pt))

    if min_x == float("inf"):
        min_x = min_y = 0.0
        max_x = max_y = 1.0
    return (min_x, min_y, max_x, max_y)


def _iterate_rings(coords: Any, kind: str):
    if kind == "polygon":
        return coords
    return [ring for polygon in coords for ring in polygon]


def _iterate_lines(coords: Any, kind: str):
    if kind == "linestring":
        return [coords]
    return coords


def _transform_points(
    points: Iterable[Iterable[float]],
    min_x: float,
    min_y: float,
    scale: float,
    offset_x: float,
    offset_y: float,
    height: int,
):
    out = []
    for x, y in points:
        sx = int((x - min_x) * scale + offset_x)
        sy = int(height - ((y - min_y) * scale + offset_y))
        out.append([sx, sy])
    if not out:
        return np.empty((0, 1, 2), dtype=np.int32)
    return np.array(out, dtype=np.int32).reshape((-1, 1, 2))


def _color_for(kind: str | None) -> tuple[int, int, int]:
    palette = {
        "wall": (180, 220, 255),
        "walls": (180, 220, 255),
        "wall_axis": (120, 180, 255),
        "door": (120, 240, 190),
        "window": (120, 190, 240),
        "space": (140, 120, 255),
    }
    return palette.get((kind or "").lower(), (130, 160, 210))


def _fill_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(c * 0.3) for c in color)


__all__ = ["save_geojson_thumbnail"]














