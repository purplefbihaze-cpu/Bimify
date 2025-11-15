from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, mapping as shp_mapping
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis


@dataclass
class OverlayArtifacts:
    overlay: dict
    metrics: dict[str, Any]


def _largest_polygon(geom: Polygon | MultiPolygon | LineString | None) -> Polygon | None:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(geom.geoms, key=lambda g: g.area)
        except ValueError:
            return None
    return None


def _iou(a: Polygon | None, b: Polygon | None) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    uni = a.union(b).area
    return float(inter / uni) if uni > 1e-6 else 0.0


def build_overlay(
    *,
    normalized: Sequence[NormalizedDet],
    axes: Sequence[WallAxis],
) -> OverlayArtifacts:
    features: list[dict[str, Any]] = []
    wall_src: list[Polygon] = []
    wall_axis_polys: list[Polygon] = []

    for det in normalized:
        if det.type != "WALL":
            continue
        poly = _largest_polygon(det.geom)
        if poly is None:
            continue
        wall_src.append(poly)
        features.append({
            "type": "Feature",
            "properties": {"type": "WALL_SRC"},
            "geometry": shp_mapping(poly),
        })

    for ax in axes:
        if getattr(ax, "axis", None) is None:
            continue
        width = float(getattr(ax, "width_mm", 0.0) or 0.0)
        geom = ax.axis.buffer(width / 2.0, cap_style=2, join_style=2) if width > 0 else None
        if geom is None or geom.is_empty:
            continue
        wall_axis_polys.append(_largest_polygon(geom) or geom)
        features.append({
            "type": "Feature",
            "properties": {"type": "WALL_AXIS", "width_mm": width},
            "geometry": shp_mapping(ax.axis),
        })

    # Compute simple IoU summary (median IoU over walls)
    ious: list[float] = []
    try:
        merged_axes = unary_union([p for p in wall_axis_polys if p is not None])
        for src in wall_src:
            ious.append(_iou(src, merged_axes))
    except Exception:
        pass

    metrics = {
        "total_walls_src": int(len(wall_src)),
        "total_axes": int(len(axes)),
        "median_iou": float(np.median(ious)) if ious else 0.0,
    }
    overlay = {"type": "FeatureCollection", "features": features}
    
    # Validate that at least some features were created (even if empty, we return valid structure)
    if not isinstance(overlay, dict) or "features" not in overlay:
        raise ValueError("Overlay-Struktur ungültig: 'features' fehlt")
    
    return OverlayArtifacts(overlay=overlay, metrics=metrics)


def write_overlay(artifacts: OverlayArtifacts, path: Path) -> Path:
    if not artifacts or not artifacts.overlay:
        raise ValueError("Overlay-Artifacts sind leer")
    try:
        json_content = json.dumps(artifacts.overlay)
        path.write_text(json_content, encoding="utf-8")
        # Verify file was written
        if not path.exists():
            raise IOError(f"Overlay-Datei konnte nicht geschrieben werden: {path}")
        return path
    except Exception as exc:
        raise IOError(f"Fehler beim Schreiben der Overlay-Datei {path}: {exc}") from exc


