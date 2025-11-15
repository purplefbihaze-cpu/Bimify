from __future__ import annotations

from typing import List, Sequence, Tuple
import math

from shapely.geometry import LineString
from shapely.ops import nearest_points

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.settings import get_settings


def _angle_of(line: LineString) -> float:
    coords = list(line.coords) if line is not None else []
    if len(coords) < 2:
        return 0.0
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _unit(direction_deg: float) -> tuple[float, float]:
    rad = math.radians(direction_deg)
    return math.cos(rad), math.sin(rad)


def _project_point_to_line_dir(cx: float, cy: float, ox: float, oy: float, ux: float, uy: float) -> tuple[float, float]:
    t = (cx - ox) * ux + (cy - oy) * uy
    return (ox + t * ux, oy + t * uy)


def refine_axes_using_edges_and_rf(
    axes: Sequence[WallAxis],
    *,
    image_lines: Sequence[LineString] | None,
    rf_norm: Sequence[NormalizedDet] | None,
) -> List[WallAxis]:
    """
    Minimal refinement: for each axis, if a nearby image line exists with small angle difference,
    project axis endpoints onto that direction preserving centroid and length.
    """
    if not axes:
        return list(axes)

    # Resolve tolerances from settings (fallback defaults)
    angle_tol_deg = 5.0
    pos_tol_mm = 20.0
    try:
        cfg = getattr(getattr(get_settings(), "geometry", None), "repair_level1", None)
        if cfg:
            angle_tol_deg = float(getattr(cfg, "angleTol_deg", angle_tol_deg))
            pos_tol_mm = float(getattr(cfg, "posTol_mm", pos_tol_mm))
    except Exception:
        pass

    out: List[WallAxis] = []
    for axis in axes:
        base = axis.axis
        if base is None or len(list(base.coords)) < 2:
            out.append(axis)
            continue

        best_line: LineString | None = None
        best_score = float("inf")
        base_angle = _angle_of(base)

        if image_lines:
            for cand in image_lines:
                if cand is None or len(list(cand.coords)) < 2:
                    continue
                cand_angle = _angle_of(cand)
                # angle diff normalized to [0, 90]
                diff = abs((cand_angle - base_angle + 180) % 180 - 90)
                angle_diff = min(abs(cand_angle - base_angle), 180 - abs(cand_angle - base_angle))
                if angle_diff > angle_tol_deg + 2.0:
                    continue
                # proximity score: distance between centroids
                try:
                    bcx, bcy = base.centroid.x, base.centroid.y
                    pcx, pcy = cand.centroid.x, cand.centroid.y
                    d = math.hypot(bcx - pcx, bcy - pcy)
                except Exception:
                    d = 1e9
                score = angle_diff * 10.0 + d
                if score < best_score:
                    best_score = score
                    best_line = cand

        if best_line is None:
            out.append(axis)
            continue

        # Project endpoints to best_line direction, anchored at original centroid
        (sx, sy), (ex, ey) = list(base.coords)[0], list(base.coords)[-1]
        try:
            length = math.hypot(ex - sx, ey - sy)
            cx, cy = base.centroid.x, base.centroid.y
        except Exception:
            out.append(axis)
            continue
        dir_deg = _angle_of(best_line)
        ux, uy = _unit(dir_deg)
        half = 0.5 * length
        start = (cx - ux * half, cy - uy * half)
        end = (cx + ux * half, cy + uy * half)
        refined = LineString([start, end])

        # If centroid drifted too far (shouldn't), fallback
        try:
            drift = refined.centroid.distance(base.centroid)
        except Exception:
            drift = pos_tol_mm + 1.0
        if drift > pos_tol_mm * 2.0:
            out.append(axis)
            continue

        out.append(
            WallAxis(
                detection=axis.detection,
                source_index=axis.source_index,
                axis=refined,
                width_mm=axis.width_mm,
                length_mm=axis.length_mm,
                centroid_mm=axis.centroid_mm,
                method=f"{axis.method}|final_fit",
                metadata=dict(axis.metadata or {}),
            )
        )

    return out


