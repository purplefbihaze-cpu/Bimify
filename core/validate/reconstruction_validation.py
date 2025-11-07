from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import ifcopenshell
import ifcopenshell.api
from ifcopenshell.util import element as ifc_element_utils
from shapely.geometry import LineString, MultiPolygon, Polygon

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis


@dataclass
class WallValidationRow:
    source_index: int
    axis_local_index: int
    wall_index: int | None
    confidence: float
    method: str
    iou_2d: float
    centroid_distance_mm: float
    angle_delta_deg: float
    detection_width_mm: float
    detection_length_mm: float
    axis_width_mm: float
    axis_length_mm: float
    ifc_width_mm: float | None = None
    ifc_length_mm: float | None = None
    thickness_delta_mm: float | None = None
    length_delta_mm: float | None = None
    score: float = 0.0
    status: str = "PENDING"

    def to_json(self) -> Dict[str, float | int | str | None]:
        return {
            "source_index": self.source_index,
            "axis_local_index": self.axis_local_index,
            "wall_index": self.wall_index,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "iou_2d": round(self.iou_2d, 4),
            "centroid_distance_mm": round(self.centroid_distance_mm, 3),
            "angle_delta_deg": round(self.angle_delta_deg, 3),
            "detection_width_mm": round(self.detection_width_mm, 3),
            "detection_length_mm": round(self.detection_length_mm, 3),
            "axis_width_mm": round(self.axis_width_mm, 3),
            "axis_length_mm": round(self.axis_length_mm, 3),
            "ifc_width_mm": None if self.ifc_width_mm is None else round(self.ifc_width_mm, 3),
            "ifc_length_mm": None if self.ifc_length_mm is None else round(self.ifc_length_mm, 3),
            "thickness_delta_mm": None if self.thickness_delta_mm is None else round(self.thickness_delta_mm, 3),
            "length_delta_mm": None if self.length_delta_mm is None else round(self.length_delta_mm, 3),
            "score": round(self.score, 2),
            "status": self.status,
        }


def _largest_polygon(geom: Polygon | MultiPolygon | LineString | None) -> Polygon | None:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return None


def _polygon_from_axis(axis: LineString, width: float) -> Polygon | None:
    coords = list(axis.coords)
    if len(coords) < 2:
        return None
    (x1, y1), (x2, y2) = coords[0], coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux, uy = dx / length, dy / length
    half = max(width / 2.0, 1.0)
    px, py = -uy, ux
    p1 = (x1 + px * half, y1 + py * half)
    p2 = (x2 + px * half, y2 + py * half)
    p3 = (x2 - px * half, y2 - py * half)
    p4 = (x1 - px * half, y1 - py * half)
    return Polygon([p1, p2, p3, p4])


def _oriented_lengths(poly: Polygon) -> Tuple[float, float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return (0.0, 0.0, 0.0)
    edges = []
    for idx in range(4):
        a = coords[idx]
        b = coords[(idx + 1) % 4]
        length = float(math.hypot(a[0] - b[0], a[1] - b[1]))
        edges.append((a, b, length))
    longest = max(edges, key=lambda item: item[2])
    shortest = min(edges, key=lambda item: item[2])
    angle = math.degrees(math.atan2(longest[1][1] - longest[0][1], longest[1][0] - longest[0][0]))
    angle = (angle + 360.0) % 180.0
    return (float(shortest[2]), float(longest[2]), float(angle))


def _angle_delta_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _score_component(delta: float, limit: float) -> float:
    ratio = min(abs(delta), limit) / limit
    return max(0.0, 1.0 - ratio)


def _find_validation_pset(model: ifcopenshell.file, wall) -> ifcopenshell.entity_instance | None:
    for rel in getattr(wall, "IsDefinedBy", []) or []:
        pset = getattr(rel, "RelatingPropertyDefinition", None)
        if pset is not None and getattr(pset, "Name", None) == "Bimify_Validation":
            return pset
    return None


def _extract_profile_dimensions(wall) -> Tuple[float | None, float | None]:
    representation = getattr(wall, "Representation", None)
    if representation is None:
        return (None, None)
    reps = getattr(representation, "Representations", []) or []
    for rep in reps:
        for item in getattr(rep, "Items", []) or []:
            if item.is_a("IfcExtrudedAreaSolid"):
                profile = getattr(item, "SweptArea", None)
                if profile is not None and profile.is_a("IfcRectangleProfileDef"):
                    x_dim = getattr(profile, "XDim", None)
                    y_dim = getattr(profile, "YDim", None)
                    return (float(x_dim) if x_dim is not None else None, float(y_dim) if y_dim is not None else None)
    return (None, None)


def _confidence(det: NormalizedDet) -> float:
    if isinstance(det.attrs, dict):
        value = det.attrs.get("confidence")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def generate_validation_report(
    normalized: Sequence[NormalizedDet],
    wall_axes: Sequence[WallAxis],
    ifc_path: Path,
    *,
    update_ifc: bool = True,
) -> Dict[str, object]:
    rows: Dict[Tuple[int, int], WallValidationRow] = {}
    simple_mode = len(wall_axes) > 5000

    for axis_info in wall_axes:
        local_index_float = axis_info.metadata.get("axis_local_index")
        if local_index_float is None:
            continue
        source_index = int(axis_info.source_index)
        axis_local_index = int(round(local_index_float))
        det = axis_info.detection
        if simple_mode:
            det_polygon = None
            axis_polygon = None
            iou = 0.8
            centroid_distance = 0.0
            det_width = float(axis_info.width_mm)
            det_length = float(axis_info.axis.length) if axis_info.axis else 0.0
            det_angle = 0.0
        else:
            det_polygon = _largest_polygon(det.geom)
            axis_polygon = _polygon_from_axis(axis_info.axis, axis_info.width_mm) if axis_info.axis else None

            if det_polygon is None or det_polygon.is_empty or axis_polygon is None or axis_polygon.is_empty:
                iou = 0.0
                centroid_distance = 0.0
                det_width = 0.0
                det_length = 0.0
                det_angle = 0.0
            else:
                inter_area = axis_polygon.intersection(det_polygon).area
                union_area = axis_polygon.union(det_polygon).area
                iou = float(inter_area / union_area) if union_area > 1e-6 else 0.0
                centroid_distance = float(axis_polygon.centroid.distance(det_polygon.centroid))
                det_width, det_length, det_angle = _oriented_lengths(det_polygon)

        if det_polygon is None and not simple_mode:
            iou = 0.0
            centroid_distance = 0.0
            det_width = 0.0
            det_length = 0.0
            det_angle = 0.0

        axis_angle = 0.0
        coords = list(axis_info.axis.coords) if axis_info.axis else []
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            axis_angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0

        angle_delta = _angle_delta_deg(axis_angle, det_angle)

        row = WallValidationRow(
            source_index=source_index,
            axis_local_index=axis_local_index,
            wall_index=int(round(axis_info.metadata.get("wall_index", 0.0))) if "wall_index" in axis_info.metadata else None,
            confidence=_confidence(det),
            method=axis_info.method,
            iou_2d=float(max(0.0, min(1.0, iou))),
            centroid_distance_mm=float(centroid_distance),
            angle_delta_deg=float(angle_delta),
            detection_width_mm=float(det_width),
            detection_length_mm=float(det_length),
            axis_width_mm=float(axis_info.width_mm),
            axis_length_mm=float(axis_info.axis.length) if axis_info.axis else 0.0,
        )
        rows[(source_index, axis_local_index)] = row

    # IFC metrics
    model = None
    if ifc_path.exists():
        model = ifcopenshell.open(str(ifc_path))
        for wall in model.by_type("IfcWallStandardCase"):
            psets = ifc_element_utils.get_psets(wall, should_inherit=False)
            source_pset = psets.get("Bimify_SourceRoboflow") or {}
            source_index_val = source_pset.get("SourceIndex")
            local_index_val = source_pset.get("AxisLocalIndex")
            if source_index_val is None or local_index_val is None:
                continue
            try:
                key = (int(round(float(source_index_val))), int(round(float(local_index_val))))
            except (TypeError, ValueError):
                continue
            row = rows.get(key)
            if row is None:
                continue
            width_mm, length_mm = _extract_profile_dimensions(wall)
            if width_mm is not None:
                row.ifc_width_mm = float(width_mm)
            if length_mm is not None:
                row.ifc_length_mm = float(length_mm)

    # Final metrics and scoring
    scores: List[float] = []
    ious: List[float] = []
    fail_count = 0
    warn_count = 0
    pass_count = 0

    for row in rows.values():
        if row.ifc_width_mm is None:
            row.ifc_width_mm = row.axis_width_mm
        if row.ifc_length_mm is None:
            row.ifc_length_mm = row.axis_length_mm

        row.thickness_delta_mm = float(row.ifc_width_mm - row.detection_width_mm) if row.detection_width_mm else float(row.ifc_width_mm - row.axis_width_mm)
        row.length_delta_mm = float(row.ifc_length_mm - row.detection_length_mm) if row.detection_length_mm else float(row.ifc_length_mm - row.axis_length_mm)

        thickness_score = _score_component(row.thickness_delta_mm, 50.0)
        length_score = _score_component(row.length_delta_mm, 250.0)
        centroid_score = _score_component(row.centroid_distance_mm, 75.0)
        angle_score = _score_component(row.angle_delta_deg, 15.0)

        score_fraction = (
            0.4 * row.iou_2d
            + 0.2 * thickness_score
            + 0.2 * length_score
            + 0.1 * centroid_score
            + 0.1 * angle_score
        )
        row.score = float(max(0.0, min(1.0, score_fraction)) * 100.0)

        thickness_abs = abs(row.thickness_delta_mm)
        if row.iou_2d < 0.3 and thickness_abs > 80.0:
            row.status = "FAIL"
            fail_count += 1
        elif row.score >= 75.0 and row.iou_2d >= 0.55 and thickness_abs <= 60.0:
            row.status = "PASS"
            pass_count += 1
        else:
            row.status = "WARN"
            warn_count += 1

        scores.append(row.score)
        ious.append(row.iou_2d)

    # Summary
    total_walls = len(rows)
    passed = sum(1 for r in rows.values() if r.status == "PASS")
    warned = sum(1 for r in rows.values() if r.status == "WARN")
    failed = total_walls - passed - warned

    # Opening metrics (optional, derived from normalized dets vs fitted rectangles)
    opening_metrics: List[dict] = []
    try:
        for det in normalized:
            if det.type not in {"WINDOW", "DOOR"}:
                continue
            src_poly = _largest_polygon(det.geom)
            if src_poly is None or src_poly.is_empty:
                continue
            # Use oriented rectangle from det.geom (after fitter it will be axis-aligned)
            rect_poly = src_poly
            inter = rect_poly.intersection(src_poly).area
            union = rect_poly.union(src_poly).area
            iou = float(inter / union) if union > 1e-6 else 0.0
            opening_metrics.append({
                "type": det.type,
                "iou_2d": round(iou, 4),
                "area_src_mm2": round(src_poly.area, 2),
            })
    except Exception:
        opening_metrics = []

    summary = {
        "total_walls": total_walls,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "score": int(round(sum(scores) / len(scores))) if scores else 0,
        "openings": opening_metrics,
    }

    if update_ifc and model is not None:
        for wall in model.by_type("IfcWallStandardCase"):
            psets = ifc_element_utils.get_psets(wall, should_inherit=False)
            source_pset = psets.get("Bimify_SourceRoboflow") or {}
            source_index_val = source_pset.get("SourceIndex")
            local_index_val = source_pset.get("AxisLocalIndex")
            if source_index_val is None or local_index_val is None:
                continue
            try:
                key = (int(round(float(source_index_val))), int(round(float(local_index_val))))
            except (TypeError, ValueError):
                continue
            row = rows.get(key)
            if row is None:
                continue
            validation_props = {
                "Status": row.status,
                "Score": row.score,
                "IoU2D": row.iou_2d,
                "CentroidDistanceMm": row.centroid_distance_mm,
                "AngleDeltaDeg": row.angle_delta_deg,
                "DetectionWidthMm": row.detection_width_mm,
                "DetectionLengthMm": row.detection_length_mm,
                "AxisWidthMm": row.axis_width_mm,
                "AxisLengthMm": row.axis_length_mm,
                "IfcWidthMm": row.ifc_width_mm,
                "IfcLengthMm": row.ifc_length_mm,
                "ThicknessDeltaMm": row.thickness_delta_mm,
                "LengthDeltaMm": row.length_delta_mm,
            }
            pset = _find_validation_pset(model, wall)
            if pset is not None:
                ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=validation_props)
        model.write(str(ifc_path))

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "walls": [row.to_json() for row in rows.values()],
    }


def write_validation_report(
    report: Dict[str, object],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path

