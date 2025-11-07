from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from uuid import uuid4

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import linemerge, snap, unary_union

from core.ml.postprocess_floorplan import NormalizedDet, estimate_wall_axes_and_thickness
from core.vector.geometry import merge_polygons as merge_polygons_helper


@dataclass
class VectorArtifacts:
    wall_polygons: List[Polygon]
    opening_polygons: List[Polygon]
    center_lines: List[LineString]


@dataclass
class PDFExportOptions:
    mode: str = "wall-fill"
    smooth_tolerance_mm: float = 5.0
    snap_tolerance_mm: float = 15.0
    orthogonal_tolerance_deg: float = 10.0
    include_background: bool = False

    @classmethod
    def from_schema(cls, schema: object) -> "PDFExportOptions":
        if schema is None:
            return cls()
        return cls(
            mode=getattr(schema, "mode", "wall-fill") or "wall-fill",
            smooth_tolerance_mm=float(getattr(schema, "smooth_tolerance_mm", 5.0) or 0.0),
            snap_tolerance_mm=float(getattr(schema, "snap_tolerance_mm", 15.0) or 0.0),
            orthogonal_tolerance_deg=float(getattr(schema, "orthogonal_tolerance_deg", 10.0) or 0.0),
            include_background=bool(getattr(schema, "include_background", False)),
        )


MM_TO_PT = 72.0 / 25.4


def generate_vector_pdf(
    *,
    normalized: Sequence[NormalizedDet],
    image_meta: dict | None,
    px_per_mm: float,
    options: PDFExportOptions,
    output_dir: Path,
) -> Tuple[Path, List[str]]:
    artifacts = _build_artifacts(normalized, options)
    if not artifacts.wall_polygons and not artifacts.center_lines:
        raise ValueError("Keine Geometrie zum Export verfügbar")

    warnings: List[str] = []

    merged_polygons = _merge_polygons(
        artifacts.wall_polygons,
        tolerance=options.smooth_tolerance_mm,
        snap_tolerance=options.snap_tolerance_mm,
    )

    merged_lines = _merge_lines(
        artifacts.center_lines,
        tolerance=options.smooth_tolerance_mm,
        snap_tolerance=options.snap_tolerance_mm,
        orthogonal_tolerance=options.orthogonal_tolerance_deg,
    )

    if options.mode == "wall-fill" and merged_polygons:
        drawable_polygons = merged_polygons
        drawable_lines: List[LineString] = []
    else:
        drawable_polygons = merged_polygons
        drawable_lines = merged_lines

    if not drawable_polygons and not drawable_lines:
        raise ValueError("Keine gültigen Vektorlinien nach der Normalisierung")

    min_x, min_y, max_x, max_y = _gather_bounds(drawable_polygons, drawable_lines)

    image_width_mm = None
    image_height_mm = None
    background_bytes: bytes | None = None
    if options.include_background and image_meta:
        width_px = _read_number(image_meta.get("width") or image_meta.get("Width"))
        height_px = _read_number(image_meta.get("height") or image_meta.get("Height"))
        if width_px and height_px and px_per_mm > 0:
            image_width_mm = width_px / px_per_mm
            image_height_mm = height_px / px_per_mm
            min_x = min(min_x, 0.0)
            min_y = min(min_y, 0.0)
            max_x = max(max_x, image_width_mm)
            max_y = max(max_y, image_height_mm)
            background_bytes = _resolve_background_bytes(image_meta)
            if background_bytes is None:
                warnings.append("Hintergrundbild konnte nicht eingebettet werden (keine Datenquelle)")
        else:
            warnings.append("Hintergrundbild konnte nicht eingebettet werden (fehlende Maße)")
    elif options.include_background:
        warnings.append("Kein Hintergrundbild verfügbar – Option ignoriert")

    margin = 50.0
    width_span = max(max_x - min_x, 1.0)
    height_span = max(max_y - min_y, 1.0)
    translated_polygons = [
        affinity.translate(poly, xoff=-min_x + margin, yoff=-min_y + margin) for poly in drawable_polygons
    ]
    translated_openings = [
        affinity.translate(poly, xoff=-min_x + margin, yoff=-min_y + margin) for poly in artifacts.opening_polygons
    ]
    translated_lines = [
        affinity.translate(line, xoff=-min_x + margin, yoff=-min_y + margin) for line in drawable_lines
    ]

    width_total = width_span + margin * 2
    height_total = height_span + margin * 2

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{uuid4().hex}.pdf"
    pdf_path = output_dir / file_name
    c = canvas.Canvas(str(pdf_path), pagesize=(width_total * MM_TO_PT, height_total * MM_TO_PT))

    if image_width_mm and image_height_mm and background_bytes:
        try:
            image_reader = ImageReader(BytesIO(background_bytes))
            c.saveState()
            c.translate((margin - min_x) * MM_TO_PT, (margin - min_y) * MM_TO_PT)
            c.drawImage(
                image_reader,
                0,
                0,
                width=image_width_mm * MM_TO_PT,
                height=image_height_mm * MM_TO_PT,
                mask="auto",
                preserveAspectRatio=True,
            )
            c.restoreState()
        except Exception:
            warnings.append("Hintergrundbild konnte nicht eingebettet werden (Ladefehler)")

    wall_fill = HexColor("#0f172a")
    wall_stroke = HexColor("#38bdf8")
    opening_stroke = HexColor("#fbbf24")
    line_stroke = HexColor("#f87171")

    for poly in translated_polygons:
        _draw_polygon(c, poly, fill_color=wall_fill, stroke_color=wall_stroke, stroke_width_mm=1.2)

    for poly in translated_openings:
        _draw_polygon(c, poly, fill_color=None, stroke_color=opening_stroke, stroke_width_mm=1.2)

    for line in translated_lines:
        _draw_line(c, line, stroke_color=line_stroke, stroke_width_mm=1.0)

    c.showPage()
    c.save()

    return pdf_path, warnings


def _build_artifacts(normalized: Sequence[NormalizedDet], options: ExportPDFOptions) -> VectorArtifacts:
    wall_polygons: List[Polygon] = []
    opening_polygons: List[Polygon] = []
    center_lines: List[LineString] = []

    for det in normalized:
        geom = det.geom
        if det.type == "WALL":
            if isinstance(geom, Polygon):
                wall_polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                wall_polygons.extend(list(geom.geoms))
        elif det.type in {"DOOR", "WINDOW"}:
            if isinstance(geom, Polygon):
                opening_polygons.append(geom)
            elif isinstance(geom, MultiPolygon):
                opening_polygons.extend(list(geom.geoms))

    axes_info = estimate_wall_axes_and_thickness(normalized)
    for axis_info in axes_info:
        axis = axis_info.axis
        width = axis_info.width_mm
        if not isinstance(axis, LineString):
            continue
        center_lines.append(axis)
        if width > 0:
            buffered = axis.buffer(width / 2.0, cap_style=2, join_style=2)
            if isinstance(buffered, Polygon):
                wall_polygons.append(buffered)
            elif isinstance(buffered, MultiPolygon):
                wall_polygons.extend(list(buffered.geoms))

    return VectorArtifacts(
        wall_polygons=wall_polygons,
        opening_polygons=opening_polygons,
        center_lines=center_lines,
    )


def _merge_polygons(polygons: Iterable[Polygon], tolerance: float, snap_tolerance: float) -> List[Polygon]:
    return merge_polygons_helper(polygons, tolerance=tolerance, snap_tolerance=snap_tolerance)


def _merge_lines(
    lines: Iterable[LineString],
    tolerance: float,
    snap_tolerance: float,
    orthogonal_tolerance: float,
) -> List[LineString]:
    cleaned: List[LineString] = []
    for line in lines:
        if line.is_empty:
            continue
        candidate = line
        if tolerance > 0 and len(candidate.coords) > 2:
            try:
                candidate = candidate.simplify(tolerance, preserve_topology=False)
            except Exception:
                pass
        cleaned.append(candidate)

    if not cleaned:
        return []

    merged_geom = unary_union(cleaned)
    if snap_tolerance > 0:
        merged_geom = snap(merged_geom, merged_geom, snap_tolerance)
    merged_geom = linemerge(merged_geom)

    merged_lines: List[LineString] = []
    if isinstance(merged_geom, LineString):
        merged_lines = [merged_geom]
    elif isinstance(merged_geom, MultiLineString):
        merged_lines = list(merged_geom.geoms)
    else:
        merged_lines = [geom for geom in merged_geom.geoms if isinstance(geom, LineString)] if hasattr(merged_geom, "geoms") else []

    if orthogonal_tolerance > 0:
        merged_lines = [
            _orthogonalize_line(line, orthogonal_tolerance) if len(line.coords) >= 2 else line for line in merged_lines
        ]

    return merged_lines


def _orthogonalize_line(line: LineString, tolerance_deg: float) -> LineString:
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    new_coords = [coords[0]]
    for index in range(1, len(coords)):
        prev = new_coords[-1]
        current = coords[index]
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]
        if dx == 0 and dy == 0:
            new_coords.append(current)
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 180.0
        if min(angle, abs(angle - 180.0)) <= tolerance_deg:
            new_coords.append((current[0], prev[1]))
        elif abs(angle - 90.0) <= tolerance_deg:
            new_coords.append((prev[0], current[1]))
        else:
            new_coords.append(current)
    return LineString(new_coords)


def _gather_bounds(polygons: Iterable[Polygon], lines: Iterable[LineString]) -> Tuple[float, float, float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for poly in polygons:
        if poly.is_empty:
            continue
        bounds = poly.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    for line in lines:
        if line.is_empty:
            continue
        bounds = line.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    if min_x == float("inf"):
        min_x = min_y = 0.0
        max_x = max_y = 1.0

    return min_x, min_y, max_x, max_y


def _draw_polygon(
    pdf: canvas.Canvas,
    polygon: Polygon,
    *,
    fill_color: HexColor | None,
    stroke_color: HexColor | None,
    stroke_width_mm: float,
) -> None:
    if polygon.is_empty:
        return

    exterior = [(float(x) * MM_TO_PT, float(y) * MM_TO_PT) for x, y in polygon.exterior.coords]
    if len(exterior) < 2:
        return

    pdf.saveState()
    pdf.setLineWidth(max(stroke_width_mm, 0.1) * MM_TO_PT)
    if stroke_color:
        pdf.setStrokeColor(stroke_color)
    if fill_color:
        pdf.setFillColor(fill_color)

    path = pdf.beginPath()
    first_x, first_y = exterior[0]
    path.moveTo(first_x, first_y)
    for x, y in exterior[1:]:
        path.lineTo(x, y)
    path.close()

    fill_flag = 1 if fill_color else 0
    stroke_flag = 1 if stroke_color else 0
    pdf.drawPath(path, fill=fill_flag, stroke=stroke_flag)

    if fill_color and polygon.interiors:
        pdf.setFillColorRGB(1.0, 1.0, 1.0)
        for interior in polygon.interiors:
            coords = [(float(x) * MM_TO_PT, float(y) * MM_TO_PT) for x, y in interior.coords]
            if len(coords) < 2:
                continue
            hole_path = pdf.beginPath()
            hx, hy = coords[0]
            hole_path.moveTo(hx, hy)
            for x, y in coords[1:]:
                hole_path.lineTo(x, y)
            hole_path.close()
            pdf.drawPath(hole_path, fill=1, stroke=0)

    pdf.restoreState()


def _draw_line(pdf: canvas.Canvas, line: LineString, *, stroke_color: HexColor, stroke_width_mm: float) -> None:
    coords = list(line.coords)
    if len(coords) < 2:
        return
    pdf.saveState()
    pdf.setStrokeColor(stroke_color)
    pdf.setLineWidth(max(stroke_width_mm, 0.1) * MM_TO_PT)
    for start, end in zip(coords[:-1], coords[1:]):
        x1, y1 = float(start[0]) * MM_TO_PT, float(start[1]) * MM_TO_PT
        x2, y2 = float(end[0]) * MM_TO_PT, float(end[1]) * MM_TO_PT
        pdf.line(x1, y1, x2, y2)
    pdf.restoreState()


def _resolve_background_bytes(image_meta: dict | None) -> bytes | None:
    if not image_meta:
        return None
    for key in ("base64", "Base64", "data"):
        value = image_meta.get(key)
        if isinstance(value, str) and value:
            encoded = value
            if value.startswith("data:"):
                _, _, encoded = value.partition(",")
            try:
                return base64.b64decode(encoded)
            except Exception:
                return None
    return None


def _read_number(value: object) -> float | None:
    try:
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number
    except (TypeError, ValueError):
        return None
    return None


