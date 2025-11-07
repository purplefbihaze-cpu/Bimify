from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import ifcopenshell
import ifcopenshell.api
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.reconstruct.spaces import SpacePoly
from core.reconstruct.openings import OpeningAssignment, snap_openings_to_walls
from core.vector.geometry import merge_polygons as merge_polygons_helper


logger = logging.getLogger(__name__)


STANDARD_WALL_THICKNESSES_MM: Sequence[float] = (115.0, 240.0, 300.0, 400.0, 500.0)
@dataclass
class OpeningPlacement:
    width_mm: float
    center_xy: Tuple[float, float]
    axis_vec: Tuple[float, float]
    depth_mm: float


DEFAULT_WALL_THICKNESS_STANDARDS_MM: Tuple[float, ...] = (115.0, 240.0, 300.0, 400.0, 500.0)


def collect_wall_polygons(
    normalized: Sequence[NormalizedDet],
    *,
    smooth_tolerance_mm: float = 3.0,
    snap_tolerance_mm: float = 10.0,
) -> Dict[int, Polygon | MultiPolygon]:
    """Return cleaned wall polygons keyed by their source index within wall detections."""

    wall_polygons: Dict[int, Polygon | MultiPolygon] = {}
    walls = [nd for nd in normalized if nd.type == "WALL"]

    for source_index, det in enumerate(walls):
        geom = det.geom
        polygons: list[Polygon] = []
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))

        if not polygons:
            continue

        merged = merge_polygons_helper(
            polygons,
            tolerance=max(smooth_tolerance_mm, 0.0),
            snap_tolerance=max(snap_tolerance_mm, 0.0),
        )
        if not merged:
            continue

        if len(merged) == 1:
            wall_polygons[source_index] = merged[0]
        else:
            wall_polygons[source_index] = unary_union(merged)

    return wall_polygons


def snap_thickness_mm(
    value_mm: float | None,
    *,
    standards: Sequence[float] = STANDARD_WALL_THICKNESSES_MM,
    is_external: bool | None = None,
    default_external_mm: float = 240.0,
    default_internal_mm: float = 115.0,
) -> float:
    """Snap measured wall thickness to the closest standard size."""
    candidates = [float(v) for v in standards if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0]
    if not candidates:
        candidates = [default_external_mm, default_internal_mm]
    candidates = [c for c in candidates if c > 0.0]
    if not candidates:
        return max(float(value_mm or 0.0), 40.0)

    fallback = default_external_mm if is_external else default_internal_mm
    if fallback <= 0.0:
        fallback = candidates[0]

    if value_mm is None or not math.isfinite(value_mm) or value_mm <= 0.0:
        return fallback

    closest = min(candidates, key=lambda option: abs(option - value_mm))
    return closest


def write_ifc_with_spaces(
    normalized: List[NormalizedDet],
    spaces: List[SpacePoly],
    out_path: Path,
    project_name: str = "Bimify Project",
    storey_name: str = "EG",
    storey_elevation: float = 0.0,
    wall_axes: List[WallAxis] | None = None,
    wall_polygons: Dict[int, Polygon | MultiPolygon] | None = None,
    storey_height_mm: float | None = None,
    door_height_mm: float | None = None,
    window_height_mm: float | None = None,
    window_head_elevation_mm: float | None = None,
    px_per_mm: float | None = None,
    calibration: dict | None = None,
    schema_version: str = "IFC4",
    wall_thickness_standards_mm: Sequence[float] | None = None,
) -> Path:
    if schema_version:
        model = ifcopenshell.api.run("project.create_file", version=schema_version)
    else:
        model = ifcopenshell.api.run("project.create_file")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name=project_name)
    ifcopenshell.api.run(
        "unit.assign_unit",
        model,
        length={
            "is_metric": True,
            "raw": "MILLIMETERS",
        },
    )
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=context,
    )
    precision_target = 1e-6
    for ctx in (context, body):
        if ctx is None:
            continue
        try:
            current_precision = getattr(ctx, "Precision", None)
        except Exception:
            current_precision = None
        if current_precision is None or (isinstance(current_precision, (int, float)) and current_precision > precision_target):
            try:
                ctx.Precision = float(precision_target)
            except Exception:
                try:
                    setattr(ctx, "Precision", float(precision_target))
                except Exception:
                    pass
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcBuildingStorey",
        name=storey_name,
    )
    try:
        storey.Elevation = float(storey_elevation)
    except Exception:
        storey.Elevation = 0.0
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])

    if calibration:
        try:
            calib_pset = ifcopenshell.api.run("pset.add_pset", model, product=project, name="Bimify_ProjectCalibration")
            raw_a = calibration.get("point_a") if isinstance(calibration.get("point_a"), (list, tuple)) else [0.0, 0.0]
            raw_b = calibration.get("point_b") if isinstance(calibration.get("point_b"), (list, tuple)) else [0.0, 0.0]
            point_a = list(raw_a)[:2]
            point_b = list(raw_b)[:2]
            if len(point_a) < 2:
                point_a.extend([0.0] * (2 - len(point_a)))
            if len(point_b) < 2:
                point_b.extend([0.0] * (2 - len(point_b)))
            properties = {
                "ScalePxPerMm": float(calibration.get("px_per_mm", px_per_mm or 0.0)),
                "PixelDistance": float(calibration.get("pixel_distance", 0.0)),
                "RealDistanceMm": float(calibration.get("real_distance_mm", 0.0)),
                "PointAXpx": float(point_a[0]),
                "PointAYpx": float(point_a[1]),
                "PointBXpx": float(point_b[0]),
                "PointBYpx": float(point_b[1]),
                "Unit": str(calibration.get("unit", "mm")),
            }
            ifcopenshell.api.run("pset.edit_pset", model, pset=calib_pset, properties=properties)
        except Exception:
            pass

    height_mm = float(storey_height_mm or 3000.0)
    thickness_standards = _prepare_thickness_standards(wall_thickness_standards_mm)

    def _make_direction(x: float, y: float, z: float):
        return model.create_entity("IfcDirection", DirectionRatios=(float(x), float(y), float(z)))

    def _make_point(x: float, y: float, z: float = 0.0):
        return model.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y), float(z)))

    def _make_local_placement(rel_to, location=(0.0, 0.0, 0.0)):
        loc = _make_point(*location)
        axis = _make_direction(0.0, 0.0, 1.0)
        ref = _make_direction(1.0, 0.0, 0.0)
        placement_3d = model.create_entity("IfcAxis2Placement3D", Location=loc, Axis=axis, RefDirection=ref)
        return model.create_entity("IfcLocalPlacement", PlacementRelTo=rel_to, RelativePlacement=placement_3d)

    def _ensure_spatial_placement(spatial, rel_to, location=(0.0, 0.0, 0.0)):
        if getattr(spatial, "ObjectPlacement", None) is None:
            spatial.ObjectPlacement = _make_local_placement(rel_to, location)

    _ensure_spatial_placement(site, None)
    _ensure_spatial_placement(building, getattr(site, "ObjectPlacement", None))
    _ensure_spatial_placement(storey, getattr(building, "ObjectPlacement", None), (0.0, 0.0, storey_elevation))

    def _ensure_product_placement(product):
        if getattr(product, "ObjectPlacement", None) is None:
            product.ObjectPlacement = _make_local_placement(storey.ObjectPlacement, (0.0, 0.0, 0.0))

    def _resolve_polygon(nd: NormalizedDet):
        geom = nd.geom
        if isinstance(geom, Polygon):
            return geom
        if isinstance(geom, MultiPolygon):
            try:
                return max(list(geom.geoms), key=lambda g: g.area)
            except ValueError:
                return None
        return None

    def _polygon_from_axis(axis: LineString, width: float) -> Polygon | None:
        coords = list(axis.coords)
        if len(coords) < 2:
            return None
        (x1, y1), (x2, y2) = coords[0], coords[-1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-3:
            return None
        ux, uy = dx / length, dy / length
        half = max(width / 2.0, 1.0)
        px, py = -uy, ux
        p1 = (x1 + px * half, y1 + py * half)
        p2 = (x2 + px * half, y2 + py * half)
        p3 = (x2 - px * half, y2 - py * half)
        p4 = (x1 - px * half, y1 - py * half)
        return Polygon([p1, p2, p3, p4])

    def _assign_wall_geometry(
        axis_info: WallAxis,
        wall_entity,
        source_geom_override: Polygon | MultiPolygon | None,
    ) -> None:
        axis = axis_info.axis
        detection = axis_info.detection
        is_external = getattr(detection, "is_external", None)
        width_raw = float(axis_info.width_mm or 0.0)
        width_value = _snap_wall_thickness(width_raw, is_external=is_external, standards=thickness_standards)
        axis_length = float(axis.length)

        polygon_candidates: list[Polygon] = []
        if source_geom_override is not None:
            if isinstance(source_geom_override, Polygon):
                polygon_candidates.append(source_geom_override)
            elif isinstance(source_geom_override, MultiPolygon):
                polygon_candidates.extend(list(source_geom_override.geoms))

        if not polygon_candidates:
            resolved = _resolve_polygon(axis_info.detection)
            if resolved is not None:
                polygon_candidates.append(resolved)

        selected_polygon: Polygon | None = None
        if polygon_candidates:
            if len(polygon_candidates) == 1:
                selected_polygon = polygon_candidates[0]
            else:
                midpoint = None
                try:
                    if axis_length > 1e-6:
                        midpoint_geom = axis.interpolate(0.5, normalized=True)
                        midpoint = (float(midpoint_geom.x), float(midpoint_geom.y))
                except Exception:
                    midpoint = None

                if midpoint is not None:
                    px, py = midpoint
                    containing = []
                    for poly in polygon_candidates:
                        try:
                            if poly.buffer(1.0).contains(Point(px, py)):
                                containing.append(poly)
                        except Exception:
                            continue
                    if containing:
                        selected_polygon = max(containing, key=lambda poly: poly.area)
                    else:
                        selected_polygon = min(
                            polygon_candidates,
                            key=lambda poly: poly.distance(Point(px, py)),
                        )
                else:
                    selected_polygon = max(polygon_candidates, key=lambda poly: poly.area)

        rect_profile = None
        rect_position = None

        if axis_length > 1e-3:
            try:
                x_start, y_start = axis.coords[0]
                x_end, y_end = axis.coords[-1]
            except Exception:
                x_start = y_start = x_end = y_end = 0.0
            dx = float(x_end) - float(x_start)
            dy = float(y_end) - float(y_start)
            length_vec = math.hypot(dx, dy)
            if length_vec > 1e-6:
                dir_x = dx / length_vec
                dir_y = dy / length_vec
                try:
                    mid_point = axis.interpolate(0.5, normalized=True)
                    mid_x, mid_y = float(mid_point.x), float(mid_point.y)
                except Exception:
                    mid_x = float((x_start + x_end) / 2.0)
                    mid_y = float((y_start + y_end) / 2.0)
                rect_profile = model.create_entity(
                    "IfcRectangleProfileDef",
                    ProfileType="AREA",
                    XDim=max(axis_length, 1.0),
                    YDim=width_value,
                )
                rect_position = model.create_entity(
                    "IfcAxis2Placement3D",
                    Location=_make_point(mid_x, mid_y, 0.0),
                    Axis=_make_direction(0.0, 0.0, 1.0),
                    RefDirection=_make_direction(dir_x, dir_y, 0.0),
                )

        profile = None
        position = None

        if selected_polygon is not None:
            # Prefer the original detection vertex order to preserve input sequence
            base_poly = None
            try:
                base_poly = getattr(detection, "geom")
            except Exception:
                base_poly = None
            if isinstance(base_poly, Polygon):
                coords = [(float(x), float(y)) for x, y in list(base_poly.exterior.coords)]
            else:
                coords = [(float(x), float(y)) for x, y in list(selected_polygon.exterior.coords)]
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                selected_polygon = None
            else:
                points = [_make_point(x, y, 0.0) for x, y in coords]
                if len(points) >= 3:
                    points.append(points[0])
                    polyline = model.create_entity("IfcPolyline", Points=points)
                    profile = model.create_entity(
                        "IfcArbitraryClosedProfileDef",
                        ProfileType="AREA",
                        OuterCurve=polyline,
                    )
                    position = model.create_entity(
                        "IfcAxis2Placement3D",
                        Location=_make_point(0.0, 0.0, 0.0),
                        Axis=_make_direction(0.0, 0.0, 1.0),
                        RefDirection=_make_direction(1.0, 0.0, 0.0),
                    )

        if profile is None:
            profile = rect_profile
            position = rect_position

        if profile is None:
            poly = _polygon_from_axis(axis, width_value)
            if poly is None:
                return
            coords = [(float(x), float(y)) for x, y in list(poly.exterior.coords)]
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) < 3:
                return
            points = [_make_point(x, y, 0.0) for x, y in coords]
            if len(points) < 3:
                return
            points.append(points[0])
            polyline = model.create_entity("IfcPolyline", Points=points)
            profile = model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline)
            position = model.create_entity(
                "IfcAxis2Placement3D",
                Location=_make_point(0.0, 0.0, 0.0),
                Axis=_make_direction(0.0, 0.0, 1.0),
                RefDirection=_make_direction(1.0, 0.0, 0.0),
            )

        if profile is None:
            return

        if position is None:
            position = model.create_entity(
                "IfcAxis2Placement3D",
                Location=_make_point(0.0, 0.0, 0.0),
                Axis=_make_direction(0.0, 0.0, 1.0),
                RefDirection=_make_direction(1.0, 0.0, 0.0),
            )

        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=position,
            ExtrudedDirection=_make_direction(0.0, 0.0, 1.0),
            Depth=height_mm,
        )
        representation = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[representation])
        wall_entity.Representation = product_shape
        _ensure_product_placement(wall_entity)

    def _fallback_wall_axis(det: NormalizedDet, source_index: int) -> WallAxis | None:
        poly = _resolve_polygon(det)
        if poly is None or poly.is_empty:
            return None
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
        if len(coords) < 4:
            return None
        edges: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
        for idx in range(4):
            a = coords[idx]
            b = coords[(idx + 1) % 4]
            length = float(math.hypot(a[0] - b[0], a[1] - b[1]))
            edges.append((a, b, length))
        longest = max(edges, key=lambda item: item[2])
        shortest = min(edges, key=lambda item: item[2])
        length_mm = longest[2]
        width_mm = max(shortest[2], 40.0)
        if length_mm <= 1e-3:
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
        return WallAxis(
            detection=det,
            source_index=source_index,
            axis=axis,
            width_mm=width_mm,
            length_mm=float(axis.length),
            centroid_mm=(float(cx), float(cy)),
            method="fallback",
            metadata={"strategy": 2.0, "source_index": float(source_index)},
        )

    walls_list = [nd for nd in normalized if nd.type == "WALL"]
    wall_polygon_sequence: List[Polygon | MultiPolygon | None] = []
    for idx, _wall_det in enumerate(walls_list):
        if wall_polygons is not None and idx in wall_polygons:
            wall_polygon_sequence.append(wall_polygons[idx])
        else:
            wall_polygon_sequence.append(None)
    axes_by_source: Dict[int, List[WallAxis]] = defaultdict(list)
    if wall_axes:
        for global_index, axis_info in enumerate(wall_axes):
            axis_info.metadata["axis_global_index"] = float(global_index)
            axes_by_source[axis_info.source_index].append(axis_info)

    wall_export_items: List[Dict[str, object]] = []
    walls_by_source: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    axis_counter = 0

    for source_index, nd in enumerate(walls_list):
        axes_for_det = axes_by_source.get(source_index, [])
        if not axes_for_det:
            fallback_axis = _fallback_wall_axis(nd, source_index)
            if fallback_axis:
                axes_for_det = [fallback_axis]
                axes_by_source[source_index] = axes_for_det
        axes_for_det = sorted(axes_for_det, key=lambda info: info.length_mm, reverse=True)
        axis_count = len(axes_for_det)
        if axis_count == 0:
            continue

        source_polygon = wall_polygon_sequence[source_index] if source_index < len(wall_polygon_sequence) else None

        for local_idx, axis_info in enumerate(axes_for_det):
            axis_counter += 1
            wall = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcWallStandardCase",
                name=f"Wall_{axis_counter}",
            )
            wall.PredefinedType = "STANDARD"
            ifcopenshell.api.run("spatial.assign_container", model, products=[wall], relating_structure=storey)
            pset_common = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Pset_WallCommon")
            is_ext = bool(nd.is_external) if nd.is_external is not None else False

            original_width = float(axis_info.width_mm) if axis_info.width_mm is not None else None
            snapped_width = snap_thickness_mm(
                original_width,
                is_external=is_ext,
                standards=thickness_standards,
            )
            if original_width is not None and math.isfinite(original_width):
                axis_info.metadata["width_raw_mm"] = float(original_width)
            axis_info.metadata["width_snapped_mm"] = float(snapped_width)
            axis_info.width_mm = snapped_width

            try:
                wall.ObjectType = "ExternalWall" if is_ext else "InternalWall"
            except Exception:
                pass
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties={"IsExternal": is_ext})

            props = {
                "HeightMm": height_mm,
                "WidthMm": snapped_width,
                "AxisLengthMm": float(axis_info.axis.length),
                "AxisMethod": axis_info.method,
            }
            if original_width is not None and math.isfinite(original_width):
                props["WidthMmRaw"] = float(original_width)
            if px_per_mm:
                props["ScalePxPerMm"] = float(px_per_mm)
            pset_bimify = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Bimify_WallParams")
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset_bimify, properties=props)

            # Source geometry metrics for metadata
            source_geom = None
            if isinstance(source_polygon, Polygon):
                source_geom = source_polygon
            elif isinstance(source_polygon, MultiPolygon):
                try:
                    source_geom = max(list(source_polygon.geoms), key=lambda g: g.area)
                except ValueError:
                    source_geom = None
            if source_geom is None:
                source_geom = _resolve_polygon(nd)

            area_mm2 = float(source_geom.area) if source_geom is not None else 0.0
            if source_geom is not None:
                minx, miny, maxx, maxy = source_geom.bounds
            else:
                minx = miny = maxx = maxy = 0.0
            bbox_width = maxx - minx
            bbox_height = maxy - miny
            confidence = 0.0
            if isinstance(nd.attrs, dict) and nd.attrs.get("confidence") is not None:
                try:
                    confidence = float(nd.attrs.get("confidence"))
                except (TypeError, ValueError):
                    confidence = 0.0

            props_source = {
                "SourceIndex": float(axis_info.source_index),
                "AxisLocalIndex": float(local_idx),
                "AxisCountForSource": float(axis_count),
                "Confidence": confidence,
                "PolygonAreaMm2": area_mm2,
                "BBoxMinX": float(minx),
                "BBoxMinY": float(miny),
                "BBoxWidthMm": float(bbox_width),
                "BBoxHeightMm": float(bbox_height),
                "AxisLengthMm": float(axis_info.axis.length),
                "AxisWidthMm": snapped_width,
                "AxisWidthSnappedMm": float(snapped_width),
                "AxisMethod": axis_info.method,
            }
            if original_width is not None and math.isfinite(original_width):
                props_source["AxisWidthMmRaw"] = float(original_width)
            for meta_key, meta_val in axis_info.metadata.items():
                try:
                    numeric_val = float(meta_val)
                except (TypeError, ValueError):
                    continue
                prop_name = f"Meta_{meta_key}".replace(" ", "_").replace("-", "_")
                props_source[prop_name] = numeric_val

            pset_source = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Bimify_SourceRoboflow")
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset_source, properties=props_source)

            pset_validation = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Bimify_Validation")
            ifcopenshell.api.run(
                "pset.edit_pset",
                model,
                pset=pset_validation,
                properties={"Status": "NotValidated"},
            )

            axis_info.metadata["wall_index"] = float(axis_counter)
            axis_info.metadata["axis_local_index"] = float(local_idx)
            axis_info.metadata["axis_count"] = float(axis_count)

            _assign_wall_geometry(axis_info, wall, source_polygon)

            export_item = {
                "axis": axis_info,
                "detection": nd,
                "wall": wall,
                "axis_count": axis_count,
                "local_index": local_idx,
            }
            wall_export_items.append(export_item)
            walls_by_source[source_index].append(export_item)

    effective_door_height = float(door_height_mm or 2100.0)
    effective_window_head = float(window_head_elevation_mm or 2000.0)
    effective_window_height = float(window_height_mm or 1000.0)
    if effective_window_height >= effective_window_head:
        effective_window_height = max(effective_window_head - 100.0, 100.0)
    window_sill_mm = max(effective_window_head - effective_window_height, 0.0)

    opening_fill_items: List[tuple[NormalizedDet, object]] = []
    fill_psets: dict[object, tuple[object, object]] = {}

    for nd in normalized:
        if nd.type == "DOOR":
            door = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcDoor", name="Door")
            ifcopenshell.api.run("spatial.assign_container", model, products=[door], relating_structure=storey)
            pset_common = ifcopenshell.api.run("pset.add_pset", model, product=door, name="Pset_DoorCommon")
            bimify = ifcopenshell.api.run("pset.add_pset", model, product=door, name="Bimify_DoorParams")
            fill_psets[door] = (pset_common, bimify)
            opening_fill_items.append((nd, door))
        elif nd.type == "WINDOW":
            win = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWindow", name="Window")
            ifcopenshell.api.run("spatial.assign_container", model, products=[win], relating_structure=storey)
            pset_common = ifcopenshell.api.run("pset.add_pset", model, product=win, name="Pset_WindowCommon")
            bimify = ifcopenshell.api.run("pset.add_pset", model, product=win, name="Bimify_WindowParams")
            fill_psets[win] = (pset_common, bimify)
            opening_fill_items.append((nd, win))

    def _compute_opening_placement(
        opening_det: NormalizedDet,
        axis: LineString | None,
        default_width: float,
        wall_thickness: float,
    ) -> OpeningPlacement:
        geom = opening_det.geom
        coords: List[tuple[float, float]] = []
        if isinstance(geom, Polygon):
            coords = [(float(x), float(y)) for x, y in list(geom.exterior.coords)]
        elif isinstance(geom, MultiPolygon):
            try:
                largest = max(list(geom.geoms), key=lambda g: g.area)
            except ValueError:
                largest = None
            if largest is not None:
                coords = [(float(x), float(y)) for x, y in list(largest.exterior.coords)]
        elif isinstance(geom, LineString):
            coords = [(float(x), float(y)) for x, y in list(geom.coords)]

        default_center = (float(geom.centroid.x), float(geom.centroid.y)) if geom is not None else (0.0, 0.0)
        width = float(default_width)
        depth = float(wall_thickness)
        axis_vec = (1.0, 0.0)

        if axis is not None and axis.length > 1e-3 and coords:
            ax_coords = list(axis.coords)
            origin_x, origin_y = float(ax_coords[0][0]), float(ax_coords[0][1])
            end_x, end_y = float(ax_coords[-1][0]), float(ax_coords[-1][1])
            dx = end_x - origin_x
            dy = end_y - origin_y
            length = math.hypot(dx, dy)
            if length > 1e-6:
                ux = dx / length
                uy = dy / length
                # Snap axis to global orthogonal to avoid slight skews (viewer-friendly)
                angle = (math.degrees(math.atan2(uy, ux)) + 360.0) % 180.0
                if abs(angle - 90.0) < 45.0 and abs(angle - 90.0) <= abs(angle - 0.0):
                    axis_vec = (0.0, 1.0)
                else:
                    axis_vec = (1.0, 0.0)
                px = -uy
                py = ux

                axis_projections: List[float] = []
                perp_projections: List[float] = []
                for x, y in coords:
                    vx = x - origin_x
                    vy = y - origin_y
                    axis_projections.append(vx * ux + vy * uy)
                    perp_projections.append(vx * px + vy * py)

                # Compute center from centroid projected onto axis (reduces offsets)
                try:
                    c = opening_det.geom.centroid
                    cx, cy = float(c.x), float(c.y)
                except Exception:
                    cx = float(sum(x for x, _ in coords) / len(coords))
                    cy = float(sum(y for _, y in coords) / len(coords))
                vcx = cx - origin_x
                vcy = cy - origin_y
                proj = vcx * ux + vcy * uy
                # clamp to axis extent
                proj = max(0.0, min(length, proj))
                default_center = (origin_x + ux * proj, origin_y + uy * proj)

                if axis_projections:
                    min_proj = min(axis_projections)
                    max_proj = max(axis_projections)
                    span = max_proj - min_proj
                    if span > 1e-3:
                        width = float(span)

                if perp_projections:
                    perp_min = min(perp_projections)
                    perp_max = max(perp_projections)
                    perp_span = perp_max - perp_min
                    if perp_span > 1e-3:
                        depth = float(min(wall_thickness, max(perp_span, min(wall_thickness, 80.0))))
        elif coords:
            # Fallback: orient by longest edge (with orthogonal snap tolerance)
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            if xs and ys:
                width = max(max(xs) - min(xs), max(ys) - min(ys), default_width * 0.5)
                default_center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
                # determine direction
                best_len_sq = -1.0
                best_vec = (1.0, 0.0)
                for i in range(len(coords) - 1):
                    vx = float(coords[i + 1][0] - coords[i][0])
                    vy = float(coords[i + 1][1] - coords[i][1])
                    cand = vx * vx + vy * vy
                    if cand > best_len_sq:
                        best_len_sq = cand
                        best_vec = (vx, vy)
                ang = math.degrees(math.atan2(best_vec[1], best_vec[0])) % 180.0
                # snap to 0/90 if close
                if min(ang, abs(180.0 - ang)) <= 10.0:
                    axis_vec = (1.0, 0.0)
                elif abs(ang - 90.0) <= 10.0:
                    axis_vec = (0.0, 1.0)
                else:
                    norm = math.hypot(best_vec[0], best_vec[1]) or 1.0
                    axis_vec = (best_vec[0] / norm, best_vec[1] / norm)
                depth = min(wall_thickness, max(width * 0.4, min(wall_thickness, 80.0)))

        width = max(width, default_width * 0.5)
        depth = max(depth, min(wall_thickness, 40.0))
        # Persist axis-aligned rectangle into detection for downstream consumers
        oriented_poly = _planar_rectangle_polygon(default_center, width, min(depth, wall_thickness), axis_vec)
        if oriented_poly is not None and not oriented_poly.is_empty:
            try:
                opening_det.geom = oriented_poly
                if isinstance(opening_det.attrs, dict):
                    opening_det.attrs["geometry_source"] = "axis_aligned"
            except Exception:
                pass

        return OpeningPlacement(width_mm=width, center_xy=default_center, axis_vec=axis_vec, depth_mm=depth)

    def _assign_rectangular_representation(product, width: float, depth: float, height: float, center_xy: tuple[float, float], base_height: float, axis_vec: tuple[float, float, float]) -> None:
        width = float(width)
        depth = float(depth)
        height = float(height)
        if width <= 1e-3 or depth <= 1e-3 or height <= 1e-3:
            return
        axis_len = math.hypot(axis_vec[0], axis_vec[1])
        if axis_len <= 1e-6:
            ref_dir = (1.0, 0.0, 0.0)
        else:
            ref_dir = (axis_vec[0] / axis_len, axis_vec[1] / axis_len, 0.0)
        profile = model.create_entity(
            "IfcRectangleProfileDef",
            ProfileType="AREA",
            XDim=width,
            YDim=depth,
        )
        placement = model.create_entity(
            "IfcAxis2Placement3D",
            Location=_make_point(center_xy[0], center_xy[1], base_height),
            Axis=_make_direction(0.0, 0.0, 1.0),
            RefDirection=_make_direction(ref_dir[0], ref_dir[1], ref_dir[2]),
        )
        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=placement,
            ExtrudedDirection=_make_direction(0.0, 0.0, 1.0),
            Depth=height,
        )
        representation = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[representation])
        product.Representation = product_shape
        _ensure_product_placement(product)

    default_door_width = 900.0
    default_window_width = 1200.0

    # Assign openings to walls (RelVoids/Fills + geometry)
    axis_catalog: List[WallAxis] = []
    effective_wall_polygons: List[Polygon] = []
    assignments: List[OpeningAssignment] = []
    if opening_fill_items:
        try:
            axis_catalog = [item["axis"] for item in wall_export_items if isinstance(item.get("axis"), WallAxis)]
            for idx, wall_det in enumerate(walls_list):
                override_geom = wall_polygon_sequence[idx] if idx < len(wall_polygon_sequence) else None
                if isinstance(override_geom, Polygon):
                    effective_wall_polygons.append(override_geom)
                    continue
                if isinstance(override_geom, MultiPolygon):
                    try:
                        effective_wall_polygons.append(max(list(override_geom.geoms), key=lambda g: g.area))
                        continue
                    except ValueError:
                        pass
                fallback_poly = _resolve_polygon(wall_det)
                effective_wall_polygons.append(fallback_poly if fallback_poly is not None else Polygon())

            assignments, _ = snap_openings_to_walls(
                normalized,
                wall_axes=axis_catalog,
                wall_polygons_override=effective_wall_polygons,
            )
        except Exception as exc:
            logger.warning("Zuordnung der Öffnungen zu Wänden fehlgeschlagen: %s", exc)
            logger.debug("opening assignment stacktrace", exc_info=True)
            assignments = []
    else:
        effective_wall_polygons = []

    def _select_wall_item_for_opening(assignment: OpeningAssignment | None, opening_geom) -> dict | None:
        if not wall_export_items:
            return None
        if assignment is None or assignment.wall_index is None:
            return wall_export_items[0]
        source_index = assignment.wall_index
        candidates = walls_by_source.get(source_index, [])
        if not candidates:
            return wall_export_items[0]
        if len(candidates) == 1 or opening_geom is None or getattr(opening_geom, "is_empty", False):
            return candidates[0]
        axis_idx = getattr(assignment, "axis_index", None)
        if axis_idx is not None:
            for item in candidates:
                axis_info = item["axis"]
                if getattr(axis_info, "source_index", None) == source_index:
                    existing_index = getattr(axis_info, "metadata", {}).get("axis_global_index")
                    if existing_index is not None and int(existing_index) == int(axis_idx):
                        return item
        try:
            centroid = opening_geom.centroid
        except Exception:
            centroid = None
        if centroid is None or getattr(centroid, "is_empty", False):
            return candidates[0]
        return min(candidates, key=lambda item: item["axis"].axis.distance(centroid))

    # First pass: compute fitted placements and collect for bias calibration
    fitted: list[tuple[NormalizedDet, object, dict, tuple[float, float], tuple[float, float], float, LineString | None, float]] = []
    for idx, (opening_det, fill_entity) in enumerate(opening_fill_items):
        assignment = assignments[idx] if idx < len(assignments) else None
        try:
            selected_item = _select_wall_item_for_opening(assignment, opening_det.geom)
            wall_det = selected_item["detection"] if selected_item else None
            wall_entity = selected_item["wall"] if selected_item else None
            axis_info = selected_item["axis"] if selected_item else None
            axis_line = axis_info.axis if axis_info else None
            if axis_info and axis_info.width_mm is not None and axis_info.width_mm > 1e-3:
                wall_thickness = _snap_wall_thickness(float(axis_info.width_mm), is_external=wall_det.is_external if wall_det else None, standards=thickness_standards)
            else:
                is_external = wall_det.is_external if wall_det is not None else None
                fallback_target = 240.0 if bool(is_external) else 115.0
                wall_thickness = _snap_wall_thickness(fallback_target, is_external=is_external, standards=thickness_standards)

            if axis_line is not None:
                placement, rect_poly, metrics = fit_opening_to_axis(opening_det, axis_line, wall_thickness)
                try:
                    opening_det.geom = rect_poly
                    if isinstance(opening_det.attrs, dict):
                        opening_det.attrs["geometry_source"] = "axis_fit"
                        opening_det.attrs["iou"] = float(metrics.get("iou", 0.0))
                except Exception:
                    pass
            else:
                default_width = default_door_width if opening_det.type == "DOOR" else default_window_width
                placement = _compute_opening_placement(opening_det, None, default_width, wall_thickness)

            target_height = effective_door_height if opening_det.type == "DOOR" else effective_window_height
            fitted.append((opening_det, fill_entity, {"placement": placement, "wall_entity": wall_entity, "wall_det": wall_det}, placement.center_xy, placement.axis_vec, wall_thickness, axis_line, target_height))
        except Exception as exc:
            opening_type = getattr(opening_det, "type", "unknown")
            logger.warning("Opening-Vorbereitung fehlgeschlagen (Index %d, Typ %s): %s", idx + 1, opening_type, exc)
            logger.debug("opening preparation stacktrace", exc_info=True)
            continue

    # Global per-orientation bias calibration
    def _orientation_key(vec: tuple[float, float]) -> str:
        return "V" if abs(vec[0]) < 1e-6 and abs(vec[1]) > 0 else "H"

    offsets: dict[str, list[float]] = {"H": [], "V": []}
    for opening_det, _fill, meta, center_xy, axis_vec_2d, _thick, axis_line, _h in fitted:
        if axis_line is None:
            continue
        ax = list(axis_line.coords)
        ox, oy = float(ax[0][0]), float(ax[0][1])
        ux, uy = axis_vec_2d
        try:
            c = _largest_polygon(opening_det.geom).centroid if isinstance(opening_det.geom, Polygon) else opening_det.geom.centroid
        except Exception:
            c = None
        if c is None:
            continue
        cx_rf = (float(c.x) - ox) * ux + (float(c.y) - oy) * uy
        cx_fit = (center_xy[0] - ox) * ux + (center_xy[1] - oy) * uy
        offsets[_orientation_key(axis_vec_2d)].append(cx_fit - cx_rf)

    biases: dict[str, float] = {}
    for key, values in offsets.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 1:
            biases[key] = float(sorted_vals[mid])
        else:
            biases[key] = float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)

    # Second pass: apply bias, build geometry and IFC
    for idx, (opening_det, fill_entity, meta, center_xy, axis_vec_2d, wall_thickness, axis_line, target_height) in enumerate(fitted):
        try:
            if axis_line is not None:
                bias = biases.get(_orientation_key(axis_vec_2d), 0.0)
                center_xy = (
                    center_xy[0] - axis_vec_2d[0] * bias,
                    center_xy[1] - axis_vec_2d[1] * bias,
                )

            base_height = 0.0 if opening_det.type == "DOOR" else window_sill_mm
            placement_data = meta.get("placement")
            if placement_data is None:
                raise ValueError("placement data missing for opening")
            width_mm = placement_data.width_mm
            depth_mm = placement_data.depth_mm
            axis_vec = (axis_vec_2d[0], axis_vec_2d[1], 0.0)

            oriented_poly = _planar_rectangle_polygon(center_xy, width_mm, min(depth_mm, wall_thickness), axis_vec_2d)
            if oriented_poly is not None and not oriented_poly.is_empty:
                try:
                    opening_det.geom = oriented_poly
                    if isinstance(opening_det.attrs, dict):
                        opening_det.attrs["geometry_source"] = "axis_fit_bias"
                except Exception:
                    pass

            opening = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcOpeningElement", name=f"Opening_{idx+1}")
            ifcopenshell.api.run("spatial.assign_container", model, products=[opening], relating_structure=storey)

            _assign_rectangular_representation(opening, width_mm, depth_mm, target_height, center_xy, base_height, axis_vec)

            fill_thickness = min(wall_thickness, max(wall_thickness * 0.7, 40.0))
            _assign_rectangular_representation(fill_entity, width_mm * 0.98, fill_thickness, target_height, center_xy, base_height, axis_vec)

            wall_entity = meta.get("wall_entity")
            if wall_entity is not None:
                try:
                    ifcopenshell.api.run("void.add_opening", model, element=wall_entity, opening=opening)
                except Exception:
                    pass
            try:
                ifcopenshell.api.run("opening.add_filling", model, opening=opening, filling=fill_entity)
            except Exception:
                pass

            has_void_rel = any(rel.is_a("IfcRelVoidsElement") for rel in (getattr(opening, "VoidsElements", []) or []))
            has_fill_rel = any(rel.is_a("IfcRelFillsElement") for rel in (getattr(opening, "HasFillings", []) or []))
            if not has_void_rel or not has_fill_rel:
                wall_guid = getattr(wall_entity, "GlobalId", None) if wall_entity is not None else None
                logger.warning(
                    "Opening %s (%s) missing relations (void=%s, fill=%s) on wall %s",
                    getattr(opening, "GlobalId", None),
                    opening_det.type,
                    has_void_rel,
                    has_fill_rel,
                    wall_guid,
                )

            opening_pset = ifcopenshell.api.run("pset.add_pset", model, product=opening, name="Bimify_OpeningParams")
            ifcopenshell.api.run(
                "pset.edit_pset",
                model,
                pset=opening_pset,
                properties={
                    "WidthMm": float(width_mm),
                    "HeightMm": float(target_height),
                    "BaseHeightMm": float(base_height),
                    "WallThicknessMm": float(wall_thickness),
                },
            )

            common_pset, bimify_pset = fill_psets.get(fill_entity, (None, None))
            if common_pset:
                common_props = {"OverallHeight": float(target_height), "OverallWidth": float(width_mm)}
                if opening_det.type == "WINDOW":
                    common_props["SillHeight"] = float(window_sill_mm)
                    common_props["HeadHeight"] = float(effective_window_head)
                ifcopenshell.api.run("pset.edit_pset", model, pset=common_pset, properties=common_props)
            if bimify_pset:
                bimify_props = {
                    "WidthMm": float(width_mm),
                    "HeightMm": float(target_height),
                    "BaseHeightMm": float(base_height),
                    "WallThicknessMm": float(wall_thickness),
                }
                if opening_det.type == "WINDOW":
                    bimify_props["HeadHeightMm"] = float(effective_window_head)
                    bimify_props["SillHeightMm"] = float(window_sill_mm)
                else:
                    bimify_props["HeadHeightMm"] = float(target_height)
                    bimify_props["SillHeightMm"] = 0.0
                ifcopenshell.api.run("pset.edit_pset", model, pset=bimify_pset, properties=bimify_props)
        except Exception as exc:
            opening_type = getattr(opening_det, "type", "unknown")
            logger.warning("Opening-Erstellung fehlgeschlagen (Index %d, Typ %s): %s", idx + 1, opening_type, exc)
            logger.debug("opening finalization stacktrace", exc_info=True)
            continue

    # Stairs
    for i, nd in enumerate([x for x in normalized if x.type == "STAIR"]):
        stair = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcStair", name=f"Stair_{i+1}")
        ifcopenshell.api.run("spatial.assign_container", model, products=[stair], relating_structure=storey)
        ifcopenshell.api.run("pset.add_pset", model, product=stair, name="Pset_StairCommon")

    # Spaces
    for i, sp in enumerate(spaces):
        space = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSpace", name=f"Space_{i+1}")
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[space])
        _ensure_product_placement(space)
        pset = ifcopenshell.api.run("pset.add_pset", model, product=space, name="Pset_SpaceCommon")
        try:
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={"Area": sp.area_m2})
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_path))
    return out_path


def _prepare_thickness_standards(standards: Sequence[float] | None) -> Tuple[float, ...]:
    values: Iterable[float] = standards if standards is not None else DEFAULT_WALL_THICKNESS_STANDARDS_MM
    filtered = sorted({float(abs(s)) for s in values if isinstance(s, (int, float)) and abs(float(s)) > 0.0})
    return tuple(filtered) if filtered else DEFAULT_WALL_THICKNESS_STANDARDS_MM


def _snap_wall_thickness(value: float, *, is_external: bool | None, standards: Sequence[float]) -> float:
    if not standards:
        return max(value, 1.0)
    reference = float(value) if value and value > 0 else (240.0 if is_external else 115.0)
    snapped = min(standards, key=lambda candidate: abs(candidate - reference))
    return float(snapped)


def _planar_rectangle_polygon(
    center: Tuple[float, float],
    width: float,
    depth: float,
    axis_vec: Tuple[float, float],
) -> Polygon | None:
    ux, uy = axis_vec
    length = math.hypot(ux, uy)
    if length <= 1e-6:
        return None
    ux /= length
    uy /= length
    px = -uy
    py = ux
    half_w = max(width / 2.0, 1.0)
    half_d = max(depth / 2.0, 1.0)

    cx, cy = center
    corners = [
        (cx - ux * half_w - px * half_d, cy - uy * half_w - py * half_d),
        (cx + ux * half_w - px * half_d, cy + uy * half_w - py * half_d),
        (cx + ux * half_w + px * half_d, cy + uy * half_w + py * half_d),
        (cx - ux * half_w + px * half_d, cy - uy * half_w + py * half_d),
    ]
    return Polygon(corners)

def _largest_polygon(geom: Polygon | MultiPolygon | LineString | None) -> Polygon | None:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            return None
    return None

def _iou(a: Polygon | None, b: Polygon | None) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return float(inter / union) if union > 1e-6 else 0.0

def fit_opening_to_axis(
    opening_det: NormalizedDet,
    axis: LineString,
    wall_thickness: float,
) -> tuple[OpeningPlacement, Polygon, dict]:
    rf_poly = _largest_polygon(opening_det.geom)
    coords = []
    if isinstance(rf_poly, Polygon):
        coords = [(float(x), float(y)) for x, y in list(rf_poly.exterior.coords)]
    if not coords:
        # Fallback: use current placement
        placement = _compute_opening_placement(opening_det, axis, 900.0 if opening_det.type == "DOOR" else 1200.0, wall_thickness)
        rect = _planar_rectangle_polygon(placement.center_xy, placement.width_mm, min(placement.depth_mm, wall_thickness), placement.axis_vec)
        return placement, rect if rect is not None else Polygon(), {"iou": 0.0}

    # Axis basis
    ax = list(axis.coords)
    ox, oy = float(ax[0][0]), float(ax[0][1])
    ex, ey = float(ax[-1][0]), float(ax[-1][1])
    dx = ex - ox
    dy = ey - oy
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        length = 1.0
    ux = dx / length
    uy = dy / length
    # Snap to 0/90 deg
    ang = (math.degrees(math.atan2(uy, ux)) + 360.0) % 180.0
    if abs(ang - 90.0) < 45.0 and abs(ang - 90.0) <= abs(ang - 0.0):
        base = (0.0, 1.0)
    else:
        base = (1.0, 0.0)
    ux, uy = base
    px, py = -uy, ux

    # Projections of RF polygon to derive spans
    along_vals = []
    perp_vals = []
    for x, y in coords:
        vx = x - ox
        vy = y - oy
        along_vals.append(vx * ux + vy * uy)
        perp_vals.append(vx * px + vy * py)
    min_a, max_a = min(along_vals), max(along_vals)
    min_p, max_p = min(perp_vals), max(perp_vals)
    span = max(1.0, max_a - min_a)
    perp_span = max(1.0, max_p - min_p)
    width = float(span)
    depth = float(min(wall_thickness, max(perp_span, min(wall_thickness, 80.0))))

    # Candidate centers
    mid_center = (ox + ux * ((min_a + max_a) / 2.0), oy + uy * ((min_a + max_a) / 2.0))
    try:
        c = rf_poly.centroid
        cproj = ((float(c.x) - ox) * ux + (float(c.y) - oy) * uy)
        centroid_center = (ox + ux * cproj, oy + uy * cproj)
    except Exception:
        centroid_center = mid_center

    # Slide search ±150mm in 15 steps
    search_radius = 150.0
    steps = 15
    step = (2.0 * search_radius) / steps
    candidates = []
    for base_center in (mid_center, centroid_center):
        for i in range(steps + 1):
            delta = -search_radius + i * step
            cx = base_center[0] + ux * delta
            cy = base_center[1] + uy * delta
            rect = _planar_rectangle_polygon((cx, cy), width, depth, (ux, uy))
            candidates.append(((cx, cy), rect))

    best_iou = -1.0
    best_center = mid_center
    best_rect = None
    for (cx, cy), rect in candidates:
        iou = _iou(rect, rf_poly)
        if iou > best_iou:
            best_iou = iou
            best_center = (cx, cy)
            best_rect = rect

    # Fallback if the IoU is too low or geometry invalid
    if best_rect is None or best_iou < 0.05 or width <= 1e-3 or depth <= 1e-3:
        default_width = 900.0 if opening_det.type == "DOOR" else 1200.0
        fallback = _compute_opening_placement(opening_det, axis, default_width, wall_thickness)
        rect = _planar_rectangle_polygon(fallback.center_xy, max(fallback.width_mm, 100.0), min(max(fallback.depth_mm, 40.0), wall_thickness), fallback.axis_vec)
        return fallback, (rect if rect is not None else Polygon()), {"iou": 0.0, "fallback": True}

    placement = OpeningPlacement(width_mm=width, center_xy=best_center, axis_vec=(ux, uy), depth_mm=depth)
    metrics = {"iou": float(max(0.0, min(1.0, best_iou))) , "span_mm": width, "perp_span_mm": perp_span}
    return placement, (best_rect if best_rect is not None else Polygon()), metrics


