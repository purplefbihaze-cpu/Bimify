"""HottCAD compatibility validation utilities.

This module inspects IFC4x3 models and evaluates them against
the HottCAD import requirements. It aggregates detailed check
results and computes a health score that can be surfaced in the
API and UI.

Key responsibilities:
  * Parse IFC products (walls, floors, spaces, openings) and their
    geometric representations.
  * Evaluate wall geometry (rectangular footprint, constant
    thickness, vertical extrusion).
  * Ensure relational completeness (RelVoidsElement, Connects,
    SpaceBoundaries, MaterialLayerSetUsage).
  * Compute proximity metrics (wall adjacency, open room loops)
    using configurable tolerances.

The output dataclasses are designed to be serialised easily with
Pydantic in the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, hypot
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import ifcopenshell
import ifcopenshell.util.element
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

Coordinate2D = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]


def _to_path(path_or_model: Any) -> Optional[Path]:
    if isinstance(path_or_model, (str, Path)):
        return Path(path_or_model)
    return None


def _open_model(path_or_model: Any) -> ifcopenshell.file:
    if isinstance(path_or_model, ifcopenshell.file):
        return path_or_model
    if isinstance(path_or_model, Path):
        return ifcopenshell.open(path_or_model.as_posix())
    if isinstance(path_or_model, str):
        return ifcopenshell.open(path_or_model)
    raise TypeError("Expected path or ifcopenshell.file")


@dataclass
class HottCADCheck:
    id: str
    title: str
    status: str  # ok | warn | fail
    details: List[str] = field(default_factory=list)
    affected: Dict[str, List[str]] = field(default_factory=dict)

    def add_detail(self, message: str) -> None:
        self.details.append(message)

    def add_affected(self, key: str, values: Iterable[str]) -> None:
        if key not in self.affected:
            self.affected[key] = []
        self.affected[key].extend(str(v) for v in values)

    def downgrade(self, status: str) -> None:
        order = {"ok": 0, "warn": 1, "fail": 2}
        if order[status] > order[self.status]:
            self.status = status


@dataclass
class HottCADMetrics:
    wall_count: int = 0
    interior_walls: int = 0
    exterior_walls: int = 0
    walls_with_rectangular_footprint: int = 0
    walls_with_constant_thickness: int = 0
    openings_with_relations: int = 0
    spaces: int = 0
    floors: int = 0
    roofs: int = 0
    connects_relations: int = 0
    space_boundaries: int = 0
    material_layer_usages: int = 0
    avg_wall_thickness_mm: Optional[float] = None


@dataclass
class HottCADValidationResult:
    schema: str
    file_info: Dict[str, Any]
    checks: List[HottCADCheck]
    metrics: HottCADMetrics
    score: int
    highlight_sets: List["HighlightSet"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "file_info": self.file_info,
            "checks": [
                {
                    "id": chk.id,
                    "title": chk.title,
                    "status": chk.status,
                    "details": chk.details,
                    "affected": chk.affected,
                }
                for chk in self.checks
            ],
            "metrics": self.metrics.__dict__,
            "score": self.score,
            "highlightSets": [
                {
                    "id": hs.id,
                    "label": hs.label,
                    "guids": hs.guids,
                    "productIds": hs.product_ids,
                }
                for hs in self.highlight_sets
            ],
        }


@dataclass
class HighlightSet:
    id: str
    label: str
    guids: List[str]
    product_ids: List[int] = field(default_factory=list)


def validate_hottcad(path_or_model: Any, *, tolerance_mm: float = 0.5) -> HottCADValidationResult:
    model = _open_model(path_or_model)
    src_path = _to_path(path_or_model)

    file_info = {
        "schema": model.schema,
        "path": src_path.as_posix() if src_path else None,
        "sizeBytes": src_path.stat().st_size if src_path and src_path.exists() else None,
        "isPlainIFC": _is_plain_ifc(src_path),
    }

    metrics = HottCADMetrics()
    checks: List[HottCADCheck] = [
        HottCADCheck("file-format", "Dateiformat", "ok"),
        HottCADCheck("walls-geometry", "Wandgeometrie", "ok"),
        HottCADCheck("walls-classification", "Wandklassifikation Innen/Außen", "warn"),
        HottCADCheck("openings", "Öffnungen & Relationen", "ok"),
        HottCADCheck("floors-roofs", "Böden & Dächer", "warn"),
        HottCADCheck("relations", "Topologische Relationen", "fail"),
        HottCADCheck("thermal", "U-Wert Angaben", "warn"),
        HottCADCheck("topology", "Raumtopologie & Spaltmaß", "warn"),
        HottCADCheck("project-structure", "Gebäude- und Projektstruktur", "ok"),
    ]

    check_map = {chk.id: chk for chk in checks}

    _validate_file_context(model, file_info, check_map["file-format"])
    wall_ctx = _validate_walls(model, metrics, tolerance_mm, check_map["walls-geometry"], check_map["walls-classification"])
    _validate_openings(model, metrics, check_map["openings"], tolerance_mm)
    _validate_floors_roofs(model, metrics, check_map["floors-roofs"], tolerance_mm)
    _validate_relations(model, metrics, check_map["relations"])
    _validate_u_values(model, check_map["thermal"])
    _validate_topology(model, metrics, wall_ctx, tolerance_mm, check_map["topology"])
    _validate_project_structure(model, check_map["project-structure"])

    score = _compute_score(metrics, checks)
    highlight_sets = _build_highlights(model, checks)

    return HottCADValidationResult(
        schema=model.schema,
        file_info=file_info,
        checks=checks,
        metrics=metrics,
        score=score,
        highlight_sets=highlight_sets,
    )


def _is_plain_ifc(path: Optional[Path]) -> Optional[bool]:
    if not path or not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            start = fh.read(64)
        return b"ISO-10303-21" in start
    except OSError:
        return None


def _has_coordination_view(model: ifcopenshell.file) -> bool:
    try:
        contexts = model.by_type("IfcGeometricRepresentationContext")
    except Exception:
        return False

    model_contexts = [ctx for ctx in contexts if (getattr(ctx, "ContextType", "") or "").strip().lower() == "model"]
    if not model_contexts:
        return False

    has_body_context = any((getattr(ctx, "ContextIdentifier", "") or "").strip().lower() == "body" for ctx in model_contexts)

    try:
        subcontexts = model.by_type("IfcGeometricRepresentationSubcontext")
    except Exception:
        subcontexts = []

    has_model_view = False
    for sub in subcontexts:
        target_view = (getattr(sub, "TargetView", "") or "").strip().upper()
        if target_view != "MODEL_VIEW":
            continue
        parent = getattr(sub, "ParentContext", None)
        parent_identifier = (getattr(parent, "ContextIdentifier", "") or "").strip().lower()
        identifier = (getattr(sub, "ContextIdentifier", "") or "").strip().lower()
        if identifier == "body" or parent_identifier == "body":
            has_model_view = True
            break

    return has_body_context and has_model_view


def _validate_file_context(model: ifcopenshell.file, file_info: Dict[str, Any], check: HottCADCheck) -> None:
    size = file_info.get("sizeBytes")
    if size and size > 50 * 1024 * 1024:
        check.downgrade("warn")
        check.add_detail("Dateigröße > 50 MB (HottCAD kann Probleme bekommen).")

    is_plain = file_info.get("isPlainIFC")
    if is_plain is False:
        check.downgrade("fail")
        check.add_detail("IFC liegt nicht im Plain-SPF-Format vor.")

    schema_name = (model.schema or "").strip().upper()
    if schema_name != "IFC4":
        check.downgrade("fail")
        check.add_detail(f"Schema muss IFC4 (Coordination View 2.0) sein – gefunden: {schema_name or 'unbekannt'}.")

    if not _has_coordination_view(model):
        check.downgrade("fail")
        check.add_detail("Koordinationssicht 2.0 nicht nachweisbar (fehlender Model View / Body-Kontext).")


def _validate_walls(
    model: ifcopenshell.file,
    metrics: HottCADMetrics,
    tolerance_mm: float,
    geom_check: HottCADCheck,
    class_check: HottCADCheck,
) -> Dict[str, Any]:
    walls = list(model.by_type("IfcWall"))
    metrics.wall_count = len(walls)

    if not walls:
        geom_check.downgrade("fail")
        geom_check.add_detail("Keine IfcWall-Elemente gefunden.")
        return {"wall_geometries": {}, "wall_spaces": {}}

    thickness_values: List[float] = []
    rectangular_count = 0
    constant_thickness = 0
    interior = 0
    exterior = 0
    wall_geometries: Dict[str, Dict[str, Any]] = {}

    for wall in walls:
        label = _element_label(wall)
        analysis = _analyse_wall_geometry(wall, tolerance_mm)
        wall_geometries[wall.GlobalId] = analysis

        if not analysis["has_body_representation"]:
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: keine Body-Repräsentation gefunden.")
            geom_check.add_affected("walls_missing_geometry", [wall.GlobalId])
            continue

        if not analysis["is_extruded"]:
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: keine IfcExtrudedAreaSolid-Geometrie (CV 2.0 verletzt).")
            geom_check.add_affected("walls_non_extruded", [wall.GlobalId])

        if analysis["has_triangulation"] and not analysis["is_extruded"]:
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: nur triangulierte Geometrie vorhanden – HottCAD benötigt gerades Prisma.")
            geom_check.add_affected("walls_triangulated", [wall.GlobalId])

        if analysis["footprint_rectangular"]:
            rectangular_count += 1
        else:
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: Grundfläche ist nicht rechteckig (4 Punkte).")
            geom_check.add_affected("walls_non_rectangular", [wall.GlobalId])

        thickness = analysis.get("thickness_mm")
        if thickness is not None:
            thickness_values.append(thickness)
        if analysis.get("constant_thickness", False):
            constant_thickness += 1
        else:
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: Wandstärke ist nicht konsistent oder nicht ermittelbar.")
            geom_check.add_affected("walls_inconsistent_thickness", [wall.GlobalId])

        if not analysis.get("extrusion_vertical", True):
            geom_check.downgrade("fail")
            geom_check.add_detail(f"{label}: Extrusionsrichtung ist nicht vertikal.")
            geom_check.add_affected("walls_non_horizontal_base", [wall.GlobalId])

        is_external = _is_wall_external(wall)
        if is_external is True:
            exterior += 1
        elif is_external is False:
            interior += 1

    metrics.walls_with_rectangular_footprint = rectangular_count
    metrics.walls_with_constant_thickness = constant_thickness
    metrics.avg_wall_thickness_mm = sum(thickness_values) / len(thickness_values) if thickness_values else None
    metrics.exterior_walls = exterior
    metrics.interior_walls = interior

    if interior == 0 or exterior == 0:
        class_check.downgrade("warn")
        class_check.add_detail("Innen-/Außenwandklassifikation unvollständig. Prüfe Pset_WallCommon.IsExternal oder Räume.")
    else:
        class_check.status = "ok"

    wall_spaces = _map_wall_spaces(model, walls)
    if wall_spaces.get("missing_space_links"):
        class_check.downgrade("warn")
        class_check.add_detail(
            "Nicht alle Wände sind Räumen gegenübergestellt – automatische Klassifikation könnte scheitern."
        )
        class_check.add_affected("walls_missing_spaces", wall_spaces["missing_space_links"])

    return {"wall_geometries": wall_geometries, "wall_spaces": wall_spaces}


def _element_label(entity: Any) -> str:
    name = getattr(entity, "Name", None) or getattr(entity, "GlobalId", "?")
    return f"{entity.is_a()}({name})"


def _iterate_body_items(product: Any) -> Iterator[Any]:
    shape = getattr(product, "Representation", None)
    if not shape:
        return
    reps = getattr(shape, "Representations", []) or []
    for rep in reps:
        identifier = (rep.RepresentationIdentifier or "").lower()
        if identifier and identifier not in {"body", "solidmodel", "model"}:
            continue
        for item in rep.Items or []:
            yield item


def _analyse_wall_geometry(wall: Any, tolerance_mm: float) -> Dict[str, Any]:
    result = {
        "has_body_representation": False,
        "is_extruded": False,
        "footprint_rectangular": False,
        "constant_thickness": False,
        "thickness_mm": None,
        "extrusion_vertical": True,
        "height_mm": None,
        "has_triangulation": False,
    }

    items = list(_iterate_body_items(wall))
    if not items:
        return result

    result["has_body_representation"] = True

    for item in items:
        if item.is_a("IfcTriangulatedFaceSet"):
            result["has_triangulation"] = True
        if item.is_a("IfcExtrudedAreaSolid"):
            result["is_extruded"] = True
            thickness, rectangular = _analyse_swept_area(item, tolerance_mm)
            if thickness is not None:
                result["thickness_mm"] = thickness
                result["constant_thickness"] = True
            result["footprint_rectangular"] = rectangular or result["footprint_rectangular"]
            result["height_mm"] = float(item.Depth)

            dir_ratios = getattr(getattr(item, "ExtrudedDirection", None), "DirectionRatios", None)
            if dir_ratios:
                result["extrusion_vertical"] = _is_vertical_direction(dir_ratios)

    return result


def _analyse_swept_area(item: Any, tolerance_mm: float) -> Tuple[Optional[float], bool]:
    swept = getattr(item, "SweptArea", None)
    if swept is None:
        return None, False

    if swept.is_a("IfcRectangleProfileDef"):
        xdim = float(getattr(swept, "XDim", 0.0))
        ydim = float(getattr(swept, "YDim", 0.0))
        thickness = min(xdim, ydim)
        return thickness, True

    if swept.is_a("IfcArbitraryClosedProfileDef"):
        curve = getattr(swept, "OuterCurve", None)
        if curve and curve.is_a("IfcPolyline"):
            points = [
                tuple(float(coord) for coord in pt.Coordinates[:2])  # type: ignore[attr-defined]
                for pt in curve.Points or []
            ]
            thickness, rectangular = _analyse_polyline(points, tolerance_mm)
            return thickness, rectangular

    return None, False


def _analyse_polyline(points: List[Coordinate2D], tolerance_mm: float) -> Tuple[Optional[float], bool]:
    if len(points) < 4:
        return None, False
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) != 4:
        return None, False

    vectors = []
    lengths = []
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        dx = x2 - x1
        dy = y2 - y1
        vectors.append((dx, dy))
        lengths.append(hypot(dx, dy))

    if any(length <= tolerance_mm * 0.1 for length in lengths):
        return None, False

    # Check orthogonality
    for i in range(len(vectors)):
        vx1, vy1 = vectors[i]
        vx2, vy2 = vectors[(i + 1) % len(vectors)]
        dot = vx1 * vx2 + vy1 * vy2
        if not isclose(dot, 0.0, abs_tol=tolerance_mm * 0.5):
            return None, False

    # Determine two distinct edge lengths
    distinct: List[float] = []
    for length in lengths:
        if not any(isclose(length, existing, rel_tol=1e-3, abs_tol=tolerance_mm) for existing in distinct):
            distinct.append(length)
    if len(distinct) != 2:
        return None, False

    thickness = min(distinct)
    return thickness, True


def _profile_to_polygon(profile: Any) -> Optional[Polygon]:
    if profile is None:
        return None

    if profile.is_a("IfcRectangleProfileDef"):
        xdim = float(getattr(profile, "XDim", 0.0))
        ydim = float(getattr(profile, "YDim", 0.0))
        if xdim <= 0.0 or ydim <= 0.0:
            return None
        hx = xdim / 2.0
        hy = ydim / 2.0
        return Polygon([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])

    if profile.is_a("IfcArbitraryClosedProfileDef"):
        curve = getattr(profile, "OuterCurve", None)
        if curve and curve.is_a("IfcPolyline") and getattr(curve, "Points", None):
            points: List[Tuple[float, float]] = []
            for pt in curve.Points:
                coords = getattr(pt, "Coordinates", None)
                if not coords or len(coords) < 2:
                    return None
                points.append((float(coords[0]), float(coords[1])))
            if len(points) < 3:
                return None
            if points[0] == points[-1]:
                points = points[:-1]
            return Polygon(points)

    return None


def _analyse_prismatic_product(product: Any, tolerance_mm: float) -> Dict[str, Any]:
    result = {
        "has_body": False,
        "has_extruded": False,
        "vertical_extrusion": True,
        "polygon_valid": False,
        "polygon": None,
        "has_triangulation": False,
    }

    items = list(_iterate_body_items(product))
    if not items:
        return result

    result["has_body"] = True

    for item in items:
        if item.is_a("IfcTriangulatedFaceSet"):
            result["has_triangulation"] = True
        if not item.is_a("IfcExtrudedAreaSolid"):
            continue

        result["has_extruded"] = True
        profile = getattr(item, "SweptArea", None)
        polygon = _profile_to_polygon(profile)
        if polygon and polygon.is_valid and not polygon.is_empty and polygon.area > max(1e-6, (tolerance_mm or 0.0) ** 2 * 0.01):
            result["polygon"] = polygon
            result["polygon_valid"] = True
        else:
            result["polygon"] = polygon
            result["polygon_valid"] = False

        dir_ratios = getattr(getattr(item, "ExtrudedDirection", None), "DirectionRatios", None)
        if dir_ratios:
            result["vertical_extrusion"] = _is_vertical_direction(dir_ratios)

    return result


def _is_vertical_direction(direction: Sequence[float]) -> bool:
    if len(direction) < 3:
        return False
    dx, dy, dz = (float(direction[0]), float(direction[1]), float(direction[2]))
    return isclose(dx, 0.0, abs_tol=1e-6) and isclose(dy, 0.0, abs_tol=1e-6) and dz > 0.0


def _is_wall_external(wall: Any) -> Optional[bool]:
    direct = getattr(wall, "IsExternal", None)
    if direct is not None:
        return bool(direct)

    psets = ifcopenshell.util.element.get_psets(wall)
    for pset_name in ("Pset_WallCommon", "Bimify_WallParams"):
        props = psets.get(pset_name)
        if props and "IsExternal" in props:
            value = props["IsExternal"]
            if isinstance(value, bool):
                return value

    return None


def _map_wall_spaces(model: ifcopenshell.file, walls: Sequence[Any]) -> Dict[str, Any]:
    # Without explicit space boundaries we approximate by checking Pset assignments.
    space_refs: Dict[str, List[str]] = {wall.GlobalId: [] for wall in walls}
    missing: List[str] = []

    rel_space_boundaries = model.by_type("IfcRelSpaceBoundary")
    if rel_space_boundaries:
        for rel in rel_space_boundaries:
            element = getattr(rel, "RelatedBuildingElement", None)
            space = getattr(rel, "RelatingSpace", None)
            if element and space and element.GlobalId in space_refs:
                space_refs[element.GlobalId].append(space.GlobalId)
    else:
        for wall in walls:
            psets = ifcopenshell.util.element.get_psets(wall)
            spaces = psets.get("Bimify_WallParams", {}).get("AdjacentSpaces")
            if spaces:
                if isinstance(spaces, str):
                    space_refs[wall.GlobalId].append(spaces)
                elif isinstance(spaces, (list, tuple)):
                    space_refs[wall.GlobalId].extend(str(sp) for sp in spaces)

    for wall in walls:
        if not space_refs.get(wall.GlobalId):
            missing.append(wall.GlobalId)

    return {"mapping": space_refs, "missing_space_links": missing}


def _validate_openings(
    model: ifcopenshell.file,
    metrics: HottCADMetrics,
    check: HottCADCheck,
    tolerance_mm: float,
) -> None:
    openings = list(model.by_type("IfcOpeningElement"))
    metrics.openings_with_relations = 0
    if not openings:
        check.downgrade("warn")
        check.add_detail("Keine IfcOpeningElement-Entitäten gefunden.")
        return

    missing_rel: List[str] = []
    invalid_geometry: List[str] = []
    triangulated: List[str] = []
    non_vertical: List[str] = []
    no_body: List[str] = []
    for opening in openings:
        has_void = False
        has_filling = False

        for rel in getattr(opening, "VoidsElements", []) or []:
            if rel.is_a("IfcRelVoidsElement"):
                has_void = True
                break

        for rel in getattr(opening, "HasFillings", []) or []:
            if rel.is_a("IfcRelFillsElement"):
                has_filling = True
                break

        if has_void and has_filling:
            metrics.openings_with_relations += 1
        else:
            missing_rel.append(opening.GlobalId)

        analysis = _analyse_prismatic_product(opening, tolerance_mm)
        label = _element_label(opening)
        if not analysis["has_body"]:
            no_body.append(opening.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: keine Body-Repräsentation vorhanden.")
            continue
        if analysis["has_triangulation"] and not analysis["has_extruded"]:
            triangulated.append(opening.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: nur triangulierter Körper – erwarte gerades Prisma.")
        if not analysis["has_extruded"]:
            invalid_geometry.append(opening.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: keine IfcExtrudedAreaSolid-Geometrie.")
        if analysis["has_extruded"] and not analysis["vertical_extrusion"]:
            non_vertical.append(opening.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: Extrusionsrichtung ist nicht vertikal.")
        if analysis["has_extruded"] and not analysis["polygon_valid"]:
            invalid_geometry.append(opening.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: Öffnungsgrundfläche ist kein gültiges Polygon.")

    if missing_rel:
        check.downgrade("warn")
        check.add_detail("Nicht alle Öffnungen besitzen RelVoidsElement & RelFillsElement.")
        check.add_affected("openings_missing_relations", missing_rel)
    if no_body:
        check.add_affected("openings_missing_geometry", no_body)
    if triangulated:
        check.add_affected("openings_triangulated", triangulated)
    if invalid_geometry:
        check.add_affected("openings_invalid_geometry", list(dict.fromkeys(invalid_geometry)))
    if non_vertical:
        check.add_affected("openings_non_vertical", non_vertical)


def _validate_floors_roofs(
    model: ifcopenshell.file,
    metrics: HottCADMetrics,
    check: HottCADCheck,
    tolerance_mm: float,
) -> None:
    floors = model.by_type("IfcSlab")
    roofs = model.by_type("IfcRoof")
    metrics.floors = len(floors)
    metrics.roofs = len(roofs)

    if not floors:
        check.downgrade("warn")
        check.add_detail("Keine IfcSlab-Elemente gefunden (Böden).")
    if not roofs:
        check.add_detail("Keine IfcRoof-Elemente gefunden – falls kein Dach vorhanden, kann ignoriert werden.")

    for slab in floors:
        label = _element_label(slab)
        analysis = _analyse_prismatic_product(slab, tolerance_mm)
        if not analysis["has_body"]:
            check.downgrade("fail")
            check.add_detail(f"{label}: keine Body-Repräsentation vorhanden.")
            check.add_affected("slabs_missing_geometry", [slab.GlobalId])
            continue
        if analysis["has_triangulation"] and not analysis["has_extruded"]:
            check.downgrade("fail")
            check.add_detail(f"{label}: nur triangulierte Geometrie – erwarte gerades Prisma.")
            check.add_affected("slabs_triangulated", [slab.GlobalId])
        if not analysis["has_extruded"]:
            check.downgrade("fail")
            check.add_detail(f"{label}: keine IfcExtrudedAreaSolid-Geometrie.")
            check.add_affected("slabs_non_extruded", [slab.GlobalId])
        if not analysis["vertical_extrusion"]:
            check.downgrade("fail")
            check.add_detail(f"{label}: Extrusionsrichtung ist nicht vertikal.")
            check.add_affected("slabs_non_vertical", [slab.GlobalId])
        if not analysis["polygon_valid"]:
            check.downgrade("fail")
            check.add_detail(f"{label}: Grundfläche ist kein gültiges Polygon.")
            check.add_affected("slabs_invalid_polygon", [slab.GlobalId])


def _validate_relations(model: ifcopenshell.file, metrics: HottCADMetrics, check: HottCADCheck) -> None:
    connects = model.by_type("IfcRelConnectsElements") + model.by_type("IfcRelConnectsPathElements")
    space_boundaries = model.by_type("IfcRelSpaceBoundary")
    material_layer_usage = model.by_type("IfcMaterialLayerSetUsage")

    metrics.connects_relations = len(connects)
    metrics.space_boundaries = len(space_boundaries)
    metrics.material_layer_usages = len(material_layer_usage)

    if not connects:
        check.add_detail("Es existieren keine IfcRelConnects(Elements/PathElements).")
    if not space_boundaries:
        check.add_detail("Es existieren keine IfcRelSpaceBoundary-Relationen.")
    if not material_layer_usage:
        check.add_detail("Es existieren keine IfcMaterialLayerSetUsage-Zuweisungen.")

    if connects or space_boundaries or material_layer_usage:
        check.status = "warn"
    else:
        check.status = "fail"


def _validate_u_values(model: ifcopenshell.file, check: HottCADCheck) -> None:
    targets = {
        "IfcWall": "walls",
        "IfcColumn": "columns",
        "IfcRoof": "roofs",
        "IfcSlab": "slabs",
        "IfcWindow": "windows",
        "IfcDoor": "doors",
    }
    candidate_keys = [
        "ThermalTransmittance",
        "UValue",
        "U-Wert",
        "U_Wert",
        "ThermTransmittance",
    ]

    for ifc_type, category in targets.items():
        missing: List[str] = []
        for element in model.by_type(ifc_type):
            if _element_has_u_value(element, candidate_keys):
                continue
            missing.append(element.GlobalId)
            check.add_detail(f"{_element_label(element)}: kein U-Wert gefunden.")
        if missing:
            check.downgrade("warn")
            check.add_affected(f"{category}_missing_u_value", missing)


def _element_has_u_value(element: Any, candidate_keys: Sequence[str]) -> bool:
    direct = getattr(element, "ThermalTransmittance", None)
    if _is_number(direct):
        return True

    psets = ifcopenshell.util.element.get_psets(element)
    for props in psets.values():
        if not isinstance(props, dict):
            continue
        for key in candidate_keys:
            if key not in props:
                continue
            if _is_number(props[key]):
                return True
    return False


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not (value != value)  # NaN check
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _validate_topology(
    model: ifcopenshell.file,
    metrics: HottCADMetrics,
    wall_ctx: Dict[str, Any],
    tolerance_mm: float,
    check: HottCADCheck,
) -> None:
    spaces = list(model.by_type("IfcSpace"))
    metrics.spaces = len(spaces)
    if not spaces:
        check.downgrade("warn")
        check.add_detail("Keine IfcSpace-Elemente vorhanden – Raumtopologie nicht prüfbar.")
        return

    missing_body: List[str] = []
    triangulated: List[str] = []
    invalid_polygons: List[str] = []
    non_vertical: List[str] = []

    for space in spaces:
        label = _element_label(space)
        analysis = _analyse_prismatic_product(space, tolerance_mm)
        if not analysis["has_body"]:
            missing_body.append(space.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: keine Geometrie vorhanden.")
            continue
        if analysis["has_triangulation"] and not analysis["has_extruded"]:
            triangulated.append(space.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: nur triangulierter Körper – gerades Prisma erforderlich.")
        if not analysis["has_extruded"]:
            invalid_polygons.append(space.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: keine IfcExtrudedAreaSolid-Geometrie.")
        if analysis["has_extruded"] and not analysis["vertical_extrusion"]:
            non_vertical.append(space.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: Extrusionsrichtung ist nicht vertikal.")
        if analysis["has_extruded"] and not analysis["polygon_valid"]:
            invalid_polygons.append(space.GlobalId)
            check.downgrade("fail")
            check.add_detail(f"{label}: Grundfläche ist kein gültiges Polygon.")

    if missing_body:
        check.add_affected("spaces_missing_geometry", missing_body)
    if triangulated:
        check.add_affected("spaces_triangulated", triangulated)
    if invalid_polygons:
        check.add_affected("spaces_invalid_geometry", list(dict.fromkeys(invalid_polygons)))
    if non_vertical:
        check.add_affected("spaces_non_vertical", non_vertical)

    # Simple heuristic: treat rooms as closed if there are >=4 surrounding walls with rectangular footprint.
    walls_with_geometry = [
        gid for gid, info in wall_ctx.get("wall_geometries", {}).items() if info.get("footprint_rectangular")
    ]
    if not walls_with_geometry:
        check.downgrade("warn")
        check.add_detail("Wenige rechteckige Wände – Raumabschlüsse unsicher.")
        return

    # No geometric computation here; flag as warn when missing data.
    check.add_detail(
        "Detaillierte Raumabschlussprüfung erfordert IfcRelSpaceBoundary – aktuell nicht verfügbar."
    )


def _validate_project_structure(model: ifcopenshell.file, check: HottCADCheck) -> None:
    projects = model.by_type("IfcProject")
    sites = model.by_type("IfcSite")
    buildings = model.by_type("IfcBuilding")
    storeys = model.by_type("IfcBuildingStorey")

    if len(projects) != 1:
        check.downgrade("warn")
        check.add_detail(f"Projektanzahl = {len(projects)} (erwartet 1).")
    if len(sites) < 1:
        check.downgrade("warn")
        check.add_detail("Kein IfcSite gefunden.")
    if len(buildings) < 1:
        check.downgrade("warn")
        check.add_detail("Kein IfcBuilding gefunden.")
    if len(buildings) > 1:
        check.add_detail("Mehrere Gebäude vorhanden – UI sollte Auswahl anbieten.")
    if len(storeys) < 1:
        check.downgrade("warn")
        check.add_detail("Keine IfcBuildingStorey-Entität gefunden.")

    missing_elevation: List[str] = []
    elevations: List[Tuple[float, str]] = []
    for storey in storeys:
        raw = getattr(storey, "Elevation", None)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            missing_elevation.append(storey.GlobalId)
            continue
        elevations.append((value, storey.GlobalId))

    if missing_elevation:
        check.downgrade("fail")
        check.add_detail("Gebäudegeschosse ohne Höhenangabe gefunden – Elevation ist Pflicht.")
        check.add_affected("storeys_missing_elevation", missing_elevation)

    if elevations:
        elevations.sort(key=lambda item: item[0])
        splitlevels: List[str] = []
        for idx in range(1, len(elevations)):
            prev_val, prev_gid = elevations[idx - 1]
            current_val, current_gid = elevations[idx]
            if abs(current_val - prev_val) < 200.0:  # 0.2 m Schwelle
                splitlevels.extend([prev_gid, current_gid])
        if splitlevels:
            unique = list(dict.fromkeys(splitlevels))
            check.downgrade("fail")
            check.add_detail("Splitlevel-Geschosse erkannt (Höhendifferenz < 200 mm). Bitte zusammenführen.")
            check.add_affected("storeys_splitlevel", unique)


def _compute_score(metrics: HottCADMetrics, checks: Sequence[HottCADCheck]) -> int:
    score = 100

    # Geometry (30 points)
    if metrics.wall_count == 0 or metrics.walls_with_rectangular_footprint == 0:
        score -= 30
    else:
        rect_ratio = metrics.walls_with_rectangular_footprint / max(metrics.wall_count, 1)
        score -= int((1 - rect_ratio) * 20)
        thickness_ratio = metrics.walls_with_constant_thickness / max(metrics.wall_count, 1)
        score -= int((1 - thickness_ratio) * 10)

    # Topology (30 points)
    if metrics.connects_relations == 0:
        score -= 15
    if metrics.space_boundaries == 0:
        score -= 10
    if metrics.material_layer_usages == 0:
        score -= 5

    # Relations already captured above.

    # Format (10 points)
    file_check = next((chk for chk in checks if chk.id == "file-format"), None)
    if file_check and file_check.status != "ok":
        score -= 10

    # Clamp between 0 and 100
    return max(0, min(100, score))


def _build_highlights(model: ifcopenshell.file, checks: Sequence[HottCADCheck]) -> List[HighlightSet]:
    guid_map = _map_guid_to_product_id(model)
    highlights: List[HighlightSet] = []

    for check in checks:
        if check.status != "fail":
            continue
        guids: List[str] = []
        for values in check.affected.values():
            for value in values:
                if _is_guid_like(value):
                    guids.append(str(value))
        unique_guids = list(dict.fromkeys(guids))
        if not unique_guids:
            continue
        product_ids = [guid_map[guid] for guid in unique_guids if guid in guid_map]
        highlights.append(
            HighlightSet(
                id=f"check-{check.id}",
                label=f"{check.title} (Fehler)",
                guids=unique_guids,
                product_ids=product_ids,
            )
        )

    return highlights


def _map_guid_to_product_id(model: ifcopenshell.file) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        for product in model.by_type("IfcProduct"):
            guid = getattr(product, "GlobalId", None)
            if not guid:
                continue
            try:
                mapping[str(guid)] = int(product.id())
            except Exception:
                continue
    except Exception:
        return mapping
    return mapping


def _is_guid_like(value: Any) -> bool:
    if value is None:
        return False
    text = str(value)
    return len(text) >= 22 and all(ch.isalnum() or ch in {"_", "-"} for ch in text)


__all__ = [
    "HottCADCheck",
    "HottCADMetrics",
    "HottCADValidationResult",
    "HighlightSet",
    "validate_hottcad",
]


