"""Auto-healing simulation helpers for HottCAD compliance.

Generates hypothetical topology repairs (wall connections, space
boundaries, material layers) based on geometric proximity and
existing IFC data. The results can be displayed in the WexBIM
viewer to guide manual or automated corrections before export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import isclose
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.placement as placement_util
import numpy as np
from shapely.geometry import Polygon

Coordinate2D = Tuple[float, float]


def _open_model(path_or_model: Any) -> ifcopenshell.file:
    if isinstance(path_or_model, ifcopenshell.file):
        return path_or_model
    if isinstance(path_or_model, (str, Path)):
        return ifcopenshell.open(str(path_or_model))
    raise TypeError("Expected path or ifcopenshell.file")


@dataclass
class WallConnectionProposal:
    walls: Tuple[str, str]
    distance_mm: float
    contact_type: str  # touch | gap | overlap
    notes: List[str] = field(default_factory=list)


@dataclass
class MaterialSuggestion:
    wall: str
    thickness_mm: Optional[float]
    note: Optional[str] = None


@dataclass
class HighlightSet:
    id: str
    label: str
    guids: List[str]
    product_ids: List[int] = field(default_factory=list)


@dataclass
class AutoReconnectSimulation:
    proposed: Dict[str, List[Any]]
    completeness: Dict[str, Any]
    highlight_sets: List[HighlightSet]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed": {
                key: [
                    {
                        **({"walls": list(item.walls), "distanceMm": item.distance_mm, "contactType": item.contact_type, "notes": item.notes}
                           if isinstance(item, WallConnectionProposal)
                           else {"wall": item.wall, "thicknessMm": item.thickness_mm, "note": item.note})
                    }
                    for item in value
                ]
                for key, value in self.proposed.items()
            },
            "completeness": self.completeness,
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


def simulate_hottcad(path_or_model: Any, *, tolerance_mm: float = 0.5) -> AutoReconnectSimulation:
    model = _open_model(path_or_model)

    walls = list(model.by_type("IfcWall"))
    spaces = list(model.by_type("IfcSpace"))

    wall_data = _collect_wall_geometry(walls)
    guid_to_product_id = _map_guid_to_product_id(model)
    wall_space_map = _map_wall_spaces(model, walls)

    connection_props = _propose_wall_connections(wall_data, tolerance_mm)
    material_props = _suggest_material_layers(wall_data)

    gaps = [prop for prop in connection_props if prop.contact_type == "gap"]
    rooms_closed = bool(spaces) and not gaps

    highlight_sets = _build_highlights(connection_props, gaps, guid_to_product_id)

    return AutoReconnectSimulation(
        proposed={
            "connects": connection_props,
            "spaceBoundaries": _propose_space_boundaries(connection_props, wall_space_map),
            "materials": material_props,
        },
        completeness={
            "roomsClosed": rooms_closed,
            "gapCount": len(gaps),
            "spaces": len(spaces),
            "walls": len(walls),
        },
        highlight_sets=highlight_sets,
    )


def _collect_wall_geometry(walls: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for wall in walls:
        footprint = _wall_footprint_polygon(wall)
        thickness = _wall_thickness_from_pset(wall)
        if footprint is None:
            continue
        data[wall.GlobalId] = {
            "entity": wall,
            "polygon": footprint,
            "thickness_mm": thickness or _estimate_polygon_thickness(footprint),
        }
    return data


def _wall_thickness_from_pset(wall: Any) -> Optional[float]:
    psets = ifcopenshell.util.element.get_psets(wall)
    for pset_name in ("Bimify_WallParams", "Pset_WallCommon"):
        props = psets.get(pset_name)
        if props:
            for key in ("WidthMm", "Thickness", "ThicknessMm"):
                if key in props:
                    value = props[key]
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
    return None


def _wall_footprint_polygon(wall: Any) -> Optional[Polygon]:
    placement_matrix = placement_util.get_local_placement(getattr(wall, "ObjectPlacement", None))

    for item in _iterate_body_items(wall):
        if not item.is_a("IfcExtrudedAreaSolid"):
            continue

        swept = getattr(item, "SweptArea", None)
        if swept is None:
            continue

        position = getattr(item, "Position", None)
        local_matrix = placement_util.get_axis2placement(position) if position else np.eye(4)
        transform = placement_matrix @ local_matrix

        if swept.is_a("IfcRectangleProfileDef"):
            xdim = float(getattr(swept, "XDim", 0.0))
            ydim = float(getattr(swept, "YDim", 0.0))
            if xdim <= 0.0 or ydim <= 0.0:
                continue
            hx = xdim / 2.0
            hy = ydim / 2.0
            coords = [
                (-hx, -hy, 0.0),
                (hx, -hy, 0.0),
                (hx, hy, 0.0),
                (-hx, hy, 0.0),
                (-hx, -hy, 0.0),
            ]
            points = _transform_points(coords, transform)
            return Polygon([(x, y) for x, y, _ in points])

        if swept.is_a("IfcArbitraryClosedProfileDef"):
            curve = getattr(swept, "OuterCurve", None)
            if curve and curve.is_a("IfcPolyline") and curve.Points:
                local_pts = [
                    (float(pt.Coordinates[0]), float(pt.Coordinates[1]), 0.0)
                    for pt in curve.Points
                ]
                points = _transform_points(local_pts, transform)
                poly = Polygon([(x, y) for x, y, _ in points])
                if poly.is_valid and not poly.is_empty:
                    return poly

    return None


def _transform_points(points: Iterable[Tuple[float, float, float]], matrix: np.ndarray) -> List[Tuple[float, float, float]]:
    arr = np.array([[x, y, z, 1.0] for x, y, z in points], dtype=float)
    transformed = arr @ matrix.T
    return [(row[0], row[1], row[2]) for row in transformed]


def _iterate_body_items(product: Any) -> Iterable[Any]:
    shape = getattr(product, "Representation", None)
    if not shape:
        return []
    reps = getattr(shape, "Representations", []) or []
    for rep in reps:
        identifier = (rep.RepresentationIdentifier or "").lower()
        if identifier and identifier not in {"body", "solidmodel", "model"}:
            continue
        for item in rep.Items or []:
            yield item


def _estimate_polygon_thickness(polygon: Polygon) -> Optional[float]:
    if polygon.is_empty:
        return None
    min_rect = polygon.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    if len(coords) < 5:
        return None
    lengths = []
    for idx in range(4):
        x1, y1 = coords[idx]
        x2, y2 = coords[idx + 1]
        lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if not lengths:
        return None
    distinct = sorted(set(round(length, 6) for length in lengths))
    if len(distinct) < 2:
        return None
    return min(distinct)


def _propose_wall_connections(wall_data: Dict[str, Dict[str, Any]], tolerance_mm: float) -> List[WallConnectionProposal]:
    proposals: List[WallConnectionProposal] = []
    for gid_a, gid_b in combinations(wall_data, 2):
        poly_a = wall_data[gid_a]["polygon"]
        poly_b = wall_data[gid_b]["polygon"]

        if not poly_a.is_valid or not poly_b.is_valid:
            continue

        if poly_a.intersects(poly_b):
            proposals.append(
                WallConnectionProposal(
                    walls=(gid_a, gid_b),
                    distance_mm=0.0,
                    contact_type="touch",
                    notes=["Wände überlappen oder berühren sich bereits."],
                )
            )
            continue

        distance = poly_a.distance(poly_b)
        distance_mm = float(distance)

        if distance_mm <= tolerance_mm:
            proposals.append(
                WallConnectionProposal(
                    walls=(gid_a, gid_b),
                    distance_mm=distance_mm,
                    contact_type="gap" if distance_mm > 0 else "touch",
                    notes=["Wände liegen innerhalb des Toleranzbereichs – RelConnectsElements empfohlen."],
                )
            )
    return proposals


def _suggest_material_layers(wall_data: Dict[str, Dict[str, Any]]) -> List[MaterialSuggestion]:
    suggestions: List[MaterialSuggestion] = []
    for gid, data in wall_data.items():
        thickness = data.get("thickness_mm")
        if thickness is None:
            suggestions.append(MaterialSuggestion(wall=gid, thickness_mm=None, note="Wandstärke unbekannt – MaterialLayerSetUsage erforderlich."))
        else:
            suggestions.append(MaterialSuggestion(wall=gid, thickness_mm=float(thickness)))
    return suggestions


def _map_guid_to_product_id(model: ifcopenshell.file) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        for product in model.by_type("IfcProduct"):
            guid = getattr(product, "GlobalId", None)
            if guid:
                try:
                    mapping[str(guid)] = int(product.id())
                except Exception:
                    continue
    except Exception:
        return mapping
    return mapping


def _map_wall_spaces(model: ifcopenshell.file, walls: Sequence[Any]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {wall.GlobalId: [] for wall in walls}
    boundaries = model.by_type("IfcRelSpaceBoundary")
    if boundaries:
        for rel in boundaries:
            element = getattr(rel, "RelatedBuildingElement", None)
            space = getattr(rel, "RelatingSpace", None)
            if element and space and element.GlobalId in mapping:
                mapping[element.GlobalId].append(space.GlobalId)
        return mapping

    for wall in walls:
        psets = ifcopenshell.util.element.get_psets(wall)
        spaces = psets.get("Bimify_WallParams", {}).get("AdjacentSpaces")
        if spaces:
            if isinstance(spaces, str):
                mapping[wall.GlobalId].append(spaces)
            elif isinstance(spaces, (list, tuple)):
                mapping[wall.GlobalId].extend(str(sp) for sp in spaces)
    return mapping


def _propose_space_boundaries(
    connections: List[WallConnectionProposal],
    wall_space_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for proposal in connections:
        walls = proposal.walls
        spaces_a = set(wall_space_map.get(walls[0], []))
        spaces_b = set(wall_space_map.get(walls[1], []))
        if not spaces_a or not spaces_b:
            continue
        shared = spaces_a.intersection(spaces_b)
        if shared:
            proposals.append(
                {
                    "walls": list(walls),
                    "spaces": list(shared),
                    "note": "Gemeinsame Räume gefunden – IfcRelSpaceBoundary1stLevel erzeugen.",
                }
            )
    return proposals


def _build_highlights(
    connections: List[WallConnectionProposal],
    gaps: List[WallConnectionProposal],
    guid_map: Dict[str, int],
) -> List[HighlightSet]:
    highlights: List[HighlightSet] = []
    if connections:
        guids = list({gid for prop in connections for gid in prop.walls})
        highlights.append(
            HighlightSet(
                id="walls-all",
                label="Erkannte Wandkontakte",
                guids=guids,
                product_ids=_product_ids_from_guids(guids, guid_map),
            )
        )
    if gaps:
        guids = list({gid for prop in gaps for gid in prop.walls})
        highlights.append(
            HighlightSet(
                id="walls-gaps",
                label="Wände mit Spalt < 0.5mm",
                guids=guids,
                product_ids=_product_ids_from_guids(guids, guid_map),
            )
        )
    return highlights


def _product_ids_from_guids(guids: Iterable[str], guid_map: Dict[str, int]) -> List[int]:
    ids: List[int] = []
    for guid in guids:
        product_id = guid_map.get(guid)
        if product_id is not None:
            ids.append(product_id)
    return ids


__all__ = [
    "AutoReconnectSimulation",
    "MaterialSuggestion",
    "WallConnectionProposal",
    "HighlightSet",
    "simulate_hottcad",
]


