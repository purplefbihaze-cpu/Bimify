"""Opening (door/window) building functions for IFC models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import ifcopenshell.api
from shapely.geometry import LineString

from core.ml.postprocess_floorplan import NormalizedDet
from core.ifc.geometry_utils import compute_opening_placement, fit_opening_to_axis


logger = logging.getLogger(__name__)


def create_door_types(
    model: Any,
    project: Any,
    *,
    is_ifc2x3: bool = False,
) -> Tuple[Any, Any]:
    """Create door type/style entities.
    
    Args:
        model: IFC model instance.
        project: IfcProject entity.
        is_ifc2x3: Whether using IFC2X3 schema.
    
    Returns:
        Tuple of (door_type, door_style).
    """
    door_type = None
    door_style = None
    
    try:
        if is_ifc2x3:
            # IFC2X3 uses IfcDoorStyle
            door_style = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcDoorStyle",
                name="StandardDoorStyle",
            )
            try:
                door_style.OperationType = "SINGLE_SWING_LEFT"
            except Exception:
                pass
        else:
            # IFC4 uses IfcDoorType
            door_type = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcDoorType",
                name="StandardDoorType",
            )
            _safe_set_predefined_type(door_type, "DOOR", is_ifc2x3)
            ifcopenshell.api.run("type.assign_type", model, related_object=project, relating_type=door_type)
    except Exception as exc:
        logger.warning("Could not create Door Type/Style: %s", exc)
    
    return door_type, door_style


def create_window_types(
    model: Any,
    project: Any,
    *,
    is_ifc2x3: bool = False,
) -> Tuple[Any, Any]:
    """Create window type/style entities.
    
    Args:
        model: IFC model instance.
        project: IfcProject entity.
        is_ifc2x3: Whether using IFC2X3 schema.
    
    Returns:
        Tuple of (window_type, window_style).
    """
    window_type = None
    window_style = None
    
    try:
        if is_ifc2x3:
            # IFC2X3 uses IfcWindowStyle
            window_style = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcWindowStyle",
                name="StandardWindowStyle",
            )
            try:
                window_style.ConstructionType = "SINGLE_PANEL"
            except Exception:
                pass
        else:
            # IFC4 uses IfcWindowType
            window_type = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcWindowType",
                name="StandardWindowType",
            )
            _safe_set_predefined_type(window_type, "WINDOW", is_ifc2x3)
            ifcopenshell.api.run("type.assign_type", model, related_object=project, relating_type=window_type)
    except Exception as exc:
        logger.warning("Could not create Window Type/Style: %s", exc)
    
    return window_type, window_style


def create_opening_fills(
    model: Any,
    storey: Any,
    normalized: List[NormalizedDet],
    *,
    door_type: Any = None,
    door_style: Any = None,
    window_type: Any = None,
    window_style: Any = None,
    is_ifc2x3: bool = False,
) -> List[Tuple[NormalizedDet, Any]]:
    """Create opening fill entities (doors and windows) from detections.
    
    Args:
        model: IFC model instance.
        storey: IfcBuildingStorey entity.
        normalized: List of normalized detections.
        door_type: IfcDoorType entity (IFC4).
        door_style: IfcDoorStyle entity (IFC2X3).
        window_type: IfcWindowType entity (IFC4).
        window_style: IfcWindowStyle entity (IFC2X3).
        is_ifc2x3: Whether using IFC2X3 schema.
    
    Returns:
        List of tuples (detection, fill_entity).
    """
    opening_fill_items: List[Tuple[NormalizedDet, Any]] = []
    
    for nd in normalized:
        if nd.type == "DOOR":
            door = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcDoor", name="Door")
            _safe_set_predefined_type(door, "DOOR", is_ifc2x3)
            ifcopenshell.api.run("spatial.assign_container", model, products=[door], relating_structure=storey)
            
            # Assign door type/style (schema-aware)
            if is_ifc2x3 and door_style is not None:
                try:
                    ifcopenshell.api.run("style.assign_style", model, products=[door], style=door_style)
                except Exception:
                    pass
            elif door_type is not None:
                try:
                    ifcopenshell.api.run("type.assign_type", model, related_object=door, relating_type=door_type)
                except Exception:
                    pass
            
            opening_fill_items.append((nd, door))
        elif nd.type == "WINDOW":
            win = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWindow", name="Window")
            _safe_set_predefined_type(win, "WINDOW", is_ifc2x3)
            ifcopenshell.api.run("spatial.assign_container", model, products=[win], relating_structure=storey)
            
            # Assign window type/style (schema-aware)
            if is_ifc2x3 and window_style is not None:
                try:
                    ifcopenshell.api.run("style.assign_style", model, products=[win], style=window_style)
                except Exception:
                    pass
            elif window_type is not None:
                try:
                    ifcopenshell.api.run("type.assign_type", model, related_object=win, relating_type=window_type)
                except Exception:
                    pass
            
            opening_fill_items.append((nd, win))
    
    return opening_fill_items


def assign_opening_to_wall(
    model: Any,
    opening_fill: Any,
    wall: Any,
    placement: Any,
    *,
    height_mm: float,
    wall_thickness: float,
) -> None:
    """Assign an opening to a wall with proper placement.
    
    Args:
        model: IFC model instance.
        opening_fill: IfcDoor or IfcWindow entity.
        wall: IfcWallStandardCase entity.
        placement: OpeningPlacement with position and dimensions.
        height_mm: Opening height in millimeters.
        wall_thickness: Wall thickness in millimeters.
    """
    try:
        # Create opening element
        opening = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcOpeningElement",
            name="Opening",
        )
        
        # Assign opening to wall
        ifcopenshell.api.run("void.add_opening", model, opening=opening, element=wall)
        
        # Assign fill to opening
        ifcopenshell.api.run("void.add_filling", model, opening=opening, element=opening_fill)
        
        # Set opening properties
        props = {
            "WidthMm": placement.width_mm,
            "HeightMm": height_mm,
            "DepthMm": min(placement.depth_mm, wall_thickness),
        }
        pset = ifcopenshell.api.run("pset.add_pset", model, product=opening, name="Bimify_OpeningParams")
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=props)
    except Exception as exc:
        logger.warning("Failed to assign opening to wall: %s", exc)


def _safe_set_predefined_type(element: Any, predefined_type_value: str, is_ifc2x3: bool) -> None:
    """Schema-safe PredefinedType setting with IFC2X3 compatibility."""
    if is_ifc2x3:
        entity_type = element.is_a() if hasattr(element, "is_a") else None
        if entity_type in ("IfcSpace", "IfcSlab", "IfcCovering"):
            try:
                element.PredefinedType = predefined_type_value
            except Exception:
                pass
        else:
            try:
                element.PredefinedType = predefined_type_value
            except Exception as exc:
                logger.debug("Failed to set PredefinedType for %s in IFC2X3: %s", entity_type, exc)
    else:
        try:
            element.PredefinedType = predefined_type_value
        except Exception as exc:
            logger.debug("Failed to set PredefinedType: %s", exc)


__all__ = [
    "create_door_types",
    "create_window_types",
    "create_opening_fills",
    "assign_opening_to_wall",
]

