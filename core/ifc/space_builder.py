"""Space building functions for IFC models."""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import ifcopenshell.api
from shapely.geometry import Polygon

from core.reconstruct.spaces import SpacePoly


logger = logging.getLogger(__name__)


def create_spaces(
    model: Any,
    storey: Any,
    spaces: List[SpacePoly],
    *,
    storey_elevation: float = 0.0,
    height_mm: float = 3000.0,
    is_ifc2x3: bool = False,
    body_context: Any = None,
) -> List[Tuple[Any, SpacePoly]]:
    """Create IfcSpace entities from space polygons.
    
    Args:
        model: IFC model instance.
        storey: IfcBuildingStorey entity.
        spaces: List of space polygons.
        storey_elevation: Elevation of the storey in millimeters.
        height_mm: Space height in millimeters.
        is_ifc2x3: Whether using IFC2X3 schema.
        body_context: IfcGeometricRepresentationContext for body representation.
    
    Returns:
        List of tuples (space_entity, space_poly).
    """
    space_entities: List[Tuple[Any, SpacePoly]] = []
    
    for i, sp in enumerate(spaces):
        try:
            space = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSpace", name=f"Space_{i+1}")
            _safe_set_predefined_type(space, "INTERNAL", is_ifc2x3)
            ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[space])
            _ensure_product_placement(model, space, storey)
            
            # Assign 3D geometry to space
            if body_context:
                _assign_space_geometry(model, space, sp.polygon, float(storey_elevation), height_mm, body_context)
            
            pset = _safe_add_pset(model, space, "Pset_SpaceCommon")
            try:
                ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={
                    "Area": sp.area_m2,
                    "IsExternal": False,
                    "LongName": f"Space {i+1}",
                })
            except Exception:
                pass
            space_entities.append((space, sp))
        except Exception as exc:
            logger.warning("Failed to create space %d: %s", i + 1, exc)
    
    return space_entities


def _assign_space_geometry(
    model: Any,
    space: Any,
    polygon: Polygon,
    elevation: float,
    height: float,
    body_context: Any,
) -> None:
    """Assign 3D geometry to a space entity.
    
    Args:
        model: IFC model instance.
        space: IfcSpace entity.
        polygon: 2D polygon representing the space footprint.
        elevation: Base elevation in millimeters.
        height: Space height in millimeters.
        body_context: IfcGeometricRepresentationContext for body representation.
    """
    try:
        # Create extrusion profile from polygon
        coords = list(polygon.exterior.coords)
        if len(coords) < 3:
            return
        
        # Create polyline from polygon coordinates
        points = []
        for x, y in coords[:-1]:  # Exclude last duplicate point
            point = model.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y), float(elevation)))
            points.append(point)
        
        # Create polyline
        polyline = model.create_entity("IfcPolyline", Points=points)
        
        # Create closed profile
        profile = model.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=polyline,
        )
        
        # Create extrusion
        placement = model.create_entity(
            "IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, float(elevation))),
            Axis=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0)),
        )
        
        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=placement,
            ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            Depth=float(height),
        )
        
        # Create representation
        representation = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body_context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        
        product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[representation])
        space.Representation = product_shape
    except Exception as exc:
        logger.warning("Failed to assign geometry to space %s: %s", getattr(space, "Name", "unknown"), exc)


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


def _safe_add_pset(model: Any, product: Any, pset_name: str, fallback_name: str | None = None) -> Any:
    """Schema-safe Property Set creation with fallback support."""
    try:
        pset = ifcopenshell.api.run("pset.add_pset", model, product=product, name=pset_name)
        return pset
    except Exception as exc:
        if fallback_name and fallback_name != pset_name:
            try:
                logger.debug("Property Set '%s' creation failed, trying fallback '%s': %s", pset_name, fallback_name, exc)
                pset = ifcopenshell.api.run("pset.add_pset", model, product=product, name=fallback_name)
                return pset
            except Exception as fallback_exc:
                logger.warning("Both Property Set '%s' and fallback '%s' failed: %s", pset_name, fallback_name, fallback_exc)
        else:
            logger.warning("Property Set '%s' creation failed: %s", pset_name, exc)
        try:
            safe_name = f"Bimify_{pset_name.replace('Pset_', '')}" if pset_name.startswith("Pset_") else f"Bimify_{pset_name}"
            pset = ifcopenshell.api.run("pset.add_pset", model, product=product, name=safe_name)
            logger.debug("Created Property Set with safe name '%s' as fallback for '%s'", safe_name, pset_name)
            return pset
        except Exception as safe_exc:
            logger.error("All Property Set creation attempts failed for '%s': %s", pset_name, safe_exc)
            return None


def _ensure_product_placement(model: Any, product: Any, storey: Any) -> None:
    """Ensure product has proper placement relative to storey."""
    try:
        if getattr(product, "ObjectPlacement", None) is None:
            storey_placement = getattr(storey, "ObjectPlacement", None)
            if storey_placement:
                placement = model.create_entity(
                    "IfcLocalPlacement",
                    PlacementRelTo=storey_placement,
                    RelativePlacement=model.create_entity(
                        "IfcAxis2Placement3D",
                        Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
                        Axis=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
                        RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0)),
                    ),
                )
                product.ObjectPlacement = placement
    except Exception as exc:
        logger.debug("Failed to ensure product placement: %s", exc)


__all__ = [
    "create_spaces",
]

