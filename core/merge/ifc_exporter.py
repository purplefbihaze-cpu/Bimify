"""IFC4 exporter for canonical plan, following template structure."""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

import ifcopenshell
import ifcopenshell.api

from .schema import CanonicalPlan
from .template_ingest import extract_template_profile as extract_template_profile_module
from core.ifc.project_setup import (
    create_project_structure,
    setup_calibration,
    setup_georeferencing,
    setup_owner_history,
)

logger = logging.getLogger(__name__)


def extract_template_profile(template_path: Path) -> Dict[str, Any]:
    """Extract export profile from template IFC."""
    return extract_template_profile_module(template_path)


def export_to_ifc(
    plan: CanonicalPlan,
    output_path: Path,
    template_path: Optional[Path] = None,
    storey_height_m: float = 3.0,
    door_height_m: float = 2.1,
    window_height_m: float = 1.0,
) -> Path:
    """Export canonical plan to IFC4, following template if provided."""
    
    # Create IFC model
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    
    # Setup owner history
    setup_owner_history(model, is_ifc2x3=False)
    
    # Extract template profile if available
    template_profile = {}
    if template_path and template_path.exists():
        template_profile = extract_template_profile(template_path)
    
    # Get spatial hierarchy from template
    spatial = template_profile.get("spatial_hierarchy", {})
    
    # Create project structure
    project, site, building, storey = create_project_structure(
        model,
        project_name=spatial.get("project_name", "Bimify Project"),
        storey_name=spatial.get("storey_name", "EG"),
        storey_elevation=spatial.get("storey_elevation", 0.0),
    )
    
    # Get contexts - they are created by create_project_structure
    contexts = model.by_type("IfcGeometricRepresentationContext")
    model_context = contexts[0] if contexts else None
    
    subcontexts = model.by_type("IfcGeometricRepresentationSubContext")
    body_context = next((sc for sc in subcontexts if (sc.ContextIdentifier or "").lower() == "body"), None)
    if not body_context and model_context:
        body_context = ifcopenshell.api.run(
            "context.add_context",
            model,
            context_type="Model",
            context_identifier="Body",
            target_view="MODEL_VIEW",
            parent=model_context,
        )
    
    axis_context = next((sc for sc in subcontexts if (sc.ContextIdentifier or "").lower() == "axis"), None)
    if not axis_context and model_context:
        axis_context = ifcopenshell.api.run(
            "context.add_context",
            model,
            context_type="Model",
            context_identifier="Axis",
            target_view="MODEL_VIEW",
            parent=model_context,
        )
    
    # Create material layer set for walls
    material = ifcopenshell.api.run("material.add_material", model, name="Wall Material")
    
    # Create walls
    wall_map: Dict[str, Any] = {}
    for wall in plan.walls:
        if len(wall.polyline) < 2:
            continue
        
        # Create wall using ifcopenshell geometry API
        # Convert wall polyline to LineString for geometry creation
        from shapely.geometry import LineString
        
        wall_coords = [(pt.x, pt.y) for pt in wall.polyline]
        wall_axis = LineString(wall_coords)
        
        # Create wall representation using ifcopenshell geometry API
        # Use the existing pattern from build_ifc43_model
        try:
            # Create wall representation
            body_rep = ifcopenshell.api.run(
                "geometry.add_wall_representation",
                model,
                context=body_context,
                length=wall_axis.length,
                height=storey_height_m,
                thickness=wall.thickness,
            )
        except:
            # Fallback: create manually
            start = wall.polyline[0]
            end = wall.polyline[-1]
            dx = end.x - start.x
            dy = end.y - start.y
            length = (dx * dx + dy * dy) ** 0.5
            
            # Perpendicular direction
            perp_x = -dy / length if length > 0 else 0.0
            perp_y = dx / length if length > 0 else 0.0
            
            half_thick = wall.thickness / 2
            profile_points = [
                model.createIfcCartesianPoint((start.x + perp_x * half_thick, start.y + perp_y * half_thick, 0.0)),
                model.createIfcCartesianPoint((end.x + perp_x * half_thick, end.y + perp_y * half_thick, 0.0)),
                model.createIfcCartesianPoint((end.x - perp_x * half_thick, end.y - perp_y * half_thick, 0.0)),
                model.createIfcCartesianPoint((start.x - perp_x * half_thick, start.y - perp_y * half_thick, 0.0)),
            ]
            profile_polyline = model.createIfcPolyline(profile_points)
            profile = model.createIfcArbitraryClosedProfileDef("AREA", None, profile_polyline)
            
            body_rep = ifcopenshell.api.run(
                "geometry.add_profile_representation",
                model,
                context=body_context,
                profile=profile,
                depth=storey_height_m,
            )
        
        # Create axis representation
        axis_points = []
        for pt in wall.polyline:
            axis_points.append(model.createIfcCartesianPoint((pt.x, pt.y, 0.0)))
        
        axis_polyline = model.createIfcPolyline(axis_points)
        try:
            axis_rep = ifcopenshell.api.run(
                "geometry.add_axis_representation",
                model,
                context=axis_context,
                axis=axis_polyline,
            )
        except:
            axis_rep = None
        
        # Create wall
        wall_elem = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcWallStandardCase",
            name=f"Wall-{wall.id}",
        )
        
        # Assign representation
        ifcopenshell.api.run(
            "geometry.assign_representation",
            model,
            product=wall_elem,
            representation=body_rep,
        )
        
        # Assign axis if available
        if axis_rep:
            try:
                ifcopenshell.api.run(
                    "geometry.assign_representation",
                    model,
                    product=wall_elem,
                    representation=axis_rep,
                )
            except:
                pass
        
        # Place in storey
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            product=wall_elem,
            relating_structure=storey,
        )
        
        # Set external property
        if wall.isExternal:
            pset = ifcopenshell.api.run("pset.add_pset", model, product=wall_elem, name="Pset_WallCommon")
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={"IsExternal": True})
        
        wall_map[wall.id] = wall_elem
    
    # Create openings and doors/windows
    for opening in plan.openings:
        host_wall = wall_map.get(opening.hostWallId)
        if not host_wall:
            continue
        
        # Get wall geometry to compute opening position
        wall_start = opening.hostWallId  # We need the actual wall points
        # For now, create opening using void.add_opening which creates the opening automatically
        # We'll need to position it correctly based on the wall axis and opening.s parameter
        
        # Find the wall to get its axis
        wall_obj = None
        for w in plan.walls:
            if w.id == opening.hostWallId:
                wall_obj = w
                break
        
        if not wall_obj or len(wall_obj.polyline) < 2:
            continue
        
        # Compute opening position along wall
        start_pt = wall_obj.polyline[0]
        end_pt = wall_obj.polyline[-1]
        dx = end_pt.x - start_pt.x
        dy = end_pt.y - start_pt.y
        length = (dx * dx + dy * dy) ** 0.5
        
        s = opening.s
        opening_x = start_pt.x + s * dx
        opening_y = start_pt.y + s * dy
        
        # Create opening element first
        opening_elem = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcOpeningElement",
            name=f"Opening-{opening.id}",
        )
        
        # Create opening representation (box)
        # Opening dimensions: width x height x wall_thickness
        opening_profile_points = [
            model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
            model.createIfcCartesianPoint((opening.width, 0.0, 0.0)),
            model.createIfcCartesianPoint((opening.width, opening.height, 0.0)),
            model.createIfcCartesianPoint((0.0, opening.height, 0.0)),
        ]
        opening_profile_polyline = model.createIfcPolyline(opening_profile_points)
        opening_profile = model.createIfcArbitraryClosedProfileDef("AREA", None, opening_profile_polyline)
        
        try:
            opening_rep = ifcopenshell.api.run(
                "geometry.add_profile_representation",
                model,
                context=body_context,
                profile=opening_profile,
                depth=wall_obj.thickness,
            )
        except:
            # Fallback: create simple box representation
            opening_rep = None
        
        if opening_rep:
            ifcopenshell.api.run(
                "geometry.assign_representation",
                model,
                product=opening_elem,
                representation=opening_rep,
            )
        
        # Position opening along wall
        # Create local placement for opening
        # Compute placement matrix based on wall axis and opening position
        wall_dir_x = dx / length if length > 0 else 1.0
        wall_dir_y = dy / length if length > 0 else 0.0
        perp_x = -wall_dir_y
        perp_y = wall_dir_x
        
        # Opening center position
        opening_center = (opening_x, opening_y, opening.height / 2)
        
        # Create placement
        opening_placement = model.createIfcLocalPlacement(
            storey.ObjectPlacement,
            model.createIfcAxis2Placement3D(
                model.createIfcCartesianPoint(opening_center),
                model.createIfcDirection((0.0, 0.0, 1.0)),
                model.createIfcDirection((wall_dir_x, wall_dir_y, 0.0)),
            )
        )
        opening_elem.ObjectPlacement = opening_placement
        
        # Void the wall
        ifcopenshell.api.run(
            "void.add_opening",
            model,
            opening=opening_elem,
            element=host_wall,
        )
        
        # Create door or window
        if opening.type == "door":
            fill_elem = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcDoor",
                name=f"Door-{opening.id}",
            )
        else:
            fill_elem = ifcopenshell.api.run(
                "root.create_entity",
                model,
                ifc_class="IfcWindow",
                name=f"Window-{opening.id}",
            )
        
        # Fill opening
        ifcopenshell.api.run(
            "void.add_filling",
            model,
            opening=opening_elem,
            element=fill_elem,
        )
        
        # Place in storey
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            product=fill_elem,
            relating_structure=storey,
        )
    
    # Create spaces
    for room in plan.rooms:
        if len(room.polygon) < 3:
            continue
        
        # Create space footprint
        space_points = []
        for pt in room.polygon:
            space_points.append(model.createIfcCartesianPoint((pt.x, pt.y, 0.0)))
        
        # Close polygon
        if space_points[0] != space_points[-1]:
            space_points.append(space_points[0])
        
        space_polyline = model.createIfcPolyline(space_points)
        space_profile = model.createIfcArbitraryClosedProfileDef("AREA", None, space_polyline)
        
        space_rep = ifcopenshell.api.run(
            "geometry.add_profile_representation",
            model,
            context=body_context,
            profile=space_profile,
            depth=storey_height_m,
        )
        
        space_elem = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcSpace",
            name=room.name or f"Space-{room.id}",
        )
        
        ifcopenshell.api.run(
            "geometry.assign_representation",
            model,
            product=space_elem,
            representation=space_rep,
        )
        
        ifcopenshell.api.run(
            "spatial.assign_container",
            model,
            product=space_elem,
            relating_structure=storey,
        )
        
        # Add area quantity
        pset = ifcopenshell.api.run("pset.add_pset", model, product=space_elem, name="Qto_SpaceBaseQuantities")
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={"GrossFloorArea": room.area})
    
    # Write file
    model.write(str(output_path))
    logger.info(f"Exported IFC to {output_path}")
    
    return output_path

