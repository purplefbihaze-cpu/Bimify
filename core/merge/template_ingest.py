"""Template IFC ingestion to extract export profile."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import ifcopenshell
except ImportError:
    ifcopenshell = None

logger = logging.getLogger(__name__)


def extract_template_profile(template_path: Path) -> Dict[str, Any]:
    """Extract export profile from template IFC.
    
    Returns profile dict with:
    - units: unit assignments
    - contexts: geometric representation contexts
    - subcontexts: subcontexts (Body, Axis, etc.)
    - spatial_hierarchy: project/site/building/storey names
    - psets: required property sets per entity type
    - representation_types: representation types used
    """
    if ifcopenshell is None:
        logger.warning("ifcopenshell not available, using defaults")
        return {}
    
    try:
        template = ifcopenshell.open(str(template_path))
    except Exception as e:
        logger.warning(f"Could not open template IFC: {e}")
        return {}
    
    profile: Dict[str, Any] = {
        "units": {},
        "contexts": {},
        "subcontexts": {},
        "spatial_hierarchy": {},
        "psets": {},
        "representation_types": {},
    }
    
    # Extract units
    try:
        unit_assignments = template.by_type("IfcUnitAssignment")
        if unit_assignments:
            for unit in unit_assignments[0].Units:
                if hasattr(unit, "UnitType"):
                    unit_type = str(unit.UnitType)
                    profile["units"][unit_type] = {
                        "type": unit_type,
                        "name": getattr(unit, "Name", None),
                    }
    except Exception as e:
        logger.debug(f"Could not extract units: {e}")
    
    # Extract contexts
    try:
        contexts = template.by_type("IfcGeometricRepresentationContext")
        for ctx in contexts:
            ctx_id = getattr(ctx, "ContextIdentifier", None) or "Model"
            profile["contexts"][ctx_id] = {
                "context_type": getattr(ctx, "ContextType", None),
                "coordinate_space_dimension": getattr(ctx, "CoordinateSpaceDimension", None),
                "precision": getattr(ctx, "Precision", None),
            }
    except Exception as e:
        logger.debug(f"Could not extract contexts: {e}")
    
    # Extract subcontexts
    try:
        subcontexts = template.by_type("IfcGeometricRepresentationSubContext")
        for subctx in subcontexts:
            ctx_id = getattr(subctx, "ContextIdentifier", None) or "Body"
            profile["subcontexts"][ctx_id] = {
                "context_type": getattr(subctx, "ContextType", None),
                "target_view": getattr(subctx, "TargetView", None),
            }
    except Exception as e:
        logger.debug(f"Could not extract subcontexts: {e}")
    
    # Extract spatial hierarchy
    try:
        projects = template.by_type("IfcProject")
        if projects:
            profile["spatial_hierarchy"]["project_name"] = getattr(projects[0], "Name", "Bimify Project")
        
        sites = template.by_type("IfcSite")
        if sites:
            profile["spatial_hierarchy"]["site_name"] = getattr(sites[0], "Name", "Site")
        
        buildings = template.by_type("IfcBuilding")
        if buildings:
            profile["spatial_hierarchy"]["building_name"] = getattr(buildings[0], "Name", "Building")
        
        storeys = template.by_type("IfcBuildingStorey")
        if storeys:
            profile["spatial_hierarchy"]["storey_name"] = getattr(storeys[0], "Name", "EG")
            profile["spatial_hierarchy"]["storey_elevation"] = getattr(storeys[0], "Elevation", 0.0)
    except Exception as e:
        logger.debug(f"Could not extract spatial hierarchy: {e}")
    
    # Extract property sets (sample from walls, doors, windows, spaces)
    try:
        walls = template.by_type("IfcWallStandardCase")
        if walls:
            sample_wall = walls[0]
            psets = ifcopenshell.util.element.get_psets(sample_wall, should_inherit=False) if hasattr(ifcopenshell, "util") else {}
            profile["psets"]["wall"] = list(psets.keys())
        
        doors = template.by_type("IfcDoor")
        if doors:
            sample_door = doors[0]
            psets = ifcopenshell.util.element.get_psets(sample_door, should_inherit=False) if hasattr(ifcopenshell, "util") else {}
            profile["psets"]["door"] = list(psets.keys())
        
        windows = template.by_type("IfcWindow")
        if windows:
            sample_window = windows[0]
            psets = ifcopenshell.util.element.get_psets(sample_window, should_inherit=False) if hasattr(ifcopenshell, "util") else {}
            profile["psets"]["window"] = list(psets.keys())
        
        spaces = template.by_type("IfcSpace")
        if spaces:
            sample_space = spaces[0]
            psets = ifcopenshell.util.element.get_psets(sample_space, should_inherit=False) if hasattr(ifcopenshell, "util") else {}
            profile["psets"]["space"] = list(psets.keys())
    except Exception as e:
        logger.debug(f"Could not extract property sets: {e}")
    
    return profile

