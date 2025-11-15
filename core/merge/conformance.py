"""IFC conformance checker against template profile."""

from pathlib import Path
from typing import Dict, Any, List
import logging

try:
    import ifcopenshell
    import ifcopenshell.util.element
except ImportError:
    ifcopenshell = None

from .template_ingest import extract_template_profile

logger = logging.getLogger(__name__)


def check_ifc_conformance(
    exported_ifc_path: Path,
    template_path: Path
) -> Dict[str, Any]:
    """Check exported IFC against template profile.
    
    Returns conformance report with:
    - matches: what matches template
    - mismatches: what doesn't match
    - missing: what's in template but not in export
    - extra: what's in export but not in template
    """
    if ifcopenshell is None:
        return {
            "error": "ifcopenshell not available",
            "matches": [],
            "mismatches": [],
            "missing": [],
            "extra": [],
        }
    
    try:
        exported = ifcopenshell.open(str(exported_ifc_path))
        template_profile = extract_template_profile(template_path)
    except Exception as e:
        return {
            "error": str(e),
            "matches": [],
            "mismatches": [],
            "missing": [],
            "extra": [],
        }
    
    report: Dict[str, Any] = {
        "matches": [],
        "mismatches": [],
        "missing": [],
        "extra": [],
        "summary": {},
    }
    
    # Check units
    try:
        exp_units = exported.by_type("IfcUnitAssignment")
        if exp_units and template_profile.get("units"):
            report["matches"].append("Units structure present")
        elif not exp_units and template_profile.get("units"):
            report["missing"].append("Unit assignment")
    except:
        pass
    
    # Check contexts
    try:
        exp_contexts = exported.by_type("IfcGeometricRepresentationContext")
        template_contexts = template_profile.get("contexts", {})
        if exp_contexts and template_contexts:
            report["matches"].append("Geometric representation contexts present")
        elif not exp_contexts and template_contexts:
            report["missing"].append("Geometric representation contexts")
    except:
        pass
    
    # Check subcontexts
    try:
        exp_subcontexts = exported.by_type("IfcGeometricRepresentationSubContext")
        template_subcontexts = template_profile.get("subcontexts", {})
        if exp_subcontexts:
            exp_ids = {getattr(sc, "ContextIdentifier", None) for sc in exp_subcontexts}
            template_ids = set(template_subcontexts.keys())
            if exp_ids & template_ids:
                report["matches"].append(f"Subcontexts present: {exp_ids & template_ids}")
            if template_ids - exp_ids:
                report["missing"].append(f"Missing subcontexts: {template_ids - exp_ids}")
    except:
        pass
    
    # Check spatial hierarchy
    try:
        exp_projects = exported.by_type("IfcProject")
        exp_sites = exported.by_type("IfcSite")
        exp_buildings = exported.by_type("IfcBuilding")
        exp_storeys = exported.by_type("IfcBuildingStorey")
        
        if exp_projects and exp_sites and exp_buildings and exp_storeys:
            report["matches"].append("Spatial hierarchy complete (Project/Site/Building/Storey)")
        else:
            missing = []
            if not exp_projects:
                missing.append("Project")
            if not exp_sites:
                missing.append("Site")
            if not exp_buildings:
                missing.append("Building")
            if not exp_storeys:
                missing.append("Storey")
            report["missing"].append(f"Missing spatial elements: {missing}")
    except:
        pass
    
    # Check entity types
    try:
        exp_walls = exported.by_type("IfcWallStandardCase")
        exp_doors = exported.by_type("IfcDoor")
        exp_windows = exported.by_type("IfcWindow")
        exp_spaces = exported.by_type("IfcSpace")
        exp_openings = exported.by_type("IfcOpeningElement")
        
        entity_counts = {
            "walls": len(exp_walls),
            "doors": len(exp_doors),
            "windows": len(exp_windows),
            "spaces": len(exp_spaces),
            "openings": len(exp_openings),
        }
        
        report["summary"]["entity_counts"] = entity_counts
        
        if exp_walls:
            report["matches"].append(f"Walls present: {len(exp_walls)}")
        if exp_doors:
            report["matches"].append(f"Doors present: {len(exp_doors)}")
        if exp_windows:
            report["matches"].append(f"Windows present: {len(exp_windows)}")
        if exp_spaces:
            report["matches"].append(f"Spaces present: {len(exp_spaces)}")
        if exp_openings:
            report["matches"].append(f"Openings present: {len(exp_openings)}")
    except Exception as e:
        report["mismatches"].append(f"Error checking entity types: {e}")
    
    # Check property sets
    try:
        template_psets = template_profile.get("psets", {})
        if template_psets:
            # Sample check on first wall
            exp_walls = exported.by_type("IfcWallStandardCase")
            if exp_walls:
                wall_psets = ifcopenshell.util.element.get_psets(exp_walls[0], should_inherit=False)
                template_wall_psets = set(template_psets.get("wall", []))
                exp_wall_psets = set(wall_psets.keys())
                
                if template_wall_psets & exp_wall_psets:
                    report["matches"].append(f"Wall property sets match: {template_wall_psets & exp_wall_psets}")
                if template_wall_psets - exp_wall_psets:
                    report["missing"].append(f"Missing wall property sets: {template_wall_psets - exp_wall_psets}")
    except Exception as e:
        logger.debug(f"Could not check property sets: {e}")
    
    report["summary"]["total_matches"] = len(report["matches"])
    report["summary"]["total_mismatches"] = len(report["mismatches"])
    report["summary"]["total_missing"] = len(report["missing"])
    report["summary"]["total_extra"] = len(report["extra"])
    
    return report

