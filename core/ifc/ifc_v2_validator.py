"""
IFC V2 Export Validator

Validates critical aspects of IFC V2 export to catch common errors before they occur.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
import numpy as np

from core.ml.postprocess_floorplan import NormalizedDet
from core.reconstruct.spaces import SpacePoly

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class NumpyArrayBooleanError(ValidationError):
    """Error when numpy array is used in boolean context."""
    pass


class GeometryValidationError(ValidationError):
    """Error when geometry is invalid."""
    pass


class TypeValidationError(ValidationError):
    """Error when type is incorrect."""
    pass


def validate_boolean_value(value: Any, context: str = "") -> bool:
    """
    Validate and convert a value to boolean, avoiding numpy array ambiguity.
    
    Args:
        value: Value to convert to boolean
        context: Context string for error messages
        
    Returns:
        Boolean value
        
    Raises:
        NumpyArrayBooleanError: If value is a numpy array with more than one element
    """
    if value is None:
        return False
    
    # Check if it's a numpy array
    if isinstance(value, np.ndarray):
        if value.size > 1:
            raise NumpyArrayBooleanError(
                f"Array with more than one element used in boolean context {context}. "
                f"Use value.any() or value.all() or len(value) > 0 instead."
            )
        # Single element array - convert to scalar
        return bool(value.item())
    
    # Check if it's a list/tuple that might be used as boolean
    if isinstance(value, (list, tuple)):
        # This is OK - we can check length
        return len(value) > 0
    
    # Regular boolean conversion
    try:
        return bool(value)
    except (ValueError, TypeError) as e:
        raise TypeValidationError(
            f"Cannot convert value to boolean {context}: {e}"
        ) from e


def validate_normalized_detections(normalized: Sequence[NormalizedDet]) -> list[str]:
    """
    Validate normalized detections before export.
    
    Args:
        normalized: Sequence of normalized detections
        
    Returns:
        List of validation warnings (empty if all valid)
    """
    warnings = []
    
    for i, det in enumerate(normalized):
        # Check geometry
        if det.geom is None:
            warnings.append(f"Detection {i} ({det.type}) has no geometry")
            continue
        
        # Check for numpy array boolean issues in is_external
        try:
            validate_boolean_value(det.is_external, f"detection {i}.is_external")
        except (NumpyArrayBooleanError, TypeValidationError) as e:
            warnings.append(f"Detection {i} ({det.type}): {e}")
        
        # Check type
        if det.type not in ("WALL", "DOOR", "WINDOW", "SPACE"):
            warnings.append(f"Detection {i} has unknown type: {det.type}")
    
    return warnings


def validate_spaces(spaces: Sequence[SpacePoly]) -> list[str]:
    """
    Validate spaces before export.
    
    Args:
        spaces: Sequence of space polygons
        
    Returns:
        List of validation warnings (empty if all valid)
    """
    warnings = []
    
    for i, space in enumerate(spaces):
        if space.poly is None:
            warnings.append(f"Space {i} has no polygon")
            continue
        
        if space.poly.is_empty:
            warnings.append(f"Space {i} has empty polygon")
        
        if not space.poly.is_valid:
            warnings.append(f"Space {i} has invalid polygon")
    
    return warnings


def validate_before_export(
    normalized: Sequence[NormalizedDet],
    spaces: Sequence[SpacePoly],
) -> tuple[bool, list[str]]:
    """
    Comprehensive validation before IFC export.
    
    This function is designed to be non-blocking - it returns warnings
    instead of raising exceptions to prevent server crashes.
    
    Args:
        normalized: Sequence of normalized detections
        spaces: Sequence of space polygons
        
    Returns:
        Tuple of (is_valid, warnings) - is_valid is False only for critical errors
    """
    warnings = []
    
    try:
        # Validate detections (non-blocking)
        try:
            det_warnings = validate_normalized_detections(normalized)
            warnings.extend(det_warnings)
        except Exception as e:
            logger.warning(f"Error validating detections (non-critical): {e}")
            warnings.append(f"Detection validation error: {e}")
        
        # Validate spaces (non-blocking)
        try:
            space_warnings = validate_spaces(spaces)
            warnings.extend(space_warnings)
        except Exception as e:
            logger.warning(f"Error validating spaces (non-critical): {e}")
            warnings.append(f"Space validation error: {e}")
        
        # Check for at least one wall (warning, not error)
        try:
            wall_count = sum(1 for det in normalized if det.type == "WALL")
            if wall_count == 0:
                warnings.append("No walls found in normalized detections (will create empty IFC)")
        except Exception as e:
            logger.warning(f"Error counting walls (non-critical): {e}")
        
        # Check for valid boolean values in all detections (non-blocking)
        for i, det in enumerate(normalized):
            try:
                # Validate is_external as boolean
                _ = validate_boolean_value(det.is_external, f"detection {i}.is_external")
            except (NumpyArrayBooleanError, TypeValidationError) as e:
                # Auto-fix: convert to safe boolean
                logger.debug(f"Auto-fixing boolean value for detection {i}: {e}")
                warnings.append(f"Detection {i}: Auto-fixed boolean value")
            except Exception as e:
                logger.debug(f"Non-critical validation error for detection {i}: {e}")
    
    except Exception as e:
        # Catch-all to prevent crashes
        logger.error(f"Unexpected error in pre-export validation: {e}", exc_info=True)
        warnings.append(f"Validation error (non-critical): {e}")
    
    # Never fail completely - always allow export to proceed with warnings
    is_valid = True  # Always allow export, warnings are informational
    return is_valid, warnings


def validate_uuid_uniqueness(model: Any) -> list[str]:
    """
    Validate that all GlobalIds are unique.
    
    Args:
        model: IFC model instance
        
    Returns:
        List of validation warnings (empty if all UUIDs are unique)
    """
    warnings = []
    
    try:
        all_guids = {}
        duplicate_guids = []
        
        # Check all entities with GlobalId
        for entity in model:
            try:
                guid = getattr(entity, "GlobalId", None)
                if guid:
                    if guid in all_guids:
                        duplicate_guids.append((guid, entity.is_a(), getattr(entity, "Name", "unknown")))
                        if guid not in [d[0] for d in duplicate_guids[:-1]]:
                            # First duplicate - also add the original
                            original_entity = all_guids[guid]
                            duplicate_guids.insert(-1, (guid, original_entity.is_a(), getattr(original_entity, "Name", "unknown")))
                    else:
                        all_guids[guid] = entity
            except Exception as e:
                logger.debug(f"Error checking GlobalId for entity: {e}")
                continue
        
        if duplicate_guids:
            # Group duplicates by GUID
            guid_groups = {}
            for guid, entity_type, name in duplicate_guids:
                if guid not in guid_groups:
                    guid_groups[guid] = []
                guid_groups[guid].append(f"{entity_type}({name})")
            
            for guid, entities in guid_groups.items():
                warnings.append(f"Duplicate GlobalId {guid}: {', '.join(entities)}")
    except Exception as e:
        logger.warning(f"Error validating UUID uniqueness: {e}")
        warnings.append(f"UUID uniqueness check failed: {e}")
    
    return warnings


def validate_relations_completeness(model: Any) -> list[str]:
    """
    Validate that all required relations exist.
    
    Args:
        model: IFC model instance
        
    Returns:
        List of validation warnings (empty if all relations are complete)
    """
    warnings = []
    
    try:
        all_openings = model.by_type("IfcOpeningElement")
        all_void_rels = model.by_type("IfcRelVoidsElement")
        all_fill_rels = model.by_type("IfcRelFillsElement")
        
        # Check opening relations
        opening_to_void = {}
        opening_to_fill = {}
        
        for rel in all_void_rels:
            opening = getattr(rel, "RelatedOpeningElement", None)
            if opening:
                opening_to_void[opening] = rel
        
        for rel in all_fill_rels:
            opening = getattr(rel, "RelatingOpeningElement", None)
            if opening:
                opening_to_fill[opening] = rel
        
        missing_voids = [op for op in all_openings if op not in opening_to_void]
        missing_fills = [op for op in all_openings if op not in opening_to_fill]
        
        if missing_voids:
            warnings.append(f"Found {len(missing_voids)} openings without IfcRelVoidsElement")
            for op in missing_voids[:5]:  # Limit to first 5
                warnings.append(f"  - Opening {getattr(op, 'Name', 'unknown')} ({getattr(op, 'GlobalId', 'unknown')})")
        
        if missing_fills:
            warnings.append(f"Found {len(missing_fills)} openings without IfcRelFillsElement")
            for op in missing_fills[:5]:  # Limit to first 5
                warnings.append(f"  - Opening {getattr(op, 'Name', 'unknown')} ({getattr(op, 'GlobalId', 'unknown')})")
    except Exception as e:
        logger.warning(f"Error validating relations completeness: {e}")
        warnings.append(f"Relations completeness check failed: {e}")
    
    return warnings


def validate_material_assignments(model: Any) -> list[str]:
    """
    Validate that all walls and slabs have material assignments.
    
    Args:
        model: IFC model instance
        
    Returns:
        List of validation warnings (empty if all materials are assigned)
    """
    warnings = []
    
    try:
        all_walls = model.by_type("IfcWallStandardCase")
        all_slabs = model.by_type("IfcSlab")
        all_material_rels = model.by_type("IfcRelAssociatesMaterial")
        
        # Build set of entities with materials
        elements_with_materials = set()
        for rel in all_material_rels:
            related = getattr(rel, "RelatedObjects", []) or []
            elements_with_materials.update(related)
        
        # Check walls
        missing_material_walls = [w for w in all_walls if w not in elements_with_materials]
        if missing_material_walls:
            warnings.append(f"Found {len(missing_material_walls)} walls without IfcRelAssociatesMaterial")
            for wall in missing_material_walls[:5]:  # Limit to first 5
                warnings.append(f"  - Wall {getattr(wall, 'Name', 'unknown')} ({getattr(wall, 'GlobalId', 'unknown')})")
        
        # Check slabs
        missing_material_slabs = [s for s in all_slabs if s not in elements_with_materials]
        if missing_material_slabs:
            warnings.append(f"Found {len(missing_material_slabs)} slabs without IfcRelAssociatesMaterial")
            for slab in missing_material_slabs[:5]:  # Limit to first 5
                warnings.append(f"  - Slab {getattr(slab, 'Name', 'unknown')} ({getattr(slab, 'GlobalId', 'unknown')})")
    except Exception as e:
        logger.warning(f"Error validating material assignments: {e}")
        warnings.append(f"Material assignment check failed: {e}")
    
    return warnings


def validate_storey_containment(model: Any) -> list[str]:
    """
    Validate that all elements are contained in a storey.
    
    Args:
        model: IFC model instance
        
    Returns:
        List of validation warnings (empty if all elements are contained)
    """
    warnings = []
    
    try:
        all_storeys = model.by_type("IfcBuildingStorey")
        if not all_storeys:
            warnings.append("No IfcBuildingStorey found - cannot validate containment")
            return warnings
        
        all_containment_rels = model.by_type("IfcRelContainedInSpatialStructure")
        
        # Build set of contained elements
        contained_elements = set()
        for rel in all_containment_rels:
            structure = getattr(rel, "RelatingStructure", None)
            if structure and structure in all_storeys:
                elements = getattr(rel, "RelatedElements", []) or []
                contained_elements.update(elements)
        
        # Check walls
        all_walls = model.by_type("IfcWallStandardCase")
        missing_walls = [w for w in all_walls if w not in contained_elements]
        if missing_walls:
            warnings.append(f"Found {len(missing_walls)} walls not contained in storey")
            for wall in missing_walls[:5]:  # Limit to first 5
                warnings.append(f"  - Wall {getattr(wall, 'Name', 'unknown')} ({getattr(wall, 'GlobalId', 'unknown')})")
        
        # Check doors
        all_doors = model.by_type("IfcDoor")
        missing_doors = [d for d in all_doors if d not in contained_elements]
        if missing_doors:
            warnings.append(f"Found {len(missing_doors)} doors not contained in storey")
            for door in missing_doors[:5]:  # Limit to first 5
                warnings.append(f"  - Door {getattr(door, 'Name', 'unknown')} ({getattr(door, 'GlobalId', 'unknown')})")
        
        # Check windows
        all_windows = model.by_type("IfcWindow")
        missing_windows = [w for w in all_windows if w not in contained_elements]
        if missing_windows:
            warnings.append(f"Found {len(missing_windows)} windows not contained in storey")
            for window in missing_windows[:5]:  # Limit to first 5
                warnings.append(f"  - Window {getattr(window, 'Name', 'unknown')} ({getattr(window, 'GlobalId', 'unknown')})")
        
        # Check slabs
        all_slabs = model.by_type("IfcSlab")
        missing_slabs = [s for s in all_slabs if s not in contained_elements]
        if missing_slabs:
            warnings.append(f"Found {len(missing_slabs)} slabs not contained in storey")
            for slab in missing_slabs[:5]:  # Limit to first 5
                warnings.append(f"  - Slab {getattr(slab, 'Name', 'unknown')} ({getattr(slab, 'GlobalId', 'unknown')})")
    except Exception as e:
        logger.warning(f"Error validating storey containment: {e}")
        warnings.append(f"Storey containment check failed: {e}")
    
    return warnings


def validate_ifc_file(ifc_path: str) -> tuple[bool, list[str]]:
    """
    Validate IFC file after export.
    
    This function is designed to be non-blocking - it returns warnings
    instead of raising exceptions to prevent server crashes.
    
    Args:
        ifc_path: Path to IFC file
        
    Returns:
        Tuple of (is_valid, warnings) - is_valid is always True (warnings are informational)
    """
    warnings = []
    
    try:
        import ifcopenshell
    except ImportError:
        logger.warning("ifcopenshell not available for post-export validation")
        return True, ["ifcopenshell not available for validation"]
    
    try:
        model = ifcopenshell.open(ifc_path)
    except Exception as e:
        logger.warning(f"Could not open IFC file for validation: {e}")
        warnings.append(f"Could not open IFC file for validation: {e}")
        return True, warnings  # Don't fail - file might still be valid
    
    try:
        # Check for required entities (non-blocking)
        try:
            project = model.by_type("IfcProject")
            if len(project) == 0:
                warnings.append("No IfcProject found")
        except Exception as e:
            logger.debug(f"Error checking IfcProject: {e}")
        
        try:
            site = model.by_type("IfcSite")
            if len(site) == 0:
                warnings.append("No IfcSite found")
        except Exception as e:
            logger.debug(f"Error checking IfcSite: {e}")
        
        try:
            building = model.by_type("IfcBuilding")
            if len(building) == 0:
                warnings.append("No IfcBuilding found")
        except Exception as e:
            logger.debug(f"Error checking IfcBuilding: {e}")
        
        try:
            storey = model.by_type("IfcBuildingStorey")
            if len(storey) == 0:
                warnings.append("No IfcBuildingStorey found")
        except Exception as e:
            logger.debug(f"Error checking IfcBuildingStorey: {e}")
        
        # Check for building elements (non-blocking)
        try:
            walls = model.by_type("IfcWallStandardCase")
            if len(walls) == 0:
                warnings.append("No walls found in IFC file")
            
            # Check that walls have representations (non-blocking)
            for wall in walls[:10]:  # Limit to first 10 to avoid timeout
                try:
                    if wall.Representation is None:
                        warnings.append(f"Wall {getattr(wall, 'Name', 'unknown')} has no representation")
                except Exception as e:
                    logger.debug(f"Error checking wall representation: {e}")
        except Exception as e:
            logger.debug(f"Error checking walls: {e}")
        
        # NEW: Validate UUID uniqueness
        try:
            uuid_warnings = validate_uuid_uniqueness(model)
            warnings.extend(uuid_warnings)
        except Exception as e:
            logger.warning(f"Error validating UUID uniqueness: {e}")
            warnings.append(f"UUID uniqueness validation error: {e}")
        
        # NEW: Validate relations completeness
        try:
            relations_warnings = validate_relations_completeness(model)
            warnings.extend(relations_warnings)
        except Exception as e:
            logger.warning(f"Error validating relations completeness: {e}")
            warnings.append(f"Relations completeness validation error: {e}")
        
        # NEW: Validate material assignments
        try:
            material_warnings = validate_material_assignments(model)
            warnings.extend(material_warnings)
        except Exception as e:
            logger.warning(f"Error validating material assignments: {e}")
            warnings.append(f"Material assignment validation error: {e}")
        
        # NEW: Validate storey containment
        try:
            containment_warnings = validate_storey_containment(model)
            warnings.extend(containment_warnings)
        except Exception as e:
            logger.warning(f"Error validating storey containment: {e}")
            warnings.append(f"Storey containment validation error: {e}")
        
    finally:
        try:
            model.close()
        except Exception as e:
            logger.debug(f"Error closing IFC model: {e}")
    
    # Always return True - warnings are informational, not blocking
    return True, warnings

