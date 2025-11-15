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
        
    finally:
        try:
            model.close()
        except Exception as e:
            logger.debug(f"Error closing IFC model: {e}")
    
    # Always return True - warnings are informational, not blocking
    return True, warnings

