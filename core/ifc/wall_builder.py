"""Wall building functions for IFC models."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import ifcopenshell.api
from ifcopenshell.util import element as ifc_element_utils
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.ifc.geometry_utils import snap_thickness_mm, prepare_thickness_standards


logger = logging.getLogger(__name__)


def create_wall_types(
    model: Any,
    project: Any,
    *,
    is_ifc2x3: bool = False,
) -> Tuple[Any, Any]:
    """Create IfcWallType entities for external and internal walls.
    
    Args:
        model: IFC model instance.
        project: IfcProject entity.
        is_ifc2x3: Whether using IFC2X3 schema.
    
    Returns:
        Tuple of (external_wall_type, internal_wall_type).
    """
    external_wall_type = None
    internal_wall_type = None
    
    try:
        external_wall_type = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcWallType",
            name="ExternalWallType",
        )
        _safe_set_predefined_type(external_wall_type, "STANDARD", is_ifc2x3)
        ifcopenshell.api.run("type.assign_type", model, related_object=project, relating_type=external_wall_type)
        
        internal_wall_type = ifcopenshell.api.run(
            "root.create_entity",
            model,
            ifc_class="IfcWallType",
            name="InternalWallType",
        )
        _safe_set_predefined_type(internal_wall_type, "STANDARD", is_ifc2x3)
        ifcopenshell.api.run("type.assign_type", model, related_object=project, relating_type=internal_wall_type)
    except Exception as exc:
        logger.warning("Could not create IfcWallType: %s", exc)
    
    return external_wall_type, internal_wall_type


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


def create_wall_from_axis(
    model: Any,
    storey: Any,
    axis_info: WallAxis,
    detection: NormalizedDet,
    *,
    height_mm: float,
    thickness_standards: Sequence[float],
    is_ifc2x3: bool = False,
    source_polygon: Polygon | MultiPolygon | None = None,
    px_per_mm: float | None = None,
) -> Dict[str, Any]:
    """Create an IfcWallStandardCase from a wall axis.
    
    Args:
        model: IFC model instance.
        storey: IfcBuildingStorey entity.
        axis_info: Wall axis information.
        detection: Normalized detection for the wall.
        height_mm: Wall height in millimeters.
        thickness_standards: Sequence of standard thickness values.
        is_ifc2x3: Whether using IFC2X3 schema.
        source_polygon: Source polygon geometry.
        px_per_mm: Pixels per millimeter ratio.
    
    Returns:
        Dictionary with wall entity and metadata.
    """
    wall = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcWallStandardCase",
        name=f"Wall_{axis_info.metadata.get('wall_index', 'unknown')}",
    )
    _safe_set_predefined_type(wall, "STANDARD", is_ifc2x3)
    ifcopenshell.api.run("spatial.assign_container", model, products=[wall], relating_structure=storey)
    
    pset_common = _safe_add_pset(model, wall, "Pset_WallCommon")
    
    # Determine if external
    is_ext = detection.is_external if detection.is_external is not None else False
    
    # Snap wall thickness
    original_width = axis_info.width_mm
    snapped_width = snap_thickness_mm(
        original_width,
        is_external=is_ext,
        standards=thickness_standards,
    )
    if original_width is not None and math.isfinite(original_width):
        axis_info.metadata["width_raw_mm"] = float(original_width)
    axis_info.metadata["width_snapped_mm"] = float(snapped_width)
    axis_info.width_mm = snapped_width
    
    # Set wall properties
    try:
        wall.ObjectType = "ExternalWall" if is_ext else "InternalWall"
    except Exception:
        pass
    
    # CRITICAL: Always set IsExternal property (BIM requirement)
    wall_props = {"IsExternal": bool(is_ext)}
    if is_ext:
        wall_props["MaterialType"] = "MASONRY"
    else:
        wall_props["MaterialType"] = "MASONRY"
    
    try:
        if pset_common is not None:
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties=wall_props)
    except Exception as pset_exc:
        logger.warning("Failed to set IsExternal property for wall %s: %s", wall.Name, pset_exc)
        try:
            existing_props = {}
            try:
                if hasattr(pset_common, "HasProperties"):
                    for prop in getattr(pset_common, "HasProperties", []):
                        if hasattr(prop, "Name"):
                            existing_props[prop.Name] = prop.NominalValue.wrappedValue if hasattr(prop, "NominalValue") else None
            except Exception:
                pass
            existing_props.update(wall_props)
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties=existing_props)
        except Exception:
            logger.error("Failed to set IsExternal property for wall %s (all methods exhausted)", wall.Name)
    
    # Add Bimify parameters
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
    
    # Add source geometry metadata
    source_geom = source_polygon
    if source_geom is None:
        source_geom = _resolve_polygon(detection)
    
    area_mm2 = float(source_geom.area) if source_geom is not None else 0.0
    if source_geom is not None:
        minx, miny, maxx, maxy = source_geom.bounds
    else:
        minx = miny = maxx = maxy = 0.0
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    confidence = 0.0
    if isinstance(detection.attrs, dict) and detection.attrs.get("confidence") is not None:
        try:
            confidence = float(detection.attrs.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
    
    props_source = {
        "SourceIndex": float(axis_info.source_index),
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
    
    return {
        "wall": wall,
        "axis_info": axis_info,
        "detection": detection,
        "is_external": is_ext,
    }


def _resolve_polygon(nd: NormalizedDet) -> Polygon | None:
    """Resolve polygon from normalized detection."""
    geom = nd.geom
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        try:
            return max(list(geom.geoms), key=lambda g: g.area)
        except ValueError:
            return None
    return None


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


def post_process_wall_classification(
    model: Any,
    wall_export_items: List[Dict[str, Any]],
) -> None:
    """Post-process wall classification to ensure 100% coverage.
    
    Analyzes all walls together after creation for more accurate perimeter detection.
    
    Args:
        model: IFC model instance.
        wall_export_items: List of wall export items with wall entities and axes.
    """
    try:
        all_walls = model.by_type("IfcWallStandardCase")
        if len(all_walls) == 0:
            return
        
        # Build comprehensive building envelope from all wall axes
        from shapely.geometry import MultiPoint
        all_axis_points = []
        wall_axis_map = {}
        
        for item in wall_export_items:
            wall_entity = item.get("wall")
            axis_info = item.get("axis")
            if wall_entity and axis_info and hasattr(axis_info, "axis") and axis_info.axis:
                wall_axis_map[wall_entity] = axis_info.axis
                coords = list(axis_info.axis.coords)
                all_axis_points.extend([Point(c) for c in coords])
        
        # Create building envelope using convex hull
        building_envelope = None
        if len(all_axis_points) >= 3:
            try:
                multi_point = MultiPoint(all_axis_points)
                building_envelope = multi_point.convex_hull
                
                if building_envelope.is_empty or not building_envelope.is_valid:
                    try:
                        building_envelope = building_envelope.buffer(0)
                        if building_envelope.is_empty or not building_envelope.is_valid:
                            all_geoms = [item.get("axis").axis for item in wall_export_items 
                                       if item.get("axis") and hasattr(item.get("axis"), "axis")]
                            if all_geoms:
                                building_envelope = unary_union(all_geoms).envelope
                    except Exception:
                        building_envelope = None
            except Exception:
                building_envelope = None
        
        # Re-analyze walls that might need classification correction
        walls_needing_reclassification = []
        for wall in all_walls:
            try:
                psets = ifc_element_utils.get_psets(wall, should_inherit=False)
                wall_common = psets.get("Pset_WallCommon", {})
                is_external = wall_common.get("IsExternal")
                
                if is_external is None:
                    walls_needing_reclassification.append((wall, None))
                elif building_envelope and wall in wall_axis_map:
                    axis_line = wall_axis_map[wall]
                    if hasattr(axis_line, "coords") and building_envelope is not None:
                        axis_coords = list(axis_line.coords)
                        if len(axis_coords) >= 2:
                            try:
                                psets = ifc_element_utils.get_psets(wall, should_inherit=False)
                                wall_params = psets.get("Bimify_WallParams", {})
                                wall_thickness = wall_params.get("WidthMm", 240.0)
                                if not isinstance(wall_thickness, (int, float)):
                                    wall_thickness = 240.0
                            except Exception:
                                wall_thickness = 240.0
                            
                            tolerance = max(min(wall_thickness * 1.5, 600.0), 300.0)
                            
                            check_points = [
                                Point(axis_coords[0]),
                                Point(axis_coords[-1]),
                            ]
                            if len(axis_coords) > 2:
                                mid_idx = len(axis_coords) // 2
                                check_points.append(Point(axis_coords[mid_idx]))
                            
                            points_on_perimeter = 0
                            min_distance = float('inf')
                            for pt in check_points:
                                try:
                                    dist = building_envelope.boundary.distance(pt)
                                    min_distance = min(min_distance, dist)
                                    if dist <= tolerance:
                                        points_on_perimeter += 1
                                except Exception:
                                    continue
                            
                            try:
                                axis_dist = axis_line.distance(building_envelope.boundary)
                                min_distance = min(min_distance, axis_dist)
                                intersects = axis_line.intersects(building_envelope.boundary)
                                is_on_perimeter = (intersects or 
                                                  points_on_perimeter >= 2 or 
                                                  min_distance <= wall_thickness)
                            except Exception:
                                is_on_perimeter = points_on_perimeter >= 1
                            
                            if is_on_perimeter and not is_external:
                                walls_needing_reclassification.append((wall, True))
                                logger.debug("Post-processing: Reclassifying wall %s from internal to external (on perimeter)", 
                                           getattr(wall, "Name", "unknown"))
            except Exception:
                walls_needing_reclassification.append((wall, False))
        
        # Reclassify walls that need it
        for wall, new_is_external in walls_needing_reclassification:
            try:
                if new_is_external is None:
                    new_is_external = False
                
                pset_common = ifc_element_utils.get_psets(wall, should_inherit=False).get("Pset_WallCommon")
                if pset_common is None:
                    pset_common = _safe_add_pset(model, wall, "Pset_WallCommon")
                
                ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties={"IsExternal": bool(new_is_external)})
                logger.debug("Post-processed: Set IsExternal=%s for wall %s", new_is_external, getattr(wall, "Name", "unknown"))
            except Exception as reclass_exc:
                logger.warning("Failed to reclassify wall %s: %s", getattr(wall, "Name", "unknown"), reclass_exc)
        
        # Final verification: ensure 100% coverage
        final_unclassified = []
        for wall in all_walls:
            try:
                psets = ifc_element_utils.get_psets(wall, should_inherit=False)
                wall_common = psets.get("Pset_WallCommon", {})
                is_external = wall_common.get("IsExternal")
                if is_external is None:
                    final_unclassified.append(wall)
            except Exception:
                final_unclassified.append(wall)
        
        if final_unclassified:
            logger.warning("Post-processing: %d wall(s) still unclassified after reanalysis, setting default (100% coverage guarantee)", len(final_unclassified))
            for wall in final_unclassified:
                try:
                    is_ext_default = False
                    try:
                        psets = ifc_element_utils.get_psets(wall, should_inherit=False)
                        wall_params = psets.get("Bimify_WallParams", {})
                        wall_thickness = wall_params.get("WidthMm", 0.0)
                        if isinstance(wall_thickness, (int, float)) and wall_thickness >= 200.0:
                            is_ext_default = True
                    except Exception:
                        pass
                    
                    pset_common = _safe_add_pset(model, wall, "Pset_WallCommon")
                    ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties={"IsExternal": bool(is_ext_default)})
                    logger.debug("Last resort: Set IsExternal=%s for wall %s (100% coverage guarantee)", 
                               is_ext_default, getattr(wall, "Name", "unknown"))
                except Exception as last_resort_exc:
                    logger.error("CRITICAL: Failed to set IsExternal for wall %s even in last resort: %s", 
                               getattr(wall, "Name", "unknown"), last_resort_exc)
        else:
            logger.info("Post-processing: 100% IsExternal coverage achieved - all walls classified")
    except Exception as postproc_exc:
        logger.warning("Wall classification post-processing failed: %s", postproc_exc)


__all__ = [
    "create_wall_types",
    "create_wall_from_axis",
    "post_process_wall_classification",
]

