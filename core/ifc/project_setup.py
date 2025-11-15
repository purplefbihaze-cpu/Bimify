"""Project structure setup for IFC models."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import ifcopenshell
import ifcopenshell.api
from ifcopenshell.util import element as ifc_element_utils

from core.exceptions import SchemaValidationError


logger = logging.getLogger(__name__)


def create_project_structure(
    model: Any,
    *,
    project_name: str = "Bimify Project",
    storey_name: str = "EG",
    storey_elevation: float = 0.0,
) -> tuple[Any, Any, Any, Any]:
    """Create IFC project structure (Project, Site, Building, Storey).
    
    Args:
        model: IFC model instance.
        project_name: Name of the project.
        storey_name: Name of the building storey.
        storey_elevation: Elevation of the storey in millimeters.
    
    Returns:
        Tuple of (project, site, building, storey) entities.
    """
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name=project_name)
    
    # Set up units
    ifcopenshell.api.run(
        "unit.assign_unit",
        model,
        length={
            "is_metric": True,
            "raw": "MILLIMETERS",
        },
    )
    
    # Create contexts
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=context,
    )
    
    # Set precision
    # Note: Precision for IfcGeometricRepresentationSubContext is DERIVED in IFC4
    # It is automatically inherited from the parent context, so we only set it for the main context
    precision_target = 1e-6
    if context is not None:
        try:
            current_precision = getattr(context, "Precision", None)
        except Exception:
            current_precision = None
        if current_precision is None or (isinstance(current_precision, (int, float)) and current_precision > precision_target):
            try:
                context.Precision = float(precision_target)
            except Exception:
                try:
                    setattr(context, "Precision", float(precision_target))
                except Exception:
                    pass
    # Do NOT set Precision for body (SubContext) - it is DERIVED and inherited from parent context
    
    # Create spatial structure
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run(
        "root.create_entity",
        model,
        ifc_class="IfcBuildingStorey",
        name=storey_name,
    )
    
    try:
        storey.Elevation = float(storey_elevation)
    except Exception:
        storey.Elevation = 0.0
    
    # Assign spatial hierarchy
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])
    
    # Set up placements
    def _make_direction(x: float, y: float, z: float):
        return model.create_entity("IfcDirection", DirectionRatios=(float(x), float(y), float(z)))

    def _make_point(x: float, y: float, z: float = 0.0):
        return model.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y), float(z)))

    def _make_local_placement(rel_to, location=(0.0, 0.0, 0.0)):
        loc = _make_point(*location)
        axis = _make_direction(0.0, 0.0, 1.0)
        ref = _make_direction(1.0, 0.0, 0.0)
        placement_3d = model.create_entity("IfcAxis2Placement3D", Location=loc, Axis=axis, RefDirection=ref)
        return model.create_entity("IfcLocalPlacement", PlacementRelTo=rel_to, RelativePlacement=placement_3d)

    def _ensure_spatial_placement(spatial, rel_to, location=(0.0, 0.0, 0.0)):
        if getattr(spatial, "ObjectPlacement", None) is None:
            spatial.ObjectPlacement = _make_local_placement(rel_to, location)

    _ensure_spatial_placement(site, None)
    _ensure_spatial_placement(building, getattr(site, "ObjectPlacement", None))
    _ensure_spatial_placement(storey, getattr(building, "ObjectPlacement", None), (0.0, 0.0, storey_elevation))
    
    return project, site, building, storey


def setup_units_complete(model: Any, project: Any | None = None) -> Any:
    """Assign a comprehensive unit set matching the template IFC.

    This recreates the unit assignment observed in the reference template:
    - Base SI units for length, area, volume, time, mass, temperature, luminous intensity
    - Conversion-based unit for degrees
    - Monetary unit in EUR
    - Derived units for thermal conductance, specific heat capacity, mass density

    Args:
        model: IFC model instance.
        project: Optional IfcProject entity; if omitted, the first project in the model is used.

    Returns:
        The created IfcUnitAssignment instance.
    """

    project_entity = project or next(iter(model.by_type("IfcProject") or []), None)
    if project_entity is None:
        raise ValueError("No IfcProject entity available to assign units")

    # Base SI units
    length_m = model.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    area_m2 = model.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
    volume_m3 = model.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    plane_angle_radian = model.create_entity(
        "IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN"
    )
    solid_angle_sr = model.create_entity(
        "IfcSIUnit", UnitType="SOLIDANGLEUNIT", Name="STERADIAN"
    )
    time_second = model.create_entity("IfcSIUnit", UnitType="TIMEUNIT", Name="SECOND")
    mass_gram = model.create_entity("IfcSIUnit", UnitType="MASSUNIT", Name="GRAM")
    temperature_celsius = model.create_entity(
        "IfcSIUnit", UnitType="THERMODYNAMICTEMPERATUREUNIT", Name="DEGREE_CELSIUS"
    )
    luminous_lumen = model.create_entity(
        "IfcSIUnit", UnitType="LUMINOUSINTENSITYUNIT", Name="LUMEN"
    )

    # Supporting SI units for derived units
    power_watt = model.create_entity("IfcSIUnit", UnitType="POWERUNIT", Name="WATT")
    length_m_for_derived = model.create_entity(
        "IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE"
    )
    temperature_kelvin = model.create_entity(
        "IfcSIUnit", UnitType="THERMODYNAMICTEMPERATUREUNIT", Name="KELVIN"
    )
    energy_joule = model.create_entity("IfcSIUnit", UnitType="ENERGYUNIT", Name="JOULE")
    mass_kilogram = model.create_entity(
        "IfcSIUnit", UnitType="MASSUNIT", Prefix="KILO", Name="GRAM"
    )

    # Conversion-based unit for degrees
    # Use ifcopenshell API to create conversion-based unit (handles IfcMeasureWithUnit correctly)
    degree_unit = None
    try:
        degree_unit = ifcopenshell.api.run(
            "unit.add_conversion_based_unit",
            model,
            name="DEGREE",
            unit_type="PLANEANGLEUNIT",
            conversion_factor=0.0174532925199,  # radians per degree
            conversion_unit=plane_angle_radian,
        )
    except (AttributeError, TypeError, ValueError, Exception) as api_exc:
        # Fallback: create manually if API fails
        logger.warning(f"unit.add_conversion_based_unit API failed: {api_exc}, creating manually")
        zero_dim = model.create_entity(
            "IfcDimensionalExponents",
            LengthExponent=0,
            MassExponent=0,
            TimeExponent=0,
            ElectricCurrentExponent=0,
            ThermodynamicTemperatureExponent=0,
            AmountOfSubstanceExponent=0,
            LuminousIntensityExponent=0,
        )
        # Create IfcMeasureWithUnit - ValueComponent should be a float for plane angle
        # Some ifcopenshell versions require explicit type wrapping
        degree_measure = None
        try:
            # Try with float directly (most common case)
            degree_measure = model.create_entity(
                "IfcMeasureWithUnit",
                ValueComponent=0.0174532925199,
                UnitComponent=plane_angle_radian,
            )
        except (TypeError, ValueError) as create_exc:
            # If that fails, try wrapping the value in a tuple to indicate it's a plane angle measure
            # In ifcopenshell, SELECT types can sometimes be passed as (type_name, value)
            try:
                degree_measure = model.create_entity(
                    "IfcMeasureWithUnit",
                    ValueComponent=(0.0174532925199,),  # Tuple might help with type inference
                    UnitComponent=plane_angle_radian,
                )
            except (TypeError, ValueError) as tuple_exc:
                # Last resort: skip degree unit and use radian only
                logger.error(f"Failed to create IfcMeasureWithUnit for degrees: {create_exc}, {tuple_exc}, using radian as fallback")
                degree_unit = plane_angle_radian  # Use radian as fallback
        
        if degree_unit is None and degree_measure is not None:
            # Create degree_unit manually if we have degree_measure
            degree_unit = model.create_entity(
                "IfcConversionBasedUnit",
                Dimensions=zero_dim,
                UnitType="PLANEANGLEUNIT",
                Name="DEGREE",
                ConversionFactor=degree_measure,
            )
        elif degree_unit is None:
            # If we couldn't create degree_measure either, use radian
            degree_unit = plane_angle_radian

    # Derived units mirroring the template
    thermal_conductance_elements: Sequence[Any] = (
        model.create_entity("IfcDerivedUnitElement", Unit=power_watt, Exponent=1),
        model.create_entity("IfcDerivedUnitElement", Unit=length_m_for_derived, Exponent=-1),
        model.create_entity("IfcDerivedUnitElement", Unit=temperature_kelvin, Exponent=-1),
    )
    thermal_conductance = model.create_entity(
        "IfcDerivedUnit",
        UnitType="THERMALCONDUCTANCEUNIT",
        Elements=thermal_conductance_elements,
    )

    specific_heat_elements: Sequence[Any] = (
        model.create_entity("IfcDerivedUnitElement", Unit=energy_joule, Exponent=1),
        model.create_entity("IfcDerivedUnitElement", Unit=mass_kilogram, Exponent=-1),
        model.create_entity("IfcDerivedUnitElement", Unit=temperature_kelvin, Exponent=-1),
    )
    specific_heat_capacity = model.create_entity(
        "IfcDerivedUnit",
        UnitType="SPECIFICHEATCAPACITYUNIT",
        Elements=specific_heat_elements,
    )

    mass_density_elements: Sequence[Any] = (
        model.create_entity("IfcDerivedUnitElement", Unit=mass_kilogram, Exponent=1),
        model.create_entity("IfcDerivedUnitElement", Unit=volume_m3, Exponent=-1),
    )
    mass_density = model.create_entity(
        "IfcDerivedUnit",
        UnitType="MASSDENSITYUNIT",
        Elements=mass_density_elements,
    )

    monetary_eur = model.create_entity("IfcMonetaryUnit", Currency="EUR")

    unit_assignment = model.create_entity(
        "IfcUnitAssignment",
        Units=(
            length_m,
            area_m2,
            volume_m3,
            degree_unit,
            solid_angle_sr,
            monetary_eur,
            time_second,
            mass_gram,
            temperature_celsius,
            luminous_lumen,
            thermal_conductance,
            specific_heat_capacity,
            mass_density,
        ),
    )

    project_entity.UnitsInContext = unit_assignment
    return unit_assignment


def setup_geometric_context_v2(
    model: Any,
    *,
    precision: float = 1.0e-5,
    true_north: Sequence[float] = (0.0, 1.0),
) -> tuple[Any, Any]:
    """Ensure geometric contexts match the template IFC configuration."""

    def _direction(values: Sequence[float]) -> Any:
        return model.create_entity("IfcDirection", DirectionRatios=tuple(float(v) for v in values))

    def _point(values: Sequence[float]) -> Any:
        coords = list(values) + [0.0] * (3 - len(values))
        return model.create_entity("IfcCartesianPoint", Coordinates=tuple(float(c) for c in coords[:3]))

    def _axis_placement() -> Any:
        return model.create_entity(
            "IfcAxis2Placement3D",
            Location=_point((0.0, 0.0, 0.0)),
            Axis=_direction((0.0, 0.0, 1.0)),
            RefDirection=_direction((1.0, 0.0, 0.0)),
        )

    contexts = model.by_type("IfcGeometricRepresentationContext")
    context = contexts[0] if contexts else None
    if context is None:
        context = model.create_entity(
            "IfcGeometricRepresentationContext",
            ContextIdentifier=None,
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=float(precision),
            WorldCoordinateSystem=_axis_placement(),
            TrueNorth=_direction(true_north),
        )
    else:
        context.ContextType = "Model"
        context.CoordinateSpaceDimension = 3
        context.Precision = float(precision)
        context.WorldCoordinateSystem = _axis_placement()
        context.TrueNorth = _direction(true_north)

    subcontexts = model.by_type("IfcGeometricRepresentationSubContext")
    body_context = next(
        (sc for sc in subcontexts if (getattr(sc, "ContextIdentifier", "") or "").lower() == "body"),
        None,
    )
    if body_context is None:
        # Note: Precision is DERIVED in IFC4 for IfcGeometricRepresentationSubContext
        # It is automatically inherited from the parent context, so we don't set it here
        body_context = model.create_entity(
            "IfcGeometricRepresentationSubContext",
            ContextIdentifier="Body",
            ContextType="Model",
            TargetView="MODEL_VIEW",
            ParentContext=context,
            # Precision is DERIVED - do not set it, it will be inherited from parent context
        )
    else:
        body_context.ContextType = "Model"
        body_context.TargetView = "MODEL_VIEW"
        body_context.ParentContext = context
        # Precision is DERIVED - do not set it, it will be inherited from parent context

    return context, body_context


def setup_calibration(
    model: Any,
    project: Any,
    *,
    calibration: dict[str, Any] | None = None,
    px_per_mm: float | None = None,
) -> None:
    """Set up calibration property set for the project.
    
    Args:
        model: IFC model instance.
        project: IfcProject entity.
        calibration: Calibration dictionary with calibration data.
        px_per_mm: Pixels per millimeter ratio.
    """
    if not calibration:
        return
    
    try:
        calib_pset = ifcopenshell.api.run("pset.add_pset", model, product=project, name="Bimify_ProjectCalibration")
        raw_a = calibration.get("point_a") if isinstance(calibration.get("point_a"), (list, tuple)) else [0.0, 0.0]
        raw_b = calibration.get("point_b") if isinstance(calibration.get("point_b"), (list, tuple)) else [0.0, 0.0]
        point_a = list(raw_a)[:2]
        point_b = list(raw_b)[:2]
        if len(point_a) < 2:
            point_a.extend([0.0] * (2 - len(point_a)))
        if len(point_b) < 2:
            point_b.extend([0.0] * (2 - len(point_b)))
        properties = {
            "ScalePxPerMm": float(calibration.get("px_per_mm", px_per_mm or 0.0)),
            "PixelDistance": float(calibration.get("pixel_distance", 0.0)),
            "RealDistanceMm": float(calibration.get("real_distance_mm", 0.0)),
            "PointAXpx": float(point_a[0]),
            "PointAYpx": float(point_a[1]),
            "PointBXpx": float(point_b[0]),
            "PointBYpx": float(point_b[1]),
            "Unit": str(calibration.get("unit", "mm")),
        }
        ifcopenshell.api.run("pset.edit_pset", model, pset=calib_pset, properties=properties)
    except Exception:
        pass


def setup_georeferencing(
    model: Any,
    project: Any,
    *,
    calibration: dict[str, Any] | None = None,
    storey_elevation: float = 0.0,
    is_ifc2x3: bool = False,
) -> None:
    """Set up georeferencing for the IFC model.
    
    Args:
        model: IFC model instance.
        project: IfcProject entity.
        calibration: Optional calibration data for coordinates.
        storey_elevation: Elevation of the storey.
        is_ifc2x3: Whether using IFC2X3 schema.
    """
    try:
        ifcopenshell.api.run("georeference.add_georeferencing", model)
        
        # Optional: Set coordinates from calibration data if available
        if calibration:
            try:
                # Extract potential real-world coordinates from calibration
                # This is a simplified approach - in production, you'd have actual GPS coordinates
                point_a = calibration.get("point_a")
                point_b = calibration.get("point_b")
                
                # If calibration has real-world coordinates, use them
                # Otherwise, use default/zero coordinates
                eastings = 0.0
                northings = 0.0
                orthogonal_height = float(storey_elevation) if storey_elevation else 0.0
                
                # Try to extract coordinates if available in calibration
                if isinstance(point_a, (list, tuple)) and len(point_a) >= 2:
                    # If point_a contains real-world coordinates, use them
                    # For now, we'll use default values as calibration typically contains pixel coordinates
                    pass
                
                # Edit georeferencing with default/calibration-based coordinates
                # For IFC2X3, this uses ePSet_MapConversion
                # For IFC4, this uses IfcMapConversion entity
                if is_ifc2x3:
                    # IFC2X3: Edit via Property Set
                    try:
                        conversion_pset = ifc_element_utils.get_pset(project, "ePSet_MapConversion")
                        if conversion_pset:
                            ifcopenshell.api.run("pset.edit_pset", model, pset=conversion_pset, properties={
                                "Eastings": eastings,
                                "Northings": northings,
                                "OrthogonalHeight": orthogonal_height,
                            })
                    except Exception:
                        pass
                else:
                    # IFC4: Edit via IfcMapConversion entity
                    ifcopenshell.api.run("georeference.edit_georeferencing", model,
                        coordinate_operation={
                            "Eastings": eastings,
                            "Northings": northings,
                            "OrthogonalHeight": orthogonal_height,
                        },
                        projected_crs={"Name": "EPSG:3857"}  # Default: Web Mercator
                    )
            except Exception as geo_exc:
                logger.debug("Failed to set georeferencing coordinates from calibration: %s", geo_exc)
    except Exception as geo_init_exc:
        logger.warning("Failed to initialize georeferencing: %s", geo_init_exc)


def setup_classification(
    model: Any,
    *,
    classification_name: str = "Uniclass 2015",
) -> tuple[Any, dict[str, Any]]:
    """Set up classification system for building elements.
    
    Args:
        model: IFC model instance.
        classification_name: Name of classification system (e.g., "Uniclass 2015").
    
    Returns:
        Tuple of (classification entity, dictionary of classification references).
    """
    classification = None
    classification_refs = {}
    
    try:
        # Create a generic classification (can be customized for Uniclass/OmniClass)
        classification = ifcopenshell.api.run("classification.add_classification", model, classification=classification_name)
        
        # Add classification references for common building elements
        # These can be customized based on actual classification codes
        
        # Wall classification
        try:
            wall_class_ref = ifcopenshell.api.run(
                "classification.add_reference",
                model,
                reference=classification,
                identification="Pr_20_70_36_02",  # Uniclass 2015 code for external walls
                name="External Wall",
            )
            classification_refs["WALL_EXTERNAL"] = wall_class_ref
        except Exception:
            pass
        
        try:
            wall_class_ref = ifcopenshell.api.run(
                "classification.add_reference",
                model,
                reference=classification,
                identification="Pr_20_70_36_03",  # Uniclass 2015 code for internal walls
                name="Internal Wall",
            )
            classification_refs["WALL_INTERNAL"] = wall_class_ref
        except Exception:
            pass
        
        # Door classification
        try:
            door_class_ref = ifcopenshell.api.run(
                "classification.add_reference",
                model,
                reference=classification,
                identification="Pr_20_70_37_02",  # Uniclass 2015 code for doors
                name="Door",
            )
            classification_refs["DOOR"] = door_class_ref
        except Exception:
            pass
        
        # Window classification
        try:
            window_class_ref = ifcopenshell.api.run(
                "classification.add_reference",
                model,
                reference=classification,
                identification="Pr_20_70_37_01",  # Uniclass 2015 code for windows
                name="Window",
            )
            classification_refs["WINDOW"] = window_class_ref
        except Exception:
            pass
        
        # Space classification
        try:
            space_class_ref = ifcopenshell.api.run(
                "classification.add_reference",
                model,
                reference=classification,
                identification="SL_20_70_95",  # Uniclass 2015 code for spaces
                name="Space",
            )
            classification_refs["SPACE"] = space_class_ref
        except Exception:
            pass
        
        logger.info("Classification system initialized: %s", classification.Name if classification else "Unknown")
    except Exception as classification_exc:
        logger.warning("Failed to initialize classification: %s", classification_exc)
        classification = None
        classification_refs = {}
    
    return classification, classification_refs


def setup_owner_history(
    model: Any,
    *,
    is_ifc2x3: bool = False,
    owner_org_name: str = "BIMMATRIX",
    owner_org_identification: str | None = None,
    app_identifier: str = "BIMMATRIX",
    app_full_name: str = "BIMMATRIX IFC Exporter",
    app_version: str = "1.0",
    person_identification: str = "BIMMATRIX_USER",
    person_given_name: str = "BIMMATRIX",
    person_family_name: str = "User",
) -> None:
    """Set up owner history (application, user, person, organisation) for IFC model.
    
    This is required for IFC2X3 (mandatory) and recommended for IFC4.
    Configures owner settings so that owner.create_owner_history can be called
    without errors.
    
    Creates entities directly via model.create_entity to avoid bootstrap issues
    where the API requires owner history to already exist.
    
    Args:
        model: IFC model instance.
        is_ifc2x3: Whether using IFC2X3 schema (owner history is mandatory).
        owner_org_name: Name of the organization (required).
        owner_org_identification: Identification of the organization (optional, not in IFC2X3).
        app_identifier: Application identifier.
        app_full_name: Full name of the application.
        app_version: Version of the application.
        person_identification: Identification of the person (optional, not in IFC2X3).
        person_given_name: Given name of the person.
        person_family_name: Family name of the person.
    """
    try:
        # Helper function to check if an entity type supports an attribute
        def _has_attribute(entity_type: str, attr_name: str) -> bool:
            """Check if an entity type supports a specific attribute in the current schema."""
            try:
                schema = model.schema
                decl = schema.declaration_by_name(entity_type)
                if decl is None:
                    return False
                entity_decl = decl.as_entity()
                if entity_decl is None:
                    return False
                # Check if attribute exists by trying to get its index
                # attribute_index returns -1 if not found, or raises exception
                try:
                    attr_idx = entity_decl.attribute_index(attr_name)
                    return attr_idx >= 0
                except Exception:
                    # If attribute_index raises, attribute doesn't exist
                    return False
            except Exception:
                # If schema lookup fails, assume attribute doesn't exist
                # This is safer than trying to create test entities
                return False
        
        # Check if entities already exist in the model
        existing_orgs = model.by_type("IfcOrganization")
        existing_persons = model.by_type("IfcPerson")
        existing_users = model.by_type("IfcPersonAndOrganization")
        existing_apps = model.by_type("IfcApplication")
        
        # Find or create organisation
        # In IFC2X3, IfcOrganization doesn't have Identification attribute
        organisation = None
        for org in existing_orgs:
            # Try to match by Identification if supported, otherwise by Name
            if not is_ifc2x3 and owner_org_identification:
                if getattr(org, "Identification", None) == owner_org_identification:
                    organisation = org
                    break
            if getattr(org, "Name", None) == owner_org_name:
                organisation = org
                break
        
        if organisation is None:
            # Build organization attributes based on schema support
            org_attrs: dict[str, Any] = {"Name": owner_org_name}
            # Only add Identification if schema supports it and value is provided
            if not is_ifc2x3 and owner_org_identification and _has_attribute("IfcOrganization", "Identification"):
                org_attrs["Identification"] = owner_org_identification
            
            organisation = model.create_entity("IfcOrganization", **org_attrs)
            logger.debug("Created IfcOrganization: %s", getattr(organisation, "Name", "Unknown"))
        else:
            logger.debug("Reusing existing IfcOrganization: %s", getattr(organisation, "Name", "Unknown"))
        
        # Find or create person
        # In IFC2X3, IfcPerson doesn't have Identification attribute
        person = None
        for p in existing_persons:
            # Try to match by Identification if supported, otherwise by name
            if not is_ifc2x3 and person_identification:
                if getattr(p, "Identification", None) == person_identification:
                    person = p
                    break
            if (getattr(p, "GivenName", None) == person_given_name and 
                getattr(p, "FamilyName", None) == person_family_name):
                person = p
                break
        
        if person is None:
            # Build person attributes based on schema support
            person_attrs: dict[str, Any] = {
                "FamilyName": person_family_name,
                "GivenName": person_given_name,
            }
            # Only add Identification if schema supports it and value is provided
            if not is_ifc2x3 and person_identification and _has_attribute("IfcPerson", "Identification"):
                person_attrs["Identification"] = person_identification
            
            person = model.create_entity("IfcPerson", **person_attrs)
            logger.debug("Created IfcPerson: %s", getattr(person, "GivenName", "Unknown"))
        else:
            logger.debug("Reusing existing IfcPerson: %s", getattr(person, "GivenName", "Unknown"))
        
        # Find or create user (person_and_organisation)
        user = None
        for u in existing_users:
            if (getattr(u, "ThePerson", None) == person and 
                getattr(u, "TheOrganization", None) == organisation):
                user = u
                break
        if user is None:
            user = model.create_entity(
                "IfcPersonAndOrganization",
                ThePerson=person,
                TheOrganization=organisation,
            )
            logger.debug("Created IfcPersonAndOrganization: id=%s", getattr(user, "id", "Unknown"))
        else:
            logger.debug("Reusing existing IfcPersonAndOrganization: id=%s", getattr(user, "id", "Unknown"))
        
        # Find or create application
        application = None
        for app in existing_apps:
            if getattr(app, "ApplicationIdentifier", None) == app_identifier:
                application = app
                break
        if application is None:
            application = model.create_entity(
                "IfcApplication",
                ApplicationDeveloper=organisation,
                Version=app_version,
                ApplicationFullName=app_full_name,
                ApplicationIdentifier=app_identifier,
            )
            logger.debug("Created IfcApplication: %s", getattr(application, "ApplicationFullName", "Unknown"))
        else:
            logger.debug("Reusing existing IfcApplication: %s", getattr(application, "ApplicationFullName", "Unknown"))
        
        # Configure owner settings so owner.create_owner_history works automatically
        # The settings are module-level, so we store the entities in closures
        # Note: These settings are shared across all models, but since we create
        # entities in each model, we return the entities we just created
        ifcopenshell.api.owner.settings.get_user = lambda file: user
        ifcopenshell.api.owner.settings.get_application = lambda file: application
        
        # Verify entities are valid
        if not hasattr(user, "id") or not hasattr(application, "id"):
            raise ValueError("Created entities are missing required attributes")
        
        # Verify the settings are working by testing them
        test_user = ifcopenshell.api.owner.settings.get_user(model)
        test_app = ifcopenshell.api.owner.settings.get_application(model)
        if test_user is None:
            raise ValueError(f"Owner settings get_user returned None (expected user entity)")
        if test_app is None:
            raise ValueError(f"Owner settings get_application returned None (expected application entity)")
        if test_user.id() != user.id():
            raise ValueError(f"Owner settings returned wrong user: expected {user.id()}, got {test_user.id()}")
        if test_app.id() != application.id():
            raise ValueError(f"Owner settings returned wrong application: expected {application.id()}, got {test_app.id()}")
        
        # Test that we can actually create an owner history (this is the real test)
        # If API call fails, try manual creation as fallback
        test_owner_history = None
        try:
            test_owner_history = ifcopenshell.api.run("owner.create_owner_history", model)
            if test_owner_history is None and is_ifc2x3:
                raise ValueError("Failed to create test owner history (required for IFC2X3)")
            logger.debug("Successfully created test owner history via API: %s", getattr(test_owner_history, "id", "None") if test_owner_history else "None")
        except Exception as api_exc:
            # Fallback: create owner history manually
            logger.warning("API owner.create_owner_history failed, creating manually: %s", api_exc)
            try:
                import time
                test_owner_history = model.create_entity(
                    "IfcOwnerHistory",
                    OwningUser=user,
                    OwningApplication=application,
                    ChangeAction="ADDED",
                    CreationDate=int(time.time()),
                )
                logger.debug("Successfully created test owner history manually: %s", getattr(test_owner_history, "id", "Unknown"))
            except Exception as manual_exc:
                if is_ifc2x3:
                    raise RuntimeError(f"Failed to create owner history (required for IFC2X3): API error: {api_exc}, Manual error: {manual_exc}") from manual_exc
                else:
                    logger.warning("Failed to create owner history (optional for IFC4): API error: %s, Manual error: %s", api_exc, manual_exc)
                    # For IFC4, we can continue without owner history, but log the issue
                    raise RuntimeError(f"Failed to create test owner history: {api_exc}") from api_exc
        
        logger.info("Owner history setup completed: application=%s, user=%s", 
                   getattr(application, "ApplicationFullName", "Unknown"),
                   getattr(user, "id", "Unknown"))
    except Exception as owner_exc:
        # For IFC2X3, owner history is mandatory, so we should raise the error
        # For IFC4, it's optional but recommended
        error_msg = f"Failed to setup owner history: {owner_exc}"
        if is_ifc2x3:
            logger.error(error_msg + " (required for IFC2X3)")
            raise RuntimeError(error_msg) from owner_exc
        else:
            logger.warning(error_msg + " (optional for IFC4, but may cause issues)")
            # Re-raise to ensure we know about the problem
            raise RuntimeError(error_msg) from owner_exc


def setup_owner_history_v2(
    model: Any,
    *,
    organization_name: str = "GRAPHISOFT",
    organization_identification: str | None = None,
    application_identifier: str = "IFC2x3 add-on version: 4009 GER FULL",
    application_full_name: str = "ARCHICAD-64",
    application_version: str = "20.0.0",
    person_family_name: str = "Nicht definiert",
    person_given_name: str | None = None,
    person_identification: str | None = None,
    creation_date: int | None = None,
    change_action: str = "ADDED",
) -> Any:
    """Configure owner history following the template IFC structure.
    
    This function is defensive and checks for existing valid OwnerHistory first.
    It ensures proper string truncation and sets required attributes.
    """
    try:
        # ERSTE Prüfung: Existiert bereits eine gültige OwnerHistory?
        existing_histories = model.by_type("IfcOwnerHistory")
        if existing_histories:
            # Prüfe, ob alle Attribute geladen sind
            hist = existing_histories[0]
            required_attrs = ["OwningUser", "OwningApplication", "State", "ChangeAction"]
            if all(hasattr(hist, attr) and getattr(hist, attr, None) is not None for attr in required_attrs):
                # Validate that the attributes are actually valid entities/values
                if (getattr(hist, "OwningUser", None) is not None and 
                    getattr(hist, "OwningApplication", None) is not None):
                    logger.debug("OwnerHistory bereits vollständig initialisiert")
                    # Ensure it's linked to project
                    try:
                        projects = model.by_type("IfcProject")
                        if projects and not hasattr(projects[0], "OwnerHistory") or getattr(projects[0], "OwnerHistory", None) is None:
                            projects[0].OwnerHistory = hist
                    except Exception:
                        pass
                    return hist

        effective_given_name = person_given_name or person_family_name

        # Truncate strings to IFC limits
        # ApplicationIdentifier: max 22 chars (IfcIdentifier is max 255, but be conservative)
        truncated_app_identifier = application_identifier[:22] if application_identifier else "bimify"
        # ApplicationFullName: max 100 chars
        truncated_app_full_name = application_full_name[:100] if application_full_name else "Bimify IFC Exporter"
        # Version: max 20 chars
        truncated_app_version = application_version[:20] if application_version else "1.0"

        setup_owner_history(
            model,
            is_ifc2x3=False,
            owner_org_name=organization_name,
            owner_org_identification=organization_identification,
            app_identifier=truncated_app_identifier,
            app_full_name=truncated_app_full_name,
            app_version=truncated_app_version,
            person_identification=person_identification or "",
            person_given_name=effective_given_name,
            person_family_name=person_family_name,
        )

        user = ifcopenshell.api.owner.settings.get_user(model)
        application = ifcopenshell.api.owner.settings.get_application(model)
        if user is None or application is None:
            raise SchemaValidationError("Owner history settings not configured correctly: user or application is None")

        # Check again for existing history after setup
        owner_history = None
        for history in model.by_type("IfcOwnerHistory"):
            if (getattr(history, "OwningUser", None) == user and 
                getattr(history, "OwningApplication", None) == application):
                owner_history = history
                break

        # Explizite, defensive Erzeugung
        if owner_history is None:
            # Use user and application from setup_owner_history (already created)
            # Erzeuge Historie mit korrekten Defaults
            owner_history = model.create_entity(
                "IfcOwnerHistory",
                OwningUser=user,
                OwningApplication=application,
                State="READWRITE",
                ChangeAction=change_action,
                CreationDate=creation_date or int(time.time()),
            )
        else:
            # Update existing history
            owner_history.ChangeAction = change_action
            owner_history.CreationDate = creation_date or getattr(owner_history, "CreationDate", int(time.time()))
            # Ensure State is set
            if not hasattr(owner_history, "State") or getattr(owner_history, "State", None) is None:
                try:
                    owner_history.State = "READWRITE"
                except Exception:
                    pass

        # Setze als globale Historie des Projekts
        try:
            projects = model.by_type("IfcProject")
            if projects:
                project = projects[0]
                if not hasattr(project, "OwnerHistory") or getattr(project, "OwnerHistory", None) is None:
                    project.OwnerHistory = owner_history
        except Exception as e:
            logger.warning(f"Could not link OwnerHistory to IfcProject: {e}")

        return owner_history

    except RuntimeError as e:
        raise SchemaValidationError(f"OwnerHistory-Setup fehlgeschlagen: {e}") from e
    except Exception as e:
        raise SchemaValidationError(f"OwnerHistory-Setup fehlgeschlagen: {e}") from e


__all__ = [
    "create_project_structure",
    "setup_calibration",
    "setup_georeferencing",
    "setup_classification",
    "setup_owner_history",
]

