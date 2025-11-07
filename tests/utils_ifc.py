from __future__ import annotations

from pathlib import Path

import ifcopenshell
import ifcopenshell.api


def _create_local_placement(
    model: ifcopenshell.file,
    rel_to,
    location: tuple[float, float, float],
) -> ifcopenshell.entity_instance:
    loc = model.create_entity("IfcCartesianPoint", Coordinates=location)
    axis = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    ref_dir = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    placement = model.create_entity("IfcAxis2Placement3D", Location=loc, Axis=axis, RefDirection=ref_dir)
    return model.create_entity("IfcLocalPlacement", PlacementRelTo=rel_to, RelativePlacement=placement)


def _create_wall(
    model: ifcopenshell.file,
    body_context,
    storey,
    name: str,
    translation: tuple[float, float, float],
) -> ifcopenshell.entity_instance:
    wall = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcWallStandardCase", name=name)
    ifcopenshell.api.run("spatial.assign_container", model, products=[wall], relating_structure=storey)
    wall.ObjectPlacement = _create_local_placement(model, storey.ObjectPlacement, translation)

    points = [
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, -50.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(500.0, -50.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(500.0, 50.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, 50.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, -50.0)),
    ]
    polyline = model.create_entity("IfcPolyline", Points=points)
    profile = model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline)
    axis = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    ref_dir = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    position = model.create_entity("IfcAxis2Placement3D", Location=origin, Axis=axis, RefDirection=ref_dir)
    extrude_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=position,
        ExtrudedDirection=extrude_dir,
        Depth=3000.0,
    )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )
    product_def = model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])
    wall.Representation = product_def
    pset_common = ifcopenshell.api.run("pset.add_pset", model, product=wall, name="Pset_WallCommon")
    ifcopenshell.api.run("pset.edit_pset", model, pset=pset_common, properties={"IsExternal": True})
    return wall


def build_sample_ifc(path: Path) -> None:
    model = ifcopenshell.api.run("project.create_file")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="HottCAD Test")
    ifcopenshell.api.run("unit.assign_unit", model, length={"is_metric": True, "raw": "MILLIMETERS"})
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=context,
    )

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="EG")

    site.ObjectPlacement = _create_local_placement(model, None, (0.0, 0.0, 0.0))
    building.ObjectPlacement = _create_local_placement(model, site.ObjectPlacement, (0.0, 0.0, 0.0))
    storey.ObjectPlacement = _create_local_placement(model, building.ObjectPlacement, (0.0, 0.0, 0.0))

    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])

    _create_wall(model, body, storey, "Wall_A", (0.0, 0.0, 0.0))
    _create_wall(model, body, storey, "Wall_B", (1000.0, 0.0, 0.0))

    space = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSpace", name="Space_1")
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[space])
    space.ObjectPlacement = _create_local_placement(model, storey.ObjectPlacement, (0.0, 0.0, 0.0))

    # Add simple prismatic geometry for the space to satisfy strict validation
    space_points = [
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, -500.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(1500.0, -500.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(1500.0, 500.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, 500.0)),
        model.create_entity("IfcCartesianPoint", Coordinates=(-500.0, -500.0)),
    ]
    space_polyline = model.create_entity("IfcPolyline", Points=space_points)
    space_profile = model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=space_polyline)
    space_axis = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    space_ref_dir = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    space_origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    space_position = model.create_entity("IfcAxis2Placement3D", Location=space_origin, Axis=space_axis, RefDirection=space_ref_dir)
    space_extrude_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    space_solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=space_profile,
        Position=space_position,
        ExtrudedDirection=space_extrude_dir,
        Depth=3000.0,
    )
    space_shape = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[space_solid],
    )
    space.Representation = model.create_entity("IfcProductDefinitionShape", Representations=[space_shape])

    path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(path))


__all__ = ["build_sample_ifc"]


