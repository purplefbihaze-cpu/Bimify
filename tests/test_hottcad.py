from __future__ import annotations

import pytest

pytest.importorskip("ifcopenshell")
pytest.importorskip("shapely")

import ifcopenshell
import ifcopenshell.api

from core.ifc.hottcad_validator import validate_hottcad
from core.ifc.hottcad_simulate import simulate_hottcad
from tests.utils_ifc import build_sample_ifc


def test_validate_hottcad_basic(tmp_path):
    path = tmp_path / "sample.ifc"
    build_sample_ifc(path)

    result = validate_hottcad(path)

    assert result.metrics.wall_count == 2
    assert result.metrics.walls_with_rectangular_footprint == 2
    assert result.metrics.walls_with_constant_thickness == 2
    assert any(check.id == "walls-geometry" and check.status != "fail" for check in result.checks)
    assert isinstance(result.highlight_sets, list)


def test_simulate_hottcad_highlights(tmp_path):
    path = tmp_path / "sample.ifc"
    build_sample_ifc(path)

    result = simulate_hottcad(path)

    assert result.proposed["connects"], "Expected wall connection proposals"
    assert result.highlight_sets, "Expected highlight sets to be generated"
    first = result.highlight_sets[0]
    assert first.guids, "Highlight set should include GUIDs"
    assert first.product_ids, "Highlight set should include product IDs"
    assert len(first.product_ids) == len(first.guids)


def _get_check(result, check_id: str):
    return next(check for check in result.checks if check.id == check_id)


def test_validate_detects_missing_storey_elevation(tmp_path):
    path = tmp_path / "no_elevation.ifc"
    build_sample_ifc(path)
    model = ifcopenshell.open(str(path))
    storey = model.by_type("IfcBuildingStorey")[0]
    storey.Elevation = None
    model.write(str(path))

    result = validate_hottcad(path)
    structure_check = _get_check(result, "project-structure")
    assert structure_check.status == "fail"
    assert "storeys_missing_elevation" in structure_check.affected


def test_validate_detects_splitlevel_storeys(tmp_path):
    path = tmp_path / "splitlevel.ifc"
    build_sample_ifc(path)
    model = ifcopenshell.open(str(path))
    building = model.by_type("IfcBuilding")[0]
    base_storey = model.by_type("IfcBuildingStorey")[0]

    point = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    axis = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    ref_dir = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    placement = model.create_entity("IfcAxis2Placement3D", Location=point, Axis=axis, RefDirection=ref_dir)
    local = model.create_entity("IfcLocalPlacement", PlacementRelTo=base_storey.ObjectPlacement, RelativePlacement=placement)

    new_storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Split")
    new_storey.Elevation = float(base_storey.Elevation or 0.0) + 100.0
    new_storey.ObjectPlacement = local
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[new_storey])

    model.write(str(path))

    result = validate_hottcad(path)
    structure_check = _get_check(result, "project-structure")
    assert structure_check.status == "fail"
    assert any(
        key in structure_check.affected for key in ("storeys_splitlevel", "storeys_missing_elevation")
    )


def test_validate_detects_space_geometry_failure(tmp_path):
    path = tmp_path / "space_fail.ifc"
    build_sample_ifc(path)
    model = ifcopenshell.open(str(path))
    space = model.by_type("IfcSpace")[0]
    space.Representation = None
    model.write(str(path))

    result = validate_hottcad(path)
    topology_check = _get_check(result, "topology")
    assert topology_check.status == "fail"
    assert "spaces_missing_geometry" in topology_check.affected


def test_validate_detects_opening_geometry_failure(tmp_path):
    path = tmp_path / "opening_fail.ifc"
    build_sample_ifc(path)
    model = ifcopenshell.open(str(path))
    storey = model.by_type("IfcBuildingStorey")[0]

    opening = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcOpeningElement", name="Opening")
    ifcopenshell.api.run("spatial.assign_container", model, products=[opening], relating_structure=storey)
    # No geometry / relations assigned to trigger fail state
    model.write(str(path))

    result = validate_hottcad(path)
    openings_check = _get_check(result, "openings")
    assert openings_check.status == "fail"
    assert "openings_missing_geometry" in openings_check.affected or "openings_invalid_geometry" in openings_check.affected
