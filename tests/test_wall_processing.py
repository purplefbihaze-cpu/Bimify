from __future__ import annotations

import math

import pytest
pytest.importorskip("ifcopenshell")
import ifcopenshell
from shapely.geometry import LineString, Polygon, MultiPolygon

from core.ifc.build_ifc43_model import collect_wall_polygons, write_ifc_with_spaces
from core.ml.postprocess_floorplan import NormalizedDet, WallAxis, estimate_wall_axes_and_thickness
from core.validate.reconstruction_validation import generate_validation_report


def _make_l_shape_polygon() -> Polygon:
    return Polygon(
        [
            (0.0, 0.0),
            (0.0, 200.0),
            (80.0, 200.0),
            (80.0, 80.0),
            (200.0, 80.0),
            (200.0, 0.0),
        ]
    )


def test_estimate_wall_axes_splits_l_shape() -> None:
    det = NormalizedDet(
        doc=0,
        page=0,
        type="WALL",
        is_external=True,
        geom=_make_l_shape_polygon(),
        attrs={"confidence": 0.95},
    )

    axes = estimate_wall_axes_and_thickness([det])

    # Skeleton-basierte Aufteilung kann je nach Filter mehrere Segmente ergeben.
    assert len(axes) >= 2, "Expected L-shaped wall to be split into multiple axes"
    longest = max(axis.axis.length for axis in axes)
    assert longest >= 100.0, "At least one segment should capture a primary wall arm"
    remaining = sum(axis.axis.length for axis in axes if axis.axis.length < longest)
    assert remaining >= 60.0, "Secondary arm should still be represented by remaining segments"


def test_generate_validation_report_basic(tmp_path) -> None:
    wall_polygon = Polygon([(0.0, 0.0), (0.0, 200.0), (3000.0, 200.0), (3000.0, 0.0)])
    det = NormalizedDet(
        doc=0,
        page=0,
        type="WALL",
        is_external=False,
        geom=wall_polygon,
        attrs={"confidence": 0.88},
    )

    axis = LineString([(0.0, 0.0), (3000.0, 0.0)])
    axis_info = WallAxis(
        detection=det,
        source_index=0,
        axis=axis,
        width_mm=200.0,
        length_mm=float(axis.length),
        centroid_mm=(1500.0, 0.0),
        method="skeleton",
        metadata={"axis_local_index": 0.0, "wall_index": 1.0},
    )

    report = generate_validation_report([det], [axis_info], tmp_path / "dummy.ifc", update_ifc=False)

    assert report["summary"]["total_walls"] == 1
    wall_entry = report["walls"][0]
    assert wall_entry["status"] in {"PASS", "WARN"}
    assert wall_entry["iou_2d"] > 0.3
    assert wall_entry["score"] >= 60.0


def test_collect_wall_polygons_merges_components() -> None:
    geom = MultiPolygon(
        [
            Polygon([(0.0, 0.0), (0.0, 50.0), (10.0, 50.0), (10.0, 0.0)]),
            Polygon([(9.5, -2.0), (9.5, 52.0), (20.0, 52.0), (20.0, -2.0)]),
        ]
    )
    det = NormalizedDet(
        doc=0,
        page=0,
        type="WALL",
        is_external=True,
        geom=geom,
        attrs={"confidence": 0.9},
    )

    merged = collect_wall_polygons([det])

    assert 0 in merged
    cleaned = merged[0]
    assert isinstance(cleaned, Polygon)
    expected_area = geom.buffer(0).area
    assert cleaned.area == pytest.approx(expected_area)


def test_write_ifc_uses_polygon_profile(tmp_path) -> None:
    polygon = Polygon(
        [
            (0.0, 0.0),
            (80.0, 20.0),
            (90.0, 120.0),
            (10.0, 100.0),
        ]
    )
    det = NormalizedDet(
        doc=0,
        page=0,
        type="WALL",
        is_external=True,
        geom=polygon,
        attrs={"confidence": 0.95},
    )

    axis_line = LineString([(10.0, 10.0), (80.0, 95.0)])
    centroid = axis_line.interpolate(0.5, normalized=True)
    axis = WallAxis(
        detection=det,
        source_index=0,
        axis=axis_line,
        width_mm=200.0,
        length_mm=float(axis_line.length),
        centroid_mm=(float(centroid.x), float(centroid.y)),
        method="test",
        metadata={},
    )

    out_path = tmp_path / "polygon_wall.ifc"
    normalized = [det]
    wall_polygons = collect_wall_polygons(normalized)

    write_ifc_with_spaces(
        normalized,
        [],
        out_path,
        wall_axes=[axis],
        wall_polygons=wall_polygons,
        storey_height_mm=3000.0,
        door_height_mm=2100.0,
        window_height_mm=1000.0,
        window_head_elevation_mm=2000.0,
    )

    model = ifcopenshell.open(str(out_path))
    walls = model.by_type("IfcWallStandardCase")
    assert len(walls) == 1
    representations = walls[0].Representation.Representations
    assert representations
    swept_solid = representations[0].Items[0]
    profile = swept_solid.SweptArea
    assert profile.is_a("IfcArbitraryClosedProfileDef")

    coords = [tuple(pt.Coordinates[:2]) for pt in profile.OuterCurve.Points]
    # Remove duplicated closing point
    coords = coords[:-1]
    expected_coords = list(polygon.exterior.coords)[:-1]
    assert len(coords) == len(expected_coords)
    for actual, expected in zip(coords, expected_coords):
        assert actual[0] == pytest.approx(expected[0], abs=1e-3)
        assert actual[1] == pytest.approx(expected[1], abs=1e-3)


def test_topview_section_includes_window_at_cut(tmp_path) -> None:
    # Build simple wall and window opening
    wall_poly = Polygon([(0.0, 0.0), (0.0, 200.0), (3000.0, 200.0), (3000.0, 0.0)])
    wall = NormalizedDet(doc=0, page=0, type="WALL", is_external=True, geom=wall_poly, attrs={"confidence": 0.9})
    window_poly = Polygon([(1200.0, 10.0), (1800.0, 10.0), (1800.0, 190.0), (1200.0, 190.0)])
    win = NormalizedDet(doc=0, page=0, type="WINDOW", is_external=None, geom=window_poly, attrs={"confidence": 0.9})

    axis = LineString([(0.0, 100.0), (3000.0, 100.0)])
    centroid = axis.interpolate(0.5, normalized=True)
    wall_axis = WallAxis(
        detection=wall,
        source_index=0,
        axis=axis,
        width_mm=200.0,
        length_mm=float(axis.length),
        centroid_mm=(float(centroid.x), float(centroid.y)),
        method="test",
        metadata={},
    )

    out_path = tmp_path / "section.ifc"
    write_ifc_with_spaces(
        [wall, win],
        [],
        out_path,
        wall_axes=[wall_axis],
        wall_polygons=collect_wall_polygons([wall]),
        storey_height_mm=3000.0,
        door_height_mm=2100.0,
        window_height_mm=1000.0,
        window_head_elevation_mm=1900.0,
    )

    # Use topview helper to slice at 1000mm and at 700mm
    from core.vector.ifc_topview import build_topview_geojson

    geojson_cut = tmp_path / "cut_1000.geojson"
    build_topview_geojson(out_path, geojson_cut, section_elevation_mm=1000.0)
    data_cut = geojson_cut.read_text(encoding="utf-8")
    assert "WINDOW" in data_cut

    geojson_low = tmp_path / "cut_700.geojson"
    build_topview_geojson(out_path, geojson_low, section_elevation_mm=700.0)
    data_low = geojson_low.read_text(encoding="utf-8")
    assert "WINDOW" not in data_low

