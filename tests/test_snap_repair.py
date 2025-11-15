from __future__ import annotations

from shapely.geometry import LineString, Polygon

from core.ml.postprocess_floorplan import NormalizedDet, WallAxis
from core.reconstruct.openings import OpeningAssignment, reproject_openings_to_snapped_axes


def test_reproject_opening_depth_matches_wall_thickness():
    # Opening roughly centered near the axis, oriented horizontally
    opening_poly = Polygon([(0, -50), (200, -50), (200, 50), (0, 50)])
    opening = NormalizedDet(
        doc=0,
        page=0,
        type="WINDOW",
        is_external=True,
        geom=opening_poly,
        attrs={},
    )
    axis = WallAxis(
        detection=NormalizedDet(doc=0, page=0, type="WALL", is_external=True, geom=Polygon([(0, -200), (1000, -200), (1000, 200), (0, 200)]), attrs={}),
        source_index=0,
        axis=LineString([(0, 0), (1000, 0)]),
        width_mm=240.0,
        length_mm=1000.0,
        centroid_mm=(500.0, 0.0),
        method="test",
        metadata={},
    )
    assignments = [OpeningAssignment(opening=opening, wall_index=0, distance_mm=0.0, axis_index=0)]
    reproject_openings_to_snapped_axes([opening], assignments, wall_axes=[axis], depth_equals_wall_thickness=True)
    result = opening.geom
    assert isinstance(result, Polygon)
    # depth equals wall thickness -> approx 240 across the normal direction
    # Check via minimum rotated rectangle dimensions
    rect = result.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = x2 - x1, y2 - y1
        edges.append((dx * dx + dy * dy) ** 0.5)
    edges.sort()
    depth = edges[0]
    assert 230.0 <= depth <= 250.0



