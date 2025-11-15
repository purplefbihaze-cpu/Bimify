"""Unit tests for GeometryProcessor chain."""

from shapely.geometry import Polygon

from core.ml.geometry_processor import GeometryProcessor, Simplifier, GridSnapper, OrthogonalEnforcer
from core.ml.pipeline_config import PipelineConfig


def test_simplifier():
    """Test Simplifier step."""
    config = PipelineConfig.default()
    simplifier = Simplifier()
    
    # Create a complex polygon
    complex_poly = Polygon([(0, 0), (10, 0), (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5)])
    pred = {"class": "wall", "confidence": 0.9}
    
    result = simplifier.process(pred, complex_poly, config)
    
    assert isinstance(result, Polygon)
    assert not result.is_empty
    # Simplified polygon should have fewer or equal points
    assert len(result.exterior.coords) <= len(complex_poly.exterior.coords)


def test_grid_snapper():
    """Test GridSnapper step."""
    config = PipelineConfig.default()
    config.enable_snap_to_grid = True
    snapper = GridSnapper()
    
    # Create polygon with non-grid coordinates
    poly = Polygon([(1.3, 2.7), (10.8, 2.7), (10.8, 5.2), (1.3, 5.2)])
    pred = {"class": "wall", "confidence": 0.9}
    
    result = snapper.process(pred, poly, config)
    
    assert isinstance(result, Polygon)
    assert not result.is_empty
    # Coordinates should be snapped to grid
    for x, y in result.exterior.coords:
        # Check that coordinates are multiples of grid size (within tolerance)
        assert abs(x % config.grid_size_walls) < 0.01 or abs(x % config.grid_size_walls - config.grid_size_walls) < 0.01


def test_orthogonal_enforcer():
    """Test OrthogonalEnforcer step."""
    config = PipelineConfig.default()
    config.enable_right_angle = True
    enforcer = OrthogonalEnforcer()
    
    # Create polygon with near-90° angles
    poly = Polygon([(0, 0), (10, 0.5), (10, 5), (0, 5)])
    pred = {"class": "wall", "confidence": 0.9}
    
    result = enforcer.process(pred, poly, config)
    
    assert isinstance(result, Polygon)
    assert not result.is_empty


def test_geometry_processor_chain():
    """Test GeometryProcessor chain processes polygons through all steps."""
    config = PipelineConfig.default()
    processor = GeometryProcessor(config)
    
    # Create a complex polygon
    complex_poly = Polygon([(0, 0), (10, 0), (10.2, 0.1), (10, 5), (0, 5)])
    pred = {"class": "wall", "confidence": 0.9, "points": []}
    
    result = processor.process(pred, complex_poly)
    
    assert isinstance(result, Polygon)
    assert not result.is_empty


def test_geometry_processor_batch():
    """Test GeometryProcessor batch processing."""
    config = PipelineConfig.default()
    processor = GeometryProcessor(config)
    
    predictions = [
        {"class": "wall", "confidence": 0.9},
        {"class": "door", "confidence": 0.8},
    ]
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        Polygon([(5, 0), (6, 0), (6, 2), (5, 2)]),
    ]
    
    results = processor.process_batch(predictions, polygons)
    
    assert len(results) == 2
    assert all(isinstance(p, Polygon) for p in results)
    assert all(not p.is_empty for p in results)

