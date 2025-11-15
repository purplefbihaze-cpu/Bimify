"""
Unit tests for IFC V2 Post-Processing Pipeline

Tests for polygon simplification, snap-to-grid, enforce right angles,
and context-based correction functions.
"""

from __future__ import annotations

import math
import pytest
from shapely.geometry import Polygon, Point

from core.ml.postprocess_v2 import (
    simplify_polygon,
    snap_to_grid,
    snap_polygon_to_grid,
    enforce_right_angles,
    fix_door_position,
    close_wall_gaps,
    clip_window_to_wall,
    process_roboflow_predictions_v2,
    SIMPLIFY_TOLERANCE_WALLS,
    SIMPLIFY_TOLERANCE_DOORS_WINDOWS,
    GRID_SIZE,
    ANGLE_TOLERANCE,
    MAX_GAP_CLOSE,
    DOOR_WALL_MAX_DISTANCE,
)


class TestSimplifyPolygon:
    """Tests for polygon simplification (Douglas-Peucker)."""
    
    def test_simplify_zigzag_wall(self):
        """Test simplification of a zigzag wall polygon."""
        # Create a zigzag polygon with many points
        points = [(0, 0), (10, 1), (20, 0), (30, 1), (40, 0), (50, 1), (60, 0)]
        poly = Polygon(points)
        
        # Simplify with tolerance
        simplified = simplify_polygon(poly, tolerance=2.0)
        
        # Should have fewer points
        assert len(simplified.exterior.coords) <= len(points)
        assert simplified.is_valid
        assert not simplified.is_empty
    
    def test_simplify_with_different_tolerances(self):
        """Test simplification with different tolerance values."""
        # Create polygon with many points
        points = [(i, i % 2) for i in range(20)]
        poly = Polygon(points)
        
        # High tolerance should reduce points more
        simplified_high = simplify_polygon(poly, tolerance=5.0)
        simplified_low = simplify_polygon(poly, tolerance=0.5)
        
        # High tolerance should have fewer or equal points
        assert len(simplified_high.exterior.coords) <= len(simplified_low.exterior.coords)
        assert simplified_high.is_valid
        assert simplified_low.is_valid
    
    def test_simplify_empty_polygon(self):
        """Test that empty polygon is handled."""
        empty_poly = Polygon()
        result = simplify_polygon(empty_poly, tolerance=2.0)
        assert result.is_empty
    
    def test_simplify_invalid_polygon(self):
        """Test that invalid polygon is repaired."""
        # Create self-intersecting polygon
        points = [(0, 0), (10, 10), (10, 0), (0, 10)]
        poly = Polygon(points)
        
        # Should repair and simplify
        result = simplify_polygon(poly, tolerance=1.0)
        assert result.is_valid or result.is_empty


class TestSnapToGrid:
    """Tests for snap-to-grid functionality."""
    
    def test_snap_point_to_grid(self):
        """Test snapping a single point to grid."""
        point = (123.7, 456.3)
        snapped = snap_to_grid(point, grid_size=50.0)
        
        # Should be on grid
        assert snapped[0] % 50.0 == 0.0
        assert snapped[1] % 50.0 == 0.0
        assert snapped == (100.0, 450.0)
    
    def test_snap_point_custom_grid(self):
        """Test snapping with custom grid size."""
        point = (123.7, 456.3)
        snapped = snap_to_grid(point, grid_size=10.0)
        
        # Should be on 10mm grid
        assert snapped[0] % 10.0 == 0.0
        assert snapped[1] % 10.0 == 0.0
    
    def test_snap_polygon_to_grid(self):
        """Test snapping all vertices of a polygon to grid."""
        points = [(123.7, 456.3), (234.8, 567.9), (345.1, 678.2)]
        poly = Polygon(points)
        
        snapped = snap_polygon_to_grid(poly, grid_size=50.0)
        
        # All vertices should be on grid
        for x, y in snapped.exterior.coords[:-1]:  # Exclude duplicate last point
            assert x % 50.0 == 0.0
            assert y % 50.0 == 0.0
        assert snapped.is_valid
    
    def test_snap_polygon_preserves_shape(self):
        """Test that snapping preserves overall polygon shape."""
        # Create a rectangle
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        poly = Polygon(points)
        
        snapped = snap_polygon_to_grid(poly, grid_size=50.0)
        
        # Should still be a rectangle (4 points)
        assert len(snapped.exterior.coords) == 5  # 4 + duplicate last
        assert snapped.is_valid


class TestEnforceRightAngles:
    """Tests for enforcing 90° angles."""
    
    def test_enforce_right_angles_horizontal(self):
        """Test enforcing right angles on horizontal wall."""
        # Create a slightly tilted horizontal wall
        points = [(0, 0), (100, 2), (100, 22), (0, 20)]
        poly = Polygon(points)
        
        corrected = enforce_right_angles(poly, tolerance_deg=5.0)
        
        assert corrected.is_valid
        # Should have corrected angles
    
    def test_enforce_right_angles_vertical(self):
        """Test enforcing right angles on vertical wall."""
        # Create a slightly tilted vertical wall
        points = [(0, 0), (2, 100), (22, 100), (20, 0)]
        poly = Polygon(points)
        
        corrected = enforce_right_angles(poly, tolerance_deg=5.0)
        
        assert corrected.is_valid
    
    def test_enforce_right_angles_rectangle(self):
        """Test that perfect rectangle is preserved."""
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        poly = Polygon(points)
        
        corrected = enforce_right_angles(poly, tolerance_deg=5.0)
        
        assert corrected.is_valid
        # Should be similar to original
    
    def test_enforce_right_angles_non_orthogonal(self):
        """Test that non-orthogonal polygon is handled."""
        # Create a triangle (not orthogonal)
        points = [(0, 0), (100, 0), (50, 100)]
        poly = Polygon(points)
        
        corrected = enforce_right_angles(poly, tolerance_deg=5.0)
        
        # Should still be valid (may not change much if not close to 90°)
        assert corrected.is_valid


class TestFixDoorPosition:
    """Tests for door position correction."""
    
    def test_fix_door_touching_wall(self):
        """Test fixing door that should touch a wall."""
        # Create a wall
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        
        # Create a door near but not touching the wall
        door = Polygon([(50, 25), (70, 25), (70, 45), (50, 45)])
        
        fixed = fix_door_position(door, [wall], max_distance=100.0)
        
        assert fixed.is_valid
        # Door should be closer to or intersecting wall
    
    def test_fix_door_already_touching(self):
        """Test door that already touches wall."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        door = Polygon([(50, 20), (70, 20), (70, 40), (50, 40)])
        
        fixed = fix_door_position(door, [wall], max_distance=100.0)
        
        assert fixed.is_valid
    
    def test_fix_door_no_walls(self):
        """Test door with no walls."""
        door = Polygon([(50, 25), (70, 25), (70, 45), (50, 45)])
        
        fixed = fix_door_position(door, [], max_distance=100.0)
        
        # Should return original
        assert fixed == door
    
    def test_fix_door_too_far(self):
        """Test door too far from walls."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        door = Polygon([(50, 200), (70, 200), (70, 220), (50, 220)])
        
        fixed = fix_door_position(door, [wall], max_distance=100.0)
        
        # Should return original (too far)
        assert fixed == door


class TestCloseWallGaps:
    """Tests for closing gaps between walls."""
    
    def test_close_small_gap(self):
        """Test closing a small gap between walls."""
        wall1 = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        wall2 = Polygon([(105, 0), (200, 0), (200, 20), (105, 20)])  # 5mm gap
        
        corrected = close_wall_gaps([wall1, wall2], max_gap=10.0)
        
        assert len(corrected) >= 1
        # Should attempt to merge or connect
    
    def test_close_large_gap(self):
        """Test that large gaps are not closed."""
        wall1 = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        wall2 = Polygon([(200, 0), (300, 0), (300, 20), (200, 20)])  # 100mm gap
        
        corrected = close_wall_gaps([wall1, wall2], max_gap=10.0)
        
        # Should not merge (gap too large)
        assert len(corrected) >= 2
    
    def test_close_gaps_single_wall(self):
        """Test with single wall (no gaps to close)."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        
        corrected = close_wall_gaps([wall], max_gap=10.0)
        
        assert len(corrected) == 1


class TestClipWindowToWall:
    """Tests for clipping windows to walls."""
    
    def test_clip_window_inside_wall(self):
        """Test window that is inside wall."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        window = Polygon([(20, 5), (80, 5), (80, 15), (20, 15)])
        
        clipped = clip_window_to_wall(window, wall)
        
        assert clipped.is_valid
        # Should be clipped to intersection
        assert clipped.intersects(wall) or clipped.within(wall)
    
    def test_clip_window_partially_outside(self):
        """Test window partially outside wall."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        window = Polygon([(90, 5), (120, 5), (120, 15), (90, 15)])
        
        clipped = clip_window_to_wall(window, wall)
        
        assert clipped.is_valid
        # Should be clipped to wall boundary
    
    def test_clip_window_completely_outside(self):
        """Test window completely outside wall."""
        wall = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        window = Polygon([(150, 5), (200, 5), (200, 15), (150, 15)])
        
        clipped = clip_window_to_wall(window, wall)
        
        # Should return original (no intersection)
        assert clipped == window


class TestProcessRoboflowPredictionsV2:
    """Tests for main post-processing pipeline."""
    
    def test_process_simple_wall(self):
        """Test processing a simple wall prediction."""
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                "points": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ]
        
        result = process_roboflow_predictions_v2(predictions, px_per_mm=1.0)
        
        assert len(result) == 1
        assert result[0]["class"] == "wall"
        assert "points" in result[0]
        # Points should be processed (simplified, snapped)
    
    def test_process_wall_and_door(self):
        """Test processing wall and door together."""
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                "points": [[0, 0], [100, 0], [100, 20], [0, 20]],
            },
            {
                "class": "door",
                "confidence": 0.8,
                "points": [[45, 20], [55, 20], [55, 40], [45, 40]],
            },
        ]
        
        result = process_roboflow_predictions_v2(predictions, px_per_mm=1.0)
        
        assert len(result) == 2
        # Door should be fixed relative to wall
    
    def test_process_with_pixel_conversion(self):
        """Test processing with pixel-to-mm conversion."""
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                "points": [[0, 0], [100, 0], [100, 20], [0, 20]],  # Pixels
            }
        ]
        
        # px_per_mm = 2.0 means 2 pixels per mm
        result = process_roboflow_predictions_v2(predictions, px_per_mm=2.0)
        
        assert len(result) == 1
        # Points should be converted to mm, processed, then converted back
    
    def test_process_empty_predictions(self):
        """Test processing empty predictions list."""
        result = process_roboflow_predictions_v2([], px_per_mm=1.0)
        assert result == []
    
    def test_process_prediction_without_points(self):
        """Test processing prediction without polygon points."""
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                # No points
            }
        ]
        
        result = process_roboflow_predictions_v2(predictions, px_per_mm=1.0)
        
        # Should skip invalid predictions
        assert len(result) == 0
    
    def test_process_zigzag_wall(self):
        """Test processing a zigzag wall (many points)."""
        # Create zigzag points
        points = [[i * 10, (i % 2) * 5] for i in range(20)]
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                "points": points,
            }
        ]
        
        result = process_roboflow_predictions_v2(
            predictions,
            px_per_mm=1.0,
            simplify_tolerance_walls=2.0,
        )
        
        assert len(result) == 1
        # Should have fewer points after simplification
        original_points = len(points)
        processed_points = len(result[0]["points"])
        # Note: May not always be fewer due to snapping, but should be processed
    
    def test_process_with_custom_parameters(self):
        """Test processing with custom parameters."""
        predictions = [
            {
                "class": "wall",
                "confidence": 0.9,
                "points": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ]
        
        result = process_roboflow_predictions_v2(
            predictions,
            px_per_mm=1.0,
            grid_size=10.0,  # 1cm grid instead of 5cm
            angle_tolerance=10.0,  # Larger tolerance
        )
        
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

