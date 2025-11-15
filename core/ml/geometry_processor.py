"""
Geometry Processor Chain for IFC Export V2

Chainable geometry processing steps for cleaning Roboflow predictions.
Each step is independently testable and can be composed into a processing pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shapely.geometry import Polygon

from core.ml.pipeline_config import PipelineConfig


class GeometryProcessorStep(ABC):
    """Base class for geometry processing steps."""
    
    @abstractmethod
    def process(self, pred: dict[str, Any], polygon: Polygon, config: PipelineConfig) -> Polygon:
        """
        Process a polygon for a given prediction.
        
        Args:
            pred: Prediction dict with 'class', 'confidence', etc.
            polygon: Shapely Polygon to process
            config: Pipeline configuration
            
        Returns:
            Processed polygon
        """
        pass


class Simplifier(GeometryProcessorStep):
    """Simplifies polygons using Douglas-Peucker algorithm."""
    
    def __init__(self, tolerance_walls: float | None = None, tolerance_doors_windows: float | None = None):
        """
        Initialize simplifier with optional tolerance overrides.
        
        Args:
            tolerance_walls: Override tolerance for walls (mm)
            tolerance_doors_windows: Override tolerance for doors/windows (mm)
        """
        self.tolerance_walls = tolerance_walls
        self.tolerance_doors_windows = tolerance_doors_windows
    
    def process(self, pred: dict[str, Any], polygon: Polygon, config: PipelineConfig) -> Polygon:
        """Simplify polygon based on object class."""
        # Lazy import to avoid circular dependency
        from core.ml.postprocess_v2 import simplify_polygon
        
        class_name = str(pred.get("class", "")).lower().strip()
        
        if "wall" in class_name:
            tolerance = self.tolerance_walls if self.tolerance_walls is not None else config.simplify_tolerance_walls
        else:
            tolerance = self.tolerance_doors_windows if self.tolerance_doors_windows is not None else config.simplify_tolerance_doors_windows
        
        return simplify_polygon(polygon, tolerance)


class GridSnapper(GeometryProcessorStep):
    """Snaps polygon vertices to a grid."""
    
    def __init__(self, grid_size: float | None = None):
        """
        Initialize grid snapper with optional grid size override.
        
        Args:
            grid_size: Override grid size (mm)
        """
        self.grid_size = grid_size
    
    def process(self, pred: dict[str, Any], polygon: Polygon, config: PipelineConfig) -> Polygon:
        """Snap polygon to grid based on object class."""
        # Lazy import to avoid circular dependency
        from core.ml.postprocess_v2 import snap_polygon_to_grid
        
        if not config.enable_snap_to_grid:
            return polygon
        
        class_name = str(pred.get("class", "")).lower().strip()
        grid_size = self.grid_size if self.grid_size is not None else config.get_grid_size(class_name)
        
        return snap_polygon_to_grid(
            polygon,
            grid_size=grid_size,
            config=config,
            class_name=class_name,
        )


class OrthogonalEnforcer(GeometryProcessorStep):
    """Enforces 90° angles in polygons."""
    
    def __init__(self, angle_tolerance: float | None = None):
        """
        Initialize orthogonal enforcer with optional angle tolerance override.
        
        Args:
            angle_tolerance: Override angle tolerance (degrees)
        """
        self.angle_tolerance = angle_tolerance
    
    def process(self, pred: dict[str, Any], polygon: Polygon, config: PipelineConfig) -> Polygon:
        """Enforce right angles if enabled and applicable."""
        # Lazy import to avoid circular dependency
        from core.ml.postprocess_v2 import enforce_right_angles
        
        if not config.enable_right_angle:
            return polygon
        
        class_name = str(pred.get("class", "")).lower().strip()
        if "wall" not in class_name and "door" not in class_name and "window" not in class_name:
            return polygon
        
        tolerance = self.angle_tolerance if self.angle_tolerance is not None else config.angle_tolerance
        return enforce_right_angles(polygon, tolerance)


class ContextCorrector(GeometryProcessorStep):
    """Context-based corrections (door/window snapping, gap closing)."""
    
    def __init__(
        self,
        walls: list[tuple[int, Polygon]] | None = None,
        max_gap_close: float | None = None,
        door_wall_max_distance: float | None = None,
    ):
        """
        Initialize context corrector.
        
        Args:
            walls: List of (index, polygon) tuples for walls
            max_gap_close: Maximum gap to close (mm)
            door_wall_max_distance: Max distance for door-wall snapping (mm)
        """
        self.walls = walls or []
        self.max_gap_close = max_gap_close
        self.door_wall_max_distance = door_wall_max_distance
    
    def process(self, pred: dict[str, Any], polygon: Polygon, config: PipelineConfig) -> Polygon:
        """
        Apply context-based corrections.
        
        Note: This step requires wall context, so it's typically applied
        after all individual polygons have been processed.
        """
        # Context correction is handled separately in the main pipeline
        # This step is a placeholder for the chain interface
        return polygon


class GeometryProcessor:
    """
    Chainable geometry processor for Roboflow predictions.
    
    Processes predictions through a sequence of geometry processing steps.
    Each step is independently testable and can be enabled/disabled via config.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize geometry processor with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipeline: list[GeometryProcessorStep] = []
        
        # Build processing chain
        self.pipeline.append(Simplifier())
        if config.enable_snap_to_grid:
            self.pipeline.append(GridSnapper())
        if config.enable_right_angle:
            self.pipeline.append(OrthogonalEnforcer())
    
    def process(self, pred: dict[str, Any], polygon: Polygon) -> Polygon:
        """
        Process a single prediction through the geometry processing chain.
        
        Args:
            pred: Prediction dict with 'class', 'confidence', 'points', etc.
            polygon: Shapely Polygon to process
            
        Returns:
            Processed polygon
        """
        result = polygon
        
        for step in self.pipeline:
            try:
                result = step.process(pred, result, self.config)
            except Exception as e:
                # Log error but continue with previous result
                from loguru import logger
                logger.warning("Error in geometry processing step %s: %s", step.__class__.__name__, e)
                break
        
        return result
    
    def process_batch(self, predictions: list[dict[str, Any]], polygons: list[Polygon]) -> list[Polygon]:
        """
        Process a batch of predictions through the geometry processing chain.
        
        Args:
            predictions: List of prediction dicts
            polygons: List of corresponding Shapely Polygons
            
        Returns:
            List of processed polygons
        """
        if len(predictions) != len(polygons):
            raise ValueError("Predictions and polygons lists must have same length")
        
        return [self.process(pred, poly) for pred, poly in zip(predictions, polygons)]

