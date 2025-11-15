"""
Pipeline Metrics Collection

Collects metrics during IFC export pipeline execution for monitoring and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineMetrics:
    """
    Metrics collected during pipeline execution.
    
    Tracks performance, quality, and validation statistics for monitoring.
    """
    
    # Input statistics
    total_input_predictions: int = 0
    valid_input_predictions: int = 0
    invalid_input_predictions: int = 0
    filtered_by_confidence: int = 0
    filtered_by_points: int = 0
    filtered_by_geometry: int = 0
    filtered_by_dimensions: int = 0
    
    # Post-processing statistics
    total_processed: int = 0
    corrected_by_snap: int = 0
    corrected_by_right_angle: int = 0
    doors_fixed: int = 0
    windows_clipped: int = 0
    
    # Reconstruction statistics
    total_walls: int = 0
    total_doors: int = 0
    total_windows: int = 0
    total_spaces: int = 0
    closed_spaces: int = 0
    open_spaces: int = 0
    overlapping_spaces: int = 0
    spaces_below_min_area: int = 0
    
    # Wall axes statistics
    total_wall_axes: int = 0
    parallel_axes: int = 0
    non_parallel_axes: int = 0
    skeletonization_quality_scores: list[float] = field(default_factory=list)
    
    # Performance metrics (in seconds)
    time_input_validation: float = 0.0
    time_post_processing: float = 0.0
    time_normalization: float = 0.0
    time_reconstruction: float = 0.0
    time_ifc_export: float = 0.0
    time_validation: float = 0.0
    time_total: float = 0.0
    
    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings_by_category: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "input": {
                "total": self.total_input_predictions,
                "valid": self.valid_input_predictions,
                "invalid": self.invalid_input_predictions,
                "filtered": {
                    "confidence": self.filtered_by_confidence,
                    "points": self.filtered_by_points,
                    "geometry": self.filtered_by_geometry,
                    "dimensions": self.filtered_by_dimensions,
                },
            },
            "post_processing": {
                "total_processed": self.total_processed,
                "corrected": {
                    "snap": self.corrected_by_snap,
                    "right_angle": self.corrected_by_right_angle,
                },
                "doors_fixed": self.doors_fixed,
                "windows_clipped": self.windows_clipped,
            },
            "reconstruction": {
                "walls": self.total_walls,
                "doors": self.total_doors,
                "windows": self.total_windows,
                "spaces": {
                    "total": self.total_spaces,
                    "closed": self.closed_spaces,
                    "open": self.open_spaces,
                    "overlapping": self.overlapping_spaces,
                    "below_min_area": self.spaces_below_min_area,
                },
            },
            "wall_axes": {
                "total": self.total_wall_axes,
                "parallel": self.parallel_axes,
                "non_parallel": self.non_parallel_axes,
                "average_quality": (
                    sum(self.skeletonization_quality_scores) / len(self.skeletonization_quality_scores)
                    if self.skeletonization_quality_scores else 0.0
                ),
            },
            "performance": {
                "input_validation": self.time_input_validation,
                "post_processing": self.time_post_processing,
                "normalization": self.time_normalization,
                "reconstruction": self.time_reconstruction,
                "ifc_export": self.time_ifc_export,
                "validation": self.time_validation,
                "total": self.time_total,
            },
            "warnings": {
                "total": len(self.warnings),
                "by_category": self.warnings_by_category,
                "list": self.warnings,
            },
            "errors": {
                "total": len(self.errors),
                "list": self.errors,
            },
        }
    
    def add_warning(self, message: str, category: str = "general") -> None:
        """Add a warning message and update category count."""
        self.warnings.append(message)
        self.warnings_by_category[category] = self.warnings_by_category.get(category, 0) + 1
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0.0 to 1.0).
        
        Based on:
        - Input validation pass rate
        - Space closure rate
        - Wall axes parallelism
        - Average skeletonization quality
        """
        score = 1.0
        
        # Input validation score
        if self.total_input_predictions > 0:
            input_ratio = self.valid_input_predictions / self.total_input_predictions
            score *= input_ratio
        
        # Space closure score
        if self.total_spaces > 0:
            closure_ratio = self.closed_spaces / self.total_spaces
            score *= closure_ratio
        
        # Wall axes parallelism score
        if self.total_wall_axes > 0:
            parallel_ratio = self.parallel_axes / self.total_wall_axes
            score *= parallel_ratio
        
        # Skeletonization quality score
        if self.skeletonization_quality_scores:
            avg_quality = sum(self.skeletonization_quality_scores) / len(self.skeletonization_quality_scores)
            score *= avg_quality
        
        # Penalize errors
        if self.errors:
            error_penalty = min(len(self.errors) * 0.1, 0.5)  # Max 50% penalty
            score *= (1.0 - error_penalty)
        
        return max(0.0, min(1.0, score))
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            "quality_score": self.calculate_quality_score(),
            "total_time_seconds": self.time_total,
            "input_validation_rate": (
                self.valid_input_predictions / self.total_input_predictions
                if self.total_input_predictions > 0 else 0.0
            ),
            "space_closure_rate": (
                self.closed_spaces / self.total_spaces
                if self.total_spaces > 0 else 0.0
            ),
            "wall_axes_parallelism_rate": (
                self.parallel_axes / self.total_wall_axes
                if self.total_wall_axes > 0 else 0.0
            ),
            "total_warnings": len(self.warnings),
            "total_errors": len(self.errors),
        }

