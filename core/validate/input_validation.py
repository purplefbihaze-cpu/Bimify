"""
Input Validation for Raw Predictions

Validates raw Roboflow predictions before post-processing to catch
invalid geometries early and prevent pipeline failures.

This module provides backward-compatible wrapper around PredictionValidator.
"""

from __future__ import annotations

from typing import Any, List, Union

from core.ml.pipeline_config import PipelineConfig
from core.ml.roboflow_client import RoboflowPrediction

from core.validate.prediction_validator import PredictionValidator, ValidationResult


def validate_raw_predictions(
    raw_predictions: List[Union[RoboflowPrediction, dict[str, Any]]],
    config: PipelineConfig | None = None,
    px_per_mm: float | None = None,
) -> ValidationResult:
    """
    Validate raw predictions from Roboflow.
    
    This is a backward-compatible wrapper around PredictionValidator.
    
    Checks:
    - Self-intersecting polygons
    - Minimum point count per shape
    - Realistic dimensional constraints
    - Confidence score filtering
    - Empty/invalid polygons
    
    Args:
        raw_predictions: List of RoboflowPrediction objects or dicts with 'points', 'class', 'confidence'
        config: Pipeline configuration (uses defaults if None)
        px_per_mm: Pixels per millimeter conversion (for dimensional checks)
        
    Returns:
        ValidationResult with valid/invalid predictions and statistics
    """
    if config is None:
        config = PipelineConfig.default()
    
    validator = PredictionValidator(config)
    return validator.validate(raw_predictions, px_per_mm=px_per_mm)

