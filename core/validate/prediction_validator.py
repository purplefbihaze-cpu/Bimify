"""
Prediction Validator

Refactored validation engine with testable individual validation rules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from loguru import logger
from shapely.geometry import Polygon

from core.ml.pipeline_config import PipelineConfig
from core.ml.roboflow_client import RoboflowPrediction


class ValidationResult:
    """Result of input validation."""
    
    def __init__(
        self,
        is_valid: bool,
        valid_predictions: list[dict[str, Any]],
        invalid_predictions: list[dict[str, Any]],
        warnings: list[str],
        statistics: dict[str, Any],
    ):
        self.is_valid = is_valid
        self.valid_predictions = valid_predictions
        self.invalid_predictions = invalid_predictions
        self.warnings = warnings
        self.statistics = statistics


class PredictionValidator:
    """Validates Roboflow predictions with individual testable rules."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize validator with configuration.
        
        Args:
            config: Pipeline configuration with validation thresholds
        """
        self.config = config
        # Per-class confidence thresholds (can be extended)
        self._confidence_thresholds: Dict[str, float] = {
            "wall": config.min_confidence_score,
            "door": config.min_confidence_score,
            "window": config.min_confidence_score,
        }
    
    def validate(
        self,
        predictions: List[Union[RoboflowPrediction, dict[str, Any]]],
        px_per_mm: float | None = None,
    ) -> ValidationResult:
        """Validate list of predictions.
        
        Args:
            predictions: List of RoboflowPrediction objects or dicts
            px_per_mm: Pixels per millimeter conversion (for dimensional checks)
            
        Returns:
            ValidationResult with valid/invalid predictions and statistics
        """
        if not predictions:
            return ValidationResult(
                is_valid=False,
                valid_predictions=[],
                invalid_predictions=[],
                warnings=["No predictions provided"],
                statistics={"total": 0, "valid": 0, "invalid": 0},
            )
        
        valid_predictions: list[dict[str, Any]] = []
        invalid_predictions: list[dict[str, Any]] = []
        warnings: list[str] = []
        
        stats = {
            "total": len(predictions),
            "valid": 0,
            "invalid": 0,
            "filtered_by_confidence": 0,
            "filtered_by_points": 0,
            "filtered_by_geometry": 0,
            "filtered_by_dimensions": 0,
            "by_class": {},
        }
        
        for idx, pred in enumerate(predictions):
            # Normalize prediction to dict format
            if isinstance(pred, RoboflowPrediction):
                class_name = pred.klass.lower().strip()
                confidence = pred.confidence
                points = pred.points or []
                pred_dict = pred.model_dump(by_alias=True)
            else:
                class_name = str(pred.get("class", "")).lower().strip()
                confidence = float(pred.get("confidence", 0.0))
                points = pred.get("points") or []
                pred_dict = dict(pred)
            
            # Track by class
            if class_name not in stats["by_class"]:
                stats["by_class"][class_name] = {"total": 0, "valid": 0, "invalid": 0}
            stats["by_class"][class_name]["total"] += 1
            
            # Run all validation checks
            is_valid, reason = self._is_valid(pred, class_name, confidence, points, px_per_mm)
            
            if is_valid:
                valid_predictions.append(pred_dict if not isinstance(pred, RoboflowPrediction) else pred.model_dump(by_alias=True))
                stats["valid"] += 1
                stats["by_class"][class_name]["valid"] += 1
            else:
                invalid_predictions.append({
                    **pred_dict,
                    "_validation_reason": reason,
                })
                stats["invalid"] += 1
                stats["by_class"][class_name]["invalid"] += 1
                
                # Update filtered statistics
                if "confidence" in reason.lower():
                    stats["filtered_by_confidence"] += 1
                elif "points" in reason.lower():
                    stats["filtered_by_points"] += 1
                elif "geometry" in reason.lower() or "polygon" in reason.lower():
                    stats["filtered_by_geometry"] += 1
                elif "dimension" in reason.lower() or "length" in reason.lower():
                    stats["filtered_by_dimensions"] += 1
        
        # Determine overall validity
        is_valid = len(valid_predictions) > 0
        
        if not is_valid:
            warnings.append("No valid predictions after validation")
        
        if len(invalid_predictions) > len(valid_predictions):
            warnings.append(
                f"More invalid ({len(invalid_predictions)}) than valid ({len(valid_predictions)}) predictions"
            )
        
        logger.info(
            "Input validation: %d valid, %d invalid out of %d total predictions",
            len(valid_predictions),
            len(invalid_predictions),
            len(predictions),
        )
        
        return ValidationResult(
            is_valid=is_valid,
            valid_predictions=valid_predictions,
            invalid_predictions=invalid_predictions,
            warnings=warnings,
            statistics=stats,
        )
    
    def _is_valid(
        self,
        pred: Union[RoboflowPrediction, dict[str, Any]],
        class_name: str,
        confidence: float,
        points: List[Tuple[float, float]],
        px_per_mm: float | None,
    ) -> Tuple[bool, str]:
        """Check if prediction passes all validation rules.
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check 1: Confidence score
        if not self._check_confidence(class_name, confidence):
            threshold = self._get_confidence_threshold(class_name)
            return False, f"Confidence {confidence:.3f} < {threshold:.3f}"
        
        # Check 2: Minimum point count
        if not self._check_min_points(class_name, points):
            min_points = self.config.get_min_points(class_name)
            return False, f"Insufficient points: {len(points) if points else 0} < {min_points}"
        
        # Check 3: Geometry validity
        geometry_valid, reason = self._check_geometry_validity(points, px_per_mm)
        if not geometry_valid:
            return False, reason
        
        # Check 4: Dimensional constraints (for walls)
        if "wall" in class_name:
            dim_valid, reason = self._check_min_dimensions(points, px_per_mm)
            if not dim_valid:
                return False, reason
        
        return True, ""
    
    def _check_confidence(self, class_name: str, confidence: float) -> bool:
        """Check if confidence meets threshold for class.
        
        Args:
            class_name: Prediction class name
            confidence: Confidence score
            
        Returns:
            True if confidence is sufficient
        """
        if not self.config.enable_input_validation:
            return True
        
        threshold = self._get_confidence_threshold(class_name)
        return confidence >= threshold
    
    def _get_confidence_threshold(self, class_name: str) -> float:
        """Get confidence threshold for class.
        
        Args:
            class_name: Prediction class name
            
        Returns:
            Confidence threshold
        """
        # Check per-class thresholds first
        for key, threshold in self._confidence_thresholds.items():
            if key in class_name.lower():
                return threshold
        
        # Fallback to global threshold
        return self.config.min_confidence_score
    
    def _check_min_points(self, class_name: str, points: List[Tuple[float, float]]) -> bool:
        """Check if prediction has minimum required points.
        
        Args:
            class_name: Prediction class name
            points: List of polygon points
            
        Returns:
            True if point count is sufficient
        """
        if not points:
            return False
        
        min_points = self.config.get_min_points(class_name)
        return len(points) >= min_points
    
    def _check_geometry_validity(
        self,
        points: List[Tuple[float, float]],
        px_per_mm: float | None,
    ) -> Tuple[bool, str]:
        """Check if geometry is valid.
        
        Args:
            points: List of polygon points
            px_per_mm: Pixels per millimeter conversion
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        try:
            # Convert points to polygon
            polygon_points: list[tuple[float, float]] = []
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = float(pt[0]), float(pt[1])
                    # Convert from pixels to mm if needed
                    if px_per_mm is not None:
                        x = x / px_per_mm
                        y = y / px_per_mm
                    polygon_points.append((x, y))
            
            if len(polygon_points) < 3:
                return False, "Less than 3 valid points after conversion"
            
            # Ensure closed polygon
            if polygon_points[0] != polygon_points[-1]:
                polygon_points.append(polygon_points[0])
            
            polygon = Polygon(polygon_points)
            
            # Check if empty
            if polygon.is_empty:
                return False, "Empty polygon"
            
            # Check if valid
            if not polygon.is_valid:
                # Try to repair
                try:
                    repaired = polygon.buffer(0)
                    if isinstance(repaired, Polygon) and repaired.is_valid and not repaired.is_empty:
                        return True, ""  # Repaired successfully
                    else:
                        return False, "Invalid polygon (self-intersecting or degenerate)"
                except Exception as repair_exc:
                    return False, f"Invalid polygon (repair failed: {repair_exc})"
            
            return True, ""
            
        except (TypeError, ValueError, Exception) as exc:
            return False, f"Geometry conversion error: {exc}"
    
    def _check_min_dimensions(
        self,
        points: List[Tuple[float, float]],
        px_per_mm: float | None,
    ) -> Tuple[bool, str]:
        """Check if dimensions meet minimum/maximum constraints (for walls).
        
        Args:
            points: List of polygon points
            px_per_mm: Pixels per millimeter conversion
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not self.config.enable_input_validation:
            return True, ""
        
        try:
            # Convert points to polygon
            polygon_points: list[tuple[float, float]] = []
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = float(pt[0]), float(pt[1])
                    if px_per_mm is not None:
                        x = x / px_per_mm
                        y = y / px_per_mm
                    polygon_points.append((x, y))
            
            if len(polygon_points) < 3:
                return False, "Less than 3 valid points for dimension check"
            
            # Ensure closed polygon
            if polygon_points[0] != polygon_points[-1]:
                polygon_points.append(polygon_points[0])
            
            polygon = Polygon(polygon_points)
            bounds = polygon.bounds
            width = abs(bounds[2] - bounds[0])  # max_x - min_x
            height = abs(bounds[3] - bounds[1])  # max_y - min_y
            max_dimension = max(width, height)
            
            # Check maximum wall length
            if max_dimension > self.config.max_wall_length_mm:
                return False, f"Wall too long: {max_dimension:.1f}mm > {self.config.max_wall_length_mm:.1f}mm"
            
            # Check minimum wall length
            if max_dimension < self.config.min_wall_length_mm:
                return False, f"Wall too short: {max_dimension:.1f}mm < {self.config.min_wall_length_mm:.1f}mm"
            
            return True, ""
            
        except Exception as exc:
            return False, f"Dimension check error: {exc}"

