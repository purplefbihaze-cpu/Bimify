"""
IFC Export V2 Pipeline Configuration

Centralized configuration with Pydantic validation for the IFC export pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class GapClosureMode(str, Enum):
    """Gap closure mode for wall gap repair."""
    PROPOSE = "propose"  # Nur Vorschläge, keine Geometrie-Änderung
    PROPOSE_ONLY = "propose_only"  # Warnen, nicht reparieren (alias for PROPOSE with clearer intent)
    REPAIR_AND_MARK = "repair_and_mark"  # Repariert + markiert im IFC
    SILENT_REPAIR = "silent_repair"  # Legacy-Verhalten (RISIKO!)


class PipelineConfig(BaseModel):
    """
    Centralized configuration for IFC Export V2 pipeline.
    
    All parameters are validated and have sensible defaults.
    Feature flags allow enabling/disabling specific pipeline steps.
    """
    
    # Post-Processing Parameters
    simplify_tolerance_walls: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Simplification tolerance for walls in mm (Douglas-Peucker)"
    )
    simplify_tolerance_doors_windows: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Simplification tolerance for doors/windows in mm"
    )
    grid_size_walls: float = Field(
        default=50.0,
        ge=5.0,
        le=200.0,
        description="Grid size for snapping walls in mm"
    )
    grid_size_doors: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Grid size for snapping doors in mm"
    )
    grid_size_windows: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Grid size for snapping windows in mm"
    )
    angle_tolerance: float = Field(
        default=5.0,
        ge=0.0,
        le=45.0,
        description="Angle tolerance for enforcing right angles in degrees"
    )
    max_gap_close: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Maximum gap to close between walls in mm"
    )
    door_wall_max_distance: float = Field(
        default=100.0,
        ge=10.0,
        le=500.0,
        description="Maximum distance to search for walls when fixing doors in mm"
    )
    
    # Validation Thresholds
    min_confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for predictions (0.0 = no filter)"
    )
    min_points_wall: int = Field(
        default=4,
        ge=3,
        le=100,
        description="Minimum number of points required for a wall polygon"
    )
    min_points_door: int = Field(
        default=3,
        ge=3,
        le=100,
        description="Minimum number of points required for a door polygon"
    )
    min_points_window: int = Field(
        default=3,
        ge=3,
        le=100,
        description="Minimum number of points required for a window polygon"
    )
    max_wall_length_mm: float = Field(
        default=100000.0,  # 100m
        ge=100.0,
        le=1000000.0,
        description="Maximum realistic wall length in mm"
    )
    min_wall_length_mm: float = Field(
        default=100.0,  # 0.1m
        ge=10.0,
        le=1000.0,
        description="Minimum realistic wall length in mm"
    )
    min_room_area_m2: float = Field(
        default=1.0,
        ge=0.1,
        le=1000.0,
        description="Minimum area for a valid room in square meters"
    )
    geometry_simplification_warning_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Area difference threshold (0.0-1.0) for warning about geometry simplification. 0.2 = 20%"
    )
    
    # Performance Parameters
    parallel_processing_threshold: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of predictions above which parallel processing is enabled"
    )
    skeletonization_max_dimension: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum dimension before downsampling for skeletonization"
    )
    skeletonization_target_dpi: float = Field(
        default=100.0,
        ge=50.0,
        le=400.0,
        description="Target DPI for downsampled skeletonization"
    )
    enable_skeletonization_cache: bool = Field(
        default=True,
        description="Enable LRU cache for skeletonization of identical polygons"
    )
    skeletonization_cache_size: int = Field(
        default=128,
        ge=1,
        le=1000,
        description="Maximum cache size for skeletonization"
    )
    
    # Feature Flags
    enable_snap_to_grid: bool = Field(
        default=True,
        description="Enable snap-to-grid post-processing"
    )
    enable_right_angle: bool = Field(
        default=True,
        description="Enable right angle enforcement"
    )
    enable_skeletonization: bool = Field(
        default=True,
        description="Enable skeletonization for wall axes"
    )
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for large datasets"
    )
    enable_adaptive_grid: bool = Field(
        default=True,
        description="Use adaptive grid sizes based on object class"
    )
    enable_input_validation: bool = Field(
        default=True,
        description="Enable input validation after raw predictions"
    )
    enable_geometry_validation: bool = Field(
        default=True,
        description="Enable geometry validation before IFC export"
    )
    preserve_exact_geometry: bool = Field(
        default=True,
        description="Preserve exact geometry (original polygons) instead of simplifying to rectangles"
    )
    gap_closure_mode: GapClosureMode = Field(
        default=GapClosureMode.PROPOSE_ONLY,
        description="Gap closure mode. PROPOSE_ONLY = warn only, no repair, REPAIR_AND_MARK = repair and mark in IFC, SILENT_REPAIR = legacy behavior"
    )
    
    @field_validator("grid_size_walls", "grid_size_doors", "grid_size_windows")
    @classmethod
    def validate_grid_sizes(cls, v: float, info) -> float:
        """Ensure grid sizes are reasonable multiples."""
        if v < 1.0:
            raise ValueError(f"Grid size {info.field_name} must be at least 1.0 mm")
        if v > 200.0:
            raise ValueError(f"Grid size {info.field_name} too large for precise architecture (max 200mm)")
        return v
    
    @model_validator(mode="after")
    def validate_grid_vs_tolerance(self) -> "PipelineConfig":
        """Ensure grid sizes are compatible with simplification tolerances."""
        if self.grid_size_walls < self.simplify_tolerance_walls:
            raise ValueError(
                f"Grid size for walls ({self.grid_size_walls}mm) should be >= "
                f"simplification tolerance ({self.simplify_tolerance_walls}mm)"
            )
        if self.grid_size_doors < self.simplify_tolerance_doors_windows:
            raise ValueError(
                f"Grid size for doors ({self.grid_size_doors}mm) should be >= "
                f"simplification tolerance ({self.simplify_tolerance_doors_windows}mm)"
            )
        if self.grid_size_windows < self.simplify_tolerance_doors_windows:
            raise ValueError(
                f"Grid size for windows ({self.grid_size_windows}mm) should be >= "
                f"simplification tolerance ({self.simplify_tolerance_doors_windows}mm)"
            )
        return self
    
    @model_validator(mode="after")
    def validate_min_max_lengths(self) -> "PipelineConfig":
        """Ensure min length is less than max length."""
        if self.min_wall_length_mm >= self.max_wall_length_mm:
            raise ValueError(
                f"Minimum wall length ({self.min_wall_length_mm}mm) must be less than "
                f"maximum wall length ({self.max_wall_length_mm}mm)"
            )
        return self
    
    def get_grid_size(self, class_name: str) -> float:
        """
        Get grid size for a given object class.
        
        Args:
            class_name: Object class name (wall, door, window, etc.)
            
        Returns:
            Grid size in mm
        """
        if not self.enable_adaptive_grid:
            # Fallback to wall grid size if adaptive grid is disabled
            return self.grid_size_walls
        
        class_lower = str(class_name).lower().strip()
        if "wall" in class_lower:
            return self.grid_size_walls
        elif "door" in class_lower:
            return self.grid_size_doors
        elif "window" in class_lower:
            return self.grid_size_windows
        else:
            # Default to smallest grid for precision
            return min(self.grid_size_doors, self.grid_size_windows)
    
    def get_min_points(self, class_name: str) -> int:
        """
        Get minimum point count for a given object class.
        
        Args:
            class_name: Object class name
            
        Returns:
            Minimum number of points required
        """
        class_lower = str(class_name).lower().strip()
        if "wall" in class_lower:
            return self.min_points_wall
        elif "door" in class_lower:
            return self.min_points_door
        elif "window" in class_lower:
            return self.min_points_window
        else:
            return 3  # Default minimum
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        """Create configuration from dictionary."""
        return cls(**data)
    
    def with_fidelity(self, level: "GeometryFidelityLevel") -> "PipelineConfig":
        """
        Create a new config with overrides based on geometry fidelity level.
        
        Args:
            level: Geometry fidelity level from GeometryFidelityLevel enum
            
        Returns:
            New PipelineConfig instance with fidelity-based overrides
        """
        from services.api.schemas import GeometryFidelityLevel
        
        # Map fidelity levels to config values
        if level == GeometryFidelityLevel.LOSSLESS:
            updates = {
                "simplify_tolerance_walls": 0.1,
                "simplify_tolerance_doors_windows": 0.1,
                "grid_size_walls": 0.0,  # Disabled (0.0 means no snapping)
                "grid_size_doors": 0.0,
                "grid_size_windows": 0.0,
                "angle_tolerance": 0.0,  # Disabled
                "enable_snap_to_grid": False,
                "enable_right_angle": False,
                "gap_closure_mode": GapClosureMode.PROPOSE_ONLY,
            }
        elif level == GeometryFidelityLevel.HIGH:
            updates = {
                "simplify_tolerance_walls": 2.0,
                "simplify_tolerance_doors_windows": 1.0,
                "grid_size_walls": 2.0,
                "grid_size_doors": 2.0,
                "grid_size_windows": 2.0,
                "angle_tolerance": 5.0,
                "enable_snap_to_grid": True,
                "enable_right_angle": True,
                "gap_closure_mode": GapClosureMode.PROPOSE_ONLY,
            }
        elif level == GeometryFidelityLevel.MEDIUM:
            updates = {
                "simplify_tolerance_walls": 5.0,
                "simplify_tolerance_doors_windows": 2.0,
                "grid_size_walls": 5.0,
                "grid_size_doors": 5.0,
                "grid_size_windows": 5.0,
                "angle_tolerance": 10.0,
                "enable_snap_to_grid": True,
                "enable_right_angle": True,
                "gap_closure_mode": GapClosureMode.REPAIR_AND_MARK,
                "max_gap_close": 50.0,  # Gaps >50mm trigger repair
            }
        elif level == GeometryFidelityLevel.LOW:
            updates = {
                "simplify_tolerance_walls": 10.0,
                "simplify_tolerance_doors_windows": 5.0,
                "grid_size_walls": 10.0,
                "grid_size_doors": 10.0,
                "grid_size_windows": 10.0,
                "angle_tolerance": 15.0,
                "enable_snap_to_grid": True,
                "enable_right_angle": True,
                "gap_closure_mode": GapClosureMode.REPAIR_AND_MARK,
                "max_gap_close": 20.0,  # Gaps >20mm trigger repair
            }
        else:
            # Unknown level, return unchanged
            return self
        
        # Create new config with updates
        config_dict = self.model_dump()
        config_dict.update(updates)
        return self.__class__(**config_dict)

