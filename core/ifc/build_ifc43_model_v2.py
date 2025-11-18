from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Sequence, Tuple

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.entity_instance
import ifcopenshell.guid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.strtree import STRtree

from core.exceptions import (
    IFCExportError,
    MaterialAssignmentError,
    SchemaValidationError,
)
from core.ifc.project_setup import (
    create_project_structure,
    setup_owner_history_v2,
    setup_units_complete,
    setup_geometric_context_v2,
)
from core.ml.postprocess_floorplan import NormalizedDet, estimate_wall_axes_and_thickness
from core.ml.pipeline_config import PipelineConfig
from core.reconstruct.spaces import SpacePoly
from core.ifc.geometry_utils import snap_thickness_mm, find_nearest_wall
from core.ifc.ifc_v2_validator import validate_before_export, validate_ifc_file
from config.ifc_standards import STANDARDS
from core.geometry.contract import MIN_OPENING_DEPTH

# Import simple input models for cleaner API
try:
    from ifc_generator_v2.models import (
        Wall as WallModel,
        Window as WindowModel,
        Door as DoorModel,
        Slab as SlabModel,
        Space as SpaceModel,
        Point2D,
    )
except ImportError:
    # Fallback if models module is not available
    WallModel = None
    WindowModel = None
    DoorModel = None
    SlabModel = None
    SpaceModel = None
    Point2D = None

logger = logging.getLogger(__name__)

# Singleton ProcessPoolExecutor for IFC exports to avoid creating new executors
# This reduces overhead and ensures proper resource management
# IMPORTANT: Must be shutdown cleanly on module reload to prevent server hangs
_executor_lock = threading.Lock()
_shared_executor: ProcessPoolExecutor | None = None
_executor_shutdown_requested = False


def _get_shared_executor() -> ProcessPoolExecutor:
    """Get or create a shared ProcessPoolExecutor instance.
    
    Automatically recreates executor if it was shutdown (e.g., during hot reload).
    Uses non-blocking lock check to prevent hangs during hot reload.
    
    Returns:
        Shared ProcessPoolExecutor instance (max_workers=1 for IFC exports)
    """
    global _shared_executor, _executor_shutdown_requested
    
    # Try to acquire lock (non-blocking to prevent hangs during hot reload)
    if _executor_lock.acquire(blocking=False):
        try:
            # Check if executor was shutdown (e.g., during hot reload)
            if _shared_executor is None or _executor_shutdown_requested:
                # Shutdown old executor if it exists but is marked for shutdown
                if _shared_executor is not None:
                    try:
                        _shared_executor.shutdown(wait=False)  # Don't wait - non-blocking
                    except Exception:
                        pass  # Ignore errors during shutdown
                # Create new executor
                _shared_executor = ProcessPoolExecutor(max_workers=1)
                _executor_shutdown_requested = False
            return _shared_executor
        finally:
            _executor_lock.release()
    else:
        # Lock is held - likely during hot reload, create new executor without lock
        # This is safe because we're creating a new executor anyway
        logger.debug("Executor lock held during hot reload, creating new executor")
        if _shared_executor is not None:
            try:
                _shared_executor.shutdown(wait=False)
            except Exception:
                pass
        _shared_executor = ProcessPoolExecutor(max_workers=1)
        _executor_shutdown_requested = False
        return _shared_executor


def _shutdown_shared_executor() -> None:
    """Shutdown the shared ProcessPoolExecutor (for cleanup/testing/hot reload).
    
    Uses non-blocking shutdown to prevent server hangs during hot reload.
    Uses non-blocking lock check to prevent hangs.
    """
    global _shared_executor, _executor_shutdown_requested
    
    # Try to acquire lock (non-blocking to prevent hangs during hot reload)
    if _executor_lock.acquire(blocking=False):
        try:
            if _shared_executor is not None:
                _executor_shutdown_requested = True
                try:
                    # Non-blocking shutdown - don't wait for running tasks
                    # This prevents server hangs during hot reload
                    _shared_executor.shutdown(wait=False)
                except Exception as e:
                    logger.debug(f"Error shutting down executor (non-critical): {e}")
                finally:
                    _shared_executor = None
        finally:
            _executor_lock.release()
    else:
        # Lock is held - likely during hot reload, force shutdown without lock
        logger.debug("Executor lock held during shutdown, forcing shutdown")
        if _shared_executor is not None:
            try:
                _shared_executor.shutdown(wait=False)
            except Exception:
                pass
        _shared_executor = None
        _executor_shutdown_requested = True


# Register cleanup function for module reload (hot reload support)
import atexit
atexit.register(_shutdown_shared_executor)


class IFCConstants:
    """All IFC dimension constants in millimeters for consistency.
    
    Values are converted from STANDARDS (meters) to millimeters.
    Values are converted to meters only at IFC entity creation.
    
    Note: GEOMETRY_SIMPLIFICATION_WARNING_THRESHOLD is now in PipelineConfig
    as the single source of truth. Use PipelineConfig.default().geometry_simplification_warning_threshold
    or access via config parameter.
    """
    # Convert from STANDARDS (meters) to millimeters
    LINING_THICKNESS_MM = STANDARDS["WINDOW_FRAME_WIDTH"] * 1000.0  # Convert m to mm
    LINING_DEPTH_MM = STANDARDS["WINDOW_FRAME_DEPTH"] * 1000.0  # Convert m to mm
    PANEL_THICKNESS_MM = STANDARDS["WINDOW_FRAME_WIDTH"] * 1000.0  # Convert m to mm (using frame width as panel thickness)
    FLOOR_THICKNESS_MM = STANDARDS["SLAB_THICKNESS"] * 1000.0  # Convert m to mm
    BUFFER_FACTOR = 0.5  # Half wall thickness for floor buffer (dimensionless)
    # Deprecated: Use PipelineConfig.geometry_simplification_warning_threshold instead
    # Kept for backward compatibility - will be removed in future version
    GEOMETRY_SIMPLIFICATION_WARNING_THRESHOLD = 0.2  # 20% area difference threshold


@dataclass
class IFCExportConfig:
    """Immutable Config-Objekt für IFC-Erzeugung.
    
    Configuration object for IFC export with sensible defaults.
    """
    schema: str = "IFC4"
    geometry_fidelity: str | None = None  # "LOW", "MEDIUM", "HIGH", "LOSSLESS"
    gap_closure_mode: str | None = None  # "propose", "repair_and_mark", "silent_repair"
    
    # Material-Mappings
    external_wall_material: str = "Masonry"
    internal_wall_material: str = "Gypsum"
    
    # Validierung
    validate_schema: bool = True
    validate_topology: bool = True


@dataclass
class WallProfile:
    points: Sequence[Sequence[float]]
    height: float
    uniform_thickness_m: float
    name: str | None = None
    is_external: bool = False
    type_name: str | None = None
    description: str | None = None
    preserve_exact_geometry: bool = False
    gap_repair_info: dict[str, any] | None = None  # Gap repair metadata if wall was gap-repaired


@dataclass
class DoorProfile:
    width: float
    height: float
    thickness: float
    location: Sequence[float] = (0.0, 0.0, 0.0)
    name: str | None = None
    type_name: str | None = None
    operation_type: str = "SINGLE_SWING_LEFT"
    is_external: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate that all dimensions are positive."""
        if self.width <= 0:
            raise ValueError(f"Door width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Door height must be positive, got {self.height}")
        if self.thickness <= 0:
            raise ValueError(f"Door thickness must be positive, got {self.thickness}")


@dataclass
class WindowProfile:
    width: float
    height: float
    thickness: float
    location: Sequence[float] = (0.0, 0.0, 0.0)
    name: str | None = None
    type_name: str | None = None
    partitioning_type: str = "SINGLE_PANEL"
    panel_position: str = "MIDDLE"
    is_external: bool = False
    description: str | None = None
    lining_thickness: float = IFCConstants.LINING_THICKNESS_MM  # Keep in mm, convert to m only at IFC write
    lining_depth: float = IFCConstants.LINING_DEPTH_MM  # Keep in mm, convert to m only at IFC write
    panel_thickness: float = IFCConstants.PANEL_THICKNESS_MM  # Keep in mm, convert to m only at IFC write
    operation_type: str = "SINGLE_PANEL"

    def __post_init__(self) -> None:
        """Validate that all dimensions are positive."""
        if self.width <= 0:
            raise ValueError(f"Window width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Window height must be positive, got {self.height}")
        if self.thickness <= 0:
            raise ValueError(f"Window thickness must be positive, got {self.thickness}")


@dataclass
class SpaceProfile:
    points: Sequence[Sequence[float]]
    height: float
    name: str | None = None
    long_name: str | None = None
    is_external: bool = False
    description: str | None = None


@dataclass
class SlabProfile:
    points: Sequence[Sequence[float]]
    thickness: float
    name: str | None = None
    is_external: bool = False
    description: str | None = None


@dataclass
class IFCV2Builder:
    schema: str = "IFC4"
    config: IFCExportConfig | None = None
    model: ifcopenshell.file = field(init=False)
    project: ifcopenshell.entity_instance = field(init=False)
    site: ifcopenshell.entity_instance = field(init=False)
    building: ifcopenshell.entity_instance = field(init=False)
    storey: ifcopenshell.entity_instance = field(init=False)
    context: ifcopenshell.entity_instance = field(init=False)
    body_context: ifcopenshell.entity_instance = field(init=False)
    walls: List[Any] = field(default_factory=list)
    doors: List[Any] = field(default_factory=list)
    windows: List[Any] = field(default_factory=list)
    spaces_created: List[Any] = field(default_factory=list)
    openings: List[Any] = field(default_factory=list)
    slabs: List[Any] = field(default_factory=list)
    # Store original polygons for spatial queries
    space_polygons: Dict[Any, Polygon] = field(default_factory=dict)  # Map space entity -> polygon
    wall_polygons: List[Polygon] = field(default_factory=list)  # Wall polygons in same order as walls

    def __post_init__(self) -> None:
        # Use config if provided, otherwise use defaults
        if self.config:
            self.schema = self.config.schema
        # Model initialization moved to __enter__ for context manager support
        pass

    def __enter__(self) -> "IFCV2Builder":
        """Context manager entry: initialize IFC model."""
        # ifcopenshell.api.project.create_file expects a `version` argument, not `schema`
        # Use the configured schema (e.g. "IFC4") as version for maximum compatibility.
        schema_version = self.schema or "IFC4"
        self.model = ifcopenshell.api.run("project.create_file", version=schema_version)
        setup_owner_history_v2(
            self.model,
            application_identifier="bimify-pipeline-v4",
            application_full_name="Bimify Pipeline V4.0",
            application_version="4.0.0",
        )
        self.project, self.site, self.building, self.storey = create_project_structure(self.model)
        setup_units_complete(self.model, project=self.project)
        self.context, self.body_context = setup_geometric_context_v2(self.model)
        self._provenance_added = False  # Flag to ensure provenance is added only once
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: clean up references.
        
        Exception-safe cleanup: all deletions are wrapped in try/except to ensure
        cleanup continues even if one deletion fails.
        """
        # ifcopenshell doesn't have explicit close(), but we can clear references
        # Use try/except for each deletion to ensure robust cleanup
        attrs_to_clean = ["model", "project", "site", "building", "storey", "context", "body_context"]
        for attr in attrs_to_clean:
            try:
                if hasattr(self, attr):
                    delattr(self, attr)
            except Exception as e:
                # Log but don't fail - continue cleanup
                logger.debug(f"Failed to delete {attr} in IFCV2Builder.__exit__: {e}")

    # --- Basic geometry helpers -------------------------------------------------

    def _direction(self, values: Sequence[float]) -> Any:
        return self.model.create_entity("IfcDirection", DirectionRatios=tuple(float(v) for v in values))

    def _point(self, values: Sequence[float]) -> Any:
        coords = list(values) + [0.0] * (3 - len(values))
        return self.model.create_entity("IfcCartesianPoint", Coordinates=tuple(float(c) for c in coords[:3]))

    def _create_ifccartesianpoint(self, x: float, y: float, z: float = 0.0) -> Any:
        return self.model.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y), float(z)))

    def _axis2placement(self, location: Sequence[float]) -> Any:
        return self.model.create_entity(
            "IfcAxis2Placement3D",
            Location=self._point(location),
            Axis=self._direction((0.0, 0.0, 1.0)),
            RefDirection=self._direction((1.0, 0.0, 0.0)),
        )

    def _ensure_material(self, name: str, category: str = "Masonry") -> Any:
        """Create or retrieve standard material for thermal calculations.
        
        Args:
            name: Material name (e.g., "Masonry", "Concrete", "Glass")
            category: Material category
            
        Returns:
            IfcMaterial entity
        """
        # Check if material already exists
        existing = next(
            (m for m in self.model.by_type("IfcMaterial") if getattr(m, "Name", None) == name),
            None
        )
        if existing:
            return existing
        
        # Create new material
        material = ifcopenshell.api.run(
            "material.add_material",
            self.model,
            name=name,
            category=category,
        )
        
        # Add material properties for thermal calculations
        try:
            pset = ifcopenshell.api.run("pset.add_pset", self.model, product=material, name="Pset_MaterialCommon")
            # Standard material properties (can be customized per material type)
            props = {
                "Name": name,
            }
            if category == "Masonry":
                props.update({
                    "MassDensity": STANDARDS["MATERIAL_MASONRY_DENSITY"],  # kg/m³
                    "ThermalConductivity": STANDARDS["MATERIAL_MASONRY_THERMAL_CONDUCTIVITY"],  # W/(m·K)
                })
            elif category == "Concrete":
                props.update({
                    "MassDensity": STANDARDS["MATERIAL_CONCRETE_DENSITY"],  # kg/m³
                    "ThermalConductivity": STANDARDS["MATERIAL_CONCRETE_THERMAL_CONDUCTIVITY"],  # W/(m·K)
                })
            elif category == "Glass":
                props.update({
                    "MassDensity": STANDARDS["MATERIAL_GLASS_DENSITY"],  # kg/m³
                    "ThermalConductivity": STANDARDS["MATERIAL_GLASS_THERMAL_CONDUCTIVITY"],  # W/(m·K)
                })
            
            ifcopenshell.api.run("pset.edit_pset", self.model, pset=pset, properties=props)
        except Exception as e:
            logger.error(f"Could not add material properties for {name}: {e}")
            raise MaterialAssignmentError(f"Failed to assign material properties to {name}: {e}") from e
        
        return material

    # Material mapping configuration for different wall types
    # Format: (material_name, default_thickness_mm, thermal_conductivity_W_per_mK)
    # Use ClassVar to make it a class variable, not a dataclass field
    # Updated to use STANDARDS from config
    MATERIAL_MAP: ClassVar[Dict[str, Tuple[str, float, float]]] = {
        "wall_external": (
            STANDARDS["WALL_MATERIAL_EXTERNAL"], 
            STANDARDS["WALL_EXTERNAL_THICKNESS"] * 1000.0,  # Convert m to mm
            STANDARDS["MATERIAL_MASONRY_THERMAL_CONDUCTIVITY"]
        ),
        "wall_internal": (
            STANDARDS["WALL_MATERIAL_INTERNAL"], 
            STANDARDS["WALL_INTERNAL_THICKNESS"] * 1000.0,  # Convert m to mm
            STANDARDS["MATERIAL_MASONRY_THERMAL_CONDUCTIVITY"]
        ),
        "stair": ("Concrete", 300.0, STANDARDS["MATERIAL_CONCRETE_THERMAL_CONDUCTIVITY"]),
        "slab": (
            STANDARDS["SLAB_MATERIAL"],
            STANDARDS["SLAB_THICKNESS"] * 1000.0,  # Convert m to mm
            STANDARDS["MATERIAL_CONCRETE_THERMAL_CONDUCTIVITY"]
        ),
    }

    def _ensure_wall_type_with_material(self, name: str | None, is_external: bool, thickness_m: float) -> Any:
        """Create or retrieve wall type with material layer set for thermal calculations.
        
        Args:
            name: Wall type name
            is_external: Whether wall is external
            thickness_m: Wall thickness in meters
            
        Returns:
            IfcWallType with attached material layer set
        """
        if not name:
            name = "StandardWallType"
        
        # Get or create wall type
        wall_type = self._ensure_wall_type(name)
        
        # Prüfe, ob bereits eine Material-Zuordnung existiert
        # Check for existing IfcRelAssociatesMaterial relationships
        if hasattr(wall_type, "HasAssociations") and wall_type.HasAssociations:
            for rel in wall_type.HasAssociations:
                if rel.is_a("IfcRelAssociatesMaterial"):
                    # Check if it has a valid material usage
                    relating_material = getattr(rel, "RelatingMaterial", None)
                    if relating_material and hasattr(relating_material, "ForLayerSet"):
                        return wall_type  # Material already attached via proper relationship
        
        # Get material configuration from mapping
        # Ensure boolean conversion to avoid numpy array boolean ambiguity
        is_ext_bool = bool(is_external) if is_external is not None else False
        material_key = "wall_external" if is_ext_bool else "wall_internal"
        material_name, default_thickness_mm, thermal_conductivity = self.MATERIAL_MAP.get(
            material_key,
            self.MATERIAL_MAP["wall_external"]  # Fallback to external wall
        )
        
        # Use actual thickness if provided, otherwise use default from mapping
        actual_thickness_m = thickness_m if thickness_m > 0 else (default_thickness_mm / 1000.0)
        
        # Determine material category based on name
        material_category = "Masonry" if material_name == "Masonry" else "Concrete" if material_name == "Concrete" else "Gypsum"
        
        material = self._ensure_material(material_name, material_category)
        
        # Create material layer with actual thickness
        material_layer = self.model.create_entity(
            "IfcMaterialLayer",
            Material=material,
            LayerThickness=float(actual_thickness_m),
        )
        
        # Create material layer set
        material_set = self.model.create_entity(
            "IfcMaterialLayerSet",
            LayerSetName=f"{name}_Layers",
            MaterialLayers=(material_layer,),
        )
        
        # Create material layer set usage
        material_layer_set_usage = self.model.create_entity(
            "IfcMaterialLayerSetUsage",
            ForLayerSet=material_set,
            LayerSetDirection="AXIS2",
            DirectionSense="POSITIVE",
            OffsetFromReferenceLine=0.0,
        )
        
        # WICHTIG: Verknüpfe über IfcRelAssociatesMaterial
        try:
            # Try using the API first
            try:
                ifcopenshell.api.run(
                    "material.assign_material",
                    self.model,
                    products=[wall_type],
                    type="IfcMaterialLayerSetUsage",
                    material=material_layer_set_usage,
                )
            except Exception as api_exc:
                # Fallback: create IfcRelAssociatesMaterial explicitly
                logger.debug(f"API material.assign_material failed, creating IfcRelAssociatesMaterial explicitly: {api_exc}")
                associates = self.model.create_entity(
                    "IfcRelAssociatesMaterial",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatingMaterial=material_layer_set_usage,
                    RelatedObjects=(wall_type,),
                )
                
                # Füge zu HasAssociations hinzu
                existing = list(getattr(wall_type, "HasAssociations", []) or [])
                existing.append(associates)
                wall_type.HasAssociations = tuple(existing)
        except Exception as e:
            logger.error(f"Could not attach material layer set to wall type {name}: {e}")
            raise MaterialAssignmentError(f"Failed to attach material layer set to wall type {name}: {e}") from e
        
        return wall_type

    def _ensure_wall_type(self, name: str | None) -> Any:
        if not name:
            name = "StandardWallType"
        existing = next((w for w in self.model.by_type("IfcWallType") if getattr(w, "Name", None) == name), None)
        if existing:
            return existing
        wall_type = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcWallType",
            name=name,
        )
        self._safe_set_predefined_type(wall_type, "STANDARD")
        return wall_type

    def _safe_set_predefined_type(self, entity: Any, value: str) -> None:
        try:
            entity.PredefinedType = value
        except Exception as e:
            logger.error(f"IFC Schema error: Could not set PredefinedType to {value}: {e}")
            raise IFCExportError(f"Incompatible IFC schema: Could not set PredefinedType to {value}: {e}") from e

    def _polyline_from_points(self, points: Sequence[Sequence[float]]) -> Any:
        if len(points) < 3:
            raise ValueError("Profile requires at least three points")
        rounded_points: List[Tuple[float, float, float]] = []
        for pt in points:
            if len(pt) < 2:
                raise ValueError("Profile point missing coordinates")
            rounded_points.append((float(pt[0]), float(pt[1]), 0.0))
        if rounded_points[0] != rounded_points[-1]:
            rounded_points.append(rounded_points[0])
        cartesian_points = [self._point(coords) for coords in rounded_points]
        return self.model.create_entity("IfcPolyline", Points=tuple(cartesian_points))

    def _ensure_axis_context(self) -> Any:
        """Ensure Axis subcontext exists for 2D wall axes.
        
        Returns:
            IfcGeometricRepresentationSubContext with ContextIdentifier="Axis"
        """
        # Check if axis context already exists
        subcontexts = self.model.by_type("IfcGeometricRepresentationSubContext")
        axis_context = next(
            (sc for sc in subcontexts if (getattr(sc, "ContextIdentifier", "") or "").lower() == "axis"),
            None
        )
        
        if axis_context is not None:
            return axis_context
        
        # Check if Plan context exists, create if not
        plan_context = next(
            (c for c in self.model.by_type("IfcGeometricRepresentationContext") 
             if getattr(c, "ContextType", "") == "Plan"),
            None
        )
        
        if plan_context is None:
            # Create Plan context
            plan_context = ifcopenshell.api.run(
                "context.add_context",
                self.model,
                context_type="Plan",
            )
        
        # Create Axis subcontext
        axis_context = ifcopenshell.api.run(
            "context.add_context",
            self.model,
            context_type="Plan",
            context_identifier="Axis",
            target_view="GRAPH_VIEW",
            parent=plan_context,
        )
        
        return axis_context

    def _create_wall_axis_representation(
        self,
        wall_points: Sequence[Sequence[float]],
        *,
        image_height: float | None = None,  # kept for backward compatibility (unused)
    ) -> Any | None:
        """Create a minimal 2D axis representation for IfcWallStandardCase."""
        if len(wall_points) < 2:
            logger.error("Wall axis requires at least 2 points")
            return None

        def _unique_point(points: Sequence[Sequence[float]], fallback_index: int) -> Tuple[float, float]:
            idx = min(max(fallback_index, 0), len(points) - 1)
            point = points[idx]
            if len(point) < 2:
                raise ValueError("Wall axis point missing coordinates")
            return float(point[0]), float(point[1])

        try:
            # Prefer axis aligned with longest edge of the polygon's minimum rotated rectangle
            try:
                from shapely.geometry import Polygon as _Polygon
                poly = _Polygon(wall_points)
                rect = poly.minimum_rotated_rectangle if poly.is_valid and not poly.is_empty else None
            except Exception:
                rect = None

            if rect is not None and hasattr(rect, "exterior"):
                rcoords = list(rect.exterior.coords)[:-1]
                if len(rcoords) >= 4:
                    # Find the longest edge of rectangle
                    max_len = 0.0
                    edge = (rcoords[0], rcoords[1])
                    for i in range(4):
                        p0 = rcoords[i]
                        p1 = rcoords[(i + 1) % 4]
                        dx = p1[0] - p0[0]
                        dy = p1[1] - p0[1]
                        L = (dx * dx + dy * dy) ** 0.5
                        if L > max_len:
                            max_len = L
                            edge = (p0, p1)
                    start_xy = (float(edge[0][0]), float(edge[0][1]))
                    end_xy = (float(edge[1][0]), float(edge[1][1]))
                else:
                    start_xy = _unique_point(wall_points, 0)
                    end_xy = _unique_point(wall_points, 1 if len(wall_points) > 1 else len(wall_points) - 1)
            else:
                start_xy = _unique_point(wall_points, 0)
                end_xy = _unique_point(wall_points, 1 if len(wall_points) > 1 else len(wall_points) - 1)

            if start_xy == end_xy and len(wall_points) > 2:
                end_xy = _unique_point(wall_points, len(wall_points) - 1)

            if start_xy == end_xy:
                logger.error("Wall axis start and end points coincide; invalid axis geometry")
                return None

            start_point = self._create_ifccartesianpoint(start_xy[0], start_xy[1])
            end_point = self._create_ifccartesianpoint(end_xy[0], end_xy[1])

            polyline = self.model.create_entity(
                "IfcPolyline",
                Points=(start_point, end_point),
            )

            axis_context = self._ensure_axis_context()
            axis_representation = self.model.create_entity(
                "IfcShapeRepresentation",
                ContextOfItems=axis_context,
                RepresentationIdentifier="Axis",
                RepresentationType="Curve2D",
                Items=(polyline,),
            )

            return axis_representation
        except Exception as exc:
            logger.error("Failed to create wall axis representation: %s", exc)
            return None

    def _create_minimal_axis_fallback(self) -> Any:
        start_point = self._create_ifccartesianpoint(0.0, 0.0)
        end_point = self._create_ifccartesianpoint(1.0, 0.0)
        polyline = self.model.create_entity(
            "IfcPolyline",
            Points=(start_point, end_point),
        )
        axis_context = self._ensure_axis_context()
        return self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=axis_context,
            RepresentationIdentifier="Axis",
            RepresentationType="Curve",
            Items=(polyline,),
        )

    def _validate_before_export(self) -> None:
        """Validate critical IFC entities before writing the file."""
        errors: list[str] = []

        if not getattr(self, "context", None):
            errors.append("IfcGeometricRepresentationContext fehlt")

        if not getattr(self, "storey", None):
            errors.append("IfcBuildingStorey fehlt")

        walls = getattr(self, "walls", [])
        openings = getattr(self, "openings", [])
        if not walls:
            errors.append("Keine Wände erzeugt – IFC-Modell wird nicht geschrieben")

        if openings and not walls:
            errors.append("Öffnungen vorhanden, aber keine Wände – ungültiges Modell")

        for wall in walls:
            representation = getattr(wall, "Representation", None)
            if not representation:
                errors.append(f"Wall {getattr(wall, 'GlobalId', 'unbekannt')} hat keine Representation")
                continue

            reps = getattr(representation, "Representations", None) or []
            has_axis = any(
                getattr(rep, "RepresentationIdentifier", None) == "Axis"
                for rep in reps
            )
            if not has_axis:
                errors.append(f"Wall {getattr(wall, 'GlobalId', 'unbekannt')} hat keine Axis-Representation")

        if errors:
            raise IFCExportError(
                "Validierung vor Export fehlgeschlagen: " + ", ".join(errors)
            )

        logger.info("Pre-Export Validierung erfolgreich abgeschlossen")

    def _validate_after_export(self) -> None:
        """Perform lightweight validations on the in-memory IFC model after export."""
        if not getattr(self, "model", None):
            logger.warning("Kein IFC-Modell für Post-Export-Validierung verfügbar")
            return

        try:
            walls = list(self.model.by_type("IfcWallStandardCase"))
        except Exception:
            walls = []
        try:
            openings = list(self.model.by_type("IfcOpeningElement"))
        except Exception:
            openings = []
        try:
            doors = list(self.model.by_type("IfcDoor"))
        except Exception:
            doors = []
        try:
            windows = list(self.model.by_type("IfcWindow"))
        except Exception:
            windows = []

        logger.info(
            "IFC Export Summary: %d Walls, %d Openings, %d Doors, %d Windows",
            len(walls),
            len(openings),
            len(doors),
            len(windows),
        )

        for opening in openings:
            fills = getattr(opening, "HasFillings", None)
            if not fills:
                logger.warning(
                    "Opening %s (%s) hat keine Filling-Relation",
                    getattr(opening, "Name", "unknown"),
                    getattr(opening, "GlobalId", "unknown"),
                )

        try:
            import ifcopenshell.validate  # type: ignore
            import io
            from contextlib import redirect_stdout

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                ifcopenshell.validate.validate(self.model)
            logger.info("IFC Schema-Validierung (in-memory) erfolgreich")
        except ImportError:
            logger.debug("ifcopenshell.validate nicht verfügbar – Schema-Validierung übersprungen")
        except Exception as exc:
            logger.error("IFC Schema-Validierung fehlgeschlagen: %s", exc)

    def validate_ifc_file(self, filepath: str) -> None:
        """Validate exported IFC file using ifcopenshell.validate if available."""
        try:
            import ifcopenshell.validate  # type: ignore
        except ImportError:
            logger.debug("ifcopenshell.validate not available; skipping validation")
            return

        try:
            logger.info("=== IFC VALIDATION START ===")
            ifcopenshell.validate.validate(filepath)
            logger.info("=== IFC VALIDATION PASSED ===")
        except Exception as exc:
            logger.error("IFC VALIDATION FAILED: %s", exc)
            try:
                with open("ifc_validation_errors.log", "w", encoding="utf-8") as log_file:
                    log_file.write(str(exc))
            except Exception as log_exc:
                logger.debug("Failed to write validation log: %s", log_exc)

    def _create_swept_solid(
        self,
        profile_points: Sequence[Sequence[float]],
        depth: float,
        direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Tuple[Any, Any, Any]:
        polyline = self._polyline_from_points(profile_points)
        profile_def = self.model.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=polyline,
        )
        extruded_solid = self.model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile_def,
            ExtrudedDirection=self._direction(direction),
            Depth=float(depth),
        )
        shape_representation = self.model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.body_context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=(extruded_solid,),
        )
        return profile_def, extruded_solid, shape_representation

    def _create_quantity(self, quantity_type: str, name: str, value: float | None) -> Any | None:
        if value is None:
            return None
        attr_map = {
            "IfcQuantityLength": "LengthValue",
            "IfcQuantityArea": "AreaValue",
            "IfcQuantityVolume": "VolumeValue",
        }
        attribute = attr_map.get(quantity_type)
        if attribute is None:
            return None
        quantity = self.model.create_entity(quantity_type, Name=name)
        setattr(quantity, attribute, float(value))
        return quantity

    def _assign_quantities(self, entity: Any, name: str, quantity_specs: Sequence[Tuple[str, str, float | None]]) -> None:
        quantities = [
            self._create_quantity(quantity_type, quantity_name, value)
            for quantity_name, quantity_type, value in quantity_specs
        ]
        quantities = [q for q in quantities if q is not None]
        if not quantities:
            return
        element_quantity = self.model.create_entity(
            "IfcElementQuantity",
            GlobalId=ifcopenshell.guid.new(),
            Name=name,
            Quantities=tuple(quantities),
        )
        self.model.create_entity(
            "IfcRelDefinesByProperties",
            GlobalId=ifcopenshell.guid.new(),
            RelatedObjects=(entity,),
            RelatingPropertyDefinition=element_quantity,
        )

    @staticmethod
    def _polygon_area(points: Sequence[Sequence[float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for idx, current in enumerate(points):
            nxt = points[(idx + 1) % len(points)]
            x1, y1 = current[:2]
            x2, y2 = nxt[:2]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0

    @staticmethod
    def _edge_length(p1: Sequence[float], p2: Sequence[float]) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def _convert_wall_to_4point_quadrilateral(
        polygon: Polygon, uniform_thickness_m: float
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Convert wall polygon to 4-point quadrilateral using minimum rotated rectangle.
        
        Maintains centroid and orientation from original polygon.
        Aligns angle to nearest 0°/90° for orthogonal plans.
        
        Args:
            polygon: Input wall polygon
            uniform_thickness_m: Uniform wall thickness in meters
            
        Returns:
            Tuple of (4 corner points, validated uniform thickness)
        """
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Invalid polygon for wall conversion")
        
        # Calculate original area for quality check
        original_area = polygon.area
        
        # Get minimum rotated rectangle
        rect = polygon.minimum_rotated_rectangle
        
        # Check area difference to detect significant geometry simplification
        # Use PipelineConfig as single source of truth (backward compatible with IFCConstants)
        threshold = PipelineConfig.default().geometry_simplification_warning_threshold
        if original_area > 1e-6:  # Avoid division by zero
            area_diff = abs(original_area - rect.area) / original_area
            if area_diff > threshold:
                logger.warning(
                    f"Wall geometry strongly simplified. "
                    f"Original area: {original_area:.2f}, Rectangle area: {rect.area:.2f}, "
                    f"Difference: {area_diff * 100:.1f}%"
                )
        
        # Extract 4 corner points (exterior coords, excluding duplicate last point)
        coords = list(rect.exterior.coords)
        if len(coords) < 5:
            raise ValueError("Minimum rotated rectangle has insufficient points")
        
        # Get 4 unique corners (exclude last duplicate)
        corners = coords[:4]
        
        # Calculate angle of first edge to align to nearest 0°/90°
        if len(corners) >= 2:
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            # Snap to nearest 0°/90° (±45° threshold)
            snapped_angle_deg = round(angle_deg / 90.0) * 90.0
            
            if abs(angle_deg - snapped_angle_deg) < 45.0:
                # Rotate corners to align with cardinal directions
                angle_diff_rad = math.radians(snapped_angle_deg - angle_deg)
                cos_a = math.cos(angle_diff_rad)
                sin_a = math.sin(angle_diff_rad)
                
                # Get centroid for rotation
                centroid = rect.centroid
                cx, cy = centroid.x, centroid.y
                
                # Rotate corners around centroid
                rotated_corners = []
                for x, y in corners:
                    rx = x - cx
                    ry = y - cy
                    nx = rx * cos_a - ry * sin_a + cx
                    ny = rx * sin_a + ry * cos_a + cy
                    rotated_corners.append((nx, ny))
                corners = rotated_corners
        
        # Ensure we have exactly 4 points
        if len(corners) != 4:
            # Fallback: use bounding box
            minx, miny, maxx, maxy = polygon.bounds
            corners = [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
            ]
        
        # Validate uniform thickness
        validated_thickness = max(float(uniform_thickness_m), 0.01)  # Minimum 1cm
        
        return corners, validated_thickness

    def _create_wall_base(
        self,
        profile: WallProfile,
        wall_points: list,
        uniform_thickness_m: float,
        wall_type_name: str = "complex",
    ) -> tuple[Any, Any]:
        """Base method to create wall entity with common setup (placement, material, representation).
        
        This method extracts the common logic shared between _create_complex_wall and _create_simple_wall
        to reduce code duplication (DRY principle).
        
        Args:
            profile: WallProfile with wall properties
            wall_points: List of corner points for the wall
            uniform_thickness_m: Wall thickness in meters
            wall_type_name: Type name for error messages ("complex" or "simple")
            
        Returns:
            Tuple of (wall_entity, wall_type_entity)
        """
        # Create wall type with material layer set
        wall_type = self._ensure_wall_type_with_material(
            profile.type_name,
            profile.is_external,
            uniform_thickness_m
        )

        # root.create_entity API only accepts ifc_class, predefined_type, and name
        # description must be set separately after creation
        wall = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcWallStandardCase",
            name=profile.name or "Wall",
        )
        # Set description separately if provided
        if profile.description:
            try:
                wall.Description = profile.description
            except Exception as desc_exc:
                logger.debug(f"Could not set description for wall {profile.name}: {desc_exc}")
        try:
            wall.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for {wall_type_name} wall {profile.name}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for {wall_type_name} wall {profile.name}: {e}") from e
        self._safe_set_predefined_type(wall, "STANDARD")

        ifcopenshell.api.run("spatial.assign_container", self.model, products=[wall], relating_structure=self.storey)
        # type.assign_type API expects related_objects (list) not related_object (single)
        ifcopenshell.api.run("type.assign_type", self.model, related_objects=[wall], relating_type=wall_type)

        # Ensure horizontal bottom: Z=0, axis=(0,0,1), ref_direction=(1,0,0)
        placement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=self.storey.ObjectPlacement,
            RelativePlacement=self._axis2placement((0.0, 0.0, 0.0)),
        )
        wall.ObjectPlacement = placement

        # Create swept solid representation
        profile_def, extruded_solid, shape_representation = self._create_swept_solid(
            wall_points,
            depth=profile.height,
            direction=(0.0, 0.0, 1.0),
        )
        
        if self.context is None:
            raise ValueError("CRITICAL: self.context is None! Viewer will crash.")

        # Create axis representation (2D polyline from start to end point)
        axis_representation = self._create_wall_axis_representation(
            wall_points,
            image_height=getattr(self, "image_height", getattr(self, "default_image_height", None)),
        )
        if axis_representation is None:
            logger.error(
                "AXIS REPRESENTATION IS NONE - VIEWER WILL CRASH! Providing fallback for wall %s",
                profile.name or "Unnamed",
            )
            logger.error("Wall points: %s", wall_points)
            axis_representation = self._create_minimal_axis_fallback()
        else:
            logger.info(
                "SUCCESS: Created axis representation for wall %s",
                profile.name or "Unnamed",
            )
            logger.debug("Axis rep: %s", axis_representation)
            logger.debug("Axis rep items: %s", getattr(axis_representation, "Items", None))
            logger.debug("Axis rep context: %s", getattr(axis_representation, "ContextOfItems", None))
        
        # Combine Body and Axis representations
        representations = [shape_representation]
        if axis_representation is not None:
            representations.append(axis_representation)
        
        wall.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=tuple(representations),
        )
        
        return wall, wall_type

    def _create_complex_wall(self, profile: WallProfile) -> Any:
        """Create wall with complex geometry using IfcBooleanResult for non-orthogonal shapes.
        
        For walls with preserve_exact_geometry=True or >8 points, this method preserves
        the exact geometry instead of simplifying to 4-point quadrilaterals.
        
        Args:
            profile: WallProfile with complex geometry
            
        Returns:
            IfcWallStandardCase entity
        """
        # For complex walls, we use the original polygon directly
        wall_points = list(profile.points)
        uniform_thickness_m = profile.uniform_thickness_m
        
        # Use base method for common setup
        wall, _ = self._create_wall_base(profile, wall_points, uniform_thickness_m, "complex")
        
        return wall

    def _create_simple_wall(self, profile: WallProfile, wall_points: list, uniform_thickness_m: float) -> Any:
        """Create simple 4-point orthogonal wall.
        
        Args:
            profile: WallProfile
            wall_points: List of 4 corner points
            uniform_thickness_m: Wall thickness in meters
            
        Returns:
            IfcWallStandardCase entity
        """
        # Use base method for common setup
        wall, _ = self._create_wall_base(profile, wall_points, uniform_thickness_m, "simple")
        
        return wall

    def create_wall(self, profile: WallProfile) -> Any:
        """Create wall, routing to simple or complex geometry method as appropriate.
        
        Args:
            profile: WallProfile with wall geometry and properties
            
        Returns:
            IfcWallStandardCase entity
        """
        if len(profile.points) < 3:
            raise ValueError("Wall profile requires at least three points")
        
        # Für einfache orthogonale Wände: Standard-Extrusion
        # Für komplexe Wände: IfcBooleanResult aus Solids
        # For simple orthogonal walls: Standard extrusion
        # For complex walls: IfcBooleanResult from solids
        if profile.preserve_exact_geometry or len(profile.points) > 8:
            wall = self._create_complex_wall(profile)
            wall_points = list(profile.points)
            uniform_thickness_m = profile.uniform_thickness_m
        else:
            # Convert to 4-point quadrilateral if needed
            wall_points = list(profile.points)
            if len(wall_points) != 4 and not profile.preserve_exact_geometry:
                # Create polygon from points and convert to 4-point quadrilateral
                # Only do this if preserve_exact_geometry is False
                poly = Polygon([(pt[0], pt[1]) for pt in wall_points])
                wall_points, validated_thickness = self._convert_wall_to_4point_quadrilateral(
                    poly, profile.uniform_thickness_m
                )
                uniform_thickness_m = validated_thickness
            else:
                # Preserve exact points when preserve_exact_geometry is True
                wall_points = list(profile.points)  # DEINE Punkte bleiben!
                uniform_thickness_m = profile.uniform_thickness_m  # DEINE Dicke!
            
            # Validate point count
            if len(wall_points) != 4:
                raise ValueError("Wall must have exactly 4 corner points for simple geometry")
            
            wall = self._create_simple_wall(profile, wall_points, uniform_thickness_m)

        pset = ifcopenshell.api.run("pset.add_pset", self.model, product=wall, name="Pset_WallCommon")
        # Determine thermal transmittance based on wall type
        thermal_transmittance = (
            STANDARDS["WALL_THERMAL_TRANSMITTANCE_EXTERNAL"] 
            if profile.is_external 
            else STANDARDS["WALL_THERMAL_TRANSMITTANCE_INTERNAL"]
        )
        ifcopenshell.api.run(
            "pset.edit_pset",
            self.model,
            pset=pset,
            properties={
                "IsExternal": bool(profile.is_external),
                "LoadBearing": STANDARDS["WALL_LOAD_BEARING"],  # IfcBoolean
                "FireRating": STANDARDS["WALL_FIRE_RATING"],  # IfcLabel
                "ThermalTransmittance": float(thermal_transmittance),  # IfcReal (U-Wert in W/m²K)
                "Reference": profile.name or "Wall",
            },
        )
        
        # Add gap repair PropertySet if wall was gap-repaired
        if profile.gap_repair_info is not None:
            gap_info = profile.gap_repair_info
            gap_pset = ifcopenshell.api.run("pset.add_pset", self.model, product=wall, name="Bimify_GapRepair")
            ifcopenshell.api.run(
                "pset.edit_pset",
                self.model,
                pset=gap_pset,
                properties={
                    "IsArtificiallyClosed": True,
                    "OriginalGapWidth_mm": float(gap_info.get("original_gap_width_mm", 0.0)),
                    "ClosureConfidence": float(gap_info.get("closure_confidence", 0.0)),
                    "ManualReviewRequired": bool(gap_info.get("manual_review_required", False)),
                },
            )

        # Calculate quantities using uniform thickness
        length = self._edge_length(wall_points[0], wall_points[1]) if len(wall_points) >= 2 else None
        area = self._polygon_area(wall_points)
        # Use uniform thickness instead of area-based calculation
        volume = area * profile.height if profile.height is not None else None
        self._assign_quantities(
            wall,
            "BaseQuantities",
            [
                ("Length", "IfcQuantityLength", length),
                ("Height", "IfcQuantityLength", profile.height),
                ("Width", "IfcQuantityLength", uniform_thickness_m),
                ("GrossArea", "IfcQuantityArea", area),
                ("GrossVolume", "IfcQuantityVolume", volume),
            ],
        )

        self.walls.append(wall)
        return wall

    def _create_opening_filling(
        self,
        wall: Any,
        profile: DoorProfile | WindowProfile,
        ifc_class: str,
        type_name: str,
        type_entity: Any,
        pset_name: str,
        default_name: str,
    ) -> tuple[Any, Any]:
        """Create opening and filling (door/window) with common logic.
        
        Args:
            wall: Wall entity to attach opening to
            profile: DoorProfile or WindowProfile
            ifc_class: "IfcDoor" or "IfcWindow"
            type_name: Type name for logging
            type_entity: Door or window type entity
            pset_name: Property set name ("Pset_DoorCommon" or "Pset_WindowCommon")
            default_name: Default name ("Door" or "Window")
            
        Returns:
            Tuple of (filling_entity, opening_entity)
        """
        if wall is None:
            raise IFCExportError("Cannot create opening/filling without a host wall")

        # Create filling entity (door or window)
        filling = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class=ifc_class,
            name=profile.name or default_name,
            description=profile.description,
        )
        try:
            filling.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for {type_name.lower()} {profile.name}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for {type_name.lower()} {profile.name}: {e}") from e
        try:
            filling.OverallHeight = float(profile.height)
        except Exception as e:
            logger.warning(f"Could not set OverallHeight for {type_name.lower()} {profile.name}: {e}")
        try:
            filling.OverallWidth = float(profile.width)
        except Exception as e:
            logger.warning(f"Could not set OverallWidth for {type_name.lower()} {profile.name}: {e}")

        ifcopenshell.api.run("spatial.assign_container", self.model, products=[filling], relating_structure=self.storey)
        # type.assign_type API expects related_objects (list) not related_object (single)
        ifcopenshell.api.run("type.assign_type", self.model, related_objects=[filling], relating_type=type_entity)

        # Get wall thickness from wall entity (from quantities or material layer set)
        wall_thickness_m = profile.thickness  # Default to profile thickness
        try:
            # Try to get wall thickness from quantities
            from ifcopenshell.util import element as ifc_element_utils
            wall_psets = ifc_element_utils.get_psets(wall, should_inherit=False)
            wall_quantities = ifc_element_utils.get_psets(wall, qtos_only=True)
            for qto_name, qto_props in wall_quantities.items():
                if "Width" in qto_props:
                    wall_thickness_m = float(qto_props["Width"])
                    break
            # If not found in quantities, try material layer set
            if wall_thickness_m == profile.thickness:
                for rel in getattr(wall, "HasAssociations", []) or []:
                    if rel.is_a("IfcRelAssociatesMaterial"):
                        material_usage = getattr(rel, "RelatingMaterial", None)
                        if material_usage and hasattr(material_usage, "ForLayerSet"):
                            layer_set = material_usage.ForLayerSet
                            if hasattr(layer_set, "MaterialLayers"):
                                total_thickness = sum(
                                    float(layer.LayerThickness) 
                                    for layer in layer_set.MaterialLayers 
                                    if hasattr(layer, "LayerThickness")
                                )
                                if total_thickness > 0:
                                    wall_thickness_m = total_thickness
                                    break
        except Exception as e:
            logger.debug(f"Could not determine wall thickness, using profile.thickness: {e}")
        
        # Determine opening depth and Z-position based on type
        is_window = ifc_class == "IfcWindow"
        if is_window:
            # For windows: opening depth = wall thickness, position at sill height (0.9m)
            opening_depth = wall_thickness_m
            opening_z_position = STANDARDS["WINDOW_SILL_HEIGHT"]  # 0.9m
        else:
            # For doors: opening depth = wall thickness, position at OKFF (0.0m)
            opening_depth = wall_thickness_m
            opening_z_position = 0.0
        try:
            opening_depth = max(float(opening_depth), float(MIN_OPENING_DEPTH))
        except Exception:
            opening_depth = float(MIN_OPENING_DEPTH)
        
        # Berechne Opening-Placement RELATIV zur Wand
        # Calculate opening placement relative to wall's local coordinate system
        wall_placement = getattr(wall, "ObjectPlacement", None) or self.storey.ObjectPlacement
        if wall_placement:
            try:
                import ifcopenshell.util.placement as placement_util
                import numpy as np
                
                # Get wall transformation matrix
                wall_matrix = placement_util.get_local_placement(wall_placement)
                
                # Transform opening coordinates from global to wall-local space
                # profile.location is in global coordinates, we need wall-local
                # Adjust Z coordinate for window sill height or door OKFF
                global_z = opening_z_position if len(profile.location) < 3 else profile.location[2]
                global_point = np.array([profile.location[0], profile.location[1], global_z, 1.0])
                
                # Transform to wall-local coordinates (inverse of wall matrix)
                wall_matrix_inv = np.linalg.inv(wall_matrix)
                local_point = wall_matrix_inv @ global_point
                
                local_origin = (float(local_point[0]), float(local_point[1]), float(local_point[2]))
            except Exception as e:
                logger.debug(f"Could not transform opening coordinates to wall-local space: {e}, using profile.location with adjusted Z")
                # Fallback: use profile location but adjust Z
                local_origin = (
                    profile.location[0] if len(profile.location) > 0 else 0.0,
                    profile.location[1] if len(profile.location) > 1 else 0.0,
                    opening_z_position
                )
        else:
            local_origin = (
                profile.location[0] if len(profile.location) > 0 else 0.0,
                profile.location[1] if len(profile.location) > 1 else 0.0,
                opening_z_position
            )

        filling.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=wall_placement,
            RelativePlacement=self._axis2placement(local_origin),
        )

        rectangle = (
            (0.0, 0.0),
            (profile.width, 0.0),
            (profile.width, opening_depth),
            (0.0, opening_depth),
        )

        # Use a thin panel for door/window filling geometry instead of full height extrusion
        height = float(profile.height)
        panel_depth = height * 0.05 if height > 0 else 0.05
        panel_depth = min(panel_depth, 0.05)
        panel_depth = max(panel_depth, 0.01)

        _, _, filling_shape = self._create_swept_solid(
            rectangle,
            depth=panel_depth,
            direction=(0.0, 0.0, 1.0),
        )

        # Create opening element
        opening = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcOpeningElement",
            name=f"Opening_{profile.name or default_name}",
            description=profile.description,
        )
        try:
            opening.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for opening {profile.name or default_name}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for opening {profile.name or default_name}: {e}") from e
        
        # Set opening OverallWidth and OverallHeight
        try:
            opening.OverallWidth = float(profile.width)
        except Exception as e:
            logger.warning(f"Could not set OverallWidth for opening: {e}")
        try:
            opening.OverallHeight = float(profile.height)
        except Exception as e:
            logger.warning(f"Could not set OverallHeight for opening: {e}")
        
        # Use same transformed coordinates for opening
        opening.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=wall_placement,
            RelativePlacement=self._axis2placement(local_origin),
        )
        # Opening depth = wall thickness (not window/door height!)
        _, _, opening_shape = self._create_swept_solid(
            rectangle,
            depth=opening_depth,  # CRITICAL: Use wall thickness, not profile.height
            direction=(0.0, 0.0, 1.0),
        )
        opening.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=(opening_shape,),
        )

        # Use ifcopenshell API for automatic relation management
        ifcopenshell.api.run("void.add_opening", self.model, element=wall, opening=opening)
        ifcopenshell.api.run("opening.add_filling", self.model, opening=opening, filling=filling)

        # Create property set with type-specific properties
        pset = ifcopenshell.api.run("pset.add_pset", self.model, product=filling, name=pset_name)
        
        if pset_name == "Pset_WindowCommon":
            # Window-specific properties
            window_props = {
                "IsExternal": bool(profile.is_external),  # IfcBoolean
                "UValue": float(STANDARDS["WINDOW_U_VALUE"]),  # IfcReal (W/m²K)
                "GlazingAreaFraction": float(STANDARDS["WINDOW_GLASS_AREA_RATIO"]),  # IfcReal (0.8 = 80%)
                "FrameDepth": float(STANDARDS["WINDOW_FRAME_DEPTH"]),  # IfcReal (0.09m)
                "FrameThickness": float(STANDARDS["WINDOW_FRAME_WIDTH"]),  # IfcReal (0.07m)
                "Reference": profile.name or default_name,
            }
            ifcopenshell.api.run(
                "pset.edit_pset",
                self.model,
                pset=pset,
                properties=window_props,
            )
        elif pset_name == "Pset_DoorCommon":
            # Door-specific properties
            door_props = {
                "FireRating": STANDARDS["DOOR_FIRE_RATING"],  # IfcLabel ("T30")
                "HandicapAccessible": STANDARDS["DOOR_HANDICAP_ACCESSIBLE"],  # IfcBoolean
                "IsExternal": bool(profile.is_external),  # IfcBoolean
                "Reference": profile.name or default_name,
            }
            ifcopenshell.api.run(
                "pset.edit_pset",
                self.model,
                pset=pset,
                properties=door_props,
            )
        else:
            # Fallback for unknown pset types
            ifcopenshell.api.run(
                "pset.edit_pset",
                self.model,
                pset=pset,
                properties={
                    "IsExternal": bool(profile.is_external),
                    "Reference": profile.name or default_name,
                },
            )

        # Assign quantities
        area = profile.width * profile.height
        volume = area * panel_depth
        self._assign_quantities(
            filling,
            "BaseQuantities",
            [
                ("Width", "IfcQuantityLength", profile.width),
                ("Height", "IfcQuantityLength", profile.height),
                ("Depth", "IfcQuantityLength", panel_depth),
                ("GrossArea", "IfcQuantityArea", area),
                ("GrossVolume", "IfcQuantityVolume", volume),
            ],
        )

        return filling, opening

    def _ensure_door_type(self, profile: DoorProfile) -> Any:
        name = profile.type_name or "StandardDoorType"
        existing = next((d for d in self.model.by_type("IfcDoorType") if getattr(d, "Name", None) == name), None)
        if existing:
            return existing
        door_type = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcDoorType",
            name=name,
        )
        self._safe_set_predefined_type(door_type, "DOOR")
        try:
            door_type.OperationType = profile.operation_type
        except Exception as e:
            logger.warning(f"Could not set OperationType for door type {name}: {e}")
        return door_type

    def create_door(self, wall: Any, profile: DoorProfile) -> Any:
        door_type = self._ensure_door_type(profile)
        door, opening = self._create_opening_filling(
            wall=wall,
            profile=profile,
            ifc_class="IfcDoor",
            type_name="Door",
            type_entity=door_type,
            pset_name="Pset_DoorCommon",
            default_name="Door",
        )
        self.doors.append(door)
        self.openings.append(opening)
        return door, opening

    def _ensure_window_type(self, profile: WindowProfile) -> Any:
        name = profile.type_name or "StandardWindowType"
        existing = next((w for w in self.model.by_type("IfcWindowType") if getattr(w, "Name", None) == name), None)
        if existing:
            return existing
        window_type = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcWindowType",
            name=name,
        )
        self._safe_set_predefined_type(window_type, "WINDOW")
        try:
            window_type.PartitioningType = profile.partitioning_type
        except Exception as e:
            logger.warning(f"Could not set PartitioningType for window type {name}: {e}")

        # Convert from mm to meters for IFC API (IFC uses meters)
        lining = self.model.create_entity(
            "IfcWindowLiningProperties",
            LiningDepth=float(profile.lining_depth / 1000.0),  # Convert mm to m
            LiningThickness=float(profile.lining_thickness / 1000.0),  # Convert mm to m
        )
        panel = self.model.create_entity(
            "IfcWindowPanelProperties",
            OperationType=profile.operation_type,
            PanelPosition=profile.panel_position,
            FrameDepth=float(profile.panel_thickness / 1000.0),  # Convert mm to m
            FrameThickness=float(profile.panel_thickness / 1000.0),  # Convert mm to m
        )
        window_type.HasPropertySets = tuple(filter(None, (lining, panel)))
        return window_type

    def create_window(self, wall: Any, profile: WindowProfile) -> Any:
        window_type = self._ensure_window_type(profile)
        window, opening = self._create_opening_filling(
            wall=wall,
            profile=profile,
            ifc_class="IfcWindow",
            type_name="Window",
            type_entity=window_type,
            pset_name="Pset_WindowCommon",
            default_name="Window",
        )
        self.windows.append(window)
        self.openings.append(opening)
        return window, opening

    def create_space(self, profile: SpaceProfile) -> Any:
        space = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcSpace",
            name=profile.name or "Space",
            description=profile.description,
        )
        try:
            space.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for space {profile.name}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for space {profile.name}: {e}") from e
        try:
            space.LongName = profile.long_name or space.Name
        except Exception as e:
            logger.warning(f"Could not set LongName for space {profile.name}: {e}")
        try:
            space.CompositionType = "ELEMENT"
        except Exception as e:
            logger.warning(f"Could not set CompositionType for space {profile.name}: {e}")

        ifcopenshell.api.run("spatial.assign_container", self.model, products=[space], relating_structure=self.storey)

        space.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=self.storey.ObjectPlacement,
            RelativePlacement=self._axis2placement((0.0, 0.0, 0.0)),
        )

        profile_def, extruded_solid, shape = self._create_swept_solid(
            profile.points,
            depth=profile.height,
            direction=(0.0, 0.0, 1.0),
        )
        space.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=(shape,),
        )

        pset = ifcopenshell.api.run("pset.add_pset", self.model, product=space, name="Pset_SpaceCommon")
        ifcopenshell.api.run(
            "pset.edit_pset",
            self.model,
            pset=pset,
            properties={
                "IsExternal": bool(profile.is_external),
                "Reference": profile.name or "Space",
            },
        )

        # HottCAD-spezifische Properties für Energiesimulation
        # HottCAD-specific properties for energy simulation
        try:
            # Extract space_type from name if available (e.g., "Office", "Bedroom", "Kitchen")
            space_type = "UNKNOWN"
            if profile.name:
                name_lower = profile.name.lower()
                if any(keyword in name_lower for keyword in ["office", "büro"]):
                    space_type = "OFFICE"
                elif any(keyword in name_lower for keyword in ["bedroom", "schlafzimmer"]):
                    space_type = "BEDROOM"
                elif any(keyword in name_lower for keyword in ["kitchen", "küche"]):
                    space_type = "KITCHEN"
                elif any(keyword in name_lower for keyword in ["bathroom", "badezimmer", "wc"]):
                    space_type = "BATHROOM"
                elif any(keyword in name_lower for keyword in ["living", "wohnzimmer"]):
                    space_type = "LIVING_ROOM"
                else:
                    space_type = "UNKNOWN"
            
            bimify_pset = ifcopenshell.api.run(
                "pset.add_pset", 
                self.model, 
                product=space, 
                name="Bimify_SpaceData"
            )
            ifcopenshell.api.run(
                "pset.edit_pset",
                self.model,
                pset=bimify_pset,
                properties={
                    "RoomCategory": space_type,
                    "DesignHeatingLoad": 0.0,  # W
                    "DesignCoolingLoad": 0.0,  # W
                    "AirChangeRate": 0.5,  # 1/h
                },
            )
        except Exception as e:
            logger.warning(f"Could not add Bimify_SpaceData PropertySet to space {profile.name}: {e}")

        area = self._polygon_area(profile.points)
        volume = area * profile.height
        self._assign_quantities(
            space,
            "BaseQuantities",
            [
                ("GrossArea", "IfcQuantityArea", area),
                ("GrossVolume", "IfcQuantityVolume", volume),
                ("Height", "IfcQuantityLength", profile.height),
            ],
        )

        self.spaces_created.append(space)
        return space

    def create_floor_covering(self, space: Any, profile: SlabProfile) -> Any:
        """Create IfcCovering (floor) for space.
        
        HottCAD expects IfcCovering for space-enclosing surfaces (floor coverings),
        especially for transmission losses.
        
        Args:
            space: IfcSpace entity
            profile: SlabProfile with floor geometry
            
        Returns:
            IfcCovering entity
        """
        covering = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcCovering",
            name=f"Floor_{getattr(space, 'Name', 'Space')}",
            description=profile.description,
        )
        try:
            covering.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for floor covering {getattr(space, 'Name', 'Space')}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for floor covering {getattr(space, 'Name', 'Space')}: {e}") from e
        try:
            covering.PredefinedType = "FLOORING"
        except Exception as e:
            logger.warning(f"Could not set PredefinedType for floor covering: {e}")

        # Assign to space using void.add_filling relationship
        ifcopenshell.api.run("void.add_filling", self.model, element=space, filling=covering)

        covering.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=getattr(space, "ObjectPlacement", None) or self.storey.ObjectPlacement,
            RelativePlacement=self._axis2placement((0.0, 0.0, 0.0)),
        )

        # Create geometry similar to slab
        profile_def, extruded_solid, shape = self._create_swept_solid(
            profile.points,
            depth=profile.thickness,
            direction=(0.0, 0.0, 1.0),
        )
        covering.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=(shape,),
        )

        # Add properties
        pset = ifcopenshell.api.run("pset.add_pset", self.model, product=covering, name="Pset_CoveringCommon")
        ifcopenshell.api.run(
            "pset.edit_pset",
            self.model,
            pset=pset,
            properties={
                "IsExternal": bool(profile.is_external),
                "Reference": f"Floor_{getattr(space, 'Name', 'Space')}",
            },
        )

        # Add quantities
        area = self._polygon_area(profile.points)
        volume = area * profile.thickness
        self._assign_quantities(
            covering,
            "BaseQuantities",
            [
                ("GrossArea", "IfcQuantityArea", area),
                ("GrossVolume", "IfcQuantityVolume", volume),
                ("Thickness", "IfcQuantityLength", profile.thickness),
            ],
        )

        return covering

    def _ensure_slab_material(self, slab: Any, thickness_m: float) -> None:
        """Assign material layer set to slab.
        
        Args:
            slab: IfcSlab entity
            thickness_m: Slab thickness in meters
        """
        # Check if material already assigned
        if hasattr(slab, "HasAssociations") and slab.HasAssociations:
            for rel in slab.HasAssociations:
                if rel.is_a("IfcRelAssociatesMaterial"):
                    relating_material = getattr(rel, "RelatingMaterial", None)
                    if relating_material and hasattr(relating_material, "ForLayerSet"):
                        return  # Material already assigned
        
        # Get material configuration from mapping
        material_name, default_thickness_mm, thermal_conductivity = self.MATERIAL_MAP.get(
            "slab",
            (STANDARDS["SLAB_MATERIAL"], STANDARDS["SLAB_THICKNESS"] * 1000.0, STANDARDS["MATERIAL_CONCRETE_THERMAL_CONDUCTIVITY"])
        )
        
        # Use actual thickness if provided, otherwise use default
        actual_thickness_m = thickness_m if thickness_m > 0 else (default_thickness_mm / 1000.0)
        
        # Create material
        material = self._ensure_material(material_name, "Concrete")
        
        # Create material layer
        material_layer = self.model.create_entity(
            "IfcMaterialLayer",
            Material=material,
            LayerThickness=float(actual_thickness_m),
        )
        
        # Create material layer set
        material_set = self.model.create_entity(
            "IfcMaterialLayerSet",
            LayerSetName="Slab_Layers",
            MaterialLayers=(material_layer,),
        )
        
        # Create material layer set usage
        material_layer_set_usage = self.model.create_entity(
            "IfcMaterialLayerSetUsage",
            ForLayerSet=material_set,
            LayerSetDirection="AXIS3",  # For slabs, direction is vertical (Z-axis)
            DirectionSense="POSITIVE",
            OffsetFromReferenceLine=0.0,
        )
        
        # Assign material via IfcRelAssociatesMaterial
        try:
            ifcopenshell.api.run(
                "material.assign_material",
                self.model,
                products=[slab],
                type="IfcMaterialLayerSetUsage",
                material=material_layer_set_usage,
            )
        except Exception as api_exc:
            # Fallback: create IfcRelAssociatesMaterial explicitly
            logger.debug(f"API material.assign_material failed for slab, creating IfcRelAssociatesMaterial explicitly: {api_exc}")
            associates = self.model.create_entity(
                "IfcRelAssociatesMaterial",
                GlobalId=ifcopenshell.guid.new(),
                RelatingMaterial=material_layer_set_usage,
                RelatedObjects=(slab,),
            )
            # Add to HasAssociations
            existing = list(getattr(slab, "HasAssociations", []) or [])
            existing.append(associates)
            slab.HasAssociations = tuple(existing)

    def create_slab(self, profile: SlabProfile) -> Any:
        """Create IfcSlab entity with uniform thickness and material layer set."""
        if len(profile.points) < 3:
            raise ValueError("Slab profile requires at least three points")
        
        slab = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcSlab",
            name=profile.name or "Slab",
            description=profile.description,
        )
        try:
            slab.GlobalId = ifcopenshell.guid.new()
        except Exception as e:
            logger.error(f"GUID generation failed for slab {profile.name}: {e}")
            raise IFCExportError(f"Critical IFC error: GUID generation failed for slab {profile.name}: {e}") from e
        try:
            slab.PredefinedType = "FLOOR"
        except Exception as e:
            logger.warning(f"Could not set PredefinedType for slab {profile.name}: {e}")

        ifcopenshell.api.run("spatial.assign_container", self.model, products=[slab], relating_structure=self.storey)

        slab.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=self.storey.ObjectPlacement,
            RelativePlacement=self._axis2placement((0.0, 0.0, 0.0)),
        )

        profile_def, extruded_solid, shape = self._create_swept_solid(
            profile.points,
            depth=profile.thickness,
            direction=(0.0, 0.0, 1.0),
        )
        slab.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=(shape,),
        )

        # Assign material layer set to slab
        self._ensure_slab_material(slab, profile.thickness)

        pset = ifcopenshell.api.run("pset.add_pset", self.model, product=slab, name="Pset_SlabCommon")
        ifcopenshell.api.run(
            "pset.edit_pset",
            self.model,
            pset=pset,
            properties={
                "IsExternal": bool(profile.is_external),
                "Reference": profile.name or "Slab",
            },
        )

        area = self._polygon_area(profile.points)
        volume = area * profile.thickness
        self._assign_quantities(
            slab,
            "BaseQuantities",
            [
                ("GrossArea", "IfcQuantityArea", area),
                ("GrossVolume", "IfcQuantityVolume", volume),
                ("Thickness", "IfcQuantityLength", profile.thickness),
            ],
        )

        self.slabs.append(slab)
        return slab

    def create_floor_slab(
        self,
        polygon_points: Sequence[Sequence[float]],
        thickness: float = 0.2,
    ) -> Any:
        """Create an IfcSlab from detected floor polygon."""
        if len(polygon_points) < 3:
            raise ValueError("Floor slab requires at least three points")

        slab = ifcopenshell.api.run(
            "root.create_entity",
            self.model,
            ifc_class="IfcSlab",
            name="Floor Slab",
        )

        try:
            slab.PredefinedType = "FLOOR"
        except Exception as exc:
            logger.debug("Could not set predefined type for floor slab: %s", exc)

        ifcopenshell.api.run(
            "spatial.assign_container",
            self.model,
            products=[slab],
            relating_structure=self.storey,
        )

        slab.ObjectPlacement = self.model.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=self.storey.ObjectPlacement,
            RelativePlacement=self._axis2placement((0.0, 0.0, 0.0)),
        )

        _, _, shape_rep = self._create_swept_solid(
            polygon_points,
            depth=float(thickness),
            direction=(0.0, 0.0, -1.0),
        )

        slab.Representation = self.model.create_entity(
            "IfcProductDefinitionShape",
            Representations=(shape_rep,),
        )

        self.slabs.append(slab)
        return slab

    def _calculate_boundary_area(self, space: Any, building_element: Any) -> float:
        """Calculate intersection area between space and building element.
        
        Args:
            space: IfcSpace entity
            building_element: Wall, Door, or Window entity
            
        Returns:
            Boundary area in square meters (approximate)
        """
        # Simplified calculation: use element dimensions
        # For walls: use wall length * height
        # For doors/windows: use door/window area
        try:
            if hasattr(building_element, "OverallWidth") and hasattr(building_element, "OverallHeight"):
                # Door or window
                width = float(getattr(building_element, "OverallWidth", 0.0) or 0.0)
                height = float(getattr(building_element, "OverallHeight", 0.0) or 0.0)
                return width * height
            else:
                # Wall - approximate from quantities
                quantities = getattr(building_element, "IsDefinedBy", None)
                if quantities:
                    for rel in quantities:
                        if hasattr(rel, "RelatingPropertyDefinition"):
                            prop_def = rel.RelatingPropertyDefinition
                            if hasattr(prop_def, "Quantities"):
                                for qty in prop_def.Quantities:
                                    if hasattr(qty, "Name") and qty.Name == "GrossArea":
                                        if hasattr(qty, "AreaValue"):
                                            return float(qty.AreaValue or 0.0)
        except Exception as e:
            logger.debug(f"Could not calculate boundary area: {e}")
        
        # Fallback: return 0 (boundary will still be created)
        return 0.0

    def _calculate_real_intersection(self, space: Any, building_element: Any) -> Any | None:
        """Calculate real intersection geometry between space and building element.
        
        Args:
            space: IfcSpace entity
            building_element: Wall, Door, or Window entity
            
        Returns:
            IfcConnectionSurfaceGeometry or None if calculation fails
        """
        try:
            # Try to use ifcopenshell.geom to get actual geometries
            import ifcopenshell.geom
            from ifcopenshell.geom import settings as geom_settings
            
            settings = geom_settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            
            # Get space geometry
            try:
                space_shape = ifcopenshell.geom.create_shape(settings, space)
                if not space_shape:
                    return None
            except Exception:
                return None
            
            # Get building element geometry
            try:
                element_shape = ifcopenshell.geom.create_shape(settings, building_element)
                if not element_shape:
                    return None
            except Exception:
                return None
            
            # For now, return None to use fallback - full intersection calculation
            # would require complex geometric operations
            # This is a placeholder for future enhancement
            return None
            
        except Exception as e:
            logger.debug(f"Could not calculate real intersection geometry: {e}")
            return None

    def _create_connection_geometry(self, space: Any, building_element: Any) -> Any | None:
        """Create connection geometry for space boundary.
        
        Args:
            space: IfcSpace entity
            building_element: Wall, Door, or Window entity
            
        Returns:
            IfcConnectionSurfaceGeometry or None if creation fails
        """
        try:
            # Try to calculate real intersection first
            real_geom = self._calculate_real_intersection(space, building_element)
            if real_geom is not None:
                return real_geom
            
            # Fallback: Create simplified surface geometry
            # Get element dimensions for better approximation
            width = 1.0
            height = 1.0
            
            if hasattr(building_element, "OverallWidth") and hasattr(building_element, "OverallHeight"):
                width = float(getattr(building_element, "OverallWidth", 1.0) or 1.0)
                height = float(getattr(building_element, "OverallHeight", 1.0) or 1.0)
            elif hasattr(building_element, "Representation"):
                # Try to extract dimensions from representation
                try:
                    # This is a simplified approach - in production, would extract from geometry
                    pass
                except Exception:
                    pass
            
            # Create surface geometry based on element dimensions
            surface = self.model.create_entity(
                "IfcSurfaceOfLinearExtrusion",
                SweptCurve=self.model.create_entity(
                    "IfcArbitraryClosedProfileDef",
                    ProfileType="AREA",
                    OuterCurve=self._polyline_from_points([(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]),
                ),
                ExtrudedDirection=self._direction((0.0, 0.0, 1.0)),
                Depth=height,
            )
            
            connection_geom = self.model.create_entity(
                "IfcConnectionSurfaceGeometry",
                SurfaceOnRelatingElement=surface,
            )
            
            return connection_geom
        except Exception as e:
            logger.debug(f"Could not create connection geometry: {e}")
            return None

    def _create_space_boundary(
        self,
        space: Any,
        building_element: Any,
        boundary_type: str = "INTERNAL",
    ) -> Any:
        """Create IfcRelSpaceBoundary between space and building element.
        
        Args:
            space: IfcSpace entity
            building_element: Wall, Door, or Window entity
            boundary_type: "INTERNAL", "EXTERNAL", "WINDOW", or "DOOR"
            
        Returns:
            IfcRelSpaceBoundary entity
        """
        # Bestimme korrekte Boundary-Eigenschaften
        # Determine if element is an opening (virtual boundary)
        is_physical = not building_element.is_a("IfcOpeningElement")
        
        # Für Öffnungen: Virtuelle Boundaries!
        # For openings: Virtual boundaries!
        physical_or_virtual = "PHYSICAL" if is_physical else "VIRTUAL"
        
        # Determine if external based on element properties
        is_external = False
        if hasattr(building_element, "IsExternal"):
            is_external = bool(getattr(building_element, "IsExternal", False))
        elif hasattr(building_element, "Pset_WallCommon"):
            try:
                pset = building_element.Pset_WallCommon
                if hasattr(pset, "IsExternal"):
                    is_external = bool(pset.IsExternal)
            except Exception:
                pass
        
        # Determine internal/external boundary
        # Ensure boolean conversion to avoid numpy array boolean ambiguity
        is_ext_bool = bool(is_external) if is_external is not None else False
        if boundary_type in ("WINDOW", "DOOR"):
            internal_or_external = "EXTERNAL" if is_ext_bool else "INTERNAL"
        else:
            internal_or_external = "EXTERNAL" if is_ext_bool else "INTERNAL"
        
        # Berechne echte Schnittgeometrie (statt Dummy)
        # Calculate real intersection geometry (instead of dummy)
        connection_geom = self._create_connection_geometry(space, building_element)
        
        # Create space boundary
        boundary = self.model.create_entity(
            "IfcRelSpaceBoundary",
            GlobalId=ifcopenshell.guid.new(),
            Name=f"Boundary_{getattr(space, 'Name', 'Space')}_{getattr(building_element, 'Name', 'Element')}",
            Description=f"Space boundary between {getattr(space, 'Name', 'Space')} and {getattr(building_element, 'Name', 'Element')}",
            RelatingSpace=space,
            RelatedBuildingElement=building_element,
            ConnectionGeometry=connection_geom,
            PhysicalOrVirtualBoundary=physical_or_virtual,  # WICHTIG!
            InternalOrExternalBoundary=internal_or_external,
            # FEHLT: ParentBoundary für 2nd Level! (Future enhancement)
        )
        
        return boundary

    def _finalize_space_boundaries(self) -> None:
        """Create space boundaries for all spaces and adjacent building elements.
        
        Uses STRtree spatial index for efficient O(log n) queries instead of O(n²) nested loops.
        """
        if not self.spaces_created:
            return
        
        # Build spatial index for walls using stored polygons (O(n log n) construction, O(log n) queries)
        if not self.wall_polygons or len(self.wall_polygons) != len(self.walls):
            # Fallback: create boundaries for all walls
            logger.warning("Wall polygons not available for spatial indexing, using fallback")
            for space in self.spaces_created:
                for wall in self.walls:
                    try:
                        self._create_space_boundary(space, wall, "INTERNAL")
                    except Exception as e:
                        logger.debug(f"Failed to create space boundary for wall {getattr(wall, 'Name', 'unknown')}: {e}")
                for door in self.doors:
                    try:
                        self._create_space_boundary(space, door, "DOOR")
                    except Exception as e:
                        logger.debug(f"Failed to create space boundary for door {getattr(door, 'Name', 'unknown')}: {e}")
                for window in self.windows:
                    try:
                        self._create_space_boundary(space, window, "WINDOW")
                    except Exception as e:
                        logger.debug(f"Failed to create space boundary for window {getattr(window, 'Name', 'unknown')}: {e}")
            return
        
        # Create STRtree for walls with real geometries
        wall_tree = STRtree(self.wall_polygons)
        
        try:
            for space in self.spaces_created:
                # Get space polygon for spatial queries
                space_poly = self.space_polygons.get(space)
                if space_poly is None:
                    # Fallback: create boundaries for all walls if space polygon not available
                    for wall in self.walls:
                        try:
                            self._create_space_boundary(space, wall, "INTERNAL")
                        except Exception as e:
                            logger.debug(f"Failed to create space boundary for wall {getattr(wall, 'Name', 'unknown')}: {e}")
                else:
                    # Use spatial index to find only nearby/intersecting walls (1m buffer for tolerance)
                    nearby_wall_indices = wall_tree.query(space_poly.buffer(1.0))  # 1m = 1000mm buffer
                    
                    # Create boundaries only for intersecting walls
                    for wall_idx in nearby_wall_indices:
                        if wall_idx < len(self.walls):
                            wall = self.walls[wall_idx]
                            wall_poly = self.wall_polygons[wall_idx]
                            # Check actual intersection
                            if space_poly.intersects(wall_poly) or space_poly.buffer(0.1).intersects(wall_poly):
                                try:
                                    self._create_space_boundary(space, wall, "INTERNAL")
                                except Exception as e:
                                    logger.debug(f"Failed to create space boundary for wall {getattr(wall, 'Name', 'unknown')}: {e}")
                
                # Create boundaries for doors (typically fewer, so no spatial index needed)
                for door in self.doors:
                    try:
                        self._create_space_boundary(space, door, "DOOR")
                    except Exception as e:
                        logger.debug(f"Failed to create space boundary for door {getattr(door, 'Name', 'unknown')}: {e}")
                
                # Create boundaries for windows (typically fewer, so no spatial index needed)
                for window in self.windows:
                    try:
                        self._create_space_boundary(space, window, "WINDOW")
                    except Exception as e:
                        logger.debug(f"Failed to create space boundary for window {getattr(window, 'Name', 'unknown')}: {e}")
        finally:
            # Explicit cleanup of STRtree to ensure memory is freed promptly
            if wall_tree is not None:
                del wall_tree

    def add_provenance(
        self,
        *,
        px_per_mm: float | None = None,
        geometry_fidelity: str | None = None,
        gap_closure_mode: str | None = None,
    ) -> None:
        """Add Bimify_Provenance PropertySet to project for audit trail."""
        if self._provenance_added:
            return
        
        try:
            from datetime import datetime
            pset = ifcopenshell.api.run("pset.add_pset", self.model, product=self.project, name="Bimify_Provenance")
            properties = {
                "PipelineVersion": "4.0.0",
                "ExportedAt": datetime.utcnow().isoformat(),
            }
            
            if geometry_fidelity:
                properties["GeometryFidelity"] = str(geometry_fidelity)
            if gap_closure_mode:
                properties["GapClosureMode"] = str(gap_closure_mode)
            if px_per_mm:
                properties["PxPerMm"] = str(px_per_mm)
            
            ifcopenshell.api.run("pset.edit_pset", self.model, pset=pset, properties=properties)
            self._provenance_added = True
        except Exception as e:
            logger.warning(f"Failed to add provenance PropertySet: {e}")

    def _verify_relations(self) -> None:
        """Verify that all required IFC relations exist.
        
        Checks:
        - IfcRelVoidsElement: Each opening must have exactly one void relation to a wall
        - IfcRelFillsElement: Each opening must have exactly one fill relation to a door/window
        - IfcRelContainedInSpatialStructure: All elements must be contained in storey
        - IfcRelAssociatesMaterial: All walls and slabs must have material associations
        - IfcRelDefinesByProperties: All property sets must be linked to entities
        """
        all_openings = self.model.by_type("IfcOpeningElement")
        all_void_rels = self.model.by_type("IfcRelVoidsElement")
        all_fill_rels = self.model.by_type("IfcRelFillsElement")
        all_containment_rels = self.model.by_type("IfcRelContainedInSpatialStructure")
        all_material_rels = self.model.by_type("IfcRelAssociatesMaterial")
        all_property_rels = self.model.by_type("IfcRelDefinesByProperties")
        
        # Verify opening relations
        opening_to_void = {}
        opening_to_fill = {}
        for rel in all_void_rels:
            opening = getattr(rel, "RelatedOpeningElement", None)
            if opening:
                opening_to_void[opening] = rel
        for rel in all_fill_rels:
            opening = getattr(rel, "RelatingOpeningElement", None)
            if opening:
                opening_to_fill[opening] = rel
        
        missing_voids = [op for op in all_openings if op not in opening_to_void]
        missing_fills = [op for op in all_openings if op not in opening_to_fill]
        
        if missing_voids:
            logger.warning(f"Found {len(missing_voids)} openings without IfcRelVoidsElement")
        if missing_fills:
            logger.warning(f"Found {len(missing_fills)} openings without IfcRelFillsElement")
        
        # Verify containment relations
        contained_elements = set()
        for rel in all_containment_rels:
            if getattr(rel, "RelatingStructure", None) == self.storey:
                elements = getattr(rel, "RelatedElements", []) or []
                contained_elements.update(elements)
        
        all_elements = set(self.walls + self.doors + self.windows + self.slabs)
        missing_containment = all_elements - contained_elements
        if missing_containment:
            logger.warning(f"Found {len(missing_containment)} elements not contained in storey")
        
        # Verify material relations
        elements_with_materials = set()
        for rel in all_material_rels:
            related = getattr(rel, "RelatedObjects", []) or []
            elements_with_materials.update(related)
        
        all_walls_and_slabs = set(self.walls + self.slabs)
        missing_materials = all_walls_and_slabs - elements_with_materials
        if missing_materials:
            logger.warning(f"Found {len(missing_materials)} walls/slabs without IfcRelAssociatesMaterial")
        
        # Verify property set relations (non-critical, just log)
        psets_with_relations = set()
        for rel in all_property_rels:
            related = getattr(rel, "RelatedObjects", []) or []
            psets_with_relations.update(related)
        
        logger.debug(f"Relations verification: {len(all_void_rels)} void relations, {len(all_fill_rels)} fill relations, "
                    f"{len(all_containment_rels)} containment relations, {len(all_material_rels)} material relations, "
                    f"{len(all_property_rels)} property relations")

    # --- Model-based API methods (simpler input interface) ---
    
    def add_wall_from_model(self, wall: "WallModel") -> Any:
        """Create wall from simple Wall model.
        
        Converts a simple Wall model (start/end points) to a WallProfile
        and creates the wall using the existing create_wall() method.
        All features (Gap Closure, Material Layers, etc.) are preserved.
        
        Args:
            wall: Wall model with start_point, end_point, height, thickness, is_external
            
        Returns:
            IfcWallStandardCase entity
        """
        if WallModel is None:
            raise ImportError("ifc_generator_v2.models module is not available")
        
        # Convert wall axis (start/end points) to 4-point quadrilateral
        # The wall has a thickness, so we create a rectangle perpendicular to the axis
        start = wall.start_point.to_tuple()
        end = wall.end_point.to_tuple()
        
        # Calculate direction vector and perpendicular vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            raise ValueError("Wall start and end points cannot be identical")
        
        # Normalized direction vector
        dir_x = dx / length
        dir_y = dy / length
        
        # Perpendicular vector (rotate 90 degrees counter-clockwise)
        perp_x = -dir_y
        perp_y = dir_x
        
        # Half thickness offset
        half_thickness = wall.thickness / 2.0
        
        # Create 4 corner points of the wall rectangle
        # Points are ordered counter-clockwise
        p1 = (
            start[0] + perp_x * half_thickness,
            start[1] + perp_y * half_thickness,
        )
        p2 = (
            end[0] + perp_x * half_thickness,
            end[1] + perp_y * half_thickness,
        )
        p3 = (
            end[0] - perp_x * half_thickness,
            end[1] - perp_y * half_thickness,
        )
        p4 = (
            start[0] - perp_x * half_thickness,
            start[1] - perp_y * half_thickness,
        )
        
        # Create WallProfile from the 4 points
        wall_profile = WallProfile(
            points=[p1, p2, p3, p4],
            height=wall.height,
            uniform_thickness_m=wall.thickness,
            name=wall.name,
            is_external=wall.is_external,
            preserve_exact_geometry=False,  # Allow simplification for simple walls
        )
        
        # Use existing create_wall() method (preserves all features)
        return self.create_wall(wall_profile)
    
    def add_window_from_model(self, window: "WindowModel", ifc_wall: Any) -> Any:
        """Create window from simple Window model.
        
        Converts a simple Window model to a WindowProfile and creates
        the window using the existing create_window() method.
        
        Args:
            window: Window model with wall_id, position_along_wall, width, height
            ifc_wall: IfcWallStandardCase entity to attach window to
            
        Returns:
            Tuple of (IfcWindow entity, IfcOpeningElement entity)
        """
        if WindowModel is None:
            raise ImportError("ifc_generator_v2.models module is not available")
        
        # Get wall geometry to calculate window position
        # We need to find the wall's start point and direction
        try:
            import ifcopenshell.util.placement as placement_util
            import numpy as np
            
            wall_placement = getattr(ifc_wall, "ObjectPlacement", None)
            if wall_placement:
                wall_matrix = placement_util.get_local_placement(wall_placement)
                # Wall's local origin is at (0, 0, 0) in wall-local space
                # Position along wall axis: (position_along_wall, 0, sill_height)
                local_position = (
                    window.position_along_wall,
                    0.0,
                    STANDARDS["WINDOW_SILL_HEIGHT"],  # Window sill height
                )
                # Transform to global coordinates
                local_point = np.array([*local_position, 1.0])
                global_point = wall_matrix @ local_point
                global_position = (float(global_point[0]), float(global_point[1]), float(global_point[2]))
            else:
                # Fallback: use position_along_wall as X coordinate
                global_position = (
                    window.position_along_wall,
                    0.0,
                    STANDARDS["WINDOW_SILL_HEIGHT"],
                )
        except Exception as e:
            logger.debug(f"Could not calculate window position from wall placement: {e}, using fallback")
            # Fallback: use position_along_wall as X coordinate
            global_position = (
                window.position_along_wall,
                0.0,
                STANDARDS["WINDOW_SILL_HEIGHT"],
            )
        
        # Get wall thickness for window thickness
        wall_thickness = window.width * 0.1  # Default: 10% of width as thickness
        try:
            from ifcopenshell.util import element as ifc_element_utils
            wall_quantities = ifc_element_utils.get_psets(ifc_wall, qtos_only=True)
            for qto_name, qto_props in wall_quantities.items():
                if "Width" in qto_props:
                    wall_thickness = float(qto_props["Width"])
                    break
        except Exception:
            pass
        
        # Create WindowProfile
        window_profile = WindowProfile(
            width=window.width,
            height=window.height,
            thickness=wall_thickness,
            location=global_position,
            name=f"Window-{window.id}",
            is_external=True,  # Windows are typically external
        )
        
        # Use existing create_window() method
        return self.create_window(ifc_wall, window_profile)
    
    def add_door_from_model(self, door: "DoorModel", ifc_wall: Any) -> Any:
        """Create door from simple Door model.
        
        Converts a simple Door model to a DoorProfile and creates
        the door using the existing create_door() method.
        
        Args:
            door: Door model with wall_id, position_along_wall, width, height
            ifc_wall: IfcWallStandardCase entity to attach door to
            
        Returns:
            Tuple of (IfcDoor entity, IfcOpeningElement entity)
        """
        if DoorModel is None:
            raise ImportError("ifc_generator_v2.models module is not available")
        
        # Get wall geometry to calculate door position
        try:
            import ifcopenshell.util.placement as placement_util
            import numpy as np
            
            wall_placement = getattr(ifc_wall, "ObjectPlacement", None)
            if wall_placement:
                wall_matrix = placement_util.get_local_placement(wall_placement)
                # Door position: (position_along_wall, 0, 0) at OKFF
                local_position = (door.position_along_wall, 0.0, 0.0)
                local_point = np.array([*local_position, 1.0])
                global_point = wall_matrix @ local_point
                global_position = (float(global_point[0]), float(global_point[1]), float(global_point[2]))
            else:
                global_position = (door.position_along_wall, 0.0, 0.0)
        except Exception as e:
            logger.debug(f"Could not calculate door position from wall placement: {e}, using fallback")
            global_position = (door.position_along_wall, 0.0, 0.0)
        
        # Get wall thickness for door thickness
        wall_thickness = door.width * 0.1  # Default: 10% of width as thickness
        try:
            from ifcopenshell.util import element as ifc_element_utils
            wall_quantities = ifc_element_utils.get_psets(ifc_wall, qtos_only=True)
            for qto_name, qto_props in wall_quantities.items():
                if "Width" in qto_props:
                    wall_thickness = float(qto_props["Width"])
                    break
        except Exception:
            pass
        
        # Create DoorProfile
        door_profile = DoorProfile(
            width=door.width,
            height=door.height,
            thickness=wall_thickness,
            location=global_position,
            name=f"Door-{door.id}",
            is_external=False,  # Doors are typically internal
        )
        
        # Use existing create_door() method
        return self.create_door(ifc_wall, door_profile)
    
    def add_slab_from_model(self, slab: "SlabModel") -> Any:
        """Create slab from simple Slab model.
        
        Converts a simple Slab model to a SlabProfile and creates
        the slab using the existing create_slab() method.
        
        Args:
            slab: Slab model with footprint_points, thickness, slab_type, elevation
            
        Returns:
            IfcSlab entity
        """
        if SlabModel is None:
            raise ImportError("ifc_generator_v2.models module is not available")
        
        # Convert Point2D list to tuple list
        points = [p.to_tuple() for p in slab.footprint_points]
        
        # Create SlabProfile
        slab_profile = SlabProfile(
            points=points,
            thickness=slab.thickness,
            name=slab.name,
            is_external=False,  # Slabs are typically internal
        )
        
        # Use existing create_slab() method
        ifc_slab = self.create_slab(slab_profile)
        
        # Set PredefinedType based on slab_type
        try:
            if slab.slab_type.upper() in ["FLOOR", "CEILING", "BASESLAB", "ROOF"]:
                ifc_slab.PredefinedType = slab.slab_type.upper()
            else:
                ifc_slab.PredefinedType = "FLOOR"  # Default
        except Exception as e:
            logger.debug(f"Could not set PredefinedType for slab {slab.name}: {e}")
        
        # Adjust elevation if needed - recreate ObjectPlacement with correct elevation
        if slab.elevation != 0.0:
            try:
                # Recreate placement with elevation
                ifc_slab.ObjectPlacement = self.model.create_entity(
                    "IfcLocalPlacement",
                    PlacementRelTo=self.storey.ObjectPlacement,
                    RelativePlacement=self._axis2placement((0.0, 0.0, slab.elevation)),
                )
            except Exception as e:
                logger.warning(f"Could not set elevation for slab {slab.name}: {e}")
        
        return ifc_slab
    
    def add_space_from_model(self, space: "SpaceModel") -> Any:
        """Create space from simple Space model.
        
        Converts a simple Space model to a SpaceProfile and creates
        the space using the existing create_space() method.
        
        Args:
            space: Space model with footprint_points, height, name
            
        Returns:
            IfcSpace entity
        """
        if SpaceModel is None:
            raise ImportError("ifc_generator_v2.models module is not available")
        
        # Convert Point2D list to tuple list
        points = [p.to_tuple() for p in space.footprint_points]
        
        # Create SpaceProfile
        space_profile = SpaceProfile(
            points=points,
            height=space.height,
            name=space.name,
            is_external=False,  # Spaces are typically internal
        )
        
        # Use existing create_space() method
        return self.create_space(space_profile)

    def finalize_relations(self) -> None:
        """Finalize all IFC relations and verify completeness."""
        if self.spaces_created:
            existing = next(
                (
                    rel
                    for rel in self.model.by_type("IfcRelAggregates")
                    if getattr(rel, "RelatingObject", None) == self.storey
                ),
                None
            )
            if existing:
                related = list(getattr(existing, "RelatedObjects", []) or [])
                for space in self.spaces_created:
                    if space not in related:
                        related.append(space)
                existing.RelatedObjects = tuple(related)
            else:
                self.model.create_entity(
                    "IfcRelAggregates",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatingObject=self.storey,
                    RelatedObjects=tuple(self.spaces_created),
                )

        # Include all elements: walls, doors, windows, and slabs
        elements = tuple({*self.walls, *self.doors, *self.windows, *self.slabs})
        if elements:
            existing_containment = next(
                (
                    rel
                    for rel in self.model.by_type("IfcRelContainedInSpatialStructure")
                    if getattr(rel, "RelatingStructure", None) == self.storey
                ),
                None
            )
            if existing_containment:
                related = list(getattr(existing_containment, "RelatedElements", []) or [])
                for element in elements:
                    if element not in related:
                        related.append(element)
                existing_containment.RelatedElements = tuple(related)
            else:
                self.model.create_entity(
                    "IfcRelContainedInSpatialStructure",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatedElements=elements,
                    RelatingStructure=self.storey,
                )
        
        # Also include openings in containment (optional but recommended)
        if self.openings:
            existing_containment = next(
                (
                    rel
                    for rel in self.model.by_type("IfcRelContainedInSpatialStructure")
                    if getattr(rel, "RelatingStructure", None) == self.storey
                ),
                None
            )
            if existing_containment:
                related = list(getattr(existing_containment, "RelatedElements", []) or [])
                for opening in self.openings:
                    if opening not in related:
                        related.append(opening)
                existing_containment.RelatedElements = tuple(related)
            else:
                # If no containment relation exists yet, create one with openings
                self.model.create_entity(
                    "IfcRelContainedInSpatialStructure",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatedElements=tuple(self.openings),
                    RelatingStructure=self.storey,
                )
        
        # Create space boundaries for thermal calculations (HottCAD requirement)
        self._finalize_space_boundaries()
        
        # Verify all relations are complete
        self._verify_relations()


def _extract_primary_polygon(geom: Any) -> Polygon | None:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon) and geom.geoms:
        return max(geom.geoms, key=lambda g: g.area)
    return None


def _polygon_to_meters(poly: Polygon) -> Polygon:
    exterior = [(x / 1000.0, y / 1000.0) for x, y in poly.exterior.coords]
    interiors = [[(x / 1000.0, y / 1000.0) for x, y in ring.coords] for ring in poly.interiors]
    return Polygon(exterior, interiors)


def _process_walls(
    builder: IFCV2Builder,
    normalized: Sequence[NormalizedDet],
    wall_axes: Sequence[Any] | None,
    storey_height_mm: float,
    px_per_mm: float | None,
) -> List[Tuple[Any, Polygon, bool, float]]:
    """Process walls and return wall records with geometry."""
    wall_axes = wall_axes or []

    # NormalizedDet type values can vary (e.g. "WALL_EXT", "wall", etc.)
    wall_labels = {"WALL", "WALL_EXT", "WALL_INT", "EXTERNAL_WALL", "INTERNAL_WALL"}

    # Create mapping from source index to wall axis info
    axis_by_source: Dict[int, List[Any]] = {}
    for axis_info in wall_axes:
        source_idx = getattr(axis_info, "source_index", None)
        if source_idx is None:
            continue
        axis_by_source.setdefault(source_idx, []).append(axis_info)

    wall_records: List[Tuple[Any, Polygon, bool, float]] = []
    wall_index = 0

    def _is_wall_detection(det: NormalizedDet) -> bool:
        det_type = (getattr(det, "type", "") or "").upper()
        return det_type in wall_labels

    wall_detections = [det for det in normalized if _is_wall_detection(det)]

    for source_index, det in enumerate(wall_detections):
        try:
            if det.geom is None:
                logger.debug("Skipping wall detection without geometry (id=%s)", getattr(det, "id", None))
                continue

            poly = _extract_primary_polygon(det.geom)
            if poly is None or poly.is_empty:
                logger.debug("Skipping wall with empty/invalid polygon (id=%s)", getattr(det, "id", None))
                continue

            coords = list(poly.exterior.coords)[:-1]
            if len(coords) < 3:
                logger.debug("Skipping wall with <3 coordinates (id=%s)", getattr(det, "id", None))
                continue

            axes_for_wall = axis_by_source.get(source_index, [])
            is_ext_bool = bool(det.is_external) if det.is_external is not None else False

            if axes_for_wall:
                thicknesses = [float(ax.width_mm) for ax in axes_for_wall if getattr(ax, "width_mm", 0) > 0]
                if thicknesses:
                    raw_thickness_mm = sorted(thicknesses)[len(thicknesses) // 2]
                else:
                    raw_thickness_mm = STANDARDS["WALL_EXTERNAL_THICKNESS"] * 1000.0 if is_ext_bool else STANDARDS["WALL_INTERNAL_THICKNESS"] * 1000.0
            else:
                raw_thickness_mm = STANDARDS["WALL_EXTERNAL_THICKNESS"] * 1000.0 if is_ext_bool else STANDARDS["WALL_INTERNAL_THICKNESS"] * 1000.0

            uniform_thickness_mm = snap_thickness_mm(
                raw_thickness_mm,
                is_external=is_ext_bool,
            )
            uniform_thickness_m = uniform_thickness_mm / 1000.0

            poly_m = _polygon_to_meters(poly)
            coords_m = list(poly_m.exterior.coords)[:-1]
            if len(coords_m) < 3:
                # Fallback: use minimum rotated rectangle when polygon is degenerate
                try:
                    rect = poly.minimum_rotated_rectangle
                    rect_m = _polygon_to_meters(rect)
                    rect_coords_m = list(rect_m.exterior.coords)[:-1]
                    if len(rect_coords_m) >= 4:
                        coords_m = rect_coords_m
                        poly_m = rect_m
                        logger.debug("Using rotated rectangle fallback for wall (id=%s)", getattr(det, "id", None))
                    else:
                        logger.debug("Skipping wall (id=%s) after meter conversion; insufficient coords (even after rectangle)", getattr(det, "id", None))
                        continue
                except Exception as rect_exc:
                    logger.debug("Skipping wall (id=%s); rectangle fallback failed: %s", getattr(det, "id", None), rect_exc)
                    continue

            wall_index += 1
            profile = WallProfile(
                points=coords_m,
                height=max(storey_height_mm, 0.0) / 1000.0,
                uniform_thickness_m=uniform_thickness_m,
                name=f"Wand-{wall_index:03d}",
                is_external=is_ext_bool,
                preserve_exact_geometry=True,
            )

            try:
                wall_entity = builder.create_wall(profile)
            except Exception as wall_exc:
                logger.error("Failed to create wall entity for detection %s: %s", getattr(det, "id", None), wall_exc, exc_info=True)
                continue

            wall_records.append((wall_entity, poly_m, is_ext_bool, uniform_thickness_mm))
            builder.wall_polygons.append(poly_m)

        except Exception as det_exc:
            logger.error("Unexpected error while processing wall detection %s: %s", getattr(det, "id", None), det_exc, exc_info=True)
            continue

    if not wall_records:
        logger.warning("No walls were created from %d wall detections", len(wall_detections))

    return wall_records


def _process_doors(
    builder: IFCV2Builder,
    normalized: Sequence[NormalizedDet],
    wall_records: List[Tuple[Any, Polygon, bool, float]],
    door_height_mm: float,
    wall_index_tree: STRtree | None,
    wall_geoms: List[Polygon],
) -> None:
    """Process doors and attach them to nearest walls."""
    if not wall_records:
        logger.warning("Skipping door processing because no wall records are available")
        return

    door_index = 0
    door_height_m = door_height_mm / 1000.0
    for det in normalized:
        if det.type != "DOOR" or det.geom is None:
            continue
        poly = _extract_primary_polygon(det.geom)
        if poly is None or poly.is_empty:
            continue
        poly_m = _polygon_to_meters(poly)
        wall_info = find_nearest_wall(poly_m, wall_records, wall_index_tree, wall_geoms)
        if wall_info is None:
            logger.debug("No wall found for door detection %s", getattr(det, "id", None))
            continue
        wall_entity, wall_geom, wall_external, _wall_thickness = wall_info
        width, thickness, origin = _rect_dimensions(poly_m)
        if width <= 0 or thickness <= 0:
            continue
        try:
            wcoords = list(wall_geom.exterior.coords)[:-1]
            if len(wcoords) >= 2:
                max_len = 0.0
                seg = (wcoords[0], wcoords[1])
                for i in range(len(wcoords)):
                    p0 = wcoords[i]
                    p1 = wcoords[(i + 1) % len(wcoords)]
                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]
                    L = (dx * dx + dy * dy) ** 0.5
                    if L > max_len:
                        max_len = L
                        seg = (p0, p1)
                p0, p1 = seg
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                vx = origin[0] - p0[0]
                vy = origin[1] - p0[1]
                t = (vx * dx + vy * dy) / (L * L)
                margin = min(max_len * 0.1, max(0.05, width * 0.6))
                t = max(margin / L, min(1.0 - margin / L, t))
                proj_x = p0[0] + t * dx
                proj_y = p0[1] + t * dy
                origin = (proj_x, proj_y, 0.0)
                max_width = max(0.1, max_len - 2.0 * margin)
                if width > max_width:
                    width = max_width
        except Exception:
            pass
        door_index += 1
        builder.create_door(
            wall_entity,
            DoorProfile(
                width=width,
                height=door_height_m,
                thickness=thickness,
                location=(origin[0], origin[1], 0.0),
                name=f"Tuer-{door_index:03d}",
                is_external=wall_external,
            ),
        )


def _process_windows(
    builder: IFCV2Builder,
    normalized: Sequence[NormalizedDet],
    wall_records: List[Tuple[Any, Polygon, bool, float]],
    window_height_mm: float,
    wall_index_tree: STRtree | None,
    wall_geoms: List[Polygon],
) -> None:
    """Process windows and attach them to nearest walls."""
    if not wall_records:
        logger.warning("Skipping window processing because no wall records are available")
        return

    window_index = 0
    window_height_m = window_height_mm / 1000.0
    for det in normalized:
        if det.type != "WINDOW" or det.geom is None:
            continue
        poly = _extract_primary_polygon(det.geom)
        if poly is None or poly.is_empty:
            continue
        poly_m = _polygon_to_meters(poly)
        wall_info = find_nearest_wall(poly_m, wall_records, wall_index_tree, wall_geoms)
        if wall_info is None:
            logger.debug("No wall found for window detection %s", getattr(det, "id", None))
            continue
        wall_entity, wall_geom, wall_external, _wall_thickness = wall_info
        width, thickness, origin = _rect_dimensions(poly_m)
        if width <= 0 or thickness <= 0:
            continue
        try:
            wcoords = list(wall_geom.exterior.coords)[:-1]
            if len(wcoords) >= 2:
                max_len = 0.0
                seg = (wcoords[0], wcoords[1])
                for i in range(len(wcoords)):
                    p0 = wcoords[i]
                    p1 = wcoords[(i + 1) % len(wcoords)]
                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]
                    L = (dx * dx + dy * dy) ** 0.5
                    if L > max_len:
                        max_len = L
                        seg = (p0, p1)
                p0, p1 = seg
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                vx = origin[0] - p0[0]
                vy = origin[1] - p0[1]
                t = (vx * dx + vy * dy) / (L * L)
                margin = min(max_len * 0.1, max(0.05, width * 0.6))
                t = max(margin / L, min(1.0 - margin / L, t))
                proj_x = p0[0] + t * dx
                proj_y = p0[1] + t * dy
                origin = (proj_x, proj_y, 0.0)
                max_width = max(0.1, max_len - 2.0 * margin)
                if width > max_width:
                    width = max_width
        except Exception:
            pass
        window_index += 1
        builder.create_window(
            wall_entity,
            WindowProfile(
                width=width,
                height=window_height_m,
                thickness=thickness,
                location=(origin[0], origin[1], 0.0),
                name=f"Fenster-{window_index:03d}",
                is_external=wall_external,
            ),
        )


def _process_spaces(
    builder: IFCV2Builder,
    spaces: Sequence[SpacePoly],
    storey_height_mm: float,
    floor_thickness_mm: float = STANDARDS["SLAB_THICKNESS"] * 1000.0,  # Convert m to mm
) -> None:
    """Process spaces and create floor coverings for each space."""
    space_index = 0
    for space in spaces:
        if space.polygon is None or space.polygon.is_empty:
            continue
        poly = _polygon_to_meters(space.polygon)
        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            continue
        space_index += 1
        space_entity = builder.create_space(
            SpaceProfile(
                points=coords,
                height=storey_height_mm / 1000.0,
                name=f"Raum-{space_index:03d}",
                long_name=None,
                is_external=False,
            )
        )
        
        # Store space polygon for spatial queries (in meters, matching wall polygons)
        builder.space_polygons[space_entity] = poly
        
        # Create floor covering for HottCAD compatibility
        try:
            builder.create_floor_covering(
                space_entity,
                SlabProfile(
                    points=coords,
                    thickness=floor_thickness_mm / 1000.0,
                    name=f"Floor_{space_index:03d}",
                    is_external=False,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to create floor covering for space {space_index}: {e}")


def _process_floor(
    builder: IFCV2Builder,
    wall_records: List[Tuple[Any, Polygon, bool, float]],
    floor_thickness_mm: float,
) -> None:
    """Create outer boundary floor from exterior walls."""
    # Ensure boolean values are properly converted (avoid numpy array boolean ambiguity)
    exterior_wall_polys = [poly_m for _wall, poly_m, is_ext, _thickness in wall_records if bool(is_ext)]
    if not exterior_wall_polys:
        exterior_wall_polys = [poly_m for _wall, poly_m, _is_ext, _thickness in wall_records]
        if not exterior_wall_polys:
            return
    
    try:
        # Create outer boundary: union of external walls, buffer by wall thickness, then envelope
        if len(exterior_wall_polys) > 1:
            union_poly = unary_union(exterior_wall_polys)
        else:
            union_poly = exterior_wall_polys[0]
        
        # Get average wall thickness for buffer
        # Prefer external walls when available; otherwise, use all walls
        ext_records = [r for r in wall_records if bool(r[2])]
        used_records = ext_records if ext_records else list(wall_records)
        avg_thickness_mm = sum(r[3] for r in used_records) / max(len(used_records), 1)
        buffer_distance_m = (avg_thickness_mm * IFCConstants.BUFFER_FACTOR) / 1000.0
        
        # Buffer and get envelope
        buffered = union_poly.buffer(buffer_distance_m)
        if hasattr(buffered, 'envelope'):
            floor_poly = buffered.envelope
        else:
            floor_poly = buffered
        
        # Convert to points (enforce orientation, simplify micro edges ~5mm)
        if not floor_poly.is_empty and floor_poly.is_valid:
            try:
                floor_poly = orient(floor_poly, sign=-1.0)
                floor_poly = floor_poly.simplify(0.005, preserve_topology=True)
            except Exception:
                pass
            floor_coords = list(floor_poly.exterior.coords)[:-1]
            if len(floor_coords) >= 3:
                builder.create_slab(
                    SlabProfile(
                        points=floor_coords,
                        thickness=floor_thickness_mm / 1000.0,
                        name="Außenboden",
                        is_external=True,
                    )
                )
    except Exception as floor_exc:
        # Log but don't fail if floor creation fails
        logger.warning("Failed to create outer boundary floor: %s", floor_exc)


def _write_ifc_v2_impl(
    normalized: Sequence[NormalizedDet],
    spaces: Sequence[SpacePoly],
    out_path: Path,
    *,
    storey_height_mm: float,
    door_height_mm: float,
    window_height_mm: float,
    floor_thickness_mm: float = STANDARDS["SLAB_THICKNESS"] * 1000.0,  # Convert m to mm
    px_per_mm: float | None = None,
    image_height_px: float | None = None,
    wall_axes: Sequence[Any] | None = None,
    geometry_fidelity: str | None = None,
    gap_closure_mode: str | None = None,
    config: IFCExportConfig | None = None,
) -> Path:
    """Internal implementation of IFC export (runs in subprocess to avoid memory leaks)."""
    # Create config from parameters if not provided
    if config is None:
        config = IFCExportConfig(
            schema="IFC4",
            geometry_fidelity=geometry_fidelity,
            gap_closure_mode=gap_closure_mode,
        )
    else:
        # Override config with explicit parameters if provided (for backward compatibility)
        if geometry_fidelity:
            config.geometry_fidelity = geometry_fidelity
        if gap_closure_mode:
            config.gap_closure_mode = gap_closure_mode
    
    # Derive image height in meters for Y-axis flipping (if possible)
    image_height_m: float | None = None
    if image_height_px is not None:
        try:
            px_value = float(image_height_px)
            if px_per_mm:
                px_per_mm_val = float(px_per_mm)
                if px_per_mm_val > 0.0:
                    image_height_m = px_value / px_per_mm_val / 1000.0
        except (TypeError, ValueError):
            logger.debug("Invalid image_height_px value: %s", image_height_px)

    if image_height_m is None and config is not None:
        cfg_height = getattr(config, "image_height", None)
        if cfg_height is not None:
            try:
                image_height_m = float(cfg_height)
            except (TypeError, ValueError):
                logger.debug("Invalid config.image_height value: %s", cfg_height)

    try:
        with IFCV2Builder(config=config) as builder:
            builder.default_image_height = image_height_m
            builder.image_height = image_height_m
            # Use provided wall axes or calculate new ones (with error handling)
            if wall_axes is None:
                try:
                    wall_axes = estimate_wall_axes_and_thickness(
                        normalized,
                        raster_px_per_mm=px_per_mm or 2.0,
                    )
                except Exception as e:
                    logger.warning(f"Error estimating wall axes (using empty list): {e}")
                    wall_axes = []
            
            # Process walls (with error handling)
            try:
                wall_records = _process_walls(
                    builder, normalized, wall_axes, storey_height_mm, px_per_mm
                )
            except Exception as e:
                logger.error(f"Error processing walls: {e}", exc_info=True)
                wall_records = []  # Continue with empty walls

            # Build spatial index for efficient wall lookup
            # NOTE: STRtree is created in function scope and will be garbage collected after use.
            # This prevents memory leaks in long-running services. The tree is passed to
            # _process_doors and _process_windows but not stored as instance/global variable.
            wall_geoms = [geom for _, geom, _, _ in wall_records]
            wall_index_tree = STRtree(wall_geoms) if wall_geoms else None

            try:
                # Process doors and windows (tree is used here and then goes out of scope)
                try:
                    _process_doors(builder, normalized, wall_records, door_height_mm, wall_index_tree, wall_geoms)
                except Exception as e:
                    logger.warning(f"Error processing doors (non-critical): {e}")
                
                try:
                    _process_windows(builder, normalized, wall_records, window_height_mm, wall_index_tree, wall_geoms)
                except Exception as e:
                    logger.warning(f"Error processing windows (non-critical): {e}")
            finally:
                # Explicit cleanup of STRtree to ensure memory is freed promptly
                # Note: STRtree doesn't have explicit close(), but clearing reference helps GC
                if wall_index_tree is not None:
                    del wall_index_tree
                # Clear wall_geoms reference as well
                wall_geoms.clear()

            # Process spaces (with error handling)
            try:
                _process_spaces(builder, spaces, storey_height_mm, floor_thickness_mm)
            except Exception as e:
                logger.warning(f"Error processing spaces (non-critical): {e}")

            # Process floor (with error handling)
            try:
                _process_floor(builder, wall_records, floor_thickness_mm)
            except Exception as e:
                logger.warning(f"Error processing floor (non-critical): {e}")

            # Add provenance PropertySet before finalizing (with error handling)
            try:
                builder.add_provenance(
                    px_per_mm=px_per_mm,
                    geometry_fidelity=geometry_fidelity,
                    gap_closure_mode=gap_closure_mode,
                )
            except Exception as e:
                logger.warning(f"Error adding provenance (non-critical): {e}")

            # Validate before writing file
            builder._validate_before_export()

            try:
                builder.finalize_relations()
            except Exception as e:
                logger.warning(f"Error finalizing relations (non-critical): {e}")

            # Write IFC file (critical - must succeed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                builder.model.write(out_path.as_posix())
            except Exception as e:
                logger.error(f"Error writing IFC file: {e}", exc_info=True)
                raise IFCExportError(f"Failed to write IFC file: {e}") from e

            # Post-export validation (non-blocking)
            builder._validate_after_export()

            # Validate IFC schema after writing (non-blocking)
            try:
                validate_ifc_schema(out_path)
            except Exception as e:
                logger.warning(f"IFC schema validation error (non-critical): {e}")
    except Exception as e:
        logger.error(f"Critical error in IFC export: {e}", exc_info=True)
        raise IFCExportError(f"IFC export failed: {e}") from e
    
    return out_path


def validate_ifc_schema(file_path: Path) -> None:
    """Validate IFC file schema conformance.
    
    This function is designed to be non-blocking - it logs warnings
    but never raises exceptions to prevent server crashes.
    
    Args:
        file_path: Path to IFC file to validate
    """
    try:
        import ifcopenshell.validate
    except ImportError:
        logger.warning("ifcopenshell.validate not available, skipping schema validation")
        return
    
    try:
        val_logger = ifcopenshell.validate.json_logger()
        ifcopenshell.validate.validate(str(file_path), val_logger)
        
        if val_logger.statements:
            errors = [s for s in val_logger.statements if s.get("type") == "error"]
            if errors:
                error_messages = [s.get("message", str(s)) for s in errors[:10]]  # Limit to first 10
                logger.warning(
                    f"IFC schema validation found {len(errors)} error(s) (non-critical): {error_messages}"
                )
            else:
                warnings = [s for s in val_logger.statements if s.get("type") == "warning"]
                if warnings:
                    logger.info(f"IFC schema validation: {len(warnings)} warning(s) found (non-critical)")
                else:
                    logger.info("IFC schema validation: No errors or warnings found")
        else:
            logger.info("IFC schema validation: No validation issues found")
    except Exception as e:
        logger.warning(f"IFC schema validation failed unexpectedly (non-critical): {e}")
        # Don't fail the export if validation itself fails, but log it
        # This allows the export to succeed even if validation has issues


def write_ifc_v2(
    normalized: Sequence[NormalizedDet],
    spaces: Sequence[SpacePoly],
    out_path: Path,
    *,
    storey_height_mm: float,
    door_height_mm: float,
    window_height_mm: float,
    floor_thickness_mm: float = STANDARDS["SLAB_THICKNESS"] * 1000.0,  # Convert m to mm
    px_per_mm: float | None = None,
    wall_axes: Sequence[Any] | None = None,
    geometry_fidelity: str | None = None,
    gap_closure_mode: str | None = None,
    use_subprocess: bool = False,
    config: IFCExportConfig | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> Path:
    """Export normalized detections and spaces to IFC file.
    
    Args:
        normalized: Normalized detections (walls, doors, windows)
        spaces: Space polygons
        out_path: Output IFC file path
        storey_height_mm: Storey height in millimeters
        door_height_mm: Door height in millimeters
        window_height_mm: Window height in millimeters
        floor_thickness_mm: Floor thickness in millimeters
        px_per_mm: Pixels per millimeter for calibration
        use_subprocess: If True, run export in separate process to avoid memory leaks.
                       Recommended for service mode with 1000+ exports.
        config: Optional IFC export configuration
        executor: Optional ProcessPoolExecutor instance for dependency injection (testing).
                 If None and use_subprocess=True, uses shared singleton executor.
    
    Returns:
        Path to written IFC file
    """
    if use_subprocess:
        # Run in subprocess to ensure C++ memory is freed when process exits
        # This is critical for service mode with many exports
        # Use injected executor if provided (for testing), otherwise use shared singleton
        if executor is None:
            executor = _get_shared_executor()
        future = executor.submit(
            _write_ifc_v2_impl,
            normalized,
            spaces,
            out_path,
            storey_height_mm=storey_height_mm,
            door_height_mm=door_height_mm,
            window_height_mm=window_height_mm,
            floor_thickness_mm=floor_thickness_mm,
            px_per_mm=px_per_mm,
            wall_axes=wall_axes,
            geometry_fidelity=geometry_fidelity,
            gap_closure_mode=gap_closure_mode,
            config=config,
        )
        return future.result()
    else:
        return _write_ifc_v2_impl(
            normalized,
            spaces,
            out_path,
            storey_height_mm=storey_height_mm,
            door_height_mm=door_height_mm,
            window_height_mm=window_height_mm,
            floor_thickness_mm=floor_thickness_mm,
            px_per_mm=px_per_mm,
            wall_axes=wall_axes,
            geometry_fidelity=geometry_fidelity,
            gap_closure_mode=gap_closure_mode,
            config=config,
        )
