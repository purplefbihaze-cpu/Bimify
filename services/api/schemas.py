from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator, field_validator

from core.ml.roboflow_client import RoboflowPrediction


class ZoneOption(BaseModel):
    name: str
    points: list[list[float]]


class LineOption(BaseModel):
    name: str
    start: list[float]
    end: list[float]


class AnalyzeOptions(BaseModel):
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    overlap: float = Field(0.3, ge=0.0, le=1.0)
    per_class_thresholds: dict[str, float] | None = None
    zones: list[ZoneOption] | None = None
    lines: list[LineOption] | None = None
    model_kind: Literal["geometry", "rooms", "verification"] | None = None


class PredictionOut(BaseModel):
    id: str | None = None
    label: str = Field("", alias="class")
    confidence: float = 0.0
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    points: list[list[float]] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True


class ZoneCount(BaseModel):
    name: str
    total: int
    per_class: dict[str, int]


class LineCount(BaseModel):
    name: str
    counts: dict[str, int]
    per_class: dict[str, dict[str, int]]


class AnalyzeResponse(BaseModel):
    model_id: str
    confidence: float
    overlap: float
    total: int
    per_class: dict[str, int]
    predictions: list[PredictionOut]
    zones: list[ZoneCount] | None = None
    lines: list[LineCount] | None = None
    annotated_image: str | None = None
    image: dict[str, Any] | None = None
    raw: dict[str, Any]


class SettingsPayload(BaseModel):
    roboflow_api_key: str | None = None
    has_roboflow_api_key: bool | None = None
    roboflow_project: str | None = None
    roboflow_version: int | None = None
    matrix_enabled: bool | None = True
    matrix_speed: float | None = 1.0
    matrix_density: float | None = 1.0
    matrix_opacity: float | None = 0.35
    matrix_color: str | None = "#00ff41"


class CalibrationPayload(BaseModel):
    px_per_mm: float = Field(..., gt=0.0)
    pixel_distance: float = Field(..., gt=0.0)
    real_distance_mm: float = Field(..., gt=0.0)
    point_a: list[float]
    point_b: list[float]
    unit: str = Field("mm")
    flip_y: bool | None = None


class ExportIFCRequest(BaseModel):
    predictions: list[dict[str, Any]]
    image: dict[str, Any] | None = None
    storey_height_mm: float = Field(..., gt=0.0)
    door_height_mm: float = Field(..., gt=0.0)
    window_height_mm: float | None = Field(default=1000.0, gt=0.0)
    window_head_elevation_mm: float = Field(2000.0, gt=0.0)
    px_per_mm: float | None = Field(default=None, gt=0.0)
    project_name: str | None = None
    storey_name: str | None = None
    calibration: CalibrationPayload | None = None
    flip_y: bool | None = None
    image_height_px: float | None = Field(default=None, gt=0.0)


class ExportIFCResponse(BaseModel):
    ifc_url: str
    improved_ifc_url: str | None = None
    improved_ifc_stats_url: str | None = None
    improved_wexbim_url: str | None = None
    file_name: str
    topview_url: str | None = None
    validation_report_url: str | None = None
    storey_height_mm: float
    door_height_mm: float
    window_height_mm: float | None
    window_head_elevation_mm: float
    px_per_mm: float | None
    warnings: list[str] | None = None


class ExportIFCJobResponse(BaseModel):
    job_id: str = Field(..., alias="job_id")

    class Config:
        populate_by_name = True


# ============================================================================
# IFC EXPORT V2 - NEW IMPLEMENTATION
# ============================================================================
# This is a completely new IFC export implementation (V2).
# It uses a different logic path than the original IFC export.
# All V2-related code is clearly marked with "V2" suffix.
# ============================================================================

class GeometryFidelityLevel(str, Enum):
    """Geometry fidelity levels for IFC export pipeline."""
    LOSSLESS = "lossless"  # Deaktiviert Snap, 90°-Enforcement, behält exakte Geometrie
    HIGH = "high"         # Snap 2mm, 90°-Enforcement 5°, kein Gap-Closure
    MEDIUM = "medium"     # Snap 5mm, 90°-Enforcement 10°, Gap-Closure bei >50mm
    LOW = "low"          # Snap 10mm, 90°-Enforcement 15°, Gap-Closure bei >20mm


class ExportIFCV2Request(BaseModel):
    """V2 IFC Export Request - New implementation with different logic."""
    predictions: list[RoboflowPrediction] = Field(..., min_length=1, description="List of validated Roboflow predictions")
    image: dict[str, Any] | None = None
    storey_height_mm: float = Field(..., gt=0.0)
    door_height_mm: float = Field(..., gt=0.0)
    window_height_mm: float | None = Field(default=None, gt=0.0)
    window_head_elevation_mm: float | None = Field(default=None, gt=0.0)
    floor_thickness_mm: float = Field(default=200.0, gt=0.0)
    px_per_mm: float | None = Field(default=None, gt=0.0)
    project_name: str | None = None
    storey_name: str | None = None
    calibration: CalibrationPayload | None = None
    flip_y: bool | None = None
    image_height_px: float | None = Field(default=None, gt=0.0)
    geometry_fidelity: GeometryFidelityLevel | None = Field(
        default=None,
        description="Geometry fidelity level. None = use legacy config flags (backward compatible)"
    )
    preserve_exact_geometry: bool = Field(
        default=True,
        description="Preserve exact geometry (original polygons) instead of simplifying to rectangles"
    )
    min_wall_thickness_mm: float = Field(
        default=50.0,
        ge=0.0,
        description="Minimum wall thickness in mm to filter thin artifacts/noise"
    )
    confidence_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0.0-1.0) for predictions to be exported"
    )
    
    @field_validator("predictions")
    @classmethod
    def validate_predictions_not_empty(cls, v: list[RoboflowPrediction]) -> list[RoboflowPrediction]:
        """Ensure predictions list is not empty."""
        if not v:
            raise ValueError("Predictions list cannot be empty")
        return v


class ExportIFCV2Response(BaseModel):
    """V2 IFC Export Response."""
    ifc_url: str
    file_name: str
    viewer_url: str | None = None
    topview_url: str | None = None
    validation_report_url: str | None = None
    comparison_report_url: str | None = None
    storey_height_mm: float
    door_height_mm: float
    window_height_mm: float | None
    window_head_elevation_mm: float
    px_per_mm: float | None
    warnings: list[str] | None = None


class ExportIFCV2JobResponse(BaseModel):
    """V2 IFC Export Job Response."""
    job_id: str = Field(..., alias="job_id")

    class Config:
        populate_by_name = True


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = 0
    result: ExportIFCResponse | ExportIFCV2Response | IFCRepairResponse | IFCRepairPreviewResponse | None = None
    error: str | None = None


## xBIM preprocess removed


class IFCTopViewRequest(BaseModel):
    file_name: str | None = Field(default=None, description="Dateiname innerhalb von data/exports")
    ifc_url: str | None = Field(default=None, description="Optionaler /files/... Pfad")
    section_elevation_mm: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _ensure_source(self) -> "IFCTopViewRequest":
        if not (self.file_name or self.ifc_url):
            raise ValueError("file_name oder ifc_url muss gesetzt sein")
        return self


class IFCTopViewResponse(BaseModel):
    topview_url: str
    file_name: str


class IFCRepairRequest(BaseModel):
    file_name: str | None = Field(default=None, description="Dateiname innerhalb von data/exports")
    ifc_url: str | None = Field(default=None, description="Optionaler /files/... Pfad")
    job_id: str | None = Field(default=None, description="Optional: Job-ID, um RF-Context (Bild, px_per_mm, Predictions) zu laden")
    image_url: str | None = Field(default=None, description="Optional: Direktlink zum Originalbild für Edge-Detection")
    level: int = Field(1, ge=1, le=5)

    @model_validator(mode="after")
    def _ensure_source(self) -> "IFCRepairRequest":
        if not (self.file_name or self.ifc_url or self.job_id):
            raise ValueError("file_name, ifc_url oder job_id muss gesetzt sein")
        return self


class IFCRepairResponse(BaseModel):
    file_name: str
    ifc_url: str
    level: int
    topview_url: str | None = None
    warnings: list[str] | None = None


class IFCRepairPreviewRequest(IFCRepairRequest):
    pass


class IFCRepairPreviewResponse(BaseModel):
    preview_id: str
    level: int
    overlay_url: str | None = None
    heatmap_url: str | None = None
    proposed_topview_url: str | None = None
    metrics: dict[str, Any] | None = None
    warnings: list[str] | None = None
    estimated_seconds: int | None = Field(default=None, description="Geschätzte Verarbeitungszeit in Sekunden basierend auf Dateigröße")


class IFCRepairCommitRequest(BaseModel):
    preview_id: str | None = Field(default=None, description="ID der erzeugten Vorschau")
    # Fallback: gleiche Quellen wie Preview, falls kein preview_id übergeben wird
    file_name: str | None = None
    ifc_url: str | None = None
    job_id: str | None = None
    image_url: str | None = None
    level: int = Field(1, ge=1, le=5)

    @model_validator(mode="after")
    def _ensure_source(self) -> "IFCRepairCommitRequest":
        if not (self.preview_id or self.file_name or self.ifc_url or self.job_id):
            raise ValueError("preview_id oder (file_name|ifc_url|job_id) muss gesetzt sein")
        return self

class ExportPDFOptions(BaseModel):
    mode: str = Field("wall-fill", pattern="^(wall-fill|centerline)$")
    smooth_tolerance_mm: float = Field(5.0, ge=0.0)
    snap_tolerance_mm: float = Field(15.0, ge=0.0)
    orthogonal_tolerance_deg: float = Field(10.0, ge=0.0, le=45.0)
    include_background: bool = False


class ExportPDFRequest(BaseModel):
    predictions: list[dict[str, Any]]
    image: dict[str, Any] | None = None
    px_per_mm: float | None = Field(default=None, gt=0.0)
    options: ExportPDFOptions | None = None


class ExportPDFResponse(BaseModel):
    pdf_url: str
    file_name: str
    warnings: list[str] | None = None



class HottCADBaseRequest(BaseModel):
    ifc_url: str | None = Field(default=None, description="Direkter Download-Link zur IFC-Datei")
    job_id: str | None = Field(default=None, description="Job-ID, falls IFC/WexBIM über Job-Metadaten geladen werden soll")
    tolerance_mm: float = Field(default=0.5, ge=0.0, le=25.0, description="Toleranz für Wandkontakte in Millimetern")


class HottCADValidateRequest(HottCADBaseRequest):
    pass


class HottCADSimulateRequest(HottCADBaseRequest):
    pass


class HottCADCheckOut(BaseModel):
    id: str
    title: str
    status: Literal["ok", "warn", "fail"]
    details: list[str] = Field(default_factory=list)
    affected: dict[str, list[str]] = Field(default_factory=dict)


class HottCADMetricsOut(BaseModel):
    wall_count: int = 0
    interior_walls: int = 0
    exterior_walls: int = 0
    walls_with_rectangular_footprint: int = 0
    walls_with_constant_thickness: int = 0
    openings_with_relations: int = 0
    spaces: int = 0
    floors: int = 0
    roofs: int = 0
    connects_relations: int = 0
    space_boundaries: int = 0
    material_layer_usages: int = 0
    avg_wall_thickness_mm: float | None = None


class HottCADFileInfo(BaseModel):
    schema: str
    path: str | None = None
    sizeBytes: int | None = None
    isPlainIFC: bool | None = None


class HottCADValidationResponse(BaseModel):
    schema: str
    file_info: HottCADFileInfo
    checks: list[HottCADCheckOut]
    metrics: HottCADMetricsOut
    score: int
    highlightSets: list[HottCADHighlightOut] = Field(default_factory=list)


class HottCADConnectionOut(BaseModel):
    walls: list[str]
    distanceMm: float
    contactType: Literal["touch", "gap", "overlap"]
    notes: list[str] = Field(default_factory=list)


class HottCADSpaceBoundaryOut(BaseModel):
    walls: list[str]
    spaces: list[str] = Field(default_factory=list)
    note: str | None = None




class HottCADMaterialSuggestionOut(BaseModel):
    wall: str
    thicknessMm: float | None = None
    note: str | None = None


class HottCADCompletenessOut(BaseModel):
    roomsClosed: bool
    gapCount: int
    spaces: int
    walls: int


class HottCADHighlightOut(BaseModel):
    id: str
    label: str
    guids: list[str]
    productIds: list[int] = Field(default_factory=list)


class HottCADSimulationProposedOut(BaseModel):
    connects: list[HottCADConnectionOut] = Field(default_factory=list)
    spaceBoundaries: list[HottCADSpaceBoundaryOut] = Field(default_factory=list)
    materials: list[HottCADMaterialSuggestionOut] = Field(default_factory=list)


class HottCADSimulationResponse(BaseModel):
    proposed: HottCADSimulationProposedOut
    completeness: HottCADCompletenessOut
    highlightSets: list[HottCADHighlightOut] = Field(default_factory=list)


