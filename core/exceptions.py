"""Custom exception hierarchy for Bimify application."""

from __future__ import annotations


class BimifyError(Exception):
    """Base exception for all Bimify-specific errors."""
    
    def __init__(self, message: str, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(BimifyError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(BimifyError):
    """Base class for validation errors."""
    pass


class IFCValidationError(ValidationError):
    """Raised when IFC file validation fails."""
    pass


class SchemaValidationError(IFCValidationError):
    """Raised when IFC schema validation fails."""
    pass


class GeometryError(BimifyError):
    """Raised when geometry operations fail."""
    pass


class GeometryProcessingError(GeometryError):
    """Raised when geometry processing fails."""
    pass


class GeometryExtractionError(GeometryError):
    """Raised when geometry extraction from IFC fails."""
    pass


class MLInferenceError(BimifyError):
    """Raised when ML inference fails."""
    pass


class RoboflowAPIError(MLInferenceError):
    """Raised when Roboflow API calls fail."""
    pass


class ExportError(BimifyError):
    """Base class for export-related errors."""
    pass


class IFCExportError(ExportError):
    """Base exception for all IFC export-related errors."""
    pass


class GeometryValidationError(IFCExportError):
    """Raised when geometry validation fails during IFC export."""
    pass


class InsufficientConfidenceError(IFCExportError):
    """Raised when predictions have insufficient confidence scores."""
    pass


class WallReconstructionError(IFCExportError):
    """Raised when wall reconstruction (skeletonization/axis estimation) fails."""
    pass


class MaterialAssignmentError(IFCExportError):
    """Raised when material assignment to IFC elements fails."""
    pass


class CoordinatePrecisionError(IFCExportError):
    """Raised when coordinate precision loss exceeds threshold."""
    pass


class IFCExportBlockedError(IFCExportError):
    """Raised when IFC export is blocked due to critical validation issues."""
    pass


class PDFExportError(ExportError):
    """Raised when PDF export fails."""
    pass


class RepairError(BimifyError):
    """Raised when IFC repair operations fail."""
    pass


class StorageError(BimifyError):
    """Raised when storage operations fail."""
    pass


class S3Error(StorageError):
    """Raised when S3 operations fail."""
    pass


class JobError(BimifyError):
    """Raised when job processing fails."""
    pass


class JobNotFoundError(JobError):
    """Raised when a job is not found."""
    pass


class FileNotFoundError(BimifyError):
    """Raised when a required file is not found."""
    pass


class TopViewError(BimifyError):
    """Raised when TopView generation fails."""
    pass

