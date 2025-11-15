"""Tests for custom exception hierarchy."""

import pytest

from core.exceptions import (
    BimifyError,
    ConfigurationError,
    IFCValidationError,
    GeometryError,
    RoboflowAPIError,
    IFCExportError,
    GeometryValidationError,
    InsufficientConfidenceError,
    WallReconstructionError,
    MaterialAssignmentError,
    RepairError,
    JobNotFoundError,
    FileNotFoundError,
    TopViewError,
)


def test_bimify_error_base():
    """Test base BimifyError."""
    error = BimifyError("Test error", {"key": "value"})
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.details == {"key": "value"}


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError("Config missing", {"setting": "api_key"})
    assert isinstance(error, BimifyError)
    assert error.message == "Config missing"


def test_ifc_validation_error():
    """Test IFCValidationError."""
    error = IFCValidationError("IFC validation failed")
    assert isinstance(error, BimifyError)
    assert isinstance(error, ConfigurationError.__bases__[0])  # ValidationError


def test_geometry_error():
    """Test GeometryError."""
    error = GeometryError("Geometry processing failed")
    assert isinstance(error, BimifyError)


def test_roboflow_api_error():
    """Test RoboflowAPIError."""
    error = RoboflowAPIError("API call failed", {"model_id": "test/1"})
    assert isinstance(error, BimifyError)
    assert error.details == {"model_id": "test/1"}


def test_ifc_export_error():
    """Test IFCExportError."""
    error = IFCExportError("Export failed")
    assert isinstance(error, BimifyError)


def test_repair_error():
    """Test RepairError."""
    error = RepairError("Repair failed", {"source_path": "/path/to/file"})
    assert isinstance(error, BimifyError)
    assert error.details == {"source_path": "/path/to/file"}


def test_job_not_found_error():
    """Test JobNotFoundError."""
    error = JobNotFoundError("Job not found", {"job_id": "123"})
    assert isinstance(error, BimifyError)


def test_file_not_found_error():
    """Test FileNotFoundError."""
    error = FileNotFoundError("File not found", {"path": "/path/to/file"})
    assert isinstance(error, BimifyError)


def test_topview_error():
    """Test TopViewError."""
    error = TopViewError("TopView generation failed")
    assert isinstance(error, BimifyError)


def test_ifc_export_error_hierarchy():
    """Test IFC export error hierarchy."""
    # Test base IFC export error
    error = IFCExportError("Base export error")
    assert isinstance(error, BimifyError)
    
    # Test specific IFC export errors
    geometry_error = GeometryValidationError("Geometry validation failed")
    assert isinstance(geometry_error, IFCExportError)
    assert isinstance(geometry_error, BimifyError)
    
    confidence_error = InsufficientConfidenceError("Confidence too low")
    assert isinstance(confidence_error, IFCExportError)
    assert isinstance(confidence_error, BimifyError)
    
    wall_error = WallReconstructionError("Wall reconstruction failed")
    assert isinstance(wall_error, IFCExportError)
    assert isinstance(wall_error, BimifyError)
    
    material_error = MaterialAssignmentError("Material assignment failed")
    assert isinstance(material_error, IFCExportError)
    assert isinstance(material_error, BimifyError)


def test_exception_inheritance():
    """Test exception inheritance hierarchy."""
    # Test that specific errors inherit from base
    assert issubclass(ConfigurationError, BimifyError)
    assert issubclass(IFCValidationError, BimifyError)
    assert issubclass(GeometryError, BimifyError)
    assert issubclass(RoboflowAPIError, BimifyError)
    assert issubclass(IFCExportError, BimifyError)
    assert issubclass(GeometryValidationError, IFCExportError)
    assert issubclass(InsufficientConfidenceError, IFCExportError)
    assert issubclass(WallReconstructionError, IFCExportError)
    assert issubclass(MaterialAssignmentError, IFCExportError)
    assert issubclass(RepairError, BimifyError)

