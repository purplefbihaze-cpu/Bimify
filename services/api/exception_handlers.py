"""FastAPI exception handlers for custom exceptions."""

from __future__ import annotations

from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from core.exceptions import (
    BimifyError,
    ConfigurationError,
    ValidationError,
    IFCValidationError,
    GeometryError,
    MLInferenceError,
    RoboflowAPIError,
    ExportError,
    IFCExportError,
    PDFExportError,
    RepairError,
    StorageError,
    JobError,
    JobNotFoundError,
    FileNotFoundError as BimifyFileNotFoundError,
    TopViewError,
)


async def bimify_exception_handler(request: Request, exc: BimifyError) -> JSONResponse:
    """Handle Bimify-specific exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Map exception types to HTTP status codes
    if isinstance(exc, (ConfigurationError, ValidationError, IFCValidationError)):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, (JobNotFoundError, BimifyFileNotFoundError)):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, (RoboflowAPIError, MLInferenceError)):
        status_code = status.HTTP_502_BAD_GATEWAY
    elif isinstance(exc, (ExportError, RepairError, GeometryError, TopViewError)):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, StorageError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    logger.error(
        "Bimify exception: {type} - {message}",
        type=type(exc).__name__,
        message=str(exc),
        details=exc.details,
    )
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )

