import asyncio
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from urllib.parse import urlparse

from core.settings import get_settings
from core.exceptions import BimifyError
from core.logging_config import setup_logging
from services.api.routes import router as v1_router
from services.api.routers.hottcad import router as hottcad_router
from services.api.exception_handlers import bimify_exception_handler
from services.api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware


if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        # Older Python versions may not expose this policy; ignore if unavailable
        pass


def create_app() -> FastAPI:
    # Cleanup on module reload (hot reload support)
    # This ensures ProcessPoolExecutor and other global resources are cleaned up
    try:
        from core.ifc.build_ifc43_model_v2 import _shutdown_shared_executor
        _shutdown_shared_executor()
    except Exception:
        pass  # Ignore errors during cleanup - module might not be loaded yet
    
    # Setup structured logging
    json_logging = os.getenv("JSON_LOGGING", "false").lower() in {"true", "1", "yes"}
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    setup_logging(
        level=log_level,
        json_format=json_logging,
        log_file=Path(log_file) if log_file else None,
    )
    
    app = FastAPI(
        title="Bimify API",
        version="0.1.0",
        description="Upload handling, job orchestration and IFC exports",
    )

    # CORS for local UI and configurable origin
    ui_origin = os.getenv("UI_ORIGIN", "http://localhost:3000")
    # Expand to include localhost/127.0.0.1 variants and default 3000 fallbacks
    allowed_origins: set[str] = {
        ui_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    }
    try:
        parsed = urlparse(ui_origin)
        if parsed.scheme and parsed.netloc:
            # Add both localhost and 127.0.0.1 variants
            if "localhost" in ui_origin:
                allowed_origins.add(ui_origin.replace("localhost", "127.0.0.1"))
            elif "127.0.0.1" in ui_origin:
                allowed_origins.add(ui_origin.replace("127.0.0.1", "localhost"))
            # Also add port variations
            if ":" in parsed.netloc:
                port = parsed.netloc.split(":")[1]
                if "localhost" in ui_origin:
                    allowed_origins.add(f"http://127.0.0.1:{port}")
                elif "127.0.0.1" in ui_origin:
                    allowed_origins.add(f"http://localhost:{port}")
    except Exception:
        pass
    
    # In development, allow all localhost origins
    if os.getenv("ENVIRONMENT", "").lower() in {"dev", "development", "local"}:
        allowed_origins.update([
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ])
    
    logger.info(f"CORS allowed origins: {sorted(allowed_origins)}")
    
    # Security middleware (applied first in execution order)
    rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"true", "1", "yes"}
    if rate_limit_enabled:
        requests_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        requests_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
        )
    
    app.add_middleware(SecurityHeadersMiddleware)
    
    # CORS middleware - add LAST so it executes FIRST (FastAPI executes middleware in reverse order)
    # In development, be very permissive with localhost
    is_dev = os.getenv("ENVIRONMENT", "").lower() in {"dev", "development", "local"} or not os.getenv("ENVIRONMENT")
    if is_dev:
        # In development, allow all localhost variants (can't use "*" with allow_credentials=True)
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ]
        # Add all ports from 3000 to 3010 for development
        for port in range(3000, 3011):
            cors_origins.extend([
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
            ])
        cors_origins = list(set(cors_origins))  # Remove duplicates
        # Also include any from allowed_origins
        cors_origins.extend(allowed_origins)
        cors_origins = list(set(cors_origins))  # Remove duplicates again
    else:
        cors_origins = list(allowed_origins)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.on_event("startup")
    async def _load_settings() -> None:
        # Ensure ProcessPoolExecutor is ready (recreate if needed after hot reload)
        try:
            from core.ifc.build_ifc43_model_v2 import _get_shared_executor
            _get_shared_executor()  # Ensure executor exists
        except Exception as e:
            logger.warning(f"Could not initialize ProcessPoolExecutor (non-critical): {e}")
        
        settings = get_settings()
        logger.info(
            "API initialised with Roboflow project={project} version={version}",
            project=settings.roboflow.project,
            version=settings.roboflow.version,
        )
    
    @app.on_event("shutdown")
    async def _cleanup_on_shutdown() -> None:
        """Cleanup resources on server shutdown (including hot reload)."""
        try:
            from core.ifc.build_ifc43_model_v2 import _shutdown_shared_executor
            _shutdown_shared_executor()
        except Exception as e:
            logger.debug(f"Error during shutdown cleanup (non-critical): {e}")

    @app.get("/healthz", tags=["meta"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    # Register exception handlers
    app.add_exception_handler(BimifyError, bimify_exception_handler)
    
    # Add general exception handler to ensure CORS headers are set even on unexpected errors
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions and ensure CORS headers are set."""
        logger.exception("Unhandled exception", exc_info=exc)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )

    # Include routers
    app.include_router(v1_router)
    app.include_router(hottcad_router)

    app.mount("/files", StaticFiles(directory="data/exports", check_dir=False), name="ifc-files")
    app.mount("/previews", StaticFiles(directory="data/previews", check_dir=False), name="previews")

    return app


app = create_app()


__all__ = ["app", "create_app"]

