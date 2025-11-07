import asyncio
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from core.settings import get_settings
from services.api.routes import router as v1_router


if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        # Older Python versions may not expose this policy; ignore if unavailable
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Bimify API",
        version="0.1.0",
        description="Upload handling, job orchestration and IFC exports",
    )

    # CORS for local UI and configurable origin
    ui_origin = os.getenv("UI_ORIGIN", "http://localhost:3000")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ui_origin, "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.on_event("startup")
    async def _load_settings() -> None:
        settings = get_settings()
        logger.info(
            "API initialised with Roboflow project={project} version={version}",
            project=settings.roboflow.project,
            version=settings.roboflow.version,
        )

    @app.get("/healthz", tags=["meta"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(v1_router)

    app.mount("/files", StaticFiles(directory="data/exports", check_dir=False), name="ifc-files")

    return app


app = create_app()


__all__ = ["app", "create_app"]

