from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic import ValidationError

from services.api.ifc_exporter import run_ifc_export
from services.api.jobs_store import FileJobStore, Job
from services.api.schemas import ExportIFCRequest, ExportIFCResponse


JOB_ROOT = Path("data") / "jobs"


def _normalize_ifc_meta(result: ExportIFCResponse) -> dict[str, str]:
    meta: dict[str, str] = {"primary": result.file_name}
    if result.improved_ifc_url:
        meta["improved"] = PurePosixPath(result.improved_ifc_url).name
    if result.improved_ifc_stats_url:
        meta["stats"] = PurePosixPath(result.improved_ifc_stats_url).name
    if result.improved_wexbim_url:
        meta["wexbim"] = PurePosixPath(result.improved_wexbim_url).name
    if result.validation_report_url:
        meta["validation"] = PurePosixPath(result.validation_report_url).name
    return meta


def _ensure_export_meta(job: Job) -> dict[str, Any]:
    meta = job.meta or {}
    export_meta = dict(meta.get("export_ifc") or {})
    meta.setdefault("export_ifc", export_meta)
    job.meta = meta
    return export_meta


async def process_ifc_job_async(job_id: UUID) -> None:
    store = FileJobStore(JOB_ROOT)
    job = store.load(job_id)
    if not job:
        logger.warning("[export-ifc-job] Job %s nicht gefunden", job_id)
        return

    export_meta = _ensure_export_meta(job)
    payload_data = export_meta.get("payload")
    if not payload_data:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Export-IFC Payload fehlt"}
        store.save(job)
        logger.error("[export-ifc-job] Payload für Job %s fehlt", job_id)
        return

    job.status = "running"
    job.progress = max(job.progress, 5)
    store.save(job)

    try:
        payload = ExportIFCRequest.model_validate(payload_data)
    except ValidationError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"Ungültige ExportIFCRequest: {exc}"}
        store.save(job)
        logger.exception("[export-ifc-job] Ungültige Payload für Job %s", job_id)
        return

    try:
        result = await run_ifc_export(payload)
    except Exception as exc:  # pragma: no cover - defensive
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": str(exc)}
        store.save(job)
        logger.exception("[export-ifc-job] Export fehlgeschlagen für Job %s", job_id)
        return

    export_meta["payload"] = payload.model_dump(mode="json")
    export_meta["result"] = result.model_dump(mode="json")
    job.meta.pop("error", None)
    job.meta["export_ifc"] = export_meta
    job.meta["ifc"] = _normalize_ifc_meta(result)
    job.status = "succeeded"
    job.progress = 100
    store.save(job)
    logger.info("[export-ifc-job] Job %s erfolgreich abgeschlossen", job_id)




