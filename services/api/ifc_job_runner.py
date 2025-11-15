from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic import ValidationError

from services.api.ifc_exporter import run_ifc_export
from services.api.ifc_exporter_v2 import run_ifc_export_v2
from services.api.jobs_store import FileJobStore, Job
from services.api.schemas import (
    ExportIFCRequest,
    ExportIFCResponse,
    ExportIFCV2Request,
    ExportIFCV2Response,
    IFCRepairRequest,
    IFCRepairResponse,
    IFCRepairPreviewRequest,
    IFCRepairPreviewResponse,
)
from services.api.ifc_repair import run_ifc_repair, RepairError
from core.vector.ifc_topview import build_topview_geojson


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


# ============================================================================
# IFC EXPORT V2 - NEW IMPLEMENTATION
# ============================================================================
# This is a completely new IFC export implementation (V2).
# It uses a different logic path than the original IFC export.
# All V2-related code is clearly marked with "V2" suffix.
# ============================================================================

def _normalize_ifc_v2_meta(result: ExportIFCV2Response) -> dict[str, str]:
    """V2: Normalize IFC metadata for V2 response."""
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


def _ensure_export_v2_meta(job: Job) -> dict[str, Any]:
    """V2: Ensure export metadata exists for V2 job."""
    meta = job.meta or {}
    export_meta = dict(meta.get("export_ifc_v2") or {})
    meta.setdefault("export_ifc_v2", export_meta)
    job.meta = meta
    return export_meta


async def process_ifc_v2_job_async(job_id: UUID) -> None:
    """Process IFC export V2 job with progress updates."""
    logger.info("[export-ifc-v2-job] Starte Verarbeitung für Job %s", job_id)
    store = FileJobStore(JOB_ROOT)
    job = store.load(job_id)
    if not job:
        logger.warning("[export-ifc-v2-job] Job %s nicht gefunden", job_id)
        return

    logger.debug("[export-ifc-v2-job] Job %s geladen, aktueller Status: %s, Progress: %d", job_id, job.status, job.progress)

    export_meta = _ensure_export_v2_meta(job)
    payload_data = export_meta.get("payload")
    if not payload_data:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Export-IFC-V2 Payload fehlt"}
        store.save(job)
        logger.error("[export-ifc-v2-job] Payload für Job %s fehlt", job_id)
        return

    logger.info("[export-ifc-v2-job] Setze Job %s auf Status 'running'", job_id)
    job.status = "running"
    job.progress = 10
    store.save(job)
    logger.debug("[export-ifc-v2-job] Job %s Status aktualisiert: running, Progress: 10", job_id)

    try:
        logger.debug("[export-ifc-v2-job] Validiere Payload für Job %s", job_id)
        payload = ExportIFCV2Request.model_validate(payload_data)
        logger.debug("[export-ifc-v2-job] Payload für Job %s erfolgreich validiert (%d predictions)", job_id, len(payload.predictions))
    except ValidationError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"Ungültige ExportIFCV2Request: {exc}"}
        store.save(job)
        logger.exception("[export-ifc-v2-job] Ungültige Payload für Job %s: %s", job_id, exc, exc_info=True)
        return

    try:
        logger.info("[export-ifc-v2-job] Starte IFC Export V2 für Job %s", job_id)
        job.progress = 20
        store.save(job)
        logger.debug("[export-ifc-v2-job] Job %s Progress aktualisiert: 20", job_id)
        
        result = await run_ifc_export_v2(payload)
        logger.info("[export-ifc-v2-job] IFC Export V2 für Job %s erfolgreich abgeschlossen", job_id)
        
        job.progress = 90
        store.save(job)
        logger.debug("[export-ifc-v2-job] Job %s Progress aktualisiert: 90", job_id)
    except Exception as exc:  # pragma: no cover - defensive
        job.status = "failed"
        job.progress = 100
        error_msg = str(exc)
        job.meta = {**(job.meta or {}), "error": error_msg}
        try:
            store.save(job)
        except Exception as save_exc:
            logger.error(f"[export-ifc-v2-job] Error saving failed job: {save_exc}")
        logger.exception(
            "[export-ifc-v2-job] Export fehlgeschlagen für Job %s: %s",
            job_id,
            error_msg,
            exc_info=True,
        )
        return

    # Save result with timeout and error handling to prevent hangs
    logger.debug("[export-ifc-v2-job] Speichere Ergebnis für Job %s", job_id)
    try:
        # Serialize payload and result with timeout (10 seconds max)
        # Run in thread to prevent blocking
        async def _serialize_data() -> tuple[dict, dict]:
            def _serialize_sync() -> tuple[dict, dict]:
                try:
                    payload_dict = payload.model_dump(mode="json")
                except Exception as e:
                    logger.warning(f"[export-ifc-v2-job] Error serializing payload: {e}")
                    payload_dict = {"error": f"Serialization error: {e}"}
                
                try:
                    result_dict = result.model_dump(mode="json")
                except Exception as e:
                    logger.warning(f"[export-ifc-v2-job] Error serializing result: {e}")
                    result_dict = {"error": f"Serialization error: {e}"}
                
                return payload_dict, result_dict
            
            return await asyncio.to_thread(_serialize_sync)
        
        payload_dict, result_dict = await asyncio.wait_for(_serialize_data(), timeout=10.0)
        
        export_meta["payload"] = payload_dict
        export_meta["result"] = result_dict
        job.meta.pop("error", None)
        job.meta["export_ifc_v2"] = export_meta
        
        # Normalize metadata with timeout (5 seconds max)
        try:
            job.meta["ifc_v2"] = await asyncio.wait_for(
                asyncio.to_thread(_normalize_ifc_v2_meta, result),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("[export-ifc-v2-job] Metadata normalization timed out, using minimal metadata")
            job.meta["ifc_v2"] = {"primary": result.file_name or "unknown"}
        except Exception as e:
            logger.warning(f"[export-ifc-v2-job] Error normalizing metadata: {e}")
            job.meta["ifc_v2"] = {"primary": result.file_name or "unknown"}
        
        job.status = "succeeded"
        job.progress = 100
        
        # Save job with timeout (5 seconds max)
        try:
            await asyncio.wait_for(asyncio.to_thread(store.save, job), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("[export-ifc-v2-job] Job save timed out - job may be incomplete")
            # Try one more time without timeout as fallback
            try:
                store.save(job)
            except Exception as save_exc:
                logger.error(f"[export-ifc-v2-job] Error saving job (fallback): {save_exc}")
        except Exception as save_exc:
            logger.error(f"[export-ifc-v2-job] Error saving job: {save_exc}")
        
        logger.info("[export-ifc-v2-job] Job %s erfolgreich abgeschlossen (Status: succeeded, Progress: 100)", job_id)
    except asyncio.TimeoutError:
        logger.error("[export-ifc-v2-job] Result serialization timed out - marking job as failed")
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Result serialization timed out"}
        try:
            store.save(job)
        except Exception:
            pass
    except Exception as save_exc:
        logger.error(f"[export-ifc-v2-job] Unexpected error saving result: {save_exc}", exc_info=True)
        # Try to save error state
        try:
            job.status = "failed"
            job.progress = 100
            job.meta = {**(job.meta or {}), "error": f"Error saving result: {save_exc}"}
            store.save(job)
        except Exception:
            pass




def _ensure_repair_meta(job: Job) -> dict[str, Any]:
    meta = job.meta or {}
    repair_meta = dict(meta.get("repair_ifc") or {})
    meta.setdefault("repair_ifc", repair_meta)
    job.meta = meta
    return repair_meta


async def process_repair_job_async(job_id: UUID) -> None:
    store = FileJobStore(JOB_ROOT)
    job = store.load(job_id)
    if not job:
        logger.warning("[repair-ifc-job] Job %s nicht gefunden", job_id)
        return

    repair_meta = _ensure_repair_meta(job)
    payload_data = repair_meta.get("payload")
    if not payload_data:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Repair-IFC Payload fehlt"}
        store.save(job)
        logger.error("[repair-ifc-job] Payload für Job %s fehlt", job_id)
        return

    job.status = "running"
    job.progress = max(job.progress, 5)
    store.save(job)

    try:
        payload = IFCRepairRequest.model_validate(payload_data)
    except ValidationError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"Ungültige IFCRepairRequest: {exc}"}
        store.save(job)
        logger.exception("[repair-ifc-job] Ungültige Payload für Job %s", job_id)
        return

    # Bestimme die Quelldatei ähnlich wie im Sync-Endpoint
    export_root = Path("data") / "exports"
    file_name = payload.file_name
    if not file_name and payload.ifc_url:
        try:
            file_name = PurePosixPath(payload.ifc_url).name
        except Exception:
            file_name = None
    if not file_name:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "file_name oder ifc_url muss gesetzt sein"}
        store.save(job)
        return

    source_path = export_root / file_name
    try:
        source_path = source_path.resolve(strict=True)
    except FileNotFoundError:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"IFC-Datei {file_name} nicht gefunden"}
        store.save(job)
        return

    if export_root.resolve() not in source_path.parents:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Ungültiger IFC-Pfad"}
        store.save(job)
        return

    # Reparatur durchführen (in Thread, um Event-Loop nicht zu blockieren)
    job.progress = 25
    store.save(job)
    try:
        import asyncio as _asyncio  # local alias to avoid confusion
        repaired_path, warnings = await _asyncio.to_thread(
            run_ifc_repair, source_path, payload.level, export_root
        )
    except RepairError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": str(exc)}
        store.save(job)
        logger.exception("[repair-ifc-job] Reparatur fehlgeschlagen für Job %s", job_id)
        return
    except Exception as exc:  # pragma: no cover - defensive
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": str(exc)}
        store.save(job)
        logger.exception("[repair-ifc-job] Unerwarteter Fehler für Job %s", job_id)
        return

    # TopView erzeugen (best effort, ebenfalls im Thread)
    job.progress = 75
    store.save(job)
    topview_url: str | None = None
    try:
        topview_path = repaired_path.with_name(f"{repaired_path.stem}_topview.geojson")
        await _asyncio.to_thread(build_topview_geojson, repaired_path, topview_path, None)
        topview_url = f"/files/{topview_path.name}"
    except Exception:
        logger.warning("[repair-ifc-job] TopView konnte nicht erstellt werden", exc_info=True)

    result = IFCRepairResponse(
        file_name=repaired_path.name,
        ifc_url=f"/files/{repaired_path.name}",
        level=payload.level,
        topview_url=topview_url,
        warnings=warnings or None,
    )

    repair_meta["payload"] = payload.model_dump(mode="json")
    repair_meta["result"] = result.model_dump(mode="json")
    job.meta.pop("error", None)
    job.meta["repair_ifc"] = repair_meta
    job.status = "succeeded"
    job.progress = 100
    store.save(job)
    logger.info("[repair-ifc-job] Job %s erfolgreich abgeschlossen", job_id)


def _ensure_repair_preview_meta(job: Job) -> dict[str, Any]:
    meta = job.meta or {}
    preview_meta = dict(meta.get("repair_preview") or {})
    meta.setdefault("repair_preview", preview_meta)
    job.meta = meta
    return preview_meta


async def process_repair_preview_job_async(job_id: UUID) -> None:
    store = FileJobStore(JOB_ROOT)
    job = store.load(job_id)
    if not job:
        logger.warning("[repair-preview-job] Job %s nicht gefunden", job_id)
        return

    preview_meta = _ensure_repair_preview_meta(job)
    payload_data = preview_meta.get("payload")
    if not payload_data:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Repair-Preview Payload fehlt"}
        store.save(job)
        logger.error("[repair-preview-job] Payload für Job %s fehlt", job_id)
        return

    job.status = "running"
    job.progress = 10
    store.save(job)

    try:
        payload = IFCRepairPreviewRequest.model_validate(payload_data)
    except ValidationError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"Ungültige IFCRepairPreviewRequest: {exc}"}
        store.save(job)
        logger.exception("[repair-preview-job] Ungültige Payload für Job %s", job_id)
        return

    # Resolve IFC source from file_name or ifc_url
    export_root = Path("data") / "exports"
    file_name = payload.file_name
    if not file_name and payload.ifc_url:
        from urllib.parse import urlparse
        parsed = urlparse(payload.ifc_url)
        path_part = parsed.path or payload.ifc_url
        file_name = PurePosixPath(path_part).name

    if not file_name:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "file_name oder ifc_url muss gesetzt sein (Preview)"}
        store.save(job)
        return

    try:
        source_path = (export_root / file_name).resolve(strict=True)
    except FileNotFoundError:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"IFC-Datei {file_name} nicht gefunden"}
        store.save(job)
        return

    if export_root.resolve() not in source_path.parents:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": "Ungültiger IFC-Pfad"}
        store.save(job)
        return

    # Estimate processing time based on file size
    try:
        file_size_mb = source_path.stat().st_size / (1024 * 1024)
        estimated_seconds = max(5, min(120, int(file_size_mb * 8)))
        logger.info("[repair-preview-job] IFC-Datei %s ist %.2f MB, geschätzte Zeit: ~%d Sekunden", file_name, file_size_mb, estimated_seconds)
    except OSError:
        estimated_seconds = 30
        logger.warning("[repair-preview-job] Konnte Dateigröße nicht ermitteln")

    # Build normalized detections and axes (with optional image refinement)
    job.progress = 30
    store.save(job)
    try:
        import asyncio as _asyncio
        import cv2
        import numpy as np
        from urllib.request import urlopen
        from services.api.ifc_repair import build_preview_axes

        image_bgr = None
        px_per_mm_val = None
        if payload.image_url:
            try:
                with urlopen(payload.image_url) as resp:
                    data = resp.read()
                    image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as img_exc:
                logger.warning("[repair-preview-job] Bild konnte nicht geladen werden: %s", img_exc)
                image_bgr = None

        logger.info("[repair-preview-job] Starte build_preview_axes für %s", source_path.name)
        normalized, axes = await _asyncio.to_thread(build_preview_axes, source_path, px_per_mm=px_per_mm_val, image_bgr=image_bgr, rf_norm=None)
        logger.info("[repair-preview-job] build_preview_axes abgeschlossen - normalized: %d, axes: %d", len(normalized), len(axes))
    except FileNotFoundError as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"IFC-Datei nicht gefunden: {exc}"}
        store.save(job)
        logger.exception("[repair-preview-job] IFC-Datei nicht gefunden")
        return
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)
        job.status = "failed"
        job.progress = 100
        if "Keine verwertbare Geometrie" in error_msg or "RepairError" in error_type:
            job.meta = {**(job.meta or {}), "error": f"IFC-Datei enthält keine verwertbare Geometrie: {error_msg}"}
        else:
            job.meta = {**(job.meta or {}), "error": f"Preview fehlgeschlagen ({error_type}): {error_msg}"}
        store.save(job)
        logger.exception("[repair-preview-job] Pipeline konnte nicht ausgeführt werden (%s: %s)", error_type, error_msg)
        return

    # Build overlay GeoJSON (axes + walls)
    job.progress = 60
    store.save(job)
    try:
        from core.reports.overlay_builder import build_overlay, write_overlay
        import json
        import time
        from uuid import UUID as UUIDType

        # Validate that we have data to work with
        if not normalized:
            job.status = "failed"
            job.progress = 100
            job.meta = {**(job.meta or {}), "error": "Keine normalisierte Geometrie gefunden - IFC-Datei enthält möglicherweise keine Wände"}
            store.save(job)
            return
        if not axes:
            logger.warning("[repair-preview-job] Keine Achsen gefunden, aber normalized vorhanden (%d Elemente)", len(normalized))

        logger.info("[repair-preview-job] Erstelle Overlay mit %d normalized, %d axes", len(normalized), len(axes))
        artifacts = build_overlay(normalized=normalized, axes=axes)

        # Validate overlay was created successfully
        if not artifacts or not artifacts.overlay:
            job.status = "failed"
            job.progress = 100
            job.meta = {**(job.meta or {}), "error": "Overlay konnte nicht erstellt werden"}
            store.save(job)
            return

        feature_count = len(artifacts.overlay.get("features", []))
        logger.info("[repair-preview-job] Overlay erstellt mit %d Features", feature_count)

        preview_root = Path("data") / "previews"
        export_root = Path("data") / "exports"
        preview_root.mkdir(parents=True, exist_ok=True)
        export_root.mkdir(parents=True, exist_ok=True)
        preview_id = str(UUIDType(int=int(time.time() * 1e6)))
        # Save overlay to EXPORT_ROOT so it can be served via /files endpoint
        overlay_path = export_root / f"repair_preview_{preview_id}.geojson"
        write_overlay(artifacts, overlay_path)

        # Validate file was written successfully
        if not overlay_path.exists():
            job.status = "failed"
            job.progress = 100
            job.meta = {**(job.meta or {}), "error": f"Overlay-Datei konnte nicht geschrieben werden: {overlay_path}"}
            store.save(job)
            return
        file_size = overlay_path.stat().st_size
        logger.info("[repair-preview-job] Overlay-Datei geschrieben: %s (%d bytes)", overlay_path.name, file_size)

        # Progress: Overlay created and saved
        job.progress = 80
        store.save(job)

        # Persist minimal preview meta
        meta_data = {
            "source_file": source_path.name,
            "level": int(payload.level or 1),
        }
        (preview_root / f"{preview_id}.json").write_text(json.dumps(meta_data), encoding="utf-8")

        # Optional: propose topview (same as current for preview)
        proposed_topview_url: str | None = None
        try:
            topview_path = source_path.with_name(f"{source_path.stem}_topview.geojson")
            if not topview_path.exists():
                await _asyncio.to_thread(build_topview_geojson, source_path, topview_path, section_elevation_mm=None)
            proposed_topview_url = f"/files/{topview_path.name}"
        except Exception:
            proposed_topview_url = None

        metrics = artifacts.metrics
        overlay_url = f"/files/{overlay_path.name}"

        # Ensure overlay_url is always set, even if empty
        if not overlay_url:
            logger.warning("[repair-preview-job] overlay_url ist leer, setze Standard-URL")
            overlay_url = f"/files/{overlay_path.name}"

        # Validate metrics exist
        if not metrics:
            metrics = {
                "total_walls_src": 0,
                "total_axes": 0,
                "median_iou": 0.0,
            }

        logger.info("[repair-preview-job] Response vorbereitet - overlay_url: %s, features: %d", overlay_url, feature_count)
        result = IFCRepairPreviewResponse(
            preview_id=preview_id,
            level=payload.level,
            overlay_url=overlay_url,
            heatmap_url=None,
            proposed_topview_url=proposed_topview_url,
            metrics=metrics,
            warnings=None,
            estimated_seconds=estimated_seconds,
        )

        preview_meta["payload"] = payload.model_dump(mode="json")
        preview_meta["result"] = result.model_dump(mode="json")
        job.meta.pop("error", None)
        job.meta["repair_preview"] = preview_meta
        job.status = "succeeded"
        job.progress = 100
        store.save(job)
        logger.info("[repair-preview-job] Job %s erfolgreich abgeschlossen", job_id)
    except Exception as exc:
        job.status = "failed"
        job.progress = 100
        job.meta = {**(job.meta or {}), "error": f"Preview fehlgeschlagen (Overlay): {exc}"}
        store.save(job)
        logger.exception("[repair-preview-job] Overlay-Erstellung fehlgeschlagen")
        return

