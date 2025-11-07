from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

from celery import shared_task
from shapely.geometry import mapping

from core.preprocess.raster_pipeline import rasterize_pdf, deskew_and_normalize
from core.ml.roboflow_client import RFOptions, infer_floorplan
from core.ml.postprocess_floorplan import normalize_predictions, estimate_wall_axes_and_thickness
from core.reconstruct.spaces import polygonize_spaces_from_walls
from core.ifc.build_ifc43_model import collect_wall_polygons, write_ifc_with_spaces
from core.ifc.preprocess import resolve_preprocess_runtime, invoke_preprocessor
from core.validate.reconstruction_validation import generate_validation_report, write_validation_report
from core.vector.preview import save_geojson_thumbnail
from core.settings import get_settings
from core.storage.s3 import S3Storage
from services.api.jobs_store import FileJobStore


async def process_job_async(job_id: str) -> str:
    jid = UUID(job_id)
    jobs = FileJobStore(Path("data") / "jobs")
    job = jobs.load(jid)
    if not job:
        return "missing"

    job.status = "running"
    job.progress = 1
    jobs.save(job)

    timeline: list[dict] = []

    settings = get_settings()
    s3_storage: S3Storage | None = None
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        s3_storage = S3Storage(bucket=settings.storage.bucket, prefix=f"artifacts/{job_id}", region=settings.storage.region)

    def step_event(name: str):
        start_ts = time.perf_counter()
        started_at = datetime.utcnow().isoformat()

        def close() -> None:
            finished_at = datetime.utcnow().isoformat()
            duration = (time.perf_counter() - start_ts) * 1000.0
            timeline.append({
                "step": name,
                "durationMs": int(duration),
                "startedAt": started_at,
                "finishedAt": finished_at,
            })

        return close

    finish_prepare = step_event("prepare_job")
    out_dir = Path("data") / "jobs" / str(jid)
    rf_dir = out_dir / "roboflow"
    recon_dir = out_dir / "recon"
    ifc_dir = out_dir / "ifc"
    thumb_dir = out_dir / "thumbnails"
    rf_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    ifc_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    finish_prepare()

    def mirror(path: Path) -> None:
        if not s3_storage:
            return
        try:
            rel = path.relative_to(out_dir).as_posix()
        except ValueError:
            rel = path.name
        s3_storage.put_file(rel, path)

    def rel_path(path: Path) -> str:
        try:
            return path.relative_to(out_dir).as_posix()
        except ValueError:
            return path.as_posix()

    total = 0
    completed = 0

    meta = job.meta or {}
    rf_conf = float(meta.get("rf_confidence", 0.01))
    rf_overlap = float(meta.get("rf_overlap", 0.3))
    per_class = meta.get("per_class_thresholds") or {}
    # dynamic overrides for project/version from server settings or job meta
    server_settings_path = Path("data") / "settings.json"
    server_proj: str | None = None
    server_ver: int | None = None
    try:
        if server_settings_path.exists():
            srv = json.loads(server_settings_path.read_text(encoding="utf-8"))
            server_proj = (srv.get("roboflow_project") or None)
            if srv.get("roboflow_version") is not None:
                server_ver = int(srv.get("roboflow_version"))
    except Exception:
        pass
    rf_project = str(meta.get("roboflow_project") or server_proj or settings.roboflow.project)
    rf_version = int(meta.get("roboflow_version") or server_ver or settings.roboflow.version)

    all_norm = []

    preprocess_started: str | None = None
    preprocess_time = 0.0
    preprocess_finished: str | None = None
    infer_started: str | None = None
    infer_time = 0.0
    infer_finished: str | None = None
    normalize_started: str | None = None
    normalize_time = 0.0
    normalize_finished: str | None = None
    normalized_total = 0

    rf_api_key = str((meta.get("roboflow_api_key") or "")).strip()

    for uri in job.input_files or []:
        p = Path(uri)
        if p.suffix.lower() == ".pdf":
            if preprocess_started is None:
                preprocess_started = datetime.utcnow().isoformat()
            raster_start = time.perf_counter()
            pages = rasterize_pdf(str(p), dpi=400, out_dir=out_dir / "raster")
            preprocess_time += time.perf_counter() - raster_start
            preprocess_finished = datetime.utcnow().isoformat()
            if not pages:
                continue
            total += len(pages)
            for idx, page in enumerate(pages, start=1):
                deskew_start = time.perf_counter()
                img_path = Path(deskew_and_normalize(str(page.path)))
                preprocess_time += time.perf_counter() - deskew_start
                preprocess_finished = datetime.utcnow().isoformat()

                label = f"{p.stem}-p{idx}"

                infer_start = time.perf_counter()
                opts = RFOptions(
                    project=rf_project,
                    version=rf_version,
                    confidence=rf_conf,
                    overlap=rf_overlap,
                    per_class=per_class,
                )
                if infer_started is None:
                    infer_started = datetime.utcnow().isoformat()
                preds = await infer_floorplan(img_path, opts=opts, api_key_override=rf_api_key or None)
                infer_time += time.perf_counter() - infer_start
                infer_finished = datetime.utcnow().isoformat()

                norm_start = time.perf_counter()
                if normalize_started is None:
                    normalize_started = datetime.utcnow().isoformat()
                # pixels-per-millimeter based on raster DPI
                px_per_mm = float(page.dpi) / 25.4
                norm = normalize_predictions(
                    preds,
                    px_to_mm=px_per_mm,
                    per_class_thresholds=per_class,
                    global_threshold=rf_conf,
                )
                for nd in norm:
                    nd.attrs = {**nd.attrs, "page_index": idx, "source_label": label}
                all_norm.extend(norm)
                normalized_total += len(norm)

                features = []
                for nd in norm:
                    features.append({
                        "type": "Feature",
                        "properties": {"type": nd.type, "is_external": nd.is_external, **nd.attrs},
                        "geometry": mapping(nd.geom),
                    })
                geojson = {"type": "FeatureCollection", "features": features}
                recon_path = recon_dir / f"{label}-norm.geojson"
                recon_path.write_text(json.dumps(geojson), encoding="utf-8")
                mirror(recon_path)
                thumb_path = thumb_dir / f"{label}.png"
                save_geojson_thumbnail(geojson, thumb_path)
                mirror(thumb_path)
                normalize_time += time.perf_counter() - norm_start
                normalize_finished = datetime.utcnow().isoformat()

                completed += 1
                job.progress = int(70 * completed / max(1, total))
                job.meta = {**(job.meta or {}), "normalized_count": normalized_total}
                jobs.save(job)
        else:
            if preprocess_started is None:
                preprocess_started = datetime.utcnow().isoformat()
            pre_start = time.perf_counter()
            img_path = Path(deskew_and_normalize(str(p)))
            preprocess_time += time.perf_counter() - pre_start
            preprocess_finished = datetime.utcnow().isoformat()

            label = p.stem
            total += 1

            infer_start = time.perf_counter()
            opts = RFOptions(
                project=rf_project,
                version=rf_version,
                confidence=rf_conf,
                overlap=rf_overlap,
                per_class=per_class,
            )
            if infer_started is None:
                infer_started = datetime.utcnow().isoformat()
            preds = await infer_floorplan(img_path, opts=opts, api_key_override=rf_api_key or None)
            infer_time += time.perf_counter() - infer_start
            infer_finished = datetime.utcnow().isoformat()

            norm_start = time.perf_counter()
            if normalize_started is None:
                normalize_started = datetime.utcnow().isoformat()
            # fallback DPI when source is not a PDF page
            fallback_dpi = int(settings.preprocess.dpi)
            px_per_mm = float(fallback_dpi) / 25.4
            norm = normalize_predictions(
                preds,
                px_to_mm=px_per_mm,
                per_class_thresholds=per_class,
                global_threshold=rf_conf,
            )
            for nd in norm:
                nd.attrs = {**nd.attrs, "page_index": 1, "source_label": label}
            all_norm.extend(norm)
            normalized_total += len(norm)

            features = []
            for nd in norm:
                features.append({
                    "type": "Feature",
                    "properties": {"type": nd.type, "is_external": nd.is_external, **nd.attrs},
                    "geometry": mapping(nd.geom),
                })
            geojson = {"type": "FeatureCollection", "features": features}
            recon_path = recon_dir / f"{label}-norm.geojson"
            recon_path.write_text(json.dumps(geojson), encoding="utf-8")
            mirror(recon_path)
            thumb_path = thumb_dir / f"{label}.png"
            save_geojson_thumbnail(geojson, thumb_path)
            mirror(thumb_path)
            normalize_time += time.perf_counter() - norm_start
            normalize_finished = datetime.utcnow().isoformat()

            completed += 1
            job.progress = int(70 * completed / max(1, total))
            job.meta = {**(job.meta or {}), "normalized_count": normalized_total}
            jobs.save(job)

    # Persist walls with internal/external separation
    wall_features = []
    for nd in all_norm:
        if nd.type == "WALL":
            wall_features.append({
                "type": "Feature",
                "properties": {"type": "WALL", "is_external": nd.is_external, **nd.attrs},
                "geometry": mapping(nd.geom),
            })
    walls_path = recon_dir / "walls.geojson"
    walls_path.write_text(json.dumps({"type": "FeatureCollection", "features": wall_features}), encoding="utf-8")
    mirror(walls_path)

    if preprocess_started:
        timeline.append({
            "step": "preprocess",
            "durationMs": int(preprocess_time * 1000),
            "startedAt": preprocess_started,
            "finishedAt": preprocess_finished or preprocess_started,
        })
    if infer_started:
        timeline.append({
            "step": "inference",
            "durationMs": int(infer_time * 1000),
            "startedAt": infer_started,
            "finishedAt": infer_finished or infer_started,
        })
    if normalize_started:
        timeline.append({
            "step": "postprocess",
            "durationMs": int(normalize_time * 1000),
            "startedAt": normalize_started,
            "finishedAt": normalize_finished or normalize_started,
        })

    close_reconstruct = step_event("reconstruction")

    # Wall axes & thickness
    axes = estimate_wall_axes_and_thickness(all_norm)
    wall_polygons = collect_wall_polygons(all_norm)
    axes_features = []
    for axis_info in axes:
        axes_features.append({
            "type": "Feature",
            "properties": {
                "type": "WALL_AXIS",
                "is_external": axis_info.detection.is_external,
                "width_mm": axis_info.width_mm,
                "length_mm": axis_info.length_mm,
                "method": axis_info.method,
                "page_index": axis_info.detection.attrs.get("page_index") if isinstance(axis_info.detection.attrs, dict) else None,
            },
            "geometry": mapping(axis_info.axis),
        })
    axes_path = recon_dir / "walls_axes.geojson"
    axes_path.write_text(json.dumps({"type": "FeatureCollection", "features": axes_features}), encoding="utf-8")
    mirror(axes_path)

    # spaces from walls
    spaces = polygonize_spaces_from_walls(all_norm)
    space_features = [{
        "type": "Feature",
        "properties": {"type": "SPACE", "area_m2": sp.area_m2},
        "geometry": mapping(sp.polygon)
    } for sp in spaces]
    spaces_path = recon_dir / "spaces.geojson"
    spaces_path.write_text(json.dumps({"type": "FeatureCollection", "features": space_features}), encoding="utf-8")
    mirror(spaces_path)
    close_reconstruct()

    close_ifc = step_event("ifc_export")
    # write IFC with walls (IsExternal) and spaces + wall params
    ifc_path = ifc_dir / "model.ifc"
    write_ifc_with_spaces(
        all_norm,
        spaces,
        ifc_path,
        wall_axes=axes,
        wall_polygons=wall_polygons,
        schema_version=getattr(getattr(settings, "ifc", None), "schema", "IFC4"),
        wall_thickness_standards_mm=getattr(getattr(settings, "ifc", None), "wall_thickness_standards_mm", None),
    )
    mirror(ifc_path)

    validation_report_path = ifc_dir / "model_validation.json"
    try:
        validation_report = generate_validation_report(all_norm, axes, ifc_path)
        write_validation_report(validation_report, validation_report_path)
        mirror(validation_report_path)
    except Exception as validation_exc:
        print(f"[validation] report generation failed: {validation_exc}")
    close_ifc()

    ifc_meta = dict((job.meta or {}).get("ifc") or {})
    ifc_meta["primary"] = rel_path(ifc_path)
    job.meta = {**(job.meta or {}), "ifc": ifc_meta}

    preprocess_runtime = resolve_preprocess_runtime(settings)
    settings_preprocess = getattr(getattr(settings, "geometry", None), "preprocess", None)
    if settings_preprocess and getattr(settings_preprocess, "enabled", False) and not preprocess_runtime:
        print("[xbim] Geometrie-Vorverarbeitung ist aktiviert, aber es wurde kein Kommando gefunden – Schritt wird übersprungen.")

    xbim_result: dict | None = None
    improved_ifc_path = ifc_dir / "model_preproc.ifc"
    stats_path = ifc_dir / "model_preproc_stats.json"
    wexbim_path: Path | None = None
    if preprocess_runtime and preprocess_runtime.emit_wexbim:
        wexbim_path = ifc_dir / "model_preproc.wexbim"

    if preprocess_runtime:
        cleanup_targets: list[Path] = [improved_ifc_path]
        if preprocess_runtime.emit_stats:
            cleanup_targets.append(stats_path)
        if wexbim_path:
            cleanup_targets.append(wexbim_path)

        for artifact in cleanup_targets:
            try:
                artifact.unlink(missing_ok=True)
            except Exception:
                pass

        close_pre = step_event("ifc_preprocess")
        xbim_result = await invoke_preprocessor(
            preprocess_runtime,
            ifc_path,
            improved_ifc_path,
            stats_path if preprocess_runtime.emit_stats else None,
            wexbim_path,
        )
        close_pre()

        improved_available = bool(xbim_result.get("success")) if xbim_result else False
        if improved_available and improved_ifc_path.exists():
            mirror(improved_ifc_path)
            if preprocess_runtime.emit_stats and stats_path and stats_path.exists():
                mirror(stats_path)
            if wexbim_path and wexbim_path.exists():
                mirror(wexbim_path)

            updated_ifc_meta = dict((job.meta or {}).get("ifc") or {})
            updated_ifc_meta.setdefault("primary", rel_path(ifc_path))
            updated_ifc_meta["improved"] = rel_path(improved_ifc_path)
            if preprocess_runtime.emit_stats and stats_path and stats_path.exists():
                updated_ifc_meta["stats"] = rel_path(stats_path)
            if wexbim_path and wexbim_path.exists():
                updated_ifc_meta["wexbim"] = rel_path(wexbim_path)
            job.meta = {**(job.meta or {}), "ifc": updated_ifc_meta}
        elif xbim_result:
            stderr_tail = (xbim_result.get("stderr") or "").strip()
            if stderr_tail:
                print(f"[xbim] Preprocessing failed: {stderr_tail}")
            else:
                print("[xbim] Preprocessing failed without stderr output")

        if xbim_result:
            preprocess_meta = {
                "command": xbim_result.get("command"),
                "returncode": xbim_result.get("returncode"),
            }
            if xbim_result.get("timeout"):
                preprocess_meta["timeout"] = True
            stdout_text = xbim_result.get("stdout")
            if stdout_text:
                preprocess_meta["stdout_tail"] = stdout_text[-4000:]
            stderr_text = xbim_result.get("stderr")
            if stderr_text:
                preprocess_meta["stderr_tail"] = stderr_text[-4000:]
            job.meta = {**(job.meta or {}), "ifc_preprocess": preprocess_meta}

    job.status = "succeeded"
    job.progress = 100
    job.meta = {**(job.meta or {}), "wall_axes_count": len(axes), "normalized_count": normalized_total, "timeline": timeline}
    jobs.save(job)
    return "succeeded"


@shared_task(name="services.worker.tasks.process_job")
def process_job(job_id: str) -> str:
    # Celery runs in a separate process without FastAPI event loop; safe to use asyncio.run
    return asyncio.run(process_job_async(job_id))


