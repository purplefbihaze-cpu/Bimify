from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urlparse

from inference_sdk import InferenceHTTPClient

from core.settings import get_settings


@dataclass
class RFOptions:
    project: str
    version: int
    confidence: float = 0.01
    overlap: float = 0.3
    per_class: dict[str, float] | None = None


@dataclass
class RFPred:
    doc: int
    page: int
    klass: str
    confidence: float
    polygon: List[Tuple[float, float]] | None
    bbox: Tuple[float, float, float, float] | None


SERVERLESS_HOSTS = {
    "serverless.roboflow.com",
    "detect.roboflow.com",
    "outline.roboflow.com",
    "classify.roboflow.com",
    "infer.roboflow.com",
}


def _normalise_url(value: str) -> str:
    if "//" in value:
        return value
    return f"https://{value}"


def _extract_host(api_url: str) -> str:
    parsed = urlparse(_normalise_url(api_url))
    host = (parsed.netloc or parsed.path).strip().lower()
    return host


def _is_serverless_endpoint(api_url: str) -> bool:
    host = _extract_host(api_url)
    return any(host == candidate or host.endswith(f".{candidate}") for candidate in SERVERLESS_HOSTS)


def _project_id_only(value: str) -> str:
    parts = [segment for segment in (value or "").split("/") if segment]
    return parts[-1] if parts else value


def _extract_point_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            try:
                return float(point["x"]), float(point["y"])
            except (TypeError, ValueError):
                return None
        candidates = [point.get(key) for key in (0, 1, "0", "1")]
        if candidates[0] is not None and candidates[1] is not None:
            try:
                return float(candidates[0]), float(candidates[1])
            except (TypeError, ValueError):
                return None
        ordered = list(point.values())
        if len(ordered) >= 2:
            try:
                return float(ordered[0]), float(ordered[1])
            except (TypeError, ValueError):
                return None
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            return float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
    return None


@lru_cache(maxsize=8)
def _serverless_base(api_url: str) -> str:
    parsed = urlparse(_normalise_url(api_url))
    scheme = parsed.scheme or "https"
    host = _extract_host(api_url)
    if host.startswith("serverless."):
        host = host.replace("serverless.", "detect.", 1)
    elif host in {"serverless.roboflow.com", "infer.roboflow.com"}:
        host = "detect.roboflow.com"
    return f"{scheme}://{host}".rstrip("/")


@lru_cache(maxsize=8)
def _get_client(api_url: str, api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    if _is_serverless_endpoint(api_url):
        client.select_api_v0()
    return client


async def _infer_with_client(
    client: InferenceHTTPClient,
    image_path: Path,
    model_id: str,
    *,
    confidence: float,
    overlap: float,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def _call() -> dict[str, Any]:
        configuration = replace(
            client.inference_configuration,
            confidence_threshold=confidence,
            iou_threshold=overlap,
        )
        with client.use_configuration(configuration):
            return client.infer(str(image_path), model_id=model_id)

    return await loop.run_in_executor(None, _call)


async def infer_floorplan_with_raw(
    image_path: Path,
    opts: RFOptions | None = None,
    api_key_override: str | None = None,
) -> tuple[list[RFPred], dict[str, Any]]:
    settings = get_settings().roboflow
    options = opts or RFOptions(
        project=settings.project,
        version=settings.version,
        confidence=settings.confidence,
        overlap=settings.overlap,
        per_class=settings.per_class_thresholds or None,
    )
    api_key = (api_key_override or "").strip() or settings.api_key
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set")
    base_url = settings.api_url or "https://serverless.roboflow.com"
    api_url = base_url.rstrip("/")
    if _is_serverless_endpoint(api_url):
        api_url = _serverless_base(api_url)
    project_id = _project_id_only(options.project)
    model_id = f"{project_id}/{options.version}"
    raw_per_class_thresholds = options.per_class or {}
    per_class_thresholds = {
        klass: float(value) for klass, value in raw_per_class_thresholds.items()
    }
    global_confidence = float(options.confidence)
    min_confidence = min(
        [global_confidence, *per_class_thresholds.values()]
    ) if per_class_thresholds else global_confidence

    try:
        client = _get_client(api_url, api_key)
        data = await _infer_with_client(
            client,
            image_path,
            model_id,
            confidence=min_confidence,
            overlap=float(options.overlap),
        )
    except Exception as exc:
        raise RuntimeError(f"Roboflow inference failed: {exc}") from exc
    if isinstance(data, dict):
        predictions_source = list(data.get("predictions", []))
    elif isinstance(data, list):
        predictions_source = list(data)
    else:
        predictions_source = []

    if per_class_thresholds or global_confidence:
        filtered_predictions: list[dict[str, Any]] = []
        for pred in predictions_source:
            klass = str(pred.get("class", ""))
            threshold = per_class_thresholds.get(klass, global_confidence)
            confidence_value = float(pred.get("confidence", 0.0))
            if confidence_value < threshold:
                continue
            filtered_predictions.append(pred)
        if isinstance(data, dict):
            data = {**data, "predictions": filtered_predictions}
        elif isinstance(data, list):
            data = filtered_predictions
        predictions_source = filtered_predictions

    preds: list[RFPred] = []
    for p in predictions_source:
        poly = None
        pts = p.get("points") or p.get("polygon")
        if pts:
            parsed: list[tuple[float, float]] = []
            for pt in pts:
                coords = _extract_point_xy(pt)
                if coords is None:
                    continue
                parsed.append(coords)
            if parsed:
                poly = parsed
        bbox = None
        if all(k in p for k in ("x", "y", "width", "height")):
            x = float(p["x"]) - float(p["width"]) / 2.0
            y = float(p["y"]) - float(p["height"]) / 2.0
            bbox = (x, y, float(p["width"]), float(p["height"]))
        preds.append(
            RFPred(
                doc=0,
                page=0,
                klass=str(p.get("class", "")),
                confidence=float(p.get("confidence", 0.0)),
                polygon=poly,
                bbox=bbox,
            )
        )
    return preds, data


async def infer_floorplan(image_path: Path, opts: RFOptions | None = None, api_key_override: str | None = None) -> list[RFPred]:
    preds, _ = await infer_floorplan_with_raw(image_path, opts=opts, api_key_override=api_key_override)
    return preds


