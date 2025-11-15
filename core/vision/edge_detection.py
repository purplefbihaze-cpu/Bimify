from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from shapely.geometry import LineString

from core.settings import get_settings


@dataclass
class EdgeDetectionParams:
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 80
    hough_min_line_length_mm: float = 200.0
    hough_max_line_gap_mm: float = 20.0


def _resolve_params() -> EdgeDetectionParams:
    settings = get_settings()
    params = EdgeDetectionParams()
    try:
        cfg = getattr(getattr(settings, "geometry", None), "repair_level1", None)
        if cfg:
            params.canny_low = int(getattr(cfg, "canny_low", params.canny_low))
            params.canny_high = int(getattr(cfg, "canny_high", params.canny_high))
            params.hough_threshold = int(getattr(cfg, "hough_threshold", params.hough_threshold))
            params.hough_min_line_length_mm = float(
                getattr(cfg, "hough_min_line_length_mm", params.hough_min_line_length_mm)
            )
            params.hough_max_line_gap_mm = float(
                getattr(cfg, "hough_max_line_gap_mm", params.hough_max_line_gap_mm)
            )
    except Exception:
        pass
    return params


def detect_edges_and_lines(
    image_bgr: np.ndarray,
    *,
    px_per_mm: Optional[float],
) -> Tuple[np.ndarray, List[LineString]]:
    """
    Run Canny + Probabilistic Hough to extract straight line candidates.
    Returns:
        edges: binary edge map (uint8)
        lines: list of LineString in millimeter coordinates
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), []

    params = _resolve_params()
    # Grayscale + slight blur to reduce pixel noise
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

    edges = cv2.Canny(gray, threshold1=params.canny_low, threshold2=params.canny_high)

    # Convert tolerances to pixels using px_per_mm
    if not px_per_mm or px_per_mm <= 0:
        px_per_mm = 1.0
    min_len_px = max(1, int(round(params.hough_min_line_length_mm * px_per_mm)))
    max_gap_px = max(1, int(round(params.hough_max_line_gap_mm * px_per_mm)))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=params.hough_threshold,
        minLineLength=min_len_px,
        maxLineGap=max_gap_px,
    )

    lines: List[LineString] = []
    if raw is not None:
        for entry in raw:
            try:
                x1, y1, x2, y2 = [float(v) for v in entry.reshape(-1)]
                # Convert to mm coordinates (origin in image pixel space)
                mm_line = LineString([(x1 / px_per_mm, y1 / px_per_mm), (x2 / px_per_mm, y2 / px_per_mm)])
                lines.append(mm_line)
            except Exception:
                continue

    return edges, lines


