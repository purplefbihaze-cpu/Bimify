from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:  # optional dependency
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


@dataclass
class ScaleEstimate:
    px_per_mm: float
    confidence: float


def estimate_scale(image_paths: List[Path]) -> ScaleEstimate:
    """Estimate px/mm from OCR of common scale notations (e.g., '1:100').

    Fallback: 1.0 px/mm with low confidence if OCR unavailable or no hit.
    """
    if pytesseract is None:
        return ScaleEstimate(px_per_mm=1.0, confidence=0.1)

    text_blobs: List[str] = []
    for p in image_paths:
        try:
            txt = pytesseract.image_to_string(str(p))  # type: ignore[attr-defined]
            text_blobs.append(txt)
        except Exception:
            continue
    text = "\n".join(text_blobs)

    # look for scale patterns like 1:50, 1:100 etc.
    m = re.search(r"\b1\s*:\s*(\d{2,4})\b", text)
    if m:
        denom = float(m.group(1))
        # assume 1 drawing unit = 1 mm at denom; so 1 mm on drawing equals denom mm in real.
        # Our images are in px; without DPI we cannot be exact. Use heuristic via 400 dpi raster (≈15.75 px/mm) / denom.
        # If rasterize_pdf used 400 dpi, approximate px/mm ≈ 15.75/denom, else fallback.
        px_per_mm_raster400 = 400.0 / 25.4
        return ScaleEstimate(px_per_mm=px_per_mm_raster400 / denom, confidence=0.6)

    return ScaleEstimate(px_per_mm=1.0, confidence=0.1)


def extract_room_labels(image_path: Path) -> List[Tuple[str, Tuple[float, float]]]:
    # Placeholder: no OCR yet
    return []


