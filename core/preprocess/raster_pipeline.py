from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pypdfium2 as pdfium


@dataclass
class RasterPage:
    path: Path
    dpi: int
    width_px: int
    height_px: int


def rasterize_pdf(pdf_path: str, dpi: int, out_dir: Path) -> List[RasterPage]:
    src = Path(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(src))
    results: List[RasterPage] = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=dpi / 72.0).to_pil()
        out_path = out_dir / f"page-{i}.png"
        pil.save(out_path)
        results.append(RasterPage(path=out_path, dpi=dpi, width_px=pil.width, height_px=pil.height))
    return results


def deskew_and_normalize(image_path: str) -> str:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return str(image_path)
    # threshold
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # edges and Hough to estimate rotation
    edges = cv2.Canny(thr, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            a = theta * 180.0 / np.pi
            # normalize around 0 or 90
            a = ((a + 45) % 90) - 45
            angles.append(a)
        if angles:
            angle = float(np.median(angles))
    # rotate around center
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out_path = Path(image_path).with_suffix(".deskew.png")
    cv2.imwrite(str(out_path), rotated)
    return str(out_path)


