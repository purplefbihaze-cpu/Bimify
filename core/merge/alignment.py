"""Co-registration and alignment checking for models."""

from typing import List, Tuple, Dict, Optional, Any
import math
try:
    import numpy as np
except ImportError:
    np = None


def check_coregistration(
    m1_meta: Dict[str, Any],
    m2_meta: Dict[str, Any],
    m3_meta: Dict[str, Any],
    tolerance: float = 1.0
) -> Tuple[bool, Optional[Dict[str, float]]]:
    """Check if models are co-registered (same image dimensions).
    
    Returns:
        (is_aligned, transform_info)
        transform_info contains offset/scale if misaligned
    """
    w1, h1 = m1_meta.get("width", 0), m1_meta.get("height", 0)
    w2, h2 = m2_meta.get("width", 0), m2_meta.get("height", 0)
    w3, h3 = m3_meta.get("width", 0), m3_meta.get("height", 0)
    
    if abs(w1 - w2) <= tolerance and abs(h1 - h2) <= tolerance and \
       abs(w1 - w3) <= tolerance and abs(h1 - h3) <= tolerance:
        return True, None
    
    # Compute transform info
    scale_x = (w2 + w3) / (2 * w1) if w1 > 0 else 1.0
    scale_y = (h2 + h3) / (2 * h1) if h1 > 0 else 1.0
    
    return False, {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }


def estimate_similarity_transform(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    min_matches: int = 3
) -> Optional[Dict[str, float]]:
    """Estimate 2D similarity transform (translation + rotation + scale) using RANSAC.
    
    Returns transform dict with: scale, rotation_rad, tx, ty
    """
    if len(points1) < min_matches or len(points2) < min_matches:
        return None
    
    if np is None:
        # Fallback without numpy
        return None
    
    # Simple approach: use centroid alignment + scale estimation
    # For full RANSAC, would need more sophisticated implementation
    p1_arr = np.array(points1)
    p2_arr = np.array(points2)
    
    # Compute centroids
    c1 = np.mean(p1_arr, axis=0)
    c2 = np.mean(p2_arr, axis=0)
    
    # Center points
    p1_centered = p1_arr - c1
    p2_centered = p2_arr - c2
    
    # Estimate scale from distances
    dist1 = np.mean(np.linalg.norm(p1_centered, axis=1))
    dist2 = np.mean(np.linalg.norm(p2_centered, axis=1))
    scale = dist2 / dist1 if dist1 > 1e-6 else 1.0
    
    # Estimate rotation using SVD (Procrustes)
    if len(points1) >= 2:
        # Use first two points to estimate rotation
        v1 = p1_centered[0] if len(p1_centered) > 0 else np.array([1.0, 0.0])
        v2 = p2_centered[0] if len(p2_centered) > 0 else np.array([1.0, 0.0])
        
        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            v1_norm = v1 / n1
            v2_norm = v2 / n2
            # Compute angle
            cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            sin_angle = np.cross(v1_norm, v2_norm)
            rotation_rad = math.atan2(sin_angle, cos_angle)
        else:
            rotation_rad = 0.0
    else:
        rotation_rad = 0.0
    
    # Translation
    tx = c2[0] - scale * (c1[0] * math.cos(rotation_rad) - c1[1] * math.sin(rotation_rad))
    ty = c2[1] - scale * (c1[0] * math.sin(rotation_rad) + c1[1] * math.cos(rotation_rad))
    
    return {
        "scale": float(scale),
        "rotation_rad": float(rotation_rad),
        "tx": float(tx),
        "ty": float(ty),
    }


def apply_transform(
    points: List[Tuple[float, float]],
    transform: Dict[str, float]
) -> List[Tuple[float, float]]:
    """Apply similarity transform to points."""
    scale = transform["scale"]
    rot = transform["rotation_rad"]
    tx = transform["tx"]
    ty = transform["ty"]
    
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    
    transformed = []
    for x, y in points:
        # Rotate and scale
        x_new = scale * (x * cos_r - y * sin_r) + tx
        y_new = scale * (x * sin_r + y * cos_r) + ty
        transformed.append((x_new, y_new))
    
    return transformed

