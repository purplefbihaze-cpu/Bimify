from __future__ import annotations

"""
Geometry Extraction Contract

Single source of truth for geometric thresholds, tolerances, and defaults used
throughout the pipeline. All modules should import from here instead of hardcoding.
"""

# Lengths in meters unless noted; mm constants end with _MM

# Walls
MIN_WALL_LENGTH = 0.40  # m
FALLBACK_WALL_THICKNESS = 0.20  # m

# Gaps
MAX_GAP_STRICT = 0.03  # m
MAX_GAP_FALLBACK = 0.12  # m
NODE_MERGE_DIST = 0.008  # m (8 mm)

# Angles
SEGMENT_MERGE_ANGLE_DEG = 5.0  # degrees tolerance to consider segments colinear

# Rooms / Spaces
MIN_ROOM_AREA_STRICT = 1.0  # m²
MIN_ROOM_AREA_FALLBACK = 0.4  # m²
MIN_POLYGON_AREA = 0.2  # m² minimal polygon area considered valid
MIN_SPACE_DIMENSION_MM = 200.0  # mm minimal dimension for a space (width/height)

# Slabs
FALLBACK_SLAB_THICKNESS = 0.20  # m

# Viewer/IFC safety
MIN_OPENING_DEPTH = 0.05  # m

# Grid snapping (used by repair)
SNAP_ENDPOINTS_MM = 8.0  # mm
MIN_SEGMENT_LEN_MM = 400.0  # mm
ANGLE_TOLERANCE_DEG = 5.0  # degrees (also used for orth snapping)


def mm(value_m: float) -> float:
    """Convert meters to millimeters."""
    return float(value_m * 1000.0)


def m(value_mm: float) -> float:
    """Convert millimeters to meters."""
    return float(value_mm / 1000.0)
