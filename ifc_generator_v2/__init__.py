"""IFC Generator V2 - Simple Input Models

This module provides simple dataclasses as a clean interface between
PNG parsers and the IFCV2Builder. These models are converted to the
more complex Profile classes internally.
"""

from .models import (
    Point2D,
    Wall,
    Window,
    Door,
    Slab,
    Space,
    create_rectangular_footprint,
)

__all__ = [
    "Point2D",
    "Wall",
    "Window",
    "Door",
    "Slab",
    "Space",
    "create_rectangular_footprint",
]

