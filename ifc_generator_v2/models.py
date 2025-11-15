"""Simple Input Models for IFC Generation

These dataclasses provide a clean, simple interface for creating IFC elements.
They are converted internally to the more complex Profile classes used by IFCV2Builder.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Point2D:
    """2D coordinate point."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)


@dataclass
class Wall:
    """Simple wall definition for IFC generation.
    
    Attributes:
        id: Unique identifier for the wall
        start_point: Starting point of the wall axis
        end_point: Ending point of the wall axis
        height: Wall height in meters
        thickness: Wall thickness in meters
        is_external: Whether the wall is external (default: True)
        name: Wall name (default: "Wall")
    """
    id: str
    start_point: Point2D
    end_point: Point2D
    height: float
    thickness: float
    is_external: bool = True
    name: str = "Wall"

    @property
    def length(self) -> float:
        """Calculate wall length from start to end point."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        return (dx**2 + dy**2)**0.5

    @property
    def axis_line(self) -> List[Tuple[float, float]]:
        """Get wall axis as list of tuples."""
        return [self.start_point.to_tuple(), self.end_point.to_tuple()]

    @property
    def direction_vector(self) -> Tuple[float, float]:
        """Get normalized direction vector of the wall."""
        length = self.length
        if length == 0:
            return (1.0, 0.0)
        dx = (self.end_point.x - self.start_point.x) / length
        dy = (self.end_point.y - self.start_point.y) / length
        return (dx, dy)


@dataclass
class Window:
    """Simple window definition for IFC generation.
    
    Attributes:
        id: Unique identifier for the window
        wall_id: ID of the wall this window belongs to
        position_along_wall: Position along wall axis from start point in meters
        width: Window width in meters
        height: Window height in meters
    """
    id: str
    wall_id: str
    position_along_wall: float
    width: float
    height: float


@dataclass
class Door:
    """Simple door definition for IFC generation.
    
    Attributes:
        id: Unique identifier for the door
        wall_id: ID of the wall this door belongs to
        position_along_wall: Position along wall axis from start point in meters
        width: Door width in meters (default: 0.9)
        height: Door height in meters (default: 2.0)
    """
    id: str
    wall_id: str
    position_along_wall: float
    width: float = 0.9
    height: float = 2.0


@dataclass
class Slab:
    """Simple slab (floor/ceiling) definition for IFC generation.
    
    Attributes:
        id: Unique identifier for the slab
        footprint_points: List of points defining the slab footprint
        thickness: Slab thickness in meters
        slab_type: Type of slab ("FLOOR", "CEILING", "BASESLAB", etc.)
        elevation: Elevation of the slab in meters (default: 0.0)
        name: Slab name (default: "Slab")
    """
    id: str
    footprint_points: List[Point2D]
    thickness: float
    slab_type: str
    elevation: float = 0.0
    name: str = "Slab"

    @property
    def footprint_tuples(self) -> List[Tuple[float, float]]:
        """Get footprint points as list of tuples."""
        return [p.to_tuple() for p in self.footprint_points]


@dataclass
class Space:
    """Simple space (room) definition for IFC generation.
    
    Attributes:
        id: Unique identifier for the space
        footprint_points: List of points defining the space footprint
        height: Space height in meters
        name: Space name (default: "Raum")
    """
    id: str
    footprint_points: List[Point2D]
    height: float
    name: str = "Raum"

    @property
    def footprint_tuples(self) -> List[Tuple[float, float]]:
        """Get footprint points as list of tuples."""
        return [p.to_tuple() for p in self.footprint_points]


def create_rectangular_footprint(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> List[Point2D]:
    """Create a rectangular footprint from bounding box coordinates.
    
    Args:
        min_x: Minimum X coordinate
        min_y: Minimum Y coordinate
        max_x: Maximum X coordinate
        max_y: Maximum Y coordinate
    
    Returns:
        List of Point2D forming a rectangle (counter-clockwise)
    """
    return [
        Point2D(min_x, min_y),
        Point2D(max_x, min_y),
        Point2D(max_x, max_y),
        Point2D(min_x, max_y),
    ]

