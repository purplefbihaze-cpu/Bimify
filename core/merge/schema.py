"""Canonical JSON schema for merged plan data."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Point2D(BaseModel):
    """2D point in meters."""
    x: float
    y: float


class Wall(BaseModel):
    """Wall element with centerline polyline."""
    id: str
    polyline: List[Point2D] = Field(..., description="Centerline points in order")
    thickness: float = Field(..., description="Wall thickness in meters")
    isExternal: bool = Field(..., description="True if external wall")
    connections: List[str] = Field(default_factory=list, description="Connected wall IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Source confidence score")


class Opening(BaseModel):
    """Door or window opening."""
    id: str
    type: Literal["door", "window"]
    hostWallId: str = Field(..., description="ID of wall containing this opening")
    s: float = Field(..., ge=0.0, le=1.0, description="Position along wall axis [0,1]")
    width: float = Field(..., description="Opening width in meters")
    height: float = Field(..., description="Opening height in meters")
    confidence: float = Field(..., ge=0.0, le=1.0)
    swingDirection: Optional[str] = Field(None, description="Door swing direction if applicable")


class Room(BaseModel):
    """Room/space polygon."""
    id: str
    polygon: List[Point2D] = Field(..., description="Closed polygon vertices")
    area: float = Field(..., description="Area in square meters")
    boundaryWallIds: List[str] = Field(default_factory=list, description="Walls forming boundary")
    confidence: float = Field(..., ge=0.0, le=1.0)
    name: Optional[str] = None


class Metadata(BaseModel):
    """Plan metadata."""
    units: Literal["m"] = "m"
    scale: float = Field(..., description="Pixel to meter scale factor")
    imageWidth: int
    imageHeight: int
    sourcePlan: Optional[str] = None


class CanonicalPlan(BaseModel):
    """Canonical merged plan JSON."""
    metadata: Metadata
    walls: List[Wall] = Field(default_factory=list)
    openings: List[Opening] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)

