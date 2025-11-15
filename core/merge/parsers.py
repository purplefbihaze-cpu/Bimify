"""Parsers for Model 1, 2, and 3 JSON outputs."""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ParsedWall:
    """Parsed wall from any model."""
    id: str
    points: List[Tuple[float, float]]  # In pixels
    isExternal: bool
    confidence: float
    source: str  # "model1", "model3"


@dataclass
class ParsedOpening:
    """Parsed door/window from any model."""
    id: str
    type: str  # "door" or "window"
    bbox: Tuple[float, float, float, float]  # x, y, width, height in pixels
    confidence: float
    source: str


@dataclass
class ParsedRoom:
    """Parsed room from Model 2."""
    id: str
    polygon: List[Tuple[float, float]]  # In pixels
    confidence: float


def parse_model1(json_path: Path) -> Tuple[List[ParsedWall], List[ParsedOpening], Dict[str, Any]]:
    """Parse Model 1 JSON (geometry/topology with points)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    walls: List[ParsedWall] = []
    openings: List[ParsedOpening] = []
    
    for pred in data.get("predictions", []):
        pred_id = pred.get("detection_id", f"m1_{len(walls) + len(openings)}")
        cls = pred.get("class", "").lower()
        conf = pred.get("confidence", 0.0)
        
        points = pred.get("points", [])
        if not points:
            continue
        
        # Convert points to list of tuples
        point_tuples = [(p["x"], p["y"]) for p in points if "x" in p and "y" in p]
        if len(point_tuples) < 2:
            continue
        
        if cls in ("internal-walls", "external-walls"):
            is_ext = cls == "external-walls"
            walls.append(ParsedWall(
                id=pred_id,
                points=point_tuples,
                isExternal=is_ext,
                confidence=conf,
                source="model1"
            ))
        elif cls in ("door", "window"):
            # For Model 1, compute bbox from points
            xs = [p[0] for p in point_tuples]
            ys = [p[1] for p in point_tuples]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            openings.append(ParsedOpening(
                id=pred_id,
                type=cls,
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                confidence=conf,
                source="model1"
            ))
    
    metadata = {
        "width": data.get("image", {}).get("width", 0),
        "height": data.get("image", {}).get("height", 0),
    }
    
    return walls, openings, metadata


def parse_model2(json_path: Path) -> Tuple[List[ParsedRoom], Dict[str, Any]]:
    """Parse Model 2 JSON (rooms)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rooms: List[ParsedRoom] = []
    
    for pred in data.get("predictions", []):
        cls = pred.get("class", "").lower()
        if cls != "room":
            continue
        
        pred_id = pred.get("detection_id", f"m2_room_{len(rooms)}")
        conf = pred.get("confidence", 0.0)
        
        points = pred.get("points", [])
        if not points:
            continue
        
        point_tuples = [(p["x"], p["y"]) for p in points if "x" in p and "y" in p]
        if len(point_tuples) < 3:  # Need at least triangle
            continue
        
        rooms.append(ParsedRoom(
            id=pred_id,
            polygon=point_tuples,
            confidence=conf
        ))
    
    metadata = {
        "width": data.get("image", {}).get("width", 0),
        "height": data.get("image", {}).get("height", 0),
    }
    
    return rooms, metadata


def parse_model3(json_path: Path) -> Tuple[List[ParsedWall], List[ParsedOpening], Dict[str, Any]]:
    """Parse Model 3 JSON (clean geometry with bboxes)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    walls: List[ParsedWall] = []
    openings: List[ParsedOpening] = []
    
    for pred in data.get("predictions", []):
        pred_id = pred.get("detection_id", f"m3_{len(walls) + len(openings)}")
        cls = pred.get("class", "").lower()
        conf = pred.get("confidence", 0.0)
        
        x = pred.get("x", 0.0)
        y = pred.get("y", 0.0)
        w = pred.get("width", 0.0)
        h = pred.get("height", 0.0)
        
        if w <= 0 or h <= 0:
            continue
        
        if cls == "wall":
            # Convert bbox to centerline approximation
            # For thin walls (h < w), it's vertical; else horizontal
            if h < w:
                # Vertical wall
                cx = x + w / 2
                points = [(cx, y), (cx, y + h)]
            else:
                # Horizontal wall
                cy = y + h / 2
                points = [(x, cy), (x + w, cy)]
            
            walls.append(ParsedWall(
                id=pred_id,
                points=points,
                isExternal=False,  # Model 3 doesn't distinguish, will infer later
                confidence=conf,
                source="model3"
            ))
        elif cls in ("door", "window"):
            openings.append(ParsedOpening(
                id=pred_id,
                type=cls,
                bbox=(x, y, w, h),
                confidence=conf,
                source="model3"
            ))
    
    metadata = {
        "width": data.get("image", {}).get("width", 0),
        "height": data.get("image", {}).get("height", 0),
    }
    
    return walls, openings, metadata

