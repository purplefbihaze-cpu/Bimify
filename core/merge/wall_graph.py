"""Build wall centerline graph from Model 1 with snapping."""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import math

from .parsers import ParsedWall


@dataclass
class WallNode:
    """Graph node representing a wall junction."""
    id: str
    x: float
    y: float
    connected_walls: Set[str]


@dataclass
class WallEdge:
    """Graph edge representing a wall segment."""
    id: str
    start_node: str
    end_node: str
    points: List[Tuple[float, float]]
    isExternal: bool
    confidence: float


class WallGraph:
    """Wall centerline graph with snapping."""
    
    def __init__(self, snap_tolerance: float = 5.0):
        self.snap_tolerance = snap_tolerance
        self.nodes: Dict[str, WallNode] = {}
        self.edges: Dict[str, WallEdge] = {}
        self.node_counter = 0
    
    def _snap_point(self, x: float, y: float) -> str:
        """Find or create node near point."""
        for node_id, node in self.nodes.items():
            dx = node.x - x
            dy = node.y - y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= self.snap_tolerance:
                return node_id
        
        # Create new node
        node_id = f"n{self.node_counter}"
        self.node_counter += 1
        self.nodes[node_id] = WallNode(
            id=node_id,
            x=x,
            y=y,
            connected_walls=set()
        )
        return node_id
    
    def add_wall(self, wall: ParsedWall) -> Optional[str]:
        """Add wall to graph with snapping."""
        if len(wall.points) < 2:
            return None
        
        # Snap endpoints
        start_pt = wall.points[0]
        end_pt = wall.points[-1]
        
        start_node = self._snap_point(start_pt[0], start_pt[1])
        end_node = self._snap_point(end_pt[0], end_pt[1])
        
        if start_node == end_node:
            # Degenerate case
            return None
        
        edge_id = wall.id
        self.edges[edge_id] = WallEdge(
            id=edge_id,
            start_node=start_node,
            end_node=end_node,
            points=wall.points,
            isExternal=wall.isExternal,
            confidence=wall.confidence
        )
        
        # Update node connections
        self.nodes[start_node].connected_walls.add(edge_id)
        self.nodes[end_node].connected_walls.add(edge_id)
        
        return edge_id
    
    def get_centerline(self, edge_id: str) -> List[Tuple[float, float]]:
        """Get centerline for edge, snapped to nodes."""
        edge = self.edges.get(edge_id)
        if not edge:
            return []
        
        start_node = self.nodes[edge.start_node]
        end_node = self.nodes[edge.end_node]
        
        # Return simplified centerline: start -> end
        return [(start_node.x, start_node.y), (end_node.x, end_node.y)]
    
    def get_connections(self, edge_id: str) -> List[str]:
        """Get IDs of edges connected to this edge."""
        edge = self.edges.get(edge_id)
        if not edge:
            return []
        
        connected = set()
        start_conns = self.nodes[edge.start_node].connected_walls
        end_conns = self.nodes[edge.end_node].connected_walls
        
        for eid in start_conns | end_conns:
            if eid != edge_id:
                connected.add(eid)
        
        return list(connected)

