"""
Network Graph Builder - Builds relationship graphs for visualization.
Creates D3.js-compatible network graphs from entities and relationships.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the network graph."""
    id: str
    label: str
    node_type: str  # 'person', 'organization', 'location', 'event', 'concept'
    size: float = 1.0
    color: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_d3(self) -> Dict[str, Any]:
        """Convert to D3.js node format."""
        return {
            "id": self.id,
            "name": self.label,
            "group": self._type_to_group(),
            "type": self.node_type,
            "size": self.size,
            "color": self.color or self._default_color(),
            **self.metadata
        }
    
    def _type_to_group(self) -> int:
        """Map type to D3 group number."""
        mapping = {
            "person": 1,
            "organization": 2,
            "location": 3,
            "event": 4,
            "concept": 5,
            "money": 6,
            "date": 7
        }
        return mapping.get(self.node_type, 0)
    
    def _default_color(self) -> str:
        """Default color by type."""
        mapping = {
            "person": "#ff6b6b",
            "organization": "#4ecdc4",
            "location": "#ffe66d",
            "event": "#95e1d3",
            "concept": "#c084fc",
            "money": "#22c55e",
            "date": "#f97316"
        }
        return mapping.get(self.node_type, "#888888")


@dataclass
class GraphEdge:
    """An edge in the network graph."""
    source: str
    target: str
    edge_type: str  # 'member_of', 'funds', 'owns', 'connected_to', 'works_for', etc.
    weight: float = 1.0
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_d3(self) -> Dict[str, Any]:
        """Convert to D3.js link format."""
        return {
            "source": self.source,
            "target": self.target,
            "value": self.weight,
            "label": self.label or self.edge_type,
            "type": self.edge_type,
            **self.metadata
        }


class NetworkGraphBuilder:
    """
    Builds network graphs for visualization.
    
    Features:
    - D3.js-compatible output
    - Force-directed layout parameters
    - Clustering support
    - Export to multiple formats
    """
    
    def __init__(self):
        """Initialize graph builder."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
    
    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        size: float = 1.0,
        color: Optional[str] = None,
        **metadata
    ) -> GraphNode:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier
            label: Display label
            node_type: Type of node
            size: Node size (for visualization)
            color: Optional color override
            **metadata: Additional metadata
            
        Returns:
            Created GraphNode
        """
        node = GraphNode(
            id=node_id,
            label=label,
            node_type=node_type,
            size=size,
            color=color,
            metadata=metadata
        )
        self.nodes[node_id] = node
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "connected_to",
        weight: float = 1.0,
        label: Optional[str] = None,
        **metadata
    ) -> Optional[GraphEdge]:
        """
        Add an edge to the graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            weight: Edge weight (for visualization)
            label: Optional display label
            **metadata: Additional metadata
            
        Returns:
            Created GraphEdge or None if nodes don't exist
        """
        # Verify nodes exist
        if source not in self.nodes or target not in self.nodes:
            logger.warning(f"Cannot create edge: node not found ({source} -> {target})")
            return None
        
        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            label=label,
            metadata=metadata
        )
        self.edges.append(edge)
        return edge
    
    def add_entities(self, entities: List[Dict[str, Any]]) -> int:
        """
        Add entities as nodes.
        
        Args:
            entities: List of entity dictionaries with 'name' and 'type'
            
        Returns:
            Number of nodes added
        """
        added = 0
        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue
            
            node_id = self._make_id(name)
            if node_id not in self.nodes:
                self.add_node(
                    node_id=node_id,
                    label=name,
                    node_type=entity.get("type", "unknown"),
                    **{k: v for k, v in entity.items() if k not in ["name", "type"]}
                )
                added += 1
        
        return added
    
    def add_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """
        Add relationships as edges.
        
        Args:
            relationships: List of relationship dicts with 'source', 'target', 'type'
            
        Returns:
            Number of edges added
        """
        added = 0
        for rel in relationships:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            
            source_id = self._make_id(source_name)
            target_id = self._make_id(target_name)
            
            # Ensure nodes exist
            if source_id not in self.nodes and source_name:
                self.add_node(source_id, source_name, "unknown")
            if target_id not in self.nodes and target_name:
                self.add_node(target_id, target_name, "unknown")
            
            edge = self.add_edge(
                source=source_id,
                target=target_id,
                edge_type=rel.get("type", "connected_to"),
                weight=rel.get("weight", 1.0),
                label=rel.get("description", None)
            )
            if edge:
                added += 1
        
        return added
    
    def _make_id(self, name: str) -> str:
        """Create a clean ID from a name."""
        return name.lower().replace(" ", "_").replace("'", "").replace(".", "")[:50]
    
    def to_d3(self) -> Dict[str, Any]:
        """
        Export graph in D3.js force-directed format.
        
        Returns:
            Dictionary with 'nodes' and 'links' arrays
        """
        return {
            "nodes": [node.to_d3() for node in self.nodes.values()],
            "links": [edge.to_d3() for edge in self.edges]
        }
    
    def to_cytoscape(self) -> Dict[str, Any]:
        """
        Export graph in Cytoscape.js format.
        
        Returns:
            Dictionary with 'elements' containing nodes and edges
        """
        elements = []
        
        for node in self.nodes.values():
            elements.append({
                "data": {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type,
                    **node.metadata
                },
                "group": "nodes"
            })
        
        for edge in self.edges:
            elements.append({
                "data": {
                    "id": f"{edge.source}_{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label or edge.edge_type,
                    "weight": edge.weight
                },
                "group": "edges"
            })
        
        return {"elements": elements}
    
    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_d3(), indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        # Count node types
        node_types = {}
        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        # Count edge types
        edge_types = {}
        for edge in self.edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
        
        # Calculate centrality (simple degree centrality)
        degree = {}
        for edge in self.edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        
        most_connected = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "most_connected": [
                {"id": node_id, "label": self.nodes[node_id].label, "connections": count}
                for node_id, count in most_connected
                if node_id in self.nodes
            ],
            "density": (2 * len(self.edges)) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0
        }
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 4) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            max_depth: Maximum path length
            
        Returns:
            List of node IDs in path, or None if no path found
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        # Build adjacency list
        adj = {}
        for edge in self.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            if edge.target not in adj:
                adj[edge.target] = []
            adj[edge.source].append(edge.target)
            adj[edge.target].append(edge.source)  # Undirected
        
        # BFS
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == target_id:
                return path
            
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_clusters(self) -> List[List[str]]:
        """
        Find connected components (clusters) in the graph.
        
        Returns:
            List of clusters, each cluster is a list of node IDs
        """
        # Build adjacency list
        adj = {}
        for edge in self.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            if edge.target not in adj:
                adj[edge.target] = []
            adj[edge.source].append(edge.target)
            adj[edge.target].append(edge.source)
        
        # Add isolated nodes
        for node_id in self.nodes:
            if node_id not in adj:
                adj[node_id] = []
        
        # Find connected components
        visited = set()
        clusters = []
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            # BFS from this node
            cluster = []
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                queue.extend([n for n in adj.get(current, []) if n not in visited])
            
            clusters.append(cluster)
        
        return clusters
    
    def clear(self):
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()


def build_network_graph(entities: List[Dict], relationships: List[Dict] = None) -> NetworkGraphBuilder:
    """
    Convenience function to build a network graph.
    
    Args:
        entities: List of entity dictionaries
        relationships: Optional list of relationship dictionaries
        
    Returns:
        Populated NetworkGraphBuilder
    """
    builder = NetworkGraphBuilder()
    builder.add_entities(entities)
    
    if relationships:
        builder.add_relationships(relationships)
    
    return builder
