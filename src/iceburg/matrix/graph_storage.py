"""
Graph Storage - NetworkX-based graph for entity relationships.
Stores entities and relationships, supports queries for connections.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not installed. Install with: pip install networkx")


@dataclass
class Entity:
    """Represents an entity in the matrix."""
    entity_id: str
    name: str
    entity_type: str  # person, organization, company, government
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "properties": self.properties,
            "sources": self.sources,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: str  # owns, funds, employs, connected_to, board_member, contributed_to
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "sources": self.sources,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


class MatrixGraph:
    """
    NetworkX-based graph storage for the Matrix.
    
    Features:
    - Add/update entities and relationships
    - Query by relationship type
    - Find paths between entities
    - Detect clusters and communities
    - Export to various formats
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the graph storage.
        
        Args:
            data_dir: Directory for persisting the graph
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required. Install with: pip install networkx")
        
        self.data_dir = data_dir or Path.home() / "Documents" / "iceburg_matrix"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph_path = self.data_dir / "matrix_graph.pkl"
        self.entities_path = self.data_dir / "entities.json"
        
        # Initialize or load graph
        self.graph: nx.DiGraph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self._load()
        
        logger.info(f"📊 Matrix Graph initialized ({len(self.entities)} entities, {self.graph.number_of_edges()} relationships)")
    
    def _load(self):
        """Load graph from disk."""
        try:
            if self.graph_path.exists():
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)
            
            if self.entities_path.exists():
                with open(self.entities_path, "r") as f:
                    data = json.load(f)
                    for eid, edata in data.items():
                        self.entities[eid] = Entity(
                            entity_id=edata["entity_id"],
                            name=edata["name"],
                            entity_type=edata["entity_type"],
                            aliases=edata.get("aliases", []),
                            properties=edata.get("properties", {}),
                            sources=edata.get("sources", []),
                            created_at=datetime.fromisoformat(edata.get("created_at", datetime.now().isoformat())),
                            updated_at=datetime.fromisoformat(edata.get("updated_at", datetime.now().isoformat())),
                        )
        except Exception as e:
            logger.warning(f"Could not load graph: {e}")
    
    def _save(self):
        """Persist graph to disk."""
        try:
            with open(self.graph_path, "wb") as f:
                pickle.dump(self.graph, f)
            
            with open(self.entities_path, "w") as f:
                json.dump({eid: e.to_dict() for eid, e in self.entities.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save graph: {e}")
    
    def add_entity(self, entity: Entity, merge: bool = True) -> Entity:
        """
        Add or update an entity.
        
        Args:
            entity: Entity to add
            merge: If True, merge with existing entity
            
        Returns:
            The added/updated entity
        """
        if entity.entity_id in self.entities and merge:
            existing = self.entities[entity.entity_id]
            # Merge aliases
            existing.aliases = list(set(existing.aliases + entity.aliases))
            # Merge sources
            existing.sources = list(set(existing.sources + entity.sources))
            # Update properties (new values override)
            existing.properties.update(entity.properties)
            existing.updated_at = datetime.now()
            entity = existing
        
        self.entities[entity.entity_id] = entity
        
        # Add to graph
        self.graph.add_node(
            entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            **entity.properties
        )
        
        return entity
    
    def add_relationship(self, relationship: Relationship) -> Relationship:
        """
        Add a relationship between entities.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            The added relationship
        """
        # Ensure both entities exist
        if relationship.source_id not in self.entities:
            logger.warning(f"Source entity not found: {relationship.source_id}")
        if relationship.target_id not in self.entities:
            logger.warning(f"Target entity not found: {relationship.target_id}")
        
        # Add edge to graph
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            relationship_id=relationship.relationship_id,
            relationship_type=relationship.relationship_type,
            confidence=relationship.confidence,
            **relationship.properties
        )
        
        return relationship
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Entity]:
        """
        Search entities by name.
        
        Args:
            query: Search query
            entity_type: Optional filter by type
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            # Check name and aliases
            if query_lower in entity.name.lower():
                results.append(entity)
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                results.append(entity)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity to get relationships for
            relationship_type: Optional filter by type
            direction: Direction of relationships
            
        Returns:
            List of relationship data
        """
        relationships = []
        
        # Outgoing edges
        if direction in ("outgoing", "both"):
            for _, target, data in self.graph.out_edges(entity_id, data=True):
                if relationship_type and data.get("relationship_type") != relationship_type:
                    continue
                relationships.append({
                    "source_id": entity_id,
                    "target_id": target,
                    "target_name": self.entities.get(target, Entity(target, target, "unknown")).name,
                    "direction": "outgoing",
                    **data
                })
        
        # Incoming edges
        if direction in ("incoming", "both"):
            for source, _, data in self.graph.in_edges(entity_id, data=True):
                if relationship_type and data.get("relationship_type") != relationship_type:
                    continue
                relationships.append({
                    "source_id": source,
                    "source_name": self.entities.get(source, Entity(source, source, "unknown")).name,
                    "target_id": entity_id,
                    "direction": "incoming",
                    **data
                })
        
        return relationships
    
    def get_network(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get the network around an entity.
        
        Args:
            entity_id: Center entity
            depth: How many hops to traverse
            limit: Maximum nodes
            
        Returns:
            Network data with nodes and edges
        """
        if entity_id not in self.graph:
            return {"nodes": [], "edges": []}
        
        # BFS to find connected nodes
        visited: Set[str] = {entity_id}
        current_level = {entity_id}
        all_edges = []
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Outgoing
                for _, target, data in self.graph.out_edges(node, data=True):
                    if len(visited) >= limit:
                        break
                    all_edges.append({"source": node, "target": target, **data})
                    if target not in visited:
                        visited.add(target)
                        next_level.add(target)
                
                # Incoming
                for source, _, data in self.graph.in_edges(node, data=True):
                    if len(visited) >= limit:
                        break
                    all_edges.append({"source": source, "target": node, **data})
                    if source not in visited:
                        visited.add(source)
                        next_level.add(source)
            
            current_level = next_level
        
        # Build node list
        nodes = []
        for node_id in visited:
            entity = self.entities.get(node_id)
            if entity:
                nodes.append({
                    "id": node_id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "is_center": node_id == entity_id,
                })
            else:
                nodes.append({
                    "id": node_id,
                    "name": node_id,
                    "type": "unknown",
                    "is_center": node_id == entity_id,
                })
        
        return {"nodes": nodes, "edges": all_edges}
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 6
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        Args:
            source_id: Starting entity
            target_id: Destination entity
            max_length: Maximum path length
            
        Returns:
            List of entity IDs in path, or None if no path
        """
        try:
            # Use undirected view for path finding
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            if len(path) <= max_length:
                return path
            return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_types = {}
        for entity in self.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        relationship_types = {}
        for _, _, data in self.graph.edges(data=True):
            rtype = data.get("relationship_type", "unknown")
            relationship_types[rtype] = relationship_types.get(rtype, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }
    
    def save(self):
        """Persist graph to disk."""
        self._save()
        logger.info(f"💾 Graph saved ({len(self.entities)} entities, {self.graph.number_of_edges()} relationships)")
    
    def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Export graph to JSON format."""
        output_path = output_path or self.data_dir / "matrix_export.json"
        
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [
                {
                    "source_id": source,
                    "target_id": target,
                    **edge_data
                }
                for source, target, edge_data in self.graph.edges(data=True)
            ],
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_stats(),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📤 Exported graph to {output_path}")
        return output_path
