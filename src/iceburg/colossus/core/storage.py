"""
COLOSSUS Storage Layer

Unified storage abstraction for polyglot persistence.
Coordinates between graph, search, and vector stores.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .graph import ColossusGraph, GraphEntity, GraphRelationship
from .search import ColossusSearch

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Storage configuration."""
    # Graph database
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    
    # Search
    elasticsearch_url: Optional[str] = None
    
    # Vector store
    milvus_host: Optional[str] = None
    milvus_port: int = 19530
    
    # Local storage
    data_dir: Path = Path.home() / "Documents" / "colossus_data"
    sqlite_db: str = "colossus.db"
    
    # Performance
    use_memory_graph: bool = True
    cache_size_mb: int = 1024


class ColossusStorage:
    """
    Unified storage layer for COLOSSUS.
    
    Coordinates:
    - Graph database (Neo4j / NetworkX)
    - Full-text search (Elasticsearch / SQLite FTS5)
    - Vector embeddings (Milvus / Chroma)
    - Object storage (local filesystem / S3)
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize storage with config."""
        self.config = config or StorageConfig()
        
        # Ensure data directory exists
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        self._graph: Optional[ColossusGraph] = None
        self._search: Optional[ColossusSearch] = None
        self._vectors = None  # Milvus/Chroma client
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all storage backends."""
        logger.info(f"🚀 Initializing COLOSSUS storage: {self.config.data_dir}")
        
        # Initialize graph
        self._graph = ColossusGraph(
            neo4j_uri=self.config.neo4j_uri,
            neo4j_user=self.config.neo4j_user,
            neo4j_password=self.config.neo4j_password,
            use_memory=self.config.use_memory_graph,
        )
        
        # Initialize search
        sqlite_path = self.config.data_dir / self.config.sqlite_db
        self._search = ColossusSearch(
            elasticsearch_url=self.config.elasticsearch_url,
            sqlite_path=str(sqlite_path),
        )
        
        # TODO: Initialize vector store
        
        logger.info("✅ COLOSSUS storage initialized")
    
    @property
    def graph(self) -> ColossusGraph:
        """Get graph database."""
        return self._graph
    
    @property
    def search(self) -> ColossusSearch:
        """Get search engine."""
        return self._search
    
    # ==================== Entity Operations ====================
    
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        countries: List[str] = None,
        sanctions: List[str] = None,
        properties: Dict[str, Any] = None,
        sources: List[str] = None,
    ) -> str:
        """
        Add entity to all storage backends.
        
        Returns:
            Entity ID
        """
        countries = countries or []
        sanctions = sanctions or []
        properties = properties or {}
        sources = sources or []
        
        # Add to graph
        graph_entity = GraphEntity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            countries=countries,
            sanctions=sanctions,
            properties=properties,
            sources=sources,
        )
        self._graph.add_entity(graph_entity)
        
        # Add to search
        self._search.index_entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            countries=countries,
            sanctions=sanctions,
            properties=properties,
        )
        
        # TODO: Add to vector store
        
        return entity_id
    
    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity from graph."""
        return self._graph.get_entity(entity_id)
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search entities using full-text search."""
        results = self._search.search(query, limit=limit)
        return [
            {
                "entity_id": r.entity_id,
                "name": r.name,
                "entity_type": r.entity_type,
                "countries": r.countries,
                "sanctions_count": r.sanctions_count,
                "score": r.score,
            }
            for r in results
        ]
    
    # ==================== Relationship Operations ====================
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
        confidence: float = 1.0,
        sources: List[str] = None,
    ) -> str:
        """Add relationship to graph."""
        import uuid
        
        rel = GraphRelationship(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {},
            confidence=confidence,
            sources=sources or [],
        )
        
        return self._graph.add_relationship(rel)
    
    def get_entity_network(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get entity network from graph."""
        return self._graph.get_network(entity_id, depth, limit)
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 6
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between entities."""
        path = self._graph.find_path(source_id, target_id, max_hops)
        if not path:
            return None
        
        return [
            {
                "entity": {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                },
                "relationship": {
                    "type": rel.relationship_type if rel else None,
                } if rel else None,
            }
            for entity, rel in path
        ]
    
    # ==================== Bulk Operations ====================
    
    def bulk_import(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]] = None,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """Bulk import entities and relationships."""
        relationships = relationships or []
        
        # Import to graph
        graph_entities = [
            GraphEntity(
                id=e["entity_id"],
                name=e["name"],
                entity_type=e.get("entity_type", "unknown"),
                countries=e.get("countries", []),
                sanctions=e.get("sanctions", []),
                properties=e.get("properties", {}),
                sources=e.get("sources", []),
            )
            for e in entities
        ]
        
        graph_rels = [
            GraphRelationship(
                id=r.get("id", str(hash(f"{r['source']}{r['target']}"))),
                source_id=r["source"],
                target_id=r["target"],
                relationship_type=r.get("type", "RELATED_TO"),
                properties=r.get("properties", {}),
                confidence=r.get("confidence", 1.0),
                sources=r.get("sources", []),
            )
            for r in relationships
        ]
        
        graph_stats = self._graph.bulk_import(graph_entities, graph_rels, batch_size)
        
        # Import to search
        search_count = self._search.bulk_index(entities, batch_size)
        
        return {
            "entities_graph": graph_stats.get("entities_imported", 0),
            "relationships_graph": graph_stats.get("relationships_imported", 0),
            "entities_search": search_count,
        }
    
    def migrate_from_matrix(self) -> Dict[str, int]:
        """Migrate entities from Matrix SQLite database."""
        matrix_db = Path.home() / "Documents" / "iceburg_matrix" / "matrix.db"
        
        if not matrix_db.exists():
            return {"error": "Matrix database not found", "path": str(matrix_db)}
        
        return self._graph.migrate_from_matrix(matrix_db)
    
    # ==================== Stats ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        graph_stats = self._graph.get_stats()
        
        return {
            "graph": graph_stats,
            "storage_path": str(self.config.data_dir),
            "backends": {
                "graph": "neo4j" if self._graph.is_neo4j else "networkx",
                "search": "elasticsearch" if self._search.is_elasticsearch else "sqlite_fts5",
                "vectors": "pending",
            }
        }
