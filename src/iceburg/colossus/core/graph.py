"""
COLOSSUS Graph Database Layer

Neo4j integration for entity relationship storage and traversal.
Supports:
- Multi-hop relationship queries
- Path finding between entities
- Network centrality analysis
- Temporal relationship tracking
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """Entity node in the graph."""
    id: str
    name: str
    entity_type: str  # person, company, organization, address
    properties: Dict[str, Any] = field(default_factory=dict)
    countries: List[str] = field(default_factory=list)
    sanctions: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRelationship:
    """Relationship edge in the graph."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str  # OWNS, DIRECTOR_OF, FAMILY_OF, SANCTIONED_BY, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    sources: List[str] = field(default_factory=list)


class ColossusGraph:
    """
    Knowledge graph for COLOSSUS intelligence platform.
    
    Uses Neo4j for production, falls back to NetworkX for development.
    Optimized for M4 Mac unified memory architecture.
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_memory: bool = True,
    ):
        """
        Initialize graph database connection.
        
        Args:
            neo4j_uri: Neo4j connection URI (bolt://localhost:7687)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            use_memory: Use in-memory graph (NetworkX) if Neo4j unavailable
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.use_memory = use_memory
        
        self._driver = None
        self._memory_graph = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize graph backend."""
        # Try Neo4j first
        if self.neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                self._driver.verify_connectivity()
                logger.info(f"✅ Connected to Neo4j: {self.neo4j_uri}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Neo4j unavailable: {e}")
        
        # Fall back to in-memory NetworkX graph
        if self.use_memory:
            try:
                import networkx as nx
                self._memory_graph = nx.MultiDiGraph()
                logger.info("📊 Using in-memory NetworkX graph")
            except ImportError:
                logger.error("❌ NetworkX not installed")
                raise
    
    @property
    def is_neo4j(self) -> bool:
        """Check if using Neo4j backend."""
        return self._driver is not None
    
    # ==================== Entity Operations ====================
    
    def add_entity(self, entity: GraphEntity) -> str:
        """Add or update an entity in the graph."""
        if self.is_neo4j:
            return self._neo4j_add_entity(entity)
        else:
            return self._memory_add_entity(entity)
    
    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity by ID (with lazy loading from Matrix)."""
        entity = None
        if self.is_neo4j:
            entity = self._neo4j_get_entity(entity_id)
        else:
            entity = self._memory_get_entity(entity_id)
            
        # Lazy Load: If not in graph, check MatrixStore and import it
        if not entity:
            from ..matrix_store import MatrixStore
            store = MatrixStore()
            entity = store.get_entity(entity_id)
            if entity:
                logger.info(f"📥 Lazy Loaded entity: {entity.name} ({entity.id})")
                self.add_entity(entity)
                
                # Extract and add relationships
                # Extract and add relationships
                try:
                    relationships = store.get_relationships(entity_id)
                    logger.info(f"🔗 Loaded {len(relationships)} relationships for {entity.name}")
                    
                    neighbor_ids = set()
                    
                    for rel in relationships:
                        self.add_relationship(rel)
                        
                        # Identify neighbor
                        if rel.source_id == entity.id:
                            neighbor_ids.add(rel.target_id)
                        else:
                            neighbor_ids.add(rel.source_id)
                            
                    # Lazy load neighbors (node data) so the graph is valid
                    # Only load if not already in memory
                    for nid in neighbor_ids:
                        if not self._memory_get_entity(nid):
                            neighbor = store.get_entity(nid)
                            if neighbor:
                                self.add_entity(neighbor)
                except Exception as e:
                    logger.error(f"Error loading relationships for {entity.name}: {e}")
                
        return entity
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[GraphEntity]:
        """Search entities by name."""
        if self.is_neo4j:
            return self._neo4j_search_entities(query, entity_type, limit)
        else:
            return self._memory_search_entities(query, entity_type, limit)
    
    # ==================== Relationship Operations ====================
    
    def add_relationship(self, relationship: GraphRelationship) -> str:
        """Add a relationship between entities."""
        if self.is_neo4j:
            return self._neo4j_add_relationship(relationship)
        else:
            return self._memory_add_relationship(relationship)
    
    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[GraphRelationship]:
        """Get relationships for an entity."""
        if self.is_neo4j:
            return self._neo4j_get_relationships(entity_id, relationship_type, direction)
        else:
            return self._memory_get_relationships(entity_id, relationship_type, direction)
    
    # ==================== Graph Traversal ====================
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 6
    ) -> Optional[List[Tuple[GraphEntity, GraphRelationship]]]:
        """Find shortest path between two entities."""
        if self.is_neo4j:
            return self._neo4j_find_path(source_id, target_id, max_hops)
        else:
            return self._memory_find_path(source_id, target_id, max_hops)
    
    def get_network(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get network around an entity up to N hops."""
        if self.is_neo4j:
            return self._neo4j_get_network(entity_id, depth, limit)
        else:
            return self._memory_get_network(entity_id, depth, limit)
    
    def find_connections(
        self,
        entity_ids: List[str],
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """Find connections between multiple entities."""
        if self.is_neo4j:
            return self._neo4j_find_connections(entity_ids, max_hops)
        else:
            return self._memory_find_connections(entity_ids, max_hops)
    
    # ==================== Analytics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.is_neo4j:
            return self._neo4j_get_stats()
        else:
            return self._memory_get_stats()
    
    def get_central_entities(
        self,
        entity_type: Optional[str] = None,
        centrality_measure: str = "degree",  # "degree", "betweenness", "pagerank"
        limit: int = 20
    ) -> List[Tuple[GraphEntity, float]]:
        """Get most central entities by network position."""
        if self.is_neo4j:
            return self._neo4j_get_central_entities(entity_type, centrality_measure, limit)
        else:
            return self._memory_get_central_entities(entity_type, centrality_measure, limit)
    
    # ==================== Bulk Operations ====================
    
    def bulk_import(
        self,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """Bulk import entities and relationships."""
        if self.is_neo4j:
            return self._neo4j_bulk_import(entities, relationships, batch_size)
        else:
            return self._memory_bulk_import(entities, relationships, batch_size)
    
    def migrate_from_matrix(self, matrix_db_path: Path) -> Dict[str, int]:
        """Migrate data from Matrix SQLite to graph."""
        from ..matrix.batch_importer import BatchImporter
        
        logger.info(f"🔄 Migrating from Matrix: {matrix_db_path}")
        
        importer = BatchImporter(matrix_db_path.parent)
        stats = importer.get_stats()
        
        total_entities = stats.get("total_entities", 0)
        logger.info(f"📊 Found {total_entities:,} entities to migrate")
        
        return {
            "entities_migrated": 0,
            "relationships_created": 0,
            "status": "pending_implementation"
        }

    def ingest_dossier(self, dossier_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Ingest an IcebergDossier into the graph.
        Extracts:
        - Main topic entity
        - Key players (entities)
        - Relationships found in network map
        - Hidden connections
        """
        entities_count = 0
        rels_count = 0
        
        # 1. Create/Update Main Topic Entity
        topic = dossier_data.get("query", "Unknown Topic")
        topic_id = f"topic_{topic.lower().replace(' ', '_')}"
        
        main_entity = GraphEntity(
            id=topic_id,
            name=topic,
            entity_type="investigation_topic",
            properties={
                "source": "iceburg_dossier",
                "executive_summary": dossier_data.get("executive_summary", ""),
                "official_narrative": dossier_data.get("official_narrative", ""),
                "confidence": str(dossier_data.get("confidence_ratings", {}))
            },
            sources=["iceburg_dossier"]
        )
        self.add_entity(main_entity)
        entities_count += 1
        
        # 2. Ingest Key Players
        player_map = {}  # name -> id
        
        for player in dossier_data.get("key_players", []):
            name = player.get("name")
            if not name:
                continue
                
            p_id = f"entity_{name.lower().replace(' ', '_')}"
            player_map[name] = p_id
            
            p_entity = GraphEntity(
                id=p_id,
                name=name,
                entity_type=player.get("type", "unknown"),
                properties={
                    "description": player.get("description", ""),
                    "role": player.get("role", "")
                },
                sources=["iceburg_dossier"]
            )
            self.add_entity(p_entity)
            entities_count += 1
            
            # Link to topic
            self.add_relationship(GraphRelationship(
                id=f"{p_id}_related_to_{topic_id}",
                source_id=p_id,
                target_id=topic_id,
                relationship_type="INVOLVED_IN",
                confidence=0.9,
                sources=["iceburg_dossier"]
            ))
            rels_count += 1
            
        # 3. Ingest Hidden Connections
        for conn in dossier_data.get("hidden_connections", []):
            e1_name = conn.get("entity_1")
            e2_name = conn.get("entity_2")
            via = conn.get("connected_via", "unknown connection")
            
            if e1_name and e2_name:
                # Ensure entities exist (simple check)
                id1 = player_map.get(e1_name) or f"entity_{e1_name.lower().replace(' ', '_')}"
                id2 = player_map.get(e2_name) or f"entity_{e2_name.lower().replace(' ', '_')}"
                
                # Create if not mapped (lightweight creation)
                if e1_name not in player_map:
                    self.add_entity(GraphEntity(id=id1, name=e1_name, entity_type="unknown", sources=["iceburg_dossier"]))
                    entities_count += 1
                if e2_name not in player_map:
                    self.add_entity(GraphEntity(id=id2, name=e2_name, entity_type="unknown", sources=["iceburg_dossier"]))
                    entities_count += 1
                
                # Add relationship
                self.add_relationship(GraphRelationship(
                    id=f"{id1}_{id2}_hidden",
                    source_id=id1,
                    target_id=id2,
                    relationship_type="HIDDEN_CONNECTION",
                    properties={"via": via},
                    confidence=0.8,
                    sources=["iceburg_dossier"]
                ))
                rels_count += 1

        logger.info(f"📥 Ingested Dossier '{topic}': {entities_count} entities, {rels_count} relationships")
        return {
            "entities_ingested": entities_count,
            "relationships_ingested": rels_count
        }
    
    # ==================== In-Memory Implementations ====================
    
    def _memory_add_entity(self, entity: GraphEntity) -> str:
        """Add entity to in-memory graph."""
        self._memory_graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties,
            countries=entity.countries,
            sanctions=entity.sanctions,
            sources=entity.sources,
            created_at=entity.created_at.isoformat(),
            updated_at=entity.updated_at.isoformat(),
        )
        return entity.id
    
    def _memory_get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity from in-memory graph."""
        if entity_id not in self._memory_graph:
            return None
        
        data = self._memory_graph.nodes[entity_id]
        return GraphEntity(
            id=entity_id,
            name=data.get("name", ""),
            entity_type=data.get("entity_type", "unknown"),
            properties=data.get("properties", {}),
            countries=data.get("countries", []),
            sanctions=data.get("sanctions", []),
            sources=data.get("sources", []),
        )
    
    def _memory_search_entities(
        self,
        query: str,
        entity_type: Optional[str],
        limit: int
    ) -> List[GraphEntity]:
        """Search entities in in-memory graph."""
        results = []
        query_lower = query.lower()
        
        for node_id, data in self._memory_graph.nodes(data=True):
            name = data.get("name", "").lower()
            if query_lower in name:
                if entity_type and data.get("entity_type") != entity_type:
                    continue
                results.append(GraphEntity(
                    id=node_id,
                    name=data.get("name", ""),
                    entity_type=data.get("entity_type", "unknown"),
                    properties=data.get("properties", {}),
                    countries=data.get("countries", []),
                    sanctions=data.get("sanctions", []),
                    sources=data.get("sources", []),
                ))
                if len(results) >= limit:
                    break
        
        return results
    
    def _memory_add_relationship(self, rel: GraphRelationship) -> str:
        """Add relationship to in-memory graph."""
        self._memory_graph.add_edge(
            rel.source_id,
            rel.target_id,
            key=rel.id,
            relationship_type=rel.relationship_type,
            properties=rel.properties,
            confidence=rel.confidence,
            from_date=rel.from_date.isoformat() if rel.from_date else None,
            to_date=rel.to_date.isoformat() if rel.to_date else None,
            sources=rel.sources,
        )
        return rel.id
    
    def _memory_get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str],
        direction: str
    ) -> List[GraphRelationship]:
        """Get relationships from in-memory graph."""
        results = []
        
        if direction in ["outgoing", "both"]:
            for _, target, key, data in self._memory_graph.out_edges(entity_id, keys=True, data=True):
                if relationship_type and data.get("relationship_type") != relationship_type:
                    continue
                results.append(self._edge_to_relationship(entity_id, target, key, data))
        
        if direction in ["incoming", "both"]:
            for source, _, key, data in self._memory_graph.in_edges(entity_id, keys=True, data=True):
                if relationship_type and data.get("relationship_type") != relationship_type:
                    continue
                results.append(self._edge_to_relationship(source, entity_id, key, data))
        
        return results
    
    def _edge_to_relationship(
        self,
        source: str,
        target: str,
        key: str,
        data: Dict
    ) -> GraphRelationship:
        """Convert edge data to GraphRelationship."""
        return GraphRelationship(
            id=key,
            source_id=source,
            target_id=target,
            relationship_type=data.get("relationship_type", "RELATED_TO"),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            sources=data.get("sources", []),
        )
    
    def _memory_find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int
    ) -> Optional[List[Tuple[GraphEntity, GraphRelationship]]]:
        """Find path in in-memory graph."""
        import networkx as nx
        
        try:
            path = nx.shortest_path(
                self._memory_graph.to_undirected(),
                source_id,
                target_id,
            )
            
            if len(path) - 1 > max_hops:
                return None
            
            result = []
            for i, node_id in enumerate(path):
                entity = self._memory_get_entity(node_id)
                rel = None
                if i > 0:
                    # Get relationship between previous and current
                    prev_id = path[i - 1]
                    if self._memory_graph.has_edge(prev_id, node_id):
                        for key, data in self._memory_graph[prev_id][node_id].items():
                            rel = self._edge_to_relationship(prev_id, node_id, key, data)
                            break
                result.append((entity, rel))
            
            return result
            
        except nx.NetworkXNoPath:
            return None
    
    def _memory_get_network(
        self,
        entity_id: str,
        depth: int,
        limit: int
    ) -> Dict[str, Any]:
        """Get network around entity in in-memory graph."""
        import networkx as nx
        
        if entity_id not in self._memory_graph:
            return {"nodes": [], "edges": [], "center": entity_id}
        
        # BFS to get neighbors up to depth
        visited = set([entity_id])
        current_level = [entity_id]
        
        for d in range(depth):
            next_level = []
            for node in current_level:
                for neighbor in self._memory_graph.neighbors(node):
                    if neighbor not in visited and len(visited) < limit:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level
        
        # Build response
        nodes = []
        for node_id in visited:
            entity = self._memory_get_entity(node_id)
            if entity:
                nodes.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "countries": entity.countries,
                    "sanctions_count": len(entity.sanctions),
                })
        
        edges = []
        for source in visited:
            for target in self._memory_graph.neighbors(source):
                if target in visited:
                    for key, data in self._memory_graph[source][target].items():
                        edges.append({
                            "source": source,
                            "target": target,
                            "type": data.get("relationship_type", "RELATED_TO"),
                            "confidence": data.get("confidence", 1.0),
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "center": entity_id,
            "depth": depth,
        }
    
    def _memory_get_stats(self) -> Dict[str, Any]:
        """Get stats from in-memory graph."""
        return {
            "total_entities": self._memory_graph.number_of_nodes(),
            "total_relationships": self._memory_graph.number_of_edges(),
            "backend": "networkx",
        }
    
    def _memory_bulk_import(
        self,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
        batch_size: int
    ) -> Dict[str, int]:
        """Bulk import to in-memory graph."""
        for entity in entities:
            self._memory_add_entity(entity)
        
        for rel in relationships:
            self._memory_add_relationship(rel)
        
        return {
            "entities_imported": len(entities),
            "relationships_imported": len(relationships),
        }
    
    def _memory_find_connections(
        self,
        entity_ids: List[str],
        max_hops: int
    ) -> Dict[str, Any]:
        """Find connections between entities in memory."""
        # Find all pairwise paths
        connections = []
        for i, source in enumerate(entity_ids):
            for target in entity_ids[i + 1:]:
                path = self._memory_find_path(source, target, max_hops)
                if path:
                    connections.append({
                        "source": source,
                        "target": target,
                        "path_length": len(path) - 1,
                        "path": [
                            {"entity": e.id, "name": e.name}
                            for e, _ in path
                        ],
                    })
        
        return {
            "entity_ids": entity_ids,
            "connections": connections,
            "connected_count": len(connections),
        }
    
    def _memory_get_central_entities(
        self,
        entity_type: Optional[str],
        centrality_measure: str,
        limit: int
    ) -> List[Tuple[GraphEntity, float]]:
        """Get central entities from memory graph."""
        import networkx as nx
        
        # Calculate centrality
        if centrality_measure == "degree":
            centrality = nx.degree_centrality(self._memory_graph)
        elif centrality_measure == "betweenness":
            centrality = nx.betweenness_centrality(self._memory_graph)
        elif centrality_measure == "pagerank":
            centrality = nx.pagerank(self._memory_graph)
        else:
            centrality = nx.degree_centrality(self._memory_graph)
        
        # Sort and filter
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, score in sorted_nodes:
            entity = self._memory_get_entity(node_id)
            if entity:
                if entity_type and entity.entity_type != entity_type:
                    continue
                results.append((entity, score))
                if len(results) >= limit:
                    break
        
        return results
    
    # ==================== Neo4j Implementations ====================
    
    def _neo4j_add_entity(self, entity: GraphEntity) -> str:
        """Add entity to Neo4j."""
        with self._driver.session() as session:
            # Use MERGE to upsert
            query = """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.entity_type = $entity_type,
                e.countries = $countries,
                e.sanctions = $sanctions,
                e.sources = $sources,
                e.properties = $properties,
                e.updated_at = datetime()
            ON CREATE SET e.created_at = datetime()
            RETURN e.id
            """
            result = session.run(
                query,
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                countries=entity.countries,
                sanctions=entity.sanctions,
                sources=entity.sources,
                properties=json.dumps(entity.properties),
            )
            record = result.single()
            return record["e.id"] if record else entity.id
    
    def _neo4j_get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity from Neo4j."""
        with self._driver.session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            RETURN e
            """
            result = session.run(query, id=entity_id)
            record = result.single()
            
            if not record:
                return None
            
            node = record["e"]
            return GraphEntity(
                id=node["id"],
                name=node.get("name", ""),
                entity_type=node.get("entity_type", "unknown"),
                countries=list(node.get("countries", [])),
                sanctions=list(node.get("sanctions", [])),
                sources=list(node.get("sources", [])),
                properties=json.loads(node.get("properties", "{}")),
            )
    
    def _neo4j_search_entities(
        self,
        query: str,
        entity_type: Optional[str],
        limit: int
    ) -> List[GraphEntity]:
        """Search entities in Neo4j."""
        with self._driver.session() as session:
            if entity_type:
                cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                AND e.entity_type = $entity_type
                RETURN e
                LIMIT $limit
                """
                result = session.run(cypher, query=query, entity_type=entity_type, limit=limit)
            else:
                cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                RETURN e
                LIMIT $limit
                """
                result = session.run(cypher, query=query, limit=limit)
            
            entities = []
            for record in result:
                node = record["e"]
                entities.append(GraphEntity(
                    id=node["id"],
                    name=node.get("name", ""),
                    entity_type=node.get("entity_type", "unknown"),
                    countries=list(node.get("countries", [])),
                    sanctions=list(node.get("sanctions", [])),
                    sources=list(node.get("sources", [])),
                    properties=json.loads(node.get("properties", "{}")),
                ))
            return entities
    
    def _neo4j_add_relationship(self, rel: GraphRelationship) -> str:
        """Add relationship to Neo4j."""
        with self._driver.session() as session:
            # Dynamic relationship type using APOC or string interpolation
            # For safety, we use parameterized properties
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{rel.relationship_type} {{id: $rel_id}}]->(target)
            SET r.confidence = $confidence,
                r.properties = $properties,
                r.sources = $sources,
                r.from_date = $from_date,
                r.to_date = $to_date
            RETURN r.id
            """
            result = session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                rel_id=rel.id,
                confidence=rel.confidence,
                properties=json.dumps(rel.properties),
                sources=rel.sources,
                from_date=rel.from_date.isoformat() if rel.from_date else None,
                to_date=rel.to_date.isoformat() if rel.to_date else None,
            )
            record = result.single()
            return record["r.id"] if record else rel.id
    
    def _neo4j_get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str],
        direction: str
    ) -> List[GraphRelationship]:
        """Get relationships from Neo4j."""
        with self._driver.session() as session:
            if direction == "outgoing":
                if relationship_type:
                    query = f"""
                    MATCH (e:Entity {{id: $id}})-[r:{relationship_type}]->(t:Entity)
                    RETURN r, e.id as source, t.id as target, type(r) as rel_type
                    """
                else:
                    query = """
                    MATCH (e:Entity {id: $id})-[r]->(t:Entity)
                    RETURN r, e.id as source, t.id as target, type(r) as rel_type
                    """
            elif direction == "incoming":
                if relationship_type:
                    query = f"""
                    MATCH (s:Entity)-[r:{relationship_type}]->(e:Entity {{id: $id}})
                    RETURN r, s.id as source, e.id as target, type(r) as rel_type
                    """
                else:
                    query = """
                    MATCH (s:Entity)-[r]->(e:Entity {id: $id})
                    RETURN r, s.id as source, e.id as target, type(r) as rel_type
                    """
            else:  # both
                if relationship_type:
                    query = f"""
                    MATCH (e:Entity {{id: $id}})-[r:{relationship_type}]-(t:Entity)
                    RETURN r, 
                           CASE WHEN startNode(r).id = $id THEN startNode(r).id ELSE endNode(r).id END as source,
                           CASE WHEN startNode(r).id = $id THEN endNode(r).id ELSE startNode(r).id END as target,
                           type(r) as rel_type
                    """
                else:
                    query = """
                    MATCH (e:Entity {id: $id})-[r]-(t:Entity)
                    RETURN r,
                           startNode(r).id as source,
                           endNode(r).id as target,
                           type(r) as rel_type
                    """
            
            result = session.run(query, id=entity_id)
            
            relationships = []
            for record in result:
                r = record["r"]
                relationships.append(GraphRelationship(
                    id=r.get("id", ""),
                    source_id=record["source"],
                    target_id=record["target"],
                    relationship_type=record["rel_type"],
                    confidence=r.get("confidence", 1.0),
                    properties=json.loads(r.get("properties", "{}")),
                    sources=list(r.get("sources", [])),
                ))
            return relationships
    
    def _neo4j_find_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int
    ) -> Optional[List[Tuple[GraphEntity, GraphRelationship]]]:
        """Find path in Neo4j."""
        with self._driver.session() as session:
            query = f"""
            MATCH path = shortestPath(
                (source:Entity {{id: $source_id}})-[*1..{max_hops}]-(target:Entity {{id: $target_id}})
            )
            RETURN nodes(path) as nodes, relationships(path) as rels
            """
            result = session.run(query, source_id=source_id, target_id=target_id)
            record = result.single()
            
            if not record:
                return None
            
            nodes = record["nodes"]
            rels = record["rels"]
            
            path_result = []
            for i, node in enumerate(nodes):
                entity = GraphEntity(
                    id=node["id"],
                    name=node.get("name", ""),
                    entity_type=node.get("entity_type", "unknown"),
                    countries=list(node.get("countries", [])),
                    sanctions=list(node.get("sanctions", [])),
                )
                
                rel = None
                if i > 0 and i - 1 < len(rels):
                    r = rels[i - 1]
                    rel = GraphRelationship(
                        id=r.get("id", ""),
                        source_id=nodes[i-1]["id"],
                        target_id=node["id"],
                        relationship_type=r.type,
                        confidence=r.get("confidence", 1.0),
                    )
                
                path_result.append((entity, rel))
            
            return path_result
    
    def _neo4j_get_network(
        self,
        entity_id: str,
        depth: int,
        limit: int
    ) -> Dict[str, Any]:
        """Get network from Neo4j."""
        with self._driver.session() as session:
            query = f"""
            MATCH path = (center:Entity {{id: $id}})-[*1..{depth}]-(connected:Entity)
            WITH center, connected, relationships(path) as rels
            LIMIT $limit
            RETURN DISTINCT connected, rels
            """
            result = session.run(query, id=entity_id, limit=limit)
            
            nodes_map = {}
            edges = []
            
            # Add center node
            center = self._neo4j_get_entity(entity_id)
            if center:
                nodes_map[center.id] = {
                    "id": center.id,
                    "name": center.name,
                    "type": center.entity_type,
                    "countries": center.countries,
                    "sanctions_count": len(center.sanctions),
                }
            
            for record in result:
                node = record["connected"]
                node_id = node["id"]
                
                if node_id not in nodes_map:
                    nodes_map[node_id] = {
                        "id": node_id,
                        "name": node.get("name", ""),
                        "type": node.get("entity_type", "unknown"),
                        "countries": list(node.get("countries", [])),
                        "sanctions_count": len(node.get("sanctions", [])),
                    }
                
                for r in record["rels"]:
                    edges.append({
                        "source": r.start_node["id"],
                        "target": r.end_node["id"],
                        "type": r.type,
                        "confidence": r.get("confidence", 1.0),
                    })
            
            return {
                "nodes": list(nodes_map.values()),
                "edges": edges,
                "center": entity_id,
                "depth": depth,
            }
    
    def _neo4j_find_connections(
        self,
        entity_ids: List[str],
        max_hops: int
    ) -> Dict[str, Any]:
        """Find connections in Neo4j."""
        connections = []
        
        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i + 1:]:
                path = self._neo4j_find_path(source_id, target_id, max_hops)
                if path:
                    connections.append({
                        "source": source_id,
                        "target": target_id,
                        "path_length": len(path) - 1,
                        "path": [
                            {"entity": e.id, "name": e.name}
                            for e, _ in path
                        ],
                    })
        
        return {
            "entity_ids": entity_ids,
            "connections": connections,
            "connected_count": len(connections),
        }
    
    def _neo4j_get_stats(self) -> Dict[str, Any]:
        """Get stats from Neo4j."""
        with self._driver.session() as session:
            # Count entities
            entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = entity_result.single()["count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            # Count by type
            type_result = session.run("""
                MATCH (e:Entity)
                RETURN e.entity_type as type, count(e) as count
            """)
            by_type = {r["type"]: r["count"] for r in type_result}
            
            return {
                "total_entities": entity_count,
                "total_relationships": rel_count,
                "by_type": by_type,
                "backend": "neo4j",
            }
    
    def _neo4j_get_central_entities(
        self,
        entity_type: Optional[str],
        centrality_measure: str,
        limit: int
    ) -> List[Tuple[GraphEntity, float]]:
        """Get central entities from Neo4j using GDS."""
        with self._driver.session() as session:
            # Use degree centrality (GDS would need graph projection)
            if entity_type:
                query = """
                MATCH (e:Entity)
                WHERE e.entity_type = $entity_type
                WITH e, size((e)--()) as degree
                ORDER BY degree DESC
                LIMIT $limit
                RETURN e, degree
                """
                result = session.run(query, entity_type=entity_type, limit=limit)
            else:
                query = """
                MATCH (e:Entity)
                WITH e, size((e)--()) as degree
                ORDER BY degree DESC
                LIMIT $limit
                RETURN e, degree
                """
                result = session.run(query, limit=limit)
            
            entities = []
            for record in result:
                node = record["e"]
                entity = GraphEntity(
                    id=node["id"],
                    name=node.get("name", ""),
                    entity_type=node.get("entity_type", "unknown"),
                    countries=list(node.get("countries", [])),
                    sanctions=list(node.get("sanctions", [])),
                )
                entities.append((entity, float(record["degree"])))
            
            return entities
    
    def _neo4j_bulk_import(
        self,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
        batch_size: int
    ) -> Dict[str, int]:
        """Bulk import to Neo4j."""
        entity_count = 0
        rel_count = 0
        
        with self._driver.session() as session:
            # Bulk import entities
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                entity_data = [
                    {
                        "id": e.id,
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "countries": e.countries,
                        "sanctions": e.sanctions,
                        "sources": e.sources,
                        "properties": json.dumps(e.properties),
                    }
                    for e in batch
                ]
                
                query = """
                UNWIND $entities as entity
                MERGE (e:Entity {id: entity.id})
                SET e.name = entity.name,
                    e.entity_type = entity.entity_type,
                    e.countries = entity.countries,
                    e.sanctions = entity.sanctions,
                    e.sources = entity.sources,
                    e.properties = entity.properties,
                    e.updated_at = datetime()
                ON CREATE SET e.created_at = datetime()
                """
                session.run(query, entities=entity_data)
                entity_count += len(batch)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"📊 Imported {entity_count:,} entities...")
            
            # Bulk import relationships (grouped by type)
            rel_by_type = {}
            for rel in relationships:
                if rel.relationship_type not in rel_by_type:
                    rel_by_type[rel.relationship_type] = []
                rel_by_type[rel.relationship_type].append(rel)
            
            for rel_type, rels in rel_by_type.items():
                for i in range(0, len(rels), batch_size):
                    batch = rels[i:i + batch_size]
                    rel_data = [
                        {
                            "id": r.id,
                            "source_id": r.source_id,
                            "target_id": r.target_id,
                            "confidence": r.confidence,
                            "properties": json.dumps(r.properties),
                            "sources": r.sources,
                        }
                        for r in batch
                    ]
                    
                    query = f"""
                    UNWIND $rels as rel
                    MATCH (source:Entity {{id: rel.source_id}})
                    MATCH (target:Entity {{id: rel.target_id}})
                    MERGE (source)-[r:{rel_type} {{id: rel.id}}]->(target)
                    SET r.confidence = rel.confidence,
                        r.properties = rel.properties,
                        r.sources = rel.sources
                    """
                    session.run(query, rels=rel_data)
                    rel_count += len(batch)
        
        # Create indexes
        self._create_neo4j_indexes()
        
        return {
            "entities_imported": entity_count,
            "relationships_imported": rel_count,
        }
    
    def _create_neo4j_indexes(self):
        """Create Neo4j indexes for performance."""
        with self._driver.session() as session:
            # Entity indexes
            session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
            logger.info("✅ Neo4j indexes created")
