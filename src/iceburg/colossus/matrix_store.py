
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from .core.graph import GraphEntity

logger = logging.getLogger(__name__)

class MatrixStore:
    """
    Direct access to the Matrix SQLite database.
    Used for full-text search across the entire dataset without loading everything into memory.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Default locations
            possible_paths = [
                Path.home() / "Documents" / "iceburg_matrix" / "matrix.db",
                Path("/Users/jackdanger/Documents/iceburg_matrix/matrix.db"),
                Path.home() / "Desktop" / "Projects" / "iceburg" / "matrix.db",
            ]
            db_path = next((p for p in possible_paths if p.exists()), None)
            
        self.db_path = db_path

    def get_connection(self):
        if not self.db_path or not self.db_path.exists():
            raise FileNotFoundError(f"Matrix DB not found at: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def search(self, query: str, limit: int = 50) -> List[GraphEntity]:
        """Search entities by name using SQL LIKE."""
        if not self.db_path:
            return []
            
        sql_query = f"%{query}%"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT entity_id, name, entity_type, source, countries, datasets, properties
                    FROM entities
                    WHERE name LIKE ?
                    LIMIT ?
                """, (sql_query, limit))
                
                return [self._row_to_entity(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Matrix Search Error: {e}")
            return []

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get full entity details by ID.
        
        Handles ID format variations:
        - Tries exact match first
        - If not found and ID starts with 'NK-', tries with 'osanc_' prefix
        - If not found and ID starts with 'osanc_', tries without prefix
        """
        if not self.db_path:
            return None
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Try exact match first
                cursor.execute("""
                    SELECT entity_id, name, entity_type, source, countries, datasets, properties
                    FROM entities
                    WHERE entity_id = ?
                """, (entity_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
                
                # Try with osanc_ prefix if ID starts with NK-
                if entity_id.startswith('NK-'):
                    prefixed_id = f'osanc_{entity_id}'
                    cursor.execute("""
                        SELECT entity_id, name, entity_type, source, countries, datasets, properties
                        FROM entities
                        WHERE entity_id = ?
                    """, (prefixed_id,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_entity(row)
                
                # Try without osanc_ prefix if ID starts with osanc_
                elif entity_id.startswith('osanc_'):
                    unprefixed_id = entity_id[6:]  # Remove 'osanc_' prefix
                    cursor.execute("""
                        SELECT entity_id, name, entity_type, source, countries, datasets, properties
                        FROM entities
                        WHERE entity_id = ?
                    """, (unprefixed_id,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_entity(row)
                
                return None
        except Exception as e:
            logger.error(f"Matrix Get Error: {e}")
            return None

    def _row_to_entity(self, row: sqlite3.Row) -> GraphEntity:
        """Convert SQLite row to GraphEntity."""
        return GraphEntity(
            id=row["entity_id"],
            name=row["name"],
            entity_type=row["entity_type"],
            countries=json.loads(row["countries"]) if row["countries"] else [],
            sanctions=json.loads(row["datasets"]) if row["datasets"] else [],
            sources=[row["source"]] if row["source"] else [],
            properties=json.loads(row["properties"]) if row["properties"] else {},
        )

    def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity (incoming and outgoing)."""
        if not self.db_path:
            return []
            
        from .core.graph import GraphRelationship
        relationships = []
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT relationship_id, source_id, target_id, relationship_type, properties
                    FROM relationships
                    WHERE source_id = ? OR target_id = ?
                """, (entity_id, entity_id))
                
                for row in cursor.fetchall():
                    relationships.append(GraphRelationship(
                        id=row["relationship_id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relationship_type=row["relationship_type"],
                        properties=json.loads(row["properties"]) if row["properties"] else {},
                        confidence=1.0,
                        sources=["matrix_db"]
                    ))
        except Exception as e:
            logger.error(f"Matrix Relationships Error: {e}")
            
        return relationships

    def get_network(self, entity_id: str, depth: int = 2, limit: int = 100) -> Dict[str, Any]:
        """
        Get network around an entity by querying SQLite directly.
        Returns nodes and edges for visualization.
        
        Guarantees:
        - Center entity always included in nodes, even if isolated
        - Edges only include nodes that exist in database
        - Comprehensive logging and diagnostic information
        """
        if not self.db_path:
            return {
                "nodes": [],
                "edges": [],
                "center": entity_id,
                "error": "Matrix database not found"
            }
        
        # Initialize query statistics
        query_stats = {
            "relationships_found": 0,
            "nodes_found": 0,
            "nodes_missing": 0,
            "edges_filtered": 0,
            "center_entity_exists": False
        }
        
        try:
            # Phase 1: Validate center entity exists FIRST (with ID normalization)
            logger.debug(f"Validating center entity: {entity_id}")
            center_entity = self.get_entity(entity_id)
            if not center_entity:
                logger.warning(f"Center entity not found: {entity_id}")
                return {
                    "nodes": [],
                    "edges": [],
                    "center": entity_id,
                    "error": f"Entity {entity_id} not found in database",
                    "query_stats": query_stats
                }
            
            # Use the actual entity ID from database (may have prefix)
            actual_entity_id = center_entity.id
            
            query_stats["center_entity_exists"] = True
            logger.info(f"Center entity found: {center_entity.name} (requested: {entity_id}, actual: {actual_entity_id})")
            
            # Phase 2: Always include center entity in result (guaranteed)
            center_props = center_entity.properties or {}
            center_node = {
                "id": center_entity.id,
                "name": center_entity.name,
                "type": center_entity.entity_type,
                "countries": center_entity.countries,
                "sanctions_count": len(center_entity.sanctions),
            }
            if center_props.get("domains") is not None:
                center_node["domains"] = center_props["domains"]
            if center_props.get("roles") is not None:
                center_node["roles"] = center_props["roles"]
            nodes_map = {actual_entity_id: center_node}
            
            # Phase 3: Handle depth=0 case explicitly
            if depth == 0:
                logger.debug(f"Depth=0, returning center entity only")
                return {
                    "nodes": list(nodes_map.values()),
                    "edges": [],
                    "center": actual_entity_id,
                    "depth": depth,
                    "query_stats": query_stats
                }
            
            # Phase 4: BFS to collect entities up to depth
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                visited_entities = {actual_entity_id}  # Start with center entity (use actual ID)
                all_edges = []
                current_level = {actual_entity_id}
                
                for d in range(depth):
                    if not current_level or len(visited_entities) >= limit:
                        logger.debug(f"BFS depth {d}: stopping (level empty or limit reached)")
                        break
                    
                    logger.debug(f"BFS depth {d}: processing {len(current_level)} entities")
                    next_level = set()
                    
                    # Query relationships for current level entities (include properties for domain)
                    if current_level:
                        placeholders = ','.join('?' for _ in current_level)
                        query_params = list(current_level) + list(current_level) + [limit * 2]
                        cursor.execute(f"""
                            SELECT r.relationship_id, r.source_id, r.target_id, r.relationship_type, r.properties
                            FROM relationships r
                            INNER JOIN entities e1 ON e1.entity_id = r.source_id
                            INNER JOIN entities e2 ON e2.entity_id = r.target_id
                            WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                            LIMIT ?
                        """, query_params)
                        
                        rows = cursor.fetchall()
                        query_stats["relationships_found"] += len(rows)
                        logger.debug(f"Found {len(rows)} relationships at depth {d}")
                        
                        for row in rows:
                            rel_props = json.loads(row["properties"]) if row["properties"] else {}
                            edge = {
                                "id": row["relationship_id"],
                                "source": row["source_id"],
                                "target": row["target_id"],
                                "type": row["relationship_type"],
                            }
                            if rel_props.get("domain") is not None:
                                edge["domain"] = rel_props["domain"]
                            all_edges.append(edge)
                            
                            # Add neighbors to next level (if not already visited)
                            if row["source_id"] not in visited_entities:
                                next_level.add(row["source_id"])
                            if row["target_id"] not in visited_entities:
                                next_level.add(row["target_id"])
                    
                    # Limit next level size
                    if len(next_level) + len(visited_entities) > limit:
                        next_level = set(list(next_level)[:limit - len(visited_entities)])
                    
                    # Mark current level as visited
                    visited_entities.update(current_level)
                    current_level = next_level
                
                # Collect final layer neighbors
                visited_entities.update(current_level)
                logger.debug(f"BFS complete: {len(visited_entities)} entities visited, {len(all_edges)} edges found")
                
                # Phase 5: Fetch node details for visited entities (include properties for domains/roles)
                if visited_entities:
                    placeholders = ','.join('?' for _ in visited_entities)
                    cursor.execute(f"""
                        SELECT entity_id, name, entity_type, countries, datasets, properties
                        FROM entities
                        WHERE entity_id IN ({placeholders})
                    """, list(visited_entities))
                    
                    for row in cursor.fetchall():
                        entity_id_found = row["entity_id"]
                        props = json.loads(row["properties"]) if row["properties"] else {}
                        node = {
                            "id": row["entity_id"],
                            "name": row["name"],
                            "type": row["entity_type"],
                            "countries": json.loads(row["countries"]) if row["countries"] else [],
                            "sanctions_count": len(json.loads(row["datasets"])) if row["datasets"] else 0,
                        }
                        if props.get("domains") is not None:
                            node["domains"] = props["domains"]
                        if props.get("roles") is not None:
                            node["roles"] = props["roles"]
                        nodes_map[entity_id_found] = node
                    
                    query_stats["nodes_found"] = len(nodes_map)
                    query_stats["nodes_missing"] = len(visited_entities) - len(nodes_map)
                    
                    if query_stats["nodes_missing"] > 0:
                        logger.warning(f"{query_stats['nodes_missing']} visited entities not found in database")
                
                # Phase 6: Filter edges AFTER fetching nodes
                # Only keep edges where both source and target exist in nodes_map
                valid_edges = []
                for e in all_edges:
                    if e["source"] in nodes_map and e["target"] in nodes_map:
                        valid_edges.append(e)
                    else:
                        query_stats["edges_filtered"] += 1
                
                if query_stats["edges_filtered"] > 0:
                    logger.warning(f"Filtered out {query_stats['edges_filtered']} edges with missing nodes")
                
                logger.info(f"Network query complete: {len(nodes_map)} nodes, {len(valid_edges)} edges")
                
                return {
                    "nodes": list(nodes_map.values()),
                    "edges": valid_edges,
                    "center": actual_entity_id,
                    "depth": depth,
                    "query_stats": query_stats
                }
                
        except Exception as e:
            logger.error(f"Matrix Get Network Error for {entity_id}: {e}", exc_info=True)
            return {
                "nodes": [],
                "edges": [],
                "center": entity_id,
                "error": str(e),
                "query_stats": query_stats
            }

