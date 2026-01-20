"""
COLOSSUS Migration Script

Migrate Matrix SQLite data to Neo4j graph database.
Extracts entities and relationships from OpenSanctions data.
"""

import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Migration statistics."""
    entities_read: int = 0
    entities_migrated: int = 0
    relationships_extracted: int = 0
    errors: int = 0
    duration_seconds: float = 0


class MatrixMigrator:
    """
    Migrate entities from Matrix SQLite to COLOSSUS graph.
    
    Extracts:
    - Entities (Person, Company, Organization)
    - Relationships (from OpenSanctions schema)
    """
    
    def __init__(
        self,
        matrix_db_path: Path,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "colossus2024",
    ):
        """Initialize migrator."""
        self.matrix_db_path = matrix_db_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        self._matrix_conn: Optional[sqlite3.Connection] = None
        self._graph = None
    
    def connect(self):
        """Connect to both databases."""
        # Connect to Matrix SQLite
        if not self.matrix_db_path.exists():
            raise FileNotFoundError(f"Matrix database not found: {self.matrix_db_path}")
        
        self._matrix_conn = sqlite3.connect(str(self.matrix_db_path))
        self._matrix_conn.row_factory = sqlite3.Row
        logger.info(f"📊 Connected to Matrix: {self.matrix_db_path}")
        
        # Connect to Neo4j
        from .core.graph import ColossusGraph
        self._graph = ColossusGraph(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            use_memory=False,
        )
        
        if not self._graph.is_neo4j:
            logger.warning("⚠️ Neo4j not available, using in-memory graph")
    
    def migrate(
        self,
        batch_size: int = 5000,
        limit: Optional[int] = None,
        extract_relationships: bool = True,
    ) -> MigrationStats:
        """
        Migrate entities from Matrix to graph.
        
        Args:
            batch_size: Entities per batch
            limit: Max entities to migrate (None for all)
            extract_relationships: Extract relationships from properties
            
        Returns:
            Migration statistics
        """
        import time
        start_time = time.time()
        
        stats = MigrationStats()
        
        # Connect to databases
        self.connect()
        
        # Get total count
        cursor = self._matrix_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        total = cursor.fetchone()[0]
        
        if limit:
            total = min(total, limit)
        
        logger.info(f"🚀 Migrating {total:,} entities...")
        
        # Migrate in batches
        from .core.graph import GraphEntity, GraphRelationship
        
        offset = 0
        all_relationships = []
        
        while offset < total:
            # Fetch batch
            cursor.execute("""
                SELECT entity_id, name, entity_type, source, countries, datasets, properties
                FROM entities
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            
            rows = cursor.fetchall()
            if not rows:
                break
            
            entities = []
            
            for row in rows:
                try:
                    entity_id = row["entity_id"]
                    name = row["name"]
                    entity_type = row["entity_type"]
                    
                    countries = json.loads(row["countries"]) if row["countries"] else []
                    datasets = json.loads(row["datasets"]) if row["datasets"] else []
                    properties = json.loads(row["properties"]) if row["properties"] else {}
                    
                    entity = GraphEntity(
                        id=entity_id,
                        name=name,
                        entity_type=entity_type,
                        countries=countries,
                        sanctions=datasets,
                        sources=[row["source"]] if row["source"] else [],
                        properties=properties,
                    )
                    entities.append(entity)
                    stats.entities_read += 1
                    
                    # Extract relationships from properties
                    if extract_relationships and properties:
                        rels = self._extract_relationships(entity_id, properties)
                        all_relationships.extend(rels)
                        
                except Exception as e:
                    logger.warning(f"Error processing entity: {e}")
                    stats.errors += 1
            
            # Import batch to graph
            result = self._graph.bulk_import(entities, [], batch_size)
            stats.entities_migrated += result.get("entities_imported", 0)
            
            offset += len(rows)
            logger.info(f"📈 Progress: {stats.entities_migrated:,} / {total:,} entities")
        
        # Import relationships in a second pass
        if all_relationships:
            logger.info(f"🔗 Importing {len(all_relationships):,} relationships...")
            result = self._graph.bulk_import([], all_relationships, batch_size)
            stats.relationships_extracted = result.get("relationships_imported", 0)
        
        stats.duration_seconds = time.time() - start_time
        
        logger.info(f"""
✅ Migration Complete
   Entities: {stats.entities_migrated:,}
   Relationships: {stats.relationships_extracted:,}
   Errors: {stats.errors}
   Duration: {stats.duration_seconds:.1f}s
""")
        
        return stats
    
    def _extract_relationships(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> List:
        """Extract relationships from entity properties."""
        from .core.graph import GraphRelationship
        
        relationships = []
        
        # OpenSanctions relationship properties
        rel_props = {
            "ownershipAsset": "OWNS",
            "ownershipOwner": "OWNED_BY",
            "directorshipDirector": "DIRECTOR_OF",
            "directorshipOrganization": "HAS_DIRECTOR",
            "familyPerson": "FAMILY_OF",
            "familyRelative": "RELATIVE_OF",
            "associateOf": "ASSOCIATED_WITH",
            "memberOf": "MEMBER_OF",
        }
        
        for prop, rel_type in rel_props.items():
            if prop in properties:
                targets = properties[prop]
                if isinstance(targets, str):
                    targets = [targets]
                
                for target_id in targets:
                    if target_id and target_id != entity_id:
                        relationships.append(GraphRelationship(
                            id=f"rel_{uuid.uuid4().hex[:12]}",
                            source_id=entity_id,
                            target_id=target_id,
                            relationship_type=rel_type,
                            confidence=0.9,
                            sources=["opensanctions"],
                        ))
        
        return relationships
    
    def close(self):
        """Close connections."""
        if self._matrix_conn:
            self._matrix_conn.close()


def run_migration():
    """Run migration from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Matrix to COLOSSUS")
    parser.add_argument("--matrix-db", type=str, 
                       default=str(Path.home() / "Documents" / "iceburg_matrix" / "matrix.db"),
                       help="Path to Matrix SQLite database")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="colossus2024")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-relationships", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    migrator = MatrixMigrator(
        matrix_db_path=Path(args.matrix_db),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    
    stats = migrator.migrate(
        batch_size=args.batch_size,
        limit=args.limit,
        extract_relationships=not args.no_relationships,
    )
    
    print(f"\nMigration Stats: {stats}")


if __name__ == "__main__":
    run_migration()
