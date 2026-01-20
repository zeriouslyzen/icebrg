"""
Batch Importer - Optimized import for large datasets.
Processes millions of records efficiently with progress tracking.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImportStats:
    """Statistics for an import job."""
    source: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_records: int = 0
    entities_imported: int = 0
    relationships_imported: int = 0
    errors: int = 0
    duration_seconds: float = 0.0


class BatchImporter:
    """
    Optimized batch importer for large datasets.
    
    Uses SQLite for fast entity storage and deferred graph building.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the batch importer.
        
        Args:
            data_dir: Directory for matrix data
        """
        self.data_dir = data_dir or Path.home() / "Documents" / "iceburg_matrix"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "matrix.db"
        self._init_db()
        
        logger.info(f"📦 Batch Importer initialized (db: {self.db_path})")
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source TEXT,
                properties TEXT,
                countries TEXT,
                datasets TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast search
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source)")
        
        # Relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                properties TEXT,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")
        
        conn.commit()
        conn.close()
    
    def import_opensanctions(
        self,
        file_path: Path,
        limit: Optional[int] = None,
        batch_size: int = 5000,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> ImportStats:
        """
        Import OpenSanctions FTM JSON file.
        
        Args:
            file_path: Path to entities.ftm.json
            limit: Optional limit on records to process
            batch_size: Number of records per batch
            on_progress: Progress callback (processed, total)
            
        Returns:
            Import statistics
        """
        stats = ImportStats(source="opensanctions", started_at=datetime.now())
        start_time = time.time()
        
        logger.info(f"📥 Starting OpenSanctions import from {file_path}")
        
        # Count total lines first (for progress)
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        stats.total_records = min(total_lines, limit) if limit else total_lines
        
        logger.info(f"📊 Processing {stats.total_records:,} records...")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        batch = []
        processed = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    schema = data.get("schema", "")
                    
                    # Only import key entity types
                    if schema in ["Person", "Company", "Organization", "LegalEntity"]:
                        entity_id = data.get("id", "")
                        name = data.get("caption", "")
                        
                        if not name or not entity_id:
                            continue
                        
                        props = data.get("properties", {})
                        countries = props.get("country", [])
                        datasets = data.get("datasets", [])
                        
                        batch.append((
                            f"osanc_{entity_id}",
                            name,
                            schema.lower(),
                            "opensanctions",
                            json.dumps(props),
                            json.dumps(countries),
                            json.dumps(datasets),
                        ))
                        
                except json.JSONDecodeError:
                    stats.errors += 1
                except Exception as e:
                    stats.errors += 1
                
                # Insert batch
                if len(batch) >= batch_size:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO entities 
                        (entity_id, name, entity_type, source, properties, countries, datasets)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    conn.commit()
                    stats.entities_imported += len(batch)
                    batch = []
                    
                    processed = i + 1
                    if on_progress:
                        on_progress(processed, stats.total_records)
                    
                    if processed % 50000 == 0:
                        logger.info(f"  Processed {processed:,}/{stats.total_records:,} ({100*processed/stats.total_records:.1f}%)")
        
        # Insert remaining batch
        if batch:
            cursor.executemany("""
                INSERT OR REPLACE INTO entities 
                (entity_id, name, entity_type, source, properties, countries, datasets)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, batch)
            conn.commit()
            stats.entities_imported += len(batch)
        
        conn.close()
        
        stats.completed_at = datetime.now()
        stats.duration_seconds = time.time() - start_time
        
        logger.info(f"✅ Import complete: {stats.entities_imported:,} entities in {stats.duration_seconds:.1f}s")
        logger.info(f"   Speed: {stats.entities_imported / stats.duration_seconds:.0f} entities/sec")
        
        return stats
    
    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search entities by name.
        
        Args:
            query: Search query
            entity_type: Optional filter by type
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if entity_type:
            cursor.execute("""
                SELECT entity_id, name, entity_type, source, countries, datasets
                FROM entities
                WHERE name LIKE ? AND entity_type = ?
                LIMIT ?
            """, (f"%{query}%", entity_type.lower(), limit))
        else:
            cursor.execute("""
                SELECT entity_id, name, entity_type, source, countries, datasets
                FROM entities
                WHERE name LIKE ?
                LIMIT ?
            """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entity_id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "source": row[3],
                "countries": json.loads(row[4]) if row[4] else [],
                "datasets": json.loads(row[5]) if row[5] else [],
            })
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total entities
        cursor.execute("SELECT COUNT(*) FROM entities")
        total_entities = cursor.fetchone()[0]
        
        # By type
        cursor.execute("""
            SELECT entity_type, COUNT(*) 
            FROM entities 
            GROUP BY entity_type
        """)
        by_type = dict(cursor.fetchall())
        
        # By source
        cursor.execute("""
            SELECT source, COUNT(*) 
            FROM entities 
            GROUP BY source
        """)
        by_source = dict(cursor.fetchall())
        
        # Total relationships
        cursor.execute("SELECT COUNT(*) FROM relationships")
        total_relationships = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "by_type": by_type,
            "by_source": by_source,
            "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
        }


# Convenience function for CLI use
def import_opensanctions_file(limit: Optional[int] = None) -> ImportStats:
    """Import OpenSanctions from default location."""
    importer = BatchImporter()
    file_path = importer.data_dir / "opensanctions" / "opensanctions_sanctions.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"OpenSanctions file not found: {file_path}")
    
    return importer.import_opensanctions(file_path, limit=limit)
