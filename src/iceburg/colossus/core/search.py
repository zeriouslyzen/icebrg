"""
COLOSSUS Search Layer

Full-text search with Elasticsearch backend.
Falls back to SQLite FTS5 for development.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with relevance score."""
    entity_id: str
    name: str
    entity_type: str
    score: float
    highlights: Dict[str, List[str]] = field(default_factory=dict)
    snippet: str = ""
    countries: List[str] = field(default_factory=list)
    sanctions_count: int = 0


@dataclass
class SearchFilters:
    """Search filter options."""
    entity_types: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    sanctions_lists: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    has_sanctions: Optional[bool] = None
    min_risk_score: Optional[float] = None


class ColossusSearch:
    """
    Full-text search for COLOSSUS.
    
    Supports Elasticsearch for production, SQLite FTS5 for development.
    """
    
    def __init__(
        self,
        elasticsearch_url: Optional[str] = None,
        sqlite_path: Optional[str] = None,
    ):
        """
        Initialize search backend.
        
        Args:
            elasticsearch_url: Elasticsearch connection URL
            sqlite_path: Path to SQLite database for FTS5 fallback
        """
        self.elasticsearch_url = elasticsearch_url
        self.sqlite_path = sqlite_path
        
        self._es_client = None
        self._sqlite_conn = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize search backend."""
        # Try Elasticsearch first
        if self.elasticsearch_url:
            try:
                from elasticsearch import Elasticsearch
                self._es_client = Elasticsearch([self.elasticsearch_url])
                if self._es_client.ping():
                    logger.info(f"✅ Connected to Elasticsearch: {self.elasticsearch_url}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Elasticsearch unavailable: {e}")
        
        # Fall back to SQLite FTS5
        if self.sqlite_path:
            try:
                import sqlite3
                self._sqlite_conn = sqlite3.connect(self.sqlite_path)
                self._ensure_fts_table()
                logger.info(f"📊 Using SQLite FTS5: {self.sqlite_path}")
            except Exception as e:
                logger.error(f"❌ SQLite FTS5 failed: {e}")
    
    def _ensure_fts_table(self):
        """Create FTS5 virtual table if not exists."""
        cursor = self._sqlite_conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                entity_id,
                name,
                entity_type,
                countries,
                sanctions,
                content='entities',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        self._sqlite_conn.commit()
    
    @property
    def is_elasticsearch(self) -> bool:
        """Check if using Elasticsearch backend."""
        return self._es_client is not None
    
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: int = 50,
        offset: int = 0,
        fuzzy: bool = True,
    ) -> List[SearchResult]:
        """
        Search entities.
        
        Args:
            query: Search query text
            filters: Optional filters
            limit: Max results to return
            offset: Pagination offset
            fuzzy: Enable fuzzy matching
            
        Returns:
            List of search results with scores
        """
        if self.is_elasticsearch:
            return self._es_search(query, filters, limit, offset, fuzzy)
        else:
            return self._sqlite_search(query, filters, limit, offset, fuzzy)
    
    def suggest(
        self,
        prefix: str,
        limit: int = 10
    ) -> List[str]:
        """
        Autocomplete suggestions.
        
        Args:
            prefix: Search prefix
            limit: Max suggestions
            
        Returns:
            List of suggested names
        """
        if self.is_elasticsearch:
            return self._es_suggest(prefix, limit)
        else:
            return self._sqlite_suggest(prefix, limit)
    
    def get_facets(
        self,
        query: str,
        facet_fields: List[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Get faceted counts for filters.
        
        Args:
            query: Search query
            facet_fields: Fields to aggregate
            
        Returns:
            Facet counts per field
        """
        facet_fields = facet_fields or ["entity_type", "countries", "sanctions"]
        
        if self.is_elasticsearch:
            return self._es_facets(query, facet_fields)
        else:
            return self._sqlite_facets(query, facet_fields)
    
    def similar(
        self,
        entity_id: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find similar entities using embeddings.
        
        Args:
            entity_id: Source entity ID
            limit: Max results
            
        Returns:
            Similar entities
        """
        # TODO: Implement vector similarity with Milvus/Chroma
        return []
    
    def index_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        countries: List[str],
        sanctions: List[str],
        properties: Dict[str, Any] = None,
    ):
        """Index an entity for search."""
        if self.is_elasticsearch:
            self._es_index(entity_id, name, entity_type, countries, sanctions, properties)
        else:
            self._sqlite_index(entity_id, name, entity_type, countries, sanctions, properties)
    
    def bulk_index(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """Bulk index entities."""
        if self.is_elasticsearch:
            return self._es_bulk_index(entities, batch_size)
        else:
            return self._sqlite_bulk_index(entities, batch_size)
    
    # ==================== SQLite FTS5 Implementation ====================
    
    def _sqlite_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        offset: int,
        fuzzy: bool
    ) -> List[SearchResult]:
        """Search using SQLite FTS5."""
        cursor = self._sqlite_conn.cursor()
        
        # Build FTS5 query
        fts_query = f'"{query}"*' if not fuzzy else f'{query}*'
        
        sql = """
            SELECT entity_id, name, entity_type, countries, sanctions,
                   bm25(entities_fts, 1.0, 0.75) as score
            FROM entities_fts
            WHERE entities_fts MATCH ?
        """
        params = [fts_query]
        
        # Apply filters
        if filters:
            if filters.entity_types:
                placeholders = ",".join("?" * len(filters.entity_types))
                sql += f" AND entity_type IN ({placeholders})"
                params.extend(filters.entity_types)
        
        sql += " ORDER BY score LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            import json
            results.append(SearchResult(
                entity_id=row[0],
                name=row[1],
                entity_type=row[2],
                countries=json.loads(row[3]) if row[3] else [],
                sanctions_count=len(json.loads(row[4])) if row[4] else 0,
                score=abs(row[5]) if row[5] else 0,
            ))
        
        return results
    
    def _sqlite_suggest(self, prefix: str, limit: int) -> List[str]:
        """Autocomplete using SQLite."""
        cursor = self._sqlite_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT name FROM entities_fts
            WHERE name MATCH ?
            LIMIT ?
        """, [f'{prefix}*', limit])
        return [row[0] for row in cursor.fetchall()]
    
    def _sqlite_facets(
        self,
        query: str,
        facet_fields: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Get facets from SQLite."""
        cursor = self._sqlite_conn.cursor()
        facets = {}
        
        for field in facet_fields:
            if field == "entity_type":
                cursor.execute("""
                    SELECT entity_type, COUNT(*) as count
                    FROM entities_fts
                    WHERE entities_fts MATCH ?
                    GROUP BY entity_type
                """, [f'{query}*'])
                facets[field] = {row[0]: row[1] for row in cursor.fetchall()}
        
        return facets
    
    def _sqlite_index(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        countries: List[str],
        sanctions: List[str],
        properties: Dict[str, Any]
    ):
        """Index entity in SQLite FTS5."""
        import json
        cursor = self._sqlite_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO entities_fts (entity_id, name, entity_type, countries, sanctions)
            VALUES (?, ?, ?, ?, ?)
        """, [entity_id, name, entity_type, json.dumps(countries), json.dumps(sanctions)])
        self._sqlite_conn.commit()
    
    def _sqlite_bulk_index(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int
    ) -> int:
        """Bulk index in SQLite."""
        import json
        cursor = self._sqlite_conn.cursor()
        count = 0
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            cursor.executemany("""
                INSERT OR REPLACE INTO entities_fts (entity_id, name, entity_type, countries, sanctions)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (e["entity_id"], e["name"], e["entity_type"],
                 json.dumps(e.get("countries", [])), json.dumps(e.get("sanctions", [])))
                for e in batch
            ])
            count += len(batch)
        
        self._sqlite_conn.commit()
        return count
    
    # ==================== Elasticsearch Implementation (TODO) ====================
    
    def _es_search(
        self,
        query: str,
        filters: Optional[SearchFilters],
        limit: int,
        offset: int,
        fuzzy: bool
    ) -> List[SearchResult]:
        """Search using Elasticsearch."""
        raise NotImplementedError("Elasticsearch integration pending")
    
    def _es_suggest(self, prefix: str, limit: int) -> List[str]:
        """Suggest using Elasticsearch."""
        raise NotImplementedError("Elasticsearch integration pending")
    
    def _es_facets(
        self,
        query: str,
        facet_fields: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Get facets from Elasticsearch."""
        raise NotImplementedError("Elasticsearch integration pending")
    
    def _es_index(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        countries: List[str],
        sanctions: List[str],
        properties: Dict[str, Any]
    ):
        """Index in Elasticsearch."""
        raise NotImplementedError("Elasticsearch integration pending")
    
    def _es_bulk_index(
        self,
        entities: List[Dict[str, Any]],
        batch_size: int
    ) -> int:
        """Bulk index in Elasticsearch."""
        raise NotImplementedError("Elasticsearch integration pending")
