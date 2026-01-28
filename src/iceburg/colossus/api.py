"""
COLOSSUS API Routes

GraphQL and REST endpoints for the intelligence platform.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/colossus", tags=["COLOSSUS"])


# ==================== Models ====================

class EntitySearchRequest(BaseModel):
    """Entity search request."""
    query: str
    entity_type: Optional[str] = None
    limit: int = Field(default=50, le=500)


class NetworkRequest(BaseModel):
    """Network request."""
    entity_id: str
    depth: int = Field(default=2, le=5)
    limit: int = Field(default=100, le=500)


class PathRequest(BaseModel):
    """Path finding request."""
    source_id: str
    target_id: str
    max_hops: int = Field(default=6, le=10)


class ConnectionsRequest(BaseModel):
    """Find connections between entities."""
    entity_ids: List[str]
    max_hops: int = Field(default=3, le=6)


class RiskRequest(BaseModel):
    """Risk assessment request."""
    entity_id: str
    include_network: bool = True


# ==================== Graph Singleton ====================

_graph = None


def get_graph():
    """Get or create graph instance."""
    global _graph
    if _graph is None:
        from .core.graph import ColossusGraph
        _graph = ColossusGraph(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="colossus2024",
            use_memory=True,  # Fall back to memory if Neo4j unavailable
        )
    return _graph


# ==================== Endpoints ====================

@router.get("/status")
async def get_status():
    """Get COLOSSUS system status."""
    graph = get_graph()
    stats = graph.get_stats()
    
    return {
        "status": "operational",
        "backend": "neo4j" if graph.is_neo4j else "networkx",
        "total_entities": stats.get("total_entities", 0),
        "total_relationships": stats.get("total_relationships", 0),
        "by_type": stats.get("by_type", {}),
    }


@router.post("/entities/search")
async def search_entities(request: EntitySearchRequest):
    """
    Search entities in the Matrix Database (Hybrid Search).
    Queries SQLite directly for 100% coverage, then returns results.
    """
    from .matrix_store import MatrixStore
    
    # Use MatrixStore for full dataset search
    store = MatrixStore()
    entities = store.search(request.query, limit=request.limit)
    
    # Optional: Filter by type if requested (can also be moved to SQL for speed)
    if request.entity_type:
        entities = [e for e in entities if e.entity_type == request.entity_type]
    
    return {
        "query": request.query,
        "count": len(entities),
        "results": [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "countries": e.countries,
                "sanctions": e.sanctions,
                "sanctions_count": len(e.sanctions),
            }
            for e in entities
        ],
    }


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get entity by ID."""
    graph = get_graph()
    entity = graph.get_entity(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Get relationships
    relationships = graph.get_relationships(entity_id)
    
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.entity_type,
        "countries": entity.countries,
        "sanctions": entity.sanctions,
        "sources": entity.sources,
        "properties": entity.properties,
        "relationships": [
            {
                "id": r.id,
                "type": r.relationship_type,
                "source": r.source_id,
                "target": r.target_id,
                "confidence": r.confidence,
            }
            for r in relationships
        ],
    }


@router.get("/entities/{entity_id}/validate")
async def validate_entity(entity_id: str):
    """Validate entity exists and return basic info."""
    from .matrix_store import MatrixStore
    store = MatrixStore()
    entity = store.get_entity(entity_id)
    
    if not entity:
        return {
            "exists": False,
            "entity_id": entity_id,
            "message": "Entity not found in database"
        }
    
    # Get relationship count
    rels = store.get_relationships(entity_id)
    
    return {
        "exists": True,
        "entity_id": entity_id,
        "name": entity.name,
        "type": entity.entity_type,
        "relationship_count": len(rels),
        "has_relationships": len(rels) > 0
    }


@router.post("/network")
async def get_network(request: NetworkRequest):
    """Get entity network directly from SQLite relationships table."""
    from .matrix_store import MatrixStore
    
    # Use MatrixStore for direct SQLite query (much faster, 1.4M relationships)
    store = MatrixStore()
    network = store.get_network(
        entity_id=request.entity_id,
        depth=request.depth,
        limit=request.limit,
    )
    
    return network


@router.post("/path")
async def find_path(request: PathRequest):
    """Find shortest path between entities."""
    graph = get_graph()
    
    path = graph.find_path(
        source_id=request.source_id,
        target_id=request.target_id,
        max_hops=request.max_hops,
    )
    
    if not path:
        return {
            "found": False,
            "source": request.source_id,
            "target": request.target_id,
            "message": "No path found within hop limit",
        }
    
    return {
        "found": True,
        "source": request.source_id,
        "target": request.target_id,
        "hops": len(path) - 1,
        "path": [
            {
                "entity": {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                },
                "relationship": {
                    "type": rel.relationship_type if rel else None,
                    "confidence": rel.confidence if rel else None,
                } if rel else None,
            }
            for entity, rel in path
        ],
    }


@router.post("/connections")
async def find_connections(request: ConnectionsRequest):
    """Find connections between multiple entities."""
    graph = get_graph()
    
    result = graph.find_connections(
        entity_ids=request.entity_ids,
        max_hops=request.max_hops,
    )
    
    return result


@router.get("/central")
async def get_central_entities(
    entity_type: Optional[str] = None,
    measure: str = Query(default="degree", regex="^(degree|betweenness|pagerank)$"),
    limit: int = Query(default=20, le=100),
):
    """Get most central entities."""
    graph = get_graph()
    
    results = graph.get_central_entities(
        entity_type=entity_type,
        centrality_measure=measure,
        limit=limit,
    )
    
    return {
        "measure": measure,
        "results": [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "score": score,
                "sanctions_count": len(entity.sanctions),
            }
            for entity, score in results
        ],
    }


@router.post("/risk")
async def assess_risk(request: RiskRequest):
    """Get risk assessment for entity."""
    from .intelligence.risk import RiskScorer
    
    graph = get_graph()
    entity = graph.get_entity(request.entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Build entity dict for scorer
    entity_data = {
        "entity_id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "countries": entity.countries,
        "sanctions": entity.sanctions,
        "datasets": entity.sanctions,
        "properties": entity.properties,
    }
    
    # Get network data if requested
    network_data = None
    if request.include_network:
        network_data = graph.get_network(entity.id, depth=1, limit=50)
    
    # Score risk
    scorer = RiskScorer()
    assessment = scorer.assess(entity_data, network_data)
    
    return {
        "entity_id": assessment.entity_id,
        "entity_name": assessment.entity_name,
        "overall_score": assessment.overall_score,
        "risk_level": assessment.risk_level.value,
        "summary": assessment.summary,
        "factors": [
            {
                "name": f.name,
                "category": f.category,
                "score": f.score,
                "evidence": f.evidence,
            }
            for f in assessment.factors
        ],
        "recommendations": assessment.recommendations,
    }


@router.get("/stats")
async def get_stats():
    """Get detailed graph statistics."""
    graph = get_graph()
    stats = graph.get_stats()
    
    return {
        "entities": {
            "total": stats.get("total_entities", 0),
            "by_type": stats.get("by_type", {}),
        },
        "relationships": {
            "total": stats.get("total_relationships", 0),
        },
        "backend": stats.get("backend", "unknown"),
    }


@router.post("/ingest")
async def ingest_matrix_data(
    limit: int = 10000,
    targets: List[str] = Query(default=["Vladimir Putin", "Donald Trump", "Elon Musk"]),
    topics: List[str] = Query(default=["role.pep", "sanction"])
):
    """
    Ingest data into the running graph.
    Prioritizes OpenSanctions JSON stream (recovery mode) over Matrix SQLite.
    
    Supports "Smart Ingestion":
    - Scans until 'limit' USEFUL entities are found (not just lines read).
    - Prioritizes PEPs (Politically Exposed Persons) and Sanctioned entities.
    - GUARANTEES ingestion of specific 'targets' if found.
    """
    from pathlib import Path
    from .migration import MatrixMigrator, JsonMigrator
    
    # 1. Check for JSON Source (Preferred - contains keys/relationships)
    json_paths = [
        Path.home() / "Documents" / "iceburg_matrix" / "opensanctions" / "opensanctions_sanctions.json",
        Path("/Users/jackdanger/Documents/iceburg_matrix/opensanctions/opensanctions_sanctions.json"),
    ]
    json_path = next((p for p in json_paths if p.exists()), None)
    
    graph = get_graph()
    
    # Strategy A: Direct JSON Stream
    if json_path:
        try:
            migrator = JsonMigrator(json_path=json_path, graph=graph)
            stats = migrator.migrate(
                limit=limit,
                priority_topics=topics,
                target_names=targets
            )
            return {
                "source": "json_stream",
                "status": "completed",
                "nodes": stats.entities_migrated,
                "scanned": stats.entities_read,
                "edges": stats.relationships_extracted,
                "duration": stats.duration_seconds
            }
        except Exception as e:
            logger.error(f"JSON Ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=f"JSON Ingest Error: {str(e)}")

    # Strategy B: Matrix SQLite (Fallback - nodes only)
    possible_db_paths = [
        Path.home() / "Documents" / "iceburg_matrix" / "matrix.db",
        Path.home() / "Desktop" / "Projects" / "iceburg" / "matrix.db",
        Path("/Users/jackdanger/Desktop/Projects/iceburg/matrix.db"),
    ]
    
    db_path = next((p for p in possible_db_paths if p.exists()), None)
    
    if not db_path:
        raise HTTPException(status_code=404, detail="No valid data source found (JSON or SQLite).")
        
    try:
        migrator = MatrixMigrator(matrix_db_path=db_path, graph=graph)
        stats = migrator.migrate(limit=limit, extract_relationships=True)
        logger.info(f"✅ Ingestion complete: {stats.entities_migrated} entities, {stats.relationships_extracted} relationships")
        return {
            "source": "sqlite_matrix",
            "status": "completed",
            "nodes": stats.entities_migrated,
            "edges": stats.relationships_extracted,
            "duration": stats.duration_seconds
        }
    except Exception as e:
        logger.error(f"SQLite Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/dossier")
async def ingest_dossier(dossier: Dict[str, Any]):
    """
    Ingest a completed Iceberg Dossier into the Colossus Graph.
    This links the investigation results into the broader intelligence network.
    """
    graph = get_graph()
    try:
        if not hasattr(graph, 'ingest_dossier'):
            # Fallback if graph instance wasn't reloaded properly in dev
            logger.warning("Graph instance missing ingest_dossier method")
            return {"status": "error", "message": "Graph capabilities not fully loaded"}
            
        result = graph.ingest_dossier(dossier)
        return {
            "status": "success",
            "message": "Dossier ingested",
            "stats": result
        }
    except Exception as e:
        logger.error(f"Dossier ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-quality")
async def get_data_quality():
    """
    Get data quality metrics for relationships.
    Shows how many relationships are valid (both entities exist).
    """
    from .matrix_store import MatrixStore
    import sqlite3
    
    store = MatrixStore()
    if not store.db_path or not store.db_path.exists():
        return {
            "error": "Matrix database not found",
            "total_relationships": 0,
            "valid_relationships": 0,
            "invalid_relationships": 0,
            "quality_percentage": 0.0
        }
    
    try:
        with store.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total relationships
            cursor.execute("SELECT COUNT(*) FROM relationships")
            total = cursor.fetchone()[0]
            
            # Valid relationships (both entities exist)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM relationships r
                WHERE EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.source_id)
                  AND EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.target_id)
            """)
            valid = cursor.fetchone()[0]
            invalid = total - valid
            quality_pct = (valid / total * 100) if total > 0 else 0.0
            
            return {
                "total_relationships": total,
                "valid_relationships": valid,
                "invalid_relationships": invalid,
                "quality_percentage": round(quality_pct, 2),
                "status": "good" if quality_pct > 90 else "needs_cleanup" if quality_pct < 10 else "fair",
                "recommendation": "cleanup_needed" if quality_pct < 50 else "ok"
            }
    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        return {
            "error": str(e),
            "total_relationships": 0,
            "valid_relationships": 0,
            "invalid_relationships": 0,
            "quality_percentage": 0.0
        }


@router.post("/cleanup")
async def cleanup_relationships():
    """
    Clean relationships table by removing invalid relationships.
    Only keeps relationships where both source and target entities exist.
    """
    from .matrix_store import MatrixStore
    import sqlite3
    from pathlib import Path
    import shutil
    
    store = MatrixStore()
    if not store.db_path or not store.db_path.exists():
        raise HTTPException(status_code=404, detail="Matrix database not found")
    
    try:
        # Create backup
        backup_path = store.db_path.parent / f"{store.db_path.stem}.backup{store.db_path.suffix}"
        shutil.copy2(store.db_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
        with store.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get counts before
            cursor.execute("SELECT COUNT(*) FROM relationships")
            total_before = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) 
                FROM relationships r
                WHERE EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.source_id)
                  AND EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.target_id)
            """)
            valid_before = cursor.fetchone()[0]
            
            # Delete invalid relationships
            cursor.execute("""
                DELETE FROM relationships
                WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = relationships.source_id)
                   OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = relationships.target_id)
            """)
            conn.commit()
            
            deleted = cursor.rowcount
            
            # Get counts after
            cursor.execute("SELECT COUNT(*) FROM relationships")
            total_after = cursor.fetchone()[0]
            
            # Recreate indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_valid ON relationships(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target_valid ON relationships(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type)")
            conn.commit()
            
            return {
                "status": "success",
                "backup_path": str(backup_path),
                "before": {
                    "total": total_before,
                    "valid": valid_before,
                    "invalid": total_before - valid_before
                },
                "after": {
                    "total": total_after,
                    "valid": total_after,
                    "invalid": 0
                },
                "deleted": deleted,
                "quality_percentage": 100.0 if total_after > 0 else 0.0
            }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/diagnostics")
async def get_diagnostics():
    """
    Get diagnostic information about the graph and data sources.
    Helps identify data quality issues and missing relationships.
    """
    graph = get_graph()
    stats = graph.get_stats()
    
    diagnostics = {
        "backend": "neo4j" if graph.is_neo4j else "networkx",
        "entities": {
            "total": stats.get("total_entities", 0),
            "by_type": stats.get("by_type", {}),
        },
        "relationships": {
            "total": stats.get("total_relationships", 0),
        },
        "data_quality": {
            "relationship_ratio": 0.0,
            "status": "unknown",
        },
        "matrix_db": {
            "available": False,
            "relationship_count": 0,
        },
        "sample_relationships": [],
    }
    
    # Calculate relationship ratio
    entity_count = diagnostics["entities"]["total"]
    rel_count = diagnostics["relationships"]["total"]
    if entity_count > 0:
        diagnostics["data_quality"]["relationship_ratio"] = rel_count / entity_count
        if diagnostics["data_quality"]["relationship_ratio"] < 0.1:
            diagnostics["data_quality"]["status"] = "low"
        elif diagnostics["data_quality"]["relationship_ratio"] < 0.5:
            diagnostics["data_quality"]["status"] = "medium"
        else:
            diagnostics["data_quality"]["status"] = "good"
    
    # Check Matrix DB availability
    try:
        from .matrix_store import MatrixStore
        store = MatrixStore()
        if store.db_path and store.db_path.exists():
            diagnostics["matrix_db"]["available"] = True
            # Try to get relationship count from Matrix DB
            try:
                with store.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM relationships")
                    diagnostics["matrix_db"]["relationship_count"] = cursor.fetchone()[0]
            except Exception as e:
                logger.debug(f"Could not query Matrix relationships: {e}")
    except Exception as e:
        logger.debug(f"Matrix DB check failed: {e}")
    
    # Get sample relationships
    if rel_count > 0:
        try:
            # Get a few sample relationships
            sample_entities = graph.get_central_entities(limit=5, centrality_measure="degree")
            for entity, _ in sample_entities[:3]:
                rels = graph.get_relationships(entity.id)
                for rel in rels[:2]:  # Max 2 per entity
                    diagnostics["sample_relationships"].append({
                        "source": entity.name,
                        "target": rel.target_id,
                        "type": rel.relationship_type,
                    })
                    if len(diagnostics["sample_relationships"]) >= 5:
                        break
                if len(diagnostics["sample_relationships"]) >= 5:
                    break
        except Exception as e:
            logger.debug(f"Could not get sample relationships: {e}")
    
    return diagnostics
