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
    """Search entities in the graph."""
    graph = get_graph()
    
    entities = graph.search_entities(
        query=request.query,
        entity_type=request.entity_type,
        limit=request.limit,
    )
    
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


@router.post("/network")
async def get_network(request: NetworkRequest):
    """Get entity network."""
    graph = get_graph()
    
    network = graph.get_network(
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
