"""
Investigation API Routes - REST endpoints for investigation management.
Provides CRUD operations, PDF export, and archive browsing.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/investigations", tags=["investigations"])


# Pydantic models for API
class InvestigationSummary(BaseModel):
    """Summary of an investigation for listing."""
    investigation_id: str
    title: str
    query: str
    created_at: str
    updated_at: str
    status: str
    tags: List[str]
    confidence_score: float
    sources_count: int


class InvestigationDetail(BaseModel):
    """Full investigation details."""
    investigation_id: str
    title: str
    query: str
    created_at: str
    updated_at: str
    status: str
    tags: List[str]
    confidence_score: float
    sources_count: int
    entities_count: int
    depth: str
    executive_summary: str
    official_narrative: str
    alternative_narratives: List[dict]
    key_players: List[dict]
    dossier_markdown: str


class UpdateStatusRequest(BaseModel):
    """Request to update investigation status."""
    status: str  # active, archived, flagged


@router.get("/", response_model=List[InvestigationSummary])
async def list_investigations(
    status: Optional[str] = Query(None, description="Filter by status (active, archived, flagged)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results")
):
    """List all investigations, optionally filtered by status."""
    try:
        from ..investigations import get_investigation_store
        
        store = get_investigation_store()
        investigations = store.list_all(status=status, limit=limit)
        
        return [InvestigationSummary(**inv) for inv in investigations]
    except Exception as e:
        logger.error(f"Failed to list investigations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_investigations(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50)
):
    """Search investigations by text."""
    try:
        from ..investigations import get_investigation_store
        
        store = get_investigation_store()
        results = store.search(q, limit=limit)
        
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{investigation_id}")
async def get_investigation(investigation_id: str):
    """Get full investigation details."""
    try:
        from ..investigations import get_investigation_store
        
        store = get_investigation_store()
        investigation = store.load(investigation_id)
        
        if investigation is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        
        return {
            "investigation_id": investigation.metadata.investigation_id,
            "title": investigation.metadata.title,
            "query": investigation.metadata.query,
            "created_at": investigation.metadata.created_at,
            "updated_at": investigation.metadata.updated_at,
            "status": investigation.metadata.status,
            "tags": investigation.metadata.tags,
            "confidence_score": investigation.metadata.confidence_score,
            "sources_count": investigation.metadata.sources_count,
            "entities_count": investigation.metadata.entities_count,
            "depth": investigation.metadata.depth,
            "executive_summary": investigation.executive_summary,
            "official_narrative": investigation.official_narrative,
            "alternative_narratives": investigation.alternative_narratives,
            "key_players": investigation.key_players,
            "contradictions": investigation.contradictions,
            "historical_parallels": investigation.historical_parallels,
            "network_graph": investigation.network_graph,
            "follow_up_suggestions": investigation.follow_up_suggestions,
            "dossier_markdown": investigation.dossier_markdown
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get investigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{investigation_id}/markdown")
async def get_investigation_markdown(investigation_id: str):
    """Get raw markdown for an investigation."""
    try:
        from ..investigations import get_investigation_store
        
        store = get_investigation_store()
        investigation = store.load(investigation_id)
        
        if investigation is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        
        return JSONResponse(
            content={"markdown": investigation.dossier_markdown},
            media_type="application/json"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{investigation_id}/pdf")
async def download_investigation_pdf(investigation_id: str):
    """Download investigation as styled PDF."""
    try:
        from ..investigations import get_investigation_store, export_investigation_to_pdf, REPORTLAB_AVAILABLE
        
        if not REPORTLAB_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="PDF export unavailable. Install reportlab: pip install reportlab"
            )
        
        store = get_investigation_store()
        investigation = store.load(investigation_id)
        
        if investigation is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        
        # Generate PDF
        pdf_path = export_investigation_to_pdf(investigation_id)
        
        if pdf_path is None or not pdf_path.exists():
            raise HTTPException(status_code=500, detail="PDF generation failed")
        
        return FileResponse(
            path=str(pdf_path),
            filename=f"iceburg_dossier_{investigation_id}.pdf",
            media_type="application/pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{investigation_id}/status")
async def update_investigation_status(investigation_id: str, request: UpdateStatusRequest):
    """Update investigation status (active, archived, flagged)."""
    try:
        from ..investigations import get_investigation_store
        
        valid_statuses = ["active", "archived", "flagged"]
        if request.status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        store = get_investigation_store()
        success = store.update_status(investigation_id, request.status)
        
        if not success:
            raise HTTPException(status_code=404, detail="Investigation not found")
        
        return {"investigation_id": investigation_id, "status": request.status, "updated": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{investigation_id}")
async def archive_investigation(investigation_id: str):
    """Archive an investigation (soft delete)."""
    try:
        from ..investigations import get_investigation_store
        
        store = get_investigation_store()
        success = store.delete(investigation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Investigation not found")
        
        return {"investigation_id": investigation_id, "status": "archived", "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Archive failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _network_graph_to_nodes_links(network_graph: dict) -> dict:
    """Normalize network_graph: if it has entities/relationships, add nodes and links for Pegasus."""
    if not network_graph:
        return network_graph
    if network_graph.get("nodes") and network_graph.get("links"):
        return network_graph
    if network_graph.get("nodes") and network_graph.get("edges"):
        return {**network_graph, "links": network_graph["edges"]}
    entities = network_graph.get("entities") or []
    relationships = network_graph.get("relationships") or []
    if not entities and not relationships:
        return network_graph
    nodes = []
    for e in entities:
        n = {
            "id": e.get("id", ""),
            "name": e.get("name", e.get("label", "")),
            "type": e.get("type", e.get("entity_type", "unknown")),
        }
        if e.get("domains") is not None:
            n["domains"] = e["domains"]
        if e.get("roles") is not None:
            n["roles"] = e["roles"]
        if e.get("themes") is not None:
            n["themes"] = e["themes"]
        nodes.append(n)
    links = []
    for r in relationships:
        link = {
            "source": r.get("source_id", r.get("source", "")),
            "target": r.get("target_id", r.get("target", "")),
            "type": r.get("relationship_type", r.get("type", "RELATED_TO")),
        }
        if r.get("domain") is not None:
            link["domain"] = r["domain"]
        links.append(link)
    return {**network_graph, "nodes": nodes, "links": links, "edges": links}


@router.get("/{investigation_id}/network")
async def get_network_graph(investigation_id: str):
    """Get network graph data for visualization (nodes/links for Pegasus)."""
    try:
        from ..investigations import get_investigation_store

        store = get_investigation_store()
        investigation = store.load(investigation_id)

        if investigation is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

        network_graph = _network_graph_to_nodes_links(investigation.network_graph or {})
        return {
            "investigation_id": investigation_id,
            "network_graph": network_graph,
            "key_players": investigation.key_players,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get network graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
