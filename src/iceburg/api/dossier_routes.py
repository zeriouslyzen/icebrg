"""
Dossier Protocol API Routes
Endpoints for dossier generation and retrieval.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import asyncio

from ..config import load_config
from ..protocols.dossier import DossierSynthesizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dossier", tags=["dossier"])


class DossierRequest(BaseModel):
    """Request body for dossier generation."""
    query: str
    depth: str = "standard"  # 'quick', 'standard', 'deep'
    format: str = "full"  # 'full', 'markdown', 'json'


class DossierResponse(BaseModel):
    """Response containing the generated dossier."""
    success: bool
    query: str
    dossier: Optional[Dict[str, Any]] = None
    markdown: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


@router.post("", response_model=DossierResponse)
@router.post("/", response_model=DossierResponse)
async def generate_dossier_endpoint(request: DossierRequest):
    """
    Generate an ICEBURG Dossier for a query.
    
    This runs the full Dossier Protocol:
    1. Gatherer - Multi-source intelligence collection
    2. Decoder - Symbol/pattern analysis
    3. Mapper - Network relationship mapping
    4. Synthesizer - Final dossier compilation
    """
    try:
        logger.info(f"🔍 Dossier request: {request.query} (depth={request.depth})")
        
        # Load config from environment
        cfg = load_config()
        
        # Generate dossier in thread pool (CPU-bound work)
        synthesizer = DossierSynthesizer(cfg)
        
        dossier = await asyncio.to_thread(
            synthesizer.generate_dossier,
            query=request.query,
            depth=request.depth
        )
        
        # Prepare response based on format
        response = DossierResponse(
            success=True,
            query=request.query,
            metadata=dossier.metadata
        )
        
        if request.format in ["full", "json"]:
            response.dossier = dossier.to_dict()
        
        if request.format in ["full", "markdown"]:
            response.markdown = dossier.to_markdown()
        
        logger.info(f"✅ Dossier generated: {dossier.metadata.get('total_sources', 0)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Dossier generation failed: {e}", exc_info=True)
        return DossierResponse(
            success=False,
            query=request.query,
            error=str(e)
        )


@router.get("/status")
async def dossier_status():
    """Check dossier protocol status."""
    return {
        "status": "operational",
        "protocol": "dossier",
        "version": "1.0.0",
        "agents": ["gatherer", "decoder", "mapper", "synthesizer"],
        "depths_available": ["quick", "standard", "deep"]
    }
