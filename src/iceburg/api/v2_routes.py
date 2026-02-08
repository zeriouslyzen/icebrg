"""
ICEBURG V2 API Routes
======================
Unified API endpoint for the simplified ICEBURG consumer product.

This provides a single /v2/query endpoint that handles:
- Fast mode (cloud API for instant responses)
- Research mode (multi-agent protocol)
- Encyclopedia mode (knowledge lookup)
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import asyncio
import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["v2"])


# Request/Response Models
class V2QueryRequest(BaseModel):
    """Unified query request for all ICEBURG modes"""
    query: str = Field(..., description="User query text", min_length=1, max_length=10000)
    mode: str = Field(default="fast", description="Query mode: 'fast', 'research', 'local', 'encyclopedia'")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    stream: bool = Field(default=True, description="Whether to stream the response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2000, ge=100, le=16000, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


class V2QueryResponse(BaseModel):
    """Response for non-streaming queries"""
    response: str
    mode: str
    provider: str
    conversation_id: str
    latency_ms: float
    usage: Optional[Dict[str, Any]] = None


@router.post("/query")
async def v2_query(request: V2QueryRequest, http_request: Request):
    """
    Unified query endpoint for all ICEBURG modes.
    
    Modes:
    - **fast**: Cloud API (Claude/Gemini) for instant responses (< 1s)
    - **research**: Multi-agent protocol for deep analysis (2-5 min)
    - **local**: Local Ollama for privacy-focused queries
    - **encyclopedia**: Query the Celestial Encyclopedia
    
    Supports SSE streaming when `stream=True` (default).
    """
    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"📡 V2 Query: mode={request.mode}, stream={request.stream}, query='{request.query[:50]}...'")
    
    # Handle encyclopedia mode specially
    if request.mode == "encyclopedia":
        return await _handle_encyclopedia(request, conversation_id)
    
    # For research mode, delegate to existing multi-agent system
    if request.mode == "research":
        return await _handle_research(request, http_request, conversation_id, start_time)
    
    # For fast/local modes, use unified provider
    if request.stream:
        return StreamingResponse(
            _stream_response(request, conversation_id, start_time),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conversation_id,
            }
        )
    else:
        return await _non_streaming_response(request, conversation_id, start_time)


async def _stream_response(request: V2QueryRequest, conversation_id: str, start_time: float):
    """Stream response using unified provider"""
    from ..providers.unified_provider import get_unified_provider
    
    provider = get_unified_provider()
    
    # Send initial status
    yield f"data: {json.dumps({'type': 'status', 'content': 'Connecting...', 'timestamp': time.time()})}\n\n"
    
    try:
        # Get response from unified provider
        response = await provider.complete(
            prompt=request.query,
            mode=request.mode,
            system=request.system_prompt or _get_default_system_prompt(request.mode),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # Stream the response in chunks
        chunk_size = 15  # Characters per chunk for smooth streaming
        total_chunks = (len(response) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.02)  # Smooth streaming effect
        
        # Send completion
        latency_ms = (time.time() - start_time) * 1000
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'latency_ms': latency_ms})}\n\n"
        
        logger.info(f"✅ V2 Query complete: {latency_ms:.0f}ms, {len(response)} chars")
        
    except Exception as e:
        logger.error(f"❌ V2 Query error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e), 'timestamp': time.time()})}\n\n"


async def _non_streaming_response(request: V2QueryRequest, conversation_id: str, start_time: float) -> V2QueryResponse:
    """Non-streaming response"""
    from ..providers.unified_provider import get_unified_provider
    
    provider = get_unified_provider()
    
    try:
        response = await provider.complete(
            prompt=request.query,
            mode=request.mode,
            system=request.system_prompt or _get_default_system_prompt(request.mode),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return V2QueryResponse(
            response=response,
            mode=request.mode,
            provider=provider.fast_provider_name if request.mode == "fast" else "ollama",
            conversation_id=conversation_id,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        logger.error(f"❌ V2 Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_encyclopedia(request: V2QueryRequest, conversation_id: str):
    """Handle encyclopedia queries"""
    import os
    from pathlib import Path
    
    try:
        # Load encyclopedia data
        data_dir = Path(os.getcwd()) / "data"
        encyclopedia_file = data_dir / "celestial_encyclopedia.json"
        
        if not encyclopedia_file.exists():
            return JSONResponse(content={
                "results": [],
                "query": request.query,
                "message": "Encyclopedia not available"
            })
        
        with open(encyclopedia_file, "r") as f:
            encyclopedia = json.load(f)
        
        # Simple keyword search
        query_lower = request.query.lower()
        entries = encyclopedia.get("entries", [])
        results = [
            entry for entry in entries
            if query_lower in entry.get("title", "").lower() 
            or query_lower in entry.get("content", "").lower()
        ][:10]  # Limit to 10 results
        
        return JSONResponse(content={
            "results": results,
            "query": request.query,
            "count": len(results),
            "conversation_id": conversation_id,
        })
        
    except Exception as e:
        logger.error(f"Encyclopedia error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_research(request: V2QueryRequest, http_request: Request, conversation_id: str, start_time: float):
    """Delegate to existing research protocol"""
    # Import the existing protocol
    try:
        from ..protocol_fixed import run as run_protocol
        from ..config import load_config
        
        cfg = load_config()
        
        if request.stream:
            async def research_stream():
                yield f"data: {json.dumps({'type': 'status', 'content': '🔬 Initializing research protocol...', 'timestamp': time.time()})}\n\n"
                
                try:
                    # Run protocol (this is synchronous, wrap in executor)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: run_protocol(cfg, request.query, verbose=False)
                    )
                    
                    # Stream the result
                    response_text = str(result)
                    chunk_size = 50
                    for i in range(0, len(response_text), chunk_size):
                        chunk = response_text[i:i + chunk_size]
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk, 'timestamp': time.time()})}\n\n"
                        await asyncio.sleep(0.01)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'latency_ms': latency_ms})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Research protocol error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e), 'timestamp': time.time()})}\n\n"
            
            return StreamingResponse(
                research_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Conversation-ID": conversation_id,
                }
            )
        else:
            # Non-streaming research
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_protocol(cfg, request.query, verbose=False)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            return V2QueryResponse(
                response=str(result),
                mode="research",
                provider="multi-agent",
                conversation_id=conversation_id,
                latency_ms=latency_ms,
            )
            
    except ImportError as e:
        logger.error(f"Could not import research protocol: {e}")
        raise HTTPException(status_code=500, detail="Research mode not available")


def _get_default_system_prompt(mode: str) -> str:
    """Get default system prompt for mode"""
    prompts = {
        "fast": "You are ICEBURG, a helpful AI assistant. Be concise, accurate, and helpful.",
        "local": "You are ICEBURG, a helpful AI assistant running locally. Be concise and helpful.",
        "research": "You are ICEBURG, a deep research AI. Provide comprehensive, well-sourced analysis.",
    }
    return prompts.get(mode, prompts["fast"])


# Health check endpoint
@router.get("/health")
async def v2_health():
    """V2 API health check with provider status"""
    from ..providers.unified_provider import get_unified_provider
    
    provider = get_unified_provider()
    
    return {
        "status": "ok",
        "version": "v2",
        "available_providers": provider.get_available_providers(),
        "fast_provider": provider.fast_provider_name,
        "research_provider": provider.research_provider_name,
        "cost_summary": provider.get_cost_summary(),
    }


# Provider configuration endpoint
@router.get("/providers")
async def v2_providers():
    """Get available providers and their configuration"""
    from ..providers.unified_provider import get_unified_provider
    
    provider = get_unified_provider()
    
    return {
        "available": provider.get_available_providers(),
        "fast_provider": provider.fast_provider_name,
        "research_provider": provider.research_provider_name,
        "fallback_chain": provider.fallback_chain,
    }
