"""
ICEBURG Production API Server
FastAPI server for web UI integration with chat, deep, and MoE modes
"""

import asyncio
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ICEBURG imports
from ..config import load_config, IceburgConfig
from ..protocol import iceberg_protocol
from ..model_select import _available_model_names


# =============================================================================
# Configuration & Auth
# =============================================================================

API_KEY = os.getenv("ICEBURG_API_KEY", "w91EEfKN3_x5_rEbdcEV_m5kipSrS4fqS_tP9-DgCTo")
CORS_ORIGINS = os.getenv("ICEBURG_CORS_ORIGINS", f"http://localhost:8081,http://localhost:3000").split(",")

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify Bearer token matches API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    mode: str = "chat"  # chat | deep | moe
    verbose: bool = False


class QueryResponse(BaseModel):
    id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: str
    processing_time: float
    mode: str


class AgentInfo(BaseModel):
    name: str
    status: str
    type: str
    model: Optional[str] = None


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="ICEBURG API",
    description="Production API for ICEBURG multi-agent research system",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime tracking
START_TIME = datetime.now()


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Basic liveness check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status/v1")
async def status_v1(authenticated: bool = Depends(verify_api_key)):
    """System status with config details"""
    cfg = load_config()
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return {
        "system": {
            "health": "OPERATIONAL",
            "uptime": f"{int(uptime)}s",
            "profile": "production",
            "routing_profile": "standard",
            "model": cfg.surveyor_model,
            "fast": cfg.fast,
        },
        "agents": {
            "active": 8,  # Core agents: surveyor, dissident, synthesist, oracle, etc.
            "queries_processed": 0,  # Could track with counter
        },
        "queues": {
            "pending": 0,
            "running": 0,
            "completed": 0,
        },
        "emergence": {
            "breakthroughs_detected": 0,
            "recent": [],
        },
        "cache": {
            "enabled": True,
            "hits": 0,
            "misses": 0,
        },
    }


@app.get("/api/v1/agents")
async def list_agents(authenticated: bool = Depends(verify_api_key)):
    """List available agents"""
    cfg = load_config()
    
    agents = [
        {"name": "Surveyor", "status": "ONLINE", "type": "consensus", "model": cfg.surveyor_model},
        {"name": "Dissident", "status": "ONLINE", "type": "challenger", "model": cfg.dissident_model},
        {"name": "Synthesist", "status": "ONLINE", "type": "synthesis", "model": cfg.synthesist_model},
        {"name": "Oracle", "status": "ONLINE", "type": "meta_analyst", "model": cfg.oracle_model},
        {"name": "Scrutineer", "status": "ONLINE", "type": "evidence_evaluator", "model": cfg.surveyor_model},
        {"name": "Scribe", "status": "ONLINE", "type": "knowledge_synthesizer", "model": cfg.oracle_model},
        {"name": "Archaeologist", "status": "ONLINE", "type": "buried_truth", "model": cfg.dissident_model},
        {"name": "Weaver", "status": "ONLINE", "type": "code_generator", "model": cfg.oracle_model},
    ]
    
    return {"agents": agents}


# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Submit query for processing (synchronous)
    Best for 'deep' mode; chat/moe should use streaming endpoint
    """
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Route based on mode - run in thread to avoid event loop conflicts
        if request.mode == "deep":
            result = await asyncio.to_thread(
                iceberg_protocol,
                request.query,
                verbose=request.verbose,
                fast=False,
                hybrid=False,
            )
        elif request.mode == "chat":
            result = await asyncio.to_thread(
                iceberg_protocol,
                request.query,
                verbose=request.verbose,
                fast=True,
                hybrid=False,
            )
        elif request.mode == "moe":
            result = await asyncio.to_thread(
                iceberg_protocol,
                request.query,
                verbose=request.verbose,
                fast=False,
                hybrid=True,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            id=query_id,
            status="completed",
            result=result,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            mode=request.mode,
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        return QueryResponse(
            id=query_id,
            status="failed",
            error=str(e),
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            mode=request.mode,
        )


@app.get("/api/v1/query/stream")
async def stream_query(
    query: str,
    mode: str = "chat",
    api_key: Optional[str] = None,
):
    """
    Stream query results via Server-Sent Events (SSE)
    Progressively sends tokens for chat/moe modes
    """
    # Verify API key from query param (for SSE compatibility)
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    async def generate_stream():
        """Generate SSE stream of tokens"""
        try:
            # Run ICEBURG protocol in background
            if mode == "chat":
                result = await asyncio.to_thread(
                    iceberg_protocol,
                    query,
                    verbose=False,
                    fast=True,
                    hybrid=False,
                )
            elif mode == "moe":
                result = await asyncio.to_thread(
                    iceberg_protocol,
                    query,
                    verbose=False,
                    fast=False,
                    hybrid=True,
                )
            else:
                result = await asyncio.to_thread(
                    iceberg_protocol,
                    query,
                    verbose=False,
                    fast=True,
                    hybrid=False,
                )
            
            # Simulate streaming by chunking result
            # (Native provider streaming can be added later)
            words = result.split()
            for i, word in enumerate(words):
                # Send token
                yield f"data: {word}\n\n"
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
            
            # Send completion
            yield "event: done\ndata: Stream complete\n\n"
            
        except Exception as e:
            # Send error event
            yield f"event: error\ndata: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
        },
    )


# =============================================================================
# Visual Generation Endpoints
# =============================================================================

class VisualGenerationRequest(BaseModel):
    description: str
    backends: List[str] = ["all"]
    project_id: str = "default"

@app.post("/api/v1/visual/generate")
async def generate_visual(
    request: VisualGenerationRequest,
    auth: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Generate UI from description"""
    try:
        from ..agents.visual_architect import VisualArchitect
        from ..storage.visual_storage import VisualArtifactStorage
        from ..iir.backends import BackendType
        
        visual_architect = VisualArchitect()
        cfg = load_config() # Load config for VisualArchitect
        
        # Convert backends
        if "all" in request.backends:
            targets = list(BackendType)
        else:
            targets = [BackendType(b) for b in request.backends if b in [e.value for e in BackendType]]
        
        # Generate
        result = visual_architect.run(cfg, request.description, verbose=False)
        
        # Store
        storage = VisualArtifactStorage()
        artifact_id = storage.store_visual_generation(result, request.project_id)
        
        return {
            "artifact_id": artifact_id,
            "spec": result.spec.to_dict(),
            "backends_generated": [b.value for b in result.artifacts.keys()],
            "validation": {
                "passed": result.validation.passed,
                "violations_count": len(result.validation.violations)
            },
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/visual/{artifact_id}/preview")
async def preview_visual(
    artifact_id: str,
    backend: str = "html5",
    project_id: str = "default",
    auth: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get preview for generated artifact"""
    try:
        from ..storage.visual_storage import VisualArtifactStorage
        from ..iir.backends import BackendType
        
        storage = VisualArtifactStorage()
        artifact = storage.load_artifact(artifact_id, project_id)
        
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        backend_type = BackendType(backend)
        
        if backend in artifact.get("backends", {}):
            # For HTML5, return preview HTML
            if backend == "html5":
                preview_path = artifact["backends"][backend] / "index.html"
                if preview_path.exists():
                    return {"preview_html": preview_path.read_text()}
                else:
                    raise HTTPException(status_code=404, detail="Preview file not found")
            else:
                return {"message": f"Preview not directly available for {backend}. Check artifacts."}
        else:
            raise HTTPException(status_code=404, detail=f"Backend {backend} not available for this artifact")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Multi-Provider Chat Endpoints (User API Keys)
# =============================================================================

class MultiProviderChatRequest(BaseModel):
    """Request for multi-provider chat with user-provided API key"""
    message: str
    provider: str  # openai, anthropic, google, xai, deepseek, hybrid
    api_key: str  # User's API key for the provider
    model: Optional[str] = None  # Optional model override
    system: Optional[str] = None  # Optional system prompt
    temperature: float = 0.2
    max_tokens: int = 4096


class MultiProviderChatResponse(BaseModel):
    """Response from multi-provider chat"""
    content: str
    provider: str
    model: str
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@app.post("/api/v1/chat/multi", response_model=MultiProviderChatResponse)
async def multi_provider_chat(request: MultiProviderChatRequest):
    """
    Chat with any AI provider using user-provided API key.
    
    Supports: openai (GPT), anthropic (Claude), google (Gemini), xai (Grok), deepseek
    
    Note: No auth required - user provides their own API key.
    """
    try:
        from ..providers.multi_provider_router import (
            get_router, Provider, ProviderConfig, ChatRequest
        )
        
        router = get_router()
        
        # Map provider string to enum
        provider_map = {
            "openai": Provider.OPENAI,
            "gpt": Provider.OPENAI,
            "anthropic": Provider.ANTHROPIC,
            "claude": Provider.ANTHROPIC,
            "google": Provider.GOOGLE,
            "gemini": Provider.GOOGLE,
            "xai": Provider.XAI,
            "grok": Provider.XAI,
            "deepseek": Provider.DEEPSEEK,
        }
        
        provider_name = request.provider.lower()
        if provider_name not in provider_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {request.provider}. Supported: openai, anthropic, google, xai, deepseek"
            )
        
        provider_enum = provider_map[provider_name]
        config = ProviderConfig(
            api_key=request.api_key,
            model=request.model,
        )
        chat_request = ChatRequest(
            message=request.message,
            system=request.system,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # Run chat in thread pool to avoid blocking
        response = await asyncio.to_thread(
            router.chat,
            provider_enum,
            config,
            chat_request,
        )
        
        return MultiProviderChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model,
            success=response.success,
            error=response.error,
            latency_ms=response.latency_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return MultiProviderChatResponse(
            content="",
            provider=request.provider,
            model=request.model or "",
            success=False,
            error=str(e),
        )


@app.post("/api/v1/chat/hybrid")
async def hybrid_provider_chat(
    message: str,
    providers: List[Dict[str, str]],  # [{"provider": "claude", "api_key": "sk-..."}]
    system: Optional[str] = None,
):
    """
    Query multiple providers in parallel (hybrid mode).
    
    Returns responses from all providers for comparison/synthesis.
    """
    try:
        from ..providers.multi_provider_router import (
            get_router, Provider, ProviderConfig, ChatRequest
        )
        
        router = get_router()
        
        provider_map = {
            "openai": Provider.OPENAI,
            "gpt": Provider.OPENAI,
            "anthropic": Provider.ANTHROPIC,
            "claude": Provider.ANTHROPIC,
            "google": Provider.GOOGLE,
            "gemini": Provider.GOOGLE,
            "xai": Provider.XAI,
            "grok": Provider.XAI,
            "deepseek": Provider.DEEPSEEK,
        }
        
        # Build provider configs
        provider_configs = []
        for p in providers:
            provider_name = p.get("provider", "").lower()
            api_key = p.get("api_key", "")
            model = p.get("model")
            
            if provider_name not in provider_map:
                continue
            
            provider_configs.append((
                provider_map[provider_name],
                ProviderConfig(api_key=api_key, model=model)
            ))
        
        if not provider_configs:
            raise HTTPException(status_code=400, detail="No valid providers specified")
        
        chat_request = ChatRequest(message=message, system=system)
        
        # Run hybrid chat
        responses = await router.chat_hybrid(provider_configs, chat_request)
        
        return {
            "responses": [
                {
                    "content": r.content,
                    "provider": r.provider,
                    "model": r.model,
                    "success": r.success,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                }
                for r in responses
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main (for direct execution)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting ICEBURG Production API Server")
    print("=" * 60)
    print(f"API Key: {API_KEY}")
    print(f"CORS Origins: {CORS_ORIGINS}")
    print(f"Port: 8083")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /status/v1")
    print("  GET  /api/v1/agents")
    print("  POST /api/v1/query")
    print("  GET  /api/v1/query/stream")
    print("=" * 60)
    print("\nUI: http://localhost:8081/iceburg-live-interface.html")
    print("    (Start UI: cd web && python -m http.server 8081)")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8083,
        log_level="info",
    )

