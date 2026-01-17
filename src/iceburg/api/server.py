"""
ICEBURG API Server
FastAPI server with WebSocket support
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime
import logging
import os
import time
import uuid

from ..core.system_integrator import SystemIntegrator
from ..formatting.response_formatter import ResponseFormatter
from ..services.streaming_handler import StreamingHandler
from ..telemetry import AdvancedTelemetry, PromptMetrics
from ..storage.local_persistence import LocalPersistence, ConversationEntry, ResearchEntry
from ..agents.reflex_agent import ReflexAgent
from ..utils.user_friendly_names import (
    get_user_friendly_name,
    format_thinking_message,
    format_action_message,
    get_context_message
)
from .routes import router
from .security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    sanitize_input,
    validate_query,
    validate_file_upload
)

logger = logging.getLogger(__name__)

# PHASE 2.1: Security event logger for audit trail
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)

# Create security log file handler
try:
    import pathlib
    security_log_dir = pathlib.Path.home() / "Documents" / "iceburg_data"
    security_log_dir.mkdir(parents=True, exist_ok=True)
    security_log_file = security_log_dir / "security.log"
    
    security_file_handler = logging.FileHandler(security_log_file)
    security_file_handler.setLevel(logging.INFO)
    security_file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    security_logger.addHandler(security_file_handler)
    security_logger.info("Security logging initialized")
except Exception as e:
    logger.warning(f"Could not initialize security logging: {e}")

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize object to ensure JSON serializability"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj if not isinstance(item, slice)]
    elif isinstance(obj, slice):
        # Convert slice to string representation
        return f"slice({obj.start}, {obj.stop}, {obj.step})"
    else:
        # Try to convert to string
        try:
            return str(obj)
        except:
            return None

# Create FastAPI app (will be configured by create_app())
app = FastAPI(
    title="ICEBURG 2.0 API",
    description="Ultimate Truth-Finding AI Civilization API",
    version="2.0.0"
)

# Security middleware (must be added first)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

# CORS middleware - PHASE 1.2: Always use whitelist, never wildcard
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # Default allowed origins for development (including mobile network access)
    # Get local network IP for mobile access
    import socket
    local_ip = None
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    
    allowed_origins = [
        "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173",
        "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002", "http://127.0.0.1:5173"
    ]
    
    # Add network IP origins for mobile access
    if local_ip:
        allowed_origins.extend([
            f"http://{local_ip}:3000",
            f"http://{local_ip}:3001",
            f"http://{local_ip}:3002",
            f"http://{local_ip}:5173"
        ])
    
    logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Always use whitelist, never wildcard
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Origin", "Upgrade", "Connection", "Sec-WebSocket-Key", "Sec-WebSocket-Version", "Sec-WebSocket-Protocol"],
    expose_headers=["X-Request-ID"],
)

# Initialize system components
system_integrator = SystemIntegrator()
response_formatter = ResponseFormatter()
streaming_handler = StreamingHandler()
telemetry = AdvancedTelemetry()
reflex_agent = ReflexAgent()

# Initialize Semantic Cache (Edge Browsing Optimization)
from ..caching.semantic_cache import SemanticCache
from ..llm import embed_texts # For generating embeddings
semantic_cache = SemanticCache(similarity_threshold=0.85)

# Warm cache in background
try:
    transcript_path = os.path.join(os.getcwd(), "fast_mode_transcript.md")
    if os.path.exists(transcript_path):
        semantic_cache.warm_from_transcript(transcript_path)
except Exception as e:
    logger.warning(f"Failed to warm semantic cache: {e}")

# Initialize local persistence (similar to browser storage but for long-term)
local_persistence = LocalPersistence()

# Initialize global agent middleware (hallucination detection & emergence tracking)
global_middleware = None
try:
    from ..middleware.global_agent_middleware import GlobalAgentMiddleware
    from ..config import load_config
    middleware_cfg = load_config()
    global_middleware = GlobalAgentMiddleware(middleware_cfg)
    logger.info("✅ Global agent middleware initialized")
except Exception as e:
    logger.warning(f"Could not initialize global middleware: {e}. Continuing without middleware.")

# Initialize bottleneck monitoring and self-healing (disabled by default - can cause errors)
# Set ICEBURG_ENABLE_MONITORING=1 to enable (may cause evolution pipeline errors)
ENABLE_MONITORING = os.getenv("ICEBURG_ENABLE_MONITORING", "0") == "1"
bottleneck_monitor = None
if ENABLE_MONITORING:
    try:
        from ..monitoring.bottleneck_detector import create_bottleneck_monitor
        from ..config import load_config
        monitor_config = {
            "latency_threshold": 1000.0,
            "memory_threshold": 80.0,
            "cpu_threshold": 80.0,
            "cache_threshold": 70.0,
            "network_threshold": 50.0,
            "disk_threshold": 90.0,
            "max_history": 1000
        }
        cfg = load_config()
        # Create monitor asynchronously (don't block startup)
        async def init_monitoring():
            try:
                monitor = await create_bottleneck_monitor(monitor_config)
                monitor.cfg = cfg
                await monitor.start_monitoring()
                return monitor
            except Exception as e:
                logger.error(f"Failed to initialize monitoring: {e}")
                return None
        
        # Start monitoring in background (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task for background initialization
                async def init_task():
                    global bottleneck_monitor
                    bottleneck_monitor = await init_monitoring()
                asyncio.create_task(init_task())
            else:
                bottleneck_monitor = asyncio.run(init_monitoring())
        except Exception as e:
            logger.warning(f"Could not start monitoring: {e}")
            bottleneck_monitor = None
        logger.info("Bottleneck monitoring and self-healing initialization started (background)")
    except Exception as e:
        logger.warning(f"Failed to initialize bottleneck monitoring: {e}")
        bottleneck_monitor = None

# Initialize system health monitor
system_health_monitor = None
if os.getenv("ICEBURG_ENABLE_HEALTH_MONITORING", "1") == "1":
    try:
        from ..monitoring.system_health_monitor import get_system_health_monitor
        from ..config import load_config
        
        health_config = {
            "check_interval": 5,
            "health_score_threshold": 70,
            "alert_on_critical": True
        }
        system_health_monitor = get_system_health_monitor(health_config)
        
        # Start monitoring in background
        async def start_health_monitoring():
            try:
                await system_health_monitor.start_monitoring()
                logger.info("System health monitoring started")
            except Exception as e:
                logger.error(f"Failed to start health monitoring: {e}")
        
        # Start monitoring asynchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(start_health_monitoring())
            else:
                loop.run_until_complete(start_health_monitoring())
        except Exception as e:
            logger.warning(f"Could not start health monitoring: {e}")
            system_health_monitor = None
    except Exception as e:
        logger.warning(f"Failed to initialize system health monitoring: {e}")
        system_health_monitor = None

# Initialize always-on AI components (disabled by default for performance - only enable if needed)
# Set ICEBURG_ENABLE_ALWAYS_ON=1 to enable (adds startup delay, uses more memory)
ENABLE_ALWAYS_ON = os.getenv("ICEBURG_ENABLE_ALWAYS_ON", "0") == "1"
# Pre-warmed agents disabled by default - only warm up agents on-demand
# Set ICEBURG_ENABLE_PRE_WARMED=1 to enable (warms up ALL agents at startup - slow)
ENABLE_PRE_WARMED = os.getenv("ICEBURG_ENABLE_PRE_WARMED", "0") == "1"
ENABLE_LOCAL_PERSONA = os.getenv("ICEBURG_ENABLE_LOCAL_PERSONA", "0") == "1"

portal = None
if ENABLE_ALWAYS_ON or ENABLE_PRE_WARMED or ENABLE_LOCAL_PERSONA:
    try:
        from ..core.iceburg_portal import ICEBURGPortal
        from ..config import load_config
        # Load config for portal
        portal_config = load_config()
        portal = ICEBURGPortal(portal_config)
        logger.info("ICEBURG Portal created (will initialize at startup)")
    except Exception as e:
        logger.warning(f"Could not create ICEBURG Portal: {e}. Continuing without always-on features.")
        portal = None
else:
    logger.info("ICEBURG Portal disabled (always-on AI disabled via env vars)")

# Include additional routes
app.include_router(router)

# Admin routes
try:
    from .admin_routes import router as admin_router
    app.include_router(admin_router)
    logger.info("Admin API routes registered")
except Exception as e:
    logger.warning(f"Could not register Admin API routes: {e}")

# Civilization routes (AGI Civilization simulation)
try:
    from .civilization_routes import router as civilization_router
    app.include_router(civilization_router)
    logger.info("Civilization API routes registered")
except Exception as e:
    logger.warning(f"Could not register Civilization API routes: {e}")


# V10: Finance & Prediction Dashboard
try:
    from ..monitoring.finance_controller import router as finance_router
    app.include_router(finance_router)
    logger.info("Finance & Prediction Dashboard API registered")
except Exception as e:
    logger.warning(f"Could not register Finance Dashboard API: {e}")

# V2-V10 Bridge: Intelligence → Finance Integration
try:
    from ..monitoring.intelligence_bridge import router as intelligence_bridge_router
    app.include_router(intelligence_bridge_router)
    logger.info("V2-V10 Intelligence Bridge API registered")
except Exception as e:
    logger.warning(f"Could not register Intelligence Bridge API: {e}")

# V2: Advanced Intelligence & Prediction System
try:
    from ..monitoring.intelligence_controller import router as intelligence_v2_router
    app.include_router(intelligence_v2_router)
    logger.info("V2 Intelligence System API registered")
except Exception as e:
    logger.warning(f"Could not register V2 Intelligence API: {e}")

# V2: Event Prediction System
try:
    from ..monitoring.prediction_controller import router as prediction_v2_router
    app.include_router(prediction_v2_router)
    logger.info("V2 Prediction System API registered")
except Exception as e:
    logger.warning(f"Could not register V2 Prediction API: {e}")

# V2: Network Analysis System
try:
    from ..monitoring.network_controller import router as network_v2_router
    app.include_router(network_v2_router)
    logger.info("V2 Network Analysis System API registered")
except Exception as e:
    logger.warning(f"Could not register V2 Network API: {e}")

# V2: Simulation & OpSec Systems
try:
    from ..monitoring.simulation_opsec_controller import router as simulation_v2_router, opsec_router
    app.include_router(simulation_v2_router)
    app.include_router(opsec_router)
    logger.info("V2 Simulation & OpSec Systems API registered")
except Exception as e:
    logger.warning(f"Could not register V2 Simulation/OpSec API: {e}")

# WebSocket connections
active_connections: List[WebSocket] = []
# Track connection metadata for debugging
connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}


# Background task for periodic cleanup
async def periodic_cleanup():
    """Periodically clean up stale connections"""
    while True:
        try:
            await asyncio.sleep(10)  # Run every 10 seconds
            await cleanup_stale_connections()
            if len(active_connections) > 0:
                logger.debug(f"Periodic cleanup: {len(active_connections)} active connections")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")


async def cleanup_stale_connections():
    """Remove stale connections from active_connections list and close them properly"""
    disconnected = []
    for websocket in list(active_connections):  # Use list() to avoid modification during iteration
        try:
            # Check if connection is still valid - be lenient, only remove if clearly disconnected
            state = websocket.client_state
            if state not in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
                # Only remove if clearly disconnected (not CONNECTED or CONNECTING)
                logger.debug(f"Removing stale connection (state: {state})")
                disconnected.append(websocket)
        except Exception as e:
            # If we can't check state, assume it's stale (but be conservative)
            logger.debug(f"Error checking connection state: {e}, assuming stale")
            # Only add to disconnected if we're sure it's an error
            try:
                # Try to send a ping to see if connection is alive
                await asyncio.wait_for(websocket.send_json({"type": "ping"}), timeout=0.1)
            except:
                # Can't send, so it's likely disconnected
                disconnected.append(websocket)
    
    # Remove and close disconnected connections properly
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            if conn in connection_metadata:
                del connection_metadata[conn]
            # Try to close the connection properly to prevent CLOSE_WAIT
            try:
                # Only try to close if not already disconnected
                if conn.client_state in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
                    await asyncio.wait_for(conn.close(code=1000), timeout=0.5)
            except (asyncio.TimeoutError, WebSocketDisconnect, Exception):
                # Connection already closed or can't be closed - that's fine
                pass
            logger.debug(f"Removed and closed stale connection from active_connections (now: {len(active_connections)})")


async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    # Clean up stale connections first
    await cleanup_stale_connections()
    
    disconnected = []
    for connection in list(active_connections):  # Use list() to avoid modification during iteration
        try:
            # Double-check connection state before sending
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_json(message)
            else:
                disconnected.append(connection)
        except Exception as e:
            logger.debug(f"Error broadcasting to connection: {e}")
            disconnected.append(connection)
    
    # Remove disconnected connections
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)
            if conn in connection_metadata:
                del connection_metadata[conn]


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "name": "ICEBURG 2.0",
        "description": "Ultimate Truth-Finding AI Civilization",
        "version": "2.0.0",
        "status": "operational"
    }


@app.get("/test-thinking-stream")
async def test_thinking_stream():
    """Test endpoint to verify thinking_stream messages work"""
    async def test_gen():
        for i, msg in enumerate([
            "Analyzing your question...",
            "Searching knowledge base...",
            "Synthesizing information..."
        ]):
            yield f"data: {json.dumps({'type': 'thinking_stream', 'content': msg, 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.3)
        yield f"data: {json.dumps({'type': 'chunk', 'content': 'Test response', 'timestamp': time.time()})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"
    
    return StreamingResponse(
        test_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_status": system_integrator.get_system_status()
    }
    
    # Add system health monitor data if available
    if system_health_monitor:
        try:
            current_health = system_health_monitor.get_current_health()
            if current_health:
                health_data["health_score"] = current_health.health_score
                health_data["health_status"] = current_health.status
                health_data["cpu_usage"] = current_health.cpu_usage
                health_data["memory_usage"] = current_health.memory_usage
                health_data["thermal_state"] = current_health.thermal_state
        except Exception as e:
            logger.debug(f"Could not get health monitor data: {e}")
    
    return health_data


@app.get("/api/health")
async def api_health_check():
    """Detailed API health check endpoint with full health metrics"""
    if system_health_monitor:
        try:
            # Get current health snapshot
            current_health = system_health_monitor.get_current_health()
            
            # Run all health checks
            health_checks = await system_health_monitor.run_health_checks()
            
            return {
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "health_score": current_health.health_score if current_health else 0.0,
                "health_status": current_health.status if current_health else "unknown",
                "metrics": {
                    "cpu_usage": current_health.cpu_usage if current_health else 0.0,
                    "memory_usage": current_health.memory_usage if current_health else 0.0,
                    "disk_usage": current_health.disk_usage if current_health else 0.0,
                    "thermal_state": current_health.thermal_state if current_health else "unknown",
                    "thermal_temperature": current_health.thermal_temperature if current_health else 0.0,
                    "network_active": current_health.network_active if current_health else False,
                    "error_rate": current_health.error_rate if current_health else 0.0,
                    "latency_p95": current_health.latency_p95 if current_health else 0.0,
                    "cache_hit_rate": current_health.cache_hit_rate if current_health else 0.0
                },
                "health_checks": health_checks,
                "system_status": system_integrator.get_system_status()
            }
        except Exception as e:
            logger.error(f"Error getting health data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    else:
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "health_monitoring": "disabled",
            "system_status": system_integrator.get_system_status()
        }


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics endpoint"""
    try:
        from ..llm import get_llm_cache_stats
        from ..caching.model_cache import get_model_cache
        
        llm_stats = get_llm_cache_stats()
        model_cache = get_model_cache()
        model_stats = model_cache.get_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "llm_cache": llm_stats,
            "model_cache": model_stats,
            "summary": {
                "total_hit_rate": llm_stats.get("hit_rate", 0.0),
                "model_cache_hit_rate": model_stats.get("model_stats", {}).get("hit_rate", 0.0),
                "cache_sizes": model_stats.get("cache_sizes", {})
            }
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }



@app.get("/api/encyclopedia")
async def get_encyclopedia():
    """Get Celestial Encyclopedia dataset"""
    try:
        from pathlib import Path
        import json
        
        # Load encyclopedia data
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        encyclopedia_path = data_dir / "celestial_encyclopedia.json"
        
        if not encyclopedia_path.exists():
            raise HTTPException(status_code=404, detail="Encyclopedia data not found")
        
        with open(encyclopedia_path, 'r', encoding='utf-8') as f:
            encyclopedia_data = json.load(f)
        
        return JSONResponse(content=encyclopedia_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading encyclopedia: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading encyclopedia: {str(e)}")


@app.get("/api/middleware/stats")
async def get_middleware_stats():
    """Get middleware statistics and analytics."""
    try:
        if not global_middleware:
            return JSONResponse({
                "error": "Middleware not initialized",
                "enabled": False
            })
        
        from ..middleware.analytics import MiddlewareAnalytics
        analytics = MiddlewareAnalytics(
            registry=global_middleware.registry,
            learning_system=global_middleware.learning_system,
            emergence_aggregator=global_middleware.emergence_aggregator
        )
        
        stats = analytics.get_comprehensive_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error getting middleware stats: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ===== LOST KNOWLEDGE API ENDPOINTS =====
@app.post("/api/knowledge/submit")
async def submit_knowledge(submission: dict):
    """
    Submit historical/suppressed knowledge to the Lost Knowledge database.
    
    Request body:
    {
        "type": "patent" | "document" | "video",
        "title": str,
        "content": str,
        "year": int (optional),
        "tags": list[str] (optional),
        "patent_number": str (required if type=="patent"),
        "suppression_evidence": str (optional),
        "source_url": str (optional)
    }
    """
    try:
        from ..ingestion import HumanCuratedSubmission
        
        submitter = HumanCuratedSubmission()
        entry_id = None
        
        submission_type = submission.get('type', 'document')
        
        if submission_type == 'patent':
            entry_id = submitter.submit_patent(
                patent_number=submission['patent_number'],
                year=submission.get('year'),
                title=submission['title'],
                description=submission.get('content', submission.get('description', '')),
                tags=submission.get('tags', []),
                suppression_evidence=submission.get('suppression_evidence')
            )
        elif submission_type == 'document':
            entry_id = submitter.submit_document(
                title=submission['title'],
                content=submission.get('content', ''),
                year=submission.get('year'),
                tags=submission.get('tags', []),
                suppression_evidence=submission.get('suppression_evidence'),
                source_url=submission.get('source_url')
            )
        elif submission_type == 'video':
            entry_id = submitter.submit_video(
                url=submission.get('source_url', ''),
                title=submission['title'],
                description=submission.get('content', ''),
                year=submission.get('year'),
                tags=submission.get('tags', [])
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid submission type: {submission_type}")
        
        return {
            "status": "success",
            "entry_id": entry_id,
            "message": "Knowledge submitted successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        logger.error(f"Error submitting knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/search")
async def search_knowledge(
    query: str = None,
    topic: str = None,
    year: int = None,
    source_type: str = None,
    min_suppression_score: float = None
):
    """Search the Lost Knowledge database"""
    try:
        from ..ingestion import HumanCuratedSubmission
        
        submitter = HumanCuratedSubmission()
        filters = {}
        
        if topic:
            filters['topic'] = topic
        if year:
            filters['year'] = year
        if source_type:
            filters['source_type'] = source_type
        
        results = submitter.search(query=query, **filters)
        
        # Filter by suppression score if requested
        if min_suppression_score is not None:
            results = [
                r for r in results
                if r.get('suppression_score', 0.0) >= min_suppression_score
            ]
        
        return {
            "status": "success",
            "query": query,
            "filters": filters,
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """Get Lost Knowledge database statistics"""
    try:
        from ..ingestion import HumanCuratedSubmission
        from pathlib import Path
        import json
        
        submitter = HumanCuratedSubmission()
        index = submitter.index
        
        # Calculate statistics
        total_entries = len(index.get('entries', []))
        
        # Group by type
        by_type = {}
        by_topic = {}
        high_suppression = []
        
        for entry in index.get('entries', []):
            # Count by type
            source_type = entry.get('source_type', 'unknown')
            by_type[source_type] = by_type.get(source_type, 0) + 1
            
            # Count by topic
            topic = entry.get('topic', 'unknown')
            by_topic[topic] = by_topic.get(topic, 0) + 1
            
            # High suppression entries
            if entry.get('suppression_score', 0.0) > 0.7:
                high_suppression.append({
                    'id': entry['id'],
                    'title': entry.get('title'),
                    'suppression_score': entry.get('suppression_score')
                })
        
        return {
            "status": "success",
            "total_entries": total_entries,
            "by_type": by_type,
            "by_topic": by_topic,
            "high_suppression_count": len(high_suppression),
            "high_suppression_entries": high_suppression,
            "last_updated": index.get('last_updated'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/middleware/agent/{agent_name}")
async def get_agent_middleware_stats(agent_name: str):
    """Get middleware statistics for a specific agent."""
    try:
        if not global_middleware:
            return JSONResponse({
                "error": "Middleware not initialized",
                "enabled": False
            })
        
        from ..middleware.analytics import MiddlewareAnalytics
        analytics = MiddlewareAnalytics(
            registry=global_middleware.registry,
            learning_system=global_middleware.learning_system,
            emergence_aggregator=global_middleware.emergence_aggregator
        )
        
        agent_stats = analytics.get_agent_analytics(agent_name)
        return JSONResponse(agent_stats)
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Phase 2: File upload endpoint using multipart/form-data.
    
    Accepts file uploads and returns a file_id for later reference.
    Files are stored temporarily and automatically cleaned up after 1 hour.
    
    Returns:
        {
            "file_id": "uuid",
            "filename": "original_filename.ext",
            "content_type": "image/png",
            "size": 12345
        }
    """
    try:
        from ..storage.file_storage import get_file_storage
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file
        is_valid, error_msg = validate_file_upload(
            file.filename or "unknown",
            file.content_type or "application/octet-stream",
            file_size
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg or "Invalid file")
        
        # Store file
        file_storage = get_file_storage()
        file_id = file_storage.store_file(
            file_content=file_content,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            file_size=file_size
        )
        
        # Return file_id
        return JSONResponse(content={
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file_size
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/api/file/{file_id}")
async def get_file(file_id: str):
    """
    Retrieve a file by file_id.
    
    Returns the file content with appropriate Content-Type header.
    """
    try:
        from ..storage.file_storage import get_file_storage
        
        file_storage = get_file_storage()
        metadata = file_storage.get_file(file_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_content = file_storage.read_file(file_id)
        if not file_content:
            raise HTTPException(status_code=404, detail="File content not found")
        
        from fastapi.responses import Response
        return Response(
            content=file_content,
            media_type=metadata["content_type"],
            headers={
                "Content-Disposition": f'inline; filename="{metadata["filename"]}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")


@app.get("/api/encyclopedia/search/{query}")
async def search_encyclopedia(query: str):
    """Search Celestial Encyclopedia"""
    try:
        from pathlib import Path
        import json
        
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        encyclopedia_path = data_dir / "celestial_encyclopedia.json"
        
        if not encyclopedia_path.exists():
            raise HTTPException(status_code=404, detail="Encyclopedia data not found")
        
        with open(encyclopedia_path, 'r', encoding='utf-8') as f:
            encyclopedia_data = json.load(f)
        
        query_lower = query.lower()
        results = []
        
        # Search across all categories
        for category_name, category_data in encyclopedia_data.items():
            if category_name == "metadata":
                continue
            
            if isinstance(category_data, dict):
                for item_id, item_data in category_data.items():
                    if isinstance(item_data, dict):
                        # Search in name, description, and other text fields
                        searchable_text = json.dumps(item_data).lower()
                        if query_lower in searchable_text:
                            results.append({
                                "category": category_name,
                                "id": item_id,
                                "data": item_data
                            })
        
        return JSONResponse(content={"query": query, "results": results, "count": len(results)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching encyclopedia: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching encyclopedia: {str(e)}")


@app.post("/api/query")
async def query_endpoint(request: Dict[str, Any], http_request: Request):
    """Query endpoint with input validation and sanitization - supports streaming via SSE"""
    try:
        # Check if client wants streaming (SSE)
        accept_header = http_request.headers.get("accept", "")
        use_streaming = "text/event-stream" in accept_header or request.get("stream", False)
        
        logger.info(f"🔍🔍🔍 STREAMING CHECK: accept_header='{accept_header}', stream_flag={request.get('stream')}, use_streaming={use_streaming}")
        
        query = request.get("query", "")
        mode = request.get("mode", "chat")
        agent = request.get("agent", "auto")
        degradation_mode = request.get("degradation_mode", False)
        settings = request.get("settings", {})
        
        # v5: Route query to appropriate handler
        from ..router.request_router import get_request_router
        router = get_request_router()
        routing_decision = router.route(query, context={"mode": mode})
        logger.info(f"🔍 v5 Routing: query='{query[:50]}' -> mode={routing_decision.mode} (confidence={routing_decision.confidence:.2f})")
        
        logger.info(f"🔍🔍🔍 REQUEST PARAMS: query='{query[:50]}', mode='{mode}', agent='{agent}'")
        
        # Validate query (skip for now to debug hanging issue)
        # is_valid, error_msg = validate_query(query)
        # if not is_valid:
        #     raise HTTPException(status_code=400, detail=error_msg or "Invalid query")
        
        # Sanitize input (skip for now to debug hanging issue)
        # query = sanitize_input(query)
        # if mode:
        #     mode = sanitize_input(str(mode), max_length=50)
        # if agent:
        #     agent = sanitize_input(str(agent), max_length=50)
        
        # Validate file uploads if present
        files = request.get("files", [])
        if files:
            for file_data in files:
                filename = file_data.get("name", "")
                file_type = file_data.get("type", "")
                file_size = file_data.get("size", 0)
                
                is_valid, error_msg = validate_file_upload(filename, file_type, file_size)
                if not is_valid:
                    raise HTTPException(status_code=400, detail=error_msg or "Invalid file")
        
        logger.info(
            "📶 HTTP /api/query request: accept=%s, stream_flag=%s, use_streaming=%s, mode=%s",
            accept_header,
            request.get("stream"),
            use_streaming,
            mode,
        )
        
        # If streaming requested, use SSE
        if use_streaming:
            logger.info("🌊🌊🌊 SSE streaming requested for mode=%s, agent=%s", mode, agent)
            
            async def generate_stream():
                # CRITICAL: First yield MUST be immediate - no try/except, no logging
                # This establishes the SSE connection immediately
                yield f"data: {json.dumps({'type': 'thinking_stream', 'content': 'Initializing...', 'timestamp': time.time()})}\n\n"
                
                try:
                    logger.info("🌊🌊🌊 GENERATOR FUNCTION CALLED - first yield sent")
                    
                    # Track prompt start time
                    prompt_start_time = time.time()
                    prompt_id = str(uuid.uuid4())
                    
                    # Get agent value from request (closure variable)
                    stream_agent = request.get("agent", "auto")
                    logger.info(f"🔍🔍🔍 SSE generate_stream: agent from request = '{stream_agent}' (type: {type(stream_agent)})")
                    
                    # FAST PATH: Check for simple queries FIRST (before any processing)
                    # CHAT MODE & FAST MODE: Secretary ONLY - completely isolated from Surveyor and all other agents
                    if mode == "chat" or mode == "fast":
                        from ..config import load_config_with_model
                        from ..providers.factory import provider_factory
                        
                        primary_model = settings.get("primaryModel", "llama3.1:8b")
                        # Force Ollama-compatible model for chat mode
                        if "gemini" in primary_model.lower() or "gpt" in primary_model.lower() or "claude" in primary_model.lower():
                            primary_model = "llama3.1:8b"
                            logger.info(f"🔄 Model override: Using {primary_model} for Ollama in chat mode")
                        temperature = settings.get("temperature", 0.7)
                        max_tokens = settings.get("maxTokens", 2000)
                        
                        logger.info("🌊 SSE chat mode engaged (model=%s, temp=%s, max_tokens=%s, stream_agent=%s)", primary_model, temperature, max_tokens, str(stream_agent))
                        
                        # Normalize agent value - treat "auto" or "surveyor" as "secretary" for chat mode
                        agent_normalized = str(stream_agent).lower().strip() if stream_agent else "auto"
                        if agent_normalized == "auto" or agent_normalized == "surveyor":
                            agent_normalized = "secretary"
                            logger.info(f"🔍 Agent forced to secretary in chat mode (was: {stream_agent})")
                        logger.info(f"🔍 SSE Agent normalization: raw='{stream_agent}' -> normalized='{agent_normalized}'")
                        
                        # Secretary agent - fast, friendly, knows ICEBURG, no VectorStore needed
                        if agent_normalized == "secretary":
                            logger.info(f"👔 Secretary agent selected - fast chat mode")
                            
                            # SEARCH-FIRST: Get real web evidence before responding
                            web_evidence_context = ""
                            search_sources = []
                            try:
                                from ..search.web_search_orchestrator import get_web_search_orchestrator
                                
                                # Quick search (5 second timeout for reliable results)
                                search_orchestrator = get_web_search_orchestrator(timeout=5.0)
                                search_response = await search_orchestrator.search(query, num_results=5)
                                
                                if search_response.results:
                                    logger.info(f"🔍 Fast mode search: {len(search_response.results)} results in {search_response.search_time_ms:.0f}ms")
                                    
                                    # Format evidence for Secretary
                                    evidence_parts = ["WEB SEARCH RESULTS (use these as sources):"]
                                    for i, result in enumerate(search_response.results, 1):
                                        evidence_parts.append(f"[Source {i}] {result.title}")
                                        evidence_parts.append(f"  URL: {result.url}")
                                        evidence_parts.append(f"  {result.snippet[:200]}...")
                                        search_sources.append({"title": result.title, "url": result.url})
                                    
                                    web_evidence_context = "\n".join(evidence_parts)
                                    yield f"data: {json.dumps({'type': 'thinking_stream', 'content': f'Found {len(search_response.results)} sources...', 'timestamp': time.time()})}\n\n"
                                else:
                                    logger.info("🔍 Fast mode search: No results found")
                                    
                            except Exception as e:
                                logger.warning(f"🔍 Fast mode search failed (continuing without): {e}")
                            
                            try:
                                from ..config import load_config_with_model
                                from ..agents.secretary import run as secretary_run
                                
                                # Initialize config - use default provider (Gemini via .env)
                                is_fast_mode = mode == "fast" or mode == "chat"
                                secretary_cfg = await asyncio.to_thread(
                                    load_config_with_model,
                                    primary_model=primary_model,
                                    use_small_models=False,
                                    fast=is_fast_mode
                                )
                                # Ensure Ollama provider is set
                                from ..config import IceburgConfig
                                secretary_cfg = IceburgConfig(
                                    data_dir=secretary_cfg.data_dir,
                                    surveyor_model=secretary_cfg.surveyor_model,
                                    dissident_model=secretary_cfg.dissident_model,
                                    synthesist_model=secretary_cfg.synthesist_model,
                                    oracle_model=secretary_cfg.oracle_model,
                                    embed_model=secretary_cfg.embed_model,
                                    llm_provider="ollama",
                                    provider_url=secretary_cfg.provider_url,
                                    timeout_s=secretary_cfg.timeout_s,
                                    enable_code_generation=secretary_cfg.enable_code_generation,
                                    disable_memory=secretary_cfg.disable_memory,
                                    enable_software_lab=secretary_cfg.enable_software_lab,
                                    max_context_length=secretary_cfg.max_context_length,
                                    fast=secretary_cfg.fast
                                )
                                
                                # Send thinking message
                                thinking_msg = "I'm thinking about your question..."
                                yield f"data: {json.dumps({'type': 'thinking_stream', 'content': thinking_msg, 'timestamp': time.time()})}\n\n"
                                await asyncio.sleep(0.1)
                                
                                # Get conversation_id and user_id from request
                                conversation_id = request.get("conversation_id")
                                if not conversation_id:
                                    conversation_id = str(uuid.uuid4())
                                
                                user_id = request.get("user_id")
                                
                                # Call Secretary agent (no VectorStore needed!)
                                # Enhance query with web evidence if available
                                enhanced_query = query
                                if web_evidence_context:
                                    enhanced_query = f"{query}\n\n{web_evidence_context}"
                                    logger.info(f"🔍 Enhanced query with {len(search_sources)} web sources")
                                
                                logger.info(f"🔍🔍🔍 SSE: Calling secretary_run with query='{query[:100]}' (length={len(enhanced_query)}), conversation_id={conversation_id[:20] if conversation_id else None}")
                                
                                # Pass routing_mode to agent (fixes double routing)
                                routing_mode_sse = routing_decision.mode if routing_decision else None
                                
                                # Check Semantic Cache (optimization)
                                cached_response = None
                                try:
                                    # Generate embedding for query
                                    query_embeddings = embed_texts("nomic-embed-text", [enhanced_query])
                                    if query_embeddings:
                                        cached_response = semantic_cache.get(enhanced_query, embedding=query_embeddings[0])
                                        if cached_response:
                                            logger.info(f"⚡️ Cache Hit for '{enhanced_query}'")
                                except Exception as e:
                                    logger.warning(f"Cache check failed: {e}")

                                if cached_response:
                                    # Stream cached response instantly
                                    yield f"data: {json.dumps({'type': 'thinking_stream', 'content': '⚡️ Instantly retrieved from knowledge cache', 'timestamp': time.time()})}\n\n"
                                    # Stream words fast
                                    for word in cached_response.split():
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': word + ' ', 'timestamp': time.time()})}\n\n"
                                        await asyncio.sleep(0.005) 
                                    yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"
                                    return

                                # Wrap with global middleware if available
                                if global_middleware:
                                    secretary_response = await global_middleware.execute_agent(
                                        agent_name="secretary",
                                        agent_func=secretary_run,
                                        cfg=secretary_cfg,
                                        query=enhanced_query,
                                        verbose=False,
                                        thinking_callback=lambda msg: None,
                                        conversation_id=conversation_id,
                                        user_id=user_id,
                                        routing_mode=routing_mode_sse,
                                        mode=mode  # Pass original mode for chat detection
                                    )
                                else:
                                    # Use asyncio.to_thread for synchronous secretary_run
                                    secretary_response = await asyncio.to_thread(
                                        secretary_run,
                                        secretary_cfg,
                                        enhanced_query,
                                        verbose=False,
                                        thinking_callback=lambda msg: None,
                                        conversation_id=conversation_id,
                                        user_id=user_id,
                                        routing_mode=routing_mode_sse,
                                        mode=mode  # Pass original mode for chat detection
                                    )
                                logger.info(f"🔍🔍🔍 SSE: Secretary returned response length={len(secretary_response) if secretary_response else 0}, preview='{secretary_response[:100] if secretary_response else None}...'")
                                
                                # SANITIZATION: Remove hallucinated "Goal-driven task" reports BEFORE streaming
                                # This prevents verbose planning logs from reaching the browser
                                if secretary_response:
                                    import re
                                    # Remove "I've executed your goal-driven task..." text
                                    secretary_response = re.sub(r"I've executed your goal-driven task.*?(?=\n\n|\Z)", "", secretary_response, flags=re.DOTALL)
                                    # Remove "Goal: X - Planned Y step(s)" lines
                                    secretary_response = re.sub(r"Goal:.*?Planned \d+ step\(s\).*?(?=\n)", "", secretary_response)
                                    # Remove "Task Execution Report" lines
                                    secretary_response = re.sub(r"\*\*Task Execution Report\*\*.*?(?=\n)", "", secretary_response)
                                    # Remove "✓ Task:" lines
                                    secretary_response = re.sub(r"✓.*?Task:.*?(?=\n)", "", secretary_response)
                                    # Remove "Planned X step(s) - Completed: Y, Failed: Z" lines
                                    secretary_response = re.sub(r"- Planned \d+ step\(s\) - Completed: \d+, Failed: \d+", "", secretary_response)
                                    # Remove "All tasks have been executed" lines
                                    secretary_response = re.sub(r"All tasks have been executed.*?(?=\n|\Z)", "", secretary_response)
                                    # Clean up multiple newlines
                                    secretary_response = re.sub(r"\n{3,}", "\n\n", secretary_response).strip()

                                    # Log metrics to telemetry (moved here to catch empty responses too)
                                    try:
                                        from ..telemetry.advanced_telemetry import PromptMetrics
                                        metric = PromptMetrics(
                                            prompt_id=prompt_id,
                                            prompt_text=query,
                                            response_time=time.time() - prompt_start_time,
                                            token_count=len(secretary_response) // 4 if secretary_response else 0,
                                            model_used=primary_model,
                                            success=bool(secretary_response),
                                            quality_score=1.0 if secretary_response else 0.0
                                        )
                                        telemetry.track_prompt(metric)
                                        logger.info(f"📊 Telemetry logged: {metric.response_time:.2f}s, {metric.token_count} tokens")
                                    except Exception as tel_err:
                                        logger.warning(f"⚠️ Failed to log telemetry: {tel_err}")

                                    logger.info(f"✅ Sanitized response length={len(secretary_response)}, preview='{secretary_response[:100]}...'")
                                
                                # Stream the response
                                if secretary_response:
                                    for word in secretary_response.split():
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': word + ' ', 'timestamp': time.time()})}\n\n"
                                        await asyncio.sleep(0.02)
                                    yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"
                                    logger.info("✅ Secretary response sent successfully")
                                    return
                                else:
                                    yield f"data: {json.dumps({'type': 'error', 'content': 'Secretary did not return a response', 'timestamp': time.time()})}\n\n"
                                    return
                                    
                            except Exception as e:
                                logger.error(f"❌ Secretary agent error: {e}", exc_info=True)
                                # Send error message instead of hardcoded fallback
                                from ..api.error_messages import format_error_for_user
                                error_msg = format_error_for_user(e, context="secretary_failure")
                                yield f"data: {json.dumps({'type': 'error', 'message': error_msg, 'timestamp': time.time()})}\n\n"
                                yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"
                                return
                        
                        # CHAT MODE ENDS HERE - Secretary is the ONLY agent
                        # Surveyor and other agents are NOT allowed in chat mode
                        # If we reach here, something is wrong
                        logger.error(f"❌ CRITICAL: Reached end of SSE chat mode without secretary handling")
                        yield f"data: {json.dumps({'type': 'error', 'content': 'Chat mode requires secretary agent', 'timestamp': time.time()})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"
                        return
                        
                    # NON-CHAT MODES CONTINUE HERE (Research, etc.)
                    # Chat mode is completely isolated above - NO SURVEYOR CODE IN CHAT MODE
                    
                    # For non-chat modes, continue with normal processing
                    
                    # Get conversation ID
                        conversation_id = request.get("conversation_id")
                        if not conversation_id:
                            conversation_id = str(uuid.uuid4())
                        
                        # DISABLED: Conversation history causes repetitive, pseudo-profound responses
                        # Each query should be answered independently based on actual research
                        conversation_history = []
                        logger.info(f"📝📝📝 Conversation history DISABLED - answering query independently")
                        
                        custom_cfg = load_config_with_model(
                            primary_model=primary_model,
                            use_small_models=False,
                            fast=True
                        )
                        
                        provider = provider_factory(custom_cfg)
                        # Build system prompt based on agent selection (same logic as WebSocket)
                        if agent and agent != "auto":
                            if agent == "surveyor":
                                from ..agents.surveyor import SURVEYOR_SYSTEM
                                system_prompt = SURVEYOR_SYSTEM
                            elif agent == "synthesist":
                                from ..agents.synthesist import SYNTHESIST_SYSTEM
                                system_prompt = SYNTHESIST_SYSTEM
                            elif agent == "dissident":
                                from ..agents.dissident import DISSIDENT_SYSTEM
                                system_prompt = DISSIDENT_SYSTEM
                            elif agent == "oracle":
                                from ..agents.oracle import ORACLE_SYSTEM
                                system_prompt = ORACLE_SYSTEM
                            else:
                                system_prompt = "You are ICEBURG, an AI civilization with deep knowledge decoding capabilities. You are designed for truth-finding and comprehensive analysis. Provide clear, insightful answers that demonstrate your advanced reasoning capabilities."
                        else:
                            system_prompt = "You are ICEBURG, an AI civilization with deep knowledge decoding capabilities. You are designed for truth-finding and comprehensive analysis. Provide clear, insightful answers that demonstrate your advanced reasoning capabilities."
                        
                        # DISABLED: Conversation history causes repetitive, pseudo-profound responses
                        # Each query should be answered independently based on actual research
                        full_prompt = query
                        logger.info(f"📝 Conversation history DISABLED - answering query independently")
                        
                        # Send thinking messages
                        thinking_msg = "I'm orchestrating a parallel swarm debate..."
                        yield f"data: {json.dumps({'type': 'thinking_stream', 'content': thinking_msg, 'timestamp': time.time()})}\n\n"
                        
                        # --- PHASE 3: PARALLEL SWARM EXECUTION ---
                        if mode == "research" or agent == "surveyor":
                            from ..core.swarm_orchestrator import swarm, AgentTask
                            from ..agents.surveyor import SURVEYOR_SYSTEM
                            from ..agents.dissident import DISSIDENT_SYSTEM
                            
                            # Define parallel tasks
                            tasks = [
                                AgentTask(
                                    name="surveyor",
                                    func=provider.chat_complete,
                                    args=(),
                                    kwargs={
                                        "model": primary_model,
                                        "prompt": query,
                                        "system": SURVEYOR_SYSTEM,
                                        "temperature": temperature
                                    }
                                ),
                                AgentTask(
                                    name="dissident",
                                    func=provider.chat_complete,
                                    args=(),
                                    kwargs={
                                        "model": primary_model,
                                        "prompt": f"Critique this query and provide counter-arguments: {query}",
                                        "system": DISSIDENT_SYSTEM,
                                        "temperature": temperature
                                    }
                                )
                            ]
                            
                            # Execute Swarm
                            results = await swarm.run_swarm("research_debate", tasks)
                            
                            # Synthesize results (Judge-in-the-Loop)
                            surveyor_out = results["surveyor"]["output"] if results["surveyor"]["success"] else "Surveyor failed."
                            dissident_out = results["dissident"]["output"] if results["dissident"]["success"] else "Dissident failed."
                            
                            # Judge Synthesis
                            judge_prompt = f"""
                            Review the following debate:
                            
                            [RESEARCHER]: {surveyor_out}
                            
                            [CRITIC]: {dissident_out}
                            
                            Synthesize a final, balanced answer that incorporates valid critiques.
                            """
                            
                            agent_response = await asyncio.to_thread(
                                provider.chat_complete,
                                model=primary_model,
                                prompt=judge_prompt,
                                system="You are an impartial Judge synthesizing a research debate.",
                                temperature=temperature
                            )
                            
                        else:
                            # Standard single agent fallback for other modes
                            thinking_msg = "I'm processing your query..."
                            yield f"data: {json.dumps({'type': 'thinking_stream', 'content': thinking_msg, 'timestamp': time.time()})}\n\n"
                            
                            agent_response = await asyncio.to_thread(
                                provider.chat_complete,
                                model=primary_model,
                                prompt=full_prompt,
                                system=system_prompt,
                                temperature=temperature,
                                options={"max_tokens": max_tokens} if max_tokens else None
                            )
                        
                        # Check for empty response
                        if not agent_response or not agent_response.strip():
                            agent_response = "I received your query but couldn't generate a response. Please try rephrasing your question."
                        
                        # Save conversation
                        try:
                            conversation_entry = ConversationEntry(
                                conversation_id=conversation_id,
                                timestamp=datetime.now().isoformat(),
                                user_message=query,
                                assistant_message=agent_response,
                                agent_used="swarm_debate" if (mode == "research" or agent == "surveyor") else "single_llm",
                                mode=mode,
                                metadata={}
                            )
                            local_persistence.save_conversation(conversation_entry)
                        except Exception as e:
                            logger.error(f"Error saving conversation: {e}")
                        
                        # Stream content character-by-character (like WebSocket)
                        chunk_delay = 0.0001  # Fast streaming
                        logger.info("🌊 SSE streaming %d characters to client", len(agent_response))
                        for i in range(0, len(agent_response)):
                            chunk = agent_response[i]
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                            if i + 1 < len(agent_response):
                                await asyncio.sleep(chunk_delay)
                    elif mode == "astrophysiology":
                        # Astro-Physiology mode - use dedicated handler with SSE streaming
                        from ..modes.astrophysiology_handler import handle_astrophysiology_query
                        from ..config import load_config_with_model
                        from ..vectorstore import VectorStore
                        
                        logger.info("🌌 SSE Astro-Physiology mode engaged")
                        
                        # Get settings for astro-physiology mode
                        primary_model = settings.get("primaryModel", "llama3.1:8b")
                        temperature = settings.get("temperature", 0.7)
                        max_tokens = settings.get("maxTokens", 2000)
                        use_small_models = False  # Use full models for astro-physiology
                        
                        # Use custom config with frontend model selection
                        custom_cfg = await asyncio.to_thread(
                            load_config_with_model,
                            primary_model=primary_model,
                            use_small_models=use_small_models,
                            fast=False  # Use full models for astro-physiology
                        )
                        
                        # Use asyncio.Queue for message streaming
                        message_queue = asyncio.Queue()
                        
                        async def queue_callback(msg: Dict[str, Any]):
                            """Put messages into queue for SSE generator"""
                            await message_queue.put(msg)
                        
                        # Run handler in background task
                        handler_task = None
                        result_container = [None]
                        
                        async def run_handler():
                            try:
                                # Skip VectorStore initialization entirely for astro-physiology to avoid ChromaDB panics
                                # VectorStore is optional and not required for basic molecular calculations
                                vs = None
                                logger.info("🌌 Skipping VectorStore initialization for astro-physiology (optional feature)")
                                
                                try:
                                    result = await asyncio.wait_for(
                                        handle_astrophysiology_query(
                                            query=query,
                                            message=request,
                                            cfg=custom_cfg,
                                            vs=vs,
                                            websocket_callback=queue_callback,
                                            temperature=temperature,
                                            max_tokens=max_tokens
                                        ),
                                        timeout=300  # 5 minute timeout
                                    )
                                    result_container[0] = result
                                    logger.info(f"🌌 Handler completed successfully, result set: {result is not None}")
                                    # Signal completion
                                    await message_queue.put({"type": "done"})
                                except asyncio.TimeoutError:
                                    logger.error("🌌 Astro-physiology handler timed out")
                                    result_container[0] = {
                                        "query": query,
                                        "mode": "astrophysiology",
                                        "error": "timeout",
                                        "results": {
                                            "error": "timeout",
                                            "content": "The calculation timed out. Please try again."
                                        }
                                    }
                                    await message_queue.put({"type": "error", "content": "Request timed out"})
                                except Exception as e:
                                    logger.error(f"🌌 Error in astro-physiology handler: {e}", exc_info=True)
                                    result_container[0] = {
                                        "query": query,
                                        "mode": "astrophysiology",
                                        "error": "handler_error",
                                        "results": {
                                            "error": "handler_error",
                                            "content": f"I encountered an error processing your query: {str(e)}"
                                        }
                                    }
                                    await message_queue.put({"type": "error", "content": str(e)})
                            except BaseException as e:
                                # Catch-all for any unexpected errors in run_handler itself (including PanicException)
                                logger.error(f"🌌 Fatal error in run_handler: {type(e).__name__}: {e}", exc_info=True)
                                result_container[0] = {
                                    "query": query,
                                    "mode": "astrophysiology",
                                    "error": "fatal_error",
                                    "results": {
                                        "error": "fatal_error",
                                        "content": f"A fatal error occurred: {type(e).__name__}: {str(e)}"
                                    }
                                }
                                try:
                                    await message_queue.put({"type": "error", "content": f"Fatal error: {str(e)}"})
                                except:
                                    pass  # Queue might be closed
                        
                        handler_task = asyncio.create_task(run_handler())
                        
                        # Stream messages from queue
                        while True:
                            try:
                                # Wait for message with timeout
                                msg = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                                
                                if msg.get("type") == "done":
                                    break
                                elif msg.get("type") == "error":
                                    yield f"data: {json.dumps({'type': 'error', 'message': msg.get('content', 'Unknown error')})}\n\n"
                                    break
                                elif msg.get("type") == "thinking":
                                    yield f"data: {json.dumps({'type': 'thinking_stream', 'content': msg.get('content', ''), 'timestamp': time.time()})}\n\n"
                                await asyncio.sleep(0.05)  # Small delay for streaming
                            except asyncio.TimeoutError:
                                # Check if handler task is done
                                if handler_task.done():
                                    break
                                continue
                        
                        # Wait for handler to complete and get final result
                        if handler_task and not handler_task.done():
                            try:
                                await asyncio.wait_for(handler_task, timeout=2.0)
                            except asyncio.TimeoutError:
                                logger.warning("Handler task still running after queue drained, waiting...")
                                # Wait a bit more for handler to complete
                                await asyncio.sleep(0.5)
                        
                        # Get final result and stream content
                        result = result_container[0]
                        logger.info(f"🌌 Final result container: {result is not None}, has results: {result.get('results') if result else False}")
                        
                        if result and result.get("results"):
                            # Stream the final content character-by-character
                            final_content = result.get("results", {}).get("content", "")
                            if final_content:
                                chunk_delay = 0.0001
                                logger.info("🌌 SSE streaming astro-physiology response: %d characters", len(final_content))
                                for i in range(0, len(final_content)):
                                    chunk = final_content[i]
                                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                                    if i + 1 < len(final_content):
                                        await asyncio.sleep(chunk_delay)
                            
                            # Send done with results
                            logger.info("🌌 Sending done message with results")
                            yield f"data: {json.dumps({'type': 'done', 'mode': 'astrophysiology', 'results': result.get('results', {})})}\n\n"
                        else:
                            logger.warning("🌌 No result or results found, sending error done message")
                            logger.warning(f"🌌 Result container value: {result}")
                            # Return error in consistent format
                            error_result = {
                                "query": query,
                                "mode": "astrophysiology",
                                "error": "no_results",
                                "results": {
                                    "error": "no_results",
                                    "content": "The calculation completed but no results were generated. Please check the server logs for details."
                                }
                            }
                            yield f"data: {json.dumps({'type': 'done', 'mode': 'astrophysiology', 'error': 'no_results', 'results': error_result.get('results', {})})}\n\n"
                    else:
                        # Use full protocol for other non-chat modes
                        # Use asyncio.Queue for message streaming (same pattern as astro-physiology)
                        research_queue = asyncio.Queue()
                        
                        # Track agent progress for step_complete events
                        agent_start_times = {}
                        agent_findings = {}
                        
                        async def research_callback(msg: Dict[str, Any]):
                            """Put messages into queue for SSE generator - emit step_complete for interactive UI"""
                            nonlocal agent_start_times, agent_findings
                            msg_type = msg.get("type", "action")
                            
                            # Detect agent start/complete for interactive step cards
                            if msg_type == "action":
                                action = msg.get("action", "")
                                status = msg.get("status", "")
                                description = msg.get("description", "Processing...")
                                
                                # Track agent starts
                                if "starting" in status.lower() or "initializing" in description.lower():
                                    agent_name = action.replace("_run", "").replace("_", " ").title()
                                    agent_start_times[agent_name] = time.time()
                                    agent_findings[agent_name] = []
                                    # Emit thinking_stream for progress indicator
                                    await research_queue.put({
                                        "type": "thinking_stream",
                                        "content": f"{agent_name}: {description}",
                                        "timestamp": time.time()
                                    })
                                
                                # Track findings/intermediate results
                                elif "found" in description.lower() or "analyzing" in description.lower() or "evidence" in description.lower():
                                    # Extract agent name from action
                                    for agent_name in agent_findings.keys():
                                        if agent_name.lower() in action.lower() or agent_name.lower() in description.lower():
                                            agent_findings[agent_name].append(description)
                                            break
                                    await research_queue.put({
                                        "type": "thinking_stream",
                                        "content": description,
                                        "timestamp": time.time()
                                    })
                                
                                # Detect agent completion - emit step_complete for interactive card
                                elif "complete" in status.lower() or "finished" in description.lower():
                                    agent_name = action.replace("_run", "").replace("_", " ").title()
                                    start_time = agent_start_times.get(agent_name, time.time())
                                    time_taken = time.time() - start_time
                                    findings = agent_findings.get(agent_name, [description])
                                    
                                    # Generate findings if we don't have any
                                    if not findings or findings == [description]:
                                        findings = [
                                            f"Analyzed query context and domain",
                                            f"Gathered relevant information",
                                            f"Processed in {time_taken:.1f}s"
                                        ]
                                    
                                    # Emit step_complete for interactive step card
                                    step_complete_event = {
                                        "type": "step_complete",
                                        "step": agent_name.lower().replace(" ", "_"),
                                        "report": {
                                            "findings": findings[:5],  # Limit to 5 findings
                                            "time_taken": round(time_taken, 2),
                                            "suggested_next": ["continue"]
                                        },
                                        "options": [
                                            {"action": "deep_dive", "label": "Deep Research", "estimated_time": "30s"},
                                            {"action": "challenge", "label": "Challenge", "estimated_time": "20s"},
                                            {"action": "skip", "label": "Continue", "estimated_time": ""}
                                        ],
                                        "timestamp": time.time()
                                    }
                                    await research_queue.put(step_complete_event)
                                else:
                                    # Default thinking_stream for other actions
                                    await research_queue.put({
                                        "type": "thinking_stream",
                                        "content": description,
                                        "timestamp": time.time()
                                    })
                                    
                            elif msg_type == "agent_thinking":
                                agent = msg.get("agent", "system")
                                content = msg.get("content", "Processing...")
                                # Store as finding
                                agent_title = agent.title()
                                if agent_title in agent_findings:
                                    agent_findings[agent_title].append(content)
                                await research_queue.put({
                                    "type": "thinking_stream",
                                    "content": f"{agent.capitalize()}: {content}",
                                    "timestamp": time.time()
                                })
                            else:
                                # Pass through other event types
                                await research_queue.put(msg)

                        # Emit initial progress event
                        await research_queue.put({
                            'type': 'thinking_stream', 
                            'content': 'Starting full research protocol...', 
                            'timestamp': time.time()
                        })
                        
                        # Run research in background task
                        research_task = asyncio.create_task(
                            system_integrator.process_query_with_full_integration(
                                query=query,
                                domain=request.get("domain"),
                                progress_callback=research_callback
                            )
                        )
                        
                        # Monitor queue and task
                        try:
                            # 2-minute total timeout tracked from start
                            while True:
                                # Check how much time remains for the global 2-minute timeout
                                elapsed = time.time() - prompt_start_time
                                remaining = 120 - elapsed
                                
                                if remaining <= 0:
                                    logger.error(f"Global research timeout reached (120s)")
                                    research_task.cancel()
                                    raise asyncio.TimeoutError()

                                try:
                                    # Wait for a message with short timeout to toggle
                                    msg = await asyncio.wait_for(research_queue.get(), timeout=0.1)
                                    yield f"data: {json.dumps(msg)}\n\n"
                                except asyncio.TimeoutError:
                                    if research_task.done():
                                        # Task is done, drain any remaining messages in queue
                                        while not research_queue.empty():
                                            m = research_queue.get_nowait()
                                            yield f"data: {json.dumps(m)}\n\n"
                                        break
                                    continue
                            
                            # Final result
                            result = await research_task
                            yield f"data: {json.dumps({'type': 'thinking_stream', 'content': 'Research complete, generating response...', 'timestamp': time.time()})}\n\n"
                            
                        except asyncio.TimeoutError:
                            logger.error(f"Query processing timed out after 120 seconds")
                            yield f"data: {json.dumps({'type': 'thinking_stream', 'content': 'Research timed out. Generating partial response...', 'timestamp': time.time()})}\n\n"
                            result = {
                                "query": query,
                                "results": {
                                    "content": "The research process timed out. Please try using **Fast** or **Chat** mode for quicker responses, or simplify your query.",
                                    "error": "timeout"
                                }
                            }
                        
                        # Stream content
                        content = result.get("results", {}).get("content", "")
                        if content and isinstance(content, str):
                            chunk_delay = 0.0001
                            for i in range(0, len(content)):
                                chunk = content[i]
                                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                                if i + 1 < len(content):
                                    await asyncio.sleep(chunk_delay)
                    
                    # Send done signal
                    logger.info("🌊 SSE stream complete for prompt %s (duration %.2fs)", prompt_id, time.time() - prompt_start_time)
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    logger.error(f"Error in streaming query: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            logger.info("🌊🌊🌊 Returning StreamingResponse with generator")
            response = StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
            logger.info("🌊🌊🌊 StreamingResponse created, returning")
            return response
        
        # Non-streaming response (fallback) - use single agent for chat mode
        if mode == "chat":
            # Use single agent LLM call for chat mode (same as WebSocket)
            from ..config import load_config_with_model
            from ..providers.factory import provider_factory
            
            primary_model = settings.get("primaryModel", "llama3.1:8b")
            temperature = settings.get("temperature", 0.7)
            max_tokens = settings.get("maxTokens", 2000)
            
            logger.info(f"📞 HTTP Chat Mode: model={primary_model}, temp={temperature}, max_tokens={max_tokens}")
            
            # Get conversation ID from request (if provided)
            conversation_id = request.get("conversation_id")
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # DISABLED: Conversation history causes repetitive, pseudo-profound responses
            # Each query should be answered independently based on actual research
            conversation_history = []
            logger.info(f"📝📝📝 HTTP: Conversation history DISABLED - answering query independently")
            
            custom_cfg = load_config_with_model(
                primary_model=primary_model,
                use_small_models=False,
                fast=True
            )
            
            try:
                # ALWAYS use Secretary agent for chat mode (with executor capabilities)
                # This ensures date injection and all enhanced features work
                agent_normalized = str(agent).lower().strip() if agent else "auto"
                # Force Secretary for chat mode regardless of agent selection
                agent_normalized = "secretary"
                logger.info(f"📞 HTTP Chat Mode: Using Secretary agent with executor (forced for chat mode)")
                
                from ..agents.secretary import run as secretary_run
                from ..config import IceburgConfig
                
                # Create config for Secretary - FORCE Ollama provider for local operation
                secretary_cfg = IceburgConfig(
                    data_dir=custom_cfg.data_dir,
                    surveyor_model=custom_cfg.surveyor_model,
                    dissident_model=custom_cfg.dissident_model,
                    synthesist_model=custom_cfg.synthesist_model,
                    oracle_model=custom_cfg.oracle_model,
                    embed_model=custom_cfg.embed_model,
                    llm_provider="ollama",  # Force Ollama for local operation
                    provider_url=custom_cfg.provider_url or "http://localhost:11434",
                    timeout_s=custom_cfg.timeout_s,
                    enable_code_generation=custom_cfg.enable_code_generation,
                    disable_memory=custom_cfg.disable_memory,
                    enable_software_lab=custom_cfg.enable_software_lab,
                    max_context_length=custom_cfg.max_context_length,
                    fast=custom_cfg.fast
                )
                
                user_id = request.get("user_id")
                
                # Get routing decision for this query
                routing_decision_http = router.route(query, context={"mode": mode})
                
                # Call Secretary agent (with executor and date injection)
                # Now async - directly await
                # Now async - directly await (via to_thread since run is sync)
                secretary_response = await asyncio.to_thread(
                    secretary_run,
                    cfg=secretary_cfg,
                    query=query,
                    verbose=False,
                    thinking_callback=None,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    routing_mode=routing_decision_http.mode,
                    mode=mode  # Pass original mode for chat detection
                )
                
                return JSONResponse(content={
                    "content": secretary_response,
                    "thinking": [],
                    "informatics": {
                        "sources": [],
                        "confidence": 0.0,
                        "processing_time": 0.0
                    },
                    "conclusions": []
                })
                
                # OLD CODE PATH BELOW - Only used for non-chat modes
                # (This code should never execute for chat mode, but kept for other modes)
                provider = provider_factory(custom_cfg)
                # Build system prompt based on agent selection (same logic as WebSocket)
                if agent and agent != "auto":
                    if agent == "surveyor":
                        from ..agents.surveyor import SURVEYOR_SYSTEM
                        system_prompt = SURVEYOR_SYSTEM
                    elif agent == "synthesist":
                        from ..agents.synthesist import SYNTHESIST_SYSTEM
                        system_prompt = SYNTHESIST_SYSTEM
                    elif agent == "dissident":
                        from ..agents.dissident import DISSIDENT_SYSTEM
                        system_prompt = DISSIDENT_SYSTEM
                    elif agent == "oracle":
                        from ..agents.oracle import ORACLE_SYSTEM
                        system_prompt = ORACLE_SYSTEM
                    elif agent == "physiology":
                        system_prompt = (
                            "You are ICEBURG Physiology Explainer, a fast, precise assistant in the physiological department.\n"
                            "- Audience: curious students and researchers.\n"
                            "- Task: given a short context about an encyclopedia entry, first produce a 3-point summary in clear, technical but accessible language.\n"
                            "- Each point should be 1–2 sentences and focus on physiological mechanisms, fields/bioelectric aspects, and what is actually known vs theoretical.\n"
                            "- After the 3 bullet points, add one short paragraph (3–5 sentences) that explains how this entry fits into the broader astro‑physiology / field-based model and any key caveats.\n"
                            "- When answering follow‑up questions, stay anchored to the same entry and its physiology, and keep answers concise and evidence oriented.\n"
                            "- If the user asks to combine multiple entries or patterns, explain the connection logically and make clear which parts are established, theoretical, or speculative."
                        )
                    else:
                        system_prompt = (
                            "You are ICEBURG, an AI civilization with deep knowledge decoding capabilities. "
                            "You are designed for truth-finding and comprehensive analysis. Provide clear, insightful answers "
                            "that demonstrate advanced reasoning while being concise and evidence-oriented."
                        )
                else:
                    system_prompt = (
                        "You are ICEBURG, an AI civilization with deep knowledge decoding capabilities. "
                        "You are designed for truth-finding and comprehensive analysis. Provide clear, insightful answers "
                        "that demonstrate advanced reasoning while being concise and evidence-oriented."
                    )
                
                logger.info(f"🎭 HTTP: Using system prompt for agent: {agent}")
                
                # Build full prompt with conversation history (natural style)
                # System prompt already handles identity - don't force it
                agent_normalized_http = str(agent).lower().strip() if agent else "auto"
                
                # DISABLED: Conversation history causes repetitive responses
                # Each query answered independently based on actual research
                full_prompt = query
                logger.info(f"📝 HTTP: Conversation history disabled - answering query independently")
                
                agent_response = await asyncio.to_thread(
                    provider.chat_complete,
                    model=primary_model,
                    prompt=full_prompt,
                    system=system_prompt,
                    temperature=temperature,
                    options={"max_tokens": max_tokens} if max_tokens else None  # Pass max_tokens via options dict
                )
                
                if not agent_response or not agent_response.strip():
                    agent_response = "I received your query but couldn't generate a response. Please try rephrasing your question."
                
                # Save conversation to local persistence (same as WebSocket)
                try:
                    conversation_entry = ConversationEntry(
                        conversation_id=conversation_id,
                        timestamp=datetime.now().isoformat(),
                        user_message=query,
                        assistant_message=agent_response,
                        agent_used="single_llm",
                        mode="chat",
                        metadata={
                            "has_history": len(conversation_history) > 0,
                            "history_count": len(conversation_history)
                        }
                    )
                    local_persistence.save_conversation(conversation_entry)
                    logger.debug(f"💾 HTTP: Saved conversation to memory (ID: {conversation_id})")
                except Exception as e:
                    logger.error(f"Error saving conversation: {e}")
                
                result = {
                    "query": query,
                    "results": {
                        "content": agent_response,
                        "mode": "chat",
                        "agent": "single_llm"
                    }
                }
                
                # Format response
                formatted = response_formatter.format_from_analysis(result.get("results", {}))
            except Exception as e:
                logger.error(f"Error in HTTP chat mode: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
        else:
            # Use full protocol for non-chat modes
            result = await system_integrator.process_query_with_full_integration(
                query=query,
                domain=request.get("domain")
            )
            
            # Format response
            formatted = response_formatter.format_from_analysis(result.get("results", {}))
        
        # Sanitize response to prevent XSS
        if isinstance(formatted, dict):
            for key, value in formatted.items():
                if isinstance(value, str):
                    formatted[key] = sanitize_input(value)
        
        return JSONResponse(content=formatted)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/device/generate")
async def generate_device(request: Dict[str, Any]):
    """Generate device endpoint"""
    try:
        device_type = request.get("device_type", "")
        requirements = request.get("requirements", {})
        domain = request.get("domain")
        
        if not device_type:
            raise HTTPException(status_code=400, detail="device_type is required")
        
        # Generate device with full integration
        device = await system_integrator.generate_device_with_full_integration(
            device_type=device_type,
            requirements=requirements,
            domain=domain
        )
        
        return JSONResponse(content=device)
        
    except Exception as e:
        logger.error(f"Error generating device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def safe_send_json(websocket: WebSocket, message: Dict[str, Any]) -> bool:
    """Safely send JSON message to WebSocket, checking connection state"""
    try:
        # Check if connection is still active
        if websocket not in active_connections:
            logger.debug("WebSocket not in active connections, cannot send")
            return False
        # Check connection state
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.debug(f"WebSocket not connected (state: {websocket.client_state}), cannot send")
            return False
        # Send message
        await websocket.send_json(message)
        return True
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected during send")
        return False
    except Exception as e:
        logger.debug(f"Error sending WebSocket message: {e}")
        return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming with security"""
    # Log connection attempt details for debugging
    origin = websocket.headers.get("origin", "no origin")
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    logger.info(f"WebSocket connection attempt from {client_host}:{client_port}, origin: {origin}")
    
    # PHASE 1.1: Authentication - Verify API key before accepting connection
    # PHASE 2.2: IP Whitelisting - Check IP against whitelist if configured
    is_production = os.getenv("ENVIRONMENT") == "production"
    # Default: require API key only in production, allow all connections in development
    api_key_required = os.getenv("ICEBURG_REQUIRE_API_KEY", "1" if is_production else "0") == "1"
    expected_api_key = os.getenv("ICEBURG_API_KEY")
    
    # In development (non-production), skip API key check if no key is configured
    if not is_production and not expected_api_key:
        api_key_required = False
        logger.debug(f"Development mode: API key check disabled (no API key configured)")
    
    # PHASE 2.2: IP Whitelisting check (before authentication)
    allowed_ips_env = os.getenv("ALLOWED_IPS", "")
    if allowed_ips_env:
        import ipaddress
        allowed_ips = [ip.strip() for ip in allowed_ips_env.split(",") if ip.strip()]
        ip_allowed = False
        
        try:
            client_ip_obj = ipaddress.ip_address(client_host)
            for allowed_ip_str in allowed_ips:
                try:
                    # Support CIDR notation (e.g., 192.168.1.0/24)
                    if '/' in allowed_ip_str:
                        network = ipaddress.ip_network(allowed_ip_str, strict=False)
                        if client_ip_obj in network:
                            ip_allowed = True
                            break
                    else:
                        # Single IP
                        if client_ip_obj == ipaddress.ip_address(allowed_ip_str):
                            ip_allowed = True
                            break
                except ValueError:
                    continue
            
            if not ip_allowed:
                logger.warning(f"WebSocket connection rejected: IP {client_host} not in whitelist")
                security_logger.warning(f"IP_REJECTED: IP={client_host}, Origin={origin}, AllowedIPs={allowed_ips_env}")
                await websocket.close(code=1008, reason="IP not allowed")
                return
        except Exception as e:
            logger.error(f"Error checking IP whitelist: {e}")
            # Continue if IP check fails (fail open for now, but log it)
            security_logger.warning(f"IP_CHECK_ERROR: IP={client_host}, Error={e}")
    
    if api_key_required and expected_api_key:
        # Extract API key from headers or query params
        api_key = None
        auth_header = websocket.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        elif auth_header.startswith("ApiKey "):
            api_key = auth_header[7:]  # Remove "ApiKey " prefix
        else:
            # Try query parameter
            api_key = websocket.query_params.get("api_key")
        
        # Verify API key
        if not api_key or api_key != expected_api_key:
            logger.warning(f"WebSocket connection rejected: Invalid or missing API key from {client_host}")
            # PHASE 2.1: Log security event
            security_logger.warning(f"AUTH_FAILURE: IP={client_host}, Origin={origin}, Reason=Invalid API key")
            await websocket.close(code=1008, reason="Authentication required")
            return
        
        # PHASE 2.1: Log successful authentication
        security_logger.info(f"AUTH_SUCCESS: IP={client_host}, Origin={origin}")
        logger.debug(f"WebSocket authentication successful for {client_host}")
    
    # Check origin in production only
    if is_production:
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
        allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
        if origin and origin not in allowed_origins:
            logger.warning(f"WebSocket connection rejected: origin {origin} not in allowed origins")
            await websocket.close(code=1008, reason="Origin not allowed")
            return
    
    # Accept WebSocket connection
    try:
        # Accept connection - no timeout, let it fail naturally if needed
        await websocket.accept()
        logger.info(f"WebSocket connection accepted from {client_host}:{client_port} (origin: {origin})")
        
        # Integrate WebSocket Manager for improved connection tracking
        from ..api.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        
        # Get conversation ID from query params if provided
        conversation_id = websocket.query_params.get("conversation_id")
        
        # Register connection with manager
        conn_info = await ws_manager.register_connection(websocket, conversation_id)
        await ws_manager.mark_connected(websocket)
        
        # Add to active connections IMMEDIATELY after accept (keep for backward compatibility)
        active_connections.append(websocket)
        # Track connection metadata
        connection_metadata[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "websocket_manager": ws_manager,
            "connection_info": conn_info,
            "client_host": client_host,
            "client_port": client_port,
            "origin": origin
        }
        logger.info(f"Active connections: {len(active_connections)}")
        
        # Clean up stale connections AFTER adding new one (don't remove the one we just added)
        await cleanup_stale_connections()
        
        # PHASE 1.3: Connection Limits - Configurable limit with graceful rejection
        max_connections = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "100"))
        if len(active_connections) >= max_connections:
            logger.warning(f"Connection limit reached ({len(active_connections)}/{max_connections}), rejecting new connection from {client_host}")
            # PHASE 2.1: Log security event
            security_logger.warning(f"CONNECTION_LIMIT: IP={client_host}, Active={len(active_connections)}, Max={max_connections}")
            await websocket.close(code=1008, reason="Server at capacity")
            return
        
        # Clean up stale connections if approaching limit
        if len(active_connections) >= max_connections * 0.8:  # Cleanup at 80% capacity
            logger.debug(f"Approaching connection limit ({len(active_connections)}/{max_connections}), cleaning up stale connections...")
            await cleanup_stale_connections()
        
        # Send initial connection confirmation immediately
        try:
            # Safely check portal status
            always_on_enabled = False
            if portal is not None:
                try:
                    always_on_enabled = getattr(portal, 'initialized', False)
                except:
                    always_on_enabled = False
            
            # Use safe_send_json instead of direct send_json for better error handling
            confirmation_sent = await safe_send_json(websocket, {
                "type": "connected",
                "message": "WebSocket connection established",
                "status": "ready",
                "always_on_enabled": always_on_enabled
            })
            if confirmation_sent:
                logger.info("✅ Sent initial connection confirmation")
            else:
                logger.warning("⚠️ Failed to send initial confirmation (connection may be closing)")
                # Don't close immediately - connection might still be valid
        except Exception as e:
            logger.error(f"❌ Error sending initial confirmation: {e}", exc_info=True)
            # Log but don't close connection - it might still work
            # The connection will be cleaned up naturally if it's actually dead
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection from {client_host}:{client_port}: {e}", exc_info=True)
        try:
            await websocket.close(code=1006, reason="Accept failed")
        except:
            pass
        return
    
    try:
        # Send ping periodically to keep connection alive
        # Use shorter interval for network connections to prevent NAT/firewall timeouts
        async def send_ping():
            ping_interval = 10  # Send ping every 10 seconds (aggressive for network stability)
            consecutive_failures = 0
            max_failures = 3  # Allow 3 consecutive failures before giving up
            
            while True:
                try:
                    await asyncio.sleep(ping_interval)
                    # Check if connection is still active
                    if websocket not in active_connections:
                        logger.debug("WebSocket no longer in active connections, stopping ping")
                        break
                    # Check connection state before sending
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.debug("WebSocket not connected, stopping ping")
                        break
                    
                    # Try to send ping
                    if await safe_send_json(websocket, {"type": "ping"}):
                        consecutive_failures = 0  # Reset failure count on success
                        logger.debug("Sent ping to keep connection alive")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Failed to send ping ({consecutive_failures}/{max_failures})")
                        if consecutive_failures >= max_failures:
                            logger.error("Too many ping failures, connection may be dead")
                            break
                except asyncio.CancelledError:
                    logger.debug("Ping task cancelled")
                    break
                except Exception as e:
                    consecutive_failures += 1
                    logger.debug(f"Ping error: {e} ({consecutive_failures}/{max_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Too many ping errors, stopping ping task")
                        break
                    # Wait before retrying with shorter interval
                    await asyncio.sleep(3)  # Shorter wait before retry
        
        # Start ping task
        ping_task = asyncio.create_task(send_ping())
        
        # Reset retry count for this connection
        websocket._retry_count = 0
        
        while True:
            # CRITICAL: Check connection state before receiving - prevent "not connected" errors
            # But be lenient - allow CONNECTING state briefly after accept
            try:
                state = websocket.client_state
                if state not in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
                    logger.info(f"WebSocket not connected (state: {state}), breaking loop")
                    # Remove from active connections
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
            except Exception as state_error:
                logger.error(f"Error checking connection state: {state_error}, breaking loop")
                if websocket in active_connections:
                    active_connections.remove(websocket)
                if websocket in connection_metadata:
                    del connection_metadata[websocket]
                break
            
            # Verify connection is still in active_connections
            if websocket not in active_connections:
                logger.info("WebSocket not in active_connections, breaking loop")
                break
            
            # Receive message from client with timeout
            try:
                # CRITICAL: Double-check connection state right before receive
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.error("WebSocket not connected right before receive, breaking loop")
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
                
                # Use asyncio.wait_for to handle timeouts gracefully
                # Increased timeout to 300 seconds (5 minutes) to prevent premature disconnections
                # Network connections may have longer idle periods
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                
                # Reset retry count on successful receive
                websocket._retry_count = 0
            except asyncio.TimeoutError:
                # Send ping to check if connection is still alive
                try:
                    # Track ping with WebSocket manager if available
                    if websocket in connection_metadata:
                        ws_manager = connection_metadata[websocket].get("websocket_manager")
                        if ws_manager:
                            await ws_manager.record_ping(websocket)
                    
                    # Check connection state before sending
                    if not await safe_send_json(websocket, {"type": "ping"}):
                        logger.info("WebSocket not connected, breaking loop")
                        break
                    logger.debug("Sent ping to check connection")
                    # Wait for pong or next message (shorter timeout for ping response)
                    try:
                        response = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                        if response == "pong" or response.strip() == "pong":
                            logger.debug("Received pong, connection alive")
                            
                            # Track pong with WebSocket manager if available
                            if websocket in connection_metadata:
                                ws_manager = connection_metadata[websocket].get("websocket_manager")
                                if ws_manager:
                                    await ws_manager.record_pong(websocket)
                            
                            continue
                    except asyncio.TimeoutError:
                        logger.debug("No pong received, but connection might still be alive")
                        continue
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected during ping")
                    break
                except Exception as e:
                    logger.info(f"WebSocket connection timeout: {e}")
                    break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected normally")
                # Mark as disconnected in WebSocket manager
                if websocket in connection_metadata:
                    ws_manager = connection_metadata[websocket].get("websocket_manager")
                    if ws_manager:
                        await ws_manager.mark_disconnected(websocket)
                break
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Error receiving WebSocket message: {e}")
                
                # CRITICAL: Break immediately on connection errors - don't create infinite loop
                if "not connected" in error_str or "need to call" in error_str or "accept" in error_str:
                    logger.error("WebSocket not properly connected, breaking loop immediately")
                    # Remove from active connections
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
                
                if "disconnect" in error_str or "closed" in error_str:
                    logger.info("Connection closed, breaking loop")
                    # Remove from active connections
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
                
                # Check connection state - if not connected, break immediately
                try:
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.error(f"WebSocket not connected (state: {websocket.client_state}), breaking loop")
                        # Remove from active connections
                        if websocket in active_connections:
                            active_connections.remove(websocket)
                        if websocket in connection_metadata:
                            del connection_metadata[websocket]
                        break
                except Exception as state_error:
                    logger.error(f"Error checking connection state: {state_error}, breaking loop")
                    # Remove from active connections
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
                
                # Only continue if connection is still valid and error is recoverable
                # Limit retries to prevent infinite loops
                retry_count = getattr(websocket, '_retry_count', 0)
                if retry_count >= 3:
                    logger.error(f"Too many retries ({retry_count}), breaking loop")
                    if websocket in active_connections:
                        active_connections.remove(websocket)
                    if websocket in connection_metadata:
                        del connection_metadata[websocket]
                    break
                
                websocket._retry_count = retry_count + 1
                logger.warning(f"Non-fatal error receiving message: {e}, retrying ({retry_count + 1}/3)")
                await asyncio.sleep(0.5)  # Wait before retry
                continue
            
            # Handle ping/pong
            if data == "pong" or data.strip() == "pong":
                # Track pong with WebSocket manager if available
                if websocket in connection_metadata:
                    ws_manager = connection_metadata[websocket].get("websocket_manager")
                    if ws_manager:
                        await ws_manager.record_pong(websocket)
                        await ws_manager.update_activity(websocket)
                continue
            
            # Validate JSON
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                # Check if it's a ping
                if data.strip() == "ping":
                    await safe_send_json(websocket, {"type": "pong"})
                    continue
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON"
                })
                continue
            
            # Handle ping message type
            if message.get("type") == "ping":
                await safe_send_json(websocket, {"type": "pong"})
                continue
            
            query = message.get("query", "")
            mode = message.get("mode", "chat")
            # Handle "fast" mode explicitly - CONVERT EARLY for fast path checks
            if mode == "fast":
                mode = "chat"
                degradation_mode = False  # Force fast mode
            else:
                degradation_mode = message.get("degradation_mode", False)
            agent = message.get("agent", "auto")
            
            # Determine fast mode BEFORE using it
            is_fast_mode = mode == "fast" or mode == "chat" or not degradation_mode
            
            # Skip validation for ping/pong messages
            if message.get("type") in ["ping", "pong"]:
                continue
            
            # Only validate if query is present and not empty
            if not query or query.strip() == "":
                # Don't send error for empty queries - just ignore them
                logger.debug("Received empty query, ignoring")
                continue
            
            # PHASE 1.4: Request Size Limits - Enforce before processing
            max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "10000"))  # Default 10k chars
            if len(query) > max_query_length:
                logger.warning(f"Query too long ({len(query)} chars, max {max_query_length}) from {client_host}")
                # PHASE 2.1: Log security event
                security_logger.warning(f"QUERY_SIZE_LIMIT: IP={client_host}, Length={len(query)}, Max={max_query_length}")
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Query too long. Maximum length is {max_query_length} characters."
                })
                continue
            
            # No hardcoded fast paths - let LLM handle everything naturally
            
            # Validate and sanitize query (only for non-simple queries)
            is_valid, error_msg = validate_query(query)
            if not is_valid:
                # Only send error if query was actually provided
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": error_msg or "Invalid query"
                })
                continue
            
            query = sanitize_input(query)
            mode = sanitize_input(str(mode), max_length=50) if mode else "chat"
            agent = sanitize_input(str(agent), max_length=50) if agent else "auto"

            # FAST PATH: Instant response for simple queries
            simple_queries = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
            if query.lower().strip() in simple_queries:
                logger.info(f"🚀 FAST PATH: Instant response for '{query}'")
                responses = {
                    "hi": "Hello! How can I help you today?",
                    "hello": "Hello! How can I help you today?",
                    "hey": "Hey there! What's on your mind?",
                    "thanks": "You're very welcome!",
                    "thank you": "You're very welcome!",
                    "bye": "Goodbye! Have a great day!",
                    "goodbye": "Goodbye! Have a great day!"
                }
                response_text = responses.get(query.lower().strip(), "Hello! How can I help you today?")
                
                # Stream the response instantly
                for chunk in response_text:
                    await safe_send_json(websocket, {"type": "chunk", "content": chunk})
                
                await safe_send_json(websocket, {
                    "type": "done",
                    "metadata": {"mode": "fast_path", "processing_time": 0.001}
                })
                continue
            
            # Track prompt start time and ID for telemetry
            prompt_start_time = time.time()
            prompt_id = str(uuid.uuid4())
            
            # Get settings from frontend (model selection, temperature, etc.)
            frontend_settings = message.get("settings", {})
            primary_model = frontend_settings.get("primaryModel", "llama3.1:8b")
            temperature = frontend_settings.get("temperature", 0.7)
            max_tokens = frontend_settings.get("maxTokens", 2000)
            
            # Determine if we should use small models (for fast chat)
            # Fast mode models (1B-3B): ultra-fast for chat
            fast_mode_models = [
                "llama3.2:1b", "phi3.5", "gemma2:2b", 
                "qwen2.5:1.5b", "llama3.1:3b"
            ]
            use_small_models = primary_model in fast_mode_models or max_tokens < 1000
            
            # Log settings for debugging
            logger.info(f"Frontend settings: model={primary_model}, temp={temperature}, tokens={max_tokens}, small={use_small_models}, fast_mode={is_fast_mode}, mode={mode}")
            
            # PHASE 1.4: Validate file uploads if present (with size limits)
            files = message.get("files", [])
            if files:
                max_file_size = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # Default 10MB
                for file_data in files:
                    filename = file_data.get("name", "")
                    file_type = file_data.get("type", "")
                    file_size = file_data.get("size", 0)
                    
                    # Enforce file size limit
                    if file_size > max_file_size:
                        logger.warning(f"File too large ({file_size} bytes, max {max_file_size}) from {client_host}")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": f"File too large. Maximum size is {max_file_size / 1024 / 1024}MB."
                        })
                        continue
                    
                    is_valid, error_msg = validate_file_upload(filename, file_type, file_size)
                    if not is_valid:
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": error_msg or "Invalid file"
                        })
                        continue
            
            # Skip old thinking message - we use thinking_stream instead
            # initial_thinking = get_context_message(mode=mode, agent=agent) or format_thinking_message(mode=mode)
            # if not await safe_send_json(websocket, {
            #     "type": "thinking",
            #     "content": initial_thinking
            # }):
            #     logger.warning("WebSocket not connected, breaking loop")
            #     break
            
            # Run prompt interpreter in background (non-blocking)
            # CHAT/FAST MODE: Skip prompt interpreter entirely for maximum speed
            # Prompt interpreter adds 0-3s delay and blocks response generation
            intent_analysis = None
            prompt_interpreter_start = time.time()
            
            # Skip prompt interpreter for chat and fast modes to improve response time
            should_skip_prompt_interpreter = (
                mode == "chat" or 
                mode == "fast" or 
                is_fast_mode or
                not degradation_mode
            )
            
            if should_skip_prompt_interpreter:
                logger.info(f"{mode.upper()} MODE: Skipping prompt interpreter for maximum speed")
                # Set minimal intent analysis for fast mode
                intent_analysis = {
                    "intent": "query",
                    "complexity": "simple",
                    "domains": [],
                    "cache_status": {"cache_hit": False}
                }
                # Record performance metric (skipped operation)
                # Note: Performance tracking is handled by UnifiedPerformanceTracker
                # which is already integrated into the system
            else:
                from ..agents.prompt_interpreter import run as prompt_interpreter_run
                from ..config import load_config_with_model
                
                # Use the selected model from frontend settings for prompt interpreter
                interpreter_cfg = load_config_with_model(
                    primary_model=primary_model,
                    use_small_models=use_small_models,
                    fast=is_fast_mode
                )
                try:
                    # Send action status (frontend already shows this, but update it)
                    action_desc = format_action_message(action="prompt_interpreter", mode=mode)
                    await safe_send_json(websocket, {
                        "type": "action",
                        "action": "prompt_interpreter",
                        "status": "processing",
                        "description": action_desc
                    })
                    
                    # Create callback for streaming word breakdown
                    word_breakdown_results = []
                    algorithm_steps = []
                    
                    async def stream_breakdown_callback(data: Dict[str, Any]):
                        """Callback to stream word breakdown results"""
                        word_breakdown_results.append(data)
                        # Stream to frontend with proper structure
                        # Include the original data type in the message
                        message = {
                            "type": "word_breakdown",
                            **data  # This includes step, word, morphological, etc.
                        }
                        if await safe_send_json(websocket, message):
                            logger.info(f"✅ Sent word breakdown: {data.get('type', 'unknown')} - {data.get('step') or data.get('word', 'N/A')}")
                        else:
                            logger.debug("Failed to send word breakdown, connection may be closed")
                    
                    # Run prompt interpreter with async support for streaming
                    # Standard mode: Use engine (cached) for speed
                    # The engine will quickly retrieve cached word analyses
                    try:
                        # Use shorter timeout for fast mode
                        timeout = 2.0
                        intent_analysis = await asyncio.wait_for(
                            prompt_interpreter_run(
                                interpreter_cfg, 
                                query, 
                                verbose=False,
                                stream_breakdown=stream_breakdown_callback
                            ),
                            timeout=timeout
                        )
                        if mode == "fast" or not degradation_mode:
                            logger.info("FAST MODE: Prompt interpreter completed (using cached engine)")
                    except asyncio.TimeoutError:
                        logger.warning(f"Prompt interpreter timed out after {timeout} seconds")
                        intent_analysis = None
                    except Exception as e:
                        logger.error(f"Prompt interpreter error: {e}", exc_info=True)
                        try:
                            await safe_send_json(websocket, {
                                "type": "action",
                                "action": "prompt_interpreter",
                                "status": "error",
                                "description": f"Prompt interpreter error: {str(e)}"
                            })
                        except Exception as send_error:
                            logger.error(f"Failed to send error message: {send_error}")
                        intent_analysis = None

                    # Stream intent analysis results
                    if intent_analysis:
                        intent_info = intent_analysis.get("intent_analysis", {})
                        domain_info = intent_analysis.get("domain_analysis", {})
                        complexity_info = intent_analysis.get("complexity_analysis", {})

                        # Include word breakdown in action result
                        word_breakdown = intent_analysis.get("word_breakdown", [])
                        algorithm_pipeline = intent_analysis.get("algorithm_pipeline", [])

                        # Get cache status from analysis
                        cache_status = intent_analysis.get("cache_status", {})
                        is_cached = cache_status.get("cache_hit", False)
                        response_time = cache_status.get("response_time", "analyzing")

                        action_desc = format_action_message(action="prompt_interpreter", description="Understanding your request", mode=mode)
                        if is_cached:
                            action_desc += " (instant - cached)"

                        await safe_send_json(websocket, {
                            "type": "action",
                            "action": "prompt_interpreter",
                            "status": "complete",
                            "description": action_desc,
                            "intent": intent_info.get("primary", "general"),
                            "confidence": intent_info.get("confidence", 0.5),
                            "domain": domain_info.get("primary", "general"),
                            "complexity": complexity_info.get("score", 0.5),
                            "routing": intent_analysis.get("agent_routing", {}).get("recommended_path", "standard"),
                            "word_breakdown_count": len(word_breakdown),
                            "algorithm_steps": len(algorithm_pipeline),
                            "cache_status": cache_status,
                            "is_cached": is_cached,
                            "response_time": response_time
                        })
                    
                    # prompt_start_time and prompt_id already set above
                except Exception as e:
                    logger.error(f"Prompt interpreter error: {e}", exc_info=True)
                    # Send error notification but continue
                    try:
                        await safe_send_json(websocket, {
                            "type": "action",
                            "action": "prompt_interpreter",
                            "status": "error",
                            "description": f"Prompt interpreter error: {str(e)}"
                        })
                    except Exception as send_error:
                        logger.error(f"Failed to send error message: {send_error}")
                    # Continue without intent analysis
                    intent_analysis = None
            
            # Thinking already sent above, skip duplicate
            
            # Handle degradation mode (slower processing)
            # GPT-5 Optimized: Character-by-character streaming with near-zero delay
            # M4 Optimized: Ultra-fast streaming for M4 chip
            # M4 Neural Engine: 30-50% faster, leverage for near-instant responses
            sleep_delay = 0.05 if degradation_mode else 0.0001  # Near-zero delay (GPT-5 speed)
            chunk_delay = 0.02 if degradation_mode else 0.0001  # Character-by-character streaming (GPT-5 speed)
            
            # Fast path already handled above (moved before thinking message)
            
            # Try portal architecture if enabled (always-on AI) - FAST PATH ONLY
            # DISABLED for chat mode - chat mode uses Secretary agent (LLM) only, no portal
            portal_available = portal is not None and mode == "fast" and mode != "chat"
            if portal_available:
                try:
                    # Only try portal for very simple queries to avoid delays
                    simple_queries = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
                    is_simple = query.lower().strip() in simple_queries
                    
                    # Skip portal for complex queries - go straight to normal processing
                    if not is_simple and len(query.split()) > 5:
                        logger.debug("Skipping portal for complex query, using normal processing")
                        portal_available = False
                    
                    if portal_available:
                        # Ensure portal is initialized (with timeout)
                        if not portal.initialized:
                            logger.warning("Portal not initialized yet, initializing now...")
                            try:
                                # Initialize with timeout to avoid hanging
                                await asyncio.wait_for(portal.initialize(), timeout=5.0)
                            except asyncio.TimeoutError:
                                logger.error("Portal initialization timed out, skipping portal")
                                portal_available = False
                            except Exception as init_error:
                                logger.error(f"Failed to initialize portal: {init_error}", exc_info=True)
                                portal_available = False
                        
                        if portal_available and portal.initialized:
                            # Get user_id from message or use default
                            user_id = message.get("user_id", "default")
                            
                            # Open portal with timeout to avoid hanging
                            try:
                                portal_response = await asyncio.wait_for(
                                    portal.open_portal(user_id, query, context=message),
                                    timeout=2.0  # 2 second timeout for portal
                                )
                            except asyncio.TimeoutError:
                                logger.warning("Portal response timed out, using normal processing")
                                portal_available = False
                            
                            if portal_available and portal_response:
                                # Check if portal handled the query (and didn't error)
                                if portal_response.get("response") and portal_response.get("source") != "error":
                                    source = portal_response.get("source", "unknown")
                                    response_time = portal_response.get("response_time", 0)
                                    
                                    # Only use portal if response is fast (<1s) or from local persona
                                    if response_time < 1.0 or source == "local_persona":
                                        logger.info(f"Portal response from {source} in {response_time:.3f}s")
                                        
                                        # Stream response
                                        response_content = portal_response.get("response", "")
                                        if response_content:
                                            # Stream response in chunks
                                            chunk_size = 30 if mode == "fast" else 10
                                            for i in range(0, len(response_content), chunk_size):
                                                chunk = response_content[i:i + chunk_size]
                                                await safe_send_json(websocket, {
                                                    "type": "chunk",
                                                    "content": chunk
                                                })
                                                await asyncio.sleep(chunk_delay)
                                            
                                            # Send done signal
                                            await safe_send_json(websocket, {
                                                "type": "done",
                                                "metadata": {
                                                    "source": source,
                                                    "response_time": response_time,
                                                    "layer": portal_response.get("metadata", {}).get("layer", "unknown")
                                                }
                                            })
                                            continue
                                    else:
                                        logger.debug(f"Portal too slow ({response_time:.3f}s), using normal processing")
                                elif portal_response.get("source") == "error":
                                    # Portal had an error, log it and fall through to normal processing
                                    logger.warning(f"Portal returned error: {portal_response.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error in portal processing: {e}", exc_info=True)
                    # Fall through to existing processing - don't crash the server
            
            # Process query based on mode and agent selection (existing flow - backward compatible)
            # Wrap all query processing in a timeout to prevent hanging
            try:
                query_timeout = 120.0  # 2 minutes max for query processing
                
                # v5: Route query to appropriate handler
                from ..router.request_router import get_request_router
                router = get_request_router()
                routing_decision = router.route(query, context={"mode": mode})
                logger.info(f"🔍 v5 Routing (WS): query='{query[:50]}' -> mode={routing_decision.mode} (confidence={routing_decision.confidence:.2f})")
                
                # PRESERVE user's original mode selection for chat detection
                user_original_mode = mode  # Store before any override
                
                # Override mode with routing decision ONLY if user selected a non-chat/non-research mode
                # If user selected "fast", "chat", or "research", keep that mode to enable proper behavior
                # "research" mode should use the full research pipeline with interactive step cards
                PRESERVE_MODE_SELECTIONS = {"fast", "chat", "research"}
                if routing_decision.mode in ["web_research", "local_rag", "hybrid"] and mode not in PRESERVE_MODE_SELECTIONS:
                    mode = routing_decision.mode
                    logger.info(f"🔍 v5 Mode override: using {mode} for query")
                else:
                    logger.info(f"🔍 Preserving user mode selection: {mode} (routing suggested: {routing_decision.mode})")
                
                # PRIORITY: Chat mode should be checked FIRST (before other modes)
                # This ensures single-agent mode is used for chat, regardless of agent setting
                # NOTE: "research" mode uses the FULL PIPELINE with engines/algorithms/informatics display
                # FULL_PIPELINE_MODES: These modes go through system_integrator with full engine/algorithm tracking
                FULL_PIPELINE_MODES = {"research", "deep_research", "unbounded", "astrophysiology", "prediction_lab"}
                # SINGLE_AGENT_MODES: These modes use Secretary agent for fast chat responses
                SINGLE_AGENT_MODES = {"chat", "fast", "web_research", "local_rag", "hybrid"}
                
                if mode in FULL_PIPELINE_MODES:
                    # FULL PIPELINE MODE - Uses system_integrator with engines/algorithms/informatics
                    logger.info(f"🔬 FULL PIPELINE MODE: mode={mode}, agent={agent}, query={query[:50]}")
                    # This will fall through to the full protocol section below
                    pass
                elif mode in SINGLE_AGENT_MODES:  # Single agent mode for chat and v5 search modes
                    # SINGLE AGENT CHAT MODE - Optimized for speed
                    # Flow: Immediate thinking → LLM call (parallel with optional etymology)
                    logger.info(f"🎯 SINGLE AGENT CHAT MODE (PRIORITY): agent={agent} (type: {type(agent).__name__}), mode={mode}, query={query[:50]}")
                    logger.debug(f"🎯 Raw agent value from message: {repr(agent)}")
                    try:
                        # Set prompt_start_time for single agent mode (needed for processing_time calculation)
                        prompt_start_time = time.time()
                        prompt_id = str(uuid.uuid4())
                        
                        # No fast path - let Secretary agent (LLM) handle everything naturally
                        
                        from ..config import load_config_with_model
                    
                        # Use custom config with frontend model selection
                        # Chat-like modes that should use Secretary with Ollama
                        CHAT_MODES = {"fast", "chat", "web_research", "local_rag", "hybrid"}
                        is_fast_mode = mode in CHAT_MODES or not degradation_mode
                        # For chat modes, use Ollama (local) provider
                        if mode in CHAT_MODES:
                            # Use a small, fast model for local operation
                            if not primary_model or "gemini" in primary_model.lower() or "gpt" in primary_model.lower():
                                primary_model = "llama3.1:8b"  # Fast, small model
                        custom_cfg = load_config_with_model(
                            primary_model=primary_model,
                            use_small_models=use_small_models or is_fast_mode,
                            fast=is_fast_mode
                        )
                        # ALWAYS force Ollama provider for chat modes (clear singleton if needed)
                        if mode in CHAT_MODES:
                            from ..providers.factory import clear_provider_singleton
                            clear_provider_singleton()  # Clear any cached Google provider
                            from ..config import IceburgConfig
                            custom_cfg = IceburgConfig(
                                data_dir=custom_cfg.data_dir,
                                surveyor_model=custom_cfg.surveyor_model,
                                dissident_model=custom_cfg.dissident_model,
                                synthesist_model=custom_cfg.synthesist_model,
                                oracle_model=custom_cfg.oracle_model,
                                embed_model=custom_cfg.embed_model,
                                llm_provider="ollama",  # Force Ollama
                                provider_url=custom_cfg.provider_url,
                                timeout_s=custom_cfg.timeout_s,
                                enable_code_generation=custom_cfg.enable_code_generation,
                                disable_memory=custom_cfg.disable_memory,
                                enable_software_lab=custom_cfg.enable_software_lab,
                                max_context_length=custom_cfg.max_context_length,
                                fast=custom_cfg.fast
                            )
                        
                        # STEP 1: Skip old thinking message - we use thinking_stream instead
                        # await safe_send_json(websocket, {
                        #     "type": "thinking",
                        #     "content": "Preparing response..."
                        # })
                        
                        # STEP 1.5: Retrieve conversation history for context
                        # Get or create conversation ID for this WebSocket connection
                        # FIX: Try to reuse conversation ID from previous connection (check query params)
                        if websocket not in connection_metadata:
                            connection_metadata[websocket] = {}
                        if "conversation_id" not in connection_metadata[websocket]:
                            # Check if frontend sent a conversation_id in query params
                            conversation_id_param = websocket.query_params.get("conversation_id")
                            if conversation_id_param:
                                connection_metadata[websocket]["conversation_id"] = conversation_id_param
                                logger.info(f"📚 Reusing conversation ID from frontend: {conversation_id_param[:8]}...")
                            else:
                                connection_metadata[websocket]["conversation_id"] = str(uuid.uuid4())
                                logger.info(f"📚 Created new conversation ID: {connection_metadata[websocket]['conversation_id'][:8]}...")
                        
                        conversation_id = connection_metadata[websocket]["conversation_id"]
                        
                        # SMART Conversation History: Use last 2-3 exchanges for context, but filter repetitive patterns
                        # This provides context without causing repetitive responses
                        conversation_history = []
                        try:
                            # Get last 2-3 exchanges from this conversation for context
                            recent_conversations = local_persistence.get_conversations(
                                conversation_id=conversation_id,
                                limit=3  # Only last 3 exchanges
                            )
                            
                            # Filter out repetitive patterns and build context
                            for conv in reversed(recent_conversations):  # Oldest first
                                user_msg = conv.get("user_message", "")
                                assistant_msg = conv.get("assistant_message", "")
                                
                                # Skip if messages are too similar (repetitive pattern)
                                if len(conversation_history) > 0:
                                    last_assistant = conversation_history[-1].get("content", "")
                                    if assistant_msg and last_assistant:
                                        # Simple similarity check - skip if >80% similar
                                        similarity = len(set(assistant_msg.split()) & set(last_assistant.split())) / max(len(set(assistant_msg.split())), len(set(last_assistant.split())), 1)
                                        if similarity > 0.8:
                                            logger.debug(f"⏭️ Skipping repetitive message (similarity: {similarity:.2f})")
                                            continue
                                
                                # Add to history if not empty
                                if user_msg and assistant_msg:
                                    conversation_history.append({
                                        "role": "user",
                                        "content": user_msg
                                    })
                                    conversation_history.append({
                                        "role": "assistant",
                                        "content": assistant_msg
                                    })
                            
                            if conversation_history:
                                logger.info(f"📝 Using {len(conversation_history)//2} previous exchanges for context")
                            else:
                                logger.info(f"📝 No previous conversation history for context")
                        except Exception as e:
                            logger.debug(f"Error loading conversation history: {e}")
                            conversation_history = []
                        
                        # STEP 2: Start LLM call IMMEDIATELY (don't wait for etymology)
                        from ..providers.factory import provider_factory
                        from ..providers.circuit_breaker_wrapper import wrap_provider_with_circuit_breaker
                        from ..api.error_messages import format_error_for_user
                        
                        try:
                            provider = provider_factory(custom_cfg)
                            
                            # Wrap provider with circuit breaker for reliability
                            provider_name = getattr(custom_cfg, "llm_provider", "unknown") or os.getenv("ICEBURG_LLM_PROVIDER", "unknown")
                            provider = wrap_provider_with_circuit_breaker(
                                provider,
                                provider_name=provider_name,
                                fallback_provider=None,  # Can add fallback provider later
                                config={"failure_threshold": 5, "success_threshold": 2, "timeout": 60.0}
                            )
                            
                            logger.debug(f"✅ Provider created with circuit breaker: {type(provider).__name__}")
                        except Exception as provider_error:
                            logger.error(f"❌ Failed to create provider: {provider_error}", exc_info=True)
                            
                            # Use user-friendly error messages
                            error_message = format_error_for_user(provider_error, context="provider_initialization")
                            
                            # Send user-friendly error message
                            await safe_send_json(websocket, {
                                "type": "error",
                                "message": error_message
                            })
                            
                            await safe_send_json(websocket, {
                                "type": "done",
                                "metadata": {"mode": "chat", "error": "provider_init_failed"}
                            })
                            continue
                        
                        # CHAT MODE: Secretary is the ONLY agent - no exceptions, no fallbacks
                        # Force secretary regardless of what frontend sends
                        agent_normalized = "secretary"
                        logger.info(f"👔 CHAT MODE: Forcing Secretary agent (only agent for chat mode)")
                        
                        # Secretary agent - fast, friendly, knows ICEBURG, no VectorStore needed
                        # This is the ONLY path for chat mode
                        if True:  # Always true - secretary is mandatory for chat mode
                            logger.info(f"👔 Secretary agent selected - fast chat mode, no VectorStore needed")
                            try:
                                from ..config import load_config_with_model
                                from ..agents.secretary import run as secretary_run
                                
                                is_fast_mode = mode == "fast" or mode == "chat" or not degradation_mode
                                secretary_cfg = load_config_with_model(
                                    primary_model=primary_model,
                                    use_small_models=use_small_models or is_fast_mode,
                                    fast=is_fast_mode
                                )
                                
                                # FORCE Ollama provider for local operation (same as HTTP endpoint)
                                from ..config import IceburgConfig
                                secretary_cfg = IceburgConfig(
                                    data_dir=secretary_cfg.data_dir,
                                    surveyor_model=secretary_cfg.surveyor_model,
                                    dissident_model=secretary_cfg.dissident_model,
                                    synthesist_model=secretary_cfg.synthesist_model,
                                    oracle_model=secretary_cfg.oracle_model,
                                    embed_model=secretary_cfg.embed_model,
                                    llm_provider="ollama",  # Force Ollama for local operation
                                    provider_url=secretary_cfg.provider_url or "http://localhost:11434",
                                    timeout_s=secretary_cfg.timeout_s,
                                    enable_code_generation=secretary_cfg.enable_code_generation,
                                    disable_memory=secretary_cfg.disable_memory,
                                    enable_software_lab=secretary_cfg.enable_software_lab,
                                    max_context_length=secretary_cfg.max_context_length,
                                    fast=secretary_cfg.fast
                                )
                                
                                # Create thinking callback for streaming
                                def thinking_callback(message: str):
                                    """Sync callback - will be called from thread"""
                                    # Queue for async processing
                                    if not hasattr(thinking_callback, 'queue'):
                                        import queue
                                        thinking_callback.queue = queue.Queue()
                                    try:
                                        thinking_callback.queue.put(message, block=False)
                                    except:
                                        pass
                                
                                # Process thinking messages in background
                                import queue
                                thinking_queue = queue.Queue()
                                thinking_task = None
                                
                                async def process_secretary_thinking():
                                    while True:
                                        try:
                                            message = thinking_queue.get_nowait()
                                            await safe_send_json(websocket, {
                                                "type": "thinking_stream",
                                                "content": message,
                                                "timestamp": time.time()
                                            })
                                        except queue.Empty:
                                            await asyncio.sleep(0.1)
                                            continue
                                
                                thinking_task = asyncio.create_task(process_secretary_thinking())
                                
                                # Get conversation_id and user_id from message or connection metadata
                                conversation_id = message.get("conversation_id")
                                if not conversation_id and websocket in connection_metadata:
                                    conn_info = connection_metadata[websocket].get("connection_info", {})
                                    conversation_id = conn_info.get("conversation_id")
                                if not conversation_id:
                                    conversation_id = str(uuid.uuid4())
                                
                                user_id = message.get("user_id")
                                
                                # Call Secretary agent (no VectorStore needed!)
                                logger.info(f"🔍🔍🔍 WS: Calling secretary_run with query='{query[:100]}' (length={len(query)}), conversation_id={conversation_id[:20] if conversation_id else None}")
                                
                                # Pass routing_mode to agent (fixes double routing)
                                routing_mode_to_pass = routing_decision.mode if routing_decision else None
                                
                                # CRITICAL: Use user_original_mode for chat detection in secretary
                                # This ensures "fast" or "chat" selection triggers chat behavior
                                # user_original_mode is defined earlier in this block before any mode override
                                mode_for_secretary = user_original_mode
                                
                                # Wrap with global middleware if available
                                if global_middleware:
                                    secretary_response = await global_middleware.execute_agent(
                                        agent_name="secretary",
                                        agent_func=secretary_run,
                                        cfg=secretary_cfg,
                                        query=query,
                                        verbose=False,
                                        thinking_callback=thinking_callback,
                                        conversation_id=conversation_id,
                                        user_id=user_id,
                                        routing_mode=routing_mode_to_pass,
                                        mode=mode_for_secretary  # Pass original user mode for chat detection
                                    )
                                else:
                                    # Directly await since run() is now async
                                    secretary_response = await secretary_run(
                                        secretary_cfg,
                                        query,
                                        verbose=False,
                                        thinking_callback=thinking_callback,
                                        conversation_id=conversation_id,
                                        user_id=user_id,
                                        routing_mode=routing_mode_to_pass,
                                        mode=mode_for_secretary  # Pass original user mode for chat detection
                                    )
                                
                                # Cancel thinking task
                                if thinking_task:
                                    thinking_task.cancel()
                                    try:
                                        await thinking_task
                                    except asyncio.CancelledError:
                                        pass
                                
                                if secretary_response:
                                    # Stream response character-by-character
                                    chunk_delay = 0.0001 if is_fast_mode else 0.02
                                    for i in range(0, len(secretary_response)):
                                        chunk = secretary_response[i]
                                        if not await safe_send_json(websocket, {
                                            "type": "chunk",
                                            "content": chunk
                                        }):
                                            break
                                        if i + 1 < len(secretary_response):
                                            await asyncio.sleep(chunk_delay)
                                    
                                    logger.info("✅ Secretary response sent successfully")
                                    
                                    await safe_send_json(websocket, {
                                        "type": "done",
                                        "metadata": {
                                            "mode": "chat",
                                            "agent": "secretary",
                                            "processing_time": time.time() - prompt_start_time
                                        }
                                    })
                                    continue
                                else:
                                    # Secretary failed - send error, NO fallback to other agents
                                    logger.error("❌ Secretary returned empty response - chat mode requires secretary")
                                    error_msg = format_error_for_user(
                                        Exception("Secretary agent returned empty response"),
                                        context="secretary_failure"
                                    )
                                    await safe_send_json(websocket, {
                                        "type": "error",
                                        "message": error_msg
                                    })
                                    await safe_send_json(websocket, {
                                        "type": "done",
                                        "metadata": {
                                            "mode": "chat",
                                            "agent": "secretary",
                                            "error": "empty_response"
                                        }
                                    })
                                    continue
                            except Exception as e:
                                # Secretary failed - send error, NO fallback to other agents
                                logger.error(f"❌ Secretary agent error in chat mode: {e}", exc_info=True)
                                error_msg = format_error_for_user(e, context="secretary_failure")
                                await safe_send_json(websocket, {
                                    "type": "error",
                                    "message": error_msg
                                })
                                await safe_send_json(websocket, {
                                    "type": "done",
                                    "metadata": {
                                        "mode": "chat",
                                        "agent": "secretary",
                                        "error": str(e)
                                    }
                                })
                                continue
                    except Exception as e:
                        # Chat mode error handling
                        logger.error(f"❌ Error in chat mode: {e}", exc_info=True)
                        error_msg = format_error_for_user(e, context="chat_mode_error")
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": error_msg
                        })
                        await safe_send_json(websocket, {
                            "type": "done",
                            "metadata": {"mode": "chat", "error": str(e)}
                        })
                        continue
                
                # NON-CHAT MODES CONTINUE HERE
                if mode == "device":
                    # Device generation mode
                    thinking_msg = get_context_message(mode="device", action="device_generator")
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": thinking_msg
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    device_type = query  # Use query as device type
                    device = await asyncio.wait_for(
                        system_integrator.generate_device_with_full_integration(
                            device_type=device_type,
                            requirements={},
                            domain=message.get("domain")
                        ),
                        timeout=query_timeout
                    )
                    result = {
                        "query": query,
                        "results": {
                            "content": device.get("specification", "Device generation completed"),
                            "device": device,
                            "mode": "device"
                        }
                    }
                elif mode == "truth":
                    # Truth finding mode - enhanced suppression detection
                    thinking_msg = get_context_message(mode="truth", action="suppression_detector")
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": thinking_msg
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    result = await asyncio.wait_for(
                        system_integrator.process_query_with_full_integration(
                            query=query,
                            domain=message.get("domain")
                        ),
                        timeout=query_timeout
                    )
                    # Emphasize truth-finding aspects
                    result["mode"] = "truth"
                    result["results"]["truth_finding"] = True
                elif mode == "research":
                    # Research mode - focus on methodology and insights
                    thinking_msg = get_context_message(mode="research")
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": thinking_msg
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    result = await asyncio.wait_for(
                        system_integrator.process_query_with_full_integration(
                            query=query,
                            domain=message.get("domain")
                        ),
                        timeout=query_timeout
                    )
                    result["mode"] = "research"
                elif mode == "astrophysiology":
                    # Astro-Physiology mode - truth-finding with celestial biological framework
                    from ..modes.astrophysiology_handler import handle_astrophysiology_query
                    from ..config import load_config_with_model
                    from ..vectorstore import VectorStore
                    
                    # Use custom config with frontend model selection
                    custom_cfg = load_config_with_model(
                        primary_model=primary_model,
                        use_small_models=use_small_models,
                        fast=False  # Use full models for astro-physiology
                    )
                    
                    # Initialize VectorStore for agent processing
                    try:
                        vs = VectorStore(custom_cfg)
                    except Exception as e:
                        logger.warning(f"Error initializing VectorStore for astro-physiology: {e}, continuing without vector store")
                        vs = None
                    
                    # Create WebSocket callback for streaming
                    async def websocket_callback(msg):
                        return await safe_send_json(websocket, msg)
                    
                    # Call dedicated handler
                    result = await asyncio.wait_for(
                        handle_astrophysiology_query(
                            query=query,
                            message=message,
                            cfg=custom_cfg,
                            vs=vs,
                            websocket_callback=websocket_callback,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=query_timeout
                    )
                    result["mode"] = "astrophysiology"
                elif mode == "prediction_lab":
                    # Prediction Lab mode - full protocol with celestial modeling
                    thinking_msg = get_context_message(mode="prediction_lab", action="prediction_analysis") or "Running prediction algorithms... Analyzing weather patterns and celestial influences..."
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": thinking_msg
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    # Use full protocol with custom config for prediction lab
                    from ..config import load_config_with_model
                    is_fast_mode = False  # Always use full protocol for predictions
                    custom_cfg = load_config_with_model(
                        primary_model=primary_model,
                        use_small_models=False,  # Use full models for predictions
                        fast=False
                    )
                    result = await asyncio.wait_for(
                        system_integrator.process_query_with_full_integration(
                            query=query,
                            domain=message.get("domain"),
                            custom_config=custom_cfg,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ),
                        timeout=query_timeout
                    )
                    result["mode"] = "prediction_lab"
                elif mode == "swarm":
                    # Swarm mode - use micro-agent swarm
                    thinking_msg = get_context_message(mode="swarm", action="swarm_creation")
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": thinking_msg
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    swarm = MicroAgentSwarm()
                    await swarm.start_swarm()
                    task_id = await swarm.submit_task(
                        task_type="research",
                        input_data={"query": query, "domain": message.get("domain")},
                        requirements=["research", "analysis"],
                        priority=7
                    )
                    swarm_result = await swarm.get_task_result(task_id, timeout=60)
                    result = {
                        "query": query,
                        "results": {
                            "content": str(swarm_result) if swarm_result else "Swarm processing completed",
                            "swarm_result": swarm_result,
                            "mode": "swarm"
                        }
                    }
                # Chat mode is now handled FIRST (priority check at line 1204)
                # This duplicate block removed - chat mode already processed above
                elif mode == "research" or mode == "deep_research":
                    # RESEARCH MODE - Full pipeline with engines/algorithms/informatics display
                    # This is the original ICEBURG research experience with glitch effects and system status
                    logger.info(f"🔬 RESEARCH MODE: Using full pipeline with engines/algorithms tracking")
                    
                    from ..config import load_config_with_model, IceburgConfig
                    from ..providers.factory import clear_provider_singleton
                    
                    # Clear any cached Google provider
                    clear_provider_singleton()
                    
                    base_cfg = load_config_with_model(
                        primary_model=primary_model or "llama3.1:8b",
                        use_small_models=False,  # Use full models for research
                        fast=False
                    )
                    
                    # Force Ollama provider for research mode
                    custom_cfg = IceburgConfig(
                        data_dir=base_cfg.data_dir,
                        surveyor_model=base_cfg.surveyor_model,
                        dissident_model=base_cfg.dissident_model,
                        synthesist_model=base_cfg.synthesist_model,
                        oracle_model=base_cfg.oracle_model,
                        embed_model=base_cfg.embed_model,
                        llm_provider="ollama",  # Force Ollama
                        provider_url=base_cfg.provider_url,
                        timeout_s=base_cfg.timeout_s,
                        enable_code_generation=base_cfg.enable_code_generation,
                        disable_memory=base_cfg.disable_memory,
                        enable_software_lab=base_cfg.enable_software_lab,
                        max_context_length=base_cfg.max_context_length,
                        fast=False
                    )
                    
                    # Send initial research mode status
                    if not await safe_send_json(websocket, {
                        "type": "thinking",
                        "content": "Initializing research protocol..."
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                    
                    # Create progress callback for real-time engine/algorithm updates
                    async def research_progress_callback(msg: Dict[str, Any]):
                        """Send real-time progress updates during research"""
                        msg_type = msg.get("type", "action")
                        
                        if msg_type == "engines":
                            # Send engines update with glitch effect trigger
                            await safe_send_json(websocket, {
                                "type": "engines",
                                "engines": msg.get("engines", [])
                            })
                        elif msg_type == "algorithms":
                            await safe_send_json(websocket, {
                                "type": "algorithms", 
                                "algorithms": msg.get("algorithms", [])
                            })
                        elif msg_type == "action":
                            # Transform to thinking_stream for glitch effect
                            await safe_send_json(websocket, {
                                "type": "thinking_stream",
                                "content": msg.get("description", "Processing...")
                            })
                        elif msg_type == "agent_thinking":
                            await safe_send_json(websocket, {
                                "type": "thinking_stream",
                                "content": f"{msg.get('agent', 'Agent')}: {msg.get('content', 'Analyzing...')}"
                            })
                    
                    # Run full integration with progress callback
                    result = await asyncio.wait_for(
                        system_integrator.process_query_with_full_integration(
                            query=query,
                            domain=message.get("domain"),
                            custom_config=custom_cfg,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            progress_callback=research_progress_callback
                        ),
                        timeout=query_timeout
                    )
                    result["mode"] = mode
                    
                elif agent == "auto":
                    # Chat mode - Intelligent routing with fast/deep paths (FALLBACK for non-chat modes)
                    try:
                        from ..config import load_config_with_model
                        from ..integration.reflexive_routing import ReflexiveRoutingSystem
                    
                        # Use custom config with frontend model selection
                        # Set fast=True for fast/chat mode
                        is_fast_mode = mode == "fast" or mode == "chat" or not degradation_mode
                        custom_cfg = load_config_with_model(
                            primary_model=primary_model,
                            use_small_models=use_small_models or is_fast_mode,  # Use small models in fast mode
                            fast=is_fast_mode
                        )
                        
                        # Initialize intelligent routing with config
                        routing_system = ReflexiveRoutingSystem(cfg=custom_cfg)
                        routing_decision = routing_system.route_query(query)
                        
                        # Route based on complexity
                        if routing_decision.complexity_score < 0.3 and not degradation_mode:
                            # Fast path: Simple query, use reflexive routing
                            fast_path_msg = get_context_message(mode="chat", action="fast_path") or "Finding the fastest answer"
                            if not await safe_send_json(websocket, {
                                "type": "thinking",
                                "content": fast_path_msg
                            }):
                                logger.warning("WebSocket not connected, breaking loop")
                                break
                            
                            try:
                                # Send immediate feedback before processing (helps with warmup delay)
                                processing_msg = format_thinking_message(mode="chat")
                                await safe_send_json(websocket, {
                                    "type": "thinking",
                                    "content": processing_msg
                                })
                                
                                # Add timeout to prevent hanging (15 seconds for fast path)
                                reflexive_response = await asyncio.wait_for(
                                    routing_system.process_reflexive(query),
                                    timeout=15.0
                                )
                                if not reflexive_response.escalation_recommended:
                                    result = {
                                        "query": query,
                                        "results": {
                                            "content": reflexive_response.response,
                                            "mode": "fast_path",
                                            "confidence": reflexive_response.confidence,
                                            "processing_time": reflexive_response.processing_time
                                        }
                                    }
                                    result["mode"] = "chat"
                                else:
                                    # Escalate to full protocol
                                    escalation_msg = get_context_message(mode="chat", action="deep_path") or "Diving deeper into your question"
                                    if not await safe_send_json(websocket, {
                                        "type": "thinking",
                                        "content": escalation_msg
                                    }):
                                        logger.warning("WebSocket not connected, breaking loop")
                                        break
                                    result = await asyncio.wait_for(
                                        system_integrator.process_query_with_full_integration(
                                            query=query,
                                            domain=message.get("domain"),
                                            custom_config=custom_cfg,
                                            temperature=temperature,
                                            max_tokens=max_tokens
                                        ),
                                        timeout=query_timeout
                                    )
                                    result["mode"] = "chat"
                            except asyncio.TimeoutError:
                                # Fast path timed out, escalate to full protocol with shorter timeout
                                logger.warning(f"Fast path timed out after 15 seconds, escalating to full protocol")
                                timeout_msg = get_context_message(mode="chat", action="deep_path") or "Using comprehensive analysis"
                                if not await safe_send_json(websocket, {
                                    "type": "thinking",
                                    "content": timeout_msg
                                }):
                                    logger.warning("WebSocket not connected, breaking loop")
                                    break
                                try:
                                    # Use shorter timeout for escalation (30 seconds max)
                                    result = await asyncio.wait_for(
                                        system_integrator.process_query_with_full_integration(
                                            query=query,
                                            domain=message.get("domain"),
                                            custom_config=custom_cfg,
                                            temperature=temperature,
                                            max_tokens=max_tokens
                                        ),
                                        timeout=30.0  # Shorter timeout for escalation
                                    )
                                    result["mode"] = "chat"
                                except asyncio.TimeoutError:
                                    # Even escalation timed out, send fallback response
                                    logger.error("Full protocol escalation also timed out after 30 seconds")
                                    result = {
                                        "query": query,
                                        "results": {
                                            "content": "I apologize, but the query processing timed out. This may indicate the LLM models are not responding. Please try a simpler query or check your LLM provider configuration.",
                                            "mode": "error",
                                            "error": "timeout"
                                        }
                                    }
                                except Exception as e:
                                    # Escalation failed, send fallback response
                                    logger.error(f"Full protocol escalation failed: {e}")
                                    result = {
                                        "query": query,
                                        "results": {
                                            "content": f"I encountered an error processing your query: {str(e)}. Please try again.",
                                            "mode": "error",
                                            "error": str(e)
                                        }
                                    }
                            except Exception as e:
                                # Fallback to full protocol on error
                                logger.warning(f"Fast path failed, using full protocol: {e}")
                                result = await asyncio.wait_for(
                                    system_integrator.process_query_with_full_integration(
                                        query=query,
                                        domain=message.get("domain"),
                                        custom_config=custom_cfg,
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    ),
                                    timeout=query_timeout
                                )
                                result["mode"] = "chat"
                        else:
                            # Deep path: Complex query, use full protocol
                            deep_path_msg = get_context_message(mode="chat", action="deep_path") or "Diving deeper into your question"
                            if not await safe_send_json(websocket, {
                                "type": "thinking",
                                "content": deep_path_msg
                            }):
                                logger.warning("WebSocket not connected, breaking loop")
                                break
                            # Create progress callback to stream action/thinking updates
                            async def progress_callback(update: Dict[str, Any]):
                                """Callback to send progress updates during processing"""
                                await safe_send_json(websocket, update)
                            
                            result = await asyncio.wait_for(
                                system_integrator.process_query_with_full_integration(
                                    query=query,
                                    domain=message.get("domain"),
                                    custom_config=custom_cfg,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    progress_callback=progress_callback
                                ),
                                timeout=query_timeout
                            )
                            result["mode"] = "chat"
                    except Exception as e:
                        # Fallback to full protocol on initialization error
                        logger.error(f"Error initializing routing system: {e}", exc_info=True)
                        fallback_msg = get_context_message(mode="chat", action="deep_path") or "Using comprehensive analysis"
                        if not await safe_send_json(websocket, {
                            "type": "thinking",
                            "content": fallback_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        # Use default config if custom config failed
                        from ..config import load_config
                        default_cfg = load_config()
                        result = await asyncio.wait_for(
                            system_integrator.process_query_with_full_integration(
                                query=query,
                                domain=message.get("domain"),
                                custom_config=default_cfg,
                                temperature=temperature,
                                max_tokens=max_tokens
                            ),
                            timeout=query_timeout
                        )
                        result["mode"] = "chat"
                else:
                    # Single agent mode
                    from ..config import load_config, load_config_with_model
                    from ..vectorstore import VectorStore
                
                    # Use custom config with frontend model selection
                    # Set fast=True for fast/chat mode
                    # Use smaller models in fast mode for speed
                    is_fast_mode = mode == "fast" or mode == "chat" or not degradation_mode
                    cfg = load_config_with_model(
                        primary_model=primary_model,
                        use_small_models=use_small_models or is_fast_mode,  # Use small models in fast mode
                        fast=is_fast_mode
                    )
                    vs = VectorStore(cfg)
                    
                    if agent == "surveyor":
                        from ..agents.surveyor import run as surveyor_run
                        agent_result = surveyor_run(cfg, vs, query, verbose=True)
                        thinking_msg = format_thinking_message(agent="surveyor", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "surveyor",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "surveyor"
                            }
                        }
                    elif agent == "dissident":
                        from ..agents.surveyor import run as surveyor_run
                        from ..agents.dissident import run as dissident_run
                        surveyor_result = surveyor_run(cfg, vs, query, verbose=True)
                        thinking_msg = format_thinking_message(agent="dissident", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "dissident",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        if global_middleware:
                            agent_result = await global_middleware.execute_agent(
                                agent_name="dissident",
                                agent_func=dissident_run,
                                cfg=cfg,
                                query=query,
                                surveyor_result=surveyor_result,
                                verbose=True
                            )
                        else:
                            agent_result = dissident_run(cfg, query, surveyor_result, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "dissident"
                            }
                        }
                    elif agent == "synthesist":
                        from ..agents.synthesist import run as synthesist_run
                        thinking_msg = format_thinking_message(agent="synthesist", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "synthesist",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Synthesist needs surveyor and dissident outputs
                        surveyor_result = surveyor_run(cfg, vs, query, verbose=True)
                        dissident_result = dissident_run(cfg, query, surveyor_result, verbose=True)
                        # Synthesist takes enhanced_context as dict with surveyor and dissident
                        enhanced_context = {
                            "surveyor": surveyor_result,
                            "dissident": dissident_result
                        }
                        if global_middleware:
                            agent_result = await global_middleware.execute_agent(
                                agent_name="synthesist",
                                agent_func=synthesist_run,
                                cfg=cfg,
                                enhanced_context=enhanced_context,
                                verbose=True
                            )
                        else:
                            agent_result = synthesist_run(cfg, enhanced_context, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "synthesist"
                            }
                        }
                    elif agent == "oracle":
                        from ..agents.oracle import run as oracle_run
                        from ..graph_store import KnowledgeGraph
                        kg = KnowledgeGraph(cfg)
                        thinking_msg = format_thinking_message(agent="oracle", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "oracle",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Oracle needs evidence-weighted input (use query as evidence)
                        if global_middleware:
                            agent_result = await global_middleware.execute_agent(
                                agent_name="oracle",
                                agent_func=oracle_run,
                                cfg=cfg,
                                kg=kg,
                                vs=vs,
                                query=query,
                                verbose=True
                            )
                        else:
                            agent_result = oracle_run(cfg, kg, vs, query, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "oracle"
                            }
                        }
                    elif agent == "archaeologist":
                        from ..agents.archaeologist import run as archaeologist_run
                        thinking_msg = format_thinking_message(agent="archaeologist", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "archaeologist",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Archaeologist can work with or without documents
                        agent_result = archaeologist_run(cfg, query, documents=None, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "archaeologist"
                            }
                        }
                    elif agent == "supervisor":
                        from ..agents.supervisor import run as supervisor_run
                        thinking_msg = format_thinking_message(agent="supervisor", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "supervisor",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Supervisor needs stage_outputs (create from query)
                        stage_outputs = {
                            "query": query,
                            "stage": "validation"
                        }
                        agent_result = supervisor_run(cfg, stage_outputs, query=query, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "supervisor"
                            }
                        }
                    elif agent == "scribe":
                        from ..agents.scribe import run as scribe_run
                        from ..agents.oracle import run as oracle_run
                        from ..graph_store import KnowledgeGraph
                        kg = KnowledgeGraph(cfg)
                        thinking_msg = format_thinking_message(agent="scribe", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "scribe",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Scribe needs oracle output first
                        oracle_result = oracle_run(cfg, kg, vs, query, verbose=True)
                        agent_result = scribe_run(cfg, oracle_result, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "scribe"
                            }
                        }
                    elif agent == "weaver":
                        from ..agents.weaver import run as weaver_run
                        from ..agents.oracle import run as oracle_run
                        from ..graph_store import KnowledgeGraph
                        kg = KnowledgeGraph(cfg)
                        thinking_msg = format_thinking_message(agent="weaver", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "weaver",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Weaver needs oracle output first
                        oracle_result = oracle_run(cfg, kg, vs, query, verbose=True)
                        agent_result = weaver_run(cfg, oracle_result, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "weaver"
                            }
                        }
                    elif agent == "scrutineer":
                        from ..agents.scrutineer import run as scrutineer_run
                        thinking_msg = format_thinking_message(agent="scrutineer", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "scrutineer",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        # Scrutineer needs synthesis (use synthesist output)
                        from ..agents.surveyor import run as surveyor_run
                        from ..agents.dissident import run as dissident_run
                        from ..agents.synthesist import run as synthesist_run
                        surveyor_result = surveyor_run(cfg, vs, query, verbose=True)
                        dissident_result = dissident_run(cfg, query, surveyor_result, verbose=True)
                        enhanced_context = {
                            "surveyor": surveyor_result,
                            "dissident": dissident_result
                        }
                        synthesist_result = synthesist_run(cfg, enhanced_context, verbose=True)
                        agent_result = scrutineer_run(cfg, synthesist_result, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "scrutineer"
                            }
                        }
                    elif agent == "ide":
                        # IDE Agent - safe command execution and code editing
                        from ..agents.ide_agent import run as ide_agent_run
                        thinking_msg = format_thinking_message(agent="ide", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "ide",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        agent_result = ide_agent_run(cfg, vs, query, verbose=True)
                        result = {
                            "query": query,
                            "results": {
                                "content": agent_result,
                                "agent": "ide"
                            }
                        }
                    elif agent == "swarm":
                        thinking_msg = format_thinking_message(agent="swarm", mode=mode)
                        if not await safe_send_json(websocket, {
                            "type": "agent_thinking",
                            "agent": "swarm",
                            "content": thinking_msg
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                        await asyncio.sleep(sleep_delay)
                        swarm = MicroAgentSwarm()
                        await swarm.start_swarm()
                        task_id = await swarm.submit_task(
                            task_type="research",
                            input_data={"query": query, "domain": message.get("domain")},
                            requirements=["research", "analysis"],
                            priority=7
                        )
                        swarm_result = await swarm.get_task_result(task_id, timeout=60)
                        result = {
                            "query": query,
                            "results": {
                                "content": str(swarm_result) if swarm_result else "Swarm processing completed",
                                "agent": "swarm"
                            }
                        }
                    else:
                        # Default to full protocol
                        result = await asyncio.wait_for(
                            system_integrator.process_query_with_full_integration(
                                query=query,
                                domain=message.get("domain")
                            ),
                            timeout=query_timeout
                        )
            
            except asyncio.TimeoutError:
                logger.error(f"Query processing timed out after {query_timeout} seconds")
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Query processing timed out. The LLM may not be responding. Please check your LLM provider configuration."
                })
                # Send a basic response so the user knows what happened
                result = {
                    "query": query,
                    "results": {
                        "content": "Query processing timed out. This usually means the LLM models aren't responding. Please check:\n1. Is your LLM provider configured correctly?\n2. Is the API key set? (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)\n3. Try a simpler query first.",
                        "mode": mode,
                        "error": "timeout"
                    }
                }
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                error_message = str(e)[:500]  # Limit error message length
                await safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Error processing query: {error_message}"
                })
                # Send a basic response so the user knows what happened
                result = {
                    "query": query,
                    "results": {
                        "content": f"I encountered an error processing your query: {error_message}. Please try rephrasing your question or check if the system is properly configured.",
                        "mode": mode,
                        "error": error_message
                    }
                }
            
            # Ensure result is never None - create fallback if needed
            if not result:
                logger.warning("Result is None, creating fallback response")
                result = {
                    "query": query,
                    "results": {
                        "content": "I apologize, but I encountered an issue processing your query. Please try again.",
                        "mode": "error"
                    }
                }
            
            # Ensure results dict exists
            if not result.get("results"):
                result["results"] = {}
            
            # Handle fast path vs full protocol results differently
            if result.get("results", {}).get("mode") == "fast_path":
                # Fast path: Simple response structure
                content = result.get("results", {}).get("content", "")
                
                # Ensure content is a string
                if content and not isinstance(content, str):
                    content = str(content)
                
                # If content is empty, provide fallback
                if not content or not content.strip():
                    content = "I received your query but couldn't generate a response. Please try rephrasing your question."
                    logger.warning("Fast path returned empty content, using fallback")
                
                # Process with Reflex Agent (compress and extract bullets)
                reflex_response = None
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    try:
                        reflex_response = reflex_agent.compress_response(content)
                        
                        # Send 3-bullet preview first (immediate value)
                        if reflex_response and reflex_response.preview:
                            await safe_send_json(websocket, {
                                "type": "reflex_preview",
                                "bullets": reflex_response.preview,
                                "compression_ratio": reflex_response.compression_ratio
                            })
                            
                            # Store reflections in knowledge tree (async, non-blocking)
                            if reflex_response.reflections:
                                try:
                                    # Store reflections for knowledge tree growth
                                    for reflection in reflex_response.reflections:
                                        try:
                                            local_persistence.update_knowledge(
                                                f"reflection_{uuid.uuid4().hex[:8]}",
                                                reflection
                                            )
                                        except Exception as e:
                                            logger.debug(f"Error storing individual reflection: {e}")
                                except Exception as e:
                                    logger.debug(f"Error storing reflections: {e}")
                    except Exception as e:
                        logger.warning(f"Reflex agent processing failed: {e}, using original content")
                        reflex_response = None
                
                # Stream content immediately (GPT-5-style character-by-character streaming)
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    # GPT-5 speed: Character-by-character or word-by-word for maximum responsiveness
                    if degradation_mode:
                        chunk_size = 20  # Slower mode: word chunks
                    else:
                        # Fast mode: Character-by-character like GPT-5
                        # Stream word-by-word for natural flow (faster than char-by-char but still instant)
                        chunk_size = 1  # Character-by-character for GPT-5-like precision
                    
                    try:
                        # Stream character-by-character for GPT-5-like precision
                        for i in range(0, len(content)):
                            chunk = content[i]
                            if not await safe_send_json(websocket, {
                                "type": "chunk",
                                "content": chunk
                            }):
                                logger.warning("WebSocket not connected, breaking loop")
                                break
                            # Near-zero delay for GPT-5-like instant streaming
                            # Only delay between characters (not after last)
                            if i + 1 < len(content):
                                await asyncio.sleep(chunk_delay)  # 0.0001s = 10,000 chars/sec (GPT-5 speed)
                    except Exception as e:
                        logger.error(f"Error streaming fast path content: {e}")
                        # Send error message but continue to send done
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": f"Error streaming response: {str(e)}"
                        })
                else:
                    # Send fallback message if no content
                    await safe_send_json(websocket, {
                        "type": "chunk",
                        "content": content if content else "I apologize, but I couldn't generate a response. Please try again."
                    })
                
                # Track prompt metrics before sending done
                response_time = time.time() - prompt_start_time
                try:
                    metrics = PromptMetrics(
                        prompt_id=prompt_id,
                        prompt_text=query[:200],  # Truncate for privacy
                        response_time=response_time,
                        token_count=len(content) if content else 0,
                        model_used=mode,
                        success=True,
                        quality_score=0.8  # Default quality score
                    )
                    telemetry.track_prompt(metrics)
                except Exception as e:
                    logger.error(f"Error tracking prompt metrics: {e}")
                
                # Save conversation to local persistence
                try:
                    conversation_entry = ConversationEntry(
                        conversation_id=str(uuid.uuid4()),
                        timestamp=datetime.now().isoformat(),
                        user_message=query,
                        assistant_message=content,
                        agent_used=agent,
                        mode=mode,
                        metadata={
                            "processing_time": time.time() - prompt_start_time,
                            "prompt_id": prompt_id
                        }
                    )
                    local_persistence.save_conversation(conversation_entry)
                except Exception as e:
                    logger.error(f"Error saving conversation: {e}")
                
                # Always send done signal, even if there were errors
                try:
                    # Include monitoring status if available
                    monitoring_data = None
                    if bottleneck_monitor:
                        try:
                            monitoring_data = {
                                "enabled": True,
                                "status": bottleneck_monitor.get_monitoring_status()
                            }
                        except Exception as e:
                            logger.debug(f"Could not get monitoring status: {e}")
                    
                    # Include full result for astro-physiology mode so frontend can display truth-finding components
                    done_message = {
                        "type": "done",
                        "monitoring": monitoring_data
                    }
                    
                    # Add result data for astro-physiology mode
                    if mode == "astrophysiology" and result:
                        done_message["mode"] = "astrophysiology"
                        done_message["results"] = result.get("results", {})
                    
                    await safe_send_json(websocket, done_message)
                except Exception as e:
                    logger.error(f"Error sending done signal: {e}")
            else:
                # Full protocol: Structured response
                # Send thinking items from methodology
                methodology = result.get("results", {}).get("methodology", {})
                if isinstance(methodology, dict):
                    thinking_steps = methodology.get("steps", [])
                    if isinstance(thinking_steps, list):
                        for step in thinking_steps[:5]:  # Limit to 5 thinking items
                            if isinstance(step, dict):
                                if not await safe_send_json(websocket, {
                                    "type": "thinking",
                                    "content": step.get("name", "Processing...")
                                }):
                                    logger.warning("WebSocket not connected, breaking loop")
                                    break
                            elif isinstance(step, str):
                                if not await safe_send_json(websocket, {
                                    "type": "thinking",
                                    "content": step
                                }):
                                    logger.warning("WebSocket not connected, breaking loop")
                                    break
                            await asyncio.sleep(sleep_delay)
                
                # Send engines and algorithms being used
                engines_used = result.get("results", {}).get("engines_used", [])
                algorithms_used = result.get("results", {}).get("algorithms_used", [])
                
                if engines_used and isinstance(engines_used, list):
                    if not await safe_send_json(websocket, {
                        "type": "engines",
                        "engines": engines_used
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                
                if algorithms_used and isinstance(algorithms_used, list):
                    if not await safe_send_json(websocket, {
                        "type": "algorithms",
                        "algorithms": algorithms_used
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                
                # Send sources if available
                sources = result.get("results", {}).get("sources", [])
                if not sources and result.get("citation_id"):
                    # Try to get sources from citation tracker
                    try:
                        citation = system_integrator.source_citation_tracker.get_citation_by_id(result.get("citation_id"))
                        if citation:
                            sources = citation.get("sources", [])
                    except:
                        pass
                
                if sources and isinstance(sources, list):
                    # Enhance sources with evidence grades and reliability scores
                    enhanced_sources = []
                    for source in sources:
                        enhanced_source = source.copy()
                        
                        # Get evidence level if available
                        evidence_level = source.get("evidence_level", "")
                        if evidence_level:
                            evidence_labels = {
                                "A": "Well-Established",
                                "B": "Plausible",
                                "C": "Speculative",
                                "S": "Suppressed but Valid",
                                "X": "Actively Censored"
                            }
                            enhanced_source["evidence_label"] = evidence_labels.get(evidence_level.upper(), evidence_level)
                        
                        # Get reliability score if available from citation tracker
                        try:
                            source_id = source.get("url") or source.get("id") or source.get("title", "")
                            if source_id and hasattr(system_integrator, 'source_citation_tracker'):
                                tracked_source = system_integrator.source_citation_tracker.sources.get(source_id)
                                if tracked_source:
                                    enhanced_source["reliability_score"] = tracked_source.get("reliability_score", 0.0)
                                    enhanced_source["reliability_grade"] = tracked_source.get("reliability_grade", "C")
                        except Exception as e:
                            logger.debug(f"Error getting reliability score: {e}")
                        
                        enhanced_sources.append(enhanced_source)
                    
                    if not await safe_send_json(websocket, {
                        "type": "sources",
                        "sources": enhanced_sources
                    }):
                        logger.warning("WebSocket not connected, breaking loop")
                        break
                
                # Send informatics
                try:
                    insights = result.get("results", {}).get("insights", {})
                    if insights and isinstance(insights, dict):
                        # Safely extract cross_domain_connections
                        cross_domain = insights.get("cross_domain_connections", [])
                        # Ensure it's a list and doesn't contain unhashable types
                        if isinstance(cross_domain, list):
                            # Filter out any unhashable types (like slice objects)
                            cross_domain = [item for item in cross_domain if isinstance(item, (str, int, float, dict, list))]
                        else:
                            cross_domain = []
                        
                        if not await safe_send_json(websocket, {
                            "type": "informatics",
                            "data": {
                                "sources": cross_domain,
                                "confidence": 0.9,
                                "methodology": "enhanced_deliberation"
                            }
                        }):
                            logger.warning("WebSocket not connected, breaking loop")
                            break
                except Exception as e:
                    logger.error(f"Error sending informatics: {e}")
                    # Continue without informatics
                
                # Stream response content
                results_dict = result.get("results", {})
                content = results_dict.get("content", "")
                
                # If no content, try to extract from agent results
                if not content:
                    agent_results = results_dict.get("agent_results", {})
                    if agent_results:
                        # Use supervisor or synthesist result as content
                        content = (
                            agent_results.get("supervisor", "") or
                            agent_results.get("synthesist", "") or
                            agent_results.get("oracle", "") or
                            str(agent_results)
                        )
                
                # Ensure content is a string
                if content and not isinstance(content, str):
                    content = str(content)
                
                # If content is empty, provide fallback
                if not content or not content.strip():
                    content = "I received your query but couldn't generate a response. Please try rephrasing your question."
                    logger.warning("Full protocol returned empty content, using fallback")
                
                # Process with Reflex Agent (compress and extract bullets)
                reflex_response = None
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    try:
                        reflex_response = reflex_agent.compress_response(content)
                        
                        # Send 3-bullet preview first (immediate value)
                        if reflex_response and reflex_response.preview:
                            await safe_send_json(websocket, {
                                "type": "reflex_preview",
                                "bullets": reflex_response.preview,
                                "compression_ratio": reflex_response.compression_ratio
                            })
                            
                            # Store reflections in knowledge tree (async, non-blocking)
                            if reflex_response.reflections:
                                try:
                                    # Store reflections for knowledge tree growth
                                    for reflection in reflex_response.reflections:
                                        try:
                                            local_persistence.update_knowledge(
                                                f"reflection_{uuid.uuid4().hex[:8]}",
                                                reflection
                                            )
                                        except Exception as e:
                                            logger.debug(f"Error storing individual reflection: {e}")
                                except Exception as e:
                                    logger.debug(f"Error storing reflections: {e}")
                    except Exception as e:
                        logger.warning(f"Reflex agent processing failed: {e}, using original content")
                        reflex_response = None
                
                # Stream content in chunks (much faster for ICEBURG experience)
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    # Larger chunks for faster streaming (ICEBURG style)
                    chunk_size = 50 if not degradation_mode else 20
                    try:
                        for i in range(0, len(content), chunk_size):
                            chunk = content[i:i+chunk_size]
                            # Ensure chunk is JSON-serializable
                            if isinstance(chunk, str):
                                if not await safe_send_json(websocket, {
                                    "type": "chunk",
                                    "content": chunk
                                }):
                                    logger.warning("WebSocket not connected, breaking loop")
                                    break
                                # Minimal delay for near-instant streaming
                                if i + chunk_size < len(content):  # Don't delay after last chunk
                                    await asyncio.sleep(chunk_delay)
                    except Exception as e:
                        logger.error(f"Error streaming content chunks: {e}")
                        # Send error message but continue to send done
                        await safe_send_json(websocket, {
                            "type": "error",
                            "message": f"Error streaming content: {str(e)}"
                        })
                else:
                    # Send fallback message if no content
                    await safe_send_json(websocket, {
                        "type": "chunk",
                        "content": content if content else "I apologize, but I couldn't generate a response. Please try again."
                    })
                
                # Send conclusions
                # Extract conclusions from results_dict or use format_from_analysis
                conclusions = []
                try:
                    if isinstance(results_dict, dict):
                        # Try to get conclusions directly
                        conclusions = results_dict.get("conclusions", [])
                        # Ensure it's a list
                        if not isinstance(conclusions, list):
                            conclusions = []
                        
                        # If not found, try format_from_analysis
                        if not conclusions:
                            try:
                                formatted = response_formatter.format_from_analysis(results_dict)
                                if isinstance(formatted, dict):
                                    extracted = formatted.get("conclusions", [])
                                    if isinstance(extracted, list):
                                        conclusions = extracted
                            except Exception as e:
                                logger.debug(f"Error formatting conclusions: {e}")
                except Exception as e:
                    logger.error(f"Error extracting conclusions: {e}")
                    conclusions = []
                
                # Stream conclusions
                if isinstance(conclusions, list):
                    for conclusion in conclusions:
                        if isinstance(conclusion, str):
                            try:
                                if not await safe_send_json(websocket, {
                                    "type": "conclusion",
                                    "content": conclusion
                                }):
                                    logger.warning("WebSocket not connected, breaking loop")
                                    break
                                await asyncio.sleep(sleep_delay)
                            except Exception as e:
                                logger.error(f"Error sending conclusion: {e}")
                                break
                
                # Track prompt metrics before sending done
                response_time = time.time() - prompt_start_time
                try:
                    # Get content length for token count
                    content_length = 0
                    content = ""
                    agent_results = {}
                    
                    if result and result.get("results"):
                        content = result.get("results", {}).get("content", "")
                        if content:
                            content_length = len(content)
                        
                        # Extract agent results for quality calculation
                        agent_results = result.get("results", {}).get("agent_results", {})
                    
                    # Calculate dynamic quality score
                    from ...utils.quality_calculator import calculate_quality_score
                    quality_score = calculate_quality_score(
                        response_text=content,
                        query_text=query,
                        agent_results=agent_results,
                        response_time=response_time,
                        metadata={}
                    )
                    
                    metrics = PromptMetrics(
                        prompt_id=prompt_id,
                        prompt_text=query[:200],  # Truncate for privacy
                        response_time=response_time,
                        token_count=content_length,
                        model_used=mode,
                        success=True,
                        quality_score=quality_score  # Dynamic quality score
                    )
                    telemetry.track_prompt(metrics)
                    
                    # Log full conversation for fine-tuning (with quality score)
                    try:
                        from ...data_collection.fine_tuning_logger import FineTuningLogger
                        fine_tuning_logger = FineTuningLogger()
                        
                        # Build full messages (not truncated)
                        full_messages = []
                        # Add system message if available
                        if result and result.get("results", {}).get("methodology"):
                            methodology = result.get("results", {}).get("methodology", {})
                            if isinstance(methodology, dict) and methodology.get("methodology"):
                                system_msg = f"You are ICEBURG, an AI civilization with {methodology.get('methodology', 'enhanced deliberation')} methodology."
                                full_messages.append({"role": "system", "content": system_msg})
                        
                        full_messages.append({"role": "user", "content": query})
                        full_messages.append({"role": "assistant", "content": content})
                        
                        metadata = {
                            "conversation_id": conversation_entry.conversation_id if 'conversation_entry' in locals() else str(uuid.uuid4()),
                            "mode": mode,
                            "agent": agent,
                            "model": primary_model or mode,
                            "response_time": response_time,
                            "quality_score": quality_score,
                            "agent_results": list(agent_results.keys()) if agent_results else []
                        }
                        
                        fine_tuning_logger.log_conversation(full_messages, metadata, quality_score=quality_score)
                    except Exception as e:
                        logger.debug(f"Failed to log conversation for fine-tuning: {e}")
                except Exception as e:
                    logger.error(f"Error tracking prompt metrics: {e}")
                
                # Save conversation and research to local persistence
                try:
                    # Extract content from result
                    result_content = ""
                    if result and result.get("results"):
                        result_content = result.get("results", {}).get("content", "")
                        if not result_content:
                            agent_results = result.get("results", {}).get("agent_results", {})
                            if agent_results:
                                result_content = (
                                    agent_results.get("supervisor", "") or
                                    agent_results.get("synthesist", "") or
                                    agent_results.get("oracle", "") or
                                    str(agent_results)
                                )
                    
                    conversation_entry = ConversationEntry(
                        conversation_id=str(uuid.uuid4()),
                        timestamp=datetime.now().isoformat(),
                        user_message=query,
                        assistant_message=result_content or str(result),
                        agent_used=agent,
                        mode=mode,
                        metadata={
                            "processing_time": time.time() - prompt_start_time,
                            "prompt_id": prompt_id,
                            "full_protocol": True
                        }
                    )
                    local_persistence.save_conversation(conversation_entry)
                    
                    # Save research output if available
                    if result and result.get("results"):
                        research_entry = ResearchEntry(
                            research_id=str(uuid.uuid4()),
                            timestamp=datetime.now().isoformat(),
                            query=query,
                            result=result_content or str(result),
                            agents_used=result.get("results", {}).get("agents_used", []),
                            sources=result.get("results", {}).get("sources", []),
                            metadata={
                                "processing_time": time.time() - prompt_start_time,
                                "prompt_id": prompt_id
                            }
                        )
                        local_persistence.save_research(research_entry)
                except Exception as e:
                    logger.error(f"Error saving conversation/research: {e}")
                
                # Always send done signal, even if there were errors
                try:
                    # Include monitoring status if available
                    monitoring_data = None
                    if bottleneck_monitor:
                        try:
                            monitoring_data = {
                                "enabled": True,
                                "status": bottleneck_monitor.get_monitoring_status()
                            }
                        except Exception as e:
                            logger.debug(f"Could not get monitoring status: {e}")
                    
                    # Include full result for astro-physiology mode so frontend can display truth-finding components
                    done_message = {
                        "type": "done",
                        "monitoring": monitoring_data
                    }
                    
                    # Add result data for astro-physiology mode
                    if mode == "astrophysiology" and result:
                        done_message["mode"] = "astrophysiology"
                        done_message["results"] = result.get("results", {})
                    
                    await safe_send_json(websocket, done_message)
                except Exception as e:
                    logger.error(f"Error sending done signal: {e}")
            
            # Continue the loop to wait for next message (don't close connection)
            # The connection stays open to handle multiple queries
            continue
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except asyncio.TimeoutError:
        logger.info("WebSocket connection timeout")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await safe_send_json(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Only close connection if we're actually exiting the loop
        # This happens when WebSocket disconnects or there's a fatal error
        # Cancel ping task if it exists - do this first to stop pings
        try:
            if 'ping_task' in locals() and ping_task:
                ping_task.cancel()
                # Wait a bit for the task to cancel
                try:
                    await asyncio.wait_for(ping_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        except Exception as e:
            logger.debug(f"Error cancelling ping task: {e}")
        
        # Remove from active connections BEFORE closing
        # This prevents ping task from trying to use the connection
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.debug("Removed websocket from active_connections")
        
        # Close connection if still open - this is critical to prevent CLOSE_WAIT state
        # CLOSE_WAIT happens when client closes but server doesn't send FIN back
        try:
            # Always try to close, even if state is not CONNECTED
            # This ensures we send FIN to complete the TCP close handshake
            # Check multiple states to catch all cases
            current_state = websocket.client_state
            if current_state in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
                logger.debug(f"Closing websocket (state: {current_state})")
                try:
                    # Use timeout to prevent hanging on close
                    await asyncio.wait_for(websocket.close(code=1000), timeout=1.0)
                    logger.debug("WebSocket closed successfully")
                except asyncio.TimeoutError:
                    logger.warning("WebSocket close timed out, forcing close")
                    # Force close by setting state directly if possible
                    try:
                        # Try to close with no wait
                        await websocket.close(code=1000)
                    except:
                        pass
                except WebSocketDisconnect:
                    # Already disconnected, that's fine
                    logger.debug("WebSocket already disconnected")
            elif current_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket already disconnected")
            else:
                logger.debug(f"WebSocket in state {current_state}, attempting close anyway")
                # Try to close even if state is unclear - this helps with CLOSE_WAIT
                try:
                    await asyncio.wait_for(websocket.close(code=1000), timeout=0.5)
                except (asyncio.TimeoutError, WebSocketDisconnect, Exception):
                    # Ignore errors - connection is likely already closed
                    pass
        except WebSocketDisconnect:
            # Already disconnected, that's fine
            logger.debug("WebSocket already disconnected during close attempt")
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")
            # Try one more time with a simpler close
            try:
                await websocket.close(code=1000)
            except:
                pass
        # Clean up WebSocket manager connection
        if websocket in connection_metadata:
            ws_manager = connection_metadata[websocket].get("websocket_manager")
            if ws_manager:
                await ws_manager.remove_connection(websocket)
        
        logger.info(f"WebSocket connection closed and removed from active connections")


@app.get("/api/status")
async def get_status():
    """Get system status with WebSocket health metrics"""
    # Clean up stale connections before reporting status
    await cleanup_stale_connections()
    
    # Get WebSocket manager health metrics
    from ..api.websocket_manager import get_websocket_manager
    ws_manager = get_websocket_manager()
    
    # Collect connection health for all active connections
    connection_health = []
    for websocket in active_connections:
        if websocket in connection_metadata:
            health = await ws_manager.get_connection_health(websocket)
            connection_health.append(health)
    
    # Get overall WebSocket statistics
    all_connections = await ws_manager.get_all_connections()
    total_connections = len(all_connections)
    
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "system_status": system_integrator.get_system_status(),
        "active_connections": len(active_connections),
        "websocket_manager": {
            "total_tracked_connections": total_connections,
            "connection_health": connection_health
        },
        "connection_details": [
            {
                "host": meta.get("client_host", "unknown"),
                "port": meta.get("client_port", "unknown"),
                "connected_at": meta.get("connected_at", "unknown")
            }
            for meta in connection_metadata.values()
        ]
    }


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(periodic_cleanup())
    logger.info("Started periodic connection cleanup task")
    
    # Initialize always-on AI components if enabled
    if portal:
        try:
            await portal.initialize()
            logger.info("ICEBURG Portal initialized at startup (always-on AI enabled)")
        except Exception as e:
            logger.error(f"Error initializing ICEBURG Portal at startup: {e}", exc_info=True)


def create_app() -> FastAPI:
    """Create and configure FastAPI app"""
    # Mount static files for frontend (must be after all routes)
    # IMPORTANT: This must be the LAST thing before returning app
    try:
        from pathlib import Path
        frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
        if frontend_dir.exists() and (frontend_dir / "index.html").exists():
            # Mount static files - this will serve index.html for / and other static files
            # API routes take precedence because they're defined first
            app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
            logger.info(f"✅ Mounted static files from {frontend_dir}")
        else:
            logger.warning(f"⚠️ Frontend directory not found or index.html missing: {frontend_dir}")
    except Exception as e:
        logger.error(f"❌ Could not mount static files: {e}", exc_info=True)
    
    return app


# Mount static files for frontend (must be after all routes)
# IMPORTANT: This must be the LAST thing before app is used
try:
    from pathlib import Path
    frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
    if frontend_dir.exists() and (frontend_dir / "index.html").exists():
        # Mount static files - this will serve index.html for / and other static files
        # API routes take precedence because they're defined first
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
        logger.info(f"✅ Mounted static files from {frontend_dir}")
    else:
        logger.warning(f"⚠️ Frontend directory not found or index.html missing: {frontend_dir}")
except Exception as e:
    logger.error(f"❌ Could not mount static files: {e}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

