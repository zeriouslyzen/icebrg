"""
API Routes
Additional API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.system_integrator import SystemIntegrator
from ..generation.device_generator import DeviceGenerator
from ..research.methodology_analyzer import MethodologyAnalyzer
from ..integration.swarming_integration import SwarmingIntegration

router = APIRouter()

system_integrator = SystemIntegrator()
device_generator = DeviceGenerator()
methodology_analyzer = MethodologyAnalyzer()
swarming_integration = SwarmingIntegration()


@router.get("/api/methodology")
async def get_methodology():
    """Get Enhanced Deliberation methodology"""
    return {
        "methodology": "enhanced_deliberation",
        "components": methodology_analyzer.get_methodology_components(),
        "description": "Truth-seeking analysis method with deep reflection pauses"
    }


@router.post("/api/swarm/create")
async def create_swarm(request: Dict[str, Any]):
    """Create truth-finding swarm"""
    try:
        query = request.get("query", "")
        swarm_type = request.get("swarm_type", "research_swarm")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        swarm = await swarming_integration.create_truth_finding_swarm(
            query=query,
            swarm_type=swarm_type
        )
        
        return swarm
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/swarm/execute")
async def execute_swarm(request: Dict[str, Any]):
    """Execute swarm"""
    try:
        swarm = request.get("swarm", {})
        parallel = request.get("parallel", True)
        
        if not swarm:
            raise HTTPException(status_code=400, detail="Swarm is required")
        
        result = await swarming_integration.execute_swarm(
            swarm=swarm,
            parallel=parallel
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/capabilities")
async def get_capabilities():
    """Get ICEBURG capabilities"""
    return {
        "capabilities": {
            "truth_finding": {
                "suppression_detection": True,
                "information_archaeology": True,
                "enhanced_deliberation": True
            },
            "device_generation": {
                "general_purpose": True,
                "schematics": True,
                "code_generation": True,
                "bom_generation": True
            },
            "research": {
                "autonomous_learning": True,
                "curiosity_driven": True,
                "swarming": True
            },
            "lab": {
                "quantum_simulation": True,
                "molecular_dynamics": True,
                "particle_physics": True,
                "cfd": True,
                "hpc_integration": True
            },
            "security": {
                "penetration_testing": True,
                "vulnerability_scanning": True,
                "autonomous_red_team": True,
                "ethical_hacking": True
            },
            "visual": {
                "reasoning": True,
                "chart_generation": True,
                "image_analysis": True
            }
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/api/export/generate")
async def generate_export(request: Dict[str, Any]):
    """Generate export in various formats (PDF, chart, code, visual)"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        format_type = request.get("format", "pdf")
        content = request.get("content", "")
        message_id = request.get("message_id", "")
        
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")
        
        # Generate based on format
        if format_type == "pdf":
            try:
                from ..visual.pdf_generator import PDFGenerator
                generator = PDFGenerator()
                result = generator.generate_pdf(content, title="ICEBURG Export")
                return {
                    "download_url": result.get("file_path"),
                    "filename": result.get("filename", "iceburg_export.pdf"),
                    "format": "pdf"
                }
            except ImportError:
                # Fallback: return content as downloadable text
                return {
                    "content": content,
                    "filename": "iceburg_export.txt",
                    "format": "pdf"
                }
        elif format_type == "chart":
            try:
                from ..visual.chart_generator import ChartGenerator
                generator = ChartGenerator()
                # Extract data from content for chart
                result = generator.generate_chart("bar", {"data": content}, format="png")
                return {
                    "download_url": result.get("output_path"),
                    "filename": result.get("filename", "iceburg_chart.png"),
                    "format": "chart"
                }
            except ImportError:
                return {
                    "content": f"Chart data: {content[:200]}",
                    "format": "chart"
                }
        elif format_type == "code":
            # Generate code from content
            return {
                "content": f"# Generated code from ICEBURG\n# {content[:100]}...\n\n# Code implementation here",
                "format": "code"
            }
        elif format_type == "visual":
            try:
                from ..agents.visual_architect import VisualArchitect
                from ..config import load_config
                visual_architect = VisualArchitect()
                cfg = load_config()
                result = visual_architect.run(cfg, content, verbose=False)
                return {
                    "content": result.spec.to_dict().get("html", ""),
                    "format": "visual"
                }
            except ImportError:
                return {
                    "content": f"<div>Visual representation: {content[:200]}</div>",
                    "format": "visual"
                }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}")
            
    except Exception as e:
        logger.error(f"Error generating export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    return system_integrator.get_system_status()


@router.get("/api/encyclopedia/news")
async def get_encyclopedia_news():
    """Get relevant news and updates for encyclopedia"""
    try:
        from .news_scraper import get_news_with_summaries
        articles = await get_news_with_summaries(max_results=15)
        return {
            "articles": articles,
            "count": len(articles),
            "fetched_at": datetime.now().isoformat()
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error fetching encyclopedia news: {e}")
        return {
            "articles": [],
            "count": 0,
            "error": str(e),
            "fetched_at": datetime.now().isoformat()
        }


@router.get("/api/test/ollama")
async def test_ollama():
    """
    Lightweight health check for local Ollama provider.

    - Does NOT change global provider configuration.
    - Simply attempts a tiny completion using OllamaProvider and reports status.
    """
    import logging
    from ..providers.ollama_provider import OllamaProvider

    logger = logging.getLogger(__name__)

    try:
        import asyncio
        provider = OllamaProvider()
        # Minimal prompt to avoid wasting tokens / time
        prompt = "Return a 1-sentence confirmation that Ollama is reachable."
        # Use a model that's likely to be available; try llama3:8b first, fallback to llama3.1:8b
        model = "llama3:8b"
        try:
            content = await asyncio.to_thread(
                provider.chat_complete,
                model=model,
                prompt=prompt,
                system="You are a diagnostic agent. Respond with a short confirmation only.",
                temperature=0.0,
                options={"max_tokens": 32},
            )
        except Exception as model_error:
            # Try fallback model
            logger.warning(f"Model {model} not available, trying llama3.1:8b: {model_error}")
            model = "llama3.1:8b"
            content = await asyncio.to_thread(
                provider.chat_complete,
                model=model,
                prompt=prompt,
                system="You are a diagnostic agent. Respond with a short confirmation only.",
                temperature=0.0,
                options={"max_tokens": 32},
            )

        return {
            "status": "ok",
            "provider": "ollama",
            "model": model,
            "message": content.strip() if isinstance(content, str) else str(content),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Ollama health check failed: {e}",
        )

