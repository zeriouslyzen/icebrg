"""
Matrix Crawler API Routes - Admin endpoints for managing data crawlers.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matrix", tags=["matrix"])

# Global crawler engine instance
_crawler_engine = None
_scheduler = None


def get_crawler_engine():
    """Get or create the crawler engine."""
    global _crawler_engine
    if _crawler_engine is None:
        from ..matrix import CrawlerEngine
        _crawler_engine = CrawlerEngine()
    return _crawler_engine


def get_scheduler():
    """Get or create the scheduler."""
    global _scheduler
    if _scheduler is None:
        from ..matrix import CrawlerScheduler
        _scheduler = CrawlerScheduler(get_crawler_engine())
    return _scheduler


# Request/Response Models

class CrawlRequest(BaseModel):
    """Request to start a crawl."""
    options: Optional[Dict[str, Any]] = None


class ScheduleRequest(BaseModel):
    """Request to schedule a crawl."""
    source: str
    schedule_type: str  # interval, daily, weekly, monthly
    interval_minutes: int = 60
    run_at_hour: int = 0
    run_at_minute: int = 0
    day_of_week: int = 0
    day_of_month: int = 1
    options: Optional[Dict[str, Any]] = None


class EntitySearchRequest(BaseModel):
    """Request to search entities."""
    query: str
    entity_type: Optional[str] = None
    limit: int = 50


# Status Endpoints

@router.get("/status")
async def get_status():
    """Get Matrix Crawler status."""
    engine = get_crawler_engine()
    crawler_stats = engine.get_stats()
    
    # Get actual database stats
    try:
        from ..matrix.batch_importer import BatchImporter
        importer = BatchImporter()
        db_stats = importer.get_stats()
        total_entities = db_stats.get("total_entities", 0)
        total_relationships = db_stats.get("total_relationships", 0)
        by_type = db_stats.get("by_type", {})
    except Exception:
        total_entities = 0
        total_relationships = 0
        by_type = {}
    
    return {
        "status": "operational",
        "sources_active": crawler_stats.sources_active,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
        "total_documents": crawler_stats.total_documents,
        "by_type": by_type,
        "last_run": crawler_stats.last_run.isoformat() if crawler_stats.last_run else None,
        "available_scrapers": list(engine.scrapers.keys()),
    }


@router.get("/sources")
async def list_sources():
    """List available data sources."""
    engine = get_crawler_engine()
    return {"sources": engine.get_available_sources()}


@router.get("/stats")
async def get_stats():
    """Get crawler statistics."""
    engine = get_crawler_engine()
    stats = engine.get_stats()
    
    # Get SQLite database stats
    try:
        from ..matrix.batch_importer import BatchImporter
        importer = BatchImporter()
        db_stats = importer.get_stats()
    except Exception:
        db_stats = {}
    
    return {
        "crawler": {
            "sources_active": stats.sources_active,
            "total_documents": stats.total_documents,
            "last_run": stats.last_run.isoformat() if stats.last_run else None,
        },
        "database": db_stats,
    }


# Crawl Endpoints

@router.post("/crawl/{source}")
async def start_crawl(source: str, request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start a crawl for a specific source."""
    engine = get_crawler_engine()
    
    if source not in engine.scrapers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown source: {source}. Available: {list(engine.scrapers.keys())}"
        )
    
    try:
        # Start crawl in background
        import asyncio
        job = await engine.start_crawl(source, request.options)
        
        return {
            "job_id": job.job_id,
            "source": job.source,
            "status": job.status.value,
            "started_at": job.started_at.isoformat() if job.started_at else None,
        }
    except Exception as e:
        logger.error(f"Failed to start crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawl/all")
async def start_all_crawls(request: CrawlRequest):
    """Start crawls for all available sources."""
    engine = get_crawler_engine()
    
    try:
        jobs = await engine.crawl_all(request.options)
        
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "source": job.source,
                    "status": job.status.value,
                }
                for job in jobs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to start crawls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/crawl/{job_id}")
async def stop_crawl(job_id: str):
    """Stop a running crawl job."""
    engine = get_crawler_engine()
    
    success = await engine.stop_crawl(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already stopped")
    
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/jobs")
async def list_jobs(limit: int = 50):
    """List all crawl jobs."""
    engine = get_crawler_engine()
    jobs = engine.get_all_jobs(limit=limit)
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "source": job.source,
                "status": job.status.value,
                "progress": job.progress,
                "items_processed": job.items_processed,
                "entities_extracted": job.entities_extracted,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "errors": job.errors,
            }
            for job in jobs
        ]
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a specific crawl job."""
    engine = get_crawler_engine()
    job = engine.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "source": job.source,
        "status": job.status.value,
        "progress": job.progress,
        "items_processed": job.items_processed,
        "items_total": job.items_total,
        "entities_extracted": job.entities_extracted,
        "relationships_extracted": job.relationships_extracted,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "errors": job.errors,
        "metadata": job.metadata,
    }


# Schedule Endpoints

@router.get("/schedules")
async def list_schedules():
    """List all scheduled crawls."""
    scheduler = get_scheduler()
    schedules = scheduler.get_schedules()
    
    return {
        "schedules": [s.to_dict() for s in schedules]
    }


@router.post("/schedules")
async def create_schedule(request: ScheduleRequest):
    """Create a new scheduled crawl."""
    scheduler = get_scheduler()
    
    schedule = scheduler.add_schedule(
        source=request.source,
        schedule_type=request.schedule_type,
        interval_minutes=request.interval_minutes,
        run_at_hour=request.run_at_hour,
        run_at_minute=request.run_at_minute,
        day_of_week=request.day_of_week,
        day_of_month=request.day_of_month,
        options=request.options,
    )
    
    return schedule.to_dict()


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a scheduled crawl."""
    scheduler = get_scheduler()
    
    success = scheduler.remove_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    return {"schedule_id": schedule_id, "deleted": True}


@router.post("/schedules/{schedule_id}/enable")
async def enable_schedule(schedule_id: str, enabled: bool = True):
    """Enable or disable a scheduled crawl."""
    scheduler = get_scheduler()
    
    success = scheduler.enable_schedule(schedule_id, enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    return {"schedule_id": schedule_id, "enabled": enabled}


# Entity Endpoints

@router.get("/entities")
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 50
):
    """List entities in the matrix graph."""
    try:
        from ..matrix import MatrixGraph
        graph = MatrixGraph()
        
        entities = []
        for entity in list(graph.entities.values())[:limit]:
            if entity_type and entity.entity_type != entity_type:
                continue
            entities.append(entity.to_dict())
        
        return {"entities": entities, "total": len(graph.entities)}
    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/search")
async def search_entities(request: EntitySearchRequest):
    """Search for entities by name."""
    try:
        from ..matrix.batch_importer import BatchImporter
        
        importer = BatchImporter()
        results = importer.search(
            query=request.query,
            entity_type=request.entity_type,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get a specific entity."""
    try:
        from ..matrix import MatrixGraph
        graph = MatrixGraph()
        
        entity = graph.get_entity(entity_id)
        if entity is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return entity.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}/network")
async def get_entity_network(entity_id: str, depth: int = 2, limit: int = 100):
    """Get the network around an entity."""
    try:
        from ..matrix import MatrixGraph
        graph = MatrixGraph()
        
        network = graph.get_network(entity_id, depth=depth, limit=limit)
        return network
    except Exception as e:
        logger.error(f"Failed to get network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}/relationships")
async def get_entity_relationships(
    entity_id: str,
    relationship_type: Optional[str] = None,
    direction: str = "both"
):
    """Get relationships for an entity."""
    try:
        from ..matrix import MatrixGraph
        graph = MatrixGraph()
        
        relationships = graph.get_relationships(
            entity_id,
            relationship_type=relationship_type,
            direction=direction
        )
        return {"relationships": relationships}
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Path Finding

@router.get("/path")
async def find_path(source_id: str, target_id: str, max_length: int = 6):
    """Find the shortest path between two entities."""
    try:
        from ..matrix import MatrixGraph
        graph = MatrixGraph()
        
        path = graph.find_path(source_id, target_id, max_length=max_length)
        
        if path is None:
            return {
                "source_id": source_id,
                "target_id": target_id,
                "path": None,
                "message": "No path found"
            }
        
        # Get entity names for the path
        path_with_names = []
        for entity_id in path:
            entity = graph.get_entity(entity_id)
            path_with_names.append({
                "id": entity_id,
                "name": entity.name if entity else entity_id,
                "type": entity.entity_type if entity else "unknown",
            })
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "path": path_with_names,
            "length": len(path) - 1,
        }
    except Exception as e:
        logger.error(f"Path finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
