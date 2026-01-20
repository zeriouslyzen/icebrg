"""
Crawler Engine - Main orchestrator for Matrix data gathering.
Manages scraper lifecycle, tracks progress, and coordinates extraction.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class CrawlerStatus(Enum):
    """Status of a crawler job."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CrawlerJob:
    """Represents a single crawler job."""
    job_id: str
    source: str
    status: CrawlerStatus = CrawlerStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    items_processed: int = 0
    items_total: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlerStats:
    """Overall crawler statistics."""
    total_entities: int = 0
    total_relationships: int = 0
    total_documents: int = 0
    sources_active: int = 0
    last_run: Optional[datetime] = None
    next_scheduled: Optional[datetime] = None


class CrawlerEngine:
    """
    Main orchestrator for Matrix data gathering.
    
    Manages:
    - Scraper lifecycle (start, stop, pause)
    - Rate limiting and politeness
    - Progress tracking
    - Event emission for UI updates
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the crawler engine.
        
        Args:
            data_dir: Directory for storing crawled data
        """
        self.data_dir = data_dir or Path.home() / "Documents" / "iceburg_matrix"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Job management
        self.jobs: Dict[str, CrawlerJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self._job_counter = 0
        self._lock = threading.Lock()
        
        # Scraper registry
        self.scrapers: Dict[str, Any] = {}
        self._register_scrapers()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Event callbacks
        self.on_progress: Optional[Callable[[CrawlerJob], None]] = None
        self.on_complete: Optional[Callable[[CrawlerJob], None]] = None
        self.on_error: Optional[Callable[[CrawlerJob, str], None]] = None
        
        # Stats
        self.stats = CrawlerStats()
        
        logger.info(f"🕸️ Matrix Crawler Engine initialized (data_dir: {self.data_dir})")
    
    def _register_scrapers(self):
        """Register available scrapers."""
        # Lazy import to avoid circular dependencies
        try:
            from .scrapers.fec_scraper import FECScraper
            self.scrapers["fec"] = FECScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.sec_edgar_scraper import SECEdgarScraper
            self.scrapers["sec_edgar"] = SECEdgarScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.icij_scraper import ICIJScraper
            self.scrapers["icij"] = ICIJScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.opensanctions_scraper import OpenSanctionsScraper
            self.scrapers["opensanctions"] = OpenSanctionsScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.congress_scraper import CongressScraper
            self.scrapers["congress"] = CongressScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.littlesis_scraper import LittleSisScraper
            self.scrapers["littlesis"] = LittleSisScraper
        except ImportError:
            pass
        
        try:
            from .scrapers.wikidata_bulk import WikidataBulkScraper
            self.scrapers["wikidata"] = WikidataBulkScraper
        except ImportError:
            pass
        
        logger.info(f"📋 Registered scrapers: {list(self.scrapers.keys())}")
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get list of available data sources."""
        sources = [
            {
                "id": "fec",
                "name": "FEC Campaign Finance",
                "description": "Federal Election Commission campaign contributions and expenditures",
                "url": "https://www.fec.gov/data/",
                "update_frequency": "daily",
                "available": "fec" in self.scrapers
            },
            {
                "id": "sec_edgar",
                "name": "SEC EDGAR Filings",
                "description": "Corporate filings, insider trading, beneficial ownership",
                "url": "https://www.sec.gov/edgar/",
                "update_frequency": "real-time",
                "available": "sec_edgar" in self.scrapers
            },
            {
                "id": "icij",
                "name": "ICIJ Offshore Leaks",
                "description": "Panama Papers, Paradise Papers, Pandora Papers",
                "url": "https://offshoreleaks.icij.org/",
                "update_frequency": "static",
                "available": "icij" in self.scrapers
            },
            {
                "id": "opensanctions",
                "name": "OpenSanctions",
                "description": "Sanctioned individuals, PEPs, wanted persons",
                "url": "https://opensanctions.org/",
                "update_frequency": "weekly",
                "available": "opensanctions" in self.scrapers
            },
            {
                "id": "congress",
                "name": "Congress.gov",
                "description": "Lobbying disclosures, bills, committee memberships",
                "url": "https://www.congress.gov/",
                "update_frequency": "weekly",
                "available": "congress" in self.scrapers
            },
            {
                "id": "littlesis",
                "name": "LittleSis Power Network",
                "description": "300K+ power players with relationships and donations",
                "url": "https://littlesis.org/",
                "update_frequency": "monthly",
                "available": "littlesis" in self.scrapers
            },
            {
                "id": "wikidata",
                "name": "Wikidata",
                "description": "Structured knowledge from Wikipedia",
                "url": "https://www.wikidata.org/",
                "update_frequency": "continuous",
                "available": "wikidata" in self.scrapers
            },
        ]
        return sources
    
    def _generate_job_id(self, source: str) -> str:
        """Generate a unique job ID."""
        with self._lock:
            self._job_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{source}_{timestamp}_{self._job_counter}"
    
    async def start_crawl(
        self,
        source: str,
        options: Optional[Dict[str, Any]] = None
    ) -> CrawlerJob:
        """
        Start a crawl job for a specific source.
        
        Args:
            source: Source ID (e.g., 'fec', 'sec_edgar')
            options: Source-specific options
            
        Returns:
            CrawlerJob with job details
        """
        if source not in self.scrapers:
            raise ValueError(f"Unknown source: {source}. Available: {list(self.scrapers.keys())}")
        
        # Check if already running
        for job_id, job in self.jobs.items():
            if job.source == source and job.status == CrawlerStatus.RUNNING:
                logger.warning(f"Crawler for {source} already running: {job_id}")
                return job
        
        # Create job
        job_id = self._generate_job_id(source)
        job = CrawlerJob(
            job_id=job_id,
            source=source,
            status=CrawlerStatus.RUNNING,
            started_at=datetime.now(),
            metadata=options or {}
        )
        self.jobs[job_id] = job
        self.stats.sources_active += 1
        
        logger.info(f"🚀 Starting crawl job: {job_id}")
        
        # Start async task
        task = asyncio.create_task(self._run_crawl(job))
        self.active_jobs[job_id] = task
        
        return job
    
    async def _run_crawl(self, job: CrawlerJob):
        """Execute a crawl job."""
        try:
            # Instantiate scraper
            scraper_class = self.scrapers[job.source]
            scraper = scraper_class(
                data_dir=self.data_dir / job.source,
                on_progress=lambda p, i, t: self._update_progress(job, p, i, t)
            )
            
            # Run scraper
            result = await scraper.run(job.metadata)
            
            # Update job
            job.status = CrawlerStatus.COMPLETED
            job.completed_at = datetime.now()
            job.entities_extracted = result.entities_count
            job.relationships_extracted = result.relationships_count
            job.items_processed = result.documents_count
            job.progress = 1.0
            
            # Update stats
            self.stats.total_entities += result.entities_count
            self.stats.total_relationships += result.relationships_count
            self.stats.total_documents += result.documents_count
            self.stats.last_run = datetime.now()
            
            logger.info(f"✅ Crawl completed: {job.job_id} ({job.entities_extracted} entities)")
            
            if self.on_complete:
                self.on_complete(job)
                
        except asyncio.CancelledError:
            job.status = CrawlerStatus.CANCELLED
            logger.info(f"🛑 Crawl cancelled: {job.job_id}")
            
        except Exception as e:
            job.status = CrawlerStatus.FAILED
            job.errors.append(str(e))
            logger.error(f"❌ Crawl failed: {job.job_id} - {e}", exc_info=True)
            
            if self.on_error:
                self.on_error(job, str(e))
        
        finally:
            self.stats.sources_active = max(0, self.stats.sources_active - 1)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _update_progress(self, job: CrawlerJob, progress: float, items: int, total: int):
        """Update job progress."""
        job.progress = progress
        job.items_processed = items
        job.items_total = total
        
        if self.on_progress:
            self.on_progress(job)
    
    async def stop_crawl(self, job_id: str) -> bool:
        """Stop a running crawl job."""
        if job_id not in self.active_jobs:
            return False
        
        task = self.active_jobs[job_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        return True
    
    def get_job(self, job_id: str) -> Optional[CrawlerJob]:
        """Get a specific job."""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self, limit: int = 50) -> List[CrawlerJob]:
        """Get all jobs, most recent first."""
        jobs = sorted(
            self.jobs.values(),
            key=lambda j: j.started_at or datetime.min,
            reverse=True
        )
        return jobs[:limit]
    
    def get_stats(self) -> CrawlerStats:
        """Get crawler statistics."""
        return self.stats
    
    async def crawl_all(self, options: Optional[Dict[str, Any]] = None) -> List[CrawlerJob]:
        """Start crawls for all available sources."""
        jobs = []
        for source in self.scrapers.keys():
            try:
                job = await self.start_crawl(source, options)
                jobs.append(job)
            except Exception as e:
                logger.error(f"Failed to start {source}: {e}")
        return jobs
