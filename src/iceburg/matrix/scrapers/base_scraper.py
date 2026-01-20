"""
Base Scraper - Abstract base class for all data source scrapers.
Provides common functionality for rate limiting, retries, and progress tracking.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ScraperResult:
    """Result of a scraper run."""
    success: bool
    documents_count: int = 0
    entities_count: int = 0
    relationships_count: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScraper(ABC):
    """
    Abstract base class for Matrix scrapers.
    
    Provides:
    - Rate limiting
    - Retry logic with exponential backoff
    - Progress reporting
    - Error handling
    - Data directory management
    """
    
    # Default rate limit (requests per second)
    RATE_LIMIT = 1.0
    
    # Default retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # User agent for requests
    USER_AGENT = "ICEBURG Matrix Crawler (research purposes)"
    
    def __init__(
        self,
        data_dir: Path,
        on_progress: Optional[Callable[[float, int, int], None]] = None
    ):
        """
        Initialize the scraper.
        
        Args:
            data_dir: Directory for storing scraped data
            on_progress: Callback for progress updates (progress, items, total)
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.on_progress = on_progress
        self._last_request_time = 0.0
        self._cancelled = False
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source name."""
        pass
    
    @property
    @abstractmethod
    def source_url(self) -> str:
        """Return the source base URL."""
        pass
    
    @abstractmethod
    async def _scrape(self, options: Dict[str, Any]) -> ScraperResult:
        """
        Implement the actual scraping logic.
        
        Args:
            options: Scraper-specific options
            
        Returns:
            ScraperResult with results
        """
        pass
    
    async def run(self, options: Optional[Dict[str, Any]] = None) -> ScraperResult:
        """
        Run the scraper.
        
        Args:
            options: Scraper-specific options
            
        Returns:
            ScraperResult with results
        """
        options = options or {}
        result = ScraperResult(success=False)
        result.started_at = datetime.now()
        
        logger.info(f"🕷️ Starting {self.source_name} scraper")
        
        try:
            # Create HTTP client
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": self.USER_AGENT},
                follow_redirects=True,
            )
            
            # Run scraping
            result = await self._scrape(options)
            result.success = True
            
        except asyncio.CancelledError:
            result.success = False
            result.errors.append("Cancelled")
            raise
            
        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Scraper error: {e}", exc_info=True)
            
        finally:
            if self.client:
                await self.client.aclose()
            result.completed_at = datetime.now()
        
        logger.info(f"✅ {self.source_name} scraper completed: {result.documents_count} documents")
        return result
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 1.0 / self.RATE_LIMIT
        
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    async def _fetch(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> httpx.Response:
        """
        Fetch a URL with rate limiting and retries.
        
        Args:
            url: URL to fetch
            method: HTTP method
            **kwargs: Additional arguments for httpx
            
        Returns:
            Response object
        """
        await self._rate_limit()
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    last_error = e
                else:
                    raise
                    
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                wait_time = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                last_error = e
        
        raise last_error or Exception("Max retries exceeded")
    
    async def _download_file(
        self,
        url: str,
        filename: str,
        chunk_size: int = 8192
    ) -> Path:
        """
        Download a file with progress tracking.
        
        Args:
            url: URL to download
            filename: Local filename
            chunk_size: Download chunk size
            
        Returns:
            Path to downloaded file
        """
        output_path = self.data_dir / filename
        
        await self._rate_limit()
        
        async with self.client.stream("GET", url) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    if self._cancelled:
                        raise asyncio.CancelledError()
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if self.on_progress and total_size:
                        self.on_progress(downloaded / total_size, downloaded, total_size)
        
        logger.info(f"📥 Downloaded: {filename} ({downloaded} bytes)")
        return output_path
    
    def _report_progress(self, progress: float, items: int, total: int):
        """Report progress to callback."""
        if self.on_progress:
            self.on_progress(progress, items, total)
    
    def cancel(self):
        """Request scraper cancellation."""
        self._cancelled = True
