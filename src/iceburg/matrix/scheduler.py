"""
Scheduler - Background job scheduler for autonomous crawling.
Supports cron-style scheduling with persistence across restarts.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """A scheduled crawl job."""
    schedule_id: str
    source: str
    schedule_type: str  # "interval", "daily", "weekly", "monthly"
    interval_minutes: int = 0  # For interval type
    run_at_hour: int = 0  # For daily/weekly/monthly
    run_at_minute: int = 0
    day_of_week: int = 0  # 0=Monday, for weekly
    day_of_month: int = 1  # For monthly
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "source": self.source,
            "schedule_type": self.schedule_type,
            "interval_minutes": self.interval_minutes,
            "run_at_hour": self.run_at_hour,
            "run_at_minute": self.run_at_minute,
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "options": self.options,
        }


class CrawlerScheduler:
    """
    Background scheduler for autonomous crawling.
    
    Features:
    - Cron-style scheduling (interval, daily, weekly, monthly)
    - Persistence across restarts
    - Enable/disable individual schedules
    """
    
    def __init__(
        self,
        crawler_engine: Any,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            crawler_engine: CrawlerEngine instance to use
            data_dir: Directory for persisting schedules
        """
        self.engine = crawler_engine
        self.data_dir = data_dir or Path.home() / "Documents" / "iceburg_matrix"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.schedules_path = self.data_dir / "schedules.json"
        self.schedules: Dict[str, ScheduledJob] = {}
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._schedule_counter = 0
        
        self._load_schedules()
        logger.info(f"⏰ Scheduler initialized ({len(self.schedules)} schedules)")
    
    def _load_schedules(self):
        """Load schedules from disk."""
        try:
            if self.schedules_path.exists():
                with open(self.schedules_path, "r") as f:
                    data = json.load(f)
                    for sid, sdata in data.items():
                        self.schedules[sid] = ScheduledJob(
                            schedule_id=sdata["schedule_id"],
                            source=sdata["source"],
                            schedule_type=sdata["schedule_type"],
                            interval_minutes=sdata.get("interval_minutes", 0),
                            run_at_hour=sdata.get("run_at_hour", 0),
                            run_at_minute=sdata.get("run_at_minute", 0),
                            day_of_week=sdata.get("day_of_week", 0),
                            day_of_month=sdata.get("day_of_month", 1),
                            enabled=sdata.get("enabled", True),
                            last_run=datetime.fromisoformat(sdata["last_run"]) if sdata.get("last_run") else None,
                            next_run=datetime.fromisoformat(sdata["next_run"]) if sdata.get("next_run") else None,
                            options=sdata.get("options", {}),
                        )
        except Exception as e:
            logger.warning(f"Could not load schedules: {e}")
    
    def _save_schedules(self):
        """Persist schedules to disk."""
        try:
            with open(self.schedules_path, "w") as f:
                json.dump({sid: s.to_dict() for sid, s in self.schedules.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save schedules: {e}")
    
    def _calculate_next_run(self, schedule: ScheduledJob) -> datetime:
        """Calculate the next run time for a schedule."""
        now = datetime.now()
        
        if schedule.schedule_type == "interval":
            if schedule.last_run:
                return schedule.last_run + timedelta(minutes=schedule.interval_minutes)
            return now + timedelta(minutes=schedule.interval_minutes)
        
        elif schedule.schedule_type == "daily":
            next_run = now.replace(
                hour=schedule.run_at_hour,
                minute=schedule.run_at_minute,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif schedule.schedule_type == "weekly":
            next_run = now.replace(
                hour=schedule.run_at_hour,
                minute=schedule.run_at_minute,
                second=0,
                microsecond=0
            )
            days_ahead = schedule.day_of_week - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and next_run <= now):
                days_ahead += 7
            next_run += timedelta(days=days_ahead)
            return next_run
        
        elif schedule.schedule_type == "monthly":
            next_run = now.replace(
                day=min(schedule.day_of_month, 28),  # Safe for all months
                hour=schedule.run_at_hour,
                minute=schedule.run_at_minute,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            return next_run
        
        return now + timedelta(days=1)
    
    def add_schedule(
        self,
        source: str,
        schedule_type: str,
        interval_minutes: int = 60,
        run_at_hour: int = 0,
        run_at_minute: int = 0,
        day_of_week: int = 0,
        day_of_month: int = 1,
        options: Optional[Dict[str, Any]] = None
    ) -> ScheduledJob:
        """
        Add a new schedule.
        
        Args:
            source: Data source to crawl
            schedule_type: "interval", "daily", "weekly", "monthly"
            interval_minutes: For interval type
            run_at_hour: Hour to run (0-23)
            run_at_minute: Minute to run (0-59)
            day_of_week: Day of week (0=Monday) for weekly
            day_of_month: Day of month (1-28) for monthly
            options: Crawler options
            
        Returns:
            Created schedule
        """
        self._schedule_counter += 1
        schedule_id = f"schedule_{source}_{self._schedule_counter}"
        
        schedule = ScheduledJob(
            schedule_id=schedule_id,
            source=source,
            schedule_type=schedule_type,
            interval_minutes=interval_minutes,
            run_at_hour=run_at_hour,
            run_at_minute=run_at_minute,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            options=options or {},
        )
        schedule.next_run = self._calculate_next_run(schedule)
        
        self.schedules[schedule_id] = schedule
        self._save_schedules()
        
        logger.info(f"📅 Added schedule: {schedule_id} ({schedule_type}, next: {schedule.next_run})")
        return schedule
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save_schedules()
            logger.info(f"🗑️ Removed schedule: {schedule_id}")
            return True
        return False
    
    def enable_schedule(self, schedule_id: str, enabled: bool = True) -> bool:
        """Enable or disable a schedule."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = enabled
            if enabled:
                self.schedules[schedule_id].next_run = self._calculate_next_run(self.schedules[schedule_id])
            self._save_schedules()
            return True
        return False
    
    def get_schedules(self) -> List[ScheduledJob]:
        """Get all schedules."""
        return list(self.schedules.values())
    
    async def start(self):
        """Start the scheduler background loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("▶️ Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Scheduler stopped")
    
    async def _run_loop(self):
        """Background loop that checks and runs scheduled jobs."""
        while self._running:
            try:
                now = datetime.now()
                
                for schedule in self.schedules.values():
                    if not schedule.enabled:
                        continue
                    
                    if schedule.next_run and schedule.next_run <= now:
                        # Time to run!
                        logger.info(f"⏰ Running scheduled job: {schedule.schedule_id}")
                        
                        try:
                            await self.engine.start_crawl(schedule.source, schedule.options)
                            schedule.last_run = now
                            schedule.next_run = self._calculate_next_run(schedule)
                            self._save_schedules()
                        except Exception as e:
                            logger.error(f"Scheduled job failed: {schedule.schedule_id} - {e}")
                
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
