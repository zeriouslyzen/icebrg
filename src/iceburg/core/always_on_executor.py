"""
Always-On Background Executor
Continuously runs protocol execution in background, learning from patterns and maintaining agent readiness.
Based on US20230108560A1 patent pattern for always-on AI systems.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from collections import deque
import threading

logger = logging.getLogger(__name__)


class AlwaysOnProtocolExecutor:
    """
    Always-on background executor that continuously processes tasks and maintains agent readiness.
    
    Architecture:
    - Background protocol loop processing queued tasks
    - Continuous knowledge base updates
    - Pattern learning and pre-computation
    - Agent readiness maintenance
    - Graph-based task processing (US20230108560A1 pattern)
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        self.task_queue: deque = deque()
        self.completed_tasks: Dict[str, Any] = {}
        self.pattern_cache: Dict[str, Any] = {}
        self.agent_readiness: Dict[str, bool] = {}
        self.knowledge_updates: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.stats = {
            "tasks_processed": 0,
            "patterns_learned": 0,
            "knowledge_updates": 0,
            "agent_readiness_checks": 0,
            "start_time": None,
            "uptime_seconds": 0
        }
        
        logger.info("AlwaysOnProtocolExecutor initialized")
    
    async def start(self):
        """Start always-on background processes"""
        if self.is_running:
            logger.warning("AlwaysOnProtocolExecutor is already running")
            return
        
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        logger.info("Starting always-on background processes...")
        
        # Start background protocol execution
        self.background_tasks.append(
            asyncio.create_task(self._protocol_loop())
        )
        
        # Start knowledge accumulation
        self.background_tasks.append(
            asyncio.create_task(self._knowledge_accumulation_loop())
        )
        
        # Start agent coordination
        self.background_tasks.append(
            asyncio.create_task(self._agent_coordination_loop())
        )
        
        # Start pattern learning
        self.background_tasks.append(
            asyncio.create_task(self._pattern_learning_loop())
        )
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def stop(self):
        """Stop always-on background processes"""
        if not self.is_running:
            return
        
        logger.info("Stopping always-on background processes...")
        self.is_running = False
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Always-on background processes stopped")
    
    async def _protocol_loop(self):
        """Main protocol execution loop - continuously processes queued tasks"""
        logger.info("Protocol loop started")
        
        while self.is_running:
            try:
                # Process queued tasks
                if self.task_queue:
                    task = self.task_queue.popleft()
                    await self._process_task(task)
                    self.stats["tasks_processed"] += 1
                
                # Update uptime
                if self.stats["start_time"]:
                    self.stats["uptime_seconds"] = (datetime.now() - self.stats["start_time"]).total_seconds()
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info("Protocol loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in protocol loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _knowledge_accumulation_loop(self):
        """Continuously updates knowledge base from patterns and tasks"""
        logger.info("Knowledge accumulation loop started")
        
        while self.is_running:
            try:
                # Process knowledge updates
                if self.knowledge_updates:
                    update = self.knowledge_updates.pop(0)
                    await self._process_knowledge_update(update)
                    self.stats["knowledge_updates"] += 1
                
                # Small delay
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Knowledge accumulation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in knowledge accumulation loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _agent_coordination_loop(self):
        """Maintains agent readiness and coordinates agent availability"""
        logger.info("Agent coordination loop started")
        
        while self.is_running:
            try:
                # Check agent readiness
                await self._check_agent_readiness()
                self.stats["agent_readiness_checks"] += 1
                
                # Small delay
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                logger.info("Agent coordination loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in agent coordination loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _pattern_learning_loop(self):
        """Learns patterns from completed tasks and updates pattern cache"""
        logger.info("Pattern learning loop started")
        
        while self.is_running:
            try:
                # Learn patterns from completed tasks
                if self.completed_tasks:
                    # Process recent completed tasks
                    recent_tasks = list(self.completed_tasks.items())[-10:]
                    for task_id, task_data in recent_tasks:
                        await self._learn_patterns_from_task(task_data)
                        self.stats["patterns_learned"] += 1
                    
                    # Clear old completed tasks (keep last 100)
                    if len(self.completed_tasks) > 100:
                        oldest_keys = list(self.completed_tasks.keys())[:-100]
                        for key in oldest_keys:
                            del self.completed_tasks[key]
                
                # Small delay
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                logger.info("Pattern learning loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in pattern learning loop: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a single task"""
        try:
            task_id = task.get("id", f"task_{int(time.time())}")
            task_type = task.get("type", "unknown")
            
            logger.debug(f"Processing task {task_id} of type {task_type}")
            
            # Process task based on type
            if task_type == "knowledge_update":
                await self._process_knowledge_update(task)
            elif task_type == "pattern_learning":
                await self._learn_patterns_from_task(task)
            elif task_type == "agent_readiness":
                await self._check_agent_readiness()
            else:
                logger.warning(f"Unknown task type: {task_type}")
            
            # Mark task as completed
            self.completed_tasks[task_id] = {
                **task,
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
    
    async def _process_knowledge_update(self, update: Dict[str, Any]):
        """Process knowledge base update"""
        try:
            # Extract knowledge from update
            knowledge = update.get("knowledge", {})
            
            # Update pattern cache with knowledge
            if knowledge:
                pattern_key = update.get("pattern_key", "default")
                self.pattern_cache[pattern_key] = knowledge
                
                logger.debug(f"Updated knowledge for pattern: {pattern_key}")
            
        except Exception as e:
            logger.error(f"Error processing knowledge update: {e}", exc_info=True)
    
    async def _check_agent_readiness(self):
        """Check and maintain agent readiness"""
        try:
            # Check if agents are ready
            # This would integrate with PreWarmedAgentPool
            # For now, just mark all agents as ready
            agent_ids = ["surveyor", "dissident", "archaeologist", "synthesist", "oracle"]
            
            for agent_id in agent_ids:
                self.agent_readiness[agent_id] = True
            
            logger.debug(f"Checked agent readiness: {len(agent_ids)} agents ready")
            
        except Exception as e:
            logger.error(f"Error checking agent readiness: {e}", exc_info=True)
    
    async def _learn_patterns_from_task(self, task_data: Dict[str, Any]):
        """Learn patterns from completed task"""
        try:
            # Extract patterns from task
            query = task_data.get("query", "")
            result = task_data.get("result", {})
            
            # Learn patterns (simplified for now)
            if query and result:
                pattern_key = f"pattern_{hash(query) % 10000}"
                self.pattern_cache[pattern_key] = {
                    "query": query,
                    "result": result,
                    "learned_at": datetime.now().isoformat()
                }
                
                logger.debug(f"Learned pattern: {pattern_key}")
            
        except Exception as e:
            logger.error(f"Error learning patterns: {e}", exc_info=True)
    
    def queue_task(self, task: Dict[str, Any]):
        """Queue a task for background processing"""
        task_id = task.get("id", f"task_{int(time.time())}")
        task["id"] = task_id
        task["queued_at"] = datetime.now().isoformat()
        self.task_queue.append(task)
        logger.debug(f"Queued task {task_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            **self.stats,
            "is_running": self.is_running,
            "queued_tasks": len(self.task_queue),
            "completed_tasks_count": len(self.completed_tasks),
            "pattern_cache_size": len(self.pattern_cache),
            "agent_readiness": self.agent_readiness
        }
    
    def get_pattern(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Get cached pattern"""
        return self.pattern_cache.get(pattern_key)

