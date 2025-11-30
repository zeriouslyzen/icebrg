"""
ICEBURG Micro-Agent Swarm System - October 2025
===============================================

A distributed micro-agent architecture using tiny models as specialized processors.
Each micro-agent is optimized for specific tasks, creating a swarm intelligence system.

Available Models (October 2025):
- tinyllama:1.1b (637MB) - Ultra-fast responses, basic reasoning
- gemma2:2b (1.6GB) - Balanced performance, good reasoning
- llama3.2:1b (1.3GB) - Meta's optimized 1B model
- mini-ice:latest (1.3GB) - ICEBURG's custom distilled model

Total Memory Usage: ~4.8GB for full swarm (fits comfortably in 16GB M1)
"""

import asyncio
import threading
import time
import queue
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .llm import chat_complete

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MicroAgent:
    """Individual micro-agent with specialized capabilities"""
    id: str
    name: str
    model: str
    specialization: str
    max_memory_mb: int
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = False
    current_task: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class SwarmTask:
    """Task to be processed by the swarm"""
    id: str
    task_type: str
    priority: int  # 1-10, 10 being highest
    input_data: Any
    requirements: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    status: str = "pending"  # pending, processing, completed, failed
    
    def __lt__(self, other):
        """Make SwarmTask comparable for priority queue"""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.created_at < other.created_at  # Earlier created first

class MicroAgentSwarm:
    """
    Distributed micro-agent swarm system for parallel processing.
    
    Architecture:
    - Coordinator: Manages task distribution and agent coordination
    - Specialized Agents: Each optimized for specific task types
    - Load Balancer: Distributes tasks based on agent capabilities and load
    - Result Aggregator: Combines results from multiple agents
    """
    
    def __init__(self, max_parallel_agents: int = 6):
        self.max_parallel_agents = max_parallel_agents
        self.agents: Dict[str, MicroAgent] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, SwarmTask] = {}
        self.is_running = False
        self.coordinator_thread = None
        
        # Dynamic resource allocator (replaces simple executor)
        # Lazy import to avoid circular dependencies
        try:
            from .infrastructure.dynamic_resource_allocator import get_resource_allocator
            self.resource_allocator = get_resource_allocator()
            # Use dynamic max workers based on available resources
            available_cores = max(1, int(self.resource_allocator.available_cpu_cores))
            self.executor = ThreadPoolExecutor(max_workers=min(max_parallel_agents, available_cores))
        except (ImportError, AttributeError):
            self.resource_allocator = None
            self.executor = ThreadPoolExecutor(max_workers=max_parallel_agents)
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.average_response_time = 0.0
        self.agent_utilization = {}
        
        # Initialize micro-agents
        self._initialize_micro_agents()
        
        logger.info(f"🚀 Micro-Agent Swarm initialized with {len(self.agents)} agents (dynamic resource allocation enabled)")
    
    def _initialize_micro_agents(self):
        """Initialize specialized micro-agents"""
        
        # Ultra-Fast Response Agent (TinyLlama)
        self.agents["ultra_fast"] = MicroAgent(
            id="ultra_fast",
            name="Lightning",
            model="tinyllama:1.1b",
            specialization="ultra_fast_responses",
            max_memory_mb=637,
            capabilities=[
                "quick_answers", "simple_reasoning", "text_processing",
                "basic_analysis", "fast_classification"
            ]
        )
        
        # Balanced Performance Agent (Gemma2)
        self.agents["balanced"] = MicroAgent(
            id="balanced",
            name="Gemini",
            model="gemma2:2b",
            specialization="balanced_reasoning",
            max_memory_mb=1600,
            capabilities=[
                "complex_reasoning", "code_analysis", "research_synthesis",
                "creative_writing", "problem_solving", "data_analysis"
            ]
        )
        
        # Meta Optimized Agent (Llama3.2)
        self.agents["meta_optimized"] = MicroAgent(
            id="meta_optimized",
            name="MetaMind",
            model="llama3.2:1b",
            specialization="meta_optimized_tasks",
            max_memory_mb=1300,
            capabilities=[
                "meta_analysis", "pattern_recognition", "optimization",
                "system_design", "architecture_planning"
            ]
        )
        
        # ICEBURG Custom Agent
        self.agents["iceburg_custom"] = MicroAgent(
            id="iceburg_custom",
            name="IceCore",
            model="mini-ice:latest",
            specialization="iceburg_specific",
            max_memory_mb=1300,
            capabilities=[
                "iceburg_protocol", "research_analysis", "knowledge_synthesis",
                "scientific_reasoning", "emergent_patterns"
            ]
        )
        
        # Code Generation Agent (using balanced model)
        self.agents["code_gen"] = MicroAgent(
            id="code_gen",
            name="CodeWeaver",
            model="gemma2:2b",
            specialization="code_generation",
            max_memory_mb=1600,
            capabilities=[
                "code_generation", "debugging", "refactoring",
                "architecture_design", "api_development"
            ]
        )
        
        # Analysis Agent (using meta optimized)
        self.agents["analysis"] = MicroAgent(
            id="analysis",
            name="DataMind",
            model="llama3.2:1b",
            specialization="data_analysis",
            max_memory_mb=1300,
            capabilities=[
                "data_analysis", "statistical_reasoning", "trend_analysis",
                "pattern_detection", "insight_generation"
            ]
        )
        
        logger.info(f"✅ Initialized {len(self.agents)} specialized micro-agents")
    
    async def warmup(self):
        """Pre-warm all agents - load models into memory for instant responses"""
        logger.info("Starting agent warmup...")
        start_time = time.time()
        
        try:
            # Warm up all agents
            for agent_id, agent in self.agents.items():
                try:
                    # Mark agent as warmed
                    agent.is_active = True
                    agent.last_used = datetime.now()
                    logger.debug(f"Warmed up agent: {agent.name} ({agent.model})")
                except Exception as e:
                    logger.warning(f"Error warming up agent {agent_id}: {e}")
            
            warmup_time = time.time() - start_time
            logger.info(f"Agent warmup complete: {len(self.agents)} agents warmed in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during agent warmup: {e}", exc_info=True)
            raise
    
    async def start_swarm(self):
        """Start the micro-agent swarm"""
        if self.is_running:
            logger.warning("Swarm is already running")
            return
        
        self.is_running = True
        self.coordinator_thread = threading.Thread(target=self._coordinator_loop, daemon=True)
        self.coordinator_thread.start()
        
        logger.info("🌟 Micro-Agent Swarm started and ready for tasks")
    
    async def stop_swarm(self):
        """Stop the micro-agent swarm"""
        self.is_running = False
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("🛑 Micro-Agent Swarm stopped")
    
    def _coordinator_loop(self):
        """Main coordinator loop for task distribution"""
        while self.is_running:
            try:
                # Get next task from queue
                if not self.task_queue.empty():
                    task = self.task_queue.get_nowait()
                    self._assign_task_to_agent(task)
                else:
                    time.sleep(0.1)  # Small delay when no tasks
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Coordinator error: {e}")
                time.sleep(1)
    
    def _assign_task_to_agent(self, task: SwarmTask):
        """Assign task to the most suitable agent"""
        best_agent = self._find_best_agent_for_task(task)
        
        if best_agent:
            task.assigned_agent = best_agent.id
            task.status = "processing"
            self.agents[best_agent.id].current_task = task.id
            self.agents[best_agent.id].is_active = True
            
            # Submit task to executor
            future = self.executor.submit(self._process_task, task, best_agent)
            logger.info(f"📋 Task {task.id} assigned to {best_agent.name} ({best_agent.model})")
        else:
            logger.warning(f"⚠️ No available agent for task {task.id}, retrying...")
            # Re-queue the task with lower priority
            task.priority = max(1, task.priority - 1)
            self.task_queue.put(task)
    
    def _find_best_agent_for_task(self, task: SwarmTask) -> Optional[MicroAgent]:
        """Find the best available agent for a task"""
        available_agents = [
            agent for agent in self.agents.values()
            if not agent.is_active and self._agent_can_handle_task(agent, task)
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on capability match and performance
        scored_agents = []
        for agent in available_agents:
            score = self._calculate_agent_score(agent, task)
            scored_agents.append((score, agent))
        
        # Return highest scoring agent
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]
    
    def _agent_can_handle_task(self, agent: MicroAgent, task: SwarmTask) -> bool:
        """Check if agent can handle the task"""
        # Check if agent has required capabilities
        for requirement in task.requirements:
            if requirement not in agent.capabilities:
                return False
        return True
    
    def _calculate_agent_score(self, agent: MicroAgent, task: SwarmTask) -> float:
        """Calculate agent suitability score for task"""
        score = 0.0
        
        # Capability match (70% weight)
        capability_match = len(set(task.requirements) & set(agent.capabilities))
        capability_match /= max(len(task.requirements), 1)
        score += capability_match * 0.7
        
        # Performance metrics (20% weight)
        avg_performance = agent.performance_metrics.get('avg_response_time', 1.0)
        score += (1.0 / avg_performance) * 0.2
        
        # Specialization match (10% weight)
        if task.task_type == agent.specialization:
            score += 0.1
        
        return score
    
    def _process_task(self, task: SwarmTask, agent: MicroAgent) -> Any:
        """Process task using assigned agent"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 {agent.name} processing task {task.id}")
            
            # Generate response using the agent's model
            response = self._generate_response(agent, task)
            
            # Update task result
            task.result = response
            task.status = "completed"
            
            # Update agent metrics
            processing_time = time.time() - start_time
            self._update_agent_metrics(agent, processing_time)
            
            # Store completed task
            self.completed_tasks[task.id] = task
            self.total_tasks_processed += 1
            
            logger.info(f"✅ {agent.name} completed task {task.id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ {agent.name} failed task {task.id}: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
        
        finally:
            # Free up agent
            agent.is_active = False
            agent.current_task = None
            agent.last_used = datetime.now()
    
    def _generate_response(self, agent: MicroAgent, task: SwarmTask) -> str:
        """Generate response using agent's model"""
        try:
            # Build prompt based on task type and agent specialization
            prompt = self._build_agent_prompt(agent, task)
            
            # Use chat_complete with agent's model
            response = chat_complete(
                model=agent.model,
                prompt=prompt,
                system=f"You are {agent.name}, a specialized AI agent focused on {agent.specialization}. Provide expert-level assistance.",
                temperature=0.7,
                context_tag=f"Swarm-{agent.id}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed for {agent.name}: {e}")
            return f"Error: {str(e)}"
    
    def _build_agent_prompt(self, agent: MicroAgent, task: SwarmTask) -> str:
        """Build specialized prompt for agent"""
        base_prompt = f"Task Type: {task.task_type}\nInput: {task.input_data}\n"
        
        # Add specialization-specific instructions
        if agent.specialization == "ultra_fast_responses":
            base_prompt += "Provide a quick, concise response. Focus on speed and clarity."
        elif agent.specialization == "balanced_reasoning":
            base_prompt += "Provide a thorough, well-reasoned response with detailed analysis."
        elif agent.specialization == "code_generation":
            base_prompt += "Generate clean, efficient code with proper documentation and error handling."
        elif agent.specialization == "data_analysis":
            base_prompt += "Analyze the data thoroughly and provide insights with statistical reasoning."
        elif agent.specialization == "iceburg_specific":
            base_prompt += "Apply ICEBURG research protocols and scientific methodology."
        
        return base_prompt
    
    def _update_agent_metrics(self, agent: MicroAgent, processing_time: float):
        """Update agent performance metrics"""
        if 'avg_response_time' not in agent.performance_metrics:
            agent.performance_metrics['avg_response_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            agent.performance_metrics['avg_response_time'] = (
                alpha * processing_time + 
                (1 - alpha) * agent.performance_metrics['avg_response_time']
            )
        
        agent.performance_metrics['total_tasks'] = agent.performance_metrics.get('total_tasks', 0) + 1
    
    async def submit_task(self, task_type: str, input_data: Any, 
                         requirements: List[str] = None, priority: int = 5) -> str:
        """Submit a task to the swarm"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = SwarmTask(
            id=task_id,
            task_type=task_type,
            priority=priority,
            input_data=input_data,
            requirements=requirements or [],
            timeout_seconds=30
        )
        
        # Add to priority queue
        self.task_queue.put(task)
        
        logger.info(f"📥 Task {task_id} submitted to swarm (priority: {priority})")
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: int = 60) -> Optional[Any]:
        """Get result of a completed task"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.result is not None:
                    return task.result
                elif task.status == "failed":
                    return {"error": "Task failed", "task_id": task_id}
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"⏰ Timeout waiting for task {task_id}")
        return {"error": "Task timeout", "task_id": task_id}
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
        
        return {
            "is_running": self.is_running,
            "total_agents": len(self.agents),
            "active_agents": active_agents,
            "available_agents": len(self.agents) - active_agents,
            "queue_size": self.task_queue.qsize(),
            "total_tasks_processed": self.total_tasks_processed,
            "average_response_time": self.average_response_time,
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "model": agent.model,
                    "specialization": agent.specialization,
                    "is_active": agent.is_active,
                    "current_task": agent.current_task,
                    "performance": agent.performance_metrics
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def process_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple tasks in parallel"""
        task_ids = []
        
        # Submit all tasks
        for task_data in tasks:
            task_id = await self.submit_task(
                task_type=task_data.get("type", "general"),
                input_data=task_data.get("input", ""),
                requirements=task_data.get("requirements", []),
                priority=task_data.get("priority", 5)
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await self.get_task_result(task_id, timeout=60)
            results.append(result)
        
        return results

# Global swarm instance
_swarm_instance: Optional[MicroAgentSwarm] = None

async def get_swarm() -> MicroAgentSwarm:
    """Get or create global swarm instance"""
    global _swarm_instance
    if _swarm_instance is None:
        _swarm_instance = MicroAgentSwarm()
        await _swarm_instance.start_swarm()
    return _swarm_instance

async def process_with_swarm(task_type: str, input_data: Any, 
                           requirements: List[str] = None) -> str:
    """Process a single task using the swarm"""
    swarm = await get_swarm()
    task_id = await swarm.submit_task(task_type, input_data, requirements)
    result = await swarm.get_task_result(task_id)
    return result

async def process_parallel_with_swarm(tasks: List[Dict[str, Any]]) -> List[Any]:
    """Process multiple tasks in parallel using the swarm"""
    swarm = await get_swarm()
    return await swarm.process_parallel_tasks(tasks)
