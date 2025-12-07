"""
Secretary Agent Planning Engine
Handles goal-driven autonomy and multi-step task planning.
"""

from typing import List, Dict, Any, Optional
import logging
import json
from dataclasses import dataclass, field
from enum import Enum

from ..config import IceburgConfig
from ..civilization.persistent_agents import GoalHierarchy, GoalPriority, Goal

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in a plan."""
    task_id: str
    description: str
    goal_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecretaryPlanner:
    """
    Planning engine for Secretary agent.
    
    Features:
    - Goal parsing from natural language
    - Task decomposition
    - Dependency resolution
    - Multi-step execution
    - Progress tracking
    """
    
    def __init__(self, cfg: IceburgConfig):
        """
        Initialize planning engine.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        self.goal_hierarchy = GoalHierarchy(max_goals=50)
        self.task_counter = 0
    
    def extract_goals(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract goals from natural language query.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of goal dictionaries with description, priority, etc.
        """
        # Use LLM to identify goals from query
        from ..providers.factory import provider_factory
        
        provider = provider_factory(self.cfg)
        model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
        
        goal_extraction_prompt = f"""Analyze this user query and identify any goals or multi-step tasks:

Query: {query}

If this query represents a goal or task that requires multiple steps, return a JSON object with:
{{
    "is_goal": true/false,
    "goals": [
        {{
            "description": "clear goal description",
            "priority": "critical|high|medium|low",
            "deadline": null or timestamp if mentioned,
            "sub_goals": ["sub-goal 1", "sub-goal 2", ...]
        }}
    ]
}}

If this is just a simple question that doesn't require planning, return:
{{
    "is_goal": false,
    "goals": []
}}

Return ONLY valid JSON, no other text."""

        try:
            response = provider.chat_complete(
                model=model_to_use,
                prompt=goal_extraction_prompt,
                system="You are a goal extraction system. Extract goals from user queries accurately.",
                temperature=0.3,
                options={"max_tokens": 500},
            )
            
            # Parse JSON response
            response = response.strip()
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            goal_data = json.loads(response)
            
            if goal_data.get("is_goal", False):
                return goal_data.get("goals", [])
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Error extracting goals: {e}. Treating as simple query.")
            return []
    
    def plan_task(self, goal: Dict[str, Any]) -> List[Task]:
        """
        Break goal into executable tasks.
        
        Args:
            goal: Goal dictionary with description, priority, etc.
            
        Returns:
            List of Task objects in execution order
        """
        goal_description = goal.get("description", "")
        sub_goals = goal.get("sub_goals", [])
        
        # If sub-goals are provided, use them
        if sub_goals:
            tasks = []
            start_counter = self.task_counter
            for i, sub_goal in enumerate(sub_goals):
                # Dependencies are previous tasks in the sequence
                dependencies = [f"task_{start_counter + j}" for j in range(i)] if i > 0 else []
                task = Task(
                    task_id=f"task_{self.task_counter}",
                    description=sub_goal,
                    goal_id=goal.get("goal_id"),
                    dependencies=dependencies,
                    status=TaskStatus.PENDING
                )
                tasks.append(task)
                self.task_counter += 1
            return tasks
        
        # Otherwise, decompose using LLM
        from ..providers.factory import provider_factory
        
        provider = provider_factory(self.cfg)
        model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
        
        decomposition_prompt = f"""Break down this goal into specific, executable steps:

Goal: {goal_description}

Return a JSON array of task descriptions in execution order:
[
    "Step 1: description",
    "Step 2: description",
    ...
]

Each step should be:
- Specific and actionable
- Clear about what needs to be done
- In the correct execution order

Return ONLY a valid JSON array, no other text."""

        try:
            response = provider.chat_complete(
                model=model_to_use,
                prompt=decomposition_prompt,
                system="You are a task planning system. Break goals into clear, executable steps.",
                temperature=0.3,
                options={"max_tokens": 800},
            )
            
            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            task_descriptions = json.loads(response)
            
            # Create Task objects
            tasks = []
            start_counter = self.task_counter
            for i, task_desc in enumerate(task_descriptions):
                # Dependencies are previous tasks in the sequence
                dependencies = [f"task_{start_counter + j}" for j in range(i)] if i > 0 else []
                task = Task(
                    task_id=f"task_{self.task_counter}",
                    description=task_desc,
                    goal_id=goal.get("goal_id"),
                    dependencies=dependencies,
                    status=TaskStatus.PENDING
                )
                tasks.append(task)
                self.task_counter += 1
            
            return tasks
            
        except Exception as e:
            logger.warning(f"Error decomposing goal: {e}. Creating single task.")
            # Fallback: create single task
            task = Task(
                task_id=f"task_{self.task_counter}",
                description=goal_description,
                goal_id=goal.get("goal_id"),
                status=TaskStatus.PENDING
            )
            self.task_counter += 1
            return [task]
    
    def execute_task(self, task: Task, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            context: Context from previous tasks
            
        Returns:
            Execution result dictionary
        """
        if context is None:
            context = {}
        
        task.status = TaskStatus.IN_PROGRESS
        logger.info(f"Executing task: {task.description}")
        
        try:
            # Use tool usage to execute task
            # For now, we'll use a simple approach: ask LLM to execute
            # In the future, this can be enhanced with actual tool execution
            
            from ..providers.factory import provider_factory
            
            provider = provider_factory(self.cfg)
            model_to_use = getattr(self.cfg, "surveyor_model", None) or getattr(self.cfg, "primary_model", None) or "gemini-2.0-flash-exp"
            
            execution_prompt = f"""Execute this task:

Task: {task.description}

Context from previous steps:
{json.dumps(context, indent=2) if context else "None"}

Provide a brief summary of what was done or what information was gathered.
If the task requires actual file operations or system commands, describe what should be done.
Be concise and specific."""

            response = provider.chat_complete(
                model=model_to_use,
                prompt=execution_prompt,
                system="You are a task execution system. Execute tasks and report results.",
                temperature=0.3,
                options={"max_tokens": 500},
            )
            
            task.status = TaskStatus.COMPLETED
            task.result = response.strip()
            
            return {
                "success": True,
                "result": task.result,
                "task_id": task.task_id
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task.task_id
            }
    
    def execute_plan(self, tasks: List[Task], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Execute a plan of tasks with dependency resolution.
        
        Args:
            tasks: List of tasks to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Execution summary with results
        """
        execution_context = {}
        results = []
        
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        completed_tasks = set()
        
        # Execute tasks in dependency order
        while len(completed_tasks) < len(tasks):
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = [
                task for task in tasks
                if task.task_id not in completed_tasks
                and all(dep_id in completed_tasks for dep_id in task.dependencies)
            ]
            
            if not ready_tasks:
                # Check for circular dependencies or missing tasks
                remaining = [t for t in tasks if t.task_id not in completed_tasks]
                if remaining:
                    logger.warning(f"Circular dependency or missing tasks detected. Remaining: {[t.task_id for t in remaining]}")
                    # Mark remaining as failed
                    for task in remaining:
                        task.status = TaskStatus.FAILED
                        task.error = "Dependency resolution failed"
                break
            
            # Execute ready tasks (can be parallel in future)
            for task in ready_tasks:
                if progress_callback:
                    progress_callback(f"Executing: {task.description}")
                
                result = self.execute_task(task, execution_context)
                results.append(result)
                
                # Update context with result
                execution_context[task.task_id] = {
                    "description": task.description,
                    "result": task.result if result["success"] else None,
                    "error": task.error if not result["success"] else None
                }
                
                completed_tasks.add(task.task_id)
                
                # If task failed, we might want to stop or continue
                if not result["success"]:
                    logger.warning(f"Task {task.task_id} failed: {result.get('error')}")
                    # Continue with other tasks
        
        # Build summary
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        
        return {
            "total_tasks": len(tasks),
            "completed": successful,
            "failed": failed,
            "results": results,
            "context": execution_context
        }
    
    def get_goal_progress(self, goal_id: str) -> Dict[str, Any]:
        """
        Get progress for a goal.
        
        Args:
            goal_id: Goal ID
            
        Returns:
            Progress information
        """
        goal = self.goal_hierarchy.goals.get(goal_id)
        if not goal:
            return {"error": "Goal not found"}
        
        return {
            "goal_id": goal_id,
            "description": goal.description,
            "progress": goal.progress,
            "priority": goal.priority.value,
            "status": "completed" if goal.progress >= 1.0 else "in_progress"
        }

