"""
ICEBURG Think Tank Department

A powerful department system for distributed intelligence and collaborative brainstorming.
Enables autonomous scaling of ICEBURG's thinking capabilities across multiple specialized departments.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

class DepartmentType(Enum):
    RESEARCH = "research"
    DEVELOPMENT = "development"
    SECURITY = "security"
    QUALITY = "quality"
    INNOVATION = "innovation"
    STRATEGY = "strategy"
    OPERATIONS = "operations"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ThinkTankTask:
    task_id: str
    department: DepartmentType
    priority: TaskPriority
    description: str
    context: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)
    collaboration_requests: List[str] = field(default_factory=list)

@dataclass
class DepartmentAgent:
    agent_id: str
    department: DepartmentType
    specialization: str
    capabilities: List[str]
    current_tasks: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    performance_score: float = 1.0
    collaboration_history: Dict[str, float] = field(default_factory=dict)

class ThinkTankDepartment:
    """
    A powerful department system for ICEBURG think tanks that enables:
    - Distributed intelligence across specialized departments
    - Autonomous brainstorming and problem-solving
    - Collaborative task distribution and management
    - Scalable department creation and management
    """
    
    def __init__(self, name: str, department_type: DepartmentType):
        self.name = name
        self.department_type = department_type
        self.agents: Dict[str, DepartmentAgent] = {}
        self.tasks: Dict[str, ThinkTankTask] = {}
        self.collaboration_network: Dict[str, List[str]] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        
    def add_agent(self, agent_id: str, specialization: str, capabilities: List[str]) -> DepartmentAgent:
        """Add a new agent to the department"""
        agent = DepartmentAgent(
            agent_id=agent_id,
            department=self.department_type,
            specialization=specialization,
            capabilities=capabilities
        )
        self.agents[agent_id] = agent
        return agent
    
    def create_task(self, description: str, priority: TaskPriority = TaskPriority.MEDIUM, 
                   context: Dict[str, Any] = None, deadline: Optional[float] = None) -> ThinkTankTask:
        """Create a new task for the department"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = ThinkTankTask(
            task_id=task_id,
            department=self.department_type,
            priority=priority,
            description=description,
            context=context or {},
            deadline=deadline
        )
        self.tasks[task_id] = task
        return task
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent"""
        if task_id not in self.tasks or agent_id not in self.agents:
            return False
        
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        # Check if agent can handle more tasks
        if len(agent.current_tasks) >= agent.max_concurrent_tasks:
            return False
        
        # Assign task
        task.assigned_agent = agent_id
        task.status = "assigned"
        agent.current_tasks.append(task_id)
        
        return True
    
    def auto_assign_tasks(self) -> Dict[str, str]:
        """Automatically assign tasks to available agents based on capabilities"""
        assignments = {}
        
        for task_id, task in self.tasks.items():
            if task.status != "pending":
                continue
            
            # Find best agent for this task
            best_agent = self._find_best_agent_for_task(task)
            if best_agent:
                if self.assign_task(task_id, best_agent.agent_id):
                    assignments[task_id] = best_agent.agent_id
        
        return assignments
    
    def _find_best_agent_for_task(self, task: ThinkTankTask) -> Optional[DepartmentAgent]:
        """Find the best agent for a given task based on capabilities and availability"""
        available_agents = [
            agent for agent in self.agents.values()
            if len(agent.current_tasks) < agent.max_concurrent_tasks
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on capabilities and performance
        scored_agents = []
        for agent in available_agents:
            score = self._calculate_agent_score(agent, task)
            scored_agents.append((score, agent))
        
        # Return highest scoring agent
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1] if scored_agents else None
    
    def _calculate_agent_score(self, agent: DepartmentAgent, task: ThinkTankTask) -> float:
        """Calculate how well an agent matches a task"""
        base_score = agent.performance_score
        
        # Capability matching
        capability_match = 0
        for capability in agent.capabilities:
            if capability.lower() in task.description.lower():
                capability_match += 1
        
        capability_score = capability_match / len(agent.capabilities) if agent.capabilities else 0
        
        # Collaboration history
        collaboration_score = 0
        if task.assigned_agent in agent.collaboration_history:
            collaboration_score = agent.collaboration_history[task.assigned_agent]
        
        return base_score + capability_score + collaboration_score
    
    def collaborate_with_department(self, other_department: 'ThinkTankDepartment', 
                                 task_id: str, collaboration_type: str) -> bool:
        """Initiate collaboration with another department"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.collaboration_requests.append(f"{other_department.name}:{collaboration_type}")
        
        # Update collaboration network
        if self.name not in self.collaboration_network:
            self.collaboration_network[self.name] = []
        if other_department.name not in self.collaboration_network[self.name]:
            self.collaboration_network[self.name].append(other_department.name)
        
        return True
    
    def brainstorm_solution(self, problem: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Conduct a brainstorming session to generate solutions"""
        solutions = []
        
        # Gather input from all agents in the department
        for agent in self.agents.values():
            if agent.specialization in ["research", "innovation", "strategy"]:
                solution = self._generate_agent_solution(agent, problem, context or {})
                if solution:
                    solutions.append(solution)
        
        # Collaborative refinement
        refined_solutions = self._refine_solutions_collaboratively(solutions, problem)
        
        return refined_solutions
    
    def _generate_agent_solution(self, agent: DepartmentAgent, problem: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a solution from a specific agent's perspective"""
        # This would integrate with ICEBURG's reasoning engines
        solution = {
            "agent_id": agent.agent_id,
            "specialization": agent.specialization,
            "approach": f"{agent.specialization}-based approach",
            "solution": f"Solution from {agent.specialization} perspective for: {problem}",
            "confidence": agent.performance_score,
            "capabilities_used": agent.capabilities,
            "timestamp": time.time()
        }
        
        return solution
    
    def _refine_solutions_collaboratively(self, solutions: List[Dict[str, Any]], problem: str) -> List[Dict[str, Any]]:
        """Refine solutions through collaborative discussion"""
        if not solutions:
            return []
        
        # Sort by confidence and capability match
        solutions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Generate collaborative insights
        collaborative_insights = []
        for i, solution in enumerate(solutions):
            insight = {
                **solution,
                "collaborative_score": self._calculate_collaborative_score(solution, solutions),
                "refinement_notes": f"Refined through {len(solutions)} agent perspectives"
            }
            collaborative_insights.append(insight)
        
        return collaborative_insights
    
    def _calculate_collaborative_score(self, solution: Dict[str, Any], all_solutions: List[Dict[str, Any]]) -> float:
        """Calculate how well a solution benefits from collaboration"""
        base_score = solution["confidence"]
        
        # Bonus for unique capabilities
        unique_capabilities = set(solution.get("capabilities_used", []))
        for other_solution in all_solutions:
            if other_solution["agent_id"] != solution["agent_id"]:
                other_capabilities = set(other_solution.get("capabilities_used", []))
                unique_capabilities.update(other_capabilities)
        
        collaboration_bonus = len(unique_capabilities) / 10.0  # Normalize
        
        return base_score + collaboration_bonus
    
    def scale_department(self, scale_factor: int) -> List[DepartmentAgent]:
        """Scale the department by adding more agents"""
        new_agents = []
        
        for i in range(scale_factor):
            agent_id = f"{self.name}_agent_{len(self.agents) + i + 1}"
            specialization = f"specialized_{i % 3}"  # Rotate specializations
            capabilities = self._generate_capabilities_for_specialization(specialization)
            
            agent = self.add_agent(agent_id, specialization, capabilities)
            new_agents.append(agent)
        
        return new_agents
    
    def _generate_capabilities_for_specialization(self, specialization: str) -> List[str]:
        """Generate appropriate capabilities for a specialization"""
        capability_map = {
            "specialized_0": ["analysis", "research", "pattern_recognition"],
            "specialized_1": ["development", "implementation", "optimization"],
            "specialized_2": ["strategy", "planning", "coordination"]
        }
        
        return capability_map.get(specialization, ["general", "problem_solving"])
    
    def get_department_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the department"""
        return {
            "name": self.name,
            "type": self.department_type.value,
            "total_agents": len(self.agents),
            "active_tasks": len([t for t in self.tasks.values() if t.status == "assigned"]),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "collaboration_network": self.collaboration_network,
            "performance_metrics": self.performance_metrics
        }

class ThinkTankCoordinator:
    """
    Coordinates multiple think tank departments for large-scale problem solving
    """
    
    def __init__(self):
        self.departments: Dict[str, ThinkTankDepartment] = {}
        self.global_tasks: Dict[str, ThinkTankTask] = {}
        self.cross_department_collaborations: Dict[str, List[str]] = {}
    
    def create_department(self, name: str, department_type: DepartmentType, 
                         initial_agents: int = 3) -> ThinkTankDepartment:
        """Create a new think tank department"""
        department = ThinkTankDepartment(name, department_type)
        
        # Add initial agents
        for i in range(initial_agents):
            agent_id = f"{name}_agent_{i+1}"
            specialization = f"specialist_{i % 3}"
            capabilities = department._generate_capabilities_for_specialization(specialization)
            department.add_agent(agent_id, specialization, capabilities)
        
        self.departments[name] = department
        return department
    
    def create_mega_department(self, name: str, scale: int = 10) -> ThinkTankDepartment:
        """Create a large-scale department for complex problems"""
        department = ThinkTankDepartment(name, DepartmentType.RESEARCH)
        
        # Add many specialized agents
        for i in range(scale):
            agent_id = f"{name}_mega_agent_{i+1}"
            specialization = f"mega_specialist_{i % 5}"
            capabilities = [
                "advanced_analysis", "complex_problem_solving", "multi_domain_expertise",
                "collaborative_reasoning", "innovative_thinking"
            ]
            department.add_agent(agent_id, specialization, capabilities)
        
        self.departments[name] = department
        return department
    
    def coordinate_cross_department_brainstorm(self, problem: str, 
                                            participating_departments: List[str],
                                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Coordinate brainstorming across multiple departments"""
        all_solutions = []
        
        for dept_name in participating_departments:
            if dept_name in self.departments:
                department = self.departments[dept_name]
                solutions = department.brainstorm_solution(problem, context)
                all_solutions.extend(solutions)
        
        # Cross-department solution synthesis
        synthesized_solution = self._synthesize_cross_department_solutions(all_solutions, problem)
        
        return {
            "problem": problem,
            "participating_departments": participating_departments,
            "total_solutions": len(all_solutions),
            "synthesized_solution": synthesized_solution,
            "department_contributions": {
                dept: len([s for s in all_solutions if dept in s.get("agent_id", "")])
                for dept in participating_departments
            }
        }
    
    def _synthesize_cross_department_solutions(self, solutions: List[Dict[str, Any]], problem: str) -> Dict[str, Any]:
        """Synthesize solutions from multiple departments"""
        if not solutions:
            return {"synthesis": "No solutions generated"}
        
        # Find the best solution
        best_solution = max(solutions, key=lambda x: x.get("collaborative_score", 0))
        
        # Generate synthesis
        synthesis = {
            "primary_solution": best_solution,
            "alternative_approaches": [s for s in solutions if s != best_solution],
            "synthesis_notes": f"Synthesized from {len(solutions)} solutions across multiple departments",
            "confidence_level": best_solution.get("collaborative_score", 0),
            "implementation_recommendation": f"Implement {best_solution['approach']} with cross-department collaboration"
        }
        
        return synthesis
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all departments and global coordination"""
        department_statuses = {
            name: dept.get_department_status() 
            for name, dept in self.departments.items()
        }
        
        return {
            "total_departments": len(self.departments),
            "total_agents": sum(status["total_agents"] for status in department_statuses.values()),
            "active_tasks": sum(status["active_tasks"] for status in department_statuses.values()),
            "department_statuses": department_statuses,
            "cross_department_collaborations": len(self.cross_department_collaborations)
        }
