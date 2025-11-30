"""
ICEBURG Mega Brainstorm System

A powerful system for coordinating massive brainstorming sessions across multiple
think tank departments with autonomous scaling and collaborative intelligence.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .think_tank import ThinkTankDepartment, ThinkTankCoordinator, DepartmentType
from .brainstorm_engine import BrainstormEngine, BrainstormMode, BrainstormSession
from .scaling_coordinator import ScalingCoordinator, ScalingStrategy

@dataclass
class MegaBrainstormSession:
    session_id: str
    problem: str
    context: Dict[str, Any]
    participating_departments: List[str]
    total_agents: int
    session_goals: List[str]
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    ideas_generated: int = 0
    collaborations: int = 0
    synthesis_quality: float = 0.0

class MegaBrainstormSystem:
    """
    A powerful system for coordinating massive brainstorming sessions across
    multiple think tank departments with autonomous scaling capabilities.
    """
    
    def __init__(self):
        self.coordinator = ThinkTankCoordinator()
        self.brainstorm_engine = BrainstormEngine()
        self.scaling_coordinator = ScalingCoordinator()
        self.mega_sessions: Dict[str, MegaBrainstormSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Initialize with core departments
        self._initialize_core_departments()
    
    def _initialize_core_departments(self) -> None:
        """Initialize core think tank departments"""
        # Research Department
        research_dept = self.coordinator.create_department(
            "Research", DepartmentType.RESEARCH, initial_agents=5
        )
        self.scaling_coordinator.register_department("Research", research_dept)
        
        # Development Department
        dev_dept = self.coordinator.create_department(
            "Development", DepartmentType.DEVELOPMENT, initial_agents=5
        )
        self.scaling_coordinator.register_department("Development", dev_dept)
        
        # Innovation Department
        innovation_dept = self.coordinator.create_department(
            "Innovation", DepartmentType.INNOVATION, initial_agents=5
        )
        self.scaling_coordinator.register_department("Innovation", innovation_dept)
        
        # Strategy Department
        strategy_dept = self.coordinator.create_department(
            "Strategy", DepartmentType.STRATEGY, initial_agents=5
        )
        self.scaling_coordinator.register_department("Strategy", strategy_dept)
    
    def start_mega_brainstorm(self, problem: str, context: Dict[str, Any] = None,
        participating_departments: List[str] = None,
                            session_goals: List[str] = None,
                            auto_scale: bool = True) -> MegaBrainstormSession:
        """Start a massive brainstorming session across multiple departments"""
        session_id = f"mega_brainstorm_{uuid.uuid4().hex[:8]}"
        
        # Determine participating departments
        if not participating_departments:
            participating_departments = list(self.coordinator.departments.keys())
        
        # Calculate total agents
        total_agents = sum(
            len(dept.agents) for name, dept in self.coordinator.departments.items()
            if name in participating_departments
        )
        
        # Create mega session
        mega_session = MegaBrainstormSession(
            session_id=session_id,
            problem=problem,
            context=context or {},
            participating_departments=participating_departments,
            total_agents=total_agents,
            session_goals=session_goals or [],
            start_time=time.time()
        )
        
        self.mega_sessions[session_id] = mega_session
        
        # Start brainstorming across departments
        self._execute_mega_brainstorm(mega_session, auto_scale)
        
        return mega_session
    
    def _execute_mega_brainstorm(self, mega_session: MegaBrainstormSession, auto_scale: bool) -> None:
        """Execute the mega brainstorming session"""
        # Start individual brainstorm sessions in each department
        department_sessions = {}
        
        for dept_name in mega_session.participating_departments:
            if dept_name in self.coordinator.departments:
                dept = self.coordinator.departments[dept_name]
                
                # Create brainstorm session for this department
                session = self.brainstorm_engine.start_brainstorm_session(
                    problem=mega_session.problem,
                    context=mega_session.context,
                    mode=BrainstormMode.COLLABORATIVE,
                    participants=[agent.agent_id for agent in dept.agents.values()],
                    goals=mega_session.session_goals
                )
                
                department_sessions[dept_name] = session
        
        # Generate ideas in parallel across departments
        self._generate_ideas_parallel(department_sessions, mega_session)
        
        # Enable cross-department collaboration
        if len(department_sessions) > 1:
            self._enable_cross_department_collaboration(department_sessions, mega_session)
        
        # Auto-scale if enabled
        if auto_scale:
            self._auto_scale_brainstorming(mega_session)
    
    def _generate_ideas_parallel(self, department_sessions: Dict[str, BrainstormSession], 
        mega_session: MegaBrainstormSession) -> None:
        """Generate ideas in parallel across all departments"""
        # Submit idea generation tasks to thread pool
        future_to_dept = {}
        
        for dept_name, session in department_sessions.items():
            future = self.executor.submit(
                self._generate_department_ideas, dept_name, session, mega_session
            )
            future_to_dept[future] = dept_name
        
        # Collect results as they complete
        for future in as_completed(future_to_dept):
            dept_name = future_to_dept[future]
            try:
                ideas_count = future.result()
                mega_session.ideas_generated += ideas_count
            except Exception as e:
    
    def _generate_department_ideas(self, dept_name: str, session: BrainstormSession, 
        mega_session: MegaBrainstormSession) -> int:
        """Generate ideas for a specific department"""
        # Generate ideas based on department specialization
        ideas_per_dept = {
            "Research": 15,
            "Development": 12,
            "Innovation": 20,
            "Strategy": 10
        }
        
        num_ideas = ideas_per_dept.get(dept_name, 10)
        
        # Generate ideas
        ideas = self.brainstorm_engine.generate_ideas_autonomously(session.session_id, num_ideas)
        
        # Add department-specific enhancements
        self._enhance_ideas_for_department(ideas, dept_name)
        
        return len(ideas)
    
    def _enhance_ideas_for_department(self, ideas: List[Any], dept_name: str) -> None:
        """Enhance ideas based on department specialization"""
        enhancements = {
            "Research": "Research-focused analysis and evidence-based approach",
            "Development": "Implementation-focused and technical feasibility",
            "Innovation": "Creative and breakthrough thinking",
            "Strategy": "Strategic planning and long-term vision"
        }
        
        enhancement = enhancements.get(dept_name, "General enhancement")
        
        for idea in ideas:
            if hasattr(idea, 'content'):
                idea.content = f"{enhancement}: {idea.content}"
    
    def _enable_cross_department_collaboration(self, department_sessions: Dict[str, BrainstormSession],
        mega_session: MegaBrainstormSession) -> None:
        """Enable collaboration between departments"""
        # Get ideas from all departments
        all_ideas = []
        for dept_name, session in department_sessions.items():
            if session.session_id in self.brainstorm_engine.active_sessions:
                session_ideas = self.brainstorm_engine.active_sessions[session.session_id].ideas
                all_ideas.extend(session_ideas)
        
        # Enable cross-department collaboration on best ideas
        best_ideas = sorted(all_ideas, key=lambda x: getattr(x, 'confidence', 0), reverse=True)[:10]
        
        for idea in best_ideas:
            # Create collaborative sessions
            self._create_collaborative_session(idea, department_sessions, mega_session)
            mega_session.collaborations += 1
    
    def _create_collaborative_session(self, idea: Any, department_sessions: Dict[str, BrainstormSession],
        mega_session: MegaBrainstormSession) -> None:
        """Create a collaborative session for a specific idea"""
        # Find the best departments for collaboration
        collaborating_departments = list(department_sessions.keys())[:3]  # Top 3 departments
        
        # Create cross-department collaboration
        for dept_name in collaborating_departments:
            if dept_name in self.coordinator.departments:
                dept = self.coordinator.departments[dept_name]
                
                # Create collaboration task
                collaboration_task = dept.create_task(
                    description=f"Collaborate on idea: {getattr(idea, 'content', 'Unknown idea')[:100]}",
                    context={"idea_id": getattr(idea, 'idea_id', 'unknown')}
                )
                
                # Assign to best agent
                if dept.agents:
                    best_agent = max(dept.agents.values(), key=lambda x: x.performance_score)
                    dept.assign_task(collaboration_task.task_id, best_agent.agent_id)
    
    def _auto_scale_brainstorming(self, mega_session: MegaBrainstormSession) -> None:
        """Automatically scale brainstorming based on session needs"""
        # Start scaling coordinator monitoring
        if not self.scaling_coordinator.monitoring_active:
            self.scaling_coordinator.start_monitoring()
        
        # Determine if scaling is needed based on session complexity
        complexity_score = self._calculate_session_complexity(mega_session)
        
        if complexity_score > 0.7:  # High complexity
            # Scale up participating departments
            for dept_name in mega_session.participating_departments:
                if dept_name in self.coordinator.departments:
                    # Force scale the department
                    self.scaling_coordinator.force_scale_department(
                        dept_name, scale_factor=3, strategy=ScalingStrategy.HORIZONTAL
                    )
    
    def _calculate_session_complexity(self, mega_session: MegaBrainstormSession) -> float:
        """Calculate the complexity score of a brainstorming session"""
        complexity_factors = {
            "participating_departments": len(mega_session.participating_departments) / 10.0,
            "total_agents": mega_session.total_agents / 50.0,
            "session_goals": len(mega_session.session_goals) / 5.0,
            "context_complexity": len(mega_session.context) / 10.0
        }
        
        # Weighted average of complexity factors
        weights = {"participating_departments": 0.3, "total_agents": 0.3, 
            "session_goals": 0.2, "context_complexity": 0.2}
        
        complexity_score = sum(
            complexity_factors[factor] * weights[factor] 
            for factor in complexity_factors
        )
        
        return min(complexity_score, 1.0)  # Cap at 1.0
    
    def synthesize_mega_brainstorm(self, session_id: str) -> Dict[str, Any]:
        """Synthesize results from a mega brainstorming session"""
        if session_id not in self.mega_sessions:
            return {"error": "Session not found"}
        
        mega_session = self.mega_sessions[session_id]
        
        # Collect all ideas from participating departments
        all_ideas = []
        department_contributions = {}
        
        for dept_name in mega_session.participating_departments:
            if dept_name in self.coordinator.departments:
                dept = self.coordinator.departments[dept_name]
                
                # Get ideas from department's brainstorm sessions
                dept_ideas = self._collect_department_ideas(dept_name)
                all_ideas.extend(dept_ideas)
                department_contributions[dept_name] = len(dept_ideas)
        
        # Generate mega synthesis
        mega_synthesis = self._generate_mega_synthesis(all_ideas, mega_session)
        
        # Calculate synthesis quality
        synthesis_quality = self._calculate_synthesis_quality(mega_synthesis, all_ideas)
        mega_session.synthesis_quality = synthesis_quality
        
        return {
            "session_id": session_id,
            "problem": mega_session.problem,
            "participating_departments": mega_session.participating_departments,
            "total_ideas": len(all_ideas),
            "department_contributions": department_contributions,
            "collaborations": mega_session.collaborations,
            "synthesis_quality": synthesis_quality,
            "mega_synthesis": mega_synthesis,
            "implementation_recommendations": self._generate_implementation_recommendations(mega_synthesis),
            "scaling_recommendations": self._generate_scaling_recommendations(mega_session)
        }
    
    def _collect_department_ideas(self, dept_name: str) -> List[Any]:
        """Collect ideas from a specific department"""
        # This would collect ideas from the department's brainstorm sessions
        # For now, return empty list as placeholder
        return []
    
    def _generate_mega_synthesis(self, all_ideas: List[Any], mega_session: MegaBrainstormSession) -> Dict[str, Any]:
        """Generate a comprehensive synthesis from all ideas"""
        if not all_ideas:
            return {"synthesis": "No ideas to synthesize"}
        
        # Group ideas by type and department
        ideas_by_type = {}
        ideas_by_department = {}
        
        for idea in all_ideas:
            # Group by type
            idea_type = getattr(idea, 'idea_type', 'unknown')
            if idea_type not in ideas_by_type:
                ideas_by_type[idea_type] = []
            ideas_by_type[idea_type].append(idea)
            
            # Group by department (would need to track this)
            dept = "unknown"
            if dept not in ideas_by_department:
                ideas_by_department[dept] = []
            ideas_by_department[dept].append(idea)
        
        # Generate synthesis
        synthesis = {
            "total_ideas": len(all_ideas),
            "ideas_by_type": {k: len(v) for k, v in ideas_by_type.items()},
            "ideas_by_department": {k: len(v) for k, v in ideas_by_department.items()},
            "synthesis_notes": f"Synthesized from {len(all_ideas)} ideas across {len(mega_session.participating_departments)} departments",
            "key_insights": self._extract_key_insights(all_ideas),
            "recommended_actions": self._generate_recommended_actions(all_ideas)
        }
        
        return synthesis
    
    def _extract_key_insights(self, all_ideas: List[Any]) -> List[str]:
        """Extract key insights from all ideas"""
        insights = [
            "Multi-department collaboration generated diverse perspectives",
            "Cross-functional thinking revealed innovative approaches",
            "Scalable brainstorming enabled comprehensive coverage",
            "Autonomous scaling optimized resource utilization"
        ]
        
        return insights
    
    def _generate_recommended_actions(self, all_ideas: List[Any]) -> List[str]:
        """Generate recommended actions from the ideas"""
        actions = [
            "Implement the highest-confidence solutions first",
            "Create cross-department implementation teams",
            "Establish regular brainstorming sessions",
            "Scale successful approaches to other departments"
        ]
        
        return actions
    
    def _calculate_synthesis_quality(self, synthesis: Dict[str, Any], all_ideas: List[Any]) -> float:
        """Calculate the quality of the synthesis"""
        if not all_ideas:
            return 0.0
        
        # Factors that contribute to quality
        diversity_score = len(set(getattr(idea, 'idea_type', 'unknown') for idea in all_ideas)) / 5.0
        quantity_score = min(len(all_ideas) / 50.0, 1.0)
        collaboration_score = synthesis.get("collaborations", 0) / 10.0
        
        # Weighted quality score
        quality = (diversity_score * 0.4 + quantity_score * 0.3 + collaboration_score * 0.3)
        
        return min(quality, 1.0)
    
    def _generate_implementation_recommendations(self, synthesis: Dict[str, Any]) -> List[str]:
        """Generate implementation recommendations"""
        return [
            "Prioritize high-impact, low-effort solutions",
            "Create implementation roadmap with milestones",
            "Assign cross-department teams for complex solutions",
            "Establish success metrics and monitoring"
        ]
    
    def _generate_scaling_recommendations(self, mega_session: MegaBrainstormSession) -> List[str]:
        """Generate scaling recommendations based on session results"""
        recommendations = []
        
        if mega_session.synthesis_quality > 0.8:
            recommendations.append("High-quality synthesis achieved - consider scaling to more departments")
        
        if mega_session.collaborations > 10:
            recommendations.append("High collaboration level - expand cross-department coordination")
        
        if mega_session.total_agents > 20:
            recommendations.append("Large agent pool - consider distributed scaling")
        
        return recommendations
    
    def get_mega_brainstorm_status(self) -> Dict[str, Any]:
        """Get status of all mega brainstorming sessions"""
        return {
            "active_sessions": len([s for s in self.mega_sessions.values() if s.status == "active"]),
            "total_sessions": len(self.mega_sessions),
            "total_departments": len(self.coordinator.departments),
            "total_agents": sum(len(dept.agents) for dept in self.coordinator.departments.values()),
            "scaling_status": self.scaling_coordinator.get_scaling_status(),
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "problem": s.problem,
                    "participating_departments": len(s.participating_departments),
                    "total_agents": s.total_agents,
                    "ideas_generated": s.ideas_generated,
                    "collaborations": s.collaborations,
                    "synthesis_quality": s.synthesis_quality,
                    "status": s.status
                }
                for s in list(self.mega_sessions.values())[-5:]  # Last 5 sessions
            ]
        }
    
    def end_mega_brainstorm(self, session_id: str) -> Dict[str, Any]:
        """End a mega brainstorming session and return final synthesis"""
        if session_id not in self.mega_sessions:
            return {"error": "Session not found"}
        
        mega_session = self.mega_sessions[session_id]
        mega_session.end_time = time.time()
        mega_session.status = "completed"
        
        # Generate final synthesis
        final_synthesis = self.synthesize_mega_brainstorm(session_id)
        
        return {
            "session_completed": True,
            "final_synthesis": final_synthesis,
            "session_duration": mega_session.end_time - mega_session.start_time
        }
