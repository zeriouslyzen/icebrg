"""
Teacher-Student Tuning for ICEBURG - October 2025
===============================================

Implements adaptive prompt refinement and lifelong learning
based on performance feedback across ICEBURG agents.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

@dataclass
class PerformanceFeedback:
    """Performance feedback for agent tuning"""
    agent_id: str
    task_type: str
    success: bool
    execution_time: float
    result_quality: float  # 0-1 score
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PromptEvolution:
    """Track prompt evolution over time"""
    agent_id: str
    version: int
    base_prompt: str
    evolved_prompt: str
    performance_improvement: float
    evolution_timestamp: datetime

class TeacherStudentTuning:
    """
    Implements teacher-student tuning for adaptive prompt refinement
    and lifelong learning across ICEBURG agents.
    """

    def __init__(self):
        self.performance_history: List[PerformanceFeedback] = []
        self.prompt_evolutions: List[PromptEvolution] = []
        self.agent_prompts: Dict[str, str] = {}
        self.feedback_threshold = 0.7  # Minimum performance for prompt evolution
        self.max_evolution_attempts = 3

    def record_performance_feedback(self, agent_id: str, task_type: str,
        success: bool, execution_time: float,
                                  result_quality: float = 1.0,
                                  error_message: Optional[str] = None):
        """Record performance feedback for an agent"""
        feedback = PerformanceFeedback(
            agent_id=agent_id,
            task_type=task_type,
            success=success,
            execution_time=execution_time,
            result_quality=result_quality,
            error_message=error_message
        )

        self.performance_history.append(feedback)

    def analyze_agent_performance(self, agent_id: str, window_size: int = 10) -> Dict[str, float]:
        """Analyze recent performance for an agent"""
        # Get recent feedback for this agent
        recent_feedback = [
            f for f in self.performance_history[-window_size:]
            if f.agent_id == agent_id
        ]

        if not recent_feedback:
            return {"success_rate": 0.0, "avg_quality": 0.0, "avg_time": 0.0}

        success_count = sum(1 for f in recent_feedback if f.success)
        success_rate = success_count / len(recent_feedback)

        avg_quality = sum(f.result_quality for f in recent_feedback) / len(recent_feedback)
        avg_time = sum(f.execution_time for f in recent_feedback) / len(recent_feedback)

        return {
            "success_rate": success_rate,
            "avg_quality": avg_quality,
            "avg_time": avg_time,
            "sample_size": len(recent_feedback)
        }

    def should_evolve_prompt(self, agent_id: str) -> bool:
        """Determine if an agent's prompt should be evolved"""
        performance = self.analyze_agent_performance(agent_id)

        # Evolution criteria
        needs_evolution = (
            performance["success_rate"] < self.feedback_threshold or
            performance["avg_quality"] < self.feedback_threshold or
            performance["sample_size"] >= 5  # Enough data for evolution
        )

        if needs_evolution:
            # Evolution needed based on performance criteria
            pass

        return needs_evolution

    async def evolve_agent_prompt(self, agent_id: str, current_prompt: str,
        performance_data: Dict[str, Any]) -> str:
        """Evolve an agent's prompt based on performance feedback"""

        try:
            # Analyze failure patterns
            failure_analysis = self._analyze_failure_patterns(agent_id)

            # Generate improved prompt based on performance data
            evolved_prompt = await self._generate_improved_prompt(
                current_prompt,
                performance_data,
                failure_analysis
            )

            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                performance_data,
                failure_analysis
            )

            # Record the evolution
            evolution = PromptEvolution(
                agent_id=agent_id,
                version=len([e for e in self.prompt_evolutions if e.agent_id == agent_id]) + 1,
                base_prompt=current_prompt,
                evolved_prompt=evolved_prompt,
                performance_improvement=expected_improvement,
                evolution_timestamp=datetime.now()
            )

            self.prompt_evolutions.append(evolution)


            return evolved_prompt

        except Exception as e:
            return current_prompt  # Return original prompt on failure

    def _analyze_failure_patterns(self, agent_id: str) -> Dict[str, Any]:
        """Analyze patterns in agent failures"""
        agent_feedback = [f for f in self.performance_history if f.agent_id == agent_id]

        if not agent_feedback:
            return {"common_errors": [], "slow_tasks": [], "quality_issues": []}

        # Find common error patterns
        error_messages = [f.error_message for f in agent_feedback if f.error_message]
        common_errors = self._find_common_patterns(error_messages)

        # Find slow tasks
        slow_threshold = sum(f.execution_time for f in agent_feedback) / len(agent_feedback) * 1.5
        slow_tasks = [f.task_type for f in agent_feedback if f.execution_time > slow_threshold]

        # Find quality issues
        quality_issues = [f.task_type for f in agent_feedback if f.result_quality < 0.7]

        return {
            "common_errors": common_errors,
            "slow_tasks": slow_tasks,
            "quality_issues": quality_issues,
            "total_failures": len([f for f in agent_feedback if not f.success])
        }

    def _find_common_patterns(self, error_messages: List[str]) -> List[str]:
        """Find common patterns in error messages"""
        if not error_messages:
            return []

        # Simple pattern matching (in production, use more sophisticated NLP)
        patterns = defaultdict(int)

        common_words = ["timeout", "memory", "parsing", "connection", "validation", "format"]

        for error in error_messages:
            error_lower = error.lower() if error else ""
            for word in common_words:
                if word in error_lower:
                    patterns[word] += 1

        # Return patterns that appear in > 20% of errors
        threshold = len(error_messages) * 0.2
        return [pattern for pattern, count in patterns.items() if count >= threshold]

    async def _generate_improved_prompt(self, current_prompt: str,
        performance_data: Dict[str, Any],
                                       failure_analysis: Dict[str, Any]) -> str:
        """Generate an improved prompt based on performance analysis"""

        # Base improvements
        improvements = []

        # Add performance-based instructions
        if performance_data["success_rate"] < 0.7:
            improvements.append("Focus on accuracy and completeness in your responses.")

        if performance_data["avg_quality"] < 0.7:
            improvements.append("Pay special attention to response quality and detail level.")

        if performance_data["avg_time"] > 10.0:  # If taking more than 10 seconds
            improvements.append("Optimize for faster response times while maintaining quality.")

        # Add failure-pattern-based instructions
        if failure_analysis["common_errors"]:
            error_focus = ", ".join(failure_analysis["common_errors"])
            improvements.append(f"Avoid common errors like: {error_focus}")

        if failure_analysis["slow_tasks"]:
            slow_focus = ", ".join(set(failure_analysis["slow_tasks"]))
            improvements.append(f"Optimize performance for these task types: {slow_focus}")

        if failure_analysis["quality_issues"]:
            quality_focus = ", ".join(set(failure_analysis["quality_issues"]))
            improvements.append(f"Improve quality for these task types: {quality_focus}")

        # Combine current prompt with improvements
        if improvements:
            enhanced_prompt = f"{current_prompt}\n\nPerformance Optimizations:\n" + "\n".join(f"• {imp}" for imp in improvements)
        else:
            enhanced_prompt = current_prompt

        return enhanced_prompt

    def _calculate_expected_improvement(self, performance_data: Dict[str, Any],
        failure_analysis: Dict[str, Any]) -> float:
        """Calculate expected improvement from prompt evolution"""
        # Simple heuristic-based calculation
        improvement_score = 0.0

        # Base improvement from success rate
        if performance_data["success_rate"] < 0.7:
            improvement_score += 0.15  # 15% improvement expected

        # Quality improvement
        if performance_data["avg_quality"] < 0.7:
            improvement_score += 0.20  # 20% improvement expected

        # Error reduction
        if failure_analysis["common_errors"]:
            improvement_score += 0.10  # 10% improvement from error reduction

        # Performance optimization
        if performance_data["avg_time"] > 10.0:
            improvement_score += 0.05  # 5% improvement from optimization

        return min(improvement_score, 0.5)  # Cap at 50% expected improvement

    def get_tuning_recommendations(self) -> Dict[str, List[str]]:
        """Get tuning recommendations for all agents"""
        recommendations = {}

        # Check each agent's performance
        agent_ids = set(f.agent_id for f in self.performance_history)

        for agent_id in agent_ids:
            performance = self.analyze_agent_performance(agent_id)

            if performance["sample_size"] < 3:
                continue  # Not enough data

            recs = []

            if performance["success_rate"] < self.feedback_threshold:
                recs.append("Low success rate - consider prompt evolution")

            if performance["avg_quality"] < self.feedback_threshold:
                recs.append("Quality issues - enhance prompt specificity")

            if performance["avg_time"] > 15.0:
                recs.append("Performance issues - optimize for speed")

            if recs:
                recommendations[agent_id] = recs

        return recommendations

    def get_evolution_history(self, agent_id: str) -> List[PromptEvolution]:
        """Get evolution history for a specific agent"""
        return [e for e in self.prompt_evolutions if e.agent_id == agent_id]

    def get_tuning_metrics(self) -> Dict[str, Any]:
        """Get overall tuning system metrics"""
        if not self.performance_history:
            return {"message": "No performance data available"}

        # Calculate overall metrics
        total_feedback = len(self.performance_history)
        successful_feedback = len([f for f in self.performance_history if f.success])
        success_rate = successful_feedback / total_feedback if total_feedback > 0 else 0.0

        avg_quality = sum(f.result_quality for f in self.performance_history) / total_feedback if total_feedback > 0 else 0.0
        avg_time = sum(f.execution_time for f in self.performance_history) / total_feedback if total_feedback > 0 else 0.0

        # Evolution metrics
        total_evolutions = len(self.prompt_evolutions)
        avg_improvement = sum(e.performance_improvement for e in self.prompt_evolutions) / total_evolutions if total_evolutions > 0 else 0.0

        return {
            "total_feedback_records": total_feedback,
            "overall_success_rate": success_rate,
            "overall_avg_quality": avg_quality,
            "overall_avg_execution_time": avg_time,
            "total_prompt_evolutions": total_evolutions,
            "average_improvement_per_evolution": avg_improvement,
            "agents_with_feedback": len(set(f.agent_id for f in self.performance_history)),
            "tuning_system_status": "active" if total_evolutions > 0 else "initializing"
        }

# Global teacher-student tuning instance
_teacher_student_tuning: Optional[TeacherStudentTuning] = None

async def get_teacher_student_tuning() -> TeacherStudentTuning:
    """Get or create the global teacher-student tuning instance"""
    global _teacher_student_tuning
    if _teacher_student_tuning is None:
        _teacher_student_tuning = TeacherStudentTuning()
    return _teacher_student_tuning

async def record_agent_feedback(agent_id: str, task_type: str, success: bool,
    execution_time: float, result_quality: float = 1.0,
                               error_message: Optional[str] = None):
    """Record performance feedback for adaptive tuning"""
    tuning = await get_teacher_student_tuning()
    tuning.record_performance_feedback(agent_id, task_type, success, execution_time, result_quality, error_message)

async def get_tuning_recommendations() -> Dict[str, List[str]]:
    """Get tuning recommendations for all agents"""
    tuning = await get_teacher_student_tuning()
    return tuning.get_tuning_recommendations()

async def evolve_agent_prompt_if_needed(agent_id: str, current_prompt: str) -> str:
    """Evolve agent prompt if performance indicates need"""
    tuning = await get_teacher_student_tuning()

    if tuning.should_evolve_prompt(agent_id):
        performance = tuning.analyze_agent_performance(agent_id)
        return await tuning.evolve_agent_prompt(agent_id, current_prompt, performance)

    return current_prompt
