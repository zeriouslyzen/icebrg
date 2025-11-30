"""
ICEBURG Scaling Coordinator

Coordinates the scaling of think tank departments and brainstorming capabilities
across multiple instances and distributed systems.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScalingStrategy(Enum):
    HORIZONTAL = "horizontal"  # Add more agents
    VERTICAL = "vertical"  # Improve agent capabilities
    DISTRIBUTED = "distributed"  # Spread across multiple systems
    HYBRID = "hybrid"  # Combination of strategies

class LoadLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScalingDecision:
    decision_id: str
    strategy: ScalingStrategy
    scale_factor: int
    target_departments: List[str]
    reasoning: str
    expected_improvement: float
    implementation_plan: List[str]
    created_at: float

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    queue_length: int
    response_time: float
    error_rate: float
    throughput: float

class ScalingCoordinator:
    """
    Coordinates the scaling of ICEBURG think tank departments and brainstorming capabilities.
    Enables autonomous scaling decisions and distributed intelligence coordination.
    """
    
    def __init__(self):
        self.departments: Dict[str, Any] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.system_metrics: SystemMetrics = SystemMetrics(0, 0, 0, 0, 0, 0, 0)
        self.scaling_thresholds: Dict[str, float] = {
            "cpu_threshold": 0.8,
            "memory_threshold": 0.85,
            "response_time_threshold": 5.0,
            "error_rate_threshold": 0.1
        }
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring_active = False
        
    def register_department(self, department_name: str, department_instance: Any) -> bool:
        """Register a department for scaling coordination"""
        self.departments[department_name] = {
            "instance": department_instance,
            "metrics": SystemMetrics(0, 0, 0, 0, 0, 0, 0),
            "scaling_history": [],
            "last_scaled": 0
        }
        return True
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring of system metrics and scaling needs"""
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for scaling decisions"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._analyze_scaling_needs()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                time.sleep(5)
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics"""
        # This would integrate with actual system monitoring
        # For now, simulate metrics collection
        import psutil
        
        self.system_metrics.cpu_usage = psutil.cpu_percent() / 100.0
        self.system_metrics.memory_usage = psutil.virtual_memory().percent / 100.0
        self.system_metrics.active_tasks = sum(
            len(dept["instance"].tasks) if hasattr(dept["instance"], 'tasks') else 0
            for dept in self.departments.values()
        )
        self.system_metrics.queue_length = sum(
            len([t for t in dept["instance"].tasks.values() if t.status == "pending"])
            if hasattr(dept["instance"], 'tasks') else 0
            for dept in self.departments.values()
        )
    
    def _analyze_scaling_needs(self) -> None:
        """Analyze current system state and determine if scaling is needed"""
        load_level = self._determine_load_level()
        
        if load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL]:
            scaling_decision = self._make_scaling_decision(load_level)
            if scaling_decision:
                self._implement_scaling(scaling_decision)
    
    def _determine_load_level(self) -> LoadLevel:
        """Determine current system load level"""
        metrics = self.system_metrics
        
        # Check multiple indicators
        high_load_indicators = 0
        
        if metrics.cpu_usage > self.scaling_thresholds["cpu_threshold"]:
            high_load_indicators += 1
        
        if metrics.memory_usage > self.scaling_thresholds["memory_threshold"]:
            high_load_indicators += 1
        
        if metrics.response_time > self.scaling_thresholds["response_time_threshold"]:
            high_load_indicators += 1
        
        if metrics.error_rate > self.scaling_thresholds["error_rate_threshold"]:
            high_load_indicators += 1
        
        # Determine load level
        if high_load_indicators >= 3:
            return LoadLevel.CRITICAL
        elif high_load_indicators >= 2:
            return LoadLevel.HIGH
        elif high_load_indicators >= 1:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LOW
    
    def _make_scaling_decision(self, load_level: LoadLevel) -> Optional[ScalingDecision]:
        """Make an intelligent scaling decision based on current load"""
        if load_level == LoadLevel.LOW:
            return None
        
        # Analyze which departments need scaling
        departments_needing_scaling = self._identify_departments_for_scaling()
        
        if not departments_needing_scaling:
            return None
        
        # Determine scaling strategy
        strategy = self._select_scaling_strategy(load_level, departments_needing_scaling)
        
        # Calculate scale factor
        scale_factor = self._calculate_scale_factor(load_level, strategy)
        
        # Create scaling decision
        decision = ScalingDecision(
            decision_id=f"scale_{uuid.uuid4().hex[:8]}",
            strategy=strategy,
            scale_factor=scale_factor,
            target_departments=departments_needing_scaling,
            reasoning=self._generate_scaling_reasoning(load_level, strategy, scale_factor),
            expected_improvement=self._calculate_expected_improvement(scale_factor),
            implementation_plan=self._create_implementation_plan(strategy, departments_needing_scaling),
            created_at=time.time()
        )
        
        self.scaling_history.append(decision)
        return decision
    
    def _identify_departments_for_scaling(self) -> List[str]:
        """Identify which departments need scaling"""
        departments_to_scale = []
        
        for dept_name, dept_info in self.departments.items():
            dept_instance = dept_info["instance"]
            
            # Check department-specific metrics
            if hasattr(dept_instance, 'tasks'):
                pending_tasks = len([t for t in dept_instance.tasks.values() if t.status == "pending"])
                active_tasks = len([t for t in dept_instance.tasks.values() if t.status == "assigned"])
                
                # Scale if there are many pending tasks or high active task ratio
                if pending_tasks > 5 or (active_tasks > 0 and pending_tasks / active_tasks > 2):
                    departments_to_scale.append(dept_name)
        
        return departments_to_scale
    
    def _select_scaling_strategy(self, load_level: LoadLevel, departments: List[str]) -> ScalingStrategy:
        """Select appropriate scaling strategy"""
        if load_level == LoadLevel.CRITICAL:
            return ScalingStrategy.HYBRID
        elif len(departments) > 3:
            return ScalingStrategy.DISTRIBUTED
        elif load_level == LoadLevel.HIGH:
            return ScalingStrategy.HORIZONTAL
        else:
            return ScalingStrategy.VERTICAL
    
    def _calculate_scale_factor(self, load_level: LoadLevel, strategy: ScalingStrategy) -> int:
        """Calculate how much to scale"""
        base_scale = {
            LoadLevel.MEDIUM: 2,
            LoadLevel.HIGH: 3,
            LoadLevel.CRITICAL: 5
        }.get(load_level, 1)
        
        strategy_multiplier = {
            ScalingStrategy.HORIZONTAL: 1.0,
            ScalingStrategy.VERTICAL: 0.5,
            ScalingStrategy.DISTRIBUTED: 1.5,
            ScalingStrategy.HYBRID: 2.0
        }.get(strategy, 1.0)
        
        return int(base_scale * strategy_multiplier)
    
    def _generate_scaling_reasoning(self, load_level: LoadLevel, strategy: ScalingStrategy, scale_factor: int) -> str:
        """Generate human-readable reasoning for scaling decision"""
        reasoning = f"Scaling decision made due to {load_level.value} load level. "
        reasoning += f"Selected {strategy.value} strategy with scale factor {scale_factor}. "
        
        if load_level == LoadLevel.CRITICAL:
            reasoning += "Critical load detected, implementing emergency scaling."
        elif load_level == LoadLevel.HIGH:
            reasoning += "High load detected, proactive scaling to prevent bottlenecks."
        else:
            reasoning += "Medium load detected, preventive scaling for optimal performance."
        
        return reasoning
    
    def _calculate_expected_improvement(self, scale_factor: int) -> float:
        """Calculate expected performance improvement from scaling"""
        # Simple linear improvement model
        return min(scale_factor * 0.3, 2.0)  # Cap at 200% improvement
    
    def _create_implementation_plan(self, strategy: ScalingStrategy, departments: List[str]) -> List[str]:
        """Create detailed implementation plan for scaling"""
        plan = []
        
        if strategy == ScalingStrategy.HORIZONTAL:
            plan.extend([
                f"Add {len(departments)} new agents to each target department",
                "Distribute workload across new agents",
                "Update load balancing configuration",
                "Monitor performance improvements"
            ])
        elif strategy == ScalingStrategy.VERTICAL:
            plan.extend([
                "Upgrade existing agent capabilities",
                "Implement advanced reasoning algorithms",
                "Optimize agent performance",
                "Update collaboration protocols"
            ])
        elif strategy == ScalingStrategy.DISTRIBUTED:
            plan.extend([
                "Deploy new department instances",
                "Configure distributed coordination",
                "Implement cross-instance communication",
                "Set up load balancing across instances"
            ])
        elif strategy == ScalingStrategy.HYBRID:
            plan.extend([
                "Combine horizontal and vertical scaling",
                "Deploy new agents with enhanced capabilities",
                "Implement distributed coordination",
                "Optimize overall system architecture"
            ])
        
        return plan
    
    def _implement_scaling(self, decision: ScalingDecision) -> bool:
        """Implement the scaling decision"""
        try:
            
            for dept_name in decision.target_departments:
                if dept_name in self.departments:
                    dept_info = self.departments[dept_name]
                    dept_instance = dept_info["instance"]
                    
                    # Apply scaling based on strategy
                    if decision.strategy == ScalingStrategy.HORIZONTAL:
                        self._scale_horizontally(dept_instance, decision.scale_factor)
                    elif decision.strategy == ScalingStrategy.VERTICAL:
                        self._scale_vertically(dept_instance, decision.scale_factor)
                    elif decision.strategy == ScalingStrategy.DISTRIBUTED:
                        self._scale_distributed(dept_name, decision.scale_factor)
                    elif decision.strategy == ScalingStrategy.HYBRID:
                        self._scale_hybrid(dept_instance, decision.scale_factor)
            
            # Update scaling history
            for dept_name in decision.target_departments:
                if dept_name in self.departments:
                    self.departments[dept_name]["last_scaled"] = time.time()
                    self.departments[dept_name]["scaling_history"].append(decision.decision_id)
            
            return True
            
        except Exception as e:
            return False
    
    def _scale_horizontally(self, department_instance: Any, scale_factor: int) -> None:
        """Scale department horizontally by adding more agents"""
        if hasattr(department_instance, 'scale_department'):
            department_instance.scale_department(scale_factor)
        else:
            # Fallback: add agents manually
            for i in range(scale_factor):
                agent_id = f"scaled_agent_{uuid.uuid4().hex[:8]}"
                if hasattr(department_instance, 'add_agent'):
                    department_instance.add_agent(
                        agent_id, 
                        f"scaled_specialist_{i}", 
                        ["scaling", "performance", "collaboration"]
                    )
    
    def _scale_vertically(self, department_instance: Any, scale_factor: int) -> None:
        """Scale department vertically by improving agent capabilities"""
        if hasattr(department_instance, 'enhance_agents'):
            department_instance.enhance_agents(scale_factor)
        else:
            # Fallback: improve existing agents
            if hasattr(department_instance, 'agents'):
                for agent in department_instance.agents.values():
                    agent.performance_score = min(agent.performance_score * 1.2, 2.0)
                    agent.max_concurrent_tasks = min(agent.max_concurrent_tasks + 1, 10)
    
    def _scale_distributed(self, department_name: str, scale_factor: int) -> None:
        """Scale department using distributed approach"""
        # Create new department instances
        for i in range(scale_factor):
            new_dept_name = f"{department_name}_distributed_{i}"
            # This would create new department instances
    
    def _scale_hybrid(self, department_instance: Any, scale_factor: int) -> None:
        """Scale department using hybrid approach"""
        # Combine horizontal and vertical scaling
        horizontal_factor = scale_factor // 2
        vertical_factor = scale_factor - horizontal_factor
        
        self._scale_horizontally(department_instance, horizontal_factor)
        self._scale_vertically(department_instance, vertical_factor)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics"""
        return {
            "total_departments": len(self.departments),
            "system_metrics": {
                "cpu_usage": self.system_metrics.cpu_usage,
                "memory_usage": self.system_metrics.memory_usage,
                "active_tasks": self.system_metrics.active_tasks,
                "queue_length": self.system_metrics.queue_length,
                "response_time": self.system_metrics.response_time,
                "error_rate": self.system_metrics.error_rate
            },
            "scaling_history": len(self.scaling_history),
            "recent_scaling_decisions": [
                {
                    "decision_id": d.decision_id,
                    "strategy": d.strategy.value,
                    "scale_factor": d.scale_factor,
                    "target_departments": d.target_departments,
                    "reasoning": d.reasoning,
                    "created_at": d.created_at
                }
                for d in self.scaling_history[-5:]  # Last 5 decisions
            ],
            "monitoring_active": self.monitoring_active
        }
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)
    
    def force_scale_department(self, department_name: str, scale_factor: int, 
                             strategy: ScalingStrategy = ScalingStrategy.HORIZONTAL) -> bool:
        """Force scaling of a specific department"""
        if department_name not in self.departments:
            return False
        
        decision = ScalingDecision(
            decision_id=f"forced_scale_{uuid.uuid4().hex[:8]}",
            strategy=strategy,
            scale_factor=scale_factor,
            target_departments=[department_name],
            reasoning=f"Forced scaling of {department_name}",
            expected_improvement=self._calculate_expected_improvement(scale_factor),
            implementation_plan=[f"Force scale {department_name} by {scale_factor}"],
            created_at=time.time()
        )
        
        return self._implement_scaling(decision)
