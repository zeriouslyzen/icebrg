"""
Multi-Agent Coordination Efficiency Optimizer
Enhances coordination between ICEBURG agents for optimal performance

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
from enum import Enum

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

class CoordinationPattern(Enum):
    """Types of coordination patterns"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"

class AgentRole(Enum):
    """Agent roles in coordination"""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    MONITOR = "monitor"

@dataclass
class AgentPerformance:
    """Performance metrics for an agent"""
    agent_name: str
    execution_time: float
    success_rate: float
    quality_score: float
    resource_usage: Dict[str, float]
    coordination_overhead: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class CoordinationStrategy:
    """Strategy for agent coordination"""
    strategy_id: str
    pattern: CoordinationPattern
    agent_roles: Dict[str, AgentRole]
    communication_protocol: str
    load_balancing: bool
    fault_tolerance: bool
    expected_efficiency: float
    implementation_complexity: str

@dataclass
class CoordinationSession:
    """Record of a coordination session"""
    session_id: str
    agents_involved: List[str]
    coordination_pattern: CoordinationPattern
    total_execution_time: float
    success: bool
    quality_score: float
    resource_efficiency: float
    coordination_overhead: float
    timestamp: float = field(default_factory=time.time)

class MultiAgentCoordinator:
    """
    Optimizes multi-agent coordination for efficiency and performance
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/multi_agent_coordination")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.performance_file = self.data_dir / "agent_performance.json"
        self.strategies_file = self.data_dir / "coordination_strategies.json"
        self.sessions_file = self.data_dir / "coordination_sessions.json"
        
        # Data structures
        self.agent_performance: Dict[str, List[AgentPerformance]] = {}
        self.coordination_strategies: Dict[str, CoordinationStrategy] = {}
        self.coordination_sessions: List[CoordinationSession] = []
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.coordination_patterns: Dict[CoordinationPattern, List[float]] = defaultdict(list)
        
        # Load existing data
        self._load_data()
        self._initialize_default_strategies()
        
        logger.info("🤝 Multi-Agent Coordinator initialized")
    
    def record_agent_performance(
        self,
        agent_name: str,
        execution_time: float,
        success: bool,
        quality_score: float,
        resource_usage: Dict[str, float],
        coordination_overhead: float = 0.0
    ) -> None:
        """Record performance metrics for an agent"""
        
        # Calculate success rate
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = []
        
        recent_performances = self.agent_performance[agent_name][-10:]  # Last 10 performances
        success_rate = np.mean([p.success_rate for p in recent_performances]) if recent_performances else 0.0
        
        # Update success rate with new data
        success_rate = (success_rate * len(recent_performances) + (1.0 if success else 0.0)) / (len(recent_performances) + 1)
        
        performance = AgentPerformance(
            agent_name=agent_name,
            execution_time=execution_time,
            success_rate=success_rate,
            quality_score=quality_score,
            resource_usage=resource_usage,
            coordination_overhead=coordination_overhead
        )
        
        self.agent_performance[agent_name].append(performance)
        
        # Keep only last 100 performances per agent
        if len(self.agent_performance[agent_name]) > 100:
            self.agent_performance[agent_name] = self.agent_performance[agent_name][-100:]
        
        # Update performance history
        self.performance_history["execution_time"].append(execution_time)
        self.performance_history["quality_score"].append(quality_score)
        self.performance_history["coordination_overhead"].append(coordination_overhead)
        
        # Analyze for coordination optimization
        self._analyze_coordination_patterns()
        
        # Save data
        self._save_data()
        
        logger.info(f"📊 Recorded performance for {agent_name}: {execution_time:.2f}s, success: {success}, quality: {quality_score:.2f}")
    
    def record_coordination_session(
        self,
        agents_involved: List[str],
        coordination_pattern: CoordinationPattern,
        total_execution_time: float,
        success: bool,
        quality_score: float,
        resource_efficiency: float,
        coordination_overhead: float
    ) -> str:
        """Record a coordination session for analysis"""
        
        session_id = f"session_{int(time.time())}_{len(agents_involved)}_agents"
        
        session = CoordinationSession(
            session_id=session_id,
            agents_involved=agents_involved,
            coordination_pattern=coordination_pattern,
            total_execution_time=total_execution_time,
            success=success,
            quality_score=quality_score,
            resource_efficiency=resource_efficiency,
            coordination_overhead=coordination_overhead
        )
        
        self.coordination_sessions.append(session)
        
        # Keep only last 500 sessions
        if len(self.coordination_sessions) > 500:
            self.coordination_sessions = self.coordination_sessions[-500:]
        
        # Update coordination pattern performance
        self.coordination_patterns[coordination_pattern].append(quality_score)
        
        # Analyze session for optimization opportunities
        self._analyze_session_performance(session)
        
        # Save data
        self._save_data()
        
        logger.info(f"🤝 Recorded coordination session: {coordination_pattern.value} with {len(agents_involved)} agents")
        
        return session_id
    
    def _analyze_coordination_patterns(self) -> None:
        """Analyze coordination patterns to identify optimization opportunities"""
        
        if len(self.coordination_sessions) < 10:
            return
        
        # Analyze performance by coordination pattern
        for pattern in CoordinationPattern:
            pattern_sessions = [s for s in self.coordination_sessions if s.coordination_pattern == pattern]
            
            if len(pattern_sessions) < 3:
                continue
            
            # Calculate pattern performance metrics
            avg_execution_time = np.mean([s.total_execution_time for s in pattern_sessions])
            avg_quality = np.mean([s.quality_score for s in pattern_sessions])
            avg_efficiency = np.mean([s.resource_efficiency for s in pattern_sessions])
            avg_overhead = np.mean([s.coordination_overhead for s in pattern_sessions])
            
            # Identify optimization opportunities
            self._identify_coordination_optimizations(pattern, avg_execution_time, avg_quality, avg_efficiency, avg_overhead)
    
    def _identify_coordination_optimizations(
        self,
        pattern: CoordinationPattern,
        avg_execution_time: float,
        avg_quality: float,
        avg_efficiency: float,
        avg_overhead: float
    ) -> None:
        """Identify optimization opportunities for coordination patterns"""
        
        strategy_id = f"optimization_{pattern.value}"
        
        # High overhead optimization
        if avg_overhead > 0.3:  # More than 30% overhead
            if strategy_id not in self.coordination_strategies:
                self.coordination_strategies[strategy_id] = CoordinationStrategy(
                    strategy_id=strategy_id,
                    pattern=pattern,
                    agent_roles=self._get_optimized_roles(pattern),
                    communication_protocol="optimized",
                    load_balancing=True,
                    fault_tolerance=True,
                    expected_efficiency=0.25,  # 25% efficiency improvement
                    implementation_complexity="medium"
                )
        
        # Low quality optimization
        elif avg_quality < 0.7:
            if strategy_id not in self.coordination_strategies:
                self.coordination_strategies[strategy_id] = CoordinationStrategy(
                    strategy_id=strategy_id,
                    pattern=pattern,
                    agent_roles=self._get_optimized_roles(pattern),
                    communication_protocol="enhanced",
                    load_balancing=False,
                    fault_tolerance=True,
                    expected_efficiency=0.2,  # 20% quality improvement
                    implementation_complexity="high"
                )
        
        # Slow execution optimization
        elif avg_execution_time > 5.0:  # More than 5 seconds
            if strategy_id not in self.coordination_strategies:
                self.coordination_strategies[strategy_id] = CoordinationStrategy(
                    strategy_id=strategy_id,
                    pattern=pattern,
                    agent_roles=self._get_optimized_roles(pattern),
                    communication_protocol="streamlined",
                    load_balancing=True,
                    fault_tolerance=False,
                    expected_efficiency=0.3,  # 30% speed improvement
                    implementation_complexity="low"
                )
    
    def _get_optimized_roles(self, pattern: CoordinationPattern) -> Dict[str, AgentRole]:
        """Get optimized agent roles for coordination pattern"""
        
        role_mappings = {
            CoordinationPattern.SEQUENTIAL: {
                "surveyor": AgentRole.COORDINATOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.VALIDATOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            },
            CoordinationPattern.PARALLEL: {
                "surveyor": AgentRole.COORDINATOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.MONITOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            },
            CoordinationPattern.PIPELINE: {
                "surveyor": AgentRole.COORDINATOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.VALIDATOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            },
            CoordinationPattern.HIERARCHICAL: {
                "surveyor": AgentRole.COORDINATOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.COORDINATOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            },
            CoordinationPattern.COLLABORATIVE: {
                "surveyor": AgentRole.EXECUTOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.MONITOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            },
            CoordinationPattern.COMPETITIVE: {
                "surveyor": AgentRole.COORDINATOR,
                "dissident": AgentRole.EXECUTOR,
                "archaeologist": AgentRole.EXECUTOR,
                "supervisor": AgentRole.VALIDATOR,
                "synthesist": AgentRole.SYNTHESIZER,
                "oracle": AgentRole.VALIDATOR
            }
        }
        
        return role_mappings.get(pattern, role_mappings[CoordinationPattern.SEQUENTIAL])
    
    def _analyze_session_performance(self, session: CoordinationSession) -> None:
        """Analyze individual session performance for insights"""
        
        # Identify bottlenecks
        if session.coordination_overhead > 0.4:
            logger.warning(f"High coordination overhead in session {session.session_id}: {session.coordination_overhead:.2f}")
        
        # Identify successful patterns
        if session.success and session.quality_score > 0.8 and session.coordination_overhead < 0.2:
            logger.info(f"Optimal coordination session {session.session_id}: {session.coordination_pattern.value}")
    
    def get_optimal_coordination_strategy(
        self,
        agents_required: List[str],
        complexity_level: str = "medium",
        priority: str = "balanced"  # "speed", "quality", "efficiency", "balanced"
    ) -> CoordinationStrategy:
        """Get optimal coordination strategy for given requirements"""
        
        # Filter strategies by agent compatibility
        compatible_strategies = []
        for strategy in self.coordination_strategies.values():
            if all(agent in strategy.agent_roles for agent in agents_required):
                compatible_strategies.append(strategy)
        
        if not compatible_strategies:
            # Return default strategy
            return self._get_default_strategy(agents_required, complexity_level)
        
        # Score strategies based on priority
        scored_strategies = []
        for strategy in compatible_strategies:
            score = self._score_strategy(strategy, complexity_level, priority)
            scored_strategies.append((strategy, score))
        
        # Return highest scoring strategy
        best_strategy, _ = max(scored_strategies, key=lambda x: x[1])
        
        logger.info(f"🎯 Selected optimal strategy: {best_strategy.strategy_id} for {len(agents_required)} agents")
        
        return best_strategy
    
    def _score_strategy(
        self,
        strategy: CoordinationStrategy,
        complexity_level: str,
        priority: str
    ) -> float:
        """Score a coordination strategy based on requirements"""
        
        base_score = strategy.expected_efficiency
        
        # Adjust for complexity level
        complexity_adjustments = {
            "low": 0.1,
            "medium": 0.0,
            "high": -0.1
        }
        base_score += complexity_adjustments.get(complexity_level, 0.0)
        
        # Adjust for priority
        priority_adjustments = {
            "speed": 0.2 if strategy.pattern == CoordinationPattern.PARALLEL else 0.0,
            "quality": 0.2 if strategy.pattern == CoordinationPattern.HIERARCHICAL else 0.0,
            "efficiency": 0.2 if strategy.load_balancing else 0.0,
            "balanced": 0.0
        }
        base_score += priority_adjustments.get(priority, 0.0)
        
        # Adjust for implementation complexity
        complexity_penalties = {
            "low": 0.0,
            "medium": -0.05,
            "high": -0.1
        }
        base_score += complexity_penalties.get(strategy.implementation_complexity, 0.0)
        
        return base_score
    
    def _get_default_strategy(
        self,
        agents_required: List[str],
        complexity_level: str
    ) -> CoordinationStrategy:
        """Get default coordination strategy"""
        
        # Choose pattern based on number of agents
        if len(agents_required) <= 2:
            pattern = CoordinationPattern.SEQUENTIAL
        elif len(agents_required) <= 4:
            pattern = CoordinationPattern.PARALLEL
        else:
            pattern = CoordinationPattern.HIERARCHICAL
        
        return CoordinationStrategy(
            strategy_id=f"default_{pattern.value}",
            pattern=pattern,
            agent_roles=self._get_optimized_roles(pattern),
            communication_protocol="standard",
            load_balancing=complexity_level == "high",
            fault_tolerance=True,
            expected_efficiency=0.1,  # 10% improvement
            implementation_complexity="low"
        )
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance and coordination efficiency"""
        
        if not self.agent_performance:
            return {"status": "no_data"}
        
        # Calculate performance metrics for each agent
        agent_summaries = {}
        for agent_name, performances in self.agent_performance.items():
            if not performances:
                continue
            
            recent_performances = performances[-10:]  # Last 10 performances
            
            agent_summaries[agent_name] = {
                "avg_execution_time": np.mean([p.execution_time for p in recent_performances]),
                "avg_success_rate": np.mean([p.success_rate for p in recent_performances]),
                "avg_quality_score": np.mean([p.quality_score for p in recent_performances]),
                "avg_coordination_overhead": np.mean([p.coordination_overhead for p in recent_performances]),
                "total_sessions": len(performances)
            }
        
        # Calculate coordination pattern performance
        pattern_performance = {}
        for pattern in CoordinationPattern:
            pattern_sessions = [s for s in self.coordination_sessions if s.coordination_pattern == pattern]
            if pattern_sessions:
                pattern_performance[pattern.value] = {
                    "avg_execution_time": np.mean([s.total_execution_time for s in pattern_sessions]),
                    "avg_quality_score": np.mean([s.quality_score for s in pattern_sessions]),
                    "avg_efficiency": np.mean([s.resource_efficiency for s in pattern_sessions]),
                    "avg_overhead": np.mean([s.coordination_overhead for s in pattern_sessions]),
                    "session_count": len(pattern_sessions)
                }
        
        return {
            "agent_performance": agent_summaries,
            "coordination_patterns": pattern_performance,
            "total_sessions": len(self.coordination_sessions),
            "optimization_strategies": len(self.coordination_strategies),
            "recommendations": self._generate_coordination_recommendations()
        }
    
    def _generate_coordination_recommendations(self) -> List[str]:
        """Generate coordination optimization recommendations"""
        
        recommendations = []
        
        if not self.coordination_sessions:
            return ["Collect more coordination session data for analysis"]
        
        # Analyze overall performance
        avg_overhead = np.mean([s.coordination_overhead for s in self.coordination_sessions])
        avg_quality = np.mean([s.quality_score for s in self.coordination_sessions])
        avg_efficiency = np.mean([s.resource_efficiency for s in self.coordination_sessions])
        
        if avg_overhead > 0.3:
            recommendations.append("High coordination overhead detected - consider optimizing communication protocols")
        
        if avg_quality < 0.7:
            recommendations.append("Low quality scores - consider implementing validation and quality control mechanisms")
        
        if avg_efficiency < 0.6:
            recommendations.append("Low resource efficiency - consider implementing load balancing and resource optimization")
        
        # Pattern-specific recommendations
        for pattern in CoordinationPattern:
            pattern_sessions = [s for s in self.coordination_sessions if s.coordination_pattern == pattern]
            if pattern_sessions:
                pattern_avg_overhead = np.mean([s.coordination_overhead for s in pattern_sessions])
                if pattern_avg_overhead > 0.4:
                    recommendations.append(f"High overhead in {pattern.value} pattern - consider optimization")
        
        if len(self.coordination_strategies) > 0:
            recommendations.append(f"Apply {len(self.coordination_strategies)} available optimization strategies")
        
        return recommendations
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default coordination strategies"""
        
        if not self.coordination_strategies:
            # Create default strategies for each pattern
            default_strategies = [
                CoordinationStrategy(
                    strategy_id="default_sequential",
                    pattern=CoordinationPattern.SEQUENTIAL,
                    agent_roles=self._get_optimized_roles(CoordinationPattern.SEQUENTIAL),
                    communication_protocol="standard",
                    load_balancing=False,
                    fault_tolerance=True,
                    expected_efficiency=0.1,
                    implementation_complexity="low"
                ),
                CoordinationStrategy(
                    strategy_id="default_parallel",
                    pattern=CoordinationPattern.PARALLEL,
                    agent_roles=self._get_optimized_roles(CoordinationPattern.PARALLEL),
                    communication_protocol="optimized",
                    load_balancing=True,
                    fault_tolerance=True,
                    expected_efficiency=0.2,
                    implementation_complexity="medium"
                ),
                CoordinationStrategy(
                    strategy_id="default_hierarchical",
                    pattern=CoordinationPattern.HIERARCHICAL,
                    agent_roles=self._get_optimized_roles(CoordinationPattern.HIERARCHICAL),
                    communication_protocol="enhanced",
                    load_balancing=True,
                    fault_tolerance=True,
                    expected_efficiency=0.15,
                    implementation_complexity="medium"
                )
            ]
            
            for strategy in default_strategies:
                self.coordination_strategies[strategy.strategy_id] = strategy
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load agent performance
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    self.agent_performance = {
                        agent_name: [
                            AgentPerformance(**perf_data)
                            for perf_data in performances
                        ]
                        for agent_name, performances in data.items()
                    }
            
            # Load coordination strategies
            if self.strategies_file.exists():
                with open(self.strategies_file, 'r') as f:
                    data = json.load(f)
                    self.coordination_strategies = {
                        strategy_id: CoordinationStrategy(**strategy_data)
                        for strategy_id, strategy_data in data.items()
                    }
            
            # Load coordination sessions
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    self.coordination_sessions = [
                        CoordinationSession(**session_data)
                        for session_data in data
                    ]
            
            logger.info(f"📁 Loaded coordination data: {len(self.agent_performance)} agents, {len(self.coordination_sessions)} sessions")
            
        except Exception as e:
            logger.warning(f"Failed to load coordination data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save agent performance
            performance_data = {
                agent_name: [
                    {
                        "agent_name": perf.agent_name,
                        "execution_time": perf.execution_time,
                        "success_rate": perf.success_rate,
                        "quality_score": perf.quality_score,
                        "resource_usage": perf.resource_usage,
                        "coordination_overhead": perf.coordination_overhead,
                        "timestamp": perf.timestamp
                    }
                    for perf in performances
                ]
                for agent_name, performances in self.agent_performance.items()
            }
            
            with open(self.performance_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            # Save coordination strategies
            strategies_data = {
                strategy_id: {
                    "strategy_id": strategy.strategy_id,
                    "pattern": strategy.pattern.value,
                    "agent_roles": {agent: role.value for agent, role in strategy.agent_roles.items()},
                    "communication_protocol": strategy.communication_protocol,
                    "load_balancing": strategy.load_balancing,
                    "fault_tolerance": strategy.fault_tolerance,
                    "expected_efficiency": strategy.expected_efficiency,
                    "implementation_complexity": strategy.implementation_complexity
                }
                for strategy_id, strategy in self.coordination_strategies.items()
            }
            
            with open(self.strategies_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
            
            # Save coordination sessions
            sessions_data = [
                {
                    "session_id": session.session_id,
                    "agents_involved": session.agents_involved,
                    "coordination_pattern": session.coordination_pattern.value,
                    "total_execution_time": session.total_execution_time,
                    "success": session.success,
                    "quality_score": session.quality_score,
                    "resource_efficiency": session.resource_efficiency,
                    "coordination_overhead": session.coordination_overhead,
                    "timestamp": session.timestamp
                }
                for session in self.coordination_sessions
            ]
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
            
            logger.debug("💾 Saved coordination data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save coordination data: {e}")


# Helper functions for integration
def create_multi_agent_coordinator(cfg: IceburgConfig) -> MultiAgentCoordinator:
    """Create multi-agent coordinator instance"""
    return MultiAgentCoordinator(cfg)

def record_agent_performance(
    coordinator: MultiAgentCoordinator,
    agent_name: str,
    execution_time: float,
    success: bool,
    quality_score: float,
    resource_usage: Dict[str, float],
    coordination_overhead: float = 0.0
) -> None:
    """Record agent performance for coordination optimization"""
    coordinator.record_agent_performance(agent_name, execution_time, success, quality_score, resource_usage, coordination_overhead)

def get_optimal_coordination_strategy(
    coordinator: MultiAgentCoordinator,
    agents_required: List[str],
    complexity_level: str = "medium",
    priority: str = "balanced"
) -> CoordinationStrategy:
    """Get optimal coordination strategy for agents"""
    return coordinator.get_optimal_coordination_strategy(agents_required, complexity_level, priority)
