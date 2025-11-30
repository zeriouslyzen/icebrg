"""
Autonomous Learning System for ICEBURG
Implements continuous learning, pattern detection, and autonomous improvement.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import hashlib

# Lazy imports to avoid circular dependencies
# from ..memory.unified_memory import UnifiedMemory
# from ..config import load_config_unified

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    """Types of improvements."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    USER_EXPERIENCE = "user_experience"
    SECURITY = "security"


class ApprovalStatus(Enum):
    """Approval status for improvements."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Improvement:
    """Represents a potential improvement."""
    improvement_id: str
    type: ImprovementType
    description: str
    expected_benefit: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    risk_level: str  # "low", "medium", "high"
    implementation_complexity: str  # "low", "medium", "high"
    affected_components: List[str]
    dependencies: List[str] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    created_time: float = 0.0
    approved_time: Optional[float] = None
    deployed_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Represents a learned pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]]
    created_time: float = 0.0
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationExperiment:
    """Represents an ablation experiment."""
    experiment_id: str
    baseline_config: Dict[str, Any]
    modified_config: Dict[str, Any]
    metrics: List[str]
    duration_hours: float = 24.0
    status: str = "pending"  # "pending", "running", "completed", "failed"
    results: Dict[str, Any] = field(default_factory=dict)
    created_time: float = 0.0
    completed_time: Optional[float] = None


class PatternDetector:
    """
    Detects patterns in ICEBURG interactions and performance.
    
    Features:
    - Pattern recognition in user queries
    - Performance pattern detection
    - Usage pattern analysis
    - Anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize pattern detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.patterns: Dict[str, LearningPattern] = {}
        self.pattern_counter = 0
        
        # Pattern detection thresholds
        self.min_frequency = self.config.get("min_frequency", 5)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.pattern_window_hours = self.config.get("pattern_window_hours", 24)
        
        # Pattern types
        self.pattern_types = [
            "query_similarity",
            "performance_bottleneck",
            "user_behavior",
            "error_pattern",
            "success_pattern",
            "resource_usage",
            "temporal_pattern"
        ]
    
    async def detect_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """
        Detect patterns in interactions.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            List of detected patterns
        """
        detected_patterns = []
        
        # Analyze different pattern types
        for pattern_type in self.pattern_types:
            patterns = await self._detect_pattern_type(pattern_type, interactions)
            detected_patterns.extend(patterns)
        
        # Update pattern database
        for pattern in detected_patterns:
            self.patterns[pattern.pattern_id] = pattern
        
        return detected_patterns
    
    async def _detect_pattern_type(self, pattern_type: str, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect patterns of a specific type."""
        if pattern_type == "query_similarity":
            return await self._detect_query_similarity_patterns(interactions)
        elif pattern_type == "performance_bottleneck":
            return await self._detect_performance_patterns(interactions)
        elif pattern_type == "user_behavior":
            return await self._detect_user_behavior_patterns(interactions)
        elif pattern_type == "error_pattern":
            return await self._detect_error_patterns(interactions)
        elif pattern_type == "success_pattern":
            return await self._detect_success_patterns(interactions)
        elif pattern_type == "resource_usage":
            return await self._detect_resource_patterns(interactions)
        elif pattern_type == "temporal_pattern":
            return await self._detect_temporal_patterns(interactions)
        
        return []
    
    async def _detect_query_similarity_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect query similarity patterns."""
        patterns = []
        
        # Group similar queries
        query_groups = {}
        for interaction in interactions:
            query = interaction.get("query", "")
            if not query:
                continue
            
            # Simple similarity grouping (in practice, use embeddings)
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
            if query_hash not in query_groups:
                query_groups[query_hash] = []
            query_groups[query_hash].append(interaction)
        
        # Create patterns for frequent query groups
        for query_hash, group in query_groups.items():
            if len(group) >= self.min_frequency:
                pattern = LearningPattern(
                    pattern_id=f"query_sim_{self.pattern_counter}",
                    pattern_type="query_similarity",
                    description=f"Similar queries pattern: {len(group)} occurrences",
                    frequency=len(group),
                    confidence=min(1.0, len(group) / 10.0),
                    examples=group[:5],  # First 5 examples
                    created_time=time.time(),
                    last_seen=time.time()
                )
                patterns.append(pattern)
                self.pattern_counter += 1
        
        return patterns
    
    async def _detect_performance_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect performance patterns."""
        patterns = []
        
        # Analyze response times
        response_times = [i.get("response_time", 0) for i in interactions if i.get("response_time")]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            if avg_response_time > 5.0:  # Slow responses
                pattern = LearningPattern(
                    pattern_id=f"perf_slow_{self.pattern_counter}",
                    pattern_type="performance_bottleneck",
                    description=f"Slow response times: avg {avg_response_time:.2f}s",
                    frequency=len(response_times),
                    confidence=0.8,
                    examples=interactions[:3],
                    created_time=time.time(),
                    last_seen=time.time()
                )
                patterns.append(pattern)
                self.pattern_counter += 1
        
        return patterns
    
    async def _detect_user_behavior_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect user behavior patterns."""
        patterns = []
        
        # Analyze user session patterns
        user_sessions = {}
        for interaction in interactions:
            user_id = interaction.get("user_id", "anonymous")
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            user_sessions[user_id].append(interaction)
        
        # Detect frequent user patterns
        for user_id, session in user_sessions.items():
            if len(session) >= self.min_frequency:
                pattern = LearningPattern(
                    pattern_id=f"user_behavior_{self.pattern_counter}",
                    pattern_type="user_behavior",
                    description=f"User {user_id} behavior pattern: {len(session)} interactions",
                    frequency=len(session),
                    confidence=0.7,
                    examples=session[:3],
                    created_time=time.time(),
                    last_seen=time.time()
                )
                patterns.append(pattern)
                self.pattern_counter += 1
        
        return patterns
    
    async def _detect_error_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect error patterns."""
        patterns = []
        
        # Analyze error patterns
        error_interactions = [i for i in interactions if i.get("status") == "error"]
        if error_interactions:
            error_types = {}
            for interaction in error_interactions:
                error_type = interaction.get("error_type", "unknown")
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(interaction)
            
            # Create patterns for frequent error types
            for error_type, errors in error_types.items():
                if len(errors) >= self.min_frequency:
                    pattern = LearningPattern(
                        pattern_id=f"error_{self.pattern_counter}",
                        pattern_type="error_pattern",
                        description=f"Error pattern: {error_type} ({len(errors)} occurrences)",
                        frequency=len(errors),
                        confidence=0.9,
                        examples=errors[:3],
                        created_time=time.time(),
                        last_seen=time.time()
                    )
                    patterns.append(pattern)
                    self.pattern_counter += 1
        
        return patterns
    
    async def _detect_success_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect success patterns."""
        patterns = []
        
        # Analyze successful interactions
        success_interactions = [i for i in interactions if i.get("status") == "success"]
        if success_interactions:
            # Group by query type
            query_types = {}
            for interaction in success_interactions:
                query_type = interaction.get("query_type", "general")
                if query_type not in query_types:
                    query_types[query_type] = []
                query_types[query_type].append(interaction)
            
            # Create patterns for successful query types
            for query_type, successes in query_types.items():
                if len(successes) >= self.min_frequency:
                    pattern = LearningPattern(
                        pattern_id=f"success_{self.pattern_counter}",
                        pattern_type="success_pattern",
                        description=f"Success pattern: {query_type} ({len(successes)} successes)",
                        frequency=len(successes),
                        confidence=0.8,
                        examples=successes[:3],
                        created_time=time.time(),
                        last_seen=time.time()
                    )
                    patterns.append(pattern)
                    self.pattern_counter += 1
        
        return patterns
    
    async def _detect_resource_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect resource usage patterns."""
        patterns = []
        
        # Analyze resource usage
        resource_data = [i for i in interactions if i.get("resource_usage")]
        if resource_data:
            # Calculate average resource usage
            avg_memory = statistics.mean([i.get("resource_usage", {}).get("memory", 0) for i in resource_data])
            avg_cpu = statistics.mean([i.get("resource_usage", {}).get("cpu", 0) for i in resource_data])
            
            if avg_memory > 1024 * 1024 * 1024 * 2:  # 2GB
                pattern = LearningPattern(
                    pattern_id=f"resource_memory_{self.pattern_counter}",
                    pattern_type="resource_usage",
                    description=f"High memory usage: avg {avg_memory / (1024**3):.2f}GB",
                    frequency=len(resource_data),
                    confidence=0.7,
                    examples=resource_data[:3],
                    created_time=time.time(),
                    last_seen=time.time()
                )
                patterns.append(pattern)
                self.pattern_counter += 1
        
        return patterns
    
    async def _detect_temporal_patterns(self, interactions: List[Dict[str, Any]]) -> List[LearningPattern]:
        """Detect temporal patterns."""
        patterns = []
        
        # Analyze time-based patterns
        timestamps = [i.get("timestamp", 0) for i in interactions if i.get("timestamp")]
        if timestamps:
            # Group by hour of day
            hour_groups = {}
            for timestamp in timestamps:
                hour = time.localtime(timestamp).tm_hour
                if hour not in hour_groups:
                    hour_groups[hour] = 0
                hour_groups[hour] += 1
            
            # Detect peak usage hours
            max_hour = max(hour_groups.items(), key=lambda x: x[1])
            if max_hour[1] >= self.min_frequency:
                pattern = LearningPattern(
                    pattern_id=f"temporal_{self.pattern_counter}",
                    pattern_type="temporal_pattern",
                    description=f"Peak usage at hour {max_hour[0]}: {max_hour[1]} interactions",
                    frequency=max_hour[1],
                    confidence=0.6,
                    examples=interactions[:3],
                    created_time=time.time(),
                    last_seen=time.time()
                )
                patterns.append(pattern)
                self.pattern_counter += 1
        
        return patterns
    
    def get_patterns(self, pattern_type: str = None) -> List[LearningPattern]:
        """Get detected patterns."""
        if pattern_type:
            return [p for p in self.patterns.values() if p.pattern_type == pattern_type]
        return list(self.patterns.values())


class AblationRunner:
    """
    Runs ablation experiments to test improvements.
    
    Features:
    - A/B testing framework
    - Performance comparison
    - Statistical significance testing
    - Automated experiment management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ablation runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.experiments: Dict[str, AblationExperiment] = {}
        self.experiment_counter = 0
        
        # Experiment configuration
        self.default_duration = self.config.get("default_duration_hours", 24.0)
        self.min_sample_size = self.config.get("min_sample_size", 100)
        self.significance_threshold = self.config.get("significance_threshold", 0.05)
    
    async def run_experiment(self, 
                           baseline_config: Dict[str, Any],
                           modified_config: Dict[str, Any],
                           metrics: List[str],
                           duration_hours: float = None) -> AblationExperiment:
        """
        Run an ablation experiment.
        
        Args:
            baseline_config: Baseline configuration
            modified_config: Modified configuration
            metrics: Metrics to measure
            duration_hours: Experiment duration
            
        Returns:
            Experiment results
        """
        if duration_hours is None:
            duration_hours = self.default_duration
        
        experiment = AblationExperiment(
            experiment_id=f"exp_{self.experiment_counter}",
            baseline_config=baseline_config,
            modified_config=modified_config,
            metrics=metrics,
            duration_hours=duration_hours,
            status="running",
            created_time=time.time()
        )
        
        self.experiments[experiment.experiment_id] = experiment
        self.experiment_counter += 1
        
        # Run experiment
        try:
            results = await self._execute_experiment(experiment)
            experiment.results = results
            experiment.status = "completed"
            experiment.completed_time = time.time()
            
            logger.info(f"Experiment {experiment.experiment_id} completed")
            
        except Exception as e:
            experiment.status = "failed"
            experiment.results = {"error": str(e)}
            logger.error(f"Experiment {experiment.experiment_id} failed: {e}")
        
        return experiment
    
    async def _execute_experiment(self, experiment: AblationExperiment) -> Dict[str, Any]:
        """Execute an ablation experiment."""
        # This would integrate with actual ICEBURG testing
        # For now, simulate experiment results
        
        baseline_results = {
            "response_time": 2.5,
            "accuracy": 0.85,
            "memory_usage": 1024 * 1024 * 1024,  # 1GB
            "cpu_usage": 45.0,
            "error_rate": 0.05
        }
        
        modified_results = {
            "response_time": 2.0,  # Improved
            "accuracy": 0.88,     # Improved
            "memory_usage": 1024 * 1024 * 1024 * 1.2,  # Slightly higher
            "cpu_usage": 50.0,    # Slightly higher
            "error_rate": 0.03   # Improved
        }
        
        # Calculate improvements
        improvements = {}
        for metric in experiment.metrics:
            if metric in baseline_results and metric in modified_results:
                baseline_val = baseline_results[metric]
                modified_val = modified_results[metric]
                
                if baseline_val > 0:
                    improvement = (modified_val - baseline_val) / baseline_val
                    improvements[metric] = improvement
        
        # Calculate overall improvement score
        overall_improvement = statistics.mean(improvements.values()) if improvements else 0.0
        
        # Determine if improvement is significant
        is_significant = abs(overall_improvement) > 0.1  # 10% improvement threshold
        
        return {
            "baseline_results": baseline_results,
            "modified_results": modified_results,
            "improvements": improvements,
            "overall_improvement": overall_improvement,
            "is_significant": is_significant,
            "sample_size": 1000,  # Mock sample size
            "confidence_interval": [0.05, 0.95],  # Mock confidence interval
            "p_value": 0.01 if is_significant else 0.5  # Mock p-value
        }
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results."""
        experiment = self.experiments.get(experiment_id)
        if experiment:
            return experiment.results
        return None
    
    def get_all_experiments(self) -> List[AblationExperiment]:
        """Get all experiments."""
        return list(self.experiments.values())


class ApprovalQueue:
    """
    Manages approval queue for improvements.
    
    Features:
    - Human oversight for major changes
    - Approval workflow
    - Risk assessment
    - Deployment tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize approval queue.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.approval_queue: List[Improvement] = []
        self.approved_improvements: List[Improvement] = []
        self.deployed_improvements: List[Improvement] = []
        
        # Approval thresholds
        self.auto_approve_threshold = self.config.get("auto_approve_threshold", 0.8)
        self.risk_threshold = self.config.get("risk_threshold", 0.3)
    
    async def add_improvement(self, improvement: Improvement):
        """Add improvement to approval queue."""
        # Check if auto-approval is possible
        if (improvement.confidence >= self.auto_approve_threshold and 
            improvement.risk_level == "low" and
            improvement.implementation_complexity == "low"):
            
            improvement.approval_status = ApprovalStatus.APPROVED
            improvement.approved_time = time.time()
            self.approved_improvements.append(improvement)
            logger.info(f"Auto-approved improvement: {improvement.improvement_id}")
        else:
            improvement.approval_status = ApprovalStatus.PENDING
            self.approval_queue.append(improvement)
            logger.info(f"Added improvement to approval queue: {improvement.improvement_id}")
    
    async def approve_improvement(self, improvement_id: str, approver: str = "human"):
        """Approve an improvement."""
        improvement = next((i for i in self.approval_queue if i.improvement_id == improvement_id), None)
        if improvement:
            improvement.approval_status = ApprovalStatus.APPROVED
            improvement.approved_time = time.time()
            improvement.metadata["approver"] = approver
            
            self.approval_queue.remove(improvement)
            self.approved_improvements.append(improvement)
            
            logger.info(f"Approved improvement: {improvement_id} by {approver}")
    
    async def reject_improvement(self, improvement_id: str, reason: str = "Not approved"):
        """Reject an improvement."""
        improvement = next((i for i in self.approval_queue if i.improvement_id == improvement_id), None)
        if improvement:
            improvement.approval_status = ApprovalStatus.REJECTED
            improvement.metadata["rejection_reason"] = reason
            
            self.approval_queue.remove(improvement)
            logger.info(f"Rejected improvement: {improvement_id} - {reason}")
    
    async def deploy_improvement(self, improvement_id: str):
        """Deploy an approved improvement."""
        improvement = next((i for i in self.approved_improvements if i.improvement_id == improvement_id), None)
        if improvement:
            improvement.approval_status = ApprovalStatus.DEPLOYED
            improvement.deployed_time = time.time()
            
            self.approved_improvements.remove(improvement)
            self.deployed_improvements.append(improvement)
            
            logger.info(f"Deployed improvement: {improvement_id}")
    
    def get_pending_approvals(self) -> List[Improvement]:
        """Get pending approvals."""
        return [i for i in self.approval_queue if i.approval_status == ApprovalStatus.PENDING]
    
    def get_approved_improvements(self) -> List[Improvement]:
        """Get approved improvements."""
        return self.approved_improvements
    
    def get_deployed_improvements(self) -> List[Improvement]:
        """Get deployed improvements."""
        return self.deployed_improvements


class AutonomousLearner:
    """
    Main autonomous learning system for ICEBURG.
    
    Features:
    - Continuous learning from interactions
    - Pattern detection and analysis
    - Autonomous improvement generation
    - Human oversight and approval
    - A/B testing and validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize autonomous learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.pattern_detector = PatternDetector(config)
        self.ablation_runner = AblationRunner(config)
        self.approval_queue = ApprovalQueue(config)
        
        # Learning state
        self.learning_active = False
        self.learning_task = None
        self.improvement_counter = 0
        
        # Learning configuration
        self.learning_interval = self.config.get("learning_interval_hours", 24.0)
        self.min_interactions = self.config.get("min_interactions", 100)
        self.improvement_threshold = self.config.get("improvement_threshold", 0.1)
    
    async def start_learning(self):
        """Start autonomous learning."""
        if self.learning_active:
            return
        
        self.learning_active = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Autonomous learning started")
    
    async def stop_learning(self):
        """Stop autonomous learning."""
        if not self.learning_active:
            return
        
        self.learning_active = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous learning stopped")
    
    async def _learning_loop(self):
        """Main learning loop."""
        while self.learning_active:
            try:
                # Collect recent interactions
                interactions = await self._collect_recent_interactions()
                
                if len(interactions) >= self.min_interactions:
                    # Detect patterns
                    patterns = await self.pattern_detector.detect_patterns(interactions)
                    
                    # Generate improvements
                    improvements = await self._generate_improvements(patterns, interactions)
                    
                    # Test improvements
                    for improvement in improvements:
                        await self._test_improvement(improvement)
                    
                    # Add to approval queue
                    for improvement in improvements:
                        await self.approval_queue.add_improvement(improvement)
                
                # Wait before next learning cycle
                await asyncio.sleep(self.learning_interval * 3600)  # Convert hours to seconds
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _collect_recent_interactions(self) -> List[Dict[str, Any]]:
        """Collect recent interactions for learning."""
        # This would integrate with unified memory system
        # For now, return mock interactions
        
        mock_interactions = []
        for i in range(100):
            interaction = {
                "query": f"Mock query {i}",
                "response_time": 2.0 + (i % 5),
                "status": "success" if i % 10 != 0 else "error",
                "user_id": f"user_{i % 10}",
                "timestamp": time.time() - (i * 3600),  # Spread over time
                "resource_usage": {
                    "memory": 1024 * 1024 * 1024 * (1 + i % 3),
                    "cpu": 40 + (i % 20)
                },
                "query_type": ["research", "chat", "software", "civilization"][i % 4],
                "error_type": "timeout" if i % 10 == 0 else None
            }
            mock_interactions.append(interaction)
        
        return mock_interactions
    
    async def _generate_improvements(self, 
                                   patterns: List[LearningPattern], 
                                   interactions: List[Dict[str, Any]]) -> List[Improvement]:
        """Generate improvements based on patterns."""
        improvements = []
        
        for pattern in patterns:
            if pattern.confidence >= self.improvement_threshold:
                improvement = await self._create_improvement_from_pattern(pattern, interactions)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    async def _create_improvement_from_pattern(self, 
                                            pattern: LearningPattern, 
                                            interactions: List[Dict[str, Any]]) -> Optional[Improvement]:
        """Create improvement from a pattern."""
        if pattern.pattern_type == "performance_bottleneck":
            return Improvement(
                improvement_id=f"improvement_{self.improvement_counter}",
                type=ImprovementType.PERFORMANCE,
                description=f"Optimize {pattern.description}",
                expected_benefit=0.2,
                confidence=pattern.confidence,
                risk_level="low",
                implementation_complexity="medium",
                affected_components=["protocol", "agents"],
                created_time=time.time()
            )
        
        elif pattern.pattern_type == "error_pattern":
            return Improvement(
                improvement_id=f"improvement_{self.improvement_counter}",
                type=ImprovementType.RELIABILITY,
                description=f"Fix {pattern.description}",
                expected_benefit=0.3,
                confidence=pattern.confidence,
                risk_level="medium",
                implementation_complexity="high",
                affected_components=["error_handling", "validation"],
                created_time=time.time()
            )
        
        elif pattern.pattern_type == "success_pattern":
            return Improvement(
                improvement_id=f"improvement_{self.improvement_counter}",
                type=ImprovementType.ACCURACY,
                description=f"Enhance {pattern.description}",
                expected_benefit=0.15,
                confidence=pattern.confidence,
                risk_level="low",
                implementation_complexity="low",
                affected_components=["agents", "synthesis"],
                created_time=time.time()
            )
        
        self.improvement_counter += 1
        return None
    
    async def _test_improvement(self, improvement: Improvement):
        """Test an improvement using ablation experiments."""
        # Create baseline and modified configurations
        baseline_config = {
            "parallel_execution": True,
            "cache_enabled": True,
            "agent_timeout": 30.0
        }
        
        modified_config = baseline_config.copy()
        # Apply improvement-specific modifications
        if improvement.type == ImprovementType.PERFORMANCE:
            modified_config["agent_timeout"] = 20.0  # Reduce timeout
        elif improvement.type == ImprovementType.RELIABILITY:
            modified_config["error_retry_count"] = 3  # Add retry logic
        
        # Run experiment
        experiment = await self.ablation_runner.run_experiment(
            baseline_config=baseline_config,
            modified_config=modified_config,
            metrics=["response_time", "accuracy", "error_rate"],
            duration_hours=1.0  # Short test
        )
        
        # Store results
        improvement.test_results = experiment.results
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get learning status."""
        return {
            "learning_active": self.learning_active,
            "patterns_detected": len(self.pattern_detector.patterns),
            "pending_approvals": len(self.approval_queue.get_pending_approvals()),
            "approved_improvements": len(self.approval_queue.get_approved_improvements()),
            "deployed_improvements": len(self.approval_queue.get_deployed_improvements()),
            "total_experiments": len(self.ablation_runner.experiments)
        }
    
    async def cleanup(self):
        """Cleanup learning resources."""
        await self.stop_learning()
        logger.info("Autonomous learning cleanup completed")


# Convenience functions
async def create_autonomous_learner(config: Dict[str, Any] = None) -> AutonomousLearner:
    """Create a new autonomous learner."""
    return AutonomousLearner(config=config)


async def start_iceburg_learning(learner: AutonomousLearner = None) -> AutonomousLearner:
    """Start ICEBURG autonomous learning."""
    if learner is None:
        learner = await create_autonomous_learner()
    
    await learner.start_learning()
    return learner


async def get_learning_status(learner: AutonomousLearner) -> Dict[str, Any]:
    """Get learning status."""
    return learner.get_learning_status()
