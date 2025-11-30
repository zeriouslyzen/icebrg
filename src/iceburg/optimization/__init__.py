"""
ICEBURG Optimization Module
Comprehensive optimization system for ICEBURG performance and capabilities

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

from .model_evolution_tracker import (
    ModelEvolutionTracker,
    ModelCapability,
    ModelEvolutionPattern,
    NextModelPrediction,
    create_model_evolution_tracker,
    track_model_performance,
    predict_next_model
)

from .cross_domain_synthesis_optimizer import (
    CrossDomainSynthesisOptimizer,
    SynthesisPattern,
    DomainConnection,
    SynthesisOptimization,
    create_synthesis_optimizer,
    record_synthesis_attempt,
    get_synthesis_strategy
)

from .technology_trend_detector import (
    TechnologyTrendDetector,
    TechnologySignal,
    TechnologyTrend,
    TrendPrediction,
    TrendType,
    EmergenceLevel,
    create_technology_trend_detector,
    add_technology_signal,
    get_emerging_technologies
)

from .next_model_predictor import (
    NextModelPredictor,
    CapabilityPrediction,
    ArchitecturePrediction,
    OptimizationRoadmap,
    create_next_model_predictor,
    predict_model_capabilities,
    create_optimization_roadmap
)

from .latent_space_optimizer import (
    LatentSpaceOptimizer,
    LatentSpaceMetrics,
    OptimizationStrategy,
    LatentSpaceProfile,
    create_latent_space_optimizer,
    record_reasoning_metrics,
    get_optimized_profile
)

from .multi_agent_coordinator import (
    MultiAgentCoordinator,
    AgentPerformance,
    CoordinationStrategy,
    CoordinationSession,
    CoordinationPattern,
    AgentRole,
    create_multi_agent_coordinator,
    record_agent_performance,
    get_optimal_coordination_strategy
)

from .synthesis_speed_optimizer import (
    SynthesisSpeedOptimizer,
    SynthesisTemplate,
    SynthesisCache,
    SpeedOptimization,
    create_synthesis_speed_optimizer,
    optimize_synthesis_speed,
    get_speed_optimization_recommendations
)

from .blockchain_performance_optimizer import (
    BlockchainPerformanceOptimizer,
    BlockPerformanceMetrics,
    BlockchainOptimization,
    MiningPool,
    create_blockchain_performance_optimizer,
    optimize_mining_performance,
    optimize_verification_performance
)

from .model_performance_registry import (
    ModelPerformanceRegistry,
    ModelPerformanceRecord,
    ModelCapabilityProfile,
    PerformanceBenchmark,
    create_model_performance_registry,
    record_model_performance,
    get_model_performance_summary
)

try:
    from .predictive_history_analyzer import (
        PredictiveHistoryAnalyzer,
        HistoricalDataPoint,
        PredictionResult,
        OptimizationOpportunity,
        create_predictive_history_analyzer,
        add_historical_data,
        get_predictions,
        get_optimization_opportunities
    )
except ImportError:
    # Optional dependency (sklearn)
    PredictiveHistoryAnalyzer = None
    HistoricalDataPoint = None
    PredictionResult = None
    OptimizationOpportunity = None
    create_predictive_history_analyzer = None
    add_historical_data = None
    get_predictions = None
    get_optimization_opportunities = None

__all__ = [
    # Model Evolution Tracker
    "ModelEvolutionTracker",
    "ModelCapability",
    "ModelEvolutionPattern",
    "NextModelPrediction",
    "create_model_evolution_tracker",
    "track_model_performance",
    "predict_next_model",
    
    # Cross-Domain Synthesis Optimizer
    "CrossDomainSynthesisOptimizer",
    "SynthesisPattern",
    "DomainConnection",
    "SynthesisOptimization",
    "create_synthesis_optimizer",
    "record_synthesis_attempt",
    "get_synthesis_strategy",
    
    # Technology Trend Detector
    "TechnologyTrendDetector",
    "TechnologySignal",
    "TechnologyTrend",
    "TrendPrediction",
    "TrendType",
    "EmergenceLevel",
    "create_technology_trend_detector",
    "add_technology_signal",
    "get_emerging_technologies",
    
    # Next Model Predictor
    "NextModelPredictor",
    "CapabilityPrediction",
    "ArchitecturePrediction",
    "OptimizationRoadmap",
    "create_next_model_predictor",
    "predict_model_capabilities",
    "create_optimization_roadmap",
    
    # Latent Space Optimizer
    "LatentSpaceOptimizer",
    "LatentSpaceMetrics",
    "OptimizationStrategy",
    "LatentSpaceProfile",
    "create_latent_space_optimizer",
    "record_reasoning_metrics",
    "get_optimized_profile",
    
    # Multi-Agent Coordinator
    "MultiAgentCoordinator",
    "AgentPerformance",
    "CoordinationStrategy",
    "CoordinationSession",
    "CoordinationPattern",
    "AgentRole",
    "create_multi_agent_coordinator",
    "record_agent_performance",
    "get_optimal_coordination_strategy",
    
    # Synthesis Speed Optimizer
    "SynthesisSpeedOptimizer",
    "SynthesisTemplate",
    "SynthesisCache",
    "SpeedOptimization",
    "create_synthesis_speed_optimizer",
    "optimize_synthesis_speed",
    "get_speed_optimization_recommendations",
    
    # Blockchain Performance Optimizer
    "BlockchainPerformanceOptimizer",
    "BlockPerformanceMetrics",
    "BlockchainOptimization",
    "MiningPool",
    "create_blockchain_performance_optimizer",
    "optimize_mining_performance",
    "optimize_verification_performance",
    
    # Model Performance Registry
    "ModelPerformanceRegistry",
    "ModelPerformanceRecord",
    "ModelCapabilityProfile",
    "PerformanceBenchmark",
    "create_model_performance_registry",
    "record_model_performance",
    "get_model_performance_summary",
    
    # Predictive History Analyzer
    "PredictiveHistoryAnalyzer",
    "HistoricalDataPoint",
    "PredictionResult",
    "OptimizationOpportunity",
    "create_predictive_history_analyzer",
    "add_historical_data",
    "get_predictions",
    "get_optimization_opportunities"
]