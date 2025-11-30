"""
ICEBURG Optimization Orchestrator
Central coordinator for all optimization systems

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..config import IceburgConfig
from .model_evolution_tracker import ModelEvolutionTracker
from .cross_domain_synthesis_optimizer import CrossDomainSynthesisOptimizer
from .technology_trend_detector import TechnologyTrendDetector
from .next_model_predictor import NextModelPredictor
from .latent_space_optimizer import LatentSpaceOptimizer
from .multi_agent_coordinator import MultiAgentCoordinator
from .synthesis_speed_optimizer import SynthesisSpeedOptimizer
from .blockchain_performance_optimizer import BlockchainPerformanceOptimizer
from .model_performance_registry import ModelPerformanceRegistry
from .predictive_history_analyzer import PredictiveHistoryAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class OptimizationReport:
    """Comprehensive optimization report"""
    report_id: str
    timestamp: float
    optimization_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    implementation_priority: List[str]
    expected_improvements: Dict[str, float]

class OptimizationOrchestrator:
    """
    Central orchestrator for all ICEBURG optimization systems
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/orchestrator")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all optimization systems
        self.model_evolution_tracker = ModelEvolutionTracker(cfg)
        self.synthesis_optimizer = CrossDomainSynthesisOptimizer(cfg)
        self.trend_detector = TechnologyTrendDetector(cfg)
        self.next_model_predictor = NextModelPredictor(cfg)
        self.latent_space_optimizer = LatentSpaceOptimizer(cfg)
        self.multi_agent_coordinator = MultiAgentCoordinator(cfg)
        self.synthesis_speed_optimizer = SynthesisSpeedOptimizer(cfg)
        self.blockchain_optimizer = BlockchainPerformanceOptimizer(cfg)
        self.performance_registry = ModelPerformanceRegistry(cfg)
        self.history_analyzer = PredictiveHistoryAnalyzer(cfg)
        
        # Storage files
        self.reports_file = self.data_dir / "optimization_reports.json"
        
        # Data structures
        self.optimization_reports: List[OptimizationReport] = []
        
        logger.info("🎯 ICEBURG Optimization Orchestrator initialized")
    
    def run_comprehensive_optimization_analysis(self) -> OptimizationReport:
        """Run comprehensive optimization analysis across all systems"""
        
        start_time = time.time()
        report_id = f"optimization_report_{int(time.time())}"
        
        logger.info("🔍 Starting comprehensive optimization analysis...")
        
        # Collect data from all optimization systems
        optimization_summary = self._collect_optimization_data()
        
        # Analyze performance metrics
        performance_metrics = self._analyze_performance_metrics()
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations()
        
        # Prioritize implementation
        implementation_priority = self._prioritize_implementations(recommendations)
        
        # Calculate expected improvements
        expected_improvements = self._calculate_expected_improvements(implementation_priority)
        
        # Create comprehensive report
        report = OptimizationReport(
            report_id=report_id,
            timestamp=time.time(),
            optimization_summary=optimization_summary,
            performance_metrics=performance_metrics,
            recommendations=recommendations,
            implementation_priority=implementation_priority,
            expected_improvements=expected_improvements
        )
        
        self.optimization_reports.append(report)
        
        # Keep only last 100 reports
        if len(self.optimization_reports) > 100:
            self.optimization_reports = self.optimization_reports[-100:]
        
        # Save report
        self._save_reports()
        
        analysis_time = time.time() - start_time
        logger.info(f"✅ Comprehensive optimization analysis completed in {analysis_time:.2f}s")
        
        return report
    
    def _collect_optimization_data(self) -> Dict[str, Any]:
        """Collect optimization data from all systems"""
        
        return {
            "model_evolution": self.model_evolution_tracker.get_evolution_summary(),
            "synthesis_optimization": self.synthesis_optimizer.get_performance_summary(),
            "technology_trends": self.trend_detector.get_trend_summary(),
            "next_model_predictions": self.next_model_predictor.get_prediction_summary(),
            "latent_space_optimization": self.latent_space_optimizer.get_performance_summary(),
            "multi_agent_coordination": self.multi_agent_coordinator.get_agent_performance_summary(),
            "synthesis_speed": self.synthesis_speed_optimizer.get_performance_summary(),
            "blockchain_performance": self.blockchain_optimizer.get_performance_summary(),
            "model_performance": self.performance_registry.get_registry_summary(),
            "predictive_analysis": self.history_analyzer.get_analysis_summary()
        }
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across all systems"""
        
        # Get performance data from all systems
        model_performance = self.performance_registry.get_registry_summary()
        synthesis_performance = self.synthesis_optimizer.get_performance_summary()
        coordination_performance = self.multi_agent_coordinator.get_agent_performance_summary()
        blockchain_performance = self.blockchain_optimizer.get_performance_summary()
        
        # Calculate overall performance metrics
        overall_metrics = {
            "system_health": self._calculate_system_health(),
            "performance_trends": self._analyze_performance_trends(),
            "optimization_opportunities": self._count_optimization_opportunities(),
            "resource_efficiency": self._calculate_resource_efficiency()
        }
        
        return {
            "overall_metrics": overall_metrics,
            "model_performance": model_performance,
            "synthesis_performance": synthesis_performance,
            "coordination_performance": coordination_performance,
            "blockchain_performance": blockchain_performance
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        
        # Get performance data from key systems
        model_performance = self.performance_registry.get_registry_summary()
        synthesis_performance = self.synthesis_optimizer.get_performance_summary()
        coordination_performance = self.multi_agent_coordinator.get_agent_performance_summary()
        
        # Calculate health scores for each system
        health_scores = []
        
        # Model performance health
        if model_performance.get("recent_performance"):
            recent_perf = model_performance["recent_performance"]
            model_health = (recent_perf.get("last_10_success_rate", 0.8) + 
                          recent_perf.get("last_10_avg_quality", 0.7)) / 2
            health_scores.append(model_health)
        
        # Synthesis performance health
        if synthesis_performance.get("overall_success_rate"):
            synthesis_health = synthesis_performance["overall_success_rate"]
            health_scores.append(synthesis_health)
        
        # Coordination performance health
        if coordination_performance.get("overall_performance"):
            coord_perf = coordination_performance["overall_performance"]
            coord_health = coord_perf.get("success_rate", 0.8)
            health_scores.append(coord_health)
        
        # Calculate overall health score
        if health_scores:
            return sum(health_scores) / len(health_scores)
        else:
            return 0.8  # Default health score
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across systems"""
        
        # Get trend data from predictive analyzer
        analysis_summary = self.history_analyzer.get_analysis_summary()
        
        return {
            "trend_direction": "improving" if analysis_summary.get("analysis_statistics", {}).get("avg_model_accuracy", 0.5) > 0.7 else "stable",
            "optimization_velocity": len(self.optimization_reports) if self.optimization_reports else 0,
            "prediction_confidence": analysis_summary.get("prediction_quality", {}).get("avg_confidence", 0.5)
        }
    
    def _count_optimization_opportunities(self) -> int:
        """Count total optimization opportunities across all systems"""
        
        total_opportunities = 0
        
        # Count opportunities from each system
        synthesis_perf = self.synthesis_optimizer.get_performance_summary()
        if synthesis_perf.get("optimization_recommendations"):
            total_opportunities += len(synthesis_perf["optimization_recommendations"])
        
        coordination_perf = self.multi_agent_coordinator.get_agent_performance_summary()
        if coordination_perf.get("recommendations"):
            total_opportunities += len(coordination_perf["recommendations"])
        
        blockchain_perf = self.blockchain_optimizer.get_performance_summary()
        if blockchain_perf.get("recommendations"):
            total_opportunities += len(blockchain_perf["recommendations"])
        
        analysis_summary = self.history_analyzer.get_analysis_summary()
        if analysis_summary.get("optimization_potential", {}).get("high_priority_opportunities"):
            total_opportunities += analysis_summary["optimization_potential"]["high_priority_opportunities"]
        
        return total_opportunities
    
    def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource efficiency score"""
        
        # Get efficiency data from key systems
        model_performance = self.performance_registry.get_registry_summary()
        synthesis_performance = self.synthesis_optimizer.get_performance_summary()
        blockchain_performance = self.blockchain_optimizer.get_performance_summary()
        
        efficiency_scores = []
        
        # Model efficiency
        if model_performance.get("recent_performance"):
            recent_perf = model_performance["recent_performance"]
            if recent_perf.get("last_10_avg_time"):
                # Lower execution time = higher efficiency
                model_efficiency = max(0.0, 1.0 - (recent_perf["last_10_avg_time"] / 10.0))
                efficiency_scores.append(model_efficiency)
        
        # Synthesis efficiency
        if synthesis_performance.get("avg_processing_time"):
            synthesis_efficiency = max(0.0, 1.0 - (synthesis_performance["avg_processing_time"] / 5.0))
            efficiency_scores.append(synthesis_efficiency)
        
        # Blockchain efficiency
        if blockchain_performance.get("overall_performance"):
            blockchain_perf = blockchain_performance["overall_performance"]
            if blockchain_perf.get("avg_processing_time"):
                blockchain_efficiency = max(0.0, 1.0 - (blockchain_perf["avg_processing_time"] / 10.0))
                efficiency_scores.append(blockchain_efficiency)
        
        # Calculate overall efficiency
        if efficiency_scores:
            return sum(efficiency_scores) / len(efficiency_scores)
        else:
            return 0.7  # Default efficiency score
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        
        recommendations = []
        
        # Collect recommendations from all systems
        synthesis_perf = self.synthesis_optimizer.get_performance_summary()
        if synthesis_perf.get("recommendations"):
            recommendations.extend(synthesis_perf["recommendations"])
        
        coordination_perf = self.multi_agent_coordinator.get_agent_performance_summary()
        if coordination_perf.get("recommendations"):
            recommendations.extend(coordination_perf["recommendations"])
        
        blockchain_perf = self.blockchain_optimizer.get_performance_summary()
        if blockchain_perf.get("recommendations"):
            recommendations.extend(blockchain_perf["recommendations"])
        
        analysis_summary = self.history_analyzer.get_analysis_summary()
        if analysis_summary.get("recommendations"):
            recommendations.extend(analysis_summary["recommendations"])
        
        # Add system-wide recommendations
        system_health = self._calculate_system_health()
        if system_health < 0.7:
            recommendations.append("System health below optimal - implement comprehensive optimization strategy")
        
        resource_efficiency = self._calculate_resource_efficiency()
        if resource_efficiency < 0.6:
            recommendations.append("Resource efficiency below optimal - optimize resource allocation and usage")
        
        # Remove duplicates and limit to top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:20]  # Top 20 recommendations
    
    def _prioritize_implementations(self, recommendations: List[str]) -> List[str]:
        """Prioritize implementation of recommendations"""
        
        # Simple prioritization based on keywords and system impact
        high_priority_keywords = ["critical", "high", "system", "performance", "efficiency"]
        medium_priority_keywords = ["medium", "optimization", "improvement", "enhancement"]
        low_priority_keywords = ["low", "fine-tune", "monitor", "analysis"]
        
        prioritized = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for recommendation in recommendations:
            recommendation_lower = recommendation.lower()
            
            if any(keyword in recommendation_lower for keyword in high_priority_keywords):
                prioritized["high"].append(recommendation)
            elif any(keyword in recommendation_lower for keyword in medium_priority_keywords):
                prioritized["medium"].append(recommendation)
            elif any(keyword in recommendation_lower for keyword in low_priority_keywords):
                prioritized["low"].append(recommendation)
            else:
                prioritized["medium"].append(recommendation)  # Default to medium
        
        # Return prioritized list
        return (prioritized["high"] + prioritized["medium"] + prioritized["low"])
    
    def _calculate_expected_improvements(self, implementation_priority: List[str]) -> Dict[str, float]:
        """Calculate expected improvements from implementing recommendations"""
        
        improvements = {
            "performance": 0.0,
            "efficiency": 0.0,
            "reliability": 0.0,
            "scalability": 0.0
        }
        
        # Analyze recommendations for improvement potential
        for recommendation in implementation_priority[:10]:  # Top 10 recommendations
            recommendation_lower = recommendation.lower()
            
            if "performance" in recommendation_lower or "speed" in recommendation_lower:
                improvements["performance"] += 0.1
            if "efficiency" in recommendation_lower or "resource" in recommendation_lower:
                improvements["efficiency"] += 0.1
            if "reliability" in recommendation_lower or "error" in recommendation_lower:
                improvements["reliability"] += 0.1
            if "scalability" in recommendation_lower or "capacity" in recommendation_lower:
                improvements["scalability"] += 0.1
        
        # Cap improvements at 1.0
        for key in improvements:
            improvements[key] = min(1.0, improvements[key])
        
        return improvements
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status across all systems"""
        
        return {
            "system_health": self._calculate_system_health(),
            "resource_efficiency": self._calculate_resource_efficiency(),
            "optimization_opportunities": self._count_optimization_opportunities(),
            "active_optimizations": len(self.optimization_reports),
            "last_analysis": self.optimization_reports[-1].timestamp if self.optimization_reports else None,
            "systems_status": {
                "model_evolution": "active" if self.model_evolution_tracker else "inactive",
                "synthesis_optimization": "active" if self.synthesis_optimizer else "inactive",
                "technology_trends": "active" if self.trend_detector else "inactive",
                "next_model_predictions": "active" if self.next_model_predictor else "inactive",
                "latent_space_optimization": "active" if self.latent_space_optimizer else "inactive",
                "multi_agent_coordination": "active" if self.multi_agent_coordinator else "inactive",
                "synthesis_speed": "active" if self.synthesis_speed_optimizer else "inactive",
                "blockchain_performance": "active" if self.blockchain_optimizer else "inactive",
                "model_performance": "active" if self.performance_registry else "inactive",
                "predictive_analysis": "active" if self.history_analyzer else "inactive"
            }
        }
    
    def _save_reports(self) -> None:
        """Save optimization reports to storage"""
        
        try:
            reports_data = [
                {
                    "report_id": report.report_id,
                    "timestamp": report.timestamp,
                    "optimization_summary": report.optimization_summary,
                    "performance_metrics": report.performance_metrics,
                    "recommendations": report.recommendations,
                    "implementation_priority": report.implementation_priority,
                    "expected_improvements": report.expected_improvements
                }
                for report in self.optimization_reports
            ]
            
            with open(self.reports_file, 'w') as f:
                json.dump(reports_data, f, indent=2)
            
            logger.debug("💾 Saved optimization reports to storage")
            
        except Exception as e:
            logger.error(f"Failed to save optimization reports: {e}")


# Helper functions for integration
def create_optimization_orchestrator(cfg: IceburgConfig) -> OptimizationOrchestrator:
    """Create optimization orchestrator instance"""
    return OptimizationOrchestrator(cfg)

def run_comprehensive_optimization_analysis(
    orchestrator: OptimizationOrchestrator
) -> OptimizationReport:
    """Run comprehensive optimization analysis"""
    return orchestrator.run_comprehensive_optimization_analysis()

def get_optimization_status(
    orchestrator: OptimizationOrchestrator
) -> Dict[str, Any]:
    """Get current optimization status"""
    return orchestrator.get_optimization_status()
