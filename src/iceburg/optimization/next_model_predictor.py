"""
Next Model Capability Prediction Framework
Predicts future AI model capabilities and optimization opportunities

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
from collections import defaultdict

from ..config import IceburgConfig
from .model_evolution_tracker import ModelEvolutionTracker, NextModelPrediction
from .cross_domain_synthesis_optimizer import CrossDomainSynthesisOptimizer
from .technology_trend_detector import TechnologyTrendDetector

logger = logging.getLogger(__name__)

@dataclass
class CapabilityPrediction:
    """Prediction for a specific model capability"""
    capability_name: str
    current_value: float
    predicted_value: float
    improvement_factor: float
    confidence: float
    timeframe: str
    optimization_potential: str  # "low", "medium", "high", "breakthrough"

@dataclass
class ArchitecturePrediction:
    """Prediction for model architecture evolution"""
    architecture_type: str
    current_architecture: str
    predicted_architecture: str
    key_changes: List[str]
    expected_benefits: List[str]
    implementation_complexity: str
    confidence: float

@dataclass
class OptimizationRoadmap:
    """Roadmap for model optimization"""
    roadmap_id: str
    model_name: str
    optimization_phases: List[Dict[str, Any]]
    expected_timeline: str
    resource_requirements: Dict[str, Any]
    success_metrics: Dict[str, float]
    risk_assessment: Dict[str, Any]

class NextModelPredictor:
    """
    Predicts future AI model capabilities and optimization opportunities
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/next_model_predictions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component systems
        self.evolution_tracker = ModelEvolutionTracker(cfg)
        self.synthesis_optimizer = CrossDomainSynthesisOptimizer(cfg)
        self.trend_detector = TechnologyTrendDetector(cfg)
        
        # Storage files
        self.predictions_file = self.data_dir / "capability_predictions.json"
        self.architectures_file = self.data_dir / "architecture_predictions.json"
        self.roadmaps_file = self.data_dir / "optimization_roadmaps.json"
        
        # Data structures
        self.capability_predictions: Dict[str, List[CapabilityPrediction]] = {}
        self.architecture_predictions: List[ArchitecturePrediction] = []
        self.optimization_roadmaps: Dict[str, OptimizationRoadmap] = {}
        
        # Prediction models
        self.capability_models: Dict[str, Any] = {}
        self.architecture_models: Dict[str, Any] = {}
        
        # Load existing data
        self._load_data()
        self._initialize_prediction_models()
        
        logger.info("🔮 Next Model Predictor initialized")
    
    def predict_model_capabilities(
        self,
        model_name: str,
        current_capabilities: Dict[str, float],
        timeframe: str = "medium_term"
    ) -> List[CapabilityPrediction]:
        """Predict future capabilities for a specific model"""
        
        predictions = []
        
        for capability_name, current_value in current_capabilities.items():
            # Get evolution pattern for this capability
            evolution_pattern = self._get_evolution_pattern(model_name, capability_name)
            
            # Get technology trend influence
            trend_influence = self._get_trend_influence(capability_name)
            
            # Calculate predicted value
            predicted_value = self._calculate_predicted_value(
                current_value, evolution_pattern, trend_influence, timeframe
            )
            
            # Calculate improvement factor
            improvement_factor = predicted_value / current_value if current_value > 0 else 1.0
            
            # Determine optimization potential
            optimization_potential = self._assess_optimization_potential(
                improvement_factor, trend_influence
            )
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                evolution_pattern, trend_influence, capability_name
            )
            
            prediction = CapabilityPrediction(
                capability_name=capability_name,
                current_value=current_value,
                predicted_value=predicted_value,
                improvement_factor=improvement_factor,
                confidence=confidence,
                timeframe=timeframe,
                optimization_potential=optimization_potential
            )
            
            predictions.append(prediction)
        
        # Store predictions
        if model_name not in self.capability_predictions:
            self.capability_predictions[model_name] = []
        
        self.capability_predictions[model_name].extend(predictions)
        
        # Keep only last 50 predictions per model
        if len(self.capability_predictions[model_name]) > 50:
            self.capability_predictions[model_name] = self.capability_predictions[model_name][-50:]
        
        self._save_data()
        
        logger.info(f"🔮 Generated {len(predictions)} capability predictions for {model_name}")
        
        return predictions
    
    def predict_architecture_evolution(
        self,
        current_architecture: str,
        model_type: str = "general_purpose"
    ) -> ArchitecturePrediction:
        """Predict architecture evolution for AI models"""
        
        # Get technology trends relevant to architecture
        architecture_trends = self._get_architecture_trends()
        
        # Predict next architecture based on trends
        predicted_architecture = self._predict_next_architecture(
            current_architecture, architecture_trends, model_type
        )
        
        # Identify key changes
        key_changes = self._identify_architecture_changes(
            current_architecture, predicted_architecture
        )
        
        # Determine expected benefits
        expected_benefits = self._calculate_architecture_benefits(
            predicted_architecture, architecture_trends
        )
        
        # Assess implementation complexity
        implementation_complexity = self._assess_implementation_complexity(
            current_architecture, predicted_architecture
        )
        
        # Calculate confidence
        confidence = self._calculate_architecture_confidence(
            architecture_trends, model_type
        )
        
        prediction = ArchitecturePrediction(
            architecture_type=model_type,
            current_architecture=current_architecture,
            predicted_architecture=predicted_architecture,
            key_changes=key_changes,
            expected_benefits=expected_benefits,
            implementation_complexity=implementation_complexity,
            confidence=confidence
        )
        
        self.architecture_predictions.append(prediction)
        
        # Keep only last 20 architecture predictions
        if len(self.architecture_predictions) > 20:
            self.architecture_predictions = self.architecture_predictions[-20:]
        
        self._save_data()
        
        logger.info(f"🏗️ Generated architecture prediction: {current_architecture} -> {predicted_architecture}")
        
        return prediction
    
    def create_optimization_roadmap(
        self,
        model_name: str,
        target_capabilities: Dict[str, float],
        current_capabilities: Dict[str, float]
    ) -> OptimizationRoadmap:
        """Create optimization roadmap for achieving target capabilities"""
        
        # Analyze capability gaps
        capability_gaps = self._analyze_capability_gaps(
            current_capabilities, target_capabilities
        )
        
        # Create optimization phases
        optimization_phases = self._create_optimization_phases(
            capability_gaps, model_name
        )
        
        # Estimate timeline
        expected_timeline = self._estimate_optimization_timeline(optimization_phases)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(optimization_phases)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(target_capabilities)
        
        # Assess risks
        risk_assessment = self._assess_optimization_risks(
            optimization_phases, model_name
        )
        
        roadmap_id = f"roadmap_{model_name}_{int(time.time())}"
        roadmap = OptimizationRoadmap(
            roadmap_id=roadmap_id,
            model_name=model_name,
            optimization_phases=optimization_phases,
            expected_timeline=expected_timeline,
            resource_requirements=resource_requirements,
            success_metrics=success_metrics,
            risk_assessment=risk_assessment
        )
        
        self.optimization_roadmaps[roadmap_id] = roadmap
        
        self._save_data()
        
        logger.info(f"🗺️ Created optimization roadmap for {model_name}: {len(optimization_phases)} phases")
        
        return roadmap
    
    def _get_evolution_pattern(self, model_name: str, capability_name: str) -> Dict[str, Any]:
        """Get evolution pattern for a specific model and capability"""
        # Get evolution summary from tracker
        evolution_summary = self.evolution_tracker.get_evolution_summary()
        
        # Look for relevant patterns
        for pattern in evolution_summary.get("recent_patterns", []):
            if capability_name in pattern.get("pattern_id", ""):
                return {
                    "trend_direction": pattern.get("trend", "stable"),
                    "confidence": pattern.get("confidence", 0.5),
                    "pattern_type": pattern.get("type", "unknown")
                }
        
        return {"trend_direction": "stable", "confidence": 0.5, "pattern_type": "unknown"}
    
    def _get_trend_influence(self, capability_name: str) -> Dict[str, Any]:
        """Get technology trend influence on a capability"""
        # Get emerging technologies
        emerging_techs = self.trend_detector.get_emerging_technologies(min_confidence=0.6)
        
        # Map capabilities to relevant technologies
        capability_tech_mapping = {
            "accuracy": ["artificial_intelligence", "machine_learning"],
            "speed": ["quantum_computing", "hardware_acceleration"],
            "efficiency": ["optimization_algorithms", "energy_efficiency"],
            "reliability": ["robust_ai", "safety_systems"],
            "creativity": ["generative_ai", "creative_algorithms"],
            "reasoning": ["logical_ai", "causal_reasoning"]
        }
        
        relevant_techs = capability_tech_mapping.get(capability_name, [])
        influence_score = 0.0
        
        for tech in emerging_techs:
            if any(relevant_tech in tech.technology.lower() for relevant_tech in relevant_techs):
                influence_score += tech.confidence * tech.growth_rate
        
        return {
            "influence_score": min(1.0, influence_score),
            "relevant_technologies": relevant_techs,
            "trend_count": len(emerging_techs)
        }
    
    def _calculate_predicted_value(
        self,
        current_value: float,
        evolution_pattern: Dict[str, Any],
        trend_influence: Dict[str, Any],
        timeframe: str
    ) -> float:
        """Calculate predicted value based on evolution pattern and trend influence"""
        
        # Base prediction from evolution pattern
        trend_direction = evolution_pattern.get("trend_direction", "stable")
        trend_confidence = evolution_pattern.get("confidence", 0.5)
        
        # Calculate trend multiplier
        if trend_direction == "increasing":
            trend_multiplier = 1.0 + (trend_confidence * 0.3)
        elif trend_direction == "decreasing":
            trend_multiplier = 1.0 - (trend_confidence * 0.2)
        else:
            trend_multiplier = 1.0
        
        # Apply trend influence
        influence_score = trend_influence.get("influence_score", 0.0)
        influence_multiplier = 1.0 + (influence_score * 0.2)
        
        # Apply timeframe scaling
        timeframe_multipliers = {
            "short_term": 1.1,
            "medium_term": 1.3,
            "long_term": 1.6
        }
        timeframe_multiplier = timeframe_multipliers.get(timeframe, 1.3)
        
        # Calculate predicted value
        predicted_value = current_value * trend_multiplier * influence_multiplier * timeframe_multiplier
        
        # Ensure reasonable bounds
        return max(0.0, min(1.0, predicted_value))
    
    def _assess_optimization_potential(
        self,
        improvement_factor: float,
        trend_influence: Dict[str, Any]
    ) -> str:
        """Assess optimization potential based on improvement factor and trends"""
        
        influence_score = trend_influence.get("influence_score", 0.0)
        
        if improvement_factor > 2.0 and influence_score > 0.7:
            return "breakthrough"
        elif improvement_factor > 1.5 and influence_score > 0.5:
            return "high"
        elif improvement_factor > 1.2 and influence_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _calculate_prediction_confidence(
        self,
        evolution_pattern: Dict[str, Any],
        trend_influence: Dict[str, Any],
        capability_name: str
    ) -> float:
        """Calculate confidence in prediction"""
        
        pattern_confidence = evolution_pattern.get("confidence", 0.5)
        influence_score = trend_influence.get("influence_score", 0.0)
        
        # Base confidence from pattern
        base_confidence = pattern_confidence
        
        # Boost confidence if strong trend influence
        if influence_score > 0.6:
            base_confidence += 0.2
        
        # Capability-specific confidence adjustments
        capability_confidence_boost = {
            "accuracy": 0.1,
            "speed": 0.15,
            "efficiency": 0.1,
            "reliability": 0.05,
            "creativity": 0.2,
            "reasoning": 0.15
        }
        
        base_confidence += capability_confidence_boost.get(capability_name, 0.0)
        
        return min(1.0, max(0.0, base_confidence))
    
    def _get_architecture_trends(self) -> List[Dict[str, Any]]:
        """Get technology trends relevant to architecture evolution"""
        emerging_techs = self.trend_detector.get_emerging_technologies(min_confidence=0.5)
        
        architecture_relevant = []
        for tech in emerging_techs:
            if any(keyword in tech.technology.lower() for keyword in [
                "transformer", "attention", "neural", "quantum", "spiking", "memristor"
            ]):
                architecture_relevant.append({
                    "technology": tech.technology,
                    "trend_type": tech.trend_type.value,
                    "confidence": tech.confidence,
                    "growth_rate": tech.growth_rate
                })
        
        return architecture_relevant
    
    def _predict_next_architecture(
        self,
        current_architecture: str,
        trends: List[Dict[str, Any]],
        model_type: str
    ) -> str:
        """Predict next architecture based on current architecture and trends"""
        
        # Architecture evolution patterns
        evolution_paths = {
            "transformer": "sparse_transformer",
            "sparse_transformer": "mixture_of_experts",
            "mixture_of_experts": "neural_architecture_search",
            "neural_architecture_search": "quantum_neural_networks",
            "cnn": "vision_transformer",
            "vision_transformer": "multimodal_transformer",
            "rnn": "transformer",
            "lstm": "transformer"
        }
        
        # Check for direct evolution path
        if current_architecture in evolution_paths:
            predicted = evolution_paths[current_architecture]
        else:
            # Predict based on trends
            if any("quantum" in trend["technology"].lower() for trend in trends):
                predicted = "quantum_enhanced_transformer"
            elif any("spiking" in trend["technology"].lower() for trend in trends):
                predicted = "spiking_neural_network"
            elif any("attention" in trend["technology"].lower() for trend in trends):
                predicted = "efficient_attention_transformer"
            else:
                predicted = "enhanced_transformer"
        
        return predicted
    
    def _identify_architecture_changes(
        self,
        current_architecture: str,
        predicted_architecture: str
    ) -> List[str]:
        """Identify key changes between current and predicted architecture"""
        
        changes = []
        
        if "transformer" in current_architecture and "quantum" in predicted_architecture:
            changes.extend([
                "Add quantum processing units",
                "Implement quantum attention mechanisms",
                "Integrate quantum error correction"
            ])
        elif "transformer" in current_architecture and "sparse" in predicted_architecture:
            changes.extend([
                "Implement sparse attention patterns",
                "Add dynamic routing mechanisms",
                "Optimize memory usage"
            ])
        elif "transformer" in current_architecture and "mixture" in predicted_architecture:
            changes.extend([
                "Add expert routing layers",
                "Implement gating mechanisms",
                "Scale to multiple experts"
            ])
        else:
            changes.append("General architecture enhancement")
        
        return changes
    
    def _calculate_architecture_benefits(
        self,
        predicted_architecture: str,
        trends: List[Dict[str, Any]]
    ) -> List[str]:
        """Calculate expected benefits of predicted architecture"""
        
        benefits = []
        
        if "quantum" in predicted_architecture:
            benefits.extend([
                "Exponential speedup for specific tasks",
                "Enhanced parallel processing",
                "Improved optimization capabilities"
            ])
        elif "sparse" in predicted_architecture:
            benefits.extend([
                "Reduced computational complexity",
                "Better memory efficiency",
                "Faster inference times"
            ])
        elif "mixture" in predicted_architecture:
            benefits.extend([
                "Specialized expert processing",
                "Improved task-specific performance",
                "Better resource utilization"
            ])
        else:
            benefits.append("General performance improvements")
        
        return benefits
    
    def _assess_implementation_complexity(
        self,
        current_architecture: str,
        predicted_architecture: str
    ) -> str:
        """Assess implementation complexity of architecture change"""
        
        if "quantum" in predicted_architecture:
            return "high"
        elif "mixture" in predicted_architecture:
            return "medium"
        elif "sparse" in predicted_architecture:
            return "low"
        else:
            return "medium"
    
    def _calculate_architecture_confidence(
        self,
        trends: List[Dict[str, Any]],
        model_type: str
    ) -> float:
        """Calculate confidence in architecture prediction"""
        
        if not trends:
            return 0.5
        
        # Average confidence of relevant trends
        avg_confidence = np.mean([trend["confidence"] for trend in trends])
        
        # Boost confidence for general purpose models
        if model_type == "general_purpose":
            avg_confidence += 0.1
        
        return min(1.0, avg_confidence)
    
    def _analyze_capability_gaps(
        self,
        current_capabilities: Dict[str, float],
        target_capabilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze gaps between current and target capabilities"""
        
        gaps = {}
        for capability, target_value in target_capabilities.items():
            current_value = current_capabilities.get(capability, 0.0)
            gap = target_value - current_value
            if gap > 0:
                gaps[capability] = gap
        
        return gaps
    
    def _create_optimization_phases(
        self,
        capability_gaps: Dict[str, float],
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Create optimization phases to address capability gaps"""
        
        phases = []
        
        # Sort gaps by size (largest first)
        sorted_gaps = sorted(capability_gaps.items(), key=lambda x: x[1], reverse=True)
        
        for i, (capability, gap) in enumerate(sorted_gaps):
            phase = {
                "phase_id": f"phase_{i+1}",
                "capability": capability,
                "target_improvement": gap,
                "optimization_techniques": self._get_optimization_techniques(capability),
                "estimated_duration": self._estimate_phase_duration(capability, gap),
                "success_criteria": {
                    "min_improvement": gap * 0.8,
                    "target_improvement": gap
                }
            }
            phases.append(phase)
        
        return phases
    
    def _get_optimization_techniques(self, capability: str) -> List[str]:
        """Get optimization techniques for a specific capability"""
        
        techniques = {
            "accuracy": [
                "Data augmentation",
                "Ensemble methods",
                "Advanced regularization",
                "Hyperparameter optimization"
            ],
            "speed": [
                "Model quantization",
                "Pruning techniques",
                "Hardware acceleration",
                "Parallel processing"
            ],
            "efficiency": [
                "Architecture optimization",
                "Memory optimization",
                "Energy-efficient algorithms",
                "Compression techniques"
            ],
            "reliability": [
                "Robust training",
                "Uncertainty quantification",
                "Error detection",
                "Fallback mechanisms"
            ]
        }
        
        return techniques.get(capability, ["General optimization"])
    
    def _estimate_phase_duration(self, capability: str, gap: float) -> str:
        """Estimate duration for optimization phase"""
        
        if gap > 0.5:
            return "3-6 months"
        elif gap > 0.3:
            return "2-4 months"
        elif gap > 0.1:
            return "1-2 months"
        else:
            return "2-4 weeks"
    
    def _estimate_optimization_timeline(self, phases: List[Dict[str, Any]]) -> str:
        """Estimate total timeline for optimization"""
        
        total_months = 0
        for phase in phases:
            duration = phase["estimated_duration"]
            if "3-6" in duration:
                total_months += 4.5
            elif "2-4" in duration:
                total_months += 3
            elif "1-2" in duration:
                total_months += 1.5
            else:
                total_months += 0.5
        
        if total_months > 12:
            return f"{int(total_months/12)}-{int(total_months/12)+1} years"
        else:
            return f"{int(total_months)}-{int(total_months)+2} months"
    
    def _calculate_resource_requirements(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for optimization"""
        
        return {
            "computing_power": "High (GPU clusters recommended)",
            "data_requirements": "Large datasets for training",
            "expertise_level": "Senior AI/ML engineers",
            "estimated_cost": "Medium to High",
            "infrastructure": "Cloud computing or dedicated hardware"
        }
    
    def _define_success_metrics(self, target_capabilities: Dict[str, float]) -> Dict[str, float]:
        """Define success metrics for optimization"""
        
        return {
            "accuracy_target": target_capabilities.get("accuracy", 0.9),
            "speed_target": target_capabilities.get("speed", 0.8),
            "efficiency_target": target_capabilities.get("efficiency", 0.85),
            "reliability_target": target_capabilities.get("reliability", 0.95)
        }
    
    def _assess_optimization_risks(
        self,
        phases: List[Dict[str, Any]],
        model_name: str
    ) -> Dict[str, Any]:
        """Assess risks associated with optimization"""
        
        return {
            "technical_risks": [
                "Model instability during optimization",
                "Performance degradation in other areas",
                "Compatibility issues with existing systems"
            ],
            "resource_risks": [
                "Insufficient computing resources",
                "Budget overruns",
                "Timeline delays"
            ],
            "mitigation_strategies": [
                "Incremental optimization approach",
                "Comprehensive testing at each phase",
                "Fallback to previous versions if needed"
            ]
        }
    
    def _initialize_prediction_models(self) -> None:
        """Initialize prediction models for capabilities and architectures"""
        
        # Initialize capability prediction models
        self.capability_models = {
            "accuracy": {"base_improvement": 0.15, "trend_influence": 0.2},
            "speed": {"base_improvement": 0.25, "trend_influence": 0.3},
            "efficiency": {"base_improvement": 0.20, "trend_influence": 0.25},
            "reliability": {"base_improvement": 0.10, "trend_influence": 0.15}
        }
        
        # Initialize architecture prediction models
        self.architecture_models = {
            "transformer": {"next_arch": "sparse_transformer", "confidence": 0.8},
            "cnn": {"next_arch": "vision_transformer", "confidence": 0.7},
            "rnn": {"next_arch": "transformer", "confidence": 0.9}
        }
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions and roadmaps"""
        
        return {
            "capability_predictions": {
                model_name: len(predictions)
                for model_name, predictions in self.capability_predictions.items()
            },
            "architecture_predictions": len(self.architecture_predictions),
            "optimization_roadmaps": len(self.optimization_roadmaps),
            "recent_capability_predictions": [
                {
                    "model_name": model_name,
                    "capability": pred.capability_name,
                    "improvement_factor": pred.improvement_factor,
                    "optimization_potential": pred.optimization_potential,
                    "confidence": pred.confidence
                }
                for model_name, predictions in self.capability_predictions.items()
                for pred in predictions[-5:]  # Last 5 predictions per model
            ],
            "recent_architecture_predictions": [
                {
                    "current": pred.current_architecture,
                    "predicted": pred.predicted_architecture,
                    "complexity": pred.implementation_complexity,
                    "confidence": pred.confidence
                }
                for pred in self.architecture_predictions[-5:]
            ]
        }
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load capability predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.capability_predictions = {
                        model_name: [
                            CapabilityPrediction(**pred_data)
                            for pred_data in predictions
                        ]
                        for model_name, predictions in data.items()
                    }
            
            # Load architecture predictions
            if self.architectures_file.exists():
                with open(self.architectures_file, 'r') as f:
                    data = json.load(f)
                    self.architecture_predictions = [
                        ArchitecturePrediction(**arch_data)
                        for arch_data in data
                    ]
            
            # Load optimization roadmaps
            if self.roadmaps_file.exists():
                with open(self.roadmaps_file, 'r') as f:
                    data = json.load(f)
                    self.optimization_roadmaps = {
                        roadmap_id: OptimizationRoadmap(**roadmap_data)
                        for roadmap_id, roadmap_data in data.items()
                    }
            
            logger.info(f"📁 Loaded prediction data: {len(self.capability_predictions)} models, {len(self.architecture_predictions)} architectures")
            
        except Exception as e:
            logger.warning(f"Failed to load prediction data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save capability predictions
            predictions_data = {
                model_name: [
                    {
                        "capability_name": pred.capability_name,
                        "current_value": pred.current_value,
                        "predicted_value": pred.predicted_value,
                        "improvement_factor": pred.improvement_factor,
                        "confidence": pred.confidence,
                        "timeframe": pred.timeframe,
                        "optimization_potential": pred.optimization_potential
                    }
                    for pred in predictions
                ]
                for model_name, predictions in self.capability_predictions.items()
            }
            
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            # Save architecture predictions
            architectures_data = [
                {
                    "architecture_type": pred.architecture_type,
                    "current_architecture": pred.current_architecture,
                    "predicted_architecture": pred.predicted_architecture,
                    "key_changes": pred.key_changes,
                    "expected_benefits": pred.expected_benefits,
                    "implementation_complexity": pred.implementation_complexity,
                    "confidence": pred.confidence
                }
                for pred in self.architecture_predictions
            ]
            
            with open(self.architectures_file, 'w') as f:
                json.dump(architectures_data, f, indent=2)
            
            # Save optimization roadmaps
            roadmaps_data = {
                roadmap_id: {
                    "roadmap_id": roadmap.roadmap_id,
                    "model_name": roadmap.model_name,
                    "optimization_phases": roadmap.optimization_phases,
                    "expected_timeline": roadmap.expected_timeline,
                    "resource_requirements": roadmap.resource_requirements,
                    "success_metrics": roadmap.success_metrics,
                    "risk_assessment": roadmap.risk_assessment
                }
                for roadmap_id, roadmap in self.optimization_roadmaps.items()
            }
            
            with open(self.roadmaps_file, 'w') as f:
                json.dump(roadmaps_data, f, indent=2)
            
            logger.debug("💾 Saved prediction data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save prediction data: {e}")


# Helper functions for integration
def create_next_model_predictor(cfg: IceburgConfig) -> NextModelPredictor:
    """Create next model predictor instance"""
    return NextModelPredictor(cfg)

def predict_model_capabilities(
    predictor: NextModelPredictor,
    model_name: str,
    current_capabilities: Dict[str, float],
    timeframe: str = "medium_term"
) -> List[CapabilityPrediction]:
    """Predict future capabilities for a model"""
    return predictor.predict_model_capabilities(model_name, current_capabilities, timeframe)

def create_optimization_roadmap(
    predictor: NextModelPredictor,
    model_name: str,
    target_capabilities: Dict[str, float],
    current_capabilities: Dict[str, float]
) -> OptimizationRoadmap:
    """Create optimization roadmap for a model"""
    return predictor.create_optimization_roadmap(model_name, target_capabilities, current_capabilities)
