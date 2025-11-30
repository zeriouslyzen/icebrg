"""
AI Model Evolution Pattern Tracking System
Tracks AI model evolution patterns and predicts next model capabilities

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelCapability:
    """Represents a model capability metric"""
    name: str
    value: float
    unit: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class ModelEvolutionPattern:
    """Represents an evolution pattern in AI models"""
    pattern_id: str
    pattern_type: str  # "performance", "architecture", "capability", "efficiency"
    trend_direction: str  # "increasing", "decreasing", "stable", "cyclical"
    confidence: float
    time_series: List[Tuple[float, float]]  # (timestamp, value)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NextModelPrediction:
    """Prediction for next model capabilities"""
    model_name: str
    predicted_capabilities: Dict[str, ModelCapability]
    confidence: float
    timeframe: str  # "short_term", "medium_term", "long_term"
    optimization_recommendations: List[str]
    timestamp: float = field(default_factory=time.time)

class ModelEvolutionTracker:
    """
    Tracks AI model evolution patterns and predicts future capabilities
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/model_evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.patterns_file = self.data_dir / "evolution_patterns.json"
        self.predictions_file = self.data_dir / "next_model_predictions.json"
        self.performance_file = self.data_dir / "model_performance.json"
        
        # Data structures
        self.evolution_patterns: Dict[str, ModelEvolutionPattern] = {}
        self.next_model_predictions: List[NextModelPrediction] = []
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing data
        self._load_data()
        
        logger.info("🧬 Model Evolution Tracker initialized")
    
    def track_model_performance(
        self, 
        model_name: str, 
        capabilities: Dict[str, float],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Track performance metrics for a model"""
        if metadata is None:
            metadata = {}
        
        performance_record = {
            "timestamp": time.time(),
            "model_name": model_name,
            "capabilities": capabilities,
            "metadata": metadata
        }
        
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = []
        
        self.model_performance_history[model_name].append(performance_record)
        
        # Keep only last 1000 records per model
        if len(self.model_performance_history[model_name]) > 1000:
            self.model_performance_history[model_name] = self.model_performance_history[model_name][-1000:]
        
        # Analyze for evolution patterns
        self._analyze_evolution_patterns(model_name)
        
        # Save data
        self._save_data()
        
        logger.info(f"📊 Tracked performance for {model_name}: {len(capabilities)} capabilities")
    
    def _analyze_evolution_patterns(self, model_name: str) -> None:
        """Analyze evolution patterns for a specific model"""
        if model_name not in self.model_performance_history:
            return
        
        history = self.model_performance_history[model_name]
        if len(history) < 3:  # Need at least 3 data points
            return
        
        # Analyze each capability
        for capability_name in history[0]["capabilities"].keys():
            values = [record["capabilities"][capability_name] for record in history]
            timestamps = [record["timestamp"] for record in history]
            
            # Calculate trend
            trend = self._calculate_trend(values)
            
            # Create pattern if significant
            if abs(trend) > 0.1:  # 10% change threshold
                pattern_id = f"{model_name}_{capability_name}_{int(time.time())}"
                pattern = ModelEvolutionPattern(
                    pattern_id=pattern_id,
                    pattern_type="capability",
                    trend_direction="increasing" if trend > 0 else "decreasing",
                    confidence=min(1.0, abs(trend)),
                    time_series=list(zip(timestamps, values)),
                    metadata={
                        "model_name": model_name,
                        "capability_name": capability_name,
                        "trend_slope": trend
                    }
                )
                
                self.evolution_patterns[pattern_id] = pattern
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope
    
    def predict_next_model_capabilities(
        self, 
        model_name: str, 
        timeframe: str = "medium_term"
    ) -> NextModelPrediction:
        """Predict next model capabilities based on evolution patterns"""
        
        # Get relevant patterns for this model
        model_patterns = [
            pattern for pattern in self.evolution_patterns.values()
            if pattern.metadata.get("model_name") == model_name
        ]
        
        if not model_patterns:
            # No patterns available, return baseline prediction
            return self._create_baseline_prediction(model_name, timeframe)
        
        # Predict capabilities based on trends
        predicted_capabilities = {}
        optimization_recommendations = []
        
        for pattern in model_patterns:
            capability_name = pattern.metadata.get("capability_name")
            if not capability_name:
                continue
            
            # Extrapolate trend
            current_value = pattern.time_series[-1][1] if pattern.time_series else 0.0
            trend_slope = pattern.metadata.get("trend_slope", 0.0)
            
            # Predict future value based on timeframe
            time_multiplier = self._get_time_multiplier(timeframe)
            predicted_value = current_value + (trend_slope * time_multiplier)
            
            # Ensure reasonable bounds
            predicted_value = max(0.0, min(1.0, predicted_value))
            
            predicted_capabilities[capability_name] = ModelCapability(
                name=capability_name,
                value=predicted_value,
                unit="normalized",
                confidence=pattern.confidence
            )
            
            # Generate optimization recommendations
            if trend_slope > 0.1:
                optimization_recommendations.append(
                    f"Continue optimizing {capability_name} - strong positive trend detected"
                )
            elif trend_slope < -0.1:
                optimization_recommendations.append(
                    f"Address declining {capability_name} - negative trend detected"
                )
        
        # Calculate overall confidence
        confidence = np.mean([p.confidence for p in predicted_capabilities.values()]) if predicted_capabilities else 0.5
        
        prediction = NextModelPrediction(
            model_name=model_name,
            predicted_capabilities=predicted_capabilities,
            confidence=confidence,
            timeframe=timeframe,
            optimization_recommendations=optimization_recommendations
        )
        
        # Store prediction
        self.next_model_predictions.append(prediction)
        
        # Keep only last 100 predictions
        if len(self.next_model_predictions) > 100:
            self.next_model_predictions = self.next_model_predictions[-100:]
        
        self._save_data()
        
        logger.info(f"🔮 Generated prediction for {model_name}: {len(predicted_capabilities)} capabilities")
        
        return prediction
    
    def _get_time_multiplier(self, timeframe: str) -> float:
        """Get time multiplier for prediction extrapolation"""
        multipliers = {
            "short_term": 1.0,
            "medium_term": 3.0,
            "long_term": 6.0
        }
        return multipliers.get(timeframe, 3.0)
    
    def _create_baseline_prediction(self, model_name: str, timeframe: str) -> NextModelPrediction:
        """Create baseline prediction when no patterns are available"""
        baseline_capabilities = {
            "accuracy": ModelCapability("accuracy", 0.85, "normalized", 0.5),
            "speed": ModelCapability("speed", 0.7, "normalized", 0.5),
            "efficiency": ModelCapability("efficiency", 0.75, "normalized", 0.5),
            "reliability": ModelCapability("reliability", 0.8, "normalized", 0.5)
        }
        
        return NextModelPrediction(
            model_name=model_name,
            predicted_capabilities=baseline_capabilities,
            confidence=0.5,
            timeframe=timeframe,
            optimization_recommendations=[
                "Insufficient data for trend analysis",
                "Collect more performance metrics",
                "Monitor capability evolution over time"
            ]
        )
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution patterns and predictions"""
        return {
            "total_patterns": len(self.evolution_patterns),
            "total_predictions": len(self.next_model_predictions),
            "models_tracked": len(self.model_performance_history),
            "recent_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "trend": pattern.trend_direction,
                    "confidence": pattern.confidence
                }
                for pattern in list(self.evolution_patterns.values())[-10:]
            ],
            "recent_predictions": [
                {
                    "model_name": pred.model_name,
                    "timeframe": pred.timeframe,
                    "confidence": pred.confidence,
                    "capabilities_count": len(pred.predicted_capabilities)
                }
                for pred in self.next_model_predictions[-5:]
            ]
        }
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load evolution patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    self.evolution_patterns = {
                        pattern_id: ModelEvolutionPattern(**pattern_data)
                        for pattern_id, pattern_data in data.items()
                    }
            
            # Load predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.next_model_predictions = [
                        NextModelPrediction(**pred_data)
                        for pred_data in data
                    ]
            
            # Load performance history
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    self.model_performance_history = json.load(f)
            
            logger.info(f"📁 Loaded evolution data: {len(self.evolution_patterns)} patterns, {len(self.next_model_predictions)} predictions")
            
        except Exception as e:
            logger.warning(f"Failed to load evolution data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save evolution patterns
            patterns_data = {
                pattern_id: {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "trend_direction": pattern.trend_direction,
                    "confidence": pattern.confidence,
                    "time_series": pattern.time_series,
                    "metadata": pattern.metadata
                }
                for pattern_id, pattern in self.evolution_patterns.items()
            }
            
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save predictions
            predictions_data = [
                {
                    "model_name": pred.model_name,
                    "predicted_capabilities": {
                        name: {
                            "name": cap.name,
                            "value": cap.value,
                            "unit": cap.unit,
                            "confidence": cap.confidence,
                            "timestamp": cap.timestamp
                        }
                        for name, cap in pred.predicted_capabilities.items()
                    },
                    "confidence": pred.confidence,
                    "timeframe": pred.timeframe,
                    "optimization_recommendations": pred.optimization_recommendations,
                    "timestamp": pred.timestamp
                }
                for pred in self.next_model_predictions
            ]
            
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            # Save performance history
            with open(self.performance_file, 'w') as f:
                json.dump(self.model_performance_history, f, indent=2)
            
            logger.debug("💾 Saved evolution data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save evolution data: {e}")


# Helper functions for integration
def create_model_evolution_tracker(cfg: IceburgConfig) -> ModelEvolutionTracker:
    """Create model evolution tracker instance"""
    return ModelEvolutionTracker(cfg)

def track_model_performance(
    tracker: ModelEvolutionTracker,
    model_name: str,
    capabilities: Dict[str, float],
    metadata: Dict[str, Any] = None
) -> None:
    """Track model performance"""
    tracker.track_model_performance(model_name, capabilities, metadata)

def predict_next_model(
    tracker: ModelEvolutionTracker,
    model_name: str,
    timeframe: str = "medium_term"
) -> NextModelPrediction:
    """Predict next model capabilities"""
    return tracker.predict_next_model_capabilities(model_name, timeframe)
