"""
Predictive History Analyzer
Analyzes historical data to predict future performance and optimization opportunities

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
import threading
from concurrent.futures import ThreadPoolExecutor
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Placeholder classes
    class LinearRegression:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return []
    class StandardScaler:
        def __init__(self, *args, **kwargs):
            pass
        def fit_transform(self, *args, **kwargs):
            return []
    def mean_squared_error(*args, **kwargs):
        return 0.0
    def r2_score(*args, **kwargs):
        return 0.0

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class HistoricalDataPoint:
    """Single data point in historical analysis"""
    timestamp: float
    metric_name: str
    metric_value: float
    context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Result of predictive analysis"""
    prediction_id: str
    metric_name: str
    predicted_value: float
    confidence: float
    prediction_horizon: str  # "short_term", "medium_term", "long_term"
    trend_direction: str  # "increasing", "decreasing", "stable"
    key_factors: List[str]
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationOpportunity:
    """Identified optimization opportunity"""
    opportunity_id: str
    opportunity_type: str  # "performance", "efficiency", "scalability", "reliability"
    current_value: float
    predicted_value: float
    improvement_potential: float
    implementation_effort: str
    priority: str  # "low", "medium", "high", "critical"
    description: str
    recommendations: List[str]

class PredictiveHistoryAnalyzer:
    """
    Analyzes historical data to predict future performance and identify optimization opportunities
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/predictive_analysis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.history_file = self.data_dir / "historical_data.json"
        self.predictions_file = self.data_dir / "predictions.json"
        self.opportunities_file = self.data_dir / "optimization_opportunities.json"
        
        # Data structures
        self.historical_data: List[HistoricalDataPoint] = []
        self.predictions: List[PredictionResult] = []
        self.optimization_opportunities: List[OptimizationOpportunity] = []
        
        # Analysis models
        self.prediction_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Performance tracking
        self.analysis_history: List[Dict[str, Any]] = []
        self.model_accuracy: Dict[str, float] = {}
        
        # Threading for concurrent analysis
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)
        self.analysis_lock = threading.Lock()
        
        # Load existing data
        self._load_data()
        self._initialize_prediction_models()
        
        logger.info("🔮 Predictive History Analyzer initialized")
    
    def add_historical_data(
        self,
        metric_name: str,
        metric_value: float,
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Add historical data point for analysis"""
        
        if context is None:
            context = {}
        if metadata is None:
            metadata = {}
        
        data_point = HistoricalDataPoint(
            timestamp=time.time(),
            metric_name=metric_name,
            metric_value=metric_value,
            context=context,
            metadata=metadata
        )
        
        with self.analysis_lock:
            self.historical_data.append(data_point)
            
            # Keep only last 50000 data points
            if len(self.historical_data) > 50000:
                self.historical_data = self.historical_data[-50000:]
        
        # Trigger analysis if enough data points
        if len([d for d in self.historical_data if d.metric_name == metric_name]) >= 10:
            self.analysis_executor.submit(self._analyze_metric_trends, metric_name)
        
        # Save data
        self._save_data()
        
        logger.debug(f"📊 Added historical data: {metric_name} = {metric_value}")
    
    def _analyze_metric_trends(self, metric_name: str) -> None:
        """Analyze trends for a specific metric"""
        
        # Get data for this metric
        metric_data = [d for d in self.historical_data if d.metric_name == metric_name]
        
        if len(metric_data) < 10:
            return
        
        # Sort by timestamp
        metric_data.sort(key=lambda x: x.timestamp)
        
        # Prepare data for analysis
        timestamps = np.array([d.timestamp for d in metric_data])
        values = np.array([d.metric_value for d in metric_data])
        
        # Normalize timestamps
        timestamps_normalized = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        
        # Fit linear regression model
        X = timestamps_normalized.reshape(-1, 1)
        y = values
        
        # Scale features
        if metric_name not in self.scalers:
            self.scalers[metric_name] = StandardScaler()
        
        scaler = self.scalers[metric_name]
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate model accuracy
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        self.model_accuracy[metric_name] = r2
        
        # Store model
        self.prediction_models[metric_name] = {
            "model": model,
            "scaler": scaler,
            "timestamps": timestamps,
            "values": values,
            "mse": mse,
            "r2": r2
        }
        
        # Generate predictions
        self._generate_predictions(metric_name, model, scaler, timestamps, values)
        
        # Identify optimization opportunities
        self._identify_optimization_opportunities(metric_name, values, model, scaler)
        
        logger.info(f"📈 Analyzed trends for {metric_name}: R² = {r2:.3f}, MSE = {mse:.3f}")
    
    def _generate_predictions(
        self,
        metric_name: str,
        model: LinearRegression,
        scaler: StandardScaler,
        timestamps: np.ndarray,
        values: np.ndarray
    ) -> None:
        """Generate predictions for different time horizons"""
        
        # Define prediction horizons
        horizons = {
            "short_term": 0.1,  # 10% of historical range
            "medium_term": 0.3,  # 30% of historical range
            "long_term": 0.5   # 50% of historical range
        }
        
        # Get current time
        current_time = timestamps[-1]
        time_range = timestamps[-1] - timestamps[0]
        
        for horizon_name, horizon_factor in horizons.items():
            # Calculate future timestamp
            future_time = current_time + (time_range * horizon_factor)
            future_time_normalized = (future_time - timestamps[0]) / (timestamps[-1] - timestamps[0])
            
            # Make prediction
            X_future = np.array([[future_time_normalized]])
            X_future_scaled = scaler.transform(X_future)
            predicted_value = model.predict(X_future_scaled)[0]
            
            # Calculate confidence based on model accuracy
            confidence = max(0.0, min(1.0, self.model_accuracy.get(metric_name, 0.5)))
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(model, scaler, timestamps, values)
            
            # Identify key factors
            key_factors = self._identify_key_factors(metric_name, values, timestamps)
            
            # Create prediction result
            prediction_id = f"pred_{metric_name}_{horizon_name}_{int(time.time())}"
            prediction = PredictionResult(
                prediction_id=prediction_id,
                metric_name=metric_name,
                predicted_value=predicted_value,
                confidence=confidence,
                prediction_horizon=horizon_name,
                trend_direction=trend_direction,
                key_factors=key_factors
            )
            
            self.predictions.append(prediction)
        
        # Keep only last 1000 predictions
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
    
    def _determine_trend_direction(
        self,
        model: LinearRegression,
        scaler: StandardScaler,
        timestamps: np.ndarray,
        values: np.ndarray
    ) -> str:
        """Determine trend direction based on model slope"""
        
        # Get model slope
        slope = model.coef_[0]
        
        # Normalize slope by value range
        value_range = np.max(values) - np.min(values)
        normalized_slope = slope / max(0.001, value_range)
        
        if normalized_slope > 0.1:
            return "increasing"
        elif normalized_slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_key_factors(
        self,
        metric_name: str,
        values: np.ndarray,
        timestamps: np.ndarray
    ) -> List[str]:
        """Identify key factors influencing the metric"""
        
        factors = []
        
        # Analyze value patterns
        value_std = np.std(values)
        value_mean = np.mean(values)
        
        if value_std / max(0.001, value_mean) > 0.3:
            factors.append("high_variability")
        
        # Analyze time patterns
        time_intervals = np.diff(timestamps)
        if len(time_intervals) > 0:
            avg_interval = np.mean(time_intervals)
            if avg_interval > 3600:  # More than 1 hour
                factors.append("infrequent_updates")
            elif avg_interval < 60:  # Less than 1 minute
                factors.append("frequent_updates")
        
        # Analyze trend patterns
        if len(values) > 5:
            recent_trend = np.mean(values[-5:]) - np.mean(values[:5])
            if recent_trend > 0:
                factors.append("recent_improvement")
            elif recent_trend < 0:
                factors.append("recent_degradation")
        
        # Metric-specific factors
        if "execution_time" in metric_name:
            factors.extend(["computational_complexity", "resource_availability"])
        elif "success_rate" in metric_name:
            factors.extend(["system_reliability", "error_handling"])
        elif "quality_score" in metric_name:
            factors.extend(["model_performance", "data_quality"])
        
        return factors[:5]  # Return top 5 factors
    
    def _identify_optimization_opportunities(
        self,
        metric_name: str,
        values: np.ndarray,
        model: LinearRegression,
        scaler: StandardScaler
    ) -> None:
        """Identify optimization opportunities based on analysis"""
        
        current_value = values[-1]
        predicted_value = model.predict(scaler.transform([[1.0]]))[0]  # Predict at end of range
        
        # Calculate improvement potential
        if "execution_time" in metric_name or "processing_time" in metric_name:
            # For time metrics, lower is better
            improvement_potential = (current_value - predicted_value) / max(0.001, current_value)
            if improvement_potential > 0.1:  # 10% improvement potential
                self._create_optimization_opportunity(
                    metric_name, current_value, predicted_value, improvement_potential,
                    "performance", "Reduce processing time through optimization"
                )
        
        elif "success_rate" in metric_name or "quality_score" in metric_name:
            # For quality metrics, higher is better
            improvement_potential = (predicted_value - current_value) / max(0.001, current_value)
            if improvement_potential > 0.1:  # 10% improvement potential
                self._create_optimization_opportunity(
                    metric_name, current_value, predicted_value, improvement_potential,
                    "quality", "Improve success rate or quality through optimization"
                )
        
        elif "resource_usage" in metric_name or "efficiency" in metric_name:
            # For resource metrics, lower is better
            improvement_potential = (current_value - predicted_value) / max(0.001, current_value)
            if improvement_potential > 0.1:  # 10% improvement potential
                self._create_optimization_opportunity(
                    metric_name, current_value, predicted_value, improvement_potential,
                    "efficiency", "Improve resource efficiency through optimization"
                )
    
    def _create_optimization_opportunity(
        self,
        metric_name: str,
        current_value: float,
        predicted_value: float,
        improvement_potential: float,
        opportunity_type: str,
        description: str
    ) -> None:
        """Create optimization opportunity"""
        
        opportunity_id = f"opp_{metric_name}_{opportunity_type}_{int(time.time())}"
        
        # Determine priority based on improvement potential
        if improvement_potential > 0.5:
            priority = "critical"
        elif improvement_potential > 0.3:
            priority = "high"
        elif improvement_potential > 0.1:
            priority = "medium"
        else:
            priority = "low"
        
        # Determine implementation effort
        if improvement_potential > 0.3:
            implementation_effort = "high"
        elif improvement_potential > 0.1:
            implementation_effort = "medium"
        else:
            implementation_effort = "low"
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            metric_name, opportunity_type, improvement_potential
        )
        
        opportunity = OptimizationOpportunity(
            opportunity_id=opportunity_id,
            opportunity_type=opportunity_type,
            current_value=current_value,
            predicted_value=predicted_value,
            improvement_potential=improvement_potential,
            implementation_effort=implementation_effort,
            priority=priority,
            description=description,
            recommendations=recommendations
        )
        
        self.optimization_opportunities.append(opportunity)
        
        # Keep only last 500 opportunities
        if len(self.optimization_opportunities) > 500:
            self.optimization_opportunities = self.optimization_opportunities[-500:]
    
    def _generate_optimization_recommendations(
        self,
        metric_name: str,
        opportunity_type: str,
        improvement_potential: float
    ) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if "execution_time" in metric_name or "processing_time" in metric_name:
            recommendations.extend([
                "Implement parallel processing",
                "Optimize algorithms for better time complexity",
                "Use caching to reduce redundant computations",
                "Consider hardware acceleration"
            ])
        
        elif "success_rate" in metric_name:
            recommendations.extend([
                "Improve error handling and recovery",
                "Add input validation and sanitization",
                "Implement retry mechanisms",
                "Enhance system monitoring and alerting"
            ])
        
        elif "quality_score" in metric_name:
            recommendations.extend([
                "Improve model training data quality",
                "Implement quality assurance processes",
                "Add validation and testing procedures",
                "Enhance feedback mechanisms"
            ])
        
        elif "resource_usage" in metric_name or "efficiency" in metric_name:
            recommendations.extend([
                "Optimize resource allocation",
                "Implement resource pooling",
                "Use more efficient data structures",
                "Consider resource scaling strategies"
            ])
        
        # Add general recommendations based on improvement potential
        if improvement_potential > 0.3:
            recommendations.append("Consider major architectural changes")
        elif improvement_potential > 0.1:
            recommendations.append("Implement incremental improvements")
        else:
            recommendations.append("Monitor and fine-tune existing processes")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def get_predictions(
        self,
        metric_name: str = None,
        prediction_horizon: str = None,
        min_confidence: float = 0.0
    ) -> List[PredictionResult]:
        """Get predictions based on criteria"""
        
        filtered_predictions = self.predictions
        
        if metric_name:
            filtered_predictions = [p for p in filtered_predictions if p.metric_name == metric_name]
        
        if prediction_horizon:
            filtered_predictions = [p for p in filtered_predictions if p.prediction_horizon == prediction_horizon]
        
        if min_confidence > 0.0:
            filtered_predictions = [p for p in filtered_predictions if p.confidence >= min_confidence]
        
        # Sort by confidence (highest first)
        filtered_predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_predictions
    
    def get_optimization_opportunities(
        self,
        opportunity_type: str = None,
        priority: str = None,
        min_improvement: float = 0.0
    ) -> List[OptimizationOpportunity]:
        """Get optimization opportunities based on criteria"""
        
        filtered_opportunities = self.optimization_opportunities
        
        if opportunity_type:
            filtered_opportunities = [o for o in filtered_opportunities if o.opportunity_type == opportunity_type]
        
        if priority:
            filtered_opportunities = [o for o in filtered_opportunities if o.priority == priority]
        
        if min_improvement > 0.0:
            filtered_opportunities = [o for o in filtered_opportunities if o.improvement_potential >= min_improvement]
        
        # Sort by improvement potential (highest first)
        filtered_opportunities.sort(key=lambda x: x.improvement_potential, reverse=True)
        
        return filtered_opportunities
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        
        # Calculate overall statistics
        total_data_points = len(self.historical_data)
        total_predictions = len(self.predictions)
        total_opportunities = len(self.optimization_opportunities)
        
        # Analyze model accuracy
        avg_model_accuracy = np.mean(list(self.model_accuracy.values())) if self.model_accuracy else 0.0
        
        # Analyze prediction confidence
        if self.predictions:
            avg_confidence = np.mean([p.confidence for p in self.predictions])
            high_confidence_predictions = len([p for p in self.predictions if p.confidence > 0.8])
        else:
            avg_confidence = 0.0
            high_confidence_predictions = 0
        
        # Analyze optimization opportunities
        if self.optimization_opportunities:
            avg_improvement_potential = np.mean([o.improvement_potential for o in self.optimization_opportunities])
            high_priority_opportunities = len([o for o in self.optimization_opportunities if o.priority in ["high", "critical"]])
        else:
            avg_improvement_potential = 0.0
            high_priority_opportunities = 0
        
        # Get top opportunities by type
        opportunities_by_type = defaultdict(list)
        for opportunity in self.optimization_opportunities:
            opportunities_by_type[opportunity.opportunity_type].append(opportunity)
        
        top_opportunities_by_type = {}
        for opp_type, opportunities in opportunities_by_type.items():
            if opportunities:
                top_opportunity = max(opportunities, key=lambda x: x.improvement_potential)
                top_opportunities_by_type[opp_type] = {
                    "improvement_potential": top_opportunity.improvement_potential,
                    "priority": top_opportunity.priority,
                    "description": top_opportunity.description
                }
        
        return {
            "analysis_statistics": {
                "total_data_points": total_data_points,
                "total_predictions": total_predictions,
                "total_opportunities": total_opportunities,
                "avg_model_accuracy": avg_model_accuracy
            },
            "prediction_quality": {
                "avg_confidence": avg_confidence,
                "high_confidence_predictions": high_confidence_predictions,
                "total_models": len(self.prediction_models)
            },
            "optimization_potential": {
                "avg_improvement_potential": avg_improvement_potential,
                "high_priority_opportunities": high_priority_opportunities,
                "top_opportunities_by_type": top_opportunities_by_type
            },
            "recommendations": self._generate_analysis_recommendations()
        }
    
    def _generate_analysis_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        if not self.historical_data:
            return ["Collect more historical data for analysis"]
        
        # Model accuracy recommendations
        if self.model_accuracy:
            low_accuracy_models = [name for name, acc in self.model_accuracy.items() if acc < 0.5]
            if low_accuracy_models:
                recommendations.append(f"Improve prediction models for: {', '.join(low_accuracy_models)}")
        
        # Prediction confidence recommendations
        if self.predictions:
            low_confidence_predictions = len([p for p in self.predictions if p.confidence < 0.6])
            if low_confidence_predictions > len(self.predictions) * 0.5:
                recommendations.append("Low prediction confidence detected - consider improving data quality")
        
        # Optimization opportunity recommendations
        if self.optimization_opportunities:
            critical_opportunities = len([o for o in self.optimization_opportunities if o.priority == "critical"])
            if critical_opportunities > 0:
                recommendations.append(f"Address {critical_opportunities} critical optimization opportunities")
        
        # Data quality recommendations
        if len(self.historical_data) < 100:
            recommendations.append("Collect more historical data for better predictions")
        
        return recommendations
    
    def _initialize_prediction_models(self) -> None:
        """Initialize prediction models for common metrics"""
        
        # Initialize models for common metrics
        common_metrics = [
            "execution_time", "success_rate", "quality_score", "resource_usage",
            "processing_time", "accuracy", "efficiency", "throughput"
        ]
        
        for metric in common_metrics:
            if metric not in self.prediction_models:
                self.prediction_models[metric] = None
                self.scalers[metric] = StandardScaler()
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load historical data
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.historical_data = [
                        HistoricalDataPoint(**data_point)
                        for data_point in data
                    ]
            
            # Load predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = [
                        PredictionResult(**prediction_data)
                        for prediction_data in data
                    ]
            
            # Load optimization opportunities
            if self.opportunities_file.exists():
                with open(self.opportunities_file, 'r') as f:
                    data = json.load(f)
                    self.optimization_opportunities = [
                        OptimizationOpportunity(**opportunity_data)
                        for opportunity_data in data
                    ]
            
            logger.info(f"📁 Loaded predictive analysis data: {len(self.historical_data)} data points, {len(self.predictions)} predictions")
            
        except Exception as e:
            logger.warning(f"Failed to load predictive analysis data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save historical data
            history_data = [
                {
                    "timestamp": data_point.timestamp,
                    "metric_name": data_point.metric_name,
                    "metric_value": data_point.metric_value,
                    "context": data_point.context,
                    "metadata": data_point.metadata
                }
                for data_point in self.historical_data
            ]
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save predictions
            predictions_data = [
                {
                    "prediction_id": prediction.prediction_id,
                    "metric_name": prediction.metric_name,
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.confidence,
                    "prediction_horizon": prediction.prediction_horizon,
                    "trend_direction": prediction.trend_direction,
                    "key_factors": prediction.key_factors,
                    "timestamp": prediction.timestamp
                }
                for prediction in self.predictions
            ]
            
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            # Save optimization opportunities
            opportunities_data = [
                {
                    "opportunity_id": opportunity.opportunity_id,
                    "opportunity_type": opportunity.opportunity_type,
                    "current_value": opportunity.current_value,
                    "predicted_value": opportunity.predicted_value,
                    "improvement_potential": opportunity.improvement_potential,
                    "implementation_effort": opportunity.implementation_effort,
                    "priority": opportunity.priority,
                    "description": opportunity.description,
                    "recommendations": opportunity.recommendations
                }
                for opportunity in self.optimization_opportunities
            ]
            
            with open(self.opportunities_file, 'w') as f:
                json.dump(opportunities_data, f, indent=2)
            
            logger.debug("💾 Saved predictive analysis data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save predictive analysis data: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'analysis_executor'):
            self.analysis_executor.shutdown(wait=True)


# Helper functions for integration
def create_predictive_history_analyzer(cfg: IceburgConfig) -> PredictiveHistoryAnalyzer:
    """Create predictive history analyzer instance"""
    return PredictiveHistoryAnalyzer(cfg)

def add_historical_data(
    analyzer: PredictiveHistoryAnalyzer,
    metric_name: str,
    metric_value: float,
    context: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> None:
    """Add historical data point for analysis"""
    analyzer.add_historical_data(metric_name, metric_value, context, metadata)

def get_predictions(
    analyzer: PredictiveHistoryAnalyzer,
    metric_name: str = None,
    prediction_horizon: str = None,
    min_confidence: float = 0.0
) -> List[PredictionResult]:
    """Get predictions based on criteria"""
    return analyzer.get_predictions(metric_name, prediction_horizon, min_confidence)

def get_optimization_opportunities(
    analyzer: PredictiveHistoryAnalyzer,
    opportunity_type: str = None,
    priority: str = None,
    min_improvement: float = 0.0
) -> List[OptimizationOpportunity]:
    """Get optimization opportunities based on criteria"""
    return analyzer.get_optimization_opportunities(opportunity_type, priority, min_improvement)
