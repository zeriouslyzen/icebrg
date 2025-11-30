"""
Technology Trend Emergence Detection System
Detects and predicts technology trends and emergence patterns

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

class TrendType(Enum):
    """Types of technology trends"""
    EMERGING = "emerging"
    ACCELERATING = "accelerating"
    MATURE = "mature"
    DECLINING = "declining"
    DISRUPTIVE = "disruptive"
    CONVERGENT = "convergent"

class EmergenceLevel(Enum):
    """Levels of emergence detection"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    BREAKTHROUGH = "breakthrough"

@dataclass
class TechnologySignal:
    """Represents a technology signal or indicator"""
    signal_id: str
    technology: str
    signal_type: str  # "research", "patent", "funding", "adoption", "media"
    strength: float  # 0.0 to 1.0
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnologyTrend:
    """Represents a detected technology trend"""
    trend_id: str
    technology: str
    trend_type: TrendType
    emergence_level: EmergenceLevel
    confidence: float
    growth_rate: float
    signal_count: int
    first_detected: float
    last_updated: float = field(default_factory=time.time)
    signals: List[TechnologySignal] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendPrediction:
    """Prediction for future technology trend"""
    prediction_id: str
    technology: str
    predicted_trend: TrendType
    confidence: float
    timeframe: str  # "short_term", "medium_term", "long_term"
    expected_impact: str  # "low", "medium", "high", "transformative"
    key_drivers: List[str]
    timestamp: float = field(default_factory=time.time)

class TechnologyTrendDetector:
    """
    Detects and predicts technology trends and emergence patterns
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/technology_trends")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.signals_file = self.data_dir / "technology_signals.json"
        self.trends_file = self.data_dir / "detected_trends.json"
        self.predictions_file = self.data_dir / "trend_predictions.json"
        
        # Data structures
        self.technology_signals: List[TechnologySignal] = []
        self.detected_trends: Dict[str, TechnologyTrend] = {}
        self.trend_predictions: List[TrendPrediction] = []
        
        # Signal processing
        self.signal_buffer: deque = deque(maxlen=1000)
        self.technology_keywords: Dict[str, List[str]] = {}
        
        # Trend analysis parameters
        self.emergence_threshold = 0.7
        self.growth_threshold = 0.3
        self.signal_window_days = 30
        
        # Load existing data
        self._load_data()
        self._initialize_technology_keywords()
        
        logger.info("🔍 Technology Trend Detector initialized")
    
    def add_technology_signal(
        self,
        technology: str,
        signal_type: str,
        strength: float,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a technology signal for trend analysis"""
        if metadata is None:
            metadata = {}
        
        signal_id = f"{technology}_{signal_type}_{int(time.time())}"
        signal = TechnologySignal(
            signal_id=signal_id,
            technology=technology,
            signal_type=signal_type,
            strength=strength,
            source=source,
            metadata=metadata
        )
        
        self.technology_signals.append(signal)
        self.signal_buffer.append(signal)
        
        # Keep only last 10000 signals
        if len(self.technology_signals) > 10000:
            self.technology_signals = self.technology_signals[-10000:]
        
        # Analyze for trends
        self._analyze_technology_trends(technology)
        
        # Save data
        self._save_data()
        
        logger.info(f"📡 Added technology signal: {technology} ({signal_type}) - strength: {strength:.2f}")
        
        return signal_id
    
    def _analyze_technology_trends(self, technology: str) -> None:
        """Analyze trends for a specific technology"""
        # Get recent signals for this technology
        recent_signals = [
            signal for signal in self.technology_signals
            if signal.technology == technology and 
            signal.timestamp > time.time() - (self.signal_window_days * 24 * 3600)
        ]
        
        if len(recent_signals) < 3:  # Need at least 3 signals
            return
        
        # Calculate trend metrics
        trend_metrics = self._calculate_trend_metrics(recent_signals)
        
        # Determine trend type and emergence level
        trend_type, emergence_level = self._classify_trend(trend_metrics)
        
        # Update or create trend
        trend_id = f"{technology}_{trend_type.value}"
        
        if trend_id in self.detected_trends:
            trend = self.detected_trends[trend_id]
            trend.last_updated = time.time()
            trend.signal_count = len(recent_signals)
            trend.growth_rate = trend_metrics["growth_rate"]
            trend.confidence = trend_metrics["confidence"]
            trend.signals = recent_signals[-10:]  # Keep last 10 signals
        else:
            trend = TechnologyTrend(
                trend_id=trend_id,
                technology=technology,
                trend_type=trend_type,
                emergence_level=emergence_level,
                confidence=trend_metrics["confidence"],
                growth_rate=trend_metrics["growth_rate"],
                signal_count=len(recent_signals),
                first_detected=time.time(),
                signals=recent_signals[-10:]
            )
            
            self.detected_trends[trend_id] = trend
        
        # Generate prediction if trend is significant
        if trend.confidence > self.emergence_threshold:
            self._generate_trend_prediction(trend)
    
    def _calculate_trend_metrics(self, signals: List[TechnologySignal]) -> Dict[str, float]:
        """Calculate trend metrics from signals"""
        if not signals:
            return {"growth_rate": 0.0, "confidence": 0.0, "volatility": 0.0}
        
        # Sort signals by timestamp
        signals.sort(key=lambda x: x.timestamp)
        
        # Calculate growth rate (linear regression slope)
        timestamps = np.array([s.timestamp for s in signals])
        strengths = np.array([s.strength for s in signals])
        
        # Normalize timestamps to 0-1 range
        if len(timestamps) > 1:
            timestamps = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        
        # Linear regression
        if len(timestamps) > 1:
            slope, _ = np.polyfit(timestamps, strengths, 1)
            growth_rate = slope
        else:
            growth_rate = 0.0
        
        # Calculate confidence based on signal consistency
        signal_consistency = 1.0 - np.std(strengths) if len(strengths) > 1 else 0.5
        signal_count_factor = min(1.0, len(signals) / 10.0)  # More signals = higher confidence
        confidence = (signal_consistency + signal_count_factor) / 2.0
        
        # Calculate volatility
        volatility = np.std(strengths) if len(strengths) > 1 else 0.0
        
        return {
            "growth_rate": growth_rate,
            "confidence": confidence,
            "volatility": volatility,
            "avg_strength": np.mean(strengths)
        }
    
    def _classify_trend(self, metrics: Dict[str, float]) -> Tuple[TrendType, EmergenceLevel]:
        """Classify trend type and emergence level based on metrics"""
        growth_rate = metrics["growth_rate"]
        confidence = metrics["confidence"]
        volatility = metrics["volatility"]
        
        # Determine trend type
        if growth_rate > self.growth_threshold:
            if volatility > 0.3:
                trend_type = TrendType.DISRUPTIVE
            else:
                trend_type = TrendType.ACCELERATING
        elif growth_rate > 0.1:
            trend_type = TrendType.EMERGING
        elif growth_rate > -0.1:
            trend_type = TrendType.MATURE
        else:
            trend_type = TrendType.DECLINING
        
        # Determine emergence level
        if confidence > 0.8 and growth_rate > 0.5:
            emergence_level = EmergenceLevel.BREAKTHROUGH
        elif confidence > 0.7 and growth_rate > 0.3:
            emergence_level = EmergenceLevel.STRONG
        elif confidence > 0.5 and growth_rate > 0.1:
            emergence_level = EmergenceLevel.MODERATE
        else:
            emergence_level = EmergenceLevel.WEAK
        
        return trend_type, emergence_level
    
    def _generate_trend_prediction(self, trend: TechnologyTrend) -> None:
        """Generate prediction for a significant trend"""
        # Determine predicted trend based on current trend
        predicted_trend = self._predict_future_trend(trend)
        
        # Calculate expected impact
        expected_impact = self._calculate_expected_impact(trend)
        
        # Identify key drivers
        key_drivers = self._identify_key_drivers(trend)
        
        prediction_id = f"pred_{trend.technology}_{int(time.time())}"
        prediction = TrendPrediction(
            prediction_id=prediction_id,
            technology=trend.technology,
            predicted_trend=predicted_trend,
            confidence=trend.confidence * 0.8,  # Reduce confidence for predictions
            timeframe="medium_term",
            expected_impact=expected_impact,
            key_drivers=key_drivers
        )
        
        self.trend_predictions.append(prediction)
        
        # Keep only last 100 predictions
        if len(self.trend_predictions) > 100:
            self.trend_predictions = self.trend_predictions[-100:]
        
        logger.info(f"🔮 Generated trend prediction: {trend.technology} -> {predicted_trend.value}")
    
    def _predict_future_trend(self, trend: TechnologyTrend) -> TrendType:
        """Predict future trend based on current trend"""
        # Simple trend continuation logic
        if trend.trend_type == TrendType.EMERGING:
            return TrendType.ACCELERATING
        elif trend.trend_type == TrendType.ACCELERATING:
            return TrendType.MATURE
        elif trend.trend_type == TrendType.MATURE:
            return TrendType.CONVERGENT
        elif trend.trend_type == TrendType.DISRUPTIVE:
            return TrendType.ACCELERATING
        else:
            return trend.trend_type
    
    def _calculate_expected_impact(self, trend: TechnologyTrend) -> str:
        """Calculate expected impact of trend"""
        impact_score = trend.confidence * abs(trend.growth_rate) * trend.signal_count / 10.0
        
        if impact_score > 0.8:
            return "transformative"
        elif impact_score > 0.6:
            return "high"
        elif impact_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_key_drivers(self, trend: TechnologyTrend) -> List[str]:
        """Identify key drivers for the trend"""
        drivers = []
        
        # Analyze signal types
        signal_types = [s.signal_type for s in trend.signals]
        type_counts = defaultdict(int)
        for signal_type in signal_types:
            type_counts[signal_type] += 1
        
        # Most common signal types are likely drivers
        for signal_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                drivers.append(f"{signal_type}_activity")
        
        # Add technology-specific drivers
        if trend.technology.lower() in ["ai", "artificial intelligence", "machine learning"]:
            drivers.extend(["computing_power", "data_availability", "algorithm_advances"])
        elif trend.technology.lower() in ["blockchain", "cryptocurrency"]:
            drivers.extend(["adoption_rate", "regulatory_clarity", "scalability_solutions"])
        elif trend.technology.lower() in ["quantum", "quantum computing"]:
            drivers.extend(["hardware_advances", "error_correction", "commercial_applications"])
        
        return drivers[:5]  # Return top 5 drivers
    
    def get_emerging_technologies(self, min_confidence: float = 0.6) -> List[TechnologyTrend]:
        """Get list of emerging technologies above confidence threshold"""
        emerging = [
            trend for trend in self.detected_trends.values()
            if trend.trend_type in [TrendType.EMERGING, TrendType.ACCELERATING, TrendType.DISRUPTIVE]
            and trend.confidence >= min_confidence
        ]
        
        # Sort by confidence and growth rate
        emerging.sort(key=lambda x: (x.confidence, x.growth_rate), reverse=True)
        
        return emerging
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of detected trends and predictions"""
        return {
            "total_signals": len(self.technology_signals),
            "detected_trends": len(self.detected_trends),
            "trend_predictions": len(self.trend_predictions),
            "trend_breakdown": {
                trend_type.value: len([t for t in self.detected_trends.values() if t.trend_type == trend_type])
                for trend_type in TrendType
            },
            "emergence_breakdown": {
                level.value: len([t for t in self.detected_trends.values() if t.emergence_level == level])
                for level in EmergenceLevel
            },
            "top_technologies": [
                {
                    "technology": trend.technology,
                    "trend_type": trend.trend_type.value,
                    "emergence_level": trend.emergence_level.value,
                    "confidence": trend.confidence,
                    "growth_rate": trend.growth_rate
                }
                for trend in sorted(
                    self.detected_trends.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:10]
            ]
        }
    
    def _initialize_technology_keywords(self) -> None:
        """Initialize technology keywords for signal processing"""
        self.technology_keywords = {
            "artificial_intelligence": ["ai", "artificial intelligence", "machine learning", "deep learning", "neural networks"],
            "blockchain": ["blockchain", "cryptocurrency", "bitcoin", "ethereum", "defi", "nft"],
            "quantum_computing": ["quantum", "quantum computing", "quantum algorithms", "quantum supremacy"],
            "biotechnology": ["biotech", "biotechnology", "gene editing", "crispr", "synthetic biology"],
            "robotics": ["robotics", "robots", "automation", "autonomous vehicles", "drones"],
            "renewable_energy": ["renewable energy", "solar", "wind", "battery", "energy storage"],
            "space_technology": ["space", "satellites", "spacex", "mars", "space exploration"],
            "nanotechnology": ["nanotechnology", "nanomaterials", "nanomedicine", "nanoelectronics"]
        }
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load technology signals
            if self.signals_file.exists():
                with open(self.signals_file, 'r') as f:
                    data = json.load(f)
                    self.technology_signals = [
                        TechnologySignal(**signal_data)
                        for signal_data in data
                    ]
            
            # Load detected trends
            if self.trends_file.exists():
                with open(self.trends_file, 'r') as f:
                    data = json.load(f)
                    self.detected_trends = {
                        trend_id: TechnologyTrend(**trend_data)
                        for trend_id, trend_data in data.items()
                    }
            
            # Load trend predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.trend_predictions = [
                        TrendPrediction(**pred_data)
                        for pred_data in data
                    ]
            
            logger.info(f"📁 Loaded trend data: {len(self.technology_signals)} signals, {len(self.detected_trends)} trends")
            
        except Exception as e:
            logger.warning(f"Failed to load trend data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save technology signals
            signals_data = [
                {
                    "signal_id": signal.signal_id,
                    "technology": signal.technology,
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "source": signal.source,
                    "timestamp": signal.timestamp,
                    "metadata": signal.metadata
                }
                for signal in self.technology_signals
            ]
            
            with open(self.signals_file, 'w') as f:
                json.dump(signals_data, f, indent=2)
            
            # Save detected trends
            trends_data = {
                trend_id: {
                    "trend_id": trend.trend_id,
                    "technology": trend.technology,
                    "trend_type": trend.trend_type.value,
                    "emergence_level": trend.emergence_level.value,
                    "confidence": trend.confidence,
                    "growth_rate": trend.growth_rate,
                    "signal_count": trend.signal_count,
                    "first_detected": trend.first_detected,
                    "last_updated": trend.last_updated,
                    "signals": [
                        {
                            "signal_id": signal.signal_id,
                            "technology": signal.technology,
                            "signal_type": signal.signal_type,
                            "strength": signal.strength,
                            "source": signal.source,
                            "timestamp": signal.timestamp,
                            "metadata": signal.metadata
                        }
                        for signal in trend.signals
                    ],
                    "metadata": trend.metadata
                }
                for trend_id, trend in self.detected_trends.items()
            }
            
            with open(self.trends_file, 'w') as f:
                json.dump(trends_data, f, indent=2)
            
            # Save trend predictions
            predictions_data = [
                {
                    "prediction_id": pred.prediction_id,
                    "technology": pred.technology,
                    "predicted_trend": pred.predicted_trend.value,
                    "confidence": pred.confidence,
                    "timeframe": pred.timeframe,
                    "expected_impact": pred.expected_impact,
                    "key_drivers": pred.key_drivers,
                    "timestamp": pred.timestamp
                }
                for pred in self.trend_predictions
            ]
            
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            logger.debug("💾 Saved trend data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save trend data: {e}")


# Helper functions for integration
def create_technology_trend_detector(cfg: IceburgConfig) -> TechnologyTrendDetector:
    """Create technology trend detector instance"""
    return TechnologyTrendDetector(cfg)

def add_technology_signal(
    detector: TechnologyTrendDetector,
    technology: str,
    signal_type: str,
    strength: float,
    source: str,
    metadata: Dict[str, Any] = None
) -> str:
    """Add technology signal for trend analysis"""
    return detector.add_technology_signal(technology, signal_type, strength, source, metadata)

def get_emerging_technologies(
    detector: TechnologyTrendDetector,
    min_confidence: float = 0.6
) -> List[TechnologyTrend]:
    """Get emerging technologies above confidence threshold"""
    return detector.get_emerging_technologies(min_confidence)
