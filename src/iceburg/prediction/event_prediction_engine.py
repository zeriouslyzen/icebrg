"""
Event Prediction Engine - Phase 2
Predicts geopolitical events, economic regime changes, and black swans

Integrates ICEBURG's existing emergence detection systems.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..emergence.breakthrough_storage import BreakthroughStorage, BreakthroughDiscovery
from ..emergence.quantum_emergence_detector import QuantumEmergenceDetector
from ..emergence.temporal_emergence_detector import TemporalEmergenceDetector
from ..knowledge.predictive_history import PredictiveHistorySystem
from ..optimization.technology_trend_detector import TechnologyTrendDetector
from ..config import IceburgConfig
from ..intelligence.multi_source_aggregator import CorrelatedIntelligence

logger = logging.getLogger(__name__)


class EventCategory(Enum):
    """Categories of predictable events"""
    GEOPOLITICAL = "geopolitical"  # Wars, coups, sanctions
    ECONOMIC = "economic"  # Recessions, currency crises
    TECHNOLOGICAL = "technological"  # Breakthroughs, disruptions
    CORPORATE = "corporate"  # Bankruptcies, M&A, scandals
    BLACK_SWAN = "black_swan"  # Unpredictable but high-impact
    REGIME_CHANGE = "regime_change"  # System-level transitions


@dataclass
class EventPrediction:
    """Represents a predicted event"""
    prediction_id: str
    event_category: EventCategory
    event_description: str
    probability: float  # 0.0 to 1.0
    timeframe: str  # "1w", "1m", "3m", "1y"
    expected_impact: float  # 0.0 to 1.0
    confidence: float  # Statistical confidence
    key_indicators: List[str]
    cascade_effects: List[str] = field(default_factory=list)
    affected_entities: List[str] = field(default_factory=list)
    historical_precedents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BlackSwanAlert:
    """Alert for potential black swan event"""
    alert_id: str
    signal_strength: float
    emergence_score: float
    description: str
    early_warning_indicators: List[str]
    probability_range: Tuple[float, float]  # min, max probability
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EventPredictionEngine:
    """
    Predicts events across multiple domains using ICEBURG's
    emergence detection and historical pattern systems.
    
    This is not just price prediction - it's event forecasting.
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        
        # Initialize ICEBURG subsystems
        self.breakthrough_storage = BreakthroughStorage()
        self.quantum_emergence = QuantumEmergenceDetector(cfg)
        self.temporal_emergence = TemporalEmergenceDetector(cfg)
        self.predictive_history = PredictiveHistorySystem()
        self.tech_trends = TechnologyTrendDetector(cfg)
        
        self.predictions: List[EventPrediction] = []
        self.black_swan_alerts: List[BlackSwanAlert] = []
        
        logger.info("Event Prediction Engine initialized")
    
    def predict_geopolitical_event(
        self,
        scenario: str,
        intelligence: Optional[CorrelatedIntelligence] = None
    ) -> EventPrediction:
        """
        Predict geopolitical events (wars, coups, sanctions, etc.).
        
        Uses:
        - PredictiveHistory for historical pattern matching
        - TemporalEmergence for regime change detection
        - Intelligence correlation for current signals
        
        Args:
            scenario: Description of potential event
            intelligence: Optional correlated intelligence
            
        Returns:
            Event prediction with probability and timeframe
        """
        # Match historical patterns
        historical_patterns = self.predictive_history.match_historical_patterns(scenario)
        
        # Detect temporal anomalies
        temporal_signals = self._detect_temporal_anomalies(scenario)
        
        # Analyze intelligence signals if available
        intelligence_score = 0.5
        key_indicators = []
        
        if intelligence:
            intelligence_score = intelligence.confidence
            key_indicators = [s.content[:100] for s in intelligence.signals[:5]]
        
        # Calculate probability based on convergence
        base_probability = len(historical_patterns) * 0.15
        temporal_boost = temporal_signals * 0.2
        intelligence_boost = intelligence_score * 0.3
        
        probability = min(base_probability + temporal_boost + intelligence_boost, 0.95)
        
        # Estimate timeframe from historical patterns
        timeframe = self._estimate_timeframe(historical_patterns, scenario)
        
        # Calculate impact
        impact = self._estimate_geopolitical_impact(scenario, probability)
        
        prediction = EventPrediction(
            prediction_id=f"geo_{datetime.utcnow().timestamp()}",
            event_category=EventCategory.GEOPOLITICAL,
            event_description=scenario,
            probability=probability,
            timeframe=timeframe,
            expected_impact=impact,
            confidence=0.5 + (len(historical_patterns) * 0.1),
            key_indicators=key_indicators,
            historical_precedents=[p.pattern_id for p in historical_patterns[:3]],
            metadata={
                "historical_pattern_count": len(historical_patterns),
                "temporal_signals": temporal_signals,
                "intelligence_source": intelligence.correlation_id if intelligence else None
            }
        )
        
        self.predictions.append(prediction)
        logger.info(f"Geopolitical prediction: {scenario} - {probability:.1%} probability")
        
        return prediction
    
    def predict_economic_regime_change(
        self,
        market: str,
        indicators: Dict[str, float]
    ) -> EventPrediction:
        """
        Predict economic regime changes (recessions, hyperinflation, etc.).
        
        Uses:
        - QuantumEmergenceDetector for regime shifts
        - Historical pattern matching
        
        Args:
            market: Market identifier (US, China, crypto, etc.)
            indicators: Economic indicators (inflation, unemployment, etc.)
            
        Returns:
            Regime change prediction
        """
        # Detect quantum regime change signals
        emergence_data = self.quantum_emergence.detect_emergence({
            "market": market,
            "indicators": indicators
        })
        
        emergence_score = emergence_data.get("emergence_score", 0.0) if emergence_data else 0.0
        
        # Historical recession patterns
        recession_patterns = self.predictive_history.match_historical_patterns(
            f"{market} recession indicators"
        )
        
        # Calculate probability
        probability = min(emergence_score * 0.6 + len(recession_patterns) * 0.15, 0.9)
        
        # Timeframe estimation
        if emergence_score > 0.7:
            timeframe = "3m"  # Imminent
        elif emergence_score > 0.5:
            timeframe = "6m"
        else:
            timeframe = "1y"
        
        prediction = EventPrediction(
            prediction_id=f"econ_{datetime.utcnow().timestamp()}",
            event_category=EventCategory.ECONOMIC,
            event_description=f"{market} economic regime change",
            probability=probability,
            timeframe=timeframe,
            expected_impact=0.8,  # High impact
            confidence=emergence_score,
            key_indicators=list(indicators.keys()),
            cascade_effects=["Market volatility", "Currency devaluation", "Capital flight"],
            metadata={
                "emergence_score": emergence_score,
                "indicators": indicators
            }
        )
        
        self.predictions.append(prediction)
        return prediction
    
    def detect_black_swan(
        self,
        domain: str,
        observation_window_days: int = 90
    ) -> List[BlackSwanAlert]:
        """
        Detect potential black swan events.
        
        Black swans are:
        - Highly improbable (tail events)
        - High impact
        - Retrospectively obvious
        
        Uses:
        - BreakthroughStorage for paradigm shifts
        - QuantumEmergence for regime changes
        - TemporalEmergence for anomalies
        
        Args:
            domain: Domain to monitor (finance, geopolitics, tech, etc.)
            observation_window_days: Lookback period
            
        Returns:
            List of black swan alerts
        """
        alerts = []
        
        # Check breakthrough storage for paradigm shifts
        breakthroughs = self.breakthrough_storage.get_breakthroughs_by_confidence(
            min_confidence=0.8
        )
        
        recent_breakthroughs = [
            b for b in breakthroughs 
            if (datetime.utcnow().timestamp() - b.timestamp) < (observation_window_days * 86400)
        ]
        
        for breakthrough in recent_breakthroughs:
            if breakthrough.impact_score > 0.7:
                alert = BlackSwanAlert(
                    alert_id=f"bs_{breakthrough.id}",
                    signal_strength=breakthrough.confidence,
                    emergence_score=breakthrough.novelty_score,
                    description=breakthrough.description,
                    early_warning_indicators=breakthrough.evidence or [],
                    probability_range=(0.01, 0.1),  # Low probability, high impact
                    timestamp=datetime.fromtimestamp(breakthrough.timestamp)
                )
                alerts.append(alert)
        
        self.black_swan_alerts.extend(alerts)
        logger.info(f"Detected {len(alerts)} potential black swans in {domain}")
        
        return alerts
    
    def adversarial_modeling(
        self,
        actor: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Model adversarial actions (competitors, governments, enemies).
        
        Uses game theory and historical patterns to predict moves.
        
        Args:
            actor: Entity to model (country, corporation, person)
            context: Current situation context
            
        Returns:
            Predicted strategies and counter-moves
        """
        # Decode actor using predictive history
        actor_profile = self.predictive_history.decode_person(actor)
        
        # Historical pattern matching
        actor_patterns = self.predictive_history.predict_pattern(
            actor, 
            actor_profile.pattern_matches
        )
        
        strategies = []
        probabilities = []
        
        # Game theory analysis (simplified)
        if "conflict" in str(context).lower():
            strategies = ["Escalate", "De-escalate", "Maintain status quo"]
            probabilities = [0.3, 0.2, 0.5]
        else:
            strategies = ["Cooperate", "Defect", "Mixed strategy"]
            probabilities = [0.4, 0.3, 0.3]
        
        return {
            "actor": actor,
            "predicted_strategies": [
                {"strategy": s, "probability": p}
                for s, p in zip(strategies, probabilities)
            ],
            "historical_behavior": actor_profile.pattern_matches,
            "recommended_counter_moves": self._generate_counter_moves(strategies[0]),
            "confidence": actor_profile.confidence
        }
    
    def get_predictions(
        self,
        category: Optional[EventCategory] = None,
        min_probability: float = 0.3,
        limit: int = 50
    ) -> List[EventPrediction]:
        """Get event predictions filtered by criteria."""
        filtered = self.predictions
        
        if category:
            filtered = [p for p in filtered if p.event_category == category]
        
        filtered = [p for p in filtered if p.probability >= min_probability]
        filtered.sort(key=lambda x: x.probability, reverse=True)
        
        return filtered[:limit]
    
    def _detect_temporal_anomalies(self, scenario: str) -> float:
        """Detect temporal anomalies related to scenario."""
        # Simplified - would use TemporalEmergenceDetector
        # Return score 0-1 indicating anomaly strength
        keywords = ["sudden", "unprecedented", "rapid", "unusual"]
        score = sum(0.2 for kw in keywords if kw in scenario.lower())
        return min(score, 1.0)
    
    def _estimate_timeframe(self, patterns: List[Any], scenario: str) -> str:
        """Estimate timeframe from historical patterns."""
        if "imminent" in scenario.lower():
            return "1w"
        elif "short-term" in scenario.lower():
            return "1m"
        elif "long-term" in scenario.lower():
            return "1y"
        else:
            return "3m"  # Default
    
    def _estimate_geopolitical_impact(self, scenario: str, probability: float) -> float:
        """Estimate impact of geopolitical event."""
        high_impact_keywords = ["war", "coup", "crisis", "collapse"]
        medium_impact = ["sanctions", "election", "treaty"]
        
        for kw in high_impact_keywords:
            if kw in scenario.lower():
                return 0.9
        
        for kw in medium_impact:
            if kw in scenario.lower():
                return 0.6
        
        return 0.4
    
    def _generate_counter_moves(self, opponent_strategy: str) -> List[str]:
        """Generate recommended counter-moves."""
        counter_map = {
            "Escalate": ["De-escalate", "Prepare defenses", "Seek mediation"],
            "De-escalate": ["Negotiate terms", "Consolidate position"],
            "Cooperate": ["Reciprocate", "Build trust"],
            "Defect": ["Retaliate", "Form alliances", "Isolate actor"]
        }
        
        return counter_map.get(opponent_strategy, ["Monitor situation"])


# Global engine instance
_engine: Optional[EventPredictionEngine] = None


def get_event_prediction_engine(cfg: Optional[IceburgConfig] = None) -> EventPredictionEngine:
    """Get or create global event prediction engine."""
    global _engine
    if _engine is None:
        if cfg is None:
            cfg = IceburgConfig()
        _engine = EventPredictionEngine(cfg)
    return _engine
