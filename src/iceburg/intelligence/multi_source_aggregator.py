"""
Multi-Source Intelligence Aggregator
Core infrastructure for V2 Advanced Prediction Market System

Aggregates intelligence from multiple sources (OSINT, CORPINT, etc.)
and correlates signals for event prediction.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IntelligenceSource(Enum):
    """Types of intelligence sources"""
    OSINT = "osint"  # Open Source Intelligence
    CORPINT = "corpint"  # Corporate Intelligence
    DARKINT = "darkint"  # Dark Web Intelligence
    NETINT = "netint"  # Network Intelligence
    GEOINT = "geoint"  # Geospatial Intelligence
    PSYINT = "psyint"  # Psychological/Sentiment Intelligence


class SignalPriority(Enum):
    """Priority levels for intelligence signals"""
    CRITICAL = "critical"  # Black swan, immediate threat
    HIGH = "high"  # High-impact event likely
    MEDIUM = "medium"  # Notable development
    LOW = "low"  # Background noise
    NOISE = "noise"  # Filter out


@dataclass
class IntelligenceSignal:
    """Represents a single intelligence signal from any source"""
    signal_id: str
    source_type: IntelligenceSource
    source_name: str
    content: str
    timestamp: datetime
    priority: SignalPriority = SignalPriority.MEDIUM
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # People, orgs, places
    related_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelatedIntelligence:
    """Represents correlated signals that tell a story"""
    correlation_id: str
    signals: List[IntelligenceSignal]
    narrative: str
    confidence: float
    predicted_event: Optional[str] = None
    timeframe: Optional[str] = None
    impact_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MultiSourceIntelligenceAggregator:
    """
    Aggregates and correlates intelligence from multiple sources.
    
    This is the foundation of V2's prediction capabilities.
    """
    
    def __init__(self):
        self.signals: List[IntelligenceSignal] = []
        self.correlations: List[CorrelatedIntelligence] = []
        self.sources: Dict[str, Any] = {}
        
        logger.info("Intelligence Aggregator initialized")
    
    def ingest_signal(self, signal: IntelligenceSignal) -> None:
        """
        Ingest a single intelligence signal.
        
        Args:
            signal: Intelligence signal to ingest
        """
        self.signals.append(signal)
        logger.debug(f"Ingested signal {signal.signal_id} from {signal.source_type.value}")
        
        # Auto-correlate if we have enough signals
        if len(self.signals) >= 10:
            self._auto_correlate()
    
    def ingest_from_news_feeds(self) -> List[IntelligenceSignal]:
        """
        Ingest intelligence from news feeds (RSS, news APIs).
        
        Returns:
            List of signals generated from news
        """
        # Placeholder - will integrate with news APIs
        logger.info("Ingesting from news feeds...")
        return []
    
    def ingest_from_social_media(self) -> List[IntelligenceSignal]:
        """
        Ingest intelligence from social media (Twitter, Reddit, etc.).
        
        Returns:
            List of signals generated from social media
        """
        # Placeholder - will integrate with social APIs
        logger.info("Ingesting from social media...")
        return []
    
    def ingest_from_sec_filings(self) -> List[IntelligenceSignal]:
        """
        Ingest intelligence from SEC filings.
        
        Returns:
            List of signals generated from SEC data
        """
        # Placeholder - will integrate with SEC EDGAR API
        logger.info("Ingesting from SEC filings...")
        return []
    
    def correlate_signals_across_sources(
        self, 
        min_signals: int = 3,
        min_confidence: float = 0.6
    ) -> List[CorrelatedIntelligence]:
        """
        Correlate signals across different sources to identify patterns.
        
        Args:
            min_signals: Minimum number of signals to form correlation
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of correlated intelligence narratives
        """
        correlations = []
        
        # Group signals by entities and tags
        entity_groups = self._group_by_entities()
        
        for entity, signals in entity_groups.items():
            if len(signals) >= min_signals:
                correlation = self._create_correlation(signals)
                if correlation.confidence >= min_confidence:
                    correlations.append(correlation)
        
        self.correlations.extend(correlations)
        logger.info(f"Created {len(correlations)} new correlations")
        
        return correlations
    
    def get_intelligence_feed(
        self,
        priority: Optional[SignalPriority] = None,
        limit: int = 100
    ) -> List[IntelligenceSignal]:
        """
        Get intelligence feed filtered by priority.
        
        Args:
            priority: Filter by priority level
            limit: Maximum number of signals to return
            
        Returns:
            List of intelligence signals
        """
        filtered = self.signals
        
        if priority:
            filtered = [s for s in filtered if s.priority == priority]
        
        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered[:limit]
    
    def get_correlations(
        self,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> List[CorrelatedIntelligence]:
        """
        Get correlated intelligence narratives.
        
        Args:
            min_confidence: Minimum confidence threshold
            limit: Maximum number of correlations to return
            
        Returns:
            List of correlated intelligence
        """
        filtered = [c for c in self.correlations if c.confidence >= min_confidence]
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered[:limit]
    
    def _group_by_entities(self) -> Dict[str, List[IntelligenceSignal]]:
        """Group signals by common entities."""
        entity_groups = {}
        
        for signal in self.signals:
            for entity in signal.entities:
                if entity not in entity_groups:
                    entity_groups[entity] = []
                entity_groups[entity].append(signal)
        
        return entity_groups
    
    def _create_correlation(self, signals: List[IntelligenceSignal]) -> CorrelatedIntelligence:
        """Create a correlated intelligence narrative from signals."""
        entities = set()
        for signal in signals:
            entities.update(signal.entities)
        
        # Generate narrative (simple version - will enhance with LLM)
        narrative = f"Multiple signals detected regarding {', '.join(list(entities)[:3])}"
        
        # Calculate confidence (average of signal confidences)
        confidence = sum(s.confidence for s in signals) / len(signals)
        
        correlation = CorrelatedIntelligence(
            correlation_id=f"corr_{datetime.utcnow().timestamp()}",
            signals=signals,
            narrative=narrative,
            confidence=confidence
        )
        
        return correlation
    
    def _auto_correlate(self) -> None:
        """Automatically correlate signals when enough are collected."""
        # Only correlate recent signals (last 24 hours)
        cutoff = datetime.utcnow().timestamp() - 86400
        recent = [s for s in self.signals if s.timestamp.timestamp() > cutoff]
        
        if len(recent) >= 10:
            self.correlate_signals_across_sources()


# Global aggregator instance
_aggregator: Optional[MultiSourceIntelligenceAggregator] = None


def get_intelligence_aggregator() -> MultiSourceIntelligenceAggregator:
    """Get or create global intelligence aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = MultiSourceIntelligenceAggregator()
    return _aggregator
