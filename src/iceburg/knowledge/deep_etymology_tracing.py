"""
Deep Etymology Tracing Engine
Traces word origins through multiple layers: astrology → stars → constellations → language origins → occult
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EtymologyLayer:
    """Represents a layer in etymology tracing"""
    layer_id: str
    term: str
    origin: str
    language: str
    meaning: str
    connections: List[str] = field(default_factory=list)
    occult_connections: List[str] = field(default_factory=list)
    secret_society_connections: List[str] = field(default_factory=list)
    suppressed_knowledge_connections: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EtymologyTrace:
    """Complete etymology trace from origins to present"""
    term: str
    layers: List[EtymologyLayer] = field(default_factory=list)
    complete_chain: str = ""
    occult_path: List[str] = field(default_factory=list)
    secret_society_path: List[str] = field(default_factory=list)
    suppressed_knowledge_path: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeepEtymologyTracing:
    """
    Traces word origins through multiple layers.
    
    Example: Astrology → stars → constellations → language origins → occult → secret societies
    """
    
    def __init__(self):
        """Initialize deep etymology tracing engine."""
        self.etymology_cache: Dict[str, EtymologyTrace] = {}
        self.occult_connections: Dict[str, List[str]] = {}
        self.secret_society_connections: Dict[str, List[str]] = {}
        self.suppressed_knowledge_connections: Dict[str, List[str]] = {}
        
        # Initialize occult connections database
        self._initialize_occult_connections()
        
        logger.info("Deep Etymology Tracing Engine initialized")
    
    def trace_deep_etymology(self, term: str) -> EtymologyTrace:
        """
        Trace deep etymology of a term.
        
        Args:
            term: Term to trace (e.g., "astrology")
            
        Returns:
            Complete etymology trace
        """
        # Check cache
        if term.lower() in self.etymology_cache:
            return self.etymology_cache[term.lower()]
        
        trace = EtymologyTrace(term=term)
        
        # Trace through layers
        layers = self._trace_etymology_layers(term)
        trace.layers = layers
        
        # Build complete chain
        trace.complete_chain = self._build_complete_chain(layers)
        
        # Trace occult path
        trace.occult_path = self._trace_occult_path(term, layers)
        
        # Trace secret society path
        trace.secret_society_path = self._trace_secret_society_path(term, layers)
        
        # Trace suppressed knowledge path
        trace.suppressed_knowledge_path = self._trace_suppressed_knowledge_path(term, layers)
        
        # Calculate confidence
        trace.confidence = self._calculate_confidence(layers, trace)
        
        # Cache result
        self.etymology_cache[term.lower()] = trace
        
        logger.info(f"Traced deep etymology for '{term}': {len(layers)} layers, "
                   f"{len(trace.occult_path)} occult connections, "
                   f"{len(trace.secret_society_path)} secret society connections")
        
        return trace
    
    def _trace_etymology_layers(self, term: str) -> List[EtymologyLayer]:
        """Trace etymology through multiple layers."""
        layers = []
        term_lower = term.lower()
        
        # Layer 1: Modern term
        layer1 = EtymologyLayer(
            layer_id=f"layer_1_{term_lower}",
            term=term,
            origin="Modern English",
            language="English",
            meaning=self._get_meaning(term)
        )
        layers.append(layer1)
        
        # Layer 2: Etymology (basic)
        etymology = self._get_basic_etymology(term)
        if etymology:
            layer2 = EtymologyLayer(
                layer_id=f"layer_2_{term_lower}",
                term=etymology.get("root", term),
                origin=etymology.get("origin", "Unknown"),
                language=etymology.get("language", "Unknown"),
                meaning=etymology.get("meaning", "")
            )
            layers.append(layer2)
        
        # Layer 3: Proto-language
        proto = self._get_proto_language(term)
        if proto:
            layer3 = EtymologyLayer(
                layer_id=f"layer_3_{term_lower}",
                term=proto.get("proto_term", ""),
                origin=proto.get("proto_language", "Proto-Indo-European"),
                language="PIE",
                meaning=proto.get("meaning", "")
            )
            layers.append(layer3)
        
        # Layer 4: Ancient connections
        ancient = self._get_ancient_connections(term)
        if ancient:
            for conn in ancient:
                layer4 = EtymologyLayer(
                    layer_id=f"layer_4_{len(layers)}",
                    term=conn.get("term", ""),
                    origin=conn.get("origin", "Ancient"),
                    language=conn.get("language", "Ancient"),
                    meaning=conn.get("meaning", "")
                )
                layers.append(layer4)
        
        return layers
    
    def _get_meaning(self, term: str) -> str:
        """Get meaning of term."""
        meanings = {
            "astrology": "Study of celestial bodies and their influence on human affairs",
            "astronomy": "Study of celestial objects and phenomena",
            "star": "Luminous celestial body",
            "constellation": "Group of stars forming a pattern",
            "zodiac": "Belt of the heavens divided into twelve signs",
            "planet": "Celestial body orbiting a star",
            "sun": "Star at the center of the solar system",
            "moon": "Natural satellite of Earth"
        }
        return meanings.get(term.lower(), f"Meaning of {term}")
    
    def _get_basic_etymology(self, term: str) -> Optional[Dict[str, Any]]:
        """Get basic etymology of term."""
        etymologies = {
            "astrology": {
                "root": "astro",
                "origin": "Greek 'astron' (star)",
                "language": "Greek",
                "meaning": "star"
            },
            "astronomy": {
                "root": "astro",
                "origin": "Greek 'astron' (star)",
                "language": "Greek",
                "meaning": "star"
            },
            "star": {
                "root": "ster",
                "origin": "Proto-Germanic '*sternō'",
                "language": "Proto-Germanic",
                "meaning": "star"
            },
            "constellation": {
                "root": "stella",
                "origin": "Latin 'stella' (star)",
                "language": "Latin",
                "meaning": "star"
            },
            "zodiac": {
                "root": "zodiakos",
                "origin": "Greek 'zōdiakos' (circle of animals)",
                "language": "Greek",
                "meaning": "circle of animals"
            },
            "planet": {
                "root": "planetes",
                "origin": "Greek 'planētēs' (wanderer)",
                "language": "Greek",
                "meaning": "wanderer"
            }
        }
        return etymologies.get(term.lower())
    
    def _get_proto_language(self, term: str) -> Optional[Dict[str, Any]]:
        """Get proto-language etymology."""
        proto_terms = {
            "astrology": {
                "proto_term": "*h₂ster-",
                "proto_language": "Proto-Indo-European",
                "meaning": "star"
            },
            "star": {
                "proto_term": "*h₂ster-",
                "proto_language": "Proto-Indo-European",
                "meaning": "star"
            },
            "constellation": {
                "proto_term": "*h₂ster-",
                "proto_language": "Proto-Indo-European",
                "meaning": "star"
            }
        }
        return proto_terms.get(term.lower())
    
    def _get_ancient_connections(self, term: str) -> List[Dict[str, Any]]:
        """Get ancient connections."""
        connections = {
            "astrology": [
                {"term": "Sanskrit 'star'", "origin": "Ancient India", "language": "Sanskrit", "meaning": "star"},
                {"term": "Babylonian astrology", "origin": "Ancient Mesopotamia", "language": "Akkadian", "meaning": "celestial divination"},
                {"term": "Egyptian star knowledge", "origin": "Ancient Egypt", "language": "Egyptian", "meaning": "celestial wisdom"}
            ],
            "star": [
                {"term": "Sanskrit 'star'", "origin": "Ancient India", "language": "Sanskrit", "meaning": "star"},
                {"term": "Babylonian star catalog", "origin": "Ancient Mesopotamia", "language": "Akkadian", "meaning": "star list"}
            ]
        }
        return connections.get(term.lower(), [])
    
    def _build_complete_chain(self, layers: List[EtymologyLayer]) -> str:
        """Build complete etymology chain."""
        chain_parts = []
        for layer in layers:
            chain_parts.append(f"{layer.term} ({layer.language}: {layer.meaning})")
        return " → ".join(chain_parts)
    
    def _trace_occult_path(self, term: str, layers: List[EtymologyLayer]) -> List[str]:
        """Trace occult connections path."""
        occult_path = []
        
        # Check occult connections database
        if term.lower() in self.occult_connections:
            occult_path.extend(self.occult_connections[term.lower()])
        
        # Trace through layers
        for layer in layers:
            if layer.term.lower() in self.occult_connections:
                occult_path.extend(self.occult_connections[layer.term.lower()])
        
        return list(set(occult_path))  # Remove duplicates
    
    def _trace_secret_society_path(self, term: str, layers: List[EtymologyLayer]) -> List[str]:
        """Trace secret society connections path."""
        secret_path = []
        
        # Check secret society connections database
        if term.lower() in self.secret_society_connections:
            secret_path.extend(self.secret_society_connections[term.lower()])
        
        # Trace through layers
        for layer in layers:
            if layer.term.lower() in self.secret_society_connections:
                secret_path.extend(self.secret_society_connections[layer.term.lower()])
        
        return list(set(secret_path))  # Remove duplicates
    
    def _trace_suppressed_knowledge_path(self, term: str, layers: List[EtymologyLayer]) -> List[str]:
        """Trace suppressed knowledge connections path."""
        suppressed_path = []
        
        # Check suppressed knowledge connections database
        if term.lower() in self.suppressed_knowledge_connections:
            suppressed_path.extend(self.suppressed_knowledge_connections[term.lower()])
        
        # Trace through layers
        for layer in layers:
            if layer.term.lower() in self.suppressed_knowledge_connections:
                suppressed_path.extend(self.suppressed_knowledge_connections[layer.term.lower()])
        
        return list(set(suppressed_path))  # Remove duplicates
    
    def _calculate_confidence(self, layers: List[EtymologyLayer], trace: EtymologyTrace) -> float:
        """Calculate confidence in etymology trace."""
        confidence = 0.0
        
        # Base confidence from number of layers
        if len(layers) >= 3:
            confidence += 0.4
        elif len(layers) >= 2:
            confidence += 0.3
        else:
            confidence += 0.2
        
        # Bonus for occult connections
        if trace.occult_path:
            confidence += 0.2
        
        # Bonus for secret society connections
        if trace.secret_society_path:
            confidence += 0.2
        
        # Bonus for suppressed knowledge connections
        if trace.suppressed_knowledge_path:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _initialize_occult_connections(self) -> None:
        """Initialize occult connections database."""
        # Astrology occult connections
        self.occult_connections["astrology"] = [
            "Hermetic astrology",
            "Kabbalistic astrology",
            "Alchemical astrology",
            "Rosicrucian astrology",
            "Masonic astrology",
            "Esoteric astrology",
            "Occult star knowledge",
            "Hidden celestial wisdom"
        ]
        
        self.secret_society_connections["astrology"] = [
            "Rosicrucians",
            "Freemasons",
            "Hermetic Order of the Golden Dawn",
            "Ordo Templi Orientis",
            "Ancient and Accepted Scottish Rite",
            "Knights Templar (astrological connections)"
        ]
        
        self.suppressed_knowledge_connections["astrology"] = [
            "Babylonian star knowledge (suppressed)",
            "Egyptian celestial wisdom (suppressed)",
            "Ancient star maps (suppressed)",
            "Precession of the equinoxes (suppressed knowledge)"
        ]
        
        # Star occult connections
        self.occult_connections["star"] = [
            "Star knowledge (occult)",
            "Celestial wisdom (hidden)",
            "Star maps (esoteric)",
            "Astronomical secrets"
        ]
        
        self.secret_society_connections["star"] = [
            "Star-based secret societies",
            "Celestial order organizations"
        ]
        
        # Constellation occult connections
        self.occult_connections["constellation"] = [
            "Constellation knowledge (occult)",
            "Star pattern wisdom (esoteric)",
            "Celestial mapping (hidden)"
        ]
        
        logger.info("Initialized occult connections database")

