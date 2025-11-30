"""
Occult Knowledge Database
Database of occult connections, secret society patterns, and hidden knowledge structures
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OccultConnection:
    """Represents an occult connection"""
    connection_id: str
    source_term: str
    target_term: str
    connection_type: str  # "occult", "secret_society", "suppressed_knowledge", "hidden_structure"
    description: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SecretSociety:
    """Represents a secret society"""
    society_id: str
    name: str
    description: str
    connections: List[str] = field(default_factory=list)
    occult_connections: List[str] = field(default_factory=list)
    suppressed_knowledge: List[str] = field(default_factory=list)
    historical_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OccultKnowledgeDatabase:
    """
    Database of occult connections, secret society patterns, and hidden knowledge structures.
    
    Maps connections between:
    - Occult knowledge
    - Secret societies
    - Suppressed knowledge
    - Hidden structures
    """
    
    def __init__(self):
        """Initialize occult knowledge database."""
        self.occult_connections: Dict[str, List[OccultConnection]] = {}
        self.secret_societies: Dict[str, SecretSociety] = {}
        self.suppressed_knowledge: Dict[str, List[str]] = {}
        self.hidden_structures: Dict[str, List[str]] = {}
        
        # Initialize database
        self._initialize_database()
        
        logger.info("Occult Knowledge Database initialized")
    
    def find_occult_connections(self, term: str) -> List[OccultConnection]:
        """
        Find occult connections for a term.
        
        Args:
            term: Term to find connections for
            
        Returns:
            List of occult connections
        """
        connections = []
        term_lower = term.lower()
        
        # Direct connections
        if term_lower in self.occult_connections:
            connections.extend(self.occult_connections[term_lower])
        
        # Related terms
        related_terms = self._find_related_terms(term)
        for related_term in related_terms:
            if related_term in self.occult_connections:
                connections.extend(self.occult_connections[related_term])
        
        logger.info(f"Found {len(connections)} occult connections for '{term}'")
        return connections
    
    def find_secret_society_connections(self, term: str) -> List[SecretSociety]:
        """
        Find secret society connections for a term.
        
        Args:
            term: Term to find connections for
            
        Returns:
            List of secret societies
        """
        societies = []
        term_lower = term.lower()
        
        # Check each secret society
        for society_id, society in self.secret_societies.items():
            # Check if term is in society's connections
            if term_lower in [c.lower() for c in society.connections]:
                societies.append(society)
            # Check if term is in society's occult connections
            elif term_lower in [c.lower() for c in society.occult_connections]:
                societies.append(society)
        
        logger.info(f"Found {len(societies)} secret society connections for '{term}'")
        return societies
    
    def find_suppressed_knowledge(self, term: str) -> List[str]:
        """
        Find suppressed knowledge related to a term.
        
        Args:
            term: Term to find suppressed knowledge for
            
        Returns:
            List of suppressed knowledge items
        """
        suppressed = []
        term_lower = term.lower()
        
        # Direct connections
        if term_lower in self.suppressed_knowledge:
            suppressed.extend(self.suppressed_knowledge[term_lower])
        
        # Related terms
        related_terms = self._find_related_terms(term)
        for related_term in related_terms:
            if related_term in self.suppressed_knowledge:
                suppressed.extend(self.suppressed_knowledge[related_term])
        
        logger.info(f"Found {len(suppressed)} suppressed knowledge items for '{term}'")
        return suppressed
    
    def decode_hidden_structure(self, term: str) -> Dict[str, Any]:
        """
        Decode hidden structure for a term.
        
        Args:
            term: Term to decode
            
        Returns:
            Dictionary with hidden structure information
        """
        structure = {
            "term": term,
            "occult_connections": [],
            "secret_society_connections": [],
            "suppressed_knowledge": [],
            "hidden_structures": [],
            "complete_path": []
        }
        
        # Find occult connections
        occult_conns = self.find_occult_connections(term)
        structure["occult_connections"] = [
            {
                "source": c.source_term,
                "target": c.target_term,
                "type": c.connection_type,
                "description": c.description
            }
            for c in occult_conns
        ]
        
        # Find secret society connections
        secret_societies = self.find_secret_society_connections(term)
        structure["secret_society_connections"] = [
            {
                "name": s.name,
                "description": s.description,
                "connections": s.connections
            }
            for s in secret_societies
        ]
        
        # Find suppressed knowledge
        suppressed = self.find_suppressed_knowledge(term)
        structure["suppressed_knowledge"] = suppressed
        
        # Find hidden structures
        term_lower = term.lower()
        if term_lower in self.hidden_structures:
            structure["hidden_structures"] = self.hidden_structures[term_lower]
        
        # Build complete path
        structure["complete_path"] = self._build_complete_path(term, occult_conns, secret_societies, suppressed)
        
        logger.info(f"Decoded hidden structure for '{term}': "
                   f"{len(structure['occult_connections'])} occult, "
                   f"{len(structure['secret_society_connections'])} secret societies, "
                   f"{len(structure['suppressed_knowledge'])} suppressed")
        
        return structure
    
    def _find_related_terms(self, term: str) -> List[str]:
        """Find related terms."""
        related = {
            "astrology": ["astronomy", "star", "constellation", "zodiac", "planet", "celestial"],
            "star": ["astrology", "astronomy", "constellation", "celestial", "zodiac"],
            "constellation": ["astrology", "star", "zodiac", "celestial"],
            "zodiac": ["astrology", "constellation", "star", "celestial"],
            "planet": ["astrology", "star", "celestial", "zodiac"]
        }
        return related.get(term.lower(), [])
    
    def _build_complete_path(self, term: str, occult_conns: List[OccultConnection], 
                            secret_societies: List[SecretSociety], suppressed: List[str]) -> List[str]:
        """Build complete path from term to hidden knowledge."""
        path = [term]
        
        # Add occult connections
        for conn in occult_conns[:3]:  # Top 3
            path.append(f"→ {conn.target_term} ({conn.connection_type})")
        
        # Add secret societies
        for society in secret_societies[:2]:  # Top 2
            path.append(f"→ {society.name} (secret society)")
        
        # Add suppressed knowledge
        for supp in suppressed[:2]:  # Top 2
            path.append(f"→ {supp} (suppressed)")
        
        return path
    
    def _initialize_database(self) -> None:
        """Initialize occult knowledge database."""
        # Astrology occult connections
        self.occult_connections["astrology"] = [
            OccultConnection(
                connection_id="occult_astrology_1",
                source_term="astrology",
                target_term="Hermetic astrology",
                connection_type="occult",
                description="Hermetic tradition of astrology",
                confidence=0.8,
                evidence=["Hermetic texts", "Alchemical astrology"]
            ),
            OccultConnection(
                connection_id="occult_astrology_2",
                source_term="astrology",
                target_term="Kabbalistic astrology",
                connection_type="occult",
                description="Kabbalistic tradition of astrology",
                confidence=0.7,
                evidence=["Kabbalistic texts", "Tree of Life astrology"]
            ),
            OccultConnection(
                connection_id="occult_astrology_3",
                source_term="astrology",
                target_term="Alchemical astrology",
                connection_type="occult",
                description="Alchemical tradition of astrology",
                confidence=0.7,
                evidence=["Alchemical texts", "Planetary metals"]
            )
        ]
        
        # Star occult connections
        self.occult_connections["star"] = [
            OccultConnection(
                connection_id="occult_star_1",
                source_term="star",
                target_term="Star knowledge (occult)",
                connection_type="occult",
                description="Occult star knowledge",
                confidence=0.6,
                evidence=["Ancient star maps", "Celestial wisdom"]
            )
        ]
        
        # Secret societies
        self.secret_societies["rosicrucians"] = SecretSociety(
            society_id="rosicrucians",
            name="Rosicrucians",
            description="Hermetic Christian secret society",
            connections=["astrology", "alchemy", "hermeticism", "kabbalah"],
            occult_connections=["Hermetic astrology", "Alchemical astrology", "Kabbalistic astrology"],
            suppressed_knowledge=["Ancient star knowledge", "Precession of the equinoxes"],
            historical_patterns=["Renaissance", "17th century", "Hermetic revival"]
        )
        
        self.secret_societies["freemasons"] = SecretSociety(
            society_id="freemasons",
            name="Freemasons",
            description="Fraternal organization with esoteric traditions",
            connections=["astrology", "geometry", "architecture", "symbolism"],
            occult_connections=["Astrological symbolism", "Celestial architecture"],
            suppressed_knowledge=["Ancient building knowledge", "Sacred geometry"],
            historical_patterns=["Medieval", "Renaissance", "Modern"]
        )
        
        self.secret_societies["golden_dawn"] = SecretSociety(
            society_id="golden_dawn",
            name="Hermetic Order of the Golden Dawn",
            description="Hermetic secret society",
            connections=["astrology", "kabbalah", "hermeticism", "ceremonial magic"],
            occult_connections=["Hermetic astrology", "Kabbalistic astrology", "Ceremonial astrology"],
            suppressed_knowledge=["Hermetic texts", "Ancient magical knowledge"],
            historical_patterns=["Victorian era", "19th century", "Occult revival"]
        )
        
        # Suppressed knowledge
        self.suppressed_knowledge["astrology"] = [
            "Babylonian star knowledge (suppressed)",
            "Egyptian celestial wisdom (suppressed)",
            "Ancient star maps (suppressed)",
            "Precession of the equinoxes (suppressed knowledge)",
            "Ancient calendar systems (suppressed)"
        ]
        
        self.suppressed_knowledge["star"] = [
            "Ancient star catalogs (suppressed)",
            "Precession knowledge (suppressed)",
            "Star-based navigation (suppressed)"
        ]
        
        # Hidden structures
        self.hidden_structures["astrology"] = [
            "Celestial matrix structure",
            "Zodiacal knowledge structure",
            "Planetary influence structure",
            "Star-based language structure"
        ]
        
        logger.info("Initialized occult knowledge database: "
                   f"{len(self.occult_connections)} occult connections, "
                   f"{len(self.secret_societies)} secret societies, "
                   f"{len(self.suppressed_knowledge)} suppressed knowledge categories")

