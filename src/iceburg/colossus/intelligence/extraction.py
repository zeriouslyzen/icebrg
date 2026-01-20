"""
COLOSSUS Entity Extraction

AI-powered entity and relationship extraction from text.
Uses local Ollama for M4-optimized inference.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity types for extraction."""
    PERSON = "person"
    COMPANY = "company"
    ORGANIZATION = "organization"
    LOCATION = "location"
    ADDRESS = "address"
    MONEY = "money"
    DATE = "date"
    POSITION = "position"


class RelationType(str, Enum):
    """Relationship types."""
    OWNS = "OWNS"
    DIRECTOR_OF = "DIRECTOR_OF"
    EMPLOYEE_OF = "EMPLOYEE_OF"
    FAMILY_OF = "FAMILY_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    LOCATED_AT = "LOCATED_AT"
    TRANSFERRED_TO = "TRANSFERRED_TO"
    SANCTIONED_BY = "SANCTIONED_BY"


@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    name: str
    entity_type: EntityType
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelationship:
    """Relationship extracted from text."""
    source: str
    target: str
    relationship_type: RelationType
    confidence: float
    evidence: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """
    Extract entities and relationships from text.
    
    Uses Ollama for LLM-powered extraction with M4 Metal acceleration.
    Falls back to spaCy NER for basic extraction.
    """
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        ollama_host: str = "http://localhost:11434",
        use_spacy_fallback: bool = True,
    ):
        """
        Initialize extractor.
        
        Args:
            model: Ollama model for extraction
            ollama_host: Ollama API endpoint
            use_spacy_fallback: Use spaCy if LLM unavailable
        """
        self.model = model
        self.ollama_host = ollama_host
        self.use_spacy_fallback = use_spacy_fallback
        
        self._ollama_available = False
        self._spacy_nlp = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize extraction backends."""
        # Check Ollama availability
        try:
            import httpx
            response = httpx.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                self._ollama_available = True
                logger.info(f"✅ Ollama available: {self.model}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama unavailable: {e}")
        
        # Load spaCy as fallback
        if self.use_spacy_fallback:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("📊 spaCy NER loaded as fallback")
            except Exception as e:
                logger.warning(f"⚠️ spaCy unavailable: {e}")
    
    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[EntityType]] = None,
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            entity_types: Types to extract (all if None)
            
        Returns:
            List of extracted entities
        """
        if self._ollama_available:
            return self._llm_extract_entities(text, entity_types)
        elif self._spacy_nlp:
            return self._spacy_extract_entities(text, entity_types)
        else:
            logger.error("No extraction backend available")
            return []
    
    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[ExtractedEntity]] = None,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships from text.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (extract if None)
            
        Returns:
            List of extracted relationships
        """
        if not entities:
            entities = self.extract_entities(text)
        
        if self._ollama_available:
            return self._llm_extract_relationships(text, entities)
        else:
            # Basic pattern matching fallback
            return self._pattern_extract_relationships(text, entities)
    
    def extract_all(
        self,
        text: str
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """
        Extract both entities and relationships.
        
        Returns:
            Tuple of (entities, relationships)
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)
        return entities, relationships
    
    # ==================== LLM Extraction ====================
    
    def _llm_extract_entities(
        self,
        text: str,
        entity_types: Optional[List[EntityType]]
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM."""
        import httpx
        import json
        
        type_list = ", ".join([t.value for t in entity_types]) if entity_types else "all types"
        
        prompt = f"""Extract all named entities from the following text.
For each entity, provide:
- name: The entity name as it appears
- type: One of [person, company, organization, location, address, money, date, position]
- confidence: 0.0 to 1.0

Focus on: {type_list}

Respond with a JSON array only, no explanation.

Text:
{text[:2000]}

JSON:"""

        try:
            response = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            
            result = response.json()
            output = result.get("response", "[]")
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
                return [
                    ExtractedEntity(
                        name=e.get("name", ""),
                        entity_type=EntityType(e.get("type", "person").lower()),
                        confidence=float(e.get("confidence", 0.8)),
                    )
                    for e in entities_data
                    if e.get("name")
                ]
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
        
        return []
    
    def _llm_extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelationship]:
        """Extract relationships using LLM."""
        import httpx
        import json
        
        entity_names = [e.name for e in entities]
        
        prompt = f"""Given these entities: {entity_names}

Extract all relationships between them from the text.
For each relationship, provide:
- source: Entity name
- target: Entity name
- type: One of [OWNS, DIRECTOR_OF, EMPLOYEE_OF, FAMILY_OF, ASSOCIATED_WITH, LOCATED_AT, TRANSFERRED_TO, SANCTIONED_BY]
- evidence: Quote from text supporting this
- confidence: 0.0 to 1.0

Respond with a JSON array only.

Text:
{text[:2000]}

JSON:"""

        try:
            response = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            
            result = response.json()
            output = result.get("response", "[]")
            
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                rels_data = json.loads(json_match.group())
                return [
                    ExtractedRelationship(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        relationship_type=RelationType(r.get("type", "ASSOCIATED_WITH")),
                        confidence=float(r.get("confidence", 0.7)),
                        evidence=r.get("evidence", ""),
                    )
                    for r in rels_data
                    if r.get("source") and r.get("target")
                ]
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
        
        return []
    
    # ==================== spaCy Extraction ====================
    
    def _spacy_extract_entities(
        self,
        text: str,
        entity_types: Optional[List[EntityType]]
    ) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        doc = self._spacy_nlp(text)
        
        # Map spaCy labels to our types
        label_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.ADDRESS,
            "MONEY": EntityType.MONEY,
            "DATE": EntityType.DATE,
        }
        
        entities = []
        for ent in doc.ents:
            entity_type = label_map.get(ent.label_)
            if not entity_type:
                continue
            
            if entity_types and entity_type not in entity_types:
                continue
            
            entities.append(ExtractedEntity(
                name=ent.text,
                entity_type=entity_type,
                confidence=0.7,  # spaCy doesn't provide confidence
                start_pos=ent.start_char,
                end_pos=ent.end_char,
            ))
        
        return entities
    
    def _pattern_extract_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedRelationship]:
        """Extract relationships using pattern matching."""
        relationships = []
        
        # Simple patterns
        patterns = [
            (r"(\w+)\s+(?:is|was)\s+(?:the\s+)?(?:owner|founder|CEO|chairman)\s+of\s+(\w+)", RelationType.DIRECTOR_OF),
            (r"(\w+)\s+owns?\s+(\w+)", RelationType.OWNS),
            (r"(\w+)\s+(?:is|was)\s+(?:married|related)\s+to\s+(\w+)", RelationType.FAMILY_OF),
            (r"(\w+)\s+works?\s+(?:for|at)\s+(\w+)", RelationType.EMPLOYEE_OF),
        ]
        
        entity_names = {e.name.lower(): e.name for e in entities}
        
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source = match.group(1).lower()
                target = match.group(2).lower()
                
                if source in entity_names and target in entity_names:
                    relationships.append(ExtractedRelationship(
                        source=entity_names[source],
                        target=entity_names[target],
                        relationship_type=rel_type,
                        confidence=0.6,
                        evidence=match.group(0),
                    ))
        
        return relationships
