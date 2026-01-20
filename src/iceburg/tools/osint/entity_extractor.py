"""
Entity Extractor - Named Entity Recognition for OSINT.
Extracts people, organizations, locations, and events from text.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """A named entity extracted from text."""
    name: str
    entity_type: str  # 'person', 'organization', 'location', 'date', 'money', 'event'
    mentions: int = 1
    context: str = ""
    confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "mentions": self.mentions,
            "context": self.context,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class EntityExtractor:
    """
    Named Entity Recognition using pattern matching and LLM.
    
    Extracts:
    - People (names)
    - Organizations (companies, governments, NGOs)
    - Locations (countries, cities, addresses)
    - Dates (specific dates, time periods)
    - Money (amounts, currencies)
    - Events (named events, incidents)
    """
    
    # Common organization suffixes
    ORG_PATTERNS = [
        r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Foundation|Institute|Association|Organization|Group|Partners|Holdings)\b',
        r'\b(?:The )?[A-Z][a-z]+ [A-Z][a-z]+ (?:Foundation|Institute|Association|Organization|Group)\b',
        r'\b[A-Z]{2,5}\b',  # Acronyms like FBI, CIA, CFR
    ]
    
    # Money patterns
    MONEY_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
        r'[\d,]+(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)',
        r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\b',
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:19|20)\d{2}\b',  # Years
    ]
    
    # Location patterns (basic - would be enhanced with gazetteer)
    LOCATION_KEYWORDS = [
        'Washington', 'New York', 'London', 'Moscow', 'Beijing', 'Brussels',
        'Geneva', 'Davos', 'United States', 'Russia', 'China', 'European Union',
        'Pentagon', 'White House', 'Kremlin', 'Capitol Hill', 'Wall Street'
    ]
    
    def __init__(self, use_llm: bool = True, cfg=None):
        """
        Initialize entity extractor.
        
        Args:
            use_llm: Whether to use LLM for enhanced extraction
            cfg: ICEBURG config for LLM access
        """
        self.use_llm = use_llm
        self.cfg = cfg
        self.provider = None
    
    def _get_provider(self):
        if self.provider is None and self.cfg:
            from ...providers.factory import provider_factory
            self.provider = provider_factory(self.cfg)
        return self.provider
    
    def extract(self, text: str, use_llm: bool = None) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            use_llm: Override default LLM setting
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Pattern-based extraction (fast)
        entities.extend(self._extract_patterns(text))
        
        # LLM-based extraction (more accurate)
        if (use_llm if use_llm is not None else self.use_llm):
            llm_entities = self._extract_with_llm(text)
            entities.extend(llm_entities)
        
        # Deduplicate and merge
        entities = self._deduplicate(entities)
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def _extract_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Extract organizations
        for pattern in self.ORG_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2:  # Skip very short matches
                    entities.append(ExtractedEntity(
                        name=match.strip(),
                        entity_type="organization",
                        confidence=0.6,
                        context="Pattern match"
                    ))
        
        # Extract money
        for pattern in self.MONEY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(ExtractedEntity(
                    name=match.strip(),
                    entity_type="money",
                    confidence=0.8,
                    context="Pattern match"
                ))
        
        # Extract dates
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append(ExtractedEntity(
                    name=match.strip(),
                    entity_type="date",
                    confidence=0.9,
                    context="Pattern match"
                ))
        
        # Extract locations
        for location in self.LOCATION_KEYWORDS:
            if location.lower() in text.lower():
                entities.append(ExtractedEntity(
                    name=location,
                    entity_type="location",
                    confidence=0.7,
                    context="Keyword match"
                ))
        
        return entities
    
    def _extract_with_llm(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using LLM."""
        entities = []
        provider = self._get_provider()
        
        if not provider:
            return entities
        
        try:
            import json
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Extract named entities from this text. Return JSON array:
[{{"name": "...", "type": "person|organization|location|date|money|event", "context": "brief context"}}]

Text (first 2500 chars):
{text[:2500]}

Return ONLY valid JSON array. Focus on:
- People (full names)
- Organizations (companies, governments, NGOs, secret societies)
- Key locations
- Significant dates
- Large money amounts
- Named events"""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Extract named entities accurately. Return valid JSON only.",
                temperature=0.2,
                options={"max_tokens": 800}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            entity_data = json.loads(response)
            
            for e in entity_data:
                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "unknown"),
                    context=e.get("context", ""),
                    confidence=0.8
                ))
                
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
        
        return entities
    
    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Deduplicate entities by name, keeping highest confidence."""
        seen = {}
        
        for entity in entities:
            key = entity.name.lower().strip()
            if key in seen:
                # Keep higher confidence version
                if entity.confidence > seen[key].confidence:
                    seen[key] = entity
                # Increment mentions
                seen[key].mentions += 1
            else:
                seen[key] = entity
        
        # Sort by mentions (most mentioned first)
        result = sorted(seen.values(), key=lambda e: e.mentions, reverse=True)
        return result


def extract_entities(text: str, cfg=None, use_llm: bool = True) -> List[ExtractedEntity]:
    """Convenience function for entity extraction."""
    extractor = EntityExtractor(use_llm=use_llm, cfg=cfg)
    return extractor.extract(text)
