"""
Entity Extractor - LLM-powered extraction of entities and relationships.
Uses local Ollama to extract structured data from raw documents.
"""

import json
import logging
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    entity_type: str  # person, organization, company, government, location
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_text: str = ""
    
    def generate_id(self) -> str:
        """Generate a stable ID for deduplication."""
        normalized = self.name.lower().strip()
        # Remove common suffixes for matching
        for suffix in [" inc", " llc", " corp", " corporation", " company", " ltd"]:
            normalized = normalized.replace(suffix, "")
        return f"{self.entity_type}_{hashlib.md5(normalized.encode()).hexdigest()[:12]}"


@dataclass
class ExtractedRelationship:
    """A relationship extracted from text."""
    source_name: str
    target_name: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_text: str = ""


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    raw_text: str
    source: str
    extracted_at: datetime = field(default_factory=datetime.now)


class EntityExtractor:
    """
    LLM-powered entity and relationship extraction.
    
    Uses local Ollama to extract structured data from raw text.
    """
    
    EXTRACTION_PROMPT = """You are an intelligence analyst extracting entities and relationships from documents.

Extract ALL entities (people, organizations, companies, government bodies, locations) and relationships from this text.

For each ENTITY, provide:
- name: The full name
- type: person/organization/company/government/location
- properties: Any relevant details (title, role, amount, date, etc.)

For each RELATIONSHIP, provide:
- source: Entity name
- target: Entity name  
- type: owns/funds/employs/leads/board_member/contributed_to/lobbied/connected_to
- properties: Details (amount, date, role, etc.)

Respond ONLY with valid JSON in this exact format:
{
  "entities": [
    {"name": "...", "type": "...", "properties": {...}}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "...", "properties": {...}}
  ]
}

TEXT TO ANALYZE:
"""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        provider_url: str = "http://localhost:11434"
    ):
        """
        Initialize the extractor.
        
        Args:
            model: Ollama model to use
            provider_url: Ollama server URL
        """
        self.model = model
        self.provider_url = provider_url
        self._provider = None
        
        logger.info(f"🔍 Entity Extractor initialized (model: {model})")
    
    def _get_provider(self):
        """Get or create LLM provider."""
        if self._provider is None:
            try:
                from ..providers.ollama_provider import OllamaProvider
                self._provider = OllamaProvider(
                    model=self.model,
                    base_url=self.provider_url
                )
            except ImportError:
                # Fallback to direct HTTP
                self._provider = None
        return self._provider
    
    async def extract(
        self,
        text: str,
        source: str = "unknown"
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Raw text to analyze
            source: Source identifier
            
        Returns:
            ExtractionResult with extracted data
        """
        if not text or len(text.strip()) < 50:
            return ExtractionResult(
                entities=[],
                relationships=[],
                raw_text=text,
                source=source
            )
        
        # Truncate very long text
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        try:
            # Build prompt
            prompt = self.EXTRACTION_PROMPT + text
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse response
            entities, relationships = self._parse_response(response, text)
            
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                raw_text=text,
                source=source
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Try fallback extraction
            return self._fallback_extraction(text, source)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for extraction."""
        import httpx
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.provider_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for structured output
                        "num_predict": 2000,
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")
    
    def _parse_response(
        self,
        response: str,
        source_text: str
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Parse LLM response into structured data."""
        entities = []
        relationships = []
        
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return entities, relationships
            
            data = json.loads(json_match.group())
            
            # Parse entities
            for e in data.get("entities", []):
                if not e.get("name"):
                    continue
                entities.append(ExtractedEntity(
                    name=e["name"],
                    entity_type=e.get("type", "unknown"),
                    properties=e.get("properties", {}),
                    source_text=source_text[:200]
                ))
            
            # Parse relationships
            for r in data.get("relationships", []):
                if not r.get("source") or not r.get("target"):
                    continue
                relationships.append(ExtractedRelationship(
                    source_name=r["source"],
                    target_name=r["target"],
                    relationship_type=r.get("type", "connected_to"),
                    properties=r.get("properties", {}),
                    source_text=source_text[:200]
                ))
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return entities, relationships
    
    def _fallback_extraction(
        self,
        text: str,
        source: str
    ) -> ExtractionResult:
        """
        Fallback extraction using regex patterns.
        Used when LLM extraction fails.
        """
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            "company": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Corporation|Company|Ltd))\b',
            ],
            "person": [
                r'\b((?:Mr|Mrs|Ms|Dr|Sen|Rep)\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)\b',  # Name M. Name
            ],
            "money": [
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?',
            ],
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                for match in matches[:10]:  # Limit matches
                    entities.append(ExtractedEntity(
                        name=match if isinstance(match, str) else match[0],
                        entity_type=entity_type,
                        confidence=0.6,
                        source_text=text[:200]
                    ))
        
        return ExtractionResult(
            entities=entities,
            relationships=[],
            raw_text=text,
            source=source
        )
    
    def extract_sync(self, text: str, source: str = "unknown") -> ExtractionResult:
        """Synchronous extraction (runs async in new event loop)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.extract(text, source)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.extract(text, source))
        except RuntimeError:
            return asyncio.run(self.extract(text, source))
