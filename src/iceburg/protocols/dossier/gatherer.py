"""
Gatherer Agent - Multi-source intelligence collection.
Collects raw intelligence from surface, alternative, academic, and deep sources.
Supports optional corpus/email ingest hook and silence/mention tracking.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import concurrent.futures

from ...config import IceburgConfig
from ...search.web_search import WebSearchAggregator, get_web_search
from ...providers.factory import provider_factory

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceSource:
    """Represents a single intelligence source."""
    url: str
    title: str
    content: str
    source_type: str  # 'surface', 'alternative', 'academic', 'historical', 'deep'
    credibility_score: float = 0.5
    bias_indicators: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligencePackage:
    """Collection of gathered intelligence on a topic."""
    query: str
    surface_sources: List[IntelligenceSource] = field(default_factory=list)
    alternative_sources: List[IntelligenceSource] = field(default_factory=list)
    academic_sources: List[IntelligenceSource] = field(default_factory=list)
    historical_sources: List[IntelligenceSource] = field(default_factory=list)
    deep_sources: List[IntelligenceSource] = field(default_factory=list)
    entities_found: List[Dict[str, Any]] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, str]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    # Optional: structured evidence objects built from sources
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_sources(self) -> int:
        return (len(self.surface_sources) + len(self.alternative_sources) + 
                len(self.academic_sources) + len(self.historical_sources) + 
                len(self.deep_sources))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_sources": self.total_sources,
            "surface_sources": [vars(s) for s in self.surface_sources],
            "alternative_sources": [vars(s) for s in self.alternative_sources],
            "academic_sources": [vars(s) for s in self.academic_sources],
            "historical_sources": [vars(s) for s in self.historical_sources],
            "deep_sources": [vars(s) for s in self.deep_sources],
            "entities_found": self.entities_found,
            "key_claims": self.key_claims,
            "contradictions": self.contradictions,
            "timestamp": self.timestamp.isoformat()
        }


class GathererAgent:
    """
    Multi-source intelligence gathering agent.
    
    Collects information from:
    - Surface: Mainstream news, Wikipedia, official statements
    - Alternative: Independent media, blogs, counter-narratives
    - Academic: arXiv, research papers, scholarly sources
    - Historical: Archives, old newspapers, declassified docs
    - Deep: Corporate filings, court records, public records
    """
    
    # Alternative/independent media domains to prioritize
    ALTERNATIVE_DOMAINS = [
        "zerohedge.com", "theintercept.com", "consortiumnews.com",
        "mintpressnews.com", "grayzone.com", "unlimitedhangout.com",
        "corbettreport.com", "activistpost.com", "truthdig.com"
    ]
    
    # Mainstream domains for surface layer
    SURFACE_DOMAINS = [
        "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
        "wsj.com", "guardian.com", "washingtonpost.com"
    ]

    # Keywords that suggest funding / ownership / power-structure questions
    FINANCE_POWER_KEYWORDS = [
        "fund", "funding", "pension", "sovereign wealth", "asset manager",
        "hedge fund", "private equity", "lp", "limited partner", "board",
        "director", "chairman", "vision fund", "investment authority"
    ]
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.web_search = get_web_search()
        self.provider = None
        self._corpus_ingest_hook: Optional[Callable[[Path], Dict[str, Any]]] = None

    def set_corpus_ingest_hook(self, hook: Callable[[Path], Dict[str, Any]]) -> None:
        """Register a hook for email/corpus ingest. Called by ingest_corpus(path)."""
        self._corpus_ingest_hook = hook

    def ingest_corpus(self, path: Path, **kwargs: Any) -> Dict[str, Any]:
        """
        Ingest a corpus (e.g. email export, document folder) for investigation.
        If a corpus ingest hook was set via set_corpus_ingest_hook, it is called;
        otherwise uses the built-in pipeline (load_corpus_from_path / ingest_corpus_for_dossier).
        """
        if self._corpus_ingest_hook:
            try:
                return self._corpus_ingest_hook(path, **kwargs)
            except Exception as e:
                logger.warning(f"Corpus ingest hook failed: {e}")
                return {"status": "error", "path": str(path), "error": str(e)}
        try:
            from .corpus_ingest import ingest_corpus_for_dossier
            max_files = int(kwargs.get("max_files", 10_000))
            return ingest_corpus_for_dossier(path, max_files=max_files)
        except Exception as e:
            logger.warning(f"Corpus ingest failed: {e}")
            return {"status": "error", "path": str(path), "error": str(e)}

    def _get_provider(self):
        if self.provider is None:
            self.provider = provider_factory(self.cfg)
        return self.provider
    
    def gather(
        self,
        query: str,
        depth: str = "standard",  # 'quick', 'standard', 'deep'
        thinking_callback: Optional[callable] = None
    ) -> IntelligencePackage:
        """
        Gather intelligence from multiple source layers.
        
        Args:
            query: Research query
            depth: How deep to search ('quick', 'standard', 'deep')
            thinking_callback: Optional callback for progress updates
            
        Returns:
            IntelligencePackage with gathered sources
        """
        package = IntelligencePackage(query=query)
        
        if thinking_callback:
            thinking_callback("🔍 Gathering surface intelligence...")
        
        # Layer 1: Surface sources (mainstream)
        package.surface_sources = self._gather_surface(query)
        logger.info(f"📰 Gathered {len(package.surface_sources)} surface sources")
        
        if thinking_callback:
            thinking_callback("🔍 Searching alternative sources...")
        
        # Layer 2: Alternative sources
        package.alternative_sources = self._gather_alternative(query)
        logger.info(f"📡 Gathered {len(package.alternative_sources)} alternative sources")
        
        if thinking_callback:
            thinking_callback("🔍 Searching academic literature...")
        
        # Layer 3: Academic sources
        package.academic_sources = self._gather_academic(query)
        logger.info(f"📚 Gathered {len(package.academic_sources)} academic sources")
        
        if depth in ["standard", "deep"]:
            if thinking_callback:
                thinking_callback("🔍 Searching historical archives...")
            
            # Layer 4: Historical sources
            package.historical_sources = self._gather_historical(query)
            logger.info(f"📜 Gathered {len(package.historical_sources)} historical sources")
        
        if depth == "deep" or self._query_looks_finance_or_power(query):
            if thinking_callback:
                thinking_callback("🔍 Searching deep records...")
            
            # Layer 5: Deep sources (corporate, legal)
            package.deep_sources = self._gather_deep(query)
            logger.info(f"🔎 Gathered {len(package.deep_sources)} deep sources")
        
        # Extract entities and claims
        if thinking_callback:
            thinking_callback("🧠 Extracting entities and claims...")
        
        package.entities_found = self._extract_entities(package)
        package.key_claims = self._extract_claims(package)
        package.contradictions = self._find_contradictions(package)
        
        logger.info(f"✅ Intelligence gathering complete: {package.total_sources} total sources")

        # Build structured evidence list from all sources for downstream modules
        try:
            from .evidence import Evidence, EvidenceStore
            store = EvidenceStore()
            for src in (
                package.surface_sources
                + package.alternative_sources
                + package.academic_sources
                + package.historical_sources
                + package.deep_sources
            ):
                store.add(Evidence.from_source(query, src))
            package.evidence = store.to_dicts()
        except Exception as e:  # pragma: no cover - evidence is best-effort
            logger.warning(f"Evidence build failed: {e}")

        return package
    
    def _gather_surface(self, query: str) -> List[IntelligenceSource]:
        """Gather mainstream/surface sources."""
        sources = []
        
        try:
            results = self.web_search.search(query, sources=['brave', 'ddg'], max_results_per_source=10)
            
            for result in results[:15]:
                # Check if from mainstream domain
                is_mainstream = any(domain in result.url for domain in self.SURFACE_DOMAINS)
                
                sources.append(IntelligenceSource(
                    url=result.url,
                    title=result.title,
                    content=result.snippet,
                    source_type='surface',
                    credibility_score=0.7 if is_mainstream else 0.5,
                    metadata={"source_api": result.source}
                ))
        except Exception as e:
            logger.warning(f"Surface gathering failed: {e}")
        
        return sources
    
    def _gather_alternative(self, query: str) -> List[IntelligenceSource]:
        """Gather alternative/independent sources."""
        sources = []
        
        # Search with alternative-focused queries
        alt_queries = [
            f"{query} alternative view",
            f"{query} hidden truth",
            f"{query} what they don't tell you"
        ]
        
        try:
            for alt_q in alt_queries[:2]:
                results = self.web_search.search(alt_q, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    is_alt = any(domain in result.url for domain in self.ALTERNATIVE_DOMAINS)
                    
                    sources.append(IntelligenceSource(
                        url=result.url,
                        title=result.title,
                        content=result.snippet,
                        source_type='alternative',
                        credibility_score=0.6 if is_alt else 0.4,
                        metadata={"query_variant": alt_q}
                    ))
        except Exception as e:
            logger.warning(f"Alternative gathering failed: {e}")
        
        return sources[:10]  # Limit to 10
    
    def _gather_academic(self, query: str) -> List[IntelligenceSource]:
        """Gather academic sources from arXiv."""
        sources = []
        
        try:
            results = self.web_search.arxiv.search(query, max_results=8)
            
            for result in results:
                sources.append(IntelligenceSource(
                    url=result.url,
                    title=result.title,
                    content=result.snippet,
                    source_type='academic',
                    credibility_score=0.85,
                    timestamp=result.published_date,
                    metadata={"source_api": "arxiv"}
                ))
        except Exception as e:
            logger.warning(f"Academic gathering failed: {e}")
        
        return sources
    
    def _gather_historical(self, query: str) -> List[IntelligenceSource]:
        """Gather historical sources."""
        sources = []
        
        # Search for historical context
        hist_queries = [
            f"{query} history",
            f"{query} origins",
            f"{query} historical precedent"
        ]
        
        try:
            for hist_q in hist_queries[:2]:
                results = self.web_search.search(hist_q, sources=['ddg'], max_results_per_source=5)
                
                for result in results:
                    sources.append(IntelligenceSource(
                        url=result.url,
                        title=result.title,
                        content=result.snippet,
                        source_type='historical',
                        credibility_score=0.6,
                        metadata={"query_variant": hist_q}
                    ))
        except Exception as e:
            logger.warning(f"Historical gathering failed: {e}")
        
        return sources[:8]
    
    def _query_looks_finance_or_power(self, query: str) -> bool:
        """Heuristic: query likely about funds/ownership/power structures."""
        lower = query.lower()
        return any(kw in lower for kw in self.FINANCE_POWER_KEYWORDS)

    def _query_looks_like_entity(self, query: str) -> bool:
        """Heuristic: query may be a company or person name (for OSINT hook)."""
        q = query.strip()
        if len(q) > 80:
            return False
        lower = q.lower()
        if lower.startswith(("how ", "what ", "why ", "when ", "where ", "is ", "are ", "did ", "does ")):
            return False
        if "?" in q:
            return False
        return True

    def _gather_deep_osint(self, query: str) -> List[IntelligenceSource]:
        """When depth=deep and query looks like entity, call OpenCorporates (and optionally OpenSecrets)."""
        sources = []
        try:
            from ...tools.osint.apis.opencorporates import OpenCorporatesClient
            client = OpenCorporatesClient()
            companies = client.search_companies(query, limit=5)
            for co in companies:
                content = f"Company: {co.name}. Jurisdiction: {co.jurisdiction_code}. Status: {co.status or 'unknown'}. Incorporation: {co.incorporation_date or 'unknown'}."
                officers = getattr(co, "officers", None) or []
                if officers:
                    content += " Officers: " + "; ".join(
                        f"{o.name} ({o.position})" if hasattr(o, "name") else str(o)
                        for o in (officers[:5] if isinstance(officers, list) else [])
                    )
                url = getattr(co, "registry_url", None) or (co.metadata or {}).get("opencorporates_url") or "#"
                sources.append(IntelligenceSource(
                    url=url,
                    title=co.name,
                    content=content,
                    source_type="deep",
                    credibility_score=0.8,
                    metadata={"source_api": "opencorporates", "company_number": co.company_number}
                ))
            if companies:
                logger.info(f"Deep OSINT: OpenCorporates returned {len(companies)} companies for query")
        except Exception as e:
            logger.debug(f"Deep OSINT (OpenCorporates) skipped or failed: {e}")
        return sources

    def _gather_deep(self, query: str) -> List[IntelligenceSource]:
        """Gather deep sources (corporate, legal records). Optionally call OSINT APIs when query looks like entity."""
        sources = []

        # When depth=deep and query looks like company/person name, call OpenCorporates
        if self._query_looks_like_entity(query):
            osint_sources = self._gather_deep_osint(query)
            sources.extend(osint_sources)

        # Web search for corporate/legal context
        deep_queries = [
            f"{query} SEC filing",
            f"{query} corporate records",
            f"{query} court case"
        ]

        try:
            for deep_q in deep_queries[:2]:
                results = self.web_search.search(deep_q, sources=['ddg'], max_results_per_source=3)

                for result in results:
                    sources.append(IntelligenceSource(
                        url=result.url,
                        title=result.title,
                        content=result.snippet,
                        source_type='deep',
                        credibility_score=0.7,
                        metadata={"query_variant": deep_q}
                    ))
        except Exception as e:
            logger.warning(f"Deep gathering failed: {e}")

        return sources[:6]
    
    def _extract_entities(self, package: IntelligencePackage) -> List[Dict[str, Any]]:
        """Extract named entities from gathered sources."""
        entities = []
        
        # Combine all source content
        all_content = " ".join([
            s.content for sources in [
                package.surface_sources, package.alternative_sources,
                package.academic_sources, package.historical_sources,
                package.deep_sources
            ] for s in sources
        ])
        
        if not all_content:
            return entities
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Extract key entities from this text. Return JSON array:
[{{"name": "...", "type": "person|organization|location|event", "mentions": N, "context": "brief context"}}]

Text (first 3000 chars):
{all_content[:3000]}

Return ONLY valid JSON array, no other text."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Extract named entities accurately. Return valid JSON only.",
                temperature=0.2,
                options={"max_tokens": 800}
            )
            
            from .llm_json import parse_llm_json
            entities = parse_llm_json(response, default=[], log_context="entity extraction")
            if not isinstance(entities, list):
                entities = [entities] if entities is not None else []
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            entities = []
        
        return entities[:20]  # Limit to 20 entities
    
    def _extract_claims(self, package: IntelligencePackage) -> List[str]:
        """Extract key claims from sources."""
        claims = []
        
        # Combine surface and alternative for claim extraction
        all_content = " ".join([
            s.content for s in package.surface_sources + package.alternative_sources
        ])
        
        if not all_content:
            return claims
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Extract the main claims/assertions from this text.
Return a JSON array of claim strings:
["Claim 1", "Claim 2", ...]

Text (first 2000 chars):
{all_content[:2000]}

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Extract factual claims accurately.",
                temperature=0.2,
                options={"max_tokens": 500}
            )
            
            from .llm_json import parse_llm_json
            claims = parse_llm_json(response, default=[], log_context="claim extraction")
            if not isinstance(claims, list):
                claims = [str(claims)] if claims is not None else []
            
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            claims = []
        
        return claims[:10]
    
    def _find_contradictions(self, package: IntelligencePackage) -> List[Dict[str, str]]:
        """Find contradictions between surface and alternative sources."""
        contradictions = []
        
        if not package.surface_sources or not package.alternative_sources:
            return contradictions
        
        surface_content = " ".join([s.content for s in package.surface_sources[:5]])
        alt_content = " ".join([s.content for s in package.alternative_sources[:5]])
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Compare these two perspectives and find contradictions.

MAINSTREAM VIEW:
{surface_content[:1500]}

ALTERNATIVE VIEW:
{alt_content[:1500]}

Return JSON array of contradictions:
[{{"mainstream": "claim from mainstream", "alternative": "contradicting claim", "topic": "what this is about"}}]

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Find genuine contradictions between viewpoints.",
                temperature=0.3,
                options={"max_tokens": 600}
            )
            
            from .llm_json import parse_llm_json
            contradictions = parse_llm_json(response, default=[], log_context="contradiction finding")
            if not isinstance(contradictions, list):
                contradictions = []
            
        except Exception as e:
            logger.warning(f"Contradiction finding failed: {e}")
            contradictions = []
        
        return contradictions[:5]


def gather_intelligence(cfg: IceburgConfig, query: str, depth: str = "standard") -> IntelligencePackage:
    """Convenience function for gathering intelligence."""
    agent = GathererAgent(cfg)
    return agent.gather(query, depth=depth)
