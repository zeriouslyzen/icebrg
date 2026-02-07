"""
Dossier Synthesizer - Combines all intelligence into final ICEBURG Dossier.
The final output of the Dossier Protocol.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from ...config import IceburgConfig
from ...providers.factory import provider_factory
from .gatherer import GathererAgent, IntelligencePackage
from .decoder import DecoderAgent, DecoderReport
from .mapper import MapperAgent, MapperReport

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """A single event in the timeline."""
    date: str
    event: str
    significance: str = ""
    sources: List[str] = field(default_factory=list)


@dataclass
class DossierSection:
    """A section of the dossier."""
    title: str
    content: str
    subsections: List[Dict[str, str]] = field(default_factory=list)
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    sources: List[str] = field(default_factory=list)


@dataclass
class IcebergDossier:
    """
    The complete ICEBURG Dossier output.
    A comprehensive intelligence report on a topic.
    """
    query: str
    executive_summary: str = ""
    official_narrative: str = ""
    alternative_narratives: List[str] = field(default_factory=list)
    key_players: List[Dict[str, Any]] = field(default_factory=list)
    network_map: Dict[str, Any] = field(default_factory=dict)
    timeline: List[TimelineEvent] = field(default_factory=list)
    symbol_analysis: Dict[str, Any] = field(default_factory=dict)
    historical_parallels: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, str]] = field(default_factory=list)
    hidden_connections: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_research: List[str] = field(default_factory=list)
    confidence_ratings: Dict[str, str] = field(default_factory=dict)
    sources: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "executive_summary": self.executive_summary,
            "official_narrative": self.official_narrative,
            "alternative_narratives": self.alternative_narratives,
            "key_players": self.key_players,
            "network_map": self.network_map,
            "timeline": [vars(e) for e in self.timeline],
            "symbol_analysis": self.symbol_analysis,
            "historical_parallels": self.historical_parallels,
            "contradictions": self.contradictions,
            "hidden_connections": self.hidden_connections,
            "follow_up_research": self.follow_up_research,
            "confidence_ratings": self.confidence_ratings,
            "sources": self.sources,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_markdown(self) -> str:
        """Render dossier as markdown for display."""
        md = []
        
        md.append(f"# 🧊 ICEBURG DOSSIER")
        md.append(f"## Query: {self.query}")
        md.append(f"*Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M')}*")
        md.append("")
        
        md.append("---")
        md.append("")
        
        # Executive Summary
        md.append("## 📋 Executive Summary")
        md.append(self.executive_summary)
        md.append("")
        
        # Official vs Alternative
        md.append("---")
        md.append("")
        md.append("## 📰 Official Narrative")
        md.append(self.official_narrative)
        md.append("")
        
        if self.alternative_narratives:
            md.append("## 🔍 Alternative Narratives")
            for i, alt in enumerate(self.alternative_narratives, 1):
                md.append(f"**{i}.** {alt}")
                md.append("")
        
        # Key Players
        if self.key_players:
            md.append("---")
            md.append("")
            md.append("## 👥 Key Players")
            md.append("")
            md.append("| Name | Type | Connections | Notes |")
            md.append("|------|------|-------------|-------|")
            for player in self.key_players[:10]:
                name = player.get("name", "Unknown")
                ptype = player.get("type", "unknown")
                conns = player.get("connections", 0)
                desc = player.get("description", "")[:50]
                md.append(f"| {name} | {ptype} | {conns} | {desc} |")
            md.append("")
        
        # Contradictions
        if self.contradictions:
            md.append("---")
            md.append("")
            md.append("## ⚔️ Narrative Contradictions")
            md.append("")
            for contr in self.contradictions[:5]:
                md.append(f"**Topic:** {contr.get('topic', 'Unknown')}")
                md.append(f"- **Mainstream:** {contr.get('mainstream', '')}")
                md.append(f"- **Alternative:** {contr.get('alternative', '')}")
                md.append("")
        
        # Symbol Analysis
        if self.symbol_analysis:
            md.append("---")
            md.append("")
            md.append("## 🔮 Symbol & Pattern Analysis")
            md.append("")
            
            if self.symbol_analysis.get("symbols_detected"):
                md.append("### Symbols Detected")
                for sym in self.symbol_analysis["symbols_detected"][:5]:
                    md.append(f"- **{sym.get('symbol_name', '')}**: {sym.get('meaning', '')}")
                md.append("")
            
            if self.symbol_analysis.get("timing_analysis"):
                md.append("### Timing Significance")
                for timing in self.symbol_analysis["timing_analysis"][:3]:
                    md.append(f"- **{timing.get('date', '')}**: {timing.get('significance', '')}")
                md.append("")
            
            if self.symbol_analysis.get("society_connections"):
                md.append("### Society Connections")
                for conn in self.symbol_analysis["society_connections"][:3]:
                    md.append(f"- **{conn.get('society', '')}**: {conn.get('match_type', '')}")
                md.append("")
        
        # Hidden Connections
        if self.hidden_connections:
            md.append("---")
            md.append("")
            md.append("## 🕸️ Hidden Connections")
            md.append("")
            for conn in self.hidden_connections[:5]:
                md.append(f"- **{conn.get('entity_1', '')}** ↔ **{conn.get('entity_2', '')}** via *{conn.get('connected_via', '')}*")
            md.append("")
        
        # Historical Parallels
        if self.historical_parallels:
            md.append("---")
            md.append("")
            md.append("## 📜 Historical Parallels")
            md.append("")
            for parallel in self.historical_parallels[:5]:
                md.append(f"- {parallel}")
            md.append("")
        
        # Timeline
        if self.timeline:
            md.append("---")
            md.append("")
            md.append("## 📅 Timeline")
            md.append("")
            md.append("| Date | Event | Significance |")
            md.append("|------|-------|--------------|")
            for event in self.timeline[:10]:
                md.append(f"| {event.date} | {event.event} | {event.significance} |")
            md.append("")
        
        # Follow-up Research
        if self.follow_up_research:
            md.append("---")
            md.append("")
            md.append("## 🔬 Recommended Follow-up Research")
            md.append("")
            for i, research in enumerate(self.follow_up_research[:5], 1):
                md.append(f"{i}. {research}")
            md.append("")
        
        # Confidence Ratings
        if self.confidence_ratings:
            md.append("---")
            md.append("")
            md.append("## 📊 Confidence Ratings")
            md.append("")
            md.append("| Finding | Confidence |")
            md.append("|---------|------------|")
            for finding, conf in self.confidence_ratings.items():
                md.append(f"| {finding} | {conf} |")
            md.append("")
        
        # Sources
        if self.sources:
            md.append("---")
            md.append("")
            md.append("## 📚 Sources")
            md.append("")
            for i, source in enumerate(self.sources[:20], 1):
                title = source.get("title", "Unknown")[:60]
                url = source.get("url", "")
                md.append(f"{i}. [{title}]({url})")
            md.append("")
        
        return "\n".join(md)


class DossierSynthesizer:
    """
    Combines output from Gatherer, Decoder, and Mapper into final dossier.
    
    The main orchestrator of the Dossier Protocol.
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.gatherer = GathererAgent(cfg)
        self.decoder = DecoderAgent(cfg)
        self.mapper = MapperAgent(cfg)
        self.provider = None
    
    def _get_provider(self):
        if self.provider is None:
            self.provider = provider_factory(self.cfg)
        return self.provider
    
    def generate_dossier(
        self,
        query: str,
        depth: str = "standard",  # 'quick', 'standard', 'deep'
        thinking_callback: Optional[callable] = None
    ) -> IcebergDossier:
        """
        Generate a complete ICEBURG Dossier.
        
        Args:
            query: Research topic
            depth: How deep to investigate
            thinking_callback: Progress callback
            
        Returns:
            Complete IcebergDossier
        """
        dossier = IcebergDossier(query=query)
        start_time = datetime.now()
        
        # === PHASE 1: GATHER ===
        if thinking_callback:
            thinking_callback("📡 Phase 1: Gathering multi-source intelligence...")
        
        intelligence = self.gatherer.gather(query, depth=depth, thinking_callback=thinking_callback)
        
        # Store sources
        dossier.sources = self._compile_sources(intelligence)
        
        # === PHASE 2: DECODE ===
        if thinking_callback:
            thinking_callback("🔮 Phase 2: Decoding symbols and patterns...")
        
        # Build content string for decoder
        content = self._combine_intelligence_content(intelligence)
        
        decoder_report = self.decoder.decode(query, content, thinking_callback=thinking_callback)
        dossier.symbol_analysis = decoder_report.to_dict()
        
        # === PHASE 3: MAP ===
        if thinking_callback:
            thinking_callback("🗺️ Phase 3: Mapping network relationships...")
        
        mapper_report = self.mapper.map_network(
            query,
            entities_from_gatherer=intelligence.entities_found,
            content=content,
            thinking_callback=thinking_callback,
            decoder_report=decoder_report,
        )

        dossier.network_map = mapper_report.network.to_dict()
        dossier.key_players = mapper_report.key_players
        dossier.hidden_connections = mapper_report.hidden_connections
        
        # === PHASE 4: SYNTHESIZE ===
        if thinking_callback:
            thinking_callback("📝 Phase 4: Synthesizing final dossier...")
        
        # Generate narratives
        dossier.official_narrative = self._generate_official_narrative(intelligence, query)
        dossier.alternative_narratives = self._generate_alternative_narratives(intelligence, query)
        
        # Copy contradictions from intelligence
        dossier.contradictions = intelligence.contradictions
        
        # Generate executive summary
        dossier.executive_summary = self._generate_executive_summary(
            query, intelligence, decoder_report, mapper_report
        )
        
        # Generate timeline
        dossier.timeline = self._build_timeline(query, content)
        
        # Find historical parallels
        dossier.historical_parallels = self._find_historical_parallels(query, content)
        
        # Generate follow-up research suggestions
        dossier.follow_up_research = self._suggest_follow_up(query, intelligence, decoder_report)
        
        # Assign confidence ratings
        dossier.confidence_ratings = self._assign_confidence(intelligence, decoder_report)
        
        # Metadata
        end_time = datetime.now()
        dossier.metadata = {
            "generation_time_seconds": (end_time - start_time).total_seconds(),
            "depth": depth,
            "total_sources": intelligence.total_sources,
            "entities_found": len(intelligence.entities_found),
            "symbols_detected": len(decoder_report.symbols_detected),
            "relationships_mapped": len(mapper_report.network.relationships)
        }
        
        # === PHASE 5: INGEST ===
        # Auto-ingest into Colossus Graph for Pegasus/Search availability
        if thinking_callback:
            thinking_callback("💾 Phase 5: Indexing dossier into Pegasus Matrix...")
        
        self._ingest_to_colossus(dossier)
        
        logger.info(f"✅ Dossier generation complete in {dossier.metadata['generation_time_seconds']:.1f}s")
        return dossier

    def _ingest_to_colossus(self, dossier: IcebergDossier):
        """Best-effort ingestion into Colossus Graph."""
        try:
            # Local import to avoid circular dependency
            from ...colossus.api import get_graph
            graph = get_graph()
            
            # Check if graph supports ingestion (it should)
            if hasattr(graph, 'ingest_dossier'):
                graph.ingest_dossier(dossier.to_dict())
                logger.info(f"✅ Dossier '{dossier.query}' ingested into Colossus")
            else:
                logger.warning("Colossus Graph missing ingest_dossier method")
                
        except Exception as e:
            # Non-blocking error - verification is optional
            logger.warning(f"Failed to ingest dossier into Colossus: {e}")
    
    def _compile_sources(self, intelligence: IntelligencePackage) -> List[Dict[str, str]]:
        """Compile all sources into a list."""
        sources = []
        
        for source_list in [
            intelligence.surface_sources,
            intelligence.alternative_sources,
            intelligence.academic_sources,
            intelligence.historical_sources,
            intelligence.deep_sources
        ]:
            for src in source_list:
                sources.append({
                    "title": src.title,
                    "url": src.url,
                    "type": src.source_type
                })
        
        return sources
    
    def _combine_intelligence_content(self, intelligence: IntelligencePackage) -> str:
        """Combine all source content into one string."""
        parts = []
        
        for source_list in [
            intelligence.surface_sources,
            intelligence.alternative_sources,
            intelligence.academic_sources
        ]:
            for src in source_list[:5]:  # Limit per category
                parts.append(src.content)
        
        return " ".join(parts)
    
    def _generate_official_narrative(self, intelligence: IntelligencePackage, query: str) -> str:
        """Generate the official/mainstream narrative summary."""
        surface_content = " ".join([s.content for s in intelligence.surface_sources[:5]])
        
        if not surface_content:
            return "No mainstream sources found."
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Summarize the mainstream/official narrative on this topic.
            
Topic: {query}
Sources: {surface_content[:2000]}

Write a concise 2-3 paragraph summary of the official position. Be objective about what the mainstream says."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Summarize mainstream narratives objectively and concisely.",
                temperature=0.3,
                options={"max_tokens": 400}
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Official narrative generation failed: {e}")
            return "Unable to generate official narrative summary."
    
    def _generate_alternative_narratives(self, intelligence: IntelligencePackage, query: str) -> List[str]:
        """Generate alternative narrative summaries."""
        alt_content = " ".join([s.content for s in intelligence.alternative_sources[:5]])
        
        if not alt_content:
            return ["No alternative sources found."]
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Identify alternative/counter-mainstream narratives on this topic.
            
Topic: {query}
Alternative Sources: {alt_content[:2000]}

Return a JSON array of 2-3 distinct alternative narratives (1-2 sentences each):
["Alternative view 1", "Alternative view 2", ...]

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Identify alternative narratives objectively.",
                temperature=0.4,
                options={"max_tokens": 400}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            return json.loads(response)
            
        except Exception as e:
            logger.warning(f"Alternative narratives generation failed: {e}")
            return ["Unable to generate alternative narratives."]
    
    def _generate_executive_summary(
        self,
        query: str,
        intelligence: IntelligencePackage,
        decoder_report: DecoderReport,
        mapper_report: MapperReport
    ) -> str:
        """Generate executive summary of entire dossier."""
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            # Build context
            context = f"""
Topic: {query}

SOURCES: {intelligence.total_sources} sources gathered
- Surface: {len(intelligence.surface_sources)}
- Alternative: {len(intelligence.alternative_sources)}
- Academic: {len(intelligence.academic_sources)}

KEY CLAIMS: {intelligence.key_claims[:3]}

CONTRADICTIONS: {len(intelligence.contradictions)} narrative contradictions found

ENTITIES: {len(intelligence.entities_found)} key entities identified

SYMBOLS: {len(decoder_report.symbols_detected)} symbols detected

NETWORK: {len(mapper_report.network.entities)} entities, {len(mapper_report.network.relationships)} relationships

KEY PLAYERS: {[p.get('name') for p in mapper_report.key_players[:5]]}
"""
            
            prompt = f"""Write a concise executive summary for this intelligence dossier.

{context}

The summary should:
1. State the main topic
2. Highlight key findings
3. Note the most important players/entities
4. Mention any significant contradictions or hidden patterns
5. Be 3-4 sentences maximum"""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Write concise, insightful executive summaries.",
                temperature=0.4,
                options={"max_tokens": 300}
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return f"Dossier on '{query}' compiled from {intelligence.total_sources} sources."
    
    def _build_timeline(self, query: str, content: str) -> List[TimelineEvent]:
        """Build a timeline of events."""
        events = []
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Extract a timeline of events related to this topic.

Topic: {query}
Content: {content[:2000]}

Return a JSON array of timeline events:
[{{"date": "YYYY-MM-DD or description", "event": "what happened", "significance": "why it matters"}}]

Return ONLY valid JSON array. Maximum 5 events."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Extract timeline events accurately.",
                temperature=0.2,
                options={"max_tokens": 500}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            event_data = json.loads(response)
            
            for event in event_data:
                events.append(TimelineEvent(
                    date=event.get("date", "Unknown"),
                    event=event.get("event", ""),
                    significance=event.get("significance", "")
                ))
                
        except Exception as e:
            logger.debug(f"Timeline building failed: {e}")
        
        return events
    
    def _find_historical_parallels(self, query: str, content: str) -> List[str]:
        """Find historical parallels to current events."""
        parallels = []
        
        try:
            provider = self._get_provider()
            model = getattr(self.cfg, "surveyor_model", None) or "gemini-2.0-flash-exp"
            
            prompt = f"""Identify historical parallels or precedents for this topic.

Topic: {query}
Context: {content[:1500]}

What similar events, patterns, or situations have occurred in history?

Return a JSON array of 2-3 historical parallels:
["Historical parallel 1", "Historical parallel 2", ...]

Return ONLY valid JSON array."""
            
            response = provider.chat_complete(
                model=model,
                prompt=prompt,
                system="Find meaningful historical parallels. Be specific about dates and events.",
                temperature=0.4,
                options={"max_tokens": 400}
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            parallels = json.loads(response)
            
        except Exception as e:
            logger.debug(f"Historical parallels failed: {e}")
        
        return parallels
    
    def _suggest_follow_up(
        self,
        query: str,
        intelligence: IntelligencePackage,
        decoder_report: DecoderReport
    ) -> List[str]:
        """Suggest follow-up research topics."""
        suggestions = []
        
        # Based on entities found
        if intelligence.entities_found:
            top_entities = [e.get("name") for e in intelligence.entities_found[:3]]
            suggestions.append(f"Investigate backgrounds of: {', '.join(top_entities)}")
        
        # Based on contradictions
        if intelligence.contradictions:
            suggestions.append("Verify contradicting claims with primary sources")
        
        # Based on society connections
        if decoder_report.society_connections:
            societies = [c.get("society") for c in decoder_report.society_connections[:2]]
            suggestions.append(f"Research connections to: {', '.join(societies)}")
        
        # Based on symbols
        if decoder_report.symbols_detected:
            suggestions.append("Analyze visual materials for additional symbolic patterns")
        
        # General
        suggestions.append("Check financial disclosures and corporate filings")
        suggestions.append("Search declassified documents related to topic")
        
        return suggestions[:5]
    
    def _assign_confidence(
        self,
        intelligence: IntelligencePackage,
        decoder_report: DecoderReport
    ) -> Dict[str, str]:
        """Assign confidence ratings to different findings."""
        ratings = {}
        
        # Source-based confidence
        if intelligence.total_sources > 20:
            ratings["Overall Source Coverage"] = "HIGH"
        elif intelligence.total_sources > 10:
            ratings["Overall Source Coverage"] = "MEDIUM"
        else:
            ratings["Overall Source Coverage"] = "LOW"
        
        # Academic confidence
        if len(intelligence.academic_sources) > 3:
            ratings["Academic Backing"] = "HIGH"
        elif len(intelligence.academic_sources) > 0:
            ratings["Academic Backing"] = "MEDIUM"
        else:
            ratings["Academic Backing"] = "LOW"
        
        # Symbol confidence (lower because interpretive)
        if decoder_report.symbols_detected:
            ratings["Symbol Analysis"] = "INTERPRETIVE"
        
        # Contradiction detection
        if intelligence.contradictions:
            ratings["Narrative Contradictions"] = "MEDIUM"
        
        return ratings


def generate_dossier(cfg: IceburgConfig, query: str, depth: str = "standard") -> IcebergDossier:
    """Convenience function to generate a dossier."""
    synthesizer = DossierSynthesizer(cfg)
    return synthesizer.generate_dossier(query, depth=depth)
