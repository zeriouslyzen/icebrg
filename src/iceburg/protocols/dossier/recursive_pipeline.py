"""
Recursive Dossier Pipeline - Deep investigation with automatic gap detection.
Generates multi-layer dossiers by using CuriosityEngine to identify what's missing
and recursively investigating sub-questions.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio

from ...config import IceburgConfig, load_config
from ...curiosity.curiosity_engine import CuriosityEngine, KnowledgeGap
from .synthesizer import DossierSynthesizer, IcebergDossier
from .gatherer import GathererAgent

logger = logging.getLogger(__name__)


@dataclass
class DossierLayer:
    """Represents one layer of investigation."""
    depth: int
    query: str
    dossier: IcebergDossier
    gaps_detected: List[KnowledgeGap] = field(default_factory=list)
    sub_layers: List['DossierLayer'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DeepDossier:
    """
    Multi-layered dossier with recursive investigation results.
    Extends the concept of IcebergDossier to include depth.
    """
    query: str
    root_layer: DossierLayer
    total_layers: int = 0
    total_sources: int = 0
    max_depth_reached: int = 0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "total_layers": self.total_layers,
            "total_sources": self.total_sources,
            "max_depth_reached": self.max_depth_reached,
            "contradictions": self.contradictions,
            "synthesis": self.synthesis,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_markdown(self) -> str:
        """Render as markdown for display."""
        md = []
        
        # Header
        md.append(f"# 🧊 ICEBURG DEEP DOSSIER")
        md.append(f"\n**Query:** {self.query}")
        md.append(f"\n**Depth Reached:** {self.max_depth_reached} layers")
        md.append(f"\n**Total Sources:** {self.total_sources}")
        md.append(f"\n**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M')}\n")
        
        # Synthesis
        md.append("---\n")
        md.append("## 📊 Deep Synthesis\n")
        md.append(self.synthesis)
        
        # Contradictions
        if self.contradictions:
            md.append("\n---\n")
            md.append("## ⚠️ Contradictions Detected\n")
            for c in self.contradictions:
                md.append(f"- **{c.get('finding1', '')}** vs **{c.get('finding2', '')}**")
                md.append(f"  - Source 1: {c.get('source1', 'Unknown')}")
                md.append(f"  - Source 2: {c.get('source2', 'Unknown')}\n")
        
        # Layer details
        md.append("\n---\n")
        md.append("## 🔍 Investigation Layers\n")
        md.append(self._render_layer(self.root_layer))
        
        return "\n".join(md)
    
    def _render_layer(self, layer: DossierLayer, indent: int = 0) -> str:
        """Recursively render layer details."""
        prefix = "  " * indent
        md = []
        
        md.append(f"{prefix}### Layer {layer.depth}: {layer.query[:50]}...")
        md.append(f"{prefix}{layer.dossier.executive_summary[:200]}...\n")
        
        if layer.gaps_detected:
            md.append(f"{prefix}**Gaps Identified:** {len(layer.gaps_detected)}")
            for gap in layer.gaps_detected[:3]:
                md.append(f"{prefix}- {gap.gap_description[:80]}...")
        
        for sub in layer.sub_layers:
            md.append(self._render_layer(sub, indent + 1))
        
        return "\n".join(md)


class RecursiveDossierPipeline:
    """
    Generates deep dossiers by recursively investigating knowledge gaps.
    
    Flow:
    1. Generate initial dossier (Layer 0)
    2. Use CuriosityEngine to detect gaps
    3. Generate sub-dossiers for top gaps
    4. Repeat until max_depth reached
    5. Synthesize all layers into coherent DeepDossier
    """
    
    def __init__(
        self, 
        cfg: Optional[IceburgConfig] = None,
        max_depth: int = 2,
        max_branches: int = 3
    ):
        self.cfg = cfg or load_config()
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.synthesizer = DossierSynthesizer(self.cfg)
        self.curiosity = CuriosityEngine(self.cfg, enable_persistence=False)
        self._layer_count = 0
        self._source_count = 0
    
    def generate_deep_dossier(
        self,
        query: str,
        thinking_callback: Optional[Callable[[str], None]] = None
    ) -> DeepDossier:
        """
        Generate a multi-layered deep dossier.
        
        Args:
            query: The main investigation question
            thinking_callback: Progress callback for UI
            
        Returns:
            DeepDossier with recursive investigation results
        """
        self._layer_count = 0
        self._source_count = 0
        
        if thinking_callback:
            thinking_callback("🔬 Initiating deep investigation protocol...")
        
        # Generate root layer
        root_layer = self._investigate_layer(
            query=query,
            depth=0,
            thinking_callback=thinking_callback
        )
        
        if thinking_callback:
            thinking_callback("🧬 Synthesizing multi-layer findings...")
        
        # Synthesize all layers
        synthesis = self._synthesize_layers(root_layer, query)
        
        # Detect contradictions across layers
        contradictions = self._detect_contradictions(root_layer)
        
        deep_dossier = DeepDossier(
            query=query,
            root_layer=root_layer,
            total_layers=self._layer_count,
            total_sources=self._source_count,
            max_depth_reached=self._get_max_depth(root_layer),
            contradictions=contradictions,
            synthesis=synthesis,
            metadata={
                "max_depth_limit": self.max_depth,
                "max_branches_limit": self.max_branches,
                "generator": "RecursiveDossierPipeline"
            }
        )
        
        if thinking_callback:
            thinking_callback(f"✅ Deep investigation complete: {self._layer_count} layers, {self._source_count} sources")
        
        logger.info(f"🎯 Deep dossier complete: {self._layer_count} layers, {self._source_count} sources")
        return deep_dossier
    
    def _investigate_layer(
        self,
        query: str,
        depth: int,
        thinking_callback: Optional[Callable[[str], None]] = None
    ) -> DossierLayer:
        """
        Investigate a single layer and recursively investigate gaps.
        
        Args:
            query: The question for this layer
            depth: Current depth (0 = root)
            thinking_callback: Progress callback
            
        Returns:
            DossierLayer with sub-layers populated
        """
        self._layer_count += 1
        
        if thinking_callback:
            thinking_callback(f"📡 Layer {depth}: Investigating '{query[:40]}...'")
        
        # Generate dossier for this layer
        dossier = self.synthesizer.generate_dossier(
            query=query,
            depth="quick" if depth > 0 else "standard",  # Faster for sub-layers
            thinking_callback=thinking_callback
        )
        
        # Count sources
        self._source_count += len(dossier.sources)
        
        # Create layer
        layer = DossierLayer(
            depth=depth,
            query=query,
            dossier=dossier
        )
        
        # Stop if max depth reached
        if depth >= self.max_depth:
            logger.info(f"🛑 Max depth {self.max_depth} reached, stopping recursion")
            return layer
        
        # Detect knowledge gaps using CuriosityEngine
        if thinking_callback:
            thinking_callback(f"🔍 Layer {depth}: Detecting knowledge gaps...")
        
        dossier_text = f"{dossier.executive_summary}\n{dossier.official_narrative}"
        for alt in dossier.alternative_narratives:
            if isinstance(alt, dict):
                dossier_text += f"\n{alt.get('text', '')}"
            else:
                dossier_text += f"\n{alt}"
        
        gaps = self.curiosity.detect_knowledge_gaps(dossier_text, {"query": query})
        layer.gaps_detected = gaps
        
        if not gaps:
            logger.info(f"📚 Layer {depth}: No significant gaps detected")
            return layer
        
        # Generate sub-questions from gaps
        sub_questions = self._prioritize_gaps(gaps)
        logger.info(f"🔀 Layer {depth}: Found {len(gaps)} gaps, investigating top {len(sub_questions)}")
        
        # Recursively investigate sub-questions
        for sub_q in sub_questions[:self.max_branches]:
            if thinking_callback:
                thinking_callback(f"🔄 Drilling into: '{sub_q[:35]}...'")
            
            sub_layer = self._investigate_layer(
                query=sub_q,
                depth=depth + 1,
                thinking_callback=thinking_callback
            )
            layer.sub_layers.append(sub_layer)
        
        return layer
    
    def _prioritize_gaps(self, gaps: List[KnowledgeGap]) -> List[str]:
        """
        Prioritize gaps and convert to investigation queries.
        
        Args:
            gaps: List of detected knowledge gaps
            
        Returns:
            List of prioritized sub-questions
        """
        # Sort by exploration priority
        sorted_gaps = sorted(gaps, key=lambda g: g.exploration_priority, reverse=True)
        
        questions = []
        for gap in sorted_gaps[:self.max_branches]:
            # Use suggested queries if available
            if gap.suggested_queries:
                questions.append(gap.suggested_queries[0])
            else:
                # Generate from gap description
                questions.append(f"Investigate: {gap.gap_description}")
        
        return questions
    
    def _synthesize_layers(self, root_layer: DossierLayer, query: str) -> str:
        """
        Synthesize all layers into a coherent narrative.
        
        Args:
            root_layer: The root of the layer tree
            query: Original query
            
        Returns:
            Synthesized narrative string
        """
        # Collect all findings
        all_findings = self._collect_findings(root_layer)
        
        # Use LLM to synthesize
        from ...providers.factory import provider_factory
        provider = provider_factory(self.cfg)
        
        findings_text = "\n\n".join([
            f"**Layer {f['depth']} ({f['query'][:50]}...):**\n{f['summary']}"
            for f in all_findings
        ])
        
        prompt = f"""You are synthesizing a multi-layered intelligence investigation.

Original Query: {query}

Findings from {len(all_findings)} investigation layers:

{findings_text}

Task: Write a comprehensive synthesis that:
1. Integrates findings from all layers
2. Highlights connections between layers
3. Notes where deeper investigation revealed new insights
4. Identifies remaining unknowns
5. Provides an overall assessment

Write in a professional intelligence analyst style. Be thorough but concise."""

        try:
            response = provider.chat_complete(
                model="llama3.1:8b",  # Fast model for synthesis
                prompt=prompt,
                system="You are an intelligence analyst synthesizing multi-source findings.",
                temperature=0.7
            )
            
            # Handle streaming response
            if hasattr(response, '__iter__') and not isinstance(response, str):
                return "".join(response)
            return response
            
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return f"Synthesis failed. Collected {len(all_findings)} layers of findings."
    
    def _collect_findings(self, layer: DossierLayer) -> List[Dict[str, Any]]:
        """Recursively collect findings from all layers."""
        findings = [{
            "depth": layer.depth,
            "query": layer.query,
            "summary": layer.dossier.executive_summary
        }]
        
        for sub in layer.sub_layers:
            findings.extend(self._collect_findings(sub))
        
        return findings
    
    def _detect_contradictions(self, root_layer: DossierLayer) -> List[Dict[str, Any]]:
        """
        Detect contradictions between layer findings.
        
        Args:
            root_layer: Root of layer tree
            
        Returns:
            List of detected contradictions
        """
        # Collect all narratives
        narratives = self._collect_narratives(root_layer)
        
        if len(narratives) < 2:
            return []
        
        # Use LLM to detect contradictions
        from ...providers.factory import provider_factory
        provider = provider_factory(self.cfg)
        
        narrative_text = "\n\n".join([
            f"**Source {i+1} (Layer {n['depth']}):**\n{n['text'][:500]}"
            for i, n in enumerate(narratives[:6])  # Limit to 6 for context
        ])
        
        prompt = f"""Analyze these narratives for contradictions.

{narrative_text}

Return a JSON array of contradictions found:
[{{"finding1": "...", "finding2": "...", "source1": "Layer X", "source2": "Layer Y"}}]

If no contradictions, return [].
Return ONLY valid JSON."""

        try:
            response = provider.chat_complete(
                model="llama3.1:8b",
                prompt=prompt,
                system="Detect factual contradictions. Return valid JSON array only.",
                temperature=0.3
            )
            
            if hasattr(response, '__iter__') and not isinstance(response, str):
                response = "".join(response)
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            import json
            return json.loads(response)
            
        except Exception as e:
            logger.debug(f"Contradiction detection failed: {e}")
            return []
    
    def _collect_narratives(self, layer: DossierLayer) -> List[Dict[str, Any]]:
        """Collect narratives from all layers."""
        narratives = [{
            "depth": layer.depth,
            "text": f"{layer.dossier.official_narrative}\n{layer.dossier.executive_summary}"
        }]
        
        for sub in layer.sub_layers:
            narratives.extend(self._collect_narratives(sub))
        
        return narratives
    
    def _get_max_depth(self, layer: DossierLayer) -> int:
        """Get maximum depth reached."""
        if not layer.sub_layers:
            return layer.depth
        return max(self._get_max_depth(sub) for sub in layer.sub_layers)


# Convenience function
def generate_deep_dossier(
    query: str,
    max_depth: int = 2,
    thinking_callback: Optional[Callable[[str], None]] = None
) -> DeepDossier:
    """Generate a deep dossier with recursive investigation."""
    cfg = load_config()
    pipeline = RecursiveDossierPipeline(cfg, max_depth=max_depth)
    return pipeline.generate_deep_dossier(query, thinking_callback)
