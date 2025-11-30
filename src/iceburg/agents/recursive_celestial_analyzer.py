"""
Recursive Celestial-Physiological Analyzer Agent

This agent performs deep recursive analysis on the physiology-celestial-chemistry
data to discover new patterns, relationships, and emergent understanding.

The agent treats celestial bodies as "meridians" - structural elements that
connect different domains of reality through STEM-based mechanisms.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..config import IceburgConfig
from ..vectorstore import VectorStore
from ..llm import chat_complete
from ..agents.celestial_biological_framework import (
    get_celestial_biological_framework,
    get_current_celestial_conditions
)

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except:
    pass

@dataclass
class RecursiveAnalysisResult:
    """Result from recursive analysis"""
    pattern_id: str
    pattern_description: str
    depth_level: int
    parent_patterns: List[str] = field(default_factory=list)
    child_patterns: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    implications: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class RecursiveCelestialAnalyzer:
    """
    Agent that performs recursive analysis on celestial-physiological data.
    
    Key insight: Stars/planets act as "meridians" - structural elements
    connecting different domains through measurable STEM mechanisms.
    """
    
    def __init__(self, cfg: IceburgConfig, vs: VectorStore):
        self.cfg = cfg
        self.vs = vs
        self.analysis_depth = 0
        self.discovered_patterns: Dict[str, RecursiveAnalysisResult] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Load base data
        self.base_data = self._load_base_data()
    
    def _load_base_data(self) -> Dict[str, Any]:
        """Load the physiology-celestial-chemistry data"""
        data_file = Path('data/physiology_celestial_chemistry_data.json')
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    async def analyze_recursively(
        self,
        query: str,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> List[RecursiveAnalysisResult]:
        """
        Perform recursive analysis on the query.
        
        Each level of recursion:
        1. Searches existing data
        2. Identifies patterns
        3. Generates new questions
        4. Recurses into deeper understanding
        """
        if current_depth >= max_depth:
            return []
        
        self.analysis_depth = current_depth
        results = []
        
        # Step 1: Search existing knowledge
        search_results = self.vs.semantic_search(
            query,
            k=10,
            where={"source": "physiology_celestial_chemistry"}
        )
        
        # Step 2: Analyze patterns
        pattern = await self._identify_pattern(query, search_results, current_depth)
        if pattern:
            results.append(pattern)
            self.discovered_patterns[pattern.pattern_id] = pattern
        
        # Step 3: Generate deeper questions
        deeper_questions = await self._generate_deeper_questions(pattern, current_depth)
        
        # Step 4: Recurse into each question
        for question in deeper_questions:
            child_results = await self.analyze_recursively(
                question,
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            results.extend(child_results)
            
            # Link parent-child relationships
            if pattern and child_results:
                pattern.child_patterns.extend([r.pattern_id for r in child_results])
                for child in child_results:
                    child.parent_patterns.append(pattern.pattern_id)
        
        return results
    
    async def _identify_pattern(
        self,
        query: str,
        search_results: List,
        depth: int
    ) -> Optional[RecursiveAnalysisResult]:
        """Identify patterns from search results"""
        
        # Extract relevant information
        context = "\n".join([hit.document for hit in search_results[:5]])
        
        system_prompt = f"""You are a recursive celestial-physiological analyzer.
You understand that celestial bodies act as "meridians" - structural elements
connecting different domains of reality through STEM-based mechanisms.

Current analysis depth: {depth}

Analyze the following data to identify patterns, relationships, and emergent properties.
Focus on:
1. Cross-domain connections (celestial → physiological → molecular → behavioral)
2. Recursive structures (how patterns repeat at different scales)
3. Meridian-like properties (how celestial bodies connect disparate systems)
4. STEM-based mechanisms (voltage gates, neurotransmitters, hormones, molecular chemistry)

Data:
{context}

Query: {query}
"""
        
        user_prompt = f"""Identify the key pattern or relationship in this data related to: {query}

Provide:
1. Pattern description
2. Evidence from the data
3. Confidence level (0.0-1.0)
4. Implications for understanding
5. What deeper questions this raises

Format as JSON with keys: pattern_description, evidence, confidence, implications, deeper_questions"""
        
        try:
            response = await chat_complete(
                self.cfg,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.cfg.model,
                temperature=0.7
            )
            
            # Parse response (may contain JSON)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                pattern_data = json.loads(json_match.group())
            else:
                # Fallback: extract from text
                pattern_data = {
                    "pattern_description": response[:200],
                    "evidence": [hit.document[:100] for hit in search_results[:3]],
                    "confidence": 0.6,
                    "implications": [response],
                    "deeper_questions": []
                }
            
            pattern_id = f"pattern_{depth}_{len(self.discovered_patterns)}"
            
            return RecursiveAnalysisResult(
                pattern_id=pattern_id,
                pattern_description=pattern_data.get("pattern_description", response),
                depth_level=depth,
                evidence=pattern_data.get("evidence", []),
                confidence=pattern_data.get("confidence", 0.5),
                implications=pattern_data.get("implications", [])
            )
            
        except Exception as e:
            if logger:
                logger.error(f"Error identifying pattern: {e}")
            return None
    
    async def _generate_deeper_questions(
        self,
        pattern: Optional[RecursiveAnalysisResult],
        depth: int
    ) -> List[str]:
        """Generate questions for deeper recursive analysis"""
        
        if not pattern:
            return []
        
        system_prompt = f"""You are generating deeper questions for recursive analysis.
Current depth: {depth}
Pattern: {pattern.pattern_description if pattern else 'None'}

Generate 2-3 questions that would lead to deeper understanding of:
1. The mechanisms underlying this pattern
2. Connections to other domains
3. Recursive structures (how this pattern appears at different scales)
4. Meridian properties (how celestial bodies connect systems)

Focus on STEM-based, measurable questions."""
        
        user_prompt = f"""Based on this pattern:
{pattern.pattern_description if pattern else 'No pattern yet'}

Generate 2-3 deeper questions for recursive analysis. Return as JSON array of strings."""
        
        try:
            response = await chat_complete(
                self.cfg,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.cfg.model,
                temperature=0.8
            )
            
            # Parse JSON array
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                return questions if isinstance(questions, list) else []
            else:
                # Fallback: extract questions from text
                questions = []
                for line in response.split('\n'):
                    if '?' in line:
                        questions.append(line.strip())
                return questions[:3]
                
        except Exception as e:
            if logger:
                logger.error(f"Error generating deeper questions: {e}")
            return []
    
    async def analyze_meridian_properties(self) -> Dict[str, Any]:
        """
        Analyze how celestial bodies act as "meridians" - structural elements
        connecting different domains through STEM mechanisms.
        """
        analysis = {
            "meridian_analysis": {},
            "cross_domain_connections": [],
            "recursive_structures": [],
            "stem_mechanisms": []
        }
        
        if not self.base_data:
            return analysis
        
        celestial_data = self.base_data.get("physiology_celestial_chemistry_mapping", {}).get("celestial_bodies", {})
        
        for body, info in celestial_data.items():
            meridian_properties = {
                "celestial_body": body,
                "organ_system": info.get("organ_system"),
                "voltage_gates": info.get("voltage_gates", {}).get("primary", []),
                "neurotransmitters": list(info.get("neurotransmitters", {}).keys()),
                "hormones": list(info.get("hormones", {}).keys()),
                "electromagnetic_frequency": info.get("electromagnetic", {}).get("frequency_range_hz", []),
                "connection_strength": len(info.get("voltage_gates", {}).get("primary", [])) +
                                       len(info.get("neurotransmitters", {})) +
                                       len(info.get("hormones", {}))
            }
            
            analysis["meridian_analysis"][body] = meridian_properties
            
            # Identify cross-domain connections
            connections = []
            if info.get("voltage_gates"):
                connections.append(f"{body} → Voltage Gates → {info.get('organ_system')}")
            if info.get("neurotransmitters"):
                for nt in info.get("neurotransmitters", {}).keys():
                    connections.append(f"{body} → {nt} → Neural Systems")
            if info.get("hormones"):
                for h in info.get("hormones", {}).keys():
                    connections.append(f"{body} → {h} → Endocrine Systems")
            
            analysis["cross_domain_connections"].extend(connections)
        
        return analysis
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all recursive analyses"""
        return {
            "total_patterns": len(self.discovered_patterns),
            "max_depth_reached": self.analysis_depth,
            "patterns_by_depth": self._group_patterns_by_depth(),
            "pattern_network": self._build_pattern_network()
        }
    
    def _group_patterns_by_depth(self) -> Dict[int, List[str]]:
        """Group patterns by analysis depth"""
        grouped = {}
        for pattern_id, pattern in self.discovered_patterns.items():
            depth = pattern.depth_level
            if depth not in grouped:
                grouped[depth] = []
            grouped[depth].append(pattern_id)
        return grouped
    
    def _build_pattern_network(self) -> Dict[str, Any]:
        """Build network graph of pattern relationships"""
        network = {
            "nodes": [],
            "edges": []
        }
        
        for pattern_id, pattern in self.discovered_patterns.items():
            network["nodes"].append({
                "id": pattern_id,
                "label": pattern.pattern_description[:50],
                "depth": pattern.depth_level,
                "confidence": pattern.confidence
            })
            
            for child_id in pattern.child_patterns:
                network["edges"].append({
                    "from": pattern_id,
                    "to": child_id,
                    "type": "parent_child"
                })
        
        return network


async def run_recursive_analysis(
    cfg: IceburgConfig,
    vs: VectorStore,
    query: str,
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Run recursive analysis on a query.
    
    This enables agents to:
    1. Access the physiology-celestial-chemistry data
    2. Perform recursive analysis
    3. Discover new patterns and relationships
    4. Build understanding recursively
    """
    analyzer = RecursiveCelestialAnalyzer(cfg, vs)
    
    # Perform recursive analysis
    results = await analyzer.analyze_recursively(query, max_depth=max_depth)
    
    # Analyze meridian properties
    meridian_analysis = await analyzer.analyze_meridian_properties()
    
    # Get summary
    summary = analyzer.get_analysis_summary()
    
    return {
        "query": query,
        "results": [{
            "pattern_id": r.pattern_id,
            "description": r.pattern_description,
            "depth": r.depth_level,
            "confidence": r.confidence,
            "implications": r.implications
        } for r in results],
        "meridian_analysis": meridian_analysis,
        "summary": summary,
        "total_patterns_discovered": len(results)
    }

