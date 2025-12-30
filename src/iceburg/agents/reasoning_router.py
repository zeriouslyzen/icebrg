"""
Reasoning Router for ICEBURG

Smart routing to specialized reasoning modules based on query type.
Routes to:
- Abstract Transformer: Formal transformations (opposites, sequences)
- Analogical Mapper: A:B::C:? relationships
- Compositional Engine: Component combination
- Hierarchical Reasoner: Tree/graph structures
- Spatial Reasoner: Containment logic
- Visual Reasoner: ARC grids and visual patterns

Part of the ARC-AGI Enhancement Project - Phase 4 Integration
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import IceburgConfig

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning modules available."""
    ABSTRACT_TRANSFORM = "abstract_transform"
    ANALOGICAL = "analogical"
    COMPOSITIONAL = "compositional"
    HIERARCHICAL = "hierarchical"
    SPATIAL = "spatial"
    VISUAL = "visual"
    GENERAL = "general"


@dataclass
class RoutingDecision:
    """Result of routing analysis."""
    primary: ReasoningType
    secondary: Optional[ReasoningType]
    confidence: float
    indicators: List[str]
    query_features: Dict[str, Any]


# Keyword patterns for each reasoning type
ROUTING_PATTERNS = {
    ReasoningType.ABSTRACT_TRANSFORM: {
        "keywords": [
            "opposite", "antonym", "transform", "pattern", "sequence",
            "next in", "what comes", "rule", "→", "->", "implies"
        ],
        "patterns": [
            r'\w+\s*[→\->]+\s*\w+',  # X → Y
            r'if\s+\w+\s+then',      # if X then Y
            r'what\s+is\s+the\s+pattern',
        ],
    },
    ReasoningType.ANALOGICAL: {
        "keywords": [
            "analogy", "is to", "as", "::", "like",
            "relationship", "relates to", "similar"
        ],
        "patterns": [
            r'\w+\s*:\s*\w+\s*::\s*\w+',  # A : B :: C
            r'\w+\s+is\s+to\s+\w+\s+as',  # A is to B as
        ],
    },
    ReasoningType.COMPOSITIONAL: {
        "keywords": [
            "compose", "combine", "add", "plus", "+",
            "together", "sum", "blend", "mix"
        ],
        "patterns": [
            r'\w+\s*\+\s*\w+',           # A + B
            r'(\d+|one|two|three)\s*\+',  # number + number
        ],
    },
    ReasoningType.HIERARCHICAL: {
        "keywords": [
            "contains", "parent", "child", "level", "depth",
            "tree", "hierarchy", "ancestor", "descendant",
            "path from", "traversal"
        ],
        "patterns": [
            r'level\s*\d+',
            r'depth\s+of',
            r'path\s+from\s+\w+\s+to',
        ],
    },
    ReasoningType.SPATIAL: {
        "keywords": [
            "inside", "outside", "contains", "within",
            "above", "below", "left", "right", "between",
            "spatial", "position", "location"
        ],
        "patterns": [
            r'\w+\s+is\s+inside\s+\w+',
            r'\w+\s+contains\s+\d+\s+elements?',
        ],
    },
    ReasoningType.VISUAL: {
        "keywords": [
            "grid", "visual", "image", "pattern", "color",
            "pixel", "cell", "rotate", "reflect", "flip",
            "arc", "transformation"
        ],
        "patterns": [
            r'\[\s*\[',                  # [[...]] grid format
            r'grid\s*:',
            r'\d+\s+\d+\s+\d+',          # Number sequences (grid rows)
        ],
    },
}


def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze a query to extract features for routing.
    
    Returns dictionary with:
    - has_grid: Whether query contains grid data
    - has_math: Whether query has mathematical elements
    - has_relationship: Whether query asks about relationships
    - entity_count: Number of distinct entities mentioned
    """
    query_lower = query.lower()
    
    return {
        "has_grid": bool(re.search(r'\[\s*\[', query) or "grid" in query_lower),
        "has_math": bool(re.search(r'[\+\-\×\÷]|\d+\s*[\+\-\*/]', query)),
        "has_relationship": any(w in query_lower for w in ["is to", "::", "contains", "inside"]),
        "has_sequence": bool(re.search(r'→|->|then|next', query_lower)),
        "has_hierarchy": any(w in query_lower for w in ["level", "depth", "parent", "child", "tree"]),
        "word_count": len(query.split()),
        "has_question": "?" in query,
    }


def route_query(query: str, verbose: bool = False) -> RoutingDecision:
    """
    Determine the best reasoning module for a query.
    
    Args:
        query: The user's reasoning query
        verbose: Print routing analysis
        
    Returns:
        RoutingDecision with primary module and confidence
    """
    query_lower = query.lower()
    scores: Dict[ReasoningType, float] = {rt: 0.0 for rt in ReasoningType}
    indicators: Dict[ReasoningType, List[str]] = {rt: [] for rt in ReasoningType}
    
    # Score each reasoning type
    for rtype, patterns in ROUTING_PATTERNS.items():
        # Check keywords
        for keyword in patterns["keywords"]:
            if keyword.lower() in query_lower:
                scores[rtype] += 1.0
                indicators[rtype].append(f"keyword: {keyword}")
        
        # Check regex patterns
        for pattern in patterns["patterns"]:
            if re.search(pattern, query, re.IGNORECASE):
                scores[rtype] += 2.0  # Patterns weighted more
                indicators[rtype].append(f"pattern: {pattern[:30]}...")
    
    # Analyze query features
    features = analyze_query(query)
    
    # Boost based on features
    if features["has_grid"]:
        scores[ReasoningType.VISUAL] += 3.0
        indicators[ReasoningType.VISUAL].append("has grid data")
    
    if features["has_math"]:
        scores[ReasoningType.COMPOSITIONAL] += 2.0
        indicators[ReasoningType.COMPOSITIONAL].append("has math operators")
    
    if features["has_hierarchy"]:
        scores[ReasoningType.HIERARCHICAL] += 2.0
        indicators[ReasoningType.HIERARCHICAL].append("has hierarchy keywords")
    
    if features["has_sequence"]:
        scores[ReasoningType.ABSTRACT_TRANSFORM] += 1.5
        indicators[ReasoningType.ABSTRACT_TRANSFORM].append("has sequence markers")
    
    # Find best matches
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_types[0][1] == 0:
        # No matches - use general reasoning
        primary = ReasoningType.GENERAL
        secondary = None
        confidence = 0.3
    else:
        primary = sorted_types[0][0]
        max_score = sorted_types[0][1]
        
        # Normalize confidence
        total_score = sum(s for _, s in sorted_types)
        confidence = min(0.95, max_score / max(total_score, 1) + 0.3)
        
        # Check for secondary if close
        if len(sorted_types) > 1 and sorted_types[1][1] >= max_score * 0.5:
            secondary = sorted_types[1][0]
        else:
            secondary = None
    
    if verbose:
        print(f"[ROUTER] Query: {query[:50]}...")
        print(f"[ROUTER] Primary: {primary.value} (confidence: {confidence:.2f})")
        if secondary:
            print(f"[ROUTER] Secondary: {secondary.value}")
        print(f"[ROUTER] Indicators: {indicators[primary]}")
    
    return RoutingDecision(
        primary=primary,
        secondary=secondary,
        confidence=confidence,
        indicators=indicators[primary],
        query_features=features
    )


def run(
    cfg: IceburgConfig,
    query: str,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Route and execute query through appropriate reasoning module.
    
    Args:
        cfg: ICEBURG configuration
        query: The reasoning query
        verbose: Print debug info
        **kwargs: Additional arguments passed to the reasoning module
        
    Returns:
        Result from the selected reasoning module
    """
    # Route the query
    decision = route_query(query, verbose=verbose)
    
    if verbose:
        print(f"[ROUTER] Routing to: {decision.primary.value}")
    
    # Import and execute appropriate module
    try:
        if decision.primary == ReasoningType.ABSTRACT_TRANSFORM:
            from . import abstract_transformer
            return abstract_transformer.run(cfg, query, verbose=verbose, **kwargs)
        
        elif decision.primary == ReasoningType.ANALOGICAL:
            from . import analogical_mapper
            return analogical_mapper.run(cfg, query, verbose=verbose, **kwargs)
        
        elif decision.primary == ReasoningType.COMPOSITIONAL:
            from . import compositional_engine
            return compositional_engine.run(cfg, query, verbose=verbose, **kwargs)
        
        elif decision.primary == ReasoningType.HIERARCHICAL:
            from . import hierarchical_reasoner
            return hierarchical_reasoner.run(cfg, query, verbose=verbose, **kwargs)
        
        elif decision.primary == ReasoningType.SPATIAL:
            from . import spatial_reasoner
            return spatial_reasoner.run(cfg, query, verbose=verbose, **kwargs)
        
        elif decision.primary == ReasoningType.VISUAL:
            from . import visual_reasoner
            return visual_reasoner.run(cfg, query, verbose=verbose, **kwargs)
        
        else:
            # General reasoning - use synthesist
            from . import synthesist
            return synthesist.run(cfg, query, verbose=verbose)
            
    except Exception as e:
        logger.error(f"[ROUTER] Module execution failed: {e}")
        
        # Try secondary if available
        if decision.secondary:
            if verbose:
                print(f"[ROUTER] Trying secondary: {decision.secondary.value}")
            # Recursively route with the secondary type
            # (In practice, you'd implement this more robustly)
        
        raise


def get_module_status() -> Dict[str, bool]:
    """Get availability status of all reasoning modules."""
    status = {}
    
    modules = [
        ("abstract_transformer", "abstract_transformer"),
        ("analogical_mapper", "analogical_mapper"),
        ("compositional_engine", "compositional_engine"),
        ("hierarchical_reasoner", "hierarchical_reasoner"),
        ("spatial_reasoner", "spatial_reasoner"),
        ("visual_reasoner", "visual_reasoner"),
    ]
    
    for name, module_name in modules:
        try:
            exec(f"from . import {module_name}")
            status[name] = True
        except ImportError:
            status[name] = False
    
    # Check V-JEPA availability
    try:
        from ..vjepa import VJEPAEncoder
        encoder = VJEPAEncoder()
        status["vjepa"] = encoder.is_available()
    except ImportError:
        status["vjepa"] = False
    
    return status
