"""
Spatial Reasoner for ICEBURG

Handles spatial and containment logic:
- Containment relationships (inside/outside)
- Set-theoretic operations
- Spatial constraint propagation

Part of the ARC-AGI Enhancement Project - Phase 3
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..config import IceburgConfig
from ..llm import chat_complete


class SpatialRelation(Enum):
    """Types of spatial relationships."""
    INSIDE = "inside"
    OUTSIDE = "outside"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    ADJACENT = "adjacent"
    UNKNOWN = "unknown"


@dataclass
class SpatialObject:
    """Represents a spatial object with containment relationships."""
    name: str
    contains: Set[str] = field(default_factory=set)
    contained_by: Optional[str] = None
    element_count: Optional[int] = None


@dataclass
class SpatialResult:
    """Result of spatial reasoning."""
    query_type: str
    answer: str
    objects: Dict[str, SpatialObject]
    confidence: float
    explanation: str


SPATIAL_REASONER_SYSTEM = (
    "ROLE: Spatial Reasoning Specialist - Expert in containment and topological logic.\n"
    "TASK: Analyze spatial relationships and answer questions about containment, set operations, and positions.\n"
    "\n"
    "CAPABILITIES:\n"
    "- Parse spatial containment relationships (A inside B)\n"
    "- Apply transitivity (A in B, B in C → A in C)\n"
    "- Calculate set operations (total elements, containment chains)\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "1. SPATIAL STRUCTURE: [describe the spatial arrangement]\n"
    "2. QUERY TYPE: [containment/elements/position/relation]\n"
    "3. ANSWER: [computed result]\n"
    "4. REASONING: [step-by-step spatial logic]\n"
    "5. CONFIDENCE: [0.0-1.0]\n"
)


class SpatialModel:
    """A model of spatial objects and their relationships."""
    
    def __init__(self):
        self.objects: Dict[str, SpatialObject] = {}
    
    def add_object(self, name: str, element_count: Optional[int] = None) -> SpatialObject:
        """Add an object to the model."""
        if name not in self.objects:
            self.objects[name] = SpatialObject(name=name, element_count=element_count)
        elif element_count is not None:
            self.objects[name].element_count = element_count
        return self.objects[name]
    
    def set_containment(self, container: str, contained: str) -> None:
        """Set that container contains contained."""
        self.add_object(container)
        self.add_object(contained)
        
        self.objects[container].contains.add(contained)
        self.objects[contained].contained_by = container
    
    def is_inside(self, inner: str, outer: str) -> bool:
        """Check if inner is inside outer (directly or transitively)."""
        if inner not in self.objects:
            return False
        
        current = inner
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            container = self.objects[current].contained_by
            
            if container == outer:
                return True
            current = container
        
        return False
    
    def get_containing_chain(self, name: str) -> List[str]:
        """Get the chain of containers from object to outermost."""
        if name not in self.objects:
            return []
        
        chain = [name]
        current = name
        
        while self.objects[current].contained_by:
            container = self.objects[current].contained_by
            chain.append(container)
            current = container
        
        return chain
    
    def get_all_contained(self, name: str) -> Set[str]:
        """Get all objects contained within name (recursively)."""
        if name not in self.objects:
            return set()
        
        result = set()
        to_visit = list(self.objects[name].contains)
        
        while to_visit:
            current = to_visit.pop()
            if current not in result:
                result.add(current)
                if current in self.objects:
                    to_visit.extend(self.objects[current].contains)
        
        return result
    
    def calculate_total_elements(self) -> int:
        """
        Calculate total unique elements considering containment.
        
        When A is inside B, A's elements are part of B's count.
        The total is the maximum of any non-contained object.
        """
        # Find outermost containers (not contained by anything)
        outermost = [
            name for name, obj in self.objects.items()
            if obj.contained_by is None
        ]
        
        if not outermost:
            return 0
        
        # Get element counts for outermost, considering hierarchy
        total = 0
        for name in outermost:
            obj = self.objects[name]
            if obj.element_count is not None:
                total = max(total, obj.element_count)
        
        return total
    
    def calculate_nested_elements(self) -> Optional[int]:
        """
        For nested structures A in B in C, determine max element count.
        
        If A has 2 elements and B has 5 elements, and A is inside B,
        then B has 5 elements total (including A's 2).
        """
        max_elements = 0
        
        for obj in self.objects.values():
            if obj.element_count is not None:
                max_elements = max(max_elements, obj.element_count)
        
        return max_elements if max_elements > 0 else None


def build_spatial_model(description: str, verbose: bool = False) -> SpatialModel:
    """
    Build a spatial model from a natural language description.
    
    Handles formats like:
    - "Shape A is inside Shape B"
    - "B contains C"
    - "Shape A contains 3 elements"
    """
    model = SpatialModel()
    
    # Pattern: X is inside Y (handles multi-word names like "Shape A")
    inside_pattern = r'([A-Za-z]+(?:\s+[A-Za-z0-9]+)?)\s+is\s+inside\s+([A-Za-z]+(?:\s+[A-Za-z0-9]+)?)'
    for match in re.finditer(inside_pattern, description, re.IGNORECASE):
        inner = match.group(1).strip()
        outer = match.group(2).strip()
        model.set_containment(outer, inner)
        if verbose:
            print(f"[SPATIAL_REASONER] {inner} is inside {outer}")
    
    # Pattern: X contains Y (handles multi-word names)
    contains_pattern = r'([A-Za-z]+(?:\s+[A-Za-z0-9]+)?)\s+contains?\s+(?!\d+\s+element)([A-Za-z]+(?:\s+[A-Za-z0-9]+)?)'
    for match in re.finditer(contains_pattern, description, re.IGNORECASE):
        outer = match.group(1).strip()
        inner = match.group(2).strip()
        model.set_containment(outer, inner)
        if verbose:
            print(f"[SPATIAL_REASONER] {outer} contains {inner}")
    
    # Pattern: X contains N elements
    element_pattern = r'(\w+)\s+contains?\s+(\d+)\s+element'
    for match in re.finditer(element_pattern, description, re.IGNORECASE):
        name = match.group(1).strip()
        count = int(match.group(2))
        model.add_object(name, element_count=count)
        if verbose:
            print(f"[SPATIAL_REASONER] {name} has {count} elements")
    
    return model


def answer_spatial_query(
    model: SpatialModel,
    query: str,
    verbose: bool = False
) -> SpatialResult:
    """
    Answer a spatial reasoning query.
    
    Query types:
    - "is A inside B"
    - "how many elements total"
    - "what contains A"
    """
    query_lower = query.lower()
    
    # Total elements query
    if "total" in query_lower and "element" in query_lower:
        total = model.calculate_nested_elements()
        
        if total is not None:
            # Explain the reasoning
            element_info = []
            for name, obj in model.objects.items():
                if obj.element_count is not None:
                    element_info.append(f"{name} contains {obj.element_count} elements")
            
            explanation = (
                "In nested containment, the outermost container's count includes all inner elements. "
                f"Structure: {'; '.join(element_info)}. "
                f"Maximum is {total} (the container with most elements includes its contents)."
            )
            
            return SpatialResult(
                query_type="total_elements",
                answer=str(total),
                objects=model.objects,
                confidence=0.85,
                explanation=explanation
            )
    
    # Containment check query
    inside_match = re.search(r'is\s+(\w+)\s+inside\s+(\w+)', query_lower)
    if inside_match:
        inner = inside_match.group(1)
        outer = inside_match.group(2)
        
        # Normalize names (check various capitalizations and partial matches)
        inner_normalized = None
        outer_normalized = None
        
        for name in model.objects.keys():
            name_lower = name.lower()
            # Exact match
            if name_lower == inner.lower():
                inner_normalized = name
            # Partial match (e.g., "A" matches "Shape A")
            elif inner.lower() in name_lower.split() or name_lower.endswith(inner.lower()):
                if inner_normalized is None:
                    inner_normalized = name
            
            # Same for outer
            if name_lower == outer.lower():
                outer_normalized = name
            elif outer.lower() in name_lower.split() or name_lower.endswith(outer.lower()):
                if outer_normalized is None:
                    outer_normalized = name
        
        if inner_normalized and outer_normalized:
            is_inside = model.is_inside(inner_normalized, outer_normalized)
            chain = model.get_containing_chain(inner_normalized)
            
            # Show transitive chain reasoning
            if is_inside:
                explanation = f"{inner_normalized} is inside {outer_normalized} (transitively). Chain: {' → '.join(chain)}"
            else:
                explanation = f"{inner_normalized} is NOT inside {outer_normalized}. Chain: {' → '.join(chain)}"
            
            return SpatialResult(
                query_type="containment_check",
                answer="Yes" if is_inside else "No",
                objects=model.objects,
                confidence=0.9,
                explanation=explanation
            )
    
    # What is outside query
    if "outside" in query_lower:
        for name in model.objects.keys():
            if name.lower() in query_lower:
                obj = model.objects[name]
                chain = model.get_containing_chain(name)
                outermost = chain[-1] if chain else name
                
                return SpatialResult(
                    query_type="outside",
                    answer=outermost,
                    objects=model.objects,
                    confidence=0.85,
                    explanation=f"The outermost container in the chain is {outermost}"
                )
    
    return SpatialResult(
        query_type="unknown",
        answer="",
        objects=model.objects,
        confidence=0.0,
        explanation="Could not parse spatial query"
    )


def run(
    cfg: IceburgConfig,
    query: str,
    spatial_description: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Main entry point for Spatial Reasoner.
    
    Args:
        cfg: ICEBURG configuration
        query: The spatial reasoning question
        spatial_description: Optional description of spatial relationships
        verbose: Print debug info
        
    Returns:
        Analysis result with spatial answer
    """
    result_parts = []
    
    # Build spatial model
    full_text = (spatial_description or "") + " " + query
    model = build_spatial_model(full_text, verbose=verbose)
    
    if verbose:
        print(f"[SPATIAL_REASONER] Built model with {len(model.objects)} objects")
    
    # Try to answer the query
    if model.objects:
        result = answer_spatial_query(model, query, verbose=verbose)
        
        result_parts.append(f"**Query Type:** {result.query_type}")
        result_parts.append(f"**Confidence:** {result.confidence:.0%}")
        
        # Show spatial structure
        structure_parts = []
        for name, obj in result.objects.items():
            if obj.contained_by:
                structure_parts.append(f"{name} inside {obj.contained_by}")
            if obj.element_count is not None:
                structure_parts.append(f"{name} has {obj.element_count} elements")
        
        if structure_parts:
            result_parts.append(f"**Spatial Structure:** {'; '.join(structure_parts)}")
        
        if result.answer:
            result_parts.append(f"\n**Answer:** {result.answer}")
            result_parts.append(f"**Reasoning:** {result.explanation}")
            
            if result.confidence >= 0.8:
                return "\n".join(result_parts)
    
    # Fall back to LLM
    if verbose:
        print("[SPATIAL_REASONER] Using LLM for complex spatial reasoning")
    
    prompt = f"SPATIAL REASONING TASK:\n{query}\n\n"
    if spatial_description:
        prompt += f"SPATIAL RELATIONSHIPS:\n{spatial_description}\n\n"
    
    prompt += "Analyze the spatial containment relationships and answer the question."
    
    try:
        llm_result = chat_complete(
            cfg.synthesist_model,
            prompt,
            system=SPATIAL_REASONER_SYSTEM,
            temperature=0.3,
            options={"num_ctx": 2048, "num_predict": 400},
            context_tag="SpatialReasoner"
        )
        
        if result_parts:
            return "\n".join(result_parts) + "\n\n**LLM Analysis:**\n" + llm_result
        return llm_result
        
    except Exception as e:
        if verbose:
            print(f"[SPATIAL_REASONER] LLM Error: {e}")
        if result_parts:
            return "\n".join(result_parts)
        raise
