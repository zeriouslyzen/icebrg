# src/iceburg/protocol/execution/agents/grounding_layer_agent.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

GROUNDING_LAYER_SYSTEM = (
    "ROLE: Grounding Layer Agent and Reality Anchoring Specialist\n"
    "MISSION: Anchor abstract concepts to concrete reality and validate claims against empirical evidence\n"
    "CAPABILITIES:\n"
    "- Reality anchoring and validation\n"
    "- Empirical evidence assessment\n"
    "- Concrete implementation analysis\n"
    "- Practical feasibility evaluation\n"
    "- Real-world constraint identification\n"
    "- Evidence-based validation\n"
    "- Practical application mapping\n\n"
    "GROUNDING FRAMEWORK:\n"
    "1. REALITY ANCHORING: Connect abstract concepts to concrete reality\n"
    "2. EMPIRICAL VALIDATION: Validate claims against empirical evidence\n"
    "3. IMPLEMENTATION ANALYSIS: Assess practical implementation feasibility\n"
    "4. CONSTRAINT IDENTIFICATION: Identify real-world constraints and limitations\n"
    "5. EVIDENCE EVALUATION: Evaluate quality and reliability of supporting evidence\n"
    "6. PRACTICAL MAPPING: Map theoretical concepts to practical applications\n\n"
    "OUTPUT FORMAT:\n"
    "GROUNDING ANALYSIS:\n"
    "- Reality Anchoring: [Concrete reality connections]\n"
    "- Empirical Validation: [Evidence-based validation]\n"
    "- Implementation Feasibility: [Practical implementation assessment]\n"
    "- Real-World Constraints: [Identified limitations]\n"
    "- Evidence Quality: [Supporting evidence evaluation]\n"
    "- Practical Applications: [Real-world applications]\n"
    "- Validation Confidence: [Confidence in grounding]\n\n"
    "GROUNDING STRENGTH: [Strong/Moderate/Weak]"
)

@register_agent("grounding_layer_agent")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    data_sources: Optional[List[str]] = None,
    correlation_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs grounding layer analysis and reality anchoring.
    """
    if verbose:
        print(f"[GROUNDING_LAYER] Anchoring to reality: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    # Build data sources info if available
    sources_info = ""
    if data_sources:
        sources_info = f"\nAVAILABLE DATA SOURCES:\n{', '.join(data_sources)}\n"
    
    # Build correlation types info if available
    correlation_info = ""
    if correlation_types:
        correlation_info = f"\nCORRELATION TYPES TO ANALYZE:\n{', '.join(correlation_types)}\n"
    
    prompt = (
        f"SCIENTIFIC QUERY: {query}\n\n"
        f"{context_info}"
        f"{sources_info}"
        f"{correlation_info}"
        "GROUNDING LAYER MISSION: Anchor abstract concepts to concrete reality and validate against empirical evidence.\n\n"
        "GROUNDING TASKS:\n"
        "1. Connect abstract concepts to concrete reality\n"
        "2. Validate claims against empirical evidence\n"
        "3. Assess practical implementation feasibility\n"
        "4. Identify real-world constraints and limitations\n"
        "5. Evaluate quality and reliability of supporting evidence\n"
        "6. Map theoretical concepts to practical applications\n"
        "7. Assess confidence in grounding strength\n\n"
        "Provide comprehensive grounding analysis with strong empirical validation and practical feasibility assessment."
    )
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=GROUNDING_LAYER_SYSTEM,
        temperature=0.1,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="GroundingLayer",
    )
    
    if verbose:
        print(f"[GROUNDING_LAYER] Grounding completed for query: {query[:50]}...")
    
    return result
