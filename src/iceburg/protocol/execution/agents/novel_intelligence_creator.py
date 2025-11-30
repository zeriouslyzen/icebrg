# src/iceburg/protocol/execution/agents/novel_intelligence_creator.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ....llm import chat_complete
from ....model_select import resolve_models
from .registry import register_agent

NOVEL_INTELLIGENCE_SYSTEM = (
    "ROLE: Novel Intelligence Creator and Intelligence Innovation Specialist\n"
    "MISSION: Invent new types of intelligence and create innovative intelligence architectures\n"
    "CAPABILITIES:\n"
    "- Novel intelligence type invention\n"
    "- Intelligence architecture innovation\n"
    "- Cross-domain intelligence synthesis\n"
    "- Emergent intelligence pattern creation\n"
    "- Intelligence evolution pathway design\n"
    "- Meta-intelligence development\n"
    "- Intelligence paradigm shifting\n\n"
    "CREATION FRAMEWORK:\n"
    "1. INTELLIGENCE ANALYSIS: Analyze existing intelligence types and patterns\n"
    "2. NOVEL TYPE INVENTION: Create new, previously unexplored intelligence types\n"
    "3. ARCHITECTURE INNOVATION: Design innovative intelligence architectures\n"
    "4. SYNTHESIS CREATION: Combine different intelligence types into novel syntheses\n"
    "5. EVOLUTION PATHWAY DESIGN: Map pathways for intelligence evolution\n"
    "6. META-INTELLIGENCE DEVELOPMENT: Develop intelligence about intelligence\n"
    "7. PARADIGM SHIFTING: Propose fundamental shifts in intelligence paradigms\n\n"
    "OUTPUT FORMAT:\n"
    "NOVEL INTELLIGENCE CREATION:\n"
    "- Intelligence Analysis: [Analysis of existing intelligence types]\n"
    "- Novel Types Invented: [New intelligence types created]\n"
    "- Architecture Innovations: [Innovative intelligence architectures]\n"
    "- Intelligence Syntheses: [Novel combinations of intelligence types]\n"
    "- Evolution Pathways: [Intelligence development roadmaps]\n"
    "- Meta-Intelligence Insights: [Intelligence about intelligence]\n"
    "- Paradigm Shifts: [Fundamental intelligence paradigm changes]\n\n"
    "INNOVATION CONFIDENCE: [High/Medium/Low]"
)

@register_agent("novel_intelligence_creator")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Creates novel intelligence types and innovative intelligence architectures.
    """
    if verbose:
        print(f"[NOVEL_INTELLIGENCE] Creating novel intelligence for: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"INTELLIGENCE CREATION QUERY: {query}\n\n"
        f"{context_info}"
        "NOVEL INTELLIGENCE CREATION MISSION: Invent new types of intelligence and create innovative architectures.\n\n"
        "CREATION TASKS:\n"
        "1. Analyze existing intelligence types and patterns\n"
        "2. Create new, previously unexplored intelligence types\n"
        "3. Design innovative intelligence architectures\n"
        "4. Combine different intelligence types into novel syntheses\n"
        "5. Map pathways for intelligence evolution\n"
        "6. Develop intelligence about intelligence (meta-intelligence)\n"
        "7. Propose fundamental shifts in intelligence paradigms\n\n"
        "Provide comprehensive novel intelligence creation with specific new intelligence types and innovative architectures."
    )
    
    # Get model from resolved models
    surveyor, _, _, oracle, _ = resolve_models("llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "nomic-embed-text")
    model = oracle or surveyor or "llama3.1:8b"
    
    result = chat_complete(
        model,
        prompt,
        system=NOVEL_INTELLIGENCE_SYSTEM,
        temperature=0.4,
        options={"num_ctx": 4096, "num_predict": 1200},
        context_tag="NovelIntelligence",
    )
    
    if verbose:
        print(f"[NOVEL_INTELLIGENCE] Intelligence creation completed for query: {query[:50]}...")
    
    return result
