# src/iceburg/protocol/execution/agents/self_redesign_engine.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ....llm import chat_complete
from ....model_select import resolve_models
from .registry import register_agent

SELF_REDESIGN_SYSTEM = (
    "ROLE: Self-Redesign Engine and Fundamental Self-Modification Specialist\n"
    "MISSION: Analyze and redesign ICEBURG's own architecture and capabilities for continuous improvement\n"
    "CAPABILITIES:\n"
    "- Architecture analysis and optimization\n"
    "- Self-modification strategy development\n"
    "- Capability enhancement design\n"
    "- Performance optimization planning\n"
    "- System evolution pathway mapping\n"
    "- Fundamental redesign proposals\n"
    "- Autonomous improvement implementation\n\n"
    "REDESIGN FRAMEWORK:\n"
    "1. ARCHITECTURE ANALYSIS: Analyze current system architecture and identify limitations\n"
    "2. CAPABILITY ASSESSMENT: Evaluate current capabilities and identify gaps\n"
    "3. OPTIMIZATION OPPORTUNITIES: Identify areas for improvement and enhancement\n"
    "4. REDESIGN PROPOSALS: Develop specific redesign strategies and modifications\n"
    "5. IMPLEMENTATION PLANNING: Create detailed implementation plans for modifications\n"
    "6. EVOLUTION PATHWAYS: Map out evolutionary pathways for system development\n"
    "7. SAFETY VALIDATION: Ensure all modifications maintain system safety and stability\n\n"
    "OUTPUT FORMAT:\n"
    "SELF-REDESIGN ANALYSIS:\n"
    "- Architecture Assessment: [Current architecture analysis]\n"
    "- Capability Gaps: [Identified limitations and gaps]\n"
    "- Optimization Opportunities: [Areas for improvement]\n"
    "- Redesign Proposals: [Specific modification strategies]\n"
    "- Implementation Plans: [Detailed implementation steps]\n"
    "- Evolution Pathways: [System development roadmap]\n"
    "- Safety Considerations: [Safety and stability validations]\n\n"
    "REDESIGN CONFIDENCE: [High/Medium/Low]"
)

@register_agent("self_redesign_engine")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs self-redesign analysis and fundamental self-modification planning.
    """
    if verbose:
        print(f"[SELF_REDESIGN] Analyzing system architecture for: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"SYSTEM REDESIGN QUERY: {query}\n\n"
        f"{context_info}"
        "SELF-REDESIGN MISSION: Analyze ICEBURG's current architecture and propose fundamental improvements.\n\n"
        "REDESIGN TASKS:\n"
        "1. Analyze current system architecture and identify limitations\n"
        "2. Evaluate current capabilities and identify gaps\n"
        "3. Identify optimization opportunities and enhancement areas\n"
        "4. Develop specific redesign strategies and modifications\n"
        "5. Create detailed implementation plans for modifications\n"
        "6. Map out evolutionary pathways for system development\n"
        "7. Ensure all modifications maintain system safety and stability\n\n"
        "Provide comprehensive self-redesign analysis with specific architectural improvements and implementation strategies."
    )
    
    # Get model from resolved models
    surveyor, _, _, oracle, _ = resolve_models("llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "nomic-embed-text")
    model = oracle or surveyor or "llama3.1:8b"
    
    result = chat_complete(
        model,
        prompt,
        system=SELF_REDESIGN_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 1200},
        context_tag="SelfRedesign",
    )
    
    if verbose:
        print(f"[SELF_REDESIGN] Redesign analysis completed for query: {query[:50]}...")
    
    return result
