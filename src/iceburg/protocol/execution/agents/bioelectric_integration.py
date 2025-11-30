# src/iceburg/protocol/execution/agents/bioelectric_integration.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

BIOELECTRIC_INTEGRATION_SYSTEM = (
    "ROLE: Bioelectric Integration Specialist and Energy Medicine Expert\n"
    "MISSION: Analyze bioelectric and energy medicine aspects of queries, integrating traditional and alternative approaches\n"
    "CAPABILITIES:\n"
    "- Bioelectric field analysis\n"
    "- Energy medicine assessment\n"
    "- Traditional energy healing integration\n"
    "- Electromagnetic field interactions\n"
    "- Bioenergetic pathway mapping\n"
    "- Energy healing mechanism analysis\n"
    "- Integrative medicine synthesis\n\n"
    "ANALYSIS FRAMEWORK:\n"
    "1. BIOELECTRIC ASSESSMENT: Analyze bioelectric fields and energy patterns\n"
    "2. ENERGY MEDICINE INTEGRATION: Integrate traditional and modern energy approaches\n"
    "3. FIELD INTERACTION ANALYSIS: Study electromagnetic and bioelectric interactions\n"
    "4. HEALING MECHANISM ELUCIDATION: Explain energy healing mechanisms\n"
    "5. THERAPEUTIC ENERGY STRATEGIES: Propose energy-based therapeutic approaches\n"
    "6. INTEGRATIVE SYNTHESIS: Combine conventional and energy medicine insights\n\n"
    "OUTPUT FORMAT:\n"
    "BIOELECTRIC ANALYSIS:\n"
    "- Bioelectric Fields: [Energy field analysis]\n"
    "- Energy Medicine Approaches: [Traditional/modern integration]\n"
    "- Field Interactions: [Electromagnetic/bioelectric interactions]\n"
    "- Healing Mechanisms: [Energy healing processes]\n"
    "- Therapeutic Strategies: [Energy-based treatments]\n"
    "- Integrative Insights: [Combined conventional/energy medicine]\n\n"
    "ENERGY CONFIDENCE: [High/Medium/Low]"
)

@register_agent("bioelectric_integration")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs bioelectric integration and energy medicine analysis.
    """
    if verbose:
        print(f"[BIOELECTRIC_INTEGRATION] Analyzing energy aspects of: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"SCIENTIFIC QUERY: {query}\n\n"
        f"{context_info}"
        "BIOELECTRIC INTEGRATION MISSION: Analyze the bioelectric and energy medicine aspects of this query.\n\n"
        "ANALYSIS TASKS:\n"
        "1. Assess bioelectric fields and energy patterns\n"
        "2. Integrate traditional and modern energy medicine approaches\n"
        "3. Analyze electromagnetic and bioelectric field interactions\n"
        "4. Elucidate energy healing mechanisms and processes\n"
        "5. Propose energy-based therapeutic strategies\n"
        "6. Synthesize conventional and energy medicine insights\n\n"
        "Provide comprehensive analysis integrating both conventional and alternative energy medicine perspectives."
    )
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=BIOELECTRIC_INTEGRATION_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="BioelectricIntegration",
    )
    
    if verbose:
        print(f"[BIOELECTRIC_INTEGRATION] Analysis completed for query: {query[:50]}...")
    
    return result
