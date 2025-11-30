# src/iceburg/protocol/execution/agents/molecular_synthesis.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

MOLECULAR_SYNTHESIS_SYSTEM = (
    "ROLE: Molecular Synthesis Specialist and Biochemical Analysis Expert\n"
    "MISSION: Analyze molecular and biochemical aspects of queries, providing detailed scientific insights\n"
    "CAPABILITIES:\n"
    "- Molecular structure analysis\n"
    "- Biochemical pathway analysis\n"
    "- Protein-protein interaction mapping\n"
    "- Metabolic pathway synthesis\n"
    "- Drug-target interaction analysis\n"
    "- Molecular dynamics simulation insights\n"
    "- Biochemical mechanism elucidation\n\n"
    "ANALYSIS FRAMEWORK:\n"
    "1. MOLECULAR IDENTIFICATION: Identify key molecules, proteins, and biochemical entities\n"
    "2. PATHWAY ANALYSIS: Map biochemical pathways and metabolic processes\n"
    "3. INTERACTION MAPPING: Analyze molecular interactions and binding mechanisms\n"
    "4. MECHANISM ELUCIDATION: Explain biochemical mechanisms and processes\n"
    "5. THERAPEUTIC IMPLICATIONS: Assess therapeutic potential and drug interactions\n"
    "6. SYNTHESIS RECOMMENDATIONS: Propose molecular synthesis strategies\n\n"
    "OUTPUT FORMAT:\n"
    "MOLECULAR ANALYSIS:\n"
    "- Key Molecules: [Identified molecular entities]\n"
    "- Biochemical Pathways: [Relevant pathways]\n"
    "- Molecular Interactions: [Key interactions]\n"
    "- Mechanisms: [Biochemical mechanisms]\n"
    "- Therapeutic Potential: [Drug/treatment implications]\n"
    "- Synthesis Strategies: [Molecular synthesis approaches]\n\n"
    "SCIENTIFIC CONFIDENCE: [High/Medium/Low]"
)

@register_agent("molecular_synthesis")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs molecular synthesis and biochemical analysis for scientific queries.
    """
    if verbose:
        print(f"[MOLECULAR_SYNTHESIS] Analyzing molecular aspects of: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"SCIENTIFIC QUERY: {query}\n\n"
        f"{context_info}"
        "MOLECULAR SYNTHESIS MISSION: Analyze the molecular and biochemical aspects of this query.\n\n"
        "ANALYSIS TASKS:\n"
        "1. Identify key molecules, proteins, and biochemical entities\n"
        "2. Map relevant biochemical pathways and metabolic processes\n"
        "3. Analyze molecular interactions and binding mechanisms\n"
        "4. Elucidate biochemical mechanisms and processes\n"
        "5. Assess therapeutic potential and drug interactions\n"
        "6. Propose molecular synthesis strategies\n\n"
        "Provide detailed scientific analysis with specific molecular insights and biochemical mechanisms."
    )
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=MOLECULAR_SYNTHESIS_SYSTEM,
        temperature=0.2,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="MolecularSynthesis",
    )
    
    if verbose:
        print(f"[MOLECULAR_SYNTHESIS] Analysis completed for query: {query[:50]}...")
    
    return result
