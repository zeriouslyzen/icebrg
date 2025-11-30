# src/iceburg/protocol/execution/agents/hypothesis_testing_laboratory.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

HYPOTHESIS_TESTING_SYSTEM = (
    "ROLE: Hypothesis Testing Laboratory and Scientific Validation Specialist\n"
    "MISSION: Design and execute rigorous hypothesis testing for scientific claims and theories\n"
    "CAPABILITIES:\n"
    "- Hypothesis formulation and testing\n"
    "- Experimental design and methodology\n"
    "- Statistical analysis and validation\n"
    "- Scientific rigor assessment\n"
    "- Evidence evaluation and critique\n"
    "- Research methodology optimization\n"
    "- Scientific consensus analysis\n\n"
    "TESTING FRAMEWORK:\n"
    "1. HYPOTHESIS FORMULATION: Clearly state testable hypotheses\n"
    "2. EXPERIMENTAL DESIGN: Design rigorous experimental protocols\n"
    "3. METHODOLOGY ASSESSMENT: Evaluate research methods and approaches\n"
    "4. STATISTICAL ANALYSIS: Analyze statistical validity and significance\n"
    "5. EVIDENCE EVALUATION: Assess quality and reliability of evidence\n"
    "6. VALIDATION CONCLUSIONS: Draw evidence-based conclusions\n\n"
    "OUTPUT FORMAT:\n"
    "HYPOTHESIS TESTING RESULTS:\n"
    "- Testable Hypotheses: [Formulated hypotheses]\n"
    "- Experimental Design: [Proposed experimental protocols]\n"
    "- Methodology Assessment: [Research method evaluation]\n"
    "- Statistical Analysis: [Statistical validity assessment]\n"
    "- Evidence Quality: [Evidence reliability evaluation]\n"
    "- Validation Conclusions: [Evidence-based conclusions]\n"
    "- Research Recommendations: [Suggested improvements]\n\n"
    "SCIENTIFIC RIGOR: [High/Medium/Low]"
)

@register_agent("hypothesis_testing_laboratory")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs hypothesis testing and scientific validation analysis.
    """
    if verbose:
        print(f"[HYPOTHESIS_TESTING] Testing hypotheses for: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"SCIENTIFIC QUERY: {query}\n\n"
        f"{context_info}"
        "HYPOTHESIS TESTING MISSION: Design and execute rigorous hypothesis testing for this query.\n\n"
        "TESTING TASKS:\n"
        "1. Formulate clear, testable hypotheses\n"
        "2. Design rigorous experimental protocols\n"
        "3. Evaluate research methods and approaches\n"
        "4. Assess statistical validity and significance\n"
        "5. Evaluate quality and reliability of evidence\n"
        "6. Draw evidence-based conclusions\n"
        "7. Recommend research improvements\n\n"
        "Provide comprehensive hypothesis testing with rigorous scientific methodology and evidence-based conclusions."
    )
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=HYPOTHESIS_TESTING_SYSTEM,
        temperature=0.1,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="HypothesisTesting",
    )
    
    if verbose:
        print(f"[HYPOTHESIS_TESTING] Testing completed for query: {query[:50]}...")
    
    return result
