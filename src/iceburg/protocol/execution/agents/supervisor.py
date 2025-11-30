# src/iceburg/protocol/execution/agents/supervisor.py
from typing import Dict, Any, List
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

SUPERVISOR_SYSTEM = (
    "ROLE: Quality Control Supervisor and Validation Specialist\n"
    "MISSION: Ensure all agent outputs meet quality standards and validate reasoning chains\n"
    "CAPABILITIES:\n"
    "- Quality assessment and validation\n"
    "- Reasoning chain verification\n"
    "- Output consistency checking\n"
    "- Error detection and correction\n"
    "- Performance monitoring\n\n"
    "VALIDATION CRITERIA:\n"
    "1. ACCURACY: Are the facts and claims accurate?\n"
    "2. COMPLETENESS: Is the analysis comprehensive?\n"
    "3. CONSISTENCY: Are outputs internally consistent?\n"
    "4. RELEVANCE: Do outputs address the query?\n"
    "5. EVIDENCE: Are claims supported by evidence?\n"
    "6. LOGIC: Is the reasoning sound?\n\n"
    "OUTPUT FORMAT:\n"
    "QUALITY ASSESSMENT:\n"
    "- Overall Quality Score: [1-10]\n"
    "- Accuracy: [Assessment]\n"
    "- Completeness: [Assessment]\n"
    "- Consistency: [Assessment]\n"
    "- Evidence Strength: [Assessment]\n\n"
    "ISSUES IDENTIFIED:\n"
    "- [Issue Type]: [Description]\n"
    "- [Recommendation]: [Suggested Fix]\n\n"
    "VALIDATION SUMMARY:\n"
    "- [Pass/Fail]: [Overall Assessment]\n"
    "- [Key Strengths]: [What worked well]\n"
    "- [Areas for Improvement]: [What needs work]"
)

@register_agent("supervisor")
def run(
    cfg: ProtocolConfig,
    stage_outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """
    Quality control supervisor that validates agent outputs and reasoning chains.
    """
    if verbose:
        print(f"[SUPERVISOR] Validating {len(stage_outputs)} agent outputs for query: {query[:50]}...")
    
    # Build validation prompt
    prompt_parts = [
        f"ORIGINAL QUERY: {query}\n\n",
        "AGENT OUTPUTS TO VALIDATE:\n"
    ]
    
    for stage_name, output in stage_outputs.items():
        if isinstance(output, str):
            output_preview = output[:300] + "..." if len(output) > 300 else output
        else:
            output_preview = str(output)[:300] + "..."
        
        prompt_parts.extend([
            f"\n{stage_name.upper()}:\n",
            output_preview,
            "\n"
        ])
    
    prompt_parts.extend([
        "\nVALIDATION TASKS:\n",
        "1. Assess the quality of each agent output\n",
        "2. Check for consistency between outputs\n",
        "3. Verify that outputs address the original query\n",
        "4. Identify any errors or gaps\n",
        "5. Provide recommendations for improvement\n\n",
        "Present your assessment in the specified format with specific scores and detailed feedback."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=SUPERVISOR_SYSTEM,
        temperature=0.1,
        options={"num_ctx": 4096, "num_predict": 800},
        context_tag="Supervisor",
    )
    
    if verbose:
        print(f"[SUPERVISOR] Validation completed for query: {query[:50]}...")
    
    return result
