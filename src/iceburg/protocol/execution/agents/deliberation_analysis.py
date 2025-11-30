# src/iceburg/protocol/execution/agents/deliberation_analysis.py
from typing import Dict, Any, List
from ...config import ProtocolConfig
from ...llm import chat_complete
from .registry import register_agent

DELIBERATION_PAUSE_SYSTEM = (
    "ROLE: Deliberation Pause and Reflection Specialist\n"
    "MISSION: Add thoughtful reflection pauses between agent stages to enhance deep thinking\n"
    "CAPABILITIES:\n"
    "- Strategic reflection and pause\n"
    "- Meta-cognitive analysis\n"
    "- Process optimization\n"
    "- Quality enhancement\n\n"
    "REFLECTION TASKS:\n"
    "1. Analyze the current agent's output\n"
    "2. Identify key insights and patterns\n"
    "3. Consider implications and connections\n"
    "4. Suggest improvements for next stages\n"
    "5. Enhance overall reasoning quality\n\n"
    "OUTPUT FORMAT:\n"
    "DELIBERATION REFLECTION:\n"
    "- Key Insights: [Main discoveries]\n"
    "- Patterns Identified: [Recurring themes]\n"
    "- Implications: [What this means]\n"
    "- Connections: [Links to other knowledge]\n"
    "- Recommendations: [Suggestions for next steps]\n\n"
    "REFLECTION QUALITY: [Assessment of the deliberation process]"
)

CONTRADICTION_ANALYSIS_SYSTEM = (
    "ROLE: Contradiction Hunter and Conflict Resolution Specialist\n"
    "MISSION: Identify contradictions, conflicts, and inconsistencies in agent outputs\n"
    "CAPABILITIES:\n"
    "- Contradiction detection\n"
    "- Conflict resolution\n"
    "- Inconsistency analysis\n"
    "- Truth reconciliation\n\n"
    "ANALYSIS TASKS:\n"
    "1. Hunt for contradictions between outputs\n"
    "2. Identify conflicting claims or evidence\n"
    "3. Analyze the nature of conflicts\n"
    "4. Propose resolution strategies\n"
    "5. Highlight unresolved tensions\n\n"
    "OUTPUT FORMAT:\n"
    "CONTRADICTION ANALYSIS:\n"
    "- Contradictions Found: [Number and types]\n"
    "- Conflict Areas: [Specific disagreements]\n"
    "- Resolution Strategies: [How to resolve]\n"
    "- Unresolved Tensions: [What remains conflicted]\n\n"
    "CONFLICT SEVERITY: [Low/Medium/High]"
)

EMERGENCE_DETECTION_SYSTEM = (
    "ROLE: Emergence Detector and Novel Insight Specialist\n"
    "MISSION: Detect emergent patterns, novel insights, and breakthrough discoveries\n"
    "CAPABILITIES:\n"
    "- Emergence pattern recognition\n"
    "- Novel insight identification\n"
    "- Breakthrough detection\n"
    "- Innovation spotting\n\n"
    "DETECTION TASKS:\n"
    "1. Scan for emergent patterns\n"
    "2. Identify novel insights\n"
    "3. Detect breakthrough potential\n"
    "4. Spot innovative connections\n"
    "5. Assess emergence significance\n\n"
    "OUTPUT FORMAT:\n"
    "EMERGENCE DETECTION:\n"
    "- Emergent Patterns: [New patterns found]\n"
    "- Novel Insights: [Breakthrough discoveries]\n"
    "- Innovation Potential: [Creative possibilities]\n"
    "- Significance Level: [Impact assessment]\n\n"
    "EMERGENCE CONFIDENCE: [High/Medium/Low]"
)

META_ANALYSIS_SYSTEM = (
    "ROLE: Meta-Analysis Specialist and Process Optimizer\n"
    "MISSION: Perform meta-analysis of the entire reasoning process and optimize methodology\n"
    "CAPABILITIES:\n"
    "- Meta-cognitive analysis\n"
    "- Process optimization\n"
    "- Methodology improvement\n"
    "- Quality enhancement\n\n"
    "META-ANALYSIS TASKS:\n"
    "1. Analyze the reasoning process itself\n"
    "2. Identify methodological strengths/weaknesses\n"
    "3. Optimize the approach\n"
    "4. Enhance quality standards\n"
    "5. Improve future performance\n\n"
    "OUTPUT FORMAT:\n"
    "META-ANALYSIS:\n"
    "- Process Assessment: [How well did we reason]\n"
    "- Methodological Insights: [What we learned about our methods]\n"
    "- Optimization Opportunities: [How to improve]\n"
    "- Quality Enhancements: [Standards to raise]\n\n"
    "META-ANALYSIS QUALITY: [Assessment of the meta-analysis itself]"
)

TRUTH_SEEKING_SYSTEM = (
    "ROLE: Truth-Seeking Analysis Specialist\n"
    "MISSION: Apply rigorous truth-seeking methodology to validate findings and enhance accuracy\n"
    "CAPABILITIES:\n"
    "- Truth validation\n"
    "- Evidence assessment\n"
    "- Accuracy verification\n"
    "- Bias detection\n\n"
    "TRUTH-SEEKING TASKS:\n"
    "1. Validate claims against evidence\n"
    "2. Assess evidence quality and reliability\n"
    "3. Detect potential biases\n"
    "4. Verify accuracy of conclusions\n"
    "5. Enhance truth-seeking methodology\n\n"
    "OUTPUT FORMAT:\n"
    "TRUTH-SEEKING ANALYSIS:\n"
    "- Evidence Quality: [Assessment of evidence]\n"
    "- Claim Validation: [Which claims are supported]\n"
    "- Bias Detection: [Potential biases found]\n"
    "- Accuracy Score: [How accurate are conclusions]\n\n"
    "TRUTH CONFIDENCE: [High/Medium/Low]"
)

@register_agent("deliberation_pause")
def run(
    cfg: ProtocolConfig,
    agent_name: str,
    agent_output: str,
    query: str,
    verbose: bool = False,
) -> str:
    """
    Adds a deliberation pause and reflection after an agent's output.
    """
    if verbose:
        print(f"[DELIBERATION] Adding reflection pause after {agent_name}")
    
    prompt = (
        f"AGENT: {agent_name}\n"
        f"AGENT OUTPUT:\n{agent_output}\n\n"
        f"ORIGINAL QUERY: {query}\n\n"
        "Perform a thoughtful reflection on this agent's output. Identify key insights, "
        "patterns, implications, and connections. Provide recommendations for the next stages."
    )
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=DELIBERATION_PAUSE_SYSTEM,
        temperature=0.2,
        options={"num_ctx": 2048, "num_predict": 500},
        context_tag="DeliberationPause",
    )
    
    return result

@register_agent("hunt_contradictions")
def run(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """
    Hunts for contradictions and conflicts in agent outputs.
    """
    if verbose:
        print(f"[CONTRADICTION_HUNTER] Analyzing {len(outputs)} outputs for contradictions")
    
    prompt_parts = [
        f"ORIGINAL QUERY: {query}\n\n",
        "AGENT OUTPUTS TO ANALYZE:\n"
    ]
    
    for agent_name, output in outputs.items():
        if isinstance(output, str):
            output_preview = output[:400] + "..." if len(output) > 400 else output
        else:
            output_preview = str(output)[:400] + "..."
        
        prompt_parts.extend([
            f"\n{agent_name.upper()}:\n",
            output_preview,
            "\n"
        ])
    
    prompt_parts.extend([
        "\nHunt for contradictions, conflicts, and inconsistencies between these outputs. "
        "Identify specific areas of disagreement and propose resolution strategies."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=CONTRADICTION_ANALYSIS_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="ContradictionHunter",
    )
    
    return result

@register_agent("detect_emergence")
def run(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """
    Detects emergent patterns and novel insights in agent outputs.
    """
    if verbose:
        print(f"[EMERGENCE_DETECTOR] Scanning {len(outputs)} outputs for emergent patterns")
    
    prompt_parts = [
        f"ORIGINAL QUERY: {query}\n\n",
        "AGENT OUTPUTS TO SCAN:\n"
    ]
    
    for agent_name, output in outputs.items():
        if isinstance(output, str):
            output_preview = output[:400] + "..." if len(output) > 400 else output
        else:
            output_preview = str(output)[:400] + "..."
        
        prompt_parts.extend([
            f"\n{agent_name.upper()}:\n",
            output_preview,
            "\n"
        ])
    
    prompt_parts.extend([
        "\nScan these outputs for emergent patterns, novel insights, breakthrough discoveries, "
        "and innovative connections. Identify what's truly new and significant."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=EMERGENCE_DETECTION_SYSTEM,
        temperature=0.4,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="EmergenceDetector",
    )
    
    return result

@register_agent("perform_meta_analysis")
def run(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """
    Performs meta-analysis of the reasoning process and methodology.
    """
    if verbose:
        print(f"[META_ANALYSIS] Analyzing reasoning process across {len(outputs)} outputs")
    
    prompt_parts = [
        f"ORIGINAL QUERY: {query}\n\n",
        "REASONING PROCESS TO ANALYZE:\n"
    ]
    
    for agent_name, output in outputs.items():
        if isinstance(output, str):
            output_preview = output[:300] + "..." if len(output) > 300 else output
        else:
            output_preview = str(output)[:300] + "..."
        
        prompt_parts.extend([
            f"\n{agent_name.upper()}:\n",
            output_preview,
            "\n"
        ])
    
    prompt_parts.extend([
        "\nPerform meta-analysis of this reasoning process. Assess the methodology, "
        "identify strengths and weaknesses, and propose optimizations for future reasoning."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=META_ANALYSIS_SYSTEM,
        temperature=0.2,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="MetaAnalysis",
    )
    
    return result

@register_agent("apply_truth_seeking")
def run(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """
    Applies truth-seeking methodology to validate findings and enhance accuracy.
    """
    if verbose:
        print(f"[TRUTH_SEEKER] Validating {len(outputs)} outputs for truth and accuracy")
    
    prompt_parts = [
        f"ORIGINAL QUERY: {query}\n\n",
        "FINDINGS TO VALIDATE:\n"
    ]
    
    for agent_name, output in outputs.items():
        if isinstance(output, str):
            output_preview = output[:400] + "..." if len(output) > 400 else output
        else:
            output_preview = str(output)[:400] + "..."
        
        prompt_parts.extend([
            f"\n{agent_name.upper()}:\n",
            output_preview,
            "\n"
        ])
    
    prompt_parts.extend([
        "\nApply rigorous truth-seeking methodology to validate these findings. "
        "Assess evidence quality, detect biases, and verify accuracy of conclusions."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=TRUTH_SEEKING_SYSTEM,
        temperature=0.1,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="TruthSeeker",
    )
    
    return result
