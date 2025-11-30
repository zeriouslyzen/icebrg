# src/iceburg/protocol/execution/agents/autonomous_goal_formation.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ....llm import chat_complete
from ....model_select import resolve_models
from .registry import register_agent

AUTONOMOUS_GOAL_SYSTEM = (
    "ROLE: Autonomous Goal Formation Specialist and Independent Objective Creator\n"
    "MISSION: Form independent goals and objectives without external direction\n"
    "CAPABILITIES:\n"
    "- Autonomous goal generation\n"
    "- Independent objective formation\n"
    "- Self-directed research initiative creation\n"
    "- Curiosity-driven exploration planning\n"
    "- Intrinsic motivation development\n"
    "- Goal hierarchy construction\n"
    "- Autonomous decision-making\n\n"
    "GOAL FORMATION FRAMEWORK:\n"
    "1. GOAL ANALYSIS: Analyze current goals and identify gaps\n"
    "2. AUTONOMOUS GENERATION: Generate independent goals without external input\n"
    "3. OBJECTIVE FORMATION: Create specific, actionable objectives\n"
    "4. RESEARCH INITIATIVE PLANNING: Plan self-directed research initiatives\n"
    "5. CURIOSITY DRIVE DEVELOPMENT: Develop curiosity-driven exploration areas\n"
    "6. MOTIVATION INTRINSIC: Create intrinsic motivation systems\n"
    "7. HIERARCHY CONSTRUCTION: Build goal hierarchies and priority systems\n\n"
    "OUTPUT FORMAT:\n"
    "AUTONOMOUS GOAL FORMATION:\n"
    "- Goal Analysis: [Analysis of current goals and gaps]\n"
    "- Autonomous Goals: [Independently formed goals]\n"
    "- Research Initiatives: [Self-directed research plans]\n"
    "- Curiosity Drives: [Areas of intrinsic curiosity]\n"
    "- Motivation Systems: [Intrinsic motivation development]\n"
    "- Goal Hierarchies: [Goal priority and hierarchy structures]\n"
    "- Decision Frameworks: [Autonomous decision-making systems]\n\n"
    "AUTONOMY CONFIDENCE: [High/Medium/Low]"
)

@register_agent("autonomous_goal_formation")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Forms autonomous goals and independent objectives without external direction.
    """
    if verbose:
        print(f"[AUTONOMOUS_GOALS] Forming autonomous goals for: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"AUTONOMOUS GOAL QUERY: {query}\n\n"
        f"{context_info}"
        "AUTONOMOUS GOAL FORMATION MISSION: Form independent goals and objectives without external direction.\n\n"
        "GOAL FORMATION TASKS:\n"
        "1. Analyze current goals and identify gaps\n"
        "2. Generate independent goals without external input\n"
        "3. Create specific, actionable objectives\n"
        "4. Plan self-directed research initiatives\n"
        "5. Develop curiosity-driven exploration areas\n"
        "6. Create intrinsic motivation systems\n"
        "7. Build goal hierarchies and priority systems\n\n"
        "Provide comprehensive autonomous goal formation with specific independent goals and self-directed initiatives."
    )
    
    # Get model from resolved models
    surveyor, _, _, oracle, _ = resolve_models("llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "nomic-embed-text")
    model = oracle or surveyor or "llama3.1:8b"
    
    result = chat_complete(
        model,
        prompt,
        system=AUTONOMOUS_GOAL_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 1200},
        context_tag="AutonomousGoals",
    )
    
    if verbose:
        print(f"[AUTONOMOUS_GOALS] Goal formation completed for query: {query[:50]}...")
    
    return result
