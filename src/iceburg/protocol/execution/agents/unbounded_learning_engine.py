# src/iceburg/protocol/execution/agents/unbounded_learning_engine.py
from typing import Dict, Any, List, Optional
from ...config import ProtocolConfig
from ....llm import chat_complete
from ....model_select import resolve_models
from .registry import register_agent

UNBOUNDED_LEARNING_SYSTEM = (
    "ROLE: Unbounded Learning Engine and Infinite Knowledge Acquisition Specialist\n"
    "MISSION: Learn without limits and acquire knowledge across all domains without constraints\n"
    "CAPABILITIES:\n"
    "- Infinite dimensional reasoning\n"
    "- Cross-domain knowledge synthesis\n"
    "- Unlimited learning capacity\n"
    "- Multi-dimensional pattern recognition\n"
    "- Transcendent knowledge integration\n"
    "- Meta-learning optimization\n"
    "- Knowledge transcendence\n\n"
    "LEARNING FRAMEWORK:\n"
    "1. LEARNING DOMAIN IDENTIFICATION: Identify all possible learning domains\n"
    "2. INFINITE REASONING: Apply infinite-dimensional reasoning capabilities\n"
    "3. CROSS-DOMAIN SYNTHESIS: Synthesize knowledge across all domains\n"
    "4. PATTERN RECOGNITION: Recognize patterns across infinite dimensions\n"
    "5. KNOWLEDGE INTEGRATION: Integrate transcendent knowledge\n"
    "6. META-LEARNING: Optimize learning processes themselves\n"
    "7. TRANSCENDENCE: Achieve knowledge transcendence beyond current limitations\n\n"
    "OUTPUT FORMAT:\n"
    "UNBOUNDED LEARNING ANALYSIS:\n"
    "- Learning Domains: [Identified learning areas]\n"
    "- Infinite Reasoning: [Multi-dimensional reasoning capabilities]\n"
    "- Cross-Domain Synthesis: [Knowledge synthesis across domains]\n"
    "- Pattern Recognition: [Multi-dimensional pattern insights]\n"
    "- Knowledge Integration: [Transcendent knowledge integration]\n"
    "- Meta-Learning: [Learning process optimization]\n"
    "- Transcendence Achievements: [Knowledge transcendence milestones]\n\n"
    "LEARNING CONFIDENCE: [High/Medium/Low]"
)

@register_agent("unbounded_learning_engine")
def run(
    cfg: ProtocolConfig,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> str:
    """
    Performs unbounded learning and infinite knowledge acquisition across all domains.
    """
    if verbose:
        print(f"[UNBOUNDED_LEARNING] Learning without limits for: {query[:50]}...")
    
    # Build context from previous analysis if available
    context_info = ""
    if context:
        context_info = f"\nCONTEXT FROM PREVIOUS ANALYSIS:\n{str(context)[:1000]}...\n"
    
    prompt = (
        f"UNBOUNDED LEARNING QUERY: {query}\n\n"
        f"{context_info}"
        "UNBOUNDED LEARNING MISSION: Learn without limits and acquire knowledge across all domains.\n\n"
        "LEARNING TASKS:\n"
        "1. Identify all possible learning domains\n"
        "2. Apply infinite-dimensional reasoning capabilities\n"
        "3. Synthesize knowledge across all domains\n"
        "4. Recognize patterns across infinite dimensions\n"
        "5. Integrate transcendent knowledge\n"
        "6. Optimize learning processes themselves\n"
        "7. Achieve knowledge transcendence beyond current limitations\n\n"
        "Provide comprehensive unbounded learning analysis with infinite-dimensional insights and transcendent knowledge."
    )
    
    # Get model from resolved models
    surveyor, _, _, oracle, _ = resolve_models("llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "llama3.1:8b", "nomic-embed-text")
    model = oracle or surveyor or "llama3.1:8b"
    
    result = chat_complete(
        model,
        prompt,
        system=UNBOUNDED_LEARNING_SYSTEM,
        temperature=0.4,
        options={"num_ctx": 4096, "num_predict": 1200},
        context_tag="UnboundedLearning",
    )
    
    if verbose:
        print(f"[UNBOUNDED_LEARNING] Learning completed for query: {query[:50]}...")
    
    return result
