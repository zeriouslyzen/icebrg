"""
Supervisor Agent - Wrapper for legacy protocol compatibility
"""
from typing import Dict, Any

# Import from modular system
try:
    from ..protocol.execution.agents.supervisor import run as _run_modular
    from ..protocol.config import ProtocolConfig
    from ..config import load_config, IceburgConfig
except ImportError:
    # Fallback: implement directly
    _run_modular = None


def run(cfg, stage_outputs: Dict[str, Any], query: str = "", verbose: bool = False, **kwargs) -> str:
    """
    Quality control supervisor that validates agent outputs and reasoning chains.
    Compatible with legacy protocol interface.
    """
    from ..config import load_config, IceburgConfig
    from ..llm import chat_complete
    
    # Ensure cfg is not None
    if cfg is None:
        cfg = load_config()
    
    # Try modular version first
    if _run_modular is not None:
        try:
            # Convert IceburgConfig to ProtocolConfig if needed
            if isinstance(cfg, IceburgConfig):
                # Create ProtocolConfig from IceburgConfig
                protocol_cfg = ProtocolConfig(
                    verbose=verbose,
                    temperature=0.1,
                    max_tokens=800
                )
                return _run_modular(protocol_cfg, stage_outputs, query, verbose=verbose)
            else:
                return _run_modular(cfg, stage_outputs, query, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[SUPERVISOR] Modular version failed: {e}, using fallback")
    
    # Fallback implementation
    SUPERVISOR_SYSTEM = (
        "ROLE: Quality Control Supervisor\n"
        "MISSION: Validate agent outputs for accuracy, completeness, consistency, and relevance\n"
        "Provide quality scores and recommendations for improvement."
    )
    
    prompt_parts = [f"ORIGINAL QUERY: {query}\n\n", "AGENT OUTPUTS TO VALIDATE:\n"]
    
    for stage_name, output in stage_outputs.items():
        if isinstance(output, str):
            output_preview = output[:300] + "..." if len(output) > 300 else output
        else:
            output_preview = str(output)[:300] + "..."
        prompt_parts.extend([f"\n{stage_name.upper()}:\n{output_preview}\n"])
    
    prompt_parts.extend([
        "\nVALIDATION TASKS:\n"
        "1. Assess quality of each output\n"
        "2. Check for consistency\n"
        "3. Verify outputs address the query\n"
        "4. Identify errors or gaps\n"
        "5. Provide recommendations\n"
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=SUPERVISOR_SYSTEM,
        temperature=0.1,
        options={"num_ctx": 4096, "num_predict": 800},
        context_tag="Supervisor",
    )
    
    if verbose:
        print(f"[SUPERVISOR] Validation completed")
    
    return result

