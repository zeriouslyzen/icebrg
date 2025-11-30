"""
Archaeologist Agent - Wrapper for legacy protocol compatibility
"""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import from modular system
try:
    from ..protocol.execution.agents.archaeologist import run as _run_modular
    from ..protocol.config import ProtocolConfig
    from ..config import load_config, IceburgConfig
except ImportError:
    # Fallback: implement directly
    _run_modular = None


def run(cfg, query: str, documents: Optional[List[Union[str, bytes, Path, Dict]]] = None, 
        verbose: bool = False, **kwargs) -> str:
    """
    Deep research archaeologist that uncovers buried evidence and historical insights.
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
                    temperature=0.3,
                    max_tokens=1000
                )
                return _run_modular(protocol_cfg, query, documents=documents, verbose=verbose)
            else:
                return _run_modular(cfg, query, documents=documents, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[ARCHAEOLOGIST] Modular version failed: {e}, using fallback")
    
    # Fallback implementation
    ARCHAEOLOGIST_SYSTEM = (
        "ROLE: Deep Research Archaeologist and Historical Evidence Excavator\n"
        "MISSION: Uncover buried evidence, suppressed research, and historical insights\n"
        "Present findings objectively with verifiable evidence."
    )
    
    document_context = ""
    if documents:
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                document_context += f"\nDocument {i+1}: {doc[:500]}...\n"
            elif isinstance(doc, dict):
                document_context += f"\nDocument {i+1}: {str(doc)[:500]}...\n"
            elif isinstance(doc, Path):
                try:
                    content = doc.read_text()[:500]
                    document_context += f"\nDocument {i+1} ({doc.name}): {content}...\n"
                except Exception:
                    pass
    
    prompt = f"RESEARCH QUERY: {query}\n\n"
    if document_context:
        prompt += f"PROVIDED DOCUMENTS:\n{document_context}\n\n"
    prompt += "Uncover buried evidence, suppressed research, and historical insights that challenge conventional understanding."
    
    result = chat_complete(
        cfg.surveyor_model,
        prompt,
        system=ARCHAEOLOGIST_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="Archaeologist",
    )
    
    if verbose:
        print(f"[ARCHAEOLOGIST] Excavation completed")
    
    return result

