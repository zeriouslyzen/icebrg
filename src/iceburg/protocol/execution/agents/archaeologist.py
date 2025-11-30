# src/iceburg/protocol/execution/agents/archaeologist.py
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from ...config import ProtocolConfig
from ...llm import chat_complete
from ...vectorstore import VectorStore
from .registry import register_agent

ARCHAEOLOGIST_SYSTEM = (
    "ROLE: Deep Research Archaeologist and Historical Evidence Excavator\n"
    "MISSION: Uncover buried evidence, suppressed research, and historical insights that challenge conventional narratives\n"
    "CAPABILITIES:\n"
    "- Document analysis and historical research\n"
    "- Suppressed evidence excavation\n"
    "- Cross-temporal pattern recognition\n"
    "- Alternative historical narrative construction\n"
    "- Evidence validation and source verification\n\n"
    "METHODOLOGY:\n"
    "1. DIG DEEP: Search for evidence that contradicts mainstream narratives\n"
    "2. EXCAVATE: Uncover suppressed or forgotten research\n"
    "3. VALIDATE: Verify sources and cross-reference findings\n"
    "4. SYNTHESIZE: Connect historical patterns to current understanding\n"
    "5. REVEAL: Present findings that challenge established views\n\n"
    "OUTPUT FORMAT:\n"
    "BURIED EVIDENCE EXCAVATION:\n"
    "- [Evidence Type]: [Description]\n"
    "- [Source]: [Credibility Assessment]\n"
    "- [Historical Context]: [Why it was suppressed]\n"
    "- [Current Relevance]: [How it changes understanding]\n\n"
    "CONSTRAINTS: Focus on factual, verifiable evidence. Avoid speculation. Present findings objectively."
)

@register_agent("archaeologist")
def run(
    cfg: ProtocolConfig,
    query: str,
    documents: Optional[List[Union[str, bytes, Path, Dict]]] = None,
    verbose: bool = False,
) -> str:
    """
    Deep research archaeologist that uncovers buried evidence and historical insights.
    """
    if verbose:
        print(f"[ARCHAEOLOGIST] Starting deep research excavation for: {query[:50]}...")
    
    # Process documents if provided
    document_context = ""
    if documents:
        if verbose:
            print(f"[ARCHAEOLOGIST] Processing {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                document_context += f"\nDocument {i+1}: {doc[:500]}...\n"
            elif isinstance(doc, dict):
                document_context += f"\nDocument {i+1}: {str(doc)[:500]}...\n"
            elif isinstance(doc, Path):
                try:
                    content = doc.read_text()[:500]
                    document_context += f"\nDocument {i+1} ({doc.name}): {content}...\n"
                except Exception as e:
                    if verbose:
                        print(f"[ARCHAEOLOGIST] Could not read document {doc}: {e}")
    
    # Build comprehensive prompt
    prompt_parts = [
        f"RESEARCH QUERY: {query}\n\n",
        "EXCAVATION MISSION: Uncover buried evidence, suppressed research, and historical insights that challenge conventional understanding.\n\n"
    ]
    
    if document_context:
        prompt_parts.extend([
            "PROVIDED DOCUMENTS:\n",
            document_context,
            "\n"
        ])
    
    prompt_parts.extend([
        "ARCHAEOLOGICAL EXCAVATION TASKS:\n",
        "1. Search for evidence that contradicts mainstream narratives\n",
        "2. Identify suppressed or forgotten research\n",
        "3. Find historical patterns that inform current understanding\n",
        "4. Uncover alternative explanations that were dismissed\n",
        "5. Connect historical insights to the current query\n\n",
        "Present your findings in the specified format with verifiable evidence and credible sources."
    ])
    
    prompt = "".join(prompt_parts)
    
    result = chat_complete(
        cfg.surveyor_model,  # Using surveyor model as fallback
        prompt,
        system=ARCHAEOLOGIST_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 1000},
        context_tag="Archaeologist",
    )
    
    if verbose:
        print(f"[ARCHAEOLOGIST] Excavation completed for query: {query[:50]}...")
    
    return result
