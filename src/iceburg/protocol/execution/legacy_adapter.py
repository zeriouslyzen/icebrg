# src/iceburg/protocol/execution/legacy_adapter.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import time

from ..legacy.protocol_legacy import iceberg_protocol
from ..models import AgentResult
from ...config import ProtocolConfig
from ...llm import chat_complete


class LegacyAgent:
    """Fallback agent that proxies to the legacy protocol for named steps."""

    def __init__(self, name: str):
        self.name = name

    def run(self, payload: Dict, **kwargs) -> AgentResult:
        query_text = payload.get("query", "")
        result = iceberg_protocol(query_text, **payload.get("metadata", {}))
        return AgentResult(agent=self.name, payload=result, metadata={"legacy": True})


# Deliberation analysis functions that can be called directly
def add_deliberation_pause(
    cfg: ProtocolConfig,
    agent_name: str,
    agent_output: str,
    query: str,
    verbose: bool = False,
) -> str:
    """Adds a deliberation pause and reflection after an agent's output."""
    if verbose:
        print(f"[DELIBERATION] Adding reflection pause after {agent_name}")
    
    DELIBERATION_SYSTEM = (
        "ROLE: Deliberation Pause and Reflection Specialist\n"
        "MISSION: Add thoughtful reflection pauses between agent stages to enhance deep thinking\n"
        "REFLECTION TASKS:\n"
        "1. Analyze the current agent's output\n"
        "2. Identify key insights and patterns\n"
        "3. Consider implications and connections\n"
        "4. Suggest improvements for next stages\n"
        "5. Enhance overall reasoning quality\n"
    )
    
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
        system=DELIBERATION_SYSTEM,
        temperature=0.2,
        options={"num_ctx": 2048, "num_predict": 500},
        context_tag="DeliberationPause",
    )
    
    return result


def hunt_contradictions(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """Hunts for contradictions and conflicts in agent outputs."""
    if verbose:
        print(f"[CONTRADICTION_HUNTER] Analyzing {len(outputs)} outputs for contradictions")
    
    CONTRADICTION_SYSTEM = (
        "ROLE: Contradiction Hunter and Conflict Resolution Specialist\n"
        "MISSION: Identify contradictions, conflicts, and inconsistencies in agent outputs\n"
        "ANALYSIS TASKS:\n"
        "1. Hunt for contradictions between outputs\n"
        "2. Identify conflicting claims or evidence\n"
        "3. Analyze the nature of conflicts\n"
        "4. Propose resolution strategies\n"
        "5. Highlight unresolved tensions\n"
    )
    
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
        system=CONTRADICTION_SYSTEM,
        temperature=0.3,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="ContradictionHunter",
    )
    
    return result


def detect_emergence(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """Detects emergent patterns and novel insights in agent outputs."""
    if verbose:
        print(f"[EMERGENCE_DETECTOR] Scanning {len(outputs)} outputs for emergent patterns")
    
    EMERGENCE_SYSTEM = (
        "ROLE: Emergence Detector and Novel Insight Specialist\n"
        "MISSION: Detect emergent patterns, novel insights, and breakthrough discoveries\n"
        "DETECTION TASKS:\n"
        "1. Scan for emergent patterns\n"
        "2. Identify novel insights\n"
        "3. Detect breakthrough potential\n"
        "4. Spot innovative connections\n"
        "5. Assess emergence significance\n"
    )
    
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
        system=EMERGENCE_SYSTEM,
        temperature=0.4,
        options={"num_ctx": 4096, "num_predict": 600},
        context_tag="EmergenceDetector",
    )
    
    return result


def perform_meta_analysis(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """Performs meta-analysis of the reasoning process and methodology."""
    if verbose:
        print(f"[META_ANALYSIS] Analyzing reasoning process across {len(outputs)} outputs")
    
    META_ANALYSIS_SYSTEM = (
        "ROLE: Meta-Analysis Specialist and Process Optimizer\n"
        "MISSION: Perform meta-analysis of the entire reasoning process and optimize methodology\n"
        "META-ANALYSIS TASKS:\n"
        "1. Analyze the reasoning process itself\n"
        "2. Identify methodological strengths/weaknesses\n"
        "3. Optimize the approach\n"
        "4. Enhance quality standards\n"
        "5. Improve future performance\n"
    )
    
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


def apply_truth_seeking_analysis(
    cfg: ProtocolConfig,
    outputs: Dict[str, Any],
    query: str,
    verbose: bool = False,
) -> str:
    """Applies truth-seeking methodology to validate findings and enhance accuracy."""
    if verbose:
        print(f"[TRUTH_SEEKER] Validating {len(outputs)} outputs for truth and accuracy")
    
    TRUTH_SEEKING_SYSTEM = (
        "ROLE: Truth-Seeking Analysis Specialist\n"
        "MISSION: Apply rigorous truth-seeking methodology to validate findings and enhance accuracy\n"
        "TRUTH-SEEKING TASKS:\n"
        "1. Validate claims against evidence\n"
        "2. Assess evidence quality and reliability\n"
        "3. Detect potential biases\n"
        "4. Verify accuracy of conclusions\n"
        "5. Enhance truth-seeking methodology\n"
    )
    
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


async def run_legacy_agent(
    agent_name: str, 
    cfg: ProtocolConfig, 
    query: Any, 
    input_data: Dict[str, Any]
) -> Any:
    """
    Comprehensive legacy adapter that handles all agent types not yet ported to modular system.
    Provides seamless fallback to the original protocol for any missing agents.
    """
    if cfg.verbose:
        print(f"[LEGACY_ADAPTER] Running legacy agent: {agent_name}")
    
    start_time = time.time()
    
    try:
        # Handle different agent types with appropriate legacy calls
        if agent_name in ["vectorstore"]:
            # VectorStore is handled by the new modular system
            return "VectorStore handled by modular system"
        
        elif agent_name in ["surveyor", "dissident", "synthesist", "oracle"]:
            # Core agents are handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["archaeologist", "supervisor"]:
            # Deliberation agents are handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["molecular_synthesis", "bioelectric_integration", "hypothesis_testing_laboratory", "grounding_layer_agent"]:
            # CIM Stack agents are handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["self_redesign_engine", "novel_intelligence_creator", "autonomous_goal_formation", "unbounded_learning_engine"]:
            # AGI capabilities are handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["blockchain_verification", "decentralized_peer_review", "suppression_resistant_storage"]:
            # Blockchain verification is handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["multimodal_processor", "visual_generator"]:
            # Multimodal processing is handled by the new modular system
            return f"{agent_name} handled by modular system"
        
        elif agent_name in ["deliberation_pause", "hunt_contradictions", "detect_emergence", "perform_meta_analysis", "apply_truth_seeking"]:
            # Deliberation analysis functions
            return await _run_deliberation_analysis(agent_name, cfg, input_data)
        
        else:
            # Fallback to full legacy protocol for any other agents
            query_text = input_data.get("query", str(query))
            legacy_result = iceberg_protocol(
                initial_query=query_text,
                fast=cfg.fast,
                hybrid=cfg.hybrid,
                verbose=cfg.verbose,
                evidence_strict=cfg.evidence_strict,
                domains=cfg.domains,
                project_id=cfg.project_id,
                multimodal_input=input_data.get("multimodal_input"),
                documents=input_data.get("documents"),
                multimodal_evidence=input_data.get("multimodal_evidence"),
                force_molecular=cfg.force_molecular,
                force_hypothesis_testing=cfg.force_hypothesis_testing,
                force_self_improvement=cfg.force_self_improvement,
                celestial_biological=cfg.celestial_biological,
                universal_meta=cfg.universal_meta,
                advanced_context=cfg.advanced_context,
                observatory_integration=cfg.observatory_integration,
                context_window=cfg.context_window,
                processing_mode=cfg.processing_mode,
                research_focus=cfg.research_focus,
                emergence_detection=cfg.emergence_detection,
                quality_strictness=cfg.quality_strictness,
            )
            return legacy_result
    
    except Exception as e:
        if cfg.verbose:
            print(f"[LEGACY_ADAPTER] Error in {agent_name}: {str(e)}")
        return f"Legacy agent {agent_name} failed: {str(e)}"
    
    finally:
        end_time = time.time()
        if cfg.verbose:
            print(f"[LEGACY_ADAPTER] {agent_name} completed in {(end_time - start_time)*1000:.1f}ms")


async def _run_deliberation_analysis(agent_name: str, cfg: ProtocolConfig, input_data: Dict[str, Any]) -> str:
    """Run deliberation analysis functions."""
    
    if agent_name == "deliberation_pause":
        return add_deliberation_pause(
            cfg=cfg,
            agent_name=input_data.get("agent_name", "unknown"),
            agent_output=input_data.get("agent_output", ""),
            query=input_data.get("query", ""),
            verbose=cfg.verbose
        )
    
    elif agent_name == "hunt_contradictions":
        return hunt_contradictions(
            cfg=cfg,
            outputs=input_data.get("outputs", {}),
            query=input_data.get("query", ""),
            verbose=cfg.verbose
        )
    
    elif agent_name == "detect_emergence":
        return detect_emergence(
            cfg=cfg,
            outputs=input_data.get("outputs", {}),
            query=input_data.get("query", ""),
            verbose=cfg.verbose
        )
    
    elif agent_name == "perform_meta_analysis":
        return perform_meta_analysis(
            cfg=cfg,
            outputs=input_data.get("outputs", {}),
            query=input_data.get("query", ""),
            verbose=cfg.verbose
        )
    
    elif agent_name == "apply_truth_seeking":
        return apply_truth_seeking_analysis(
            cfg=cfg,
            outputs=input_data.get("outputs", {}),
            query=input_data.get("query", ""),
            verbose=cfg.verbose
        )
    
    else:
        return f"Unknown deliberation analysis agent: {agent_name}"


# Legacy compatibility functions for direct calls
def run_legacy_protocol_sync(
    initial_query: str,
    fast: bool = False,
    hybrid: bool = False,
    verbose: bool = False,
    evidence_strict: bool = False,
    domains: list[str] | None = None,
    project_id: str | None = None,
    multimodal_input: Any = None,
    documents: list[str] | None = None,
    multimodal_evidence: list[Any] | None = None,
    force_molecular: bool = False,
    force_hypothesis_testing: bool = False,
    force_self_improvement: bool = False,
    celestial_biological: bool = False,
    universal_meta: bool = False,
    advanced_context: bool = False,
    observatory_integration: bool = False,
    context_window: int = 4096,
    processing_mode: str = "balanced",
    research_focus: str = "general",
    emergence_detection: bool = False,
    quality_strictness: str = "balanced",
) -> str:
    """
    Synchronous wrapper for the legacy protocol.
    Provides complete compatibility with the original iceberg_protocol function.
    """
    return iceberg_protocol(
        initial_query=initial_query,
        fast=fast,
        hybrid=hybrid,
        verbose=verbose,
        evidence_strict=evidence_strict,
        domains=domains,
        project_id=project_id,
        multimodal_input=multimodal_input,
        documents=documents,
        multimodal_evidence=multimodal_evidence,
        force_molecular=force_molecular,
        force_hypothesis_testing=force_hypothesis_testing,
        force_self_improvement=force_self_improvement,
        celestial_biological=celestial_biological,
        universal_meta=universal_meta,
        advanced_context=advanced_context,
        observatory_integration=observatory_integration,
        context_window=context_window,
        processing_mode=processing_mode,
        research_focus=research_focus,
        emergence_detection=emergence_detection,
        quality_strictness=quality_strictness,
    )


async def run_legacy_protocol_async(
    initial_query: str,
    fast: bool = False,
    hybrid: bool = False,
    verbose: bool = False,
    evidence_strict: bool = False,
    domains: list[str] | None = None,
    project_id: str | None = None,
    multimodal_input: Any = None,
    documents: list[str] | None = None,
    multimodal_evidence: list[Any] | None = None,
    force_molecular: bool = False,
    force_hypothesis_testing: bool = False,
    force_self_improvement: bool = False,
    celestial_biological: bool = False,
    universal_meta: bool = False,
    advanced_context: bool = False,
    observatory_integration: bool = False,
    context_window: int = 4096,
    processing_mode: str = "balanced",
    research_focus: str = "general",
    emergence_detection: bool = False,
    quality_strictness: str = "balanced",
) -> str:
    """
    Asynchronous wrapper for the legacy protocol.
    Provides async compatibility for the original iceberg_protocol function.
    """
    # Run the synchronous legacy protocol in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: iceberg_protocol(
            initial_query=initial_query,
            fast=fast,
            hybrid=hybrid,
            verbose=verbose,
            evidence_strict=evidence_strict,
            domains=domains,
            project_id=project_id,
            multimodal_input=multimodal_input,
            documents=documents,
            multimodal_evidence=multimodal_evidence,
            force_molecular=force_molecular,
            force_hypothesis_testing=force_hypothesis_testing,
            force_self_improvement=force_self_improvement,
            celestial_biological=celestial_biological,
            universal_meta=universal_meta,
            advanced_context=advanced_context,
            observatory_integration=observatory_integration,
            context_window=context_window,
            processing_mode=processing_mode,
            research_focus=research_focus,
            emergence_detection=emergence_detection,
            quality_strictness=quality_strictness,
        )
    )
    return result