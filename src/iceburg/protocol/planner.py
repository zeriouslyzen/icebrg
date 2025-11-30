from __future__ import annotations

from typing import List, Dict, Set, Optional
import logging

from .config import ProtocolConfig
from .models import AgentTask, Mode, Query
from iceburg.agents.capability_registry import get_registry, AgentCapability

logger = logging.getLogger(__name__)


_CORE_AGENT_SEQUENCE = [
    "vectorstore",
    "surveyor",
    "dissident",
    "archaeologist",
    "synthesist",
    "oracle",
]

_DELIBERATION_AGENTS = [
    "deliberation_pause",
    "hunt_contradictions",
    "detect_emergence",
    "perform_meta_analysis",
    "apply_truth_seeking",
]

_OPTIONAL_AGENTS = {
    "capability_gap": "capability_gap_detector",
    "scribe": "scribe",
    "architect": "architect",
    "supervisor": "supervisor",
}


def plan(query: Query, mode: Mode, config: ProtocolConfig) -> List[AgentTask]:
    tasks: List[AgentTask] = []

    # Phase 1: Multimodal Processing (if enabled)
    if config.enable_multimodal_processing:
        if config.verbose:
            print("[MULTIMODAL] Processing multimodal inputs...")
        
        tasks.append(
            AgentTask(
                agent="multimodal_processor",
                payload={
                    "query": query.text,
                    "multimodal_input": query.multimodal_input,
                    "documents": query.documents,
                    "multimodal_evidence": query.multimodal_evidence,
                    "requires": [],  # Can run independently
                },
            )
        )

    # Phase 2: Core Research Agents
    tasks.append(
        AgentTask(
            agent="vectorstore",
            payload={"query": query.text, "metadata": query.metadata},
        )
    )

    tasks.append(
        AgentTask(
            agent="surveyor",
            payload={"query": query.text, "metadata": query.metadata},
        )
    )

    # Deliberation pause after surveyor
    if not config.fast:
        tasks.append(
            AgentTask(
                agent="deliberation_pause",
                payload={
                    "agent_name": "surveyor",
                    "agent_output": "surveyor_output",  # Will be resolved by runner
                    "query": query.text,
                    "requires": ["surveyor"],
                },
            )
        )

    tasks.append(
        AgentTask(
            agent="dissident",
            payload={"query": query.text, "metadata": query.metadata, "requires": ["surveyor"]},
        )
    )

    # Deliberation pause after dissident
    if not config.fast:
        tasks.append(
            AgentTask(
                agent="deliberation_pause",
                payload={
                    "agent_name": "dissident",
                    "agent_output": "dissident_output",  # Will be resolved by runner
                    "query": query.text,
                    "requires": ["dissident"],
                },
            )
        )

    # Archaeologist for deep research
    tasks.append(
        AgentTask(
            agent="archaeologist",
            payload={
                "query": query.text,
                "documents": query.documents,
                "metadata": query.metadata,
                "requires": ["surveyor", "dissident"],
            },
        )
    )

    tasks.append(
        AgentTask(
            agent="synthesist",
            payload={
                "query": query.text,
                "metadata": query.metadata,
                "requires": ["surveyor", "dissident", "archaeologist"],
            },
        )
    )

    # Phase 2: CIM Stack Components (if enabled)
    if config.force_molecular:
        tasks.append(
            AgentTask(
                agent="molecular_synthesis",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident"],
                },
            )
        )

    if config.force_bioelectric:
        tasks.append(
            AgentTask(
                agent="bioelectric_integration",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident"],
                },
            )
        )

    if config.force_hypothesis_testing:
        tasks.append(
            AgentTask(
                agent="hypothesis_testing_laboratory",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident", "archaeologist"],
                },
            )
        )

    # Grounding layer always runs for scientific queries
    tasks.append(
        AgentTask(
            agent="grounding_layer_agent",
            payload={
                "query": query.text,
                "context": "previous_analysis",  # Will be resolved by runner
                "data_sources": ["scientific_literature", "empirical_data"],
                "correlation_types": ["causal", "correlational", "mechanistic"],
                "requires": ["surveyor", "dissident", "archaeologist"],
            },
        )
    )

    # Phase 4: AGI Capabilities (if enabled)
    agi_keywords = ['agi', 'self-redesign', 'novel intelligence', 'autonomous goal', 'unbounded learning', 'true agi', 'artificial general intelligence']
    should_activate_agi = any(keyword in query.text.lower() for keyword in agi_keywords)
    
    if should_activate_agi or config.force_agi:
        if config.verbose:
            print("[AGI] Activating True AGI Capabilities...")
        
        # Self-Redesign Engine - Fundamental Self-Modification
        tasks.append(
            AgentTask(
                agent="self_redesign_engine",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )
        
        # Novel Intelligence Creator - Invent New Intelligence Types
        tasks.append(
            AgentTask(
                agent="novel_intelligence_creator",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )
        
        # Autonomous Goal Formation - Form Own Goals
        tasks.append(
            AgentTask(
                agent="autonomous_goal_formation",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )
        
        # Unbounded Learning Engine - Learn Without Limits
        tasks.append(
            AgentTask(
                agent="unbounded_learning_engine",
                payload={
                    "query": query.text,
                    "context": "previous_analysis",  # Will be resolved by runner
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )

    # Phase 5: Deliberation Analysis (if not in fast mode)
    if not config.fast:
        # Collect all outputs for deliberation analysis
        all_outputs = {
            "surveyor": "surveyor_output",
            "dissident": "dissident_output", 
            "archaeologist": "archaeologist_output",
            "synthesist": "synthesist_output",
        }

        tasks.append(
            AgentTask(
                agent="hunt_contradictions",
                payload={
                    "outputs": all_outputs,
                    "query": query.text,
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )

        tasks.append(
            AgentTask(
                agent="detect_emergence",
                payload={
                    "outputs": all_outputs,
                    "query": query.text,
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )

        tasks.append(
            AgentTask(
                agent="perform_meta_analysis",
                payload={
                    "outputs": all_outputs,
                    "query": query.text,
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )

        tasks.append(
            AgentTask(
                agent="apply_truth_seeking",
                payload={
                    "outputs": all_outputs,
                    "query": query.text,
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist"],
                },
            )
        )

    tasks.append(
        AgentTask(
            agent="oracle",
            payload={
                "query": query.text,
                "metadata": query.metadata,
                "requires": ["synthesist"],
            },
        )
    )

    # Phase 6: Blockchain Verification and Immutable Storage (if enabled)
    if config.enable_blockchain_verification:
        if config.verbose:
            print("[BLOCKCHAIN] Activating blockchain verification...")
        
        # Blockchain verification - create immutable record
        tasks.append(
            AgentTask(
                agent="blockchain_verification",
                payload={
                    "query": query.text,
                    "research_content": "final_report",  # Will be resolved by runner
                    "metadata": {
                        "protocol_version": "4.0_modular",
                        "query_type": "research",
                        "timestamp": "current_timestamp"
                    },
                    "requires": ["oracle"],  # Run after oracle for final content
                },
            )
        )
        
        # Decentralized peer review
        tasks.append(
            AgentTask(
                agent="decentralized_peer_review",
                payload={
                    "query": query.text,
                    "research_content": "final_report",  # Will be resolved by runner
                    "blockchain_record_id": "blockchain_record_id",  # Will be resolved by runner
                    "requires": ["blockchain_verification"],
                },
            )
        )
        
        # Suppression-resistant storage
        tasks.append(
            AgentTask(
                agent="suppression_resistant_storage",
                payload={
                    "query": query.text,
                    "research_content": "final_report",  # Will be resolved by runner
                    "blockchain_record_id": "blockchain_record_id",  # Will be resolved by runner
                    "peer_review_id": "peer_review_id",  # Will be resolved by runner
                    "requires": ["blockchain_verification", "decentralized_peer_review"],
                },
            )
        )

    # Phase 6.5: Visual Generation (if enabled)
    if config.enable_visual_generation:
        if config.verbose:
            print("[VISUAL_GENERATOR] Generating visual content...")
        
        tasks.append(
            AgentTask(
                agent="visual_generator",
                payload={
                    "query": query.text,
                    "visual_requirements": {
                        "platform": "html5",
                        "responsive": True,
                        "accessibility": True
                    },
                    "platform": "html5",
                    "requires": ["oracle"],  # Run after oracle for final content
                },
            )
        )

    # Phase 7: Quality Control (if verbose mode)
    if config.verbose:
        stage_outputs = {
            "surveyor": "surveyor_output",
            "dissident": "dissident_output",
            "archaeologist": "archaeologist_output", 
            "synthesist": "synthesist_output",
            "oracle": "oracle_output",
        }
        
        tasks.append(
            AgentTask(
                agent="supervisor",
                payload={
                    "stage_outputs": stage_outputs,
                    "query": query.text,
                    "requires": ["surveyor", "dissident", "archaeologist", "synthesist", "oracle"],
                },
            )
        )

    return tasks


def optimize_plan(tasks: List[AgentTask], config: ProtocolConfig) -> List[AgentTask]:
    """
    Optimize task plan using capability registry for dependency resolution
    and parallelization opportunities.
    
    Uses topological sort to order tasks and groups parallelizable agents.
    """
    if not tasks:
        return tasks
    
    registry = get_registry()
    
    # Build dependency graph from tasks and registry
    dependency_graph: Dict[str, Set[str]] = {}
    agent_tasks: Dict[str, AgentTask] = {}
    
    for task in tasks:
        agent_id = task.agent
        agent_tasks[agent_id] = task
        
        # Get dependencies from task payload
        requires = set()
        if isinstance(task.payload, dict) and "requires" in task.payload:
            requires.update(task.payload.get("requires", []))
        if task.dependencies:
            requires.update(task.dependencies)
        
        # Get dependencies from registry
        agent_capability = registry.get_agent(agent_id)
        if agent_capability:
            requires.update(agent_capability.dependencies)
        
        dependency_graph[agent_id] = requires
    
    # Topological sort
    ordered_agents = _topological_sort(dependency_graph)
    
    # Reorder tasks based on topological sort
    ordered_tasks = []
    for agent_id in ordered_agents:
        if agent_id in agent_tasks:
            ordered_tasks.append(agent_tasks[agent_id])
    
    # Add any tasks not in dependency graph (no dependencies)
    for task in tasks:
        if task.agent not in ordered_agents:
            ordered_tasks.append(task)
    
    if config.verbose:
        logger.info(f"Optimized plan: {len(ordered_tasks)} tasks ordered with dependencies resolved")
    
    return ordered_tasks


def get_parallelizable_groups(tasks: List[AgentTask], config: ProtocolConfig) -> List[List[AgentTask]]:
    """
    Group tasks into parallelizable execution groups.
    
    Returns list of groups, where tasks in each group can run in parallel.
    """
    if not tasks:
        return []
    
    registry = get_registry()
    
    # Build dependency graph
    dependency_graph: Dict[str, Set[str]] = {}
    agent_tasks: Dict[str, AgentTask] = {}
    
    for task in tasks:
        agent_id = task.agent
        agent_tasks[agent_id] = task
        
        requires = set()
        if isinstance(task.payload, dict) and "requires" in task.payload:
            requires.update(task.payload.get("requires", []))
        if task.dependencies:
            requires.update(task.dependencies)
        
        agent_capability = registry.get_agent(agent_id)
        if agent_capability:
            requires.update(agent_capability.dependencies)
        
        dependency_graph[agent_id] = requires
    
    # Group by dependency level
    groups: List[List[AgentTask]] = []
    completed: Set[str] = set()
    
    while len(completed) < len(agent_tasks):
        # Find tasks ready to execute (all dependencies completed)
        ready = []
        for agent_id, task in agent_tasks.items():
            if agent_id in completed:
                continue
            
            deps = dependency_graph.get(agent_id, set())
            if deps.issubset(completed):
                # Check if agent is parallelizable
                agent_capability = registry.get_agent(agent_id)
                if agent_capability and agent_capability.parallelizable:
                    ready.append(task)
                elif not agent_capability:
                    # If not in registry, assume parallelizable
                    ready.append(task)
        
        if not ready:
            # No tasks ready - might be circular dependency or missing agent
            remaining = [aid for aid in agent_tasks.keys() if aid not in completed]
            if remaining:
                logger.warning(f"Could not resolve dependencies for: {remaining}")
                # Force add remaining tasks
                for agent_id in remaining:
                    if agent_id in agent_tasks:
                        ready.append(agent_tasks[agent_id])
        
        if ready:
            groups.append(ready)
            completed.update([task.agent for task in ready])
        else:
            break
    
    if config.verbose:
        logger.info(f"Grouped {len(tasks)} tasks into {len(groups)} parallelizable groups")
    
    return groups


def _topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    """
    Perform topological sort on dependency graph.
    
    Returns ordered list of nodes respecting dependencies.
    """
    ordered = []
    visited = set()
    temp_visited = set()
    
    def visit(node: str):
        if node in temp_visited:
            # Circular dependency detected
            logger.warning(f"Circular dependency detected involving {node}")
            return
        if node in visited:
            return
        
        temp_visited.add(node)
        
        # Visit dependencies first
        for dep in graph.get(node, set()):
            if dep in graph:  # Only visit if in graph
                visit(dep)
        
        temp_visited.remove(node)
        visited.add(node)
        ordered.append(node)
    
    for node in graph:
        if node not in visited:
            visit(node)
    
    return ordered
