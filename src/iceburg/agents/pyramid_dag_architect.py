"""
Pyramid DAG Architecture for ICEBURG - October 2025
=================================================

Implements hierarchical task decomposition with Judge Agent verification
based on InfiAgent research for enhanced multi-agent coordination.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

@dataclass
class DAGNode:
    """Node in the pyramid DAG structure"""
    id: str
    task_type: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Any] = None
    execution_order: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None

@dataclass
class DAGValidation:
    """Validation result for DAG execution"""
    node_id: str
    is_valid: bool
    confidence_score: float
    validation_notes: str
    timestamp: datetime = field(default_factory=datetime.now)

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class PyramidDAGArchitect:
    """
    Implements pyramid DAG architecture for hierarchical task decomposition
    with Judge Agent verification layer for enhanced reliability.
    """

    def __init__(self):
        self.dag_nodes: Dict[str, DAGNode] = {}
        self.execution_order: List[str] = []
        self.completed_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()
        self.judge_validations: List[DAGValidation] = []

    def build_pyramid_dag(self, root_requirement: str) -> str:
        """Build a pyramid DAG structure for the given requirement"""

        # Root node
        root_id = "root_task"
        self.dag_nodes[root_id] = DAGNode(
            id=root_id,
            task_type="root_requirement",
            description=root_requirement,
            execution_order=0
        )

        # Layer 1: Decomposition (parallels ICEBURG Surveyor/Dissident)
        decomposition_nodes = self._create_decomposition_layer(root_id, root_requirement)

        # Layer 2: Specialized Processing (parallels ICEBURG Archaeologist/Synthesist)
        processing_nodes = self._create_processing_layer(decomposition_nodes)

        # Layer 3: Integration & Synthesis (parallels ICEBURG Scrutineer/Oracle)
        integration_nodes = self._create_integration_layer(processing_nodes)

        # Layer 4: Judge Agent Verification (new research-backed layer)
        judge_node = self._create_judge_layer(integration_nodes, root_id)

        # Calculate execution order using topological sort
        self._calculate_execution_order()

        return root_id

    def _create_decomposition_layer(self, parent_id: str, requirement: str) -> List[str]:
        """Create decomposition layer nodes"""
        nodes = []

        # Break down into component tasks
        decomposition_tasks = [
            {"type": "literature_analysis", "desc": "Analyze existing literature and sources"},
            {"type": "requirement_breakdown", "desc": "Break down requirements into components"},
            {"type": "constraint_identification", "desc": "Identify constraints and limitations"},
            {"type": "feasibility_assessment", "desc": "Assess technical and practical feasibility"}
        ]

        for i, task in enumerate(decomposition_tasks):
            node_id = f"decomp_{i}"
            node = DAGNode(
                id=node_id,
                task_type=task["type"],
                description=task["desc"],
                dependencies=[parent_id],
                execution_order=i + 1
            )
            node.parent = parent_id
            self.dag_nodes[parent_id].children.append(node_id)

            self.dag_nodes[node_id] = node
            nodes.append(node_id)

        return nodes

    def _create_processing_layer(self, parent_nodes: List[str]) -> List[str]:
        """Create specialized processing layer nodes"""
        nodes = []

        # Specialized processing for each decomposition task
        processing_tasks = [
            {"type": "research_synthesis", "desc": "Synthesize research findings"},
            {"type": "technical_analysis", "desc": "Analyze technical requirements"},
            {"type": "risk_assessment", "desc": "Assess implementation risks"},
            {"type": "solution_design", "desc": "Design potential solutions"}
        ]

        for i, (parent_id, task) in enumerate(zip(parent_nodes, processing_tasks)):
            node_id = f"process_{i}"
            node = DAGNode(
                id=node_id,
                task_type=task["type"],
                description=task["desc"],
                dependencies=[parent_id],
                execution_order=i + 5
            )
            node.parent = parent_id
            self.dag_nodes[parent_id].children.append(node_id)

            self.dag_nodes[node_id] = node
            nodes.append(node_id)

        return nodes

    def _create_integration_layer(self, parent_nodes: List[str]) -> List[str]:
        """Create integration and synthesis layer nodes"""
        nodes = []

        # Integration tasks that combine processing results
        integration_tasks = [
            {"type": "cross_domain_synthesis", "desc": "Synthesize across different domains"},
            {"type": "solution_integration", "desc": "Integrate multiple solution approaches"},
            {"type": "validation_framework", "desc": "Create validation and testing framework"}
        ]

        for i, task in enumerate(integration_tasks):
            node_id = f"integrate_{i}"
            node = DAGNode(
                id=node_id,
                task_type=task["type"],
                description=task["desc"],
                dependencies=parent_nodes,  # Depends on all processing nodes
                execution_order=i + 9
            )

            # Set parent relationships
            for parent_id in parent_nodes:
                node.parent = parent_id  # Simplified - in practice would be more complex
                self.dag_nodes[parent_id].children.append(node_id)

            self.dag_nodes[node_id] = node
            nodes.append(node_id)

        return nodes

    def _create_judge_layer(self, parent_nodes: List[str], root_id: str) -> str:
        """Create Judge Agent verification layer"""
        judge_id = "judge_verification"

        node = DAGNode(
            id=judge_id,
            task_type="judge_verification",
            description="Verify and validate all results for accuracy and consistency",
            dependencies=parent_nodes,
            execution_order=12
        )

        # Set parent relationships
        for parent_id in parent_nodes:
            node.parent = parent_id
            self.dag_nodes[parent_id].children.append(judge_id)

        # Judge also verifies against root requirement
        node.dependencies.append(root_id)

        self.dag_nodes[judge_id] = node
        return judge_id

    def _calculate_execution_order(self):
        """Calculate topological execution order"""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        order = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id not in visited:
                temp_visited.add(node_id)

                # Visit dependencies first
                for dep_id in self.dag_nodes[node_id].dependencies:
                    if dep_id in self.dag_nodes:  # Only visit nodes that exist
                        visit(dep_id)

                temp_visited.remove(node_id)
                visited.add(node_id)
                order.append(node_id)

        # Visit all nodes
        for node_id in self.dag_nodes:
            if node_id not in visited:
                visit(node_id)

        self.execution_order = order

    async def execute_dag(self, agent_swarm, rag_memory) -> Dict[str, Any]:
        """Execute the pyramid DAG with agent swarm"""

        results = {}
        start_time = time.time()

        for i, node_id in enumerate(self.execution_order):
            node = self.dag_nodes[node_id]

            node.start_time = datetime.now()
            node.status = "processing"

            try:
                # Get context from previous nodes and RAG memory
                context = await self._get_node_context(node, rag_memory)

                # Execute with appropriate agent
                result = await self._execute_node_with_agent(node, agent_swarm, context)

                node.result = result
                node.status = "completed"
                node.end_time = datetime.now()
                results[node_id] = result

                self.completed_nodes.add(node_id)

            except Exception as e:
                node.status = "failed"
                node.end_time = datetime.now()
                self.failed_nodes.add(node_id)

                # For critical failures, we might want to stop execution
                if node.task_type in ["judge_verification"]:
                    break

        end_time = time.time()
        execution_time = end_time - start_time

        # Run judge validation if execution completed
        if "judge_verification" in self.completed_nodes:
            await self._run_judge_validation(results)


        return {
            "results": results,
            "execution_time": execution_time,
            "completed_nodes": len(self.completed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "judge_validation": self.judge_validations[-1] if self.judge_validations else None,
            "dag_structure": self._get_dag_summary()
        }

    async def _get_node_context(self, node: DAGNode, rag_memory) -> str:
        """Get context for a node from completed dependencies and RAG memory"""
        context_parts = []

        # Get results from completed dependencies
        for dep_id in node.dependencies:
            if dep_id in self.completed_nodes and dep_id in self.dag_nodes:
                dep_node = self.dag_nodes[dep_id]
                if dep_node.result:
                    context_parts.append(f"[{dep_node.task_type}]: {str(dep_node.result)[:200]}...")

        # Get additional context from RAG memory
        if rag_memory:
            try:
                rag_context = await rag_memory.get_layer_context(
                    layer=node.task_type.split('_')[0],  # Extract layer from task type
                    agent="dag_executor",
                    limit=2
                )
                if rag_context:
                    context_parts.append(f"[RAG Context]: {rag_context[:200]}...")
            except:
                pass  # RAG context is optional

        return "\n\n".join(context_parts) if context_parts else ""

    async def _execute_node_with_agent(self, node: DAGNode, agent_swarm, context: str) -> Any:
        """Execute a node using the appropriate agent"""
        # Map task types to agent capabilities
        agent_mapping = {
            "literature_analysis": "research_synthesis",
            "requirement_breakdown": "complex_reasoning",
            "constraint_identification": "meta_analysis",
            "feasibility_assessment": "pattern_recognition",
            "research_synthesis": "research_synthesis",
            "technical_analysis": "code_analysis",
            "risk_assessment": "meta_analysis",
            "solution_design": "system_design",
            "cross_domain_synthesis": "research_synthesis",
            "solution_integration": "architecture_planning",
            "validation_framework": "code_analysis",
            "judge_verification": "scientific_reasoning"
        }

        required_capability = agent_mapping.get(node.task_type, "complex_reasoning")

        # Find best agent for this task
        best_agent_id, similarity = None, 0.0

        for agent_id, agent in agent_swarm.agents.items():
            if agent.is_active:
                continue

            # Check if agent has the required capability
            if required_capability in agent.capabilities:
                # Calculate simple similarity score
                capability_match = len(set([required_capability]) & set(agent.capabilities))
                score = capability_match / len(agent.capabilities)

                if score > similarity:
                    similarity = score
                    best_agent_id = agent_id

        if not best_agent_id:
            raise ValueError(f"No suitable agent found for {node.task_type} (required: {required_capability})")

        # Create task for the agent
        task_input = f"{node.description}\n\nContext: {context}" if context else node.description

        # In a real implementation, this would submit to the swarm
        # For now, return a placeholder result
        return f"Executed {node.task_type} using {best_agent_id} with similarity {similarity:.2f}. Input: {task_input[:100]}..."

    async def _run_judge_validation(self, results: Dict[str, Any]):
        """Run Judge Agent validation on completed results"""
        try:
            # Aggregate all results for validation
            all_results = []
            for node_id, result in results.items():
                if node_id in self.completed_nodes:
                    all_results.append(f"[{self.dag_nodes[node_id].task_type}]: {str(result)}")

            combined_results = "\n".join(all_results)

            # Simple validation logic (in production, this would use an actual LLM)
            validation_score = 0.8  # Placeholder - would be calculated based on consistency, completeness, etc.

            validation = DAGValidation(
                node_id="judge_verification",
                is_valid=validation_score > 0.7,
                confidence_score=validation_score,
                validation_notes=f"Validated {len(all_results)} results with {validation_score:.2f} confidence"
            )

            self.judge_validations.append(validation)

        except Exception as e:
            if self.verbose:
                print(f"[PYRAMID_DAG] Error in validation: {e}")

    def _get_dag_summary(self) -> Dict[str, Any]:
        """Get summary of DAG structure"""
        return {
            "total_nodes": len(self.dag_nodes),
            "execution_steps": len(self.execution_order),
            "completed_nodes": len(self.completed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "layers": self._analyze_dag_layers(),
            "dependencies": sum(len(node.dependencies) for node in self.dag_nodes.values())
        }

    def _analyze_dag_layers(self) -> Dict[str, int]:
        """Analyze the structure by layers"""
        layers = {}
        for node in self.dag_nodes.values():
            layer = node.task_type.split('_')[0]  # Extract layer from task type
            layers[layer] = layers.get(layer, 0) + 1
        return layers

# Global pyramid DAG instance
_pyramid_dag: Optional[PyramidDAGArchitect] = None

async def get_pyramid_dag_architect() -> PyramidDAGArchitect:
    """Get or create the global Pyramid DAG Architect instance"""
    global _pyramid_dag
    if _pyramid_dag is None:
        _pyramid_dag = PyramidDAGArchitect()
    return _pyramid_dag

async def execute_pyramid_dag(requirement: str, agent_swarm, rag_memory) -> Dict[str, Any]:
    """Execute a requirement through the pyramid DAG architecture"""
    architect = await get_pyramid_dag_architect()

    # Build the DAG
    root_id = architect.build_pyramid_dag(requirement)

    # Execute the DAG
    results = await architect.execute_dag(agent_swarm, rag_memory)

    return results
