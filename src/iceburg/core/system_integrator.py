"""
System Integrator
Integrates all ICEBURG systems
"""

from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
import logging
import time
from ..integration.blackboard_integration import BlackboardIntegration
from ..integration.curiosity_integration import CuriosityIntegration
from ..integration.swarming_integration import SwarmingIntegration
from ..truth.suppression_detector import SuppressionDetector
from ..curiosity.curiosity_engine import CuriosityEngine
from ..global_workspace import GlobalWorkspace
from ..micro_agent_swarm import MicroAgentSwarm
from ..generation.device_generator import DeviceGenerator
from ..research.methodology_analyzer import MethodologyAnalyzer
from ..tracking.source_citation_tracker import SourceCitationTracker
from ..tracking.copyright_vault import CopyrightVault
from ..tracking.emergent_intelligence_tracker import EmergentIntelligenceTracker
from ..utils.user_friendly_names import (
    format_thinking_message,
    format_action_message
)
from ..infrastructure.retry_manager import RetryManager, RetryConfig
from ..distributed.load_balancer import IntelligentLoadBalancer, LoadBalancingStrategy
from ..infrastructure.dynamic_resource_allocator import get_resource_allocator

logger = logging.getLogger(__name__)


class SystemIntegrator:
    """Integrates all ICEBURG systems"""
    
    def __init__(self):
        self.blackboard_integration = BlackboardIntegration()
        self.curiosity_integration = CuriosityIntegration()
        self.swarming_integration = SwarmingIntegration()
        self.suppression_detector = SuppressionDetector()
        self.curiosity_engine = CuriosityEngine()
        self.global_workspace = GlobalWorkspace()
        self.micro_swarm = MicroAgentSwarm()
        self.device_generator = DeviceGenerator()
        self.methodology_analyzer = MethodologyAnalyzer()
        self.source_citation_tracker = SourceCitationTracker()
        self.copyright_vault = CopyrightVault()
        self.emergent_intelligence_tracker = EmergentIntelligenceTracker()
        self.integrated = False
        
        # Initialize retry manager with circuit breakers for agent execution
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=300.0  # 5 minutes
        )
        self.retry_manager = RetryManager(retry_config)
        
        # Initialize load balancer for agent distribution
        self.load_balancer = IntelligentLoadBalancer(
            strategy=LoadBalancingStrategy.ADAPTIVE
        )
        
        # Initialize dynamic resource allocator
        self.resource_allocator = get_resource_allocator()
        
        # Register agents as workers in load balancer
        self._register_agents_as_workers()
    
    def integrate_all_systems(self) -> Dict[str, Any]:
        """Integrate all ICEBURG systems"""
        integration = {
            "blackboard": True,
            "curiosity": True,
            "swarming": True,
            "truth_finding": True,
            "device_generation": True,
            "research_methodology": True,
            "integrated_at": datetime.now().isoformat()
        }
        
        # Verify integrations
        integration["blackboard_status"] = self.blackboard_integration.get_blackboard_status()
        integration["curiosity_status"] = self.curiosity_integration.get_curiosity_status()
        integration["swarm_capabilities"] = self.swarming_integration.get_swarm_capabilities()
        
        self.integrated = True
        
        return integration
    
    def _register_agents_as_workers(self):
        """Register agents as workers in load balancer"""
        from ..agents.capability_registry import get_registry
        registry = get_registry()
        
        # Register each agent as a worker
        for agent_id, agent_capability in registry.get_all_agents().items():
            self.load_balancer.add_worker(
                worker_id=agent_id,
                weight=1.0 / max(1, agent_capability.complexity_level.value),  # Lower complexity = higher weight
                capabilities=agent_capability.capabilities,
                metadata={
                    "agent_name": agent_capability.agent_name,
                    "agent_type": agent_capability.agent_type.value,
                    "complexity": agent_capability.complexity_level.value,
                    "speed": agent_capability.speed_rating.value,
                    "memory_mb": agent_capability.memory_mb,
                    "cpu_cores": agent_capability.cpu_cores
                }
            )
        
        logger.info(f"Registered {len(registry.get_all_agents())} agents as workers in load balancer")
    
    async def process_query_with_full_integration(
        self,
        query: str,
        domain: Optional[str] = None,
        custom_config: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process query with full system integration - executes all agents in sequence"""
        try:
            from ..config import load_config
            from ..vectorstore import VectorStore
            from ..graph_store import KnowledgeGraph
            from ..agents.surveyor import run as surveyor_run
            from ..agents.dissident import run as dissident_run
            from ..agents.archaeologist import run as archaeologist_run
            from ..agents.synthesist import run as synthesist_run
            from ..agents.oracle import run as oracle_run
            from ..agents.scribe import run as scribe_run
            from ..agents.weaver import run as weaver_run
            from ..agents.scrutineer import run as scrutineer_run
            from ..agents.supervisor import run as supervisor_run
            
            # Use custom config if provided (for model selection from frontend)
            cfg = custom_config if custom_config else load_config()
            
            # Initialize stores with error handling
            try:
                vs = VectorStore(cfg)
            except Exception as e:
                logger.warning(f"Error initializing VectorStore: {e}, continuing without vector store")
                vs = None
            
            try:
                kg = KnowledgeGraph(cfg)
            except Exception as e:
                logger.warning(f"Error initializing KnowledgeGraph: {e}, continuing without knowledge graph")
                kg = None
        except Exception as e:
            logger.error(f"Error importing modules or initializing stores: {e}", exc_info=True)
            # Return fallback result
            return {
                "query": query,
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
                "methodology": "error_fallback",
                "results": {
                    "content": f"Error initializing system: {str(e)}. Please check system configuration.",
                    "agent_results": {},
                    "engines_used": [],
                    "algorithms_used": [],
                    "error": str(e)
                }
            }
        
        result = {
            "query": query,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "methodology": "enhanced_deliberation",
            "results": {
                "agent_results": {},
                "engines_used": [],
                "algorithms_used": []
            }
        }
        
        # Step 1: Apply Enhanced Deliberation methodology
        try:
            methodology = self.methodology_analyzer.apply_methodology(query, domain)
            result["results"]["methodology"] = methodology
        except Exception as e:
            logger.error(f"Error applying methodology: {e}", exc_info=True)
            result["results"]["methodology"] = {
                "methodology": "error_fallback",
                "steps": [{"name": "Error applying methodology", "status": "error"}]
            }
        
        # Track engines and algorithms
        result["results"]["engines_used"].append({
            "engine": "MethodologyAnalyzer",
            "algorithm": "Enhanced Deliberation",
            "step": "methodology_application",
            "description": "Applying Enhanced Deliberation methodology with 6-step process"
        })
        
        # Send initial action update
        if progress_callback:
            await progress_callback({
                "type": "action",
                "action": "system_integration",
                "status": "processing",
                "description": "Initializing agent pipeline..."
            })
        
        # Step 2: Execute core agents (parallel if possible, otherwise sequential)
        # Check if we should use parallel execution (for complex queries)
        use_parallel = not cfg.fast and domain is None  # Use parallel for non-fast, non-domain-specific queries
        
        if use_parallel:
            # Try parallel execution for independent agents
            try:
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # Execute independent agents in parallel using thread pool
                # (agents are synchronous, so we use threads for parallelization)
                executor = ThreadPoolExecutor(max_workers=2)
                
                # Submit both agents to thread pool
                surveyor_future = executor.submit(surveyor_run, cfg, vs, query, True)
                archaeologist_future = executor.submit(archaeologist_run, cfg, query, None, True)
                
                # Wait for both to complete
                surveyor_result = surveyor_future.result(timeout=300)
                archaeologist_result = archaeologist_future.result(timeout=300)
                
                # Store results
                result["results"]["agent_results"]["surveyor"] = surveyor_result
                result["results"]["agent_results"]["archaeologist"] = archaeologist_result
                
                result["results"]["algorithms_used"].append({
                    "algorithm": "Parallel Execution",
                    "method": "Concurrent Agent Processing",
                    "step": 2
                })
                
                executor.shutdown(wait=False)
            except Exception as e:
                # Fallback to sequential execution
                logger.warning(f"Parallel execution failed, using sequential: {e}")
                use_parallel = False
        
        if not use_parallel:
            # Sequential execution (original code)
            # Agent 1: Surveyor
            try:
                if progress_callback:
                    action_desc = format_action_message(action="surveyor_analysis", mode=None)
                    await progress_callback({
                        "type": "action",
                        "action": "surveyor",
                        "status": "processing",
                        "description": action_desc
                    })
                    thinking_msg = format_thinking_message(agent="surveyor", mode=None)
                    await progress_callback({
                        "type": "agent_thinking",
                        "agent": "surveyor",
                        "content": thinking_msg
                    })
                engine_info = {
                    "engine": "VectorStore",
                    "algorithm": "Semantic Search",
                    "step": "surveyor_analysis",
                    "description": "Searching knowledge base with semantic similarity"
                }
                result["results"]["engines_used"].append(engine_info)
                # Send engine update during processing
                if progress_callback:
                    await progress_callback({
                        "type": "engines",
                        "engines": [engine_info]
                    })
                # Select worker using load balancer
                selected_worker = await self.load_balancer.select_worker(
                    query=query,
                    context={"agent": "surveyor", "mode": domain or "default"},
                    required_capabilities=["information_gathering", "research_synthesis"]
                )
                # Note: load_balancer.select_worker returns just the worker_id, not a tuple
                
                # Execute with circuit breaker, retry, and resource allocation
                async def run_surveyor():
                    # Allocate resources dynamically
                    allocation = await self.resource_allocator.allocate_resources(
                        agent_id="surveyor",
                        priority=8  # High priority for core agent
                    )
                    
                    if not allocation:
                        logger.warning("Resource allocation failed for surveyor, proceeding anyway")
                    
                    try:
                        result = surveyor_run(cfg, vs, query, verbose=True)
                        return result
                    finally:
                        if allocation:
                            # Release resources
                            allocation_id = f"surveyor_{int(time.time() * 1000)}"
                            self.resource_allocator.release_resources(allocation_id)
                
                retry_result = await self.retry_manager.execute_with_retry(
                    run_surveyor,
                    operation_name="agent_surveyor"
                )
                
                if retry_result.success:
                    surveyor_result = retry_result.result
                    # Record success in load balancer
                    self.load_balancer.record_request_result(
                        worker_id="surveyor",
                        success=True,
                        response_time=retry_result.total_time
                    )
                else:
                    surveyor_result = f"Error: {retry_result.final_error}"
                    logger.error(f"Surveyor failed after {retry_result.attempts} attempts: {retry_result.final_error}")
                    # Record failure in load balancer
                    self.load_balancer.record_request_result(
                        worker_id="surveyor",
                        success=False,
                        response_time=retry_result.total_time,
                        error=retry_result.final_error
                    )
                
                result["results"]["agent_results"]["surveyor"] = surveyor_result
                if progress_callback:
                    action_desc = format_action_message(action="surveyor_analysis", description="Information gathering complete", mode=None)
                    await progress_callback({
                        "type": "action",
                        "action": "surveyor",
                        "status": "complete",
                        "description": action_desc
                    })
                algo_info = {
                    "algorithm": "Surveyor Agent",
                    "method": "Semantic Search",
                    "step": 1
                }
                result["results"]["algorithms_used"].append(algo_info)
                # Send algorithm update during processing
                if progress_callback:
                    await progress_callback({
                        "type": "algorithms",
                        "algorithms": [algo_info]
                    })
            except Exception as e:
                result["results"]["agent_results"]["surveyor"] = f"Error: {e}"
            
            # Agent 3: Archaeologist (independent, can run in parallel with Surveyor)
            try:
                if progress_callback:
                    action_desc = format_action_message(action="archaeologist_analysis", mode=None)
                    await progress_callback({
                        "type": "action",
                        "action": "archaeologist",
                        "status": "processing",
                        "description": action_desc
                    })
                    thinking_msg = format_thinking_message(agent="archaeologist", mode=None)
                    await progress_callback({
                        "type": "agent_thinking",
                        "agent": "archaeologist",
                        "content": thinking_msg
                    })
                archaeologist_result = archaeologist_run(cfg, query, documents=None, verbose=True)
                result["results"]["agent_results"]["archaeologist"] = archaeologist_result
                if progress_callback:
                    action_desc = format_action_message(action="archaeologist_analysis", description="Historical research complete", mode=None)
                    await progress_callback({
                        "type": "action",
                        "action": "archaeologist",
                        "status": "complete",
                        "description": action_desc
                    })
            except Exception as e:
                result["results"]["agent_results"]["archaeologist"] = f"Error: {e}"
        
        # Agent 2: Dissident (needs surveyor output - must be sequential)
        try:
            if progress_callback:
                action_desc = format_action_message(action="dissident_analysis", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "dissident",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="dissident", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "dissident",
                    "content": thinking_msg
                })
            surveyor_output = result["results"]["agent_results"].get("surveyor", "")
            dissident_result = dissident_run(cfg, query, surveyor_output, verbose=True)
            result["results"]["agent_results"]["dissident"] = dissident_result
            if progress_callback:
                action_desc = format_action_message(action="dissident_analysis", description="Alternative analysis complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "dissident",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["dissident"] = f"Error: {e}"
        
        # Agent 4: Synthesist (needs surveyor and dissident - must be sequential)
        try:
            if progress_callback:
                action_desc = format_action_message(action="synthesist_analysis", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "synthesist",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="synthesist", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "synthesist",
                    "content": thinking_msg
                })
            surveyor_output = result["results"]["agent_results"].get("surveyor", "")
            dissident_output = result["results"]["agent_results"].get("dissident", "")
            enhanced_context = {
                "surveyor": surveyor_output,
                "dissident": dissident_output
            }
            synthesist_result = synthesist_run(cfg, enhanced_context, verbose=True)
            result["results"]["agent_results"]["synthesist"] = synthesist_result
            if progress_callback:
                action_desc = format_action_message(action="synthesist_analysis", description="Synthesis complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "synthesist",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["synthesist"] = f"Error: {e}"
        
        # Agent 5: Oracle (needs synthesist output)
        try:
            if progress_callback:
                action_desc = format_action_message(action="oracle_analysis", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "oracle",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="oracle", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "oracle",
                    "content": thinking_msg
                })
            synthesis_output = result["results"]["agent_results"].get("synthesist", "")
            oracle_result = oracle_run(cfg, kg, vs, synthesis_output, verbose=True)
            result["results"]["agent_results"]["oracle"] = oracle_result
            if progress_callback:
                action_desc = format_action_message(action="oracle_analysis", description="Evidence synthesis complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "oracle",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["oracle"] = f"Error: {e}"
        
        # Agent 6: Scrutineer (needs synthesist output)
        try:
            if progress_callback:
                action_desc = format_action_message(action="scrutineer_analysis", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "scrutineer",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="scrutineer", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "scrutineer",
                    "content": thinking_msg
                })
            synthesis_output = result["results"]["agent_results"].get("synthesist", "")
            scrutineer_result = scrutineer_run(cfg, synthesis_output, verbose=True)
            result["results"]["agent_results"]["scrutineer"] = scrutineer_result
            if progress_callback:
                action_desc = format_action_message(action="scrutineer_analysis", description="Contradiction check complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "scrutineer",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["scrutineer"] = f"Error: {e}"
        
        # Agent 7: Scribe (needs oracle output)
        try:
            if progress_callback:
                action_desc = format_action_message(action="scribe", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "scribe",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="scribe", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "scribe",
                    "content": thinking_msg
                })
            oracle_output = result["results"]["agent_results"].get("oracle", "")
            scribe_result = scribe_run(cfg, oracle_output, verbose=True)
            result["results"]["agent_results"]["scribe"] = scribe_result
            if progress_callback:
                action_desc = format_action_message(action="scribe", description="Knowledge documentation complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "scribe",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["scribe"] = f"Error: {e}"
        
        # Agent 8: Weaver (needs oracle output)
        try:
            if progress_callback:
                action_desc = format_action_message(action="weaver", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "weaver",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="weaver", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "weaver",
                    "content": thinking_msg
                })
            oracle_output = result["results"]["agent_results"].get("oracle", "")
            weaver_result = weaver_run(cfg, oracle_output, verbose=True)
            result["results"]["agent_results"]["weaver"] = weaver_result
            if progress_callback:
                action_desc = format_action_message(action="weaver", description="Code generation complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "weaver",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["weaver"] = f"Error: {e}"
        
        # Agent 9: Supervisor (validates all outputs)
        try:
            if progress_callback:
                action_desc = format_action_message(action="supervisor_validation", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "supervisor",
                    "status": "processing",
                    "description": action_desc
                })
                thinking_msg = format_thinking_message(agent="supervisor", mode=None)
                await progress_callback({
                    "type": "agent_thinking",
                    "agent": "supervisor",
                    "content": thinking_msg
                })
            stage_outputs = {
                "query": query,
                "surveyor": result["results"]["agent_results"].get("surveyor", ""),
                "dissident": result["results"]["agent_results"].get("dissident", ""),
                "synthesist": result["results"]["agent_results"].get("synthesist", ""),
                "oracle": result["results"]["agent_results"].get("oracle", ""),
                "scrutineer": result["results"]["agent_results"].get("scrutineer", "")
            }
            supervisor_result = supervisor_run(cfg, stage_outputs, query=query, verbose=True)
            result["results"]["agent_results"]["supervisor"] = supervisor_result
            if progress_callback:
                action_desc = format_action_message(action="supervisor_validation", description="Validation complete", mode=None)
                await progress_callback({
                    "type": "action",
                    "action": "supervisor",
                    "status": "complete",
                    "description": action_desc
                })
        except Exception as e:
            result["results"]["agent_results"]["supervisor"] = f"Error: {e}"
        
        # Step 3: Generate curiosity-driven queries
        try:
            if progress_callback:
                await progress_callback({
                    "type": "action",
                    "action": "curiosity_engine",
                    "status": "processing",
                    "description": "Curiosity Engine: Generating research queries..."
                })
                await progress_callback({
                    "type": "thinking",
                    "content": "Generating curiosity-driven research queries"
                })
            engine_info = {
                "engine": "CuriosityEngine",
                "algorithm": "Autonomous Query Generation",
                "step": "curiosity_driven_research",
                "description": "Generating curiosity-driven research queries"
            }
            result["results"]["engines_used"].append(engine_info)
            # Send engine update during processing
            if progress_callback:
                await progress_callback({
                    "type": "engines",
                    "engines": [engine_info]
                })
            if hasattr(self.curiosity_engine, 'generate_queries'):
                curiosity_queries = self.curiosity_engine.generate_queries(domain=domain, limit=3)
            elif hasattr(self.curiosity_engine, 'generate_curiosity_query'):
                curiosity_queries = [self.curiosity_engine.generate_curiosity_query(domain=domain)]
            else:
                # Fallback: create simple curiosity queries
                curiosity_queries = [
                    f"What suppressed knowledge exists about {domain or 'this topic'}?",
                    f"How can Enhanced Deliberation reveal hidden patterns in {domain or 'this domain'}?",
                    f"What contradictions exist in narratives about {domain or 'this topic'}?"
                ]
            result["results"]["curiosity_queries"] = curiosity_queries
            result["results"]["algorithms_used"].append({
                "algorithm": "Curiosity-Driven Query Generation",
                "method": "Autonomous Research Query Generation",
                "step": 3
            })
        except Exception as e:
            # Fallback curiosity queries
            result["results"]["curiosity_queries"] = [
                f"What suppressed knowledge exists about {domain or 'this topic'}?",
                f"How can Enhanced Deliberation reveal hidden patterns?",
                f"What contradictions exist in narratives?"
            ]
        
        # Step 4: Create truth-finding swarm
        try:
            if progress_callback:
                await progress_callback({
                    "type": "action",
                    "action": "swarm",
                    "status": "processing",
                    "description": "Swarm: Creating micro-agent swarm..."
                })
                await progress_callback({
                    "type": "thinking",
                    "content": "Coordinating micro-agent swarm for parallel processing"
                })
            engine_info = {
                "engine": "MicroAgentSwarm",
                "algorithm": "Swarm Coordination",
                "step": "swarm_creation",
                "description": "Creating micro-agent swarm for parallel processing"
            }
            result["results"]["engines_used"].append(engine_info)
            # Send engine update during processing
            if progress_callback:
                await progress_callback({
                    "type": "engines",
                    "engines": [engine_info]
                })
            swarm = await self.swarming_integration.create_truth_finding_swarm(
                query,
                swarm_type="research_swarm"
            )
            result["results"]["swarm"] = swarm
            
            # Step 5: Execute swarm
            if progress_callback:
                await progress_callback({
                    "type": "action",
                    "action": "swarm",
                    "status": "processing",
                    "description": "Swarm: Executing parallel tasks..."
                })
            engine_info = {
                "engine": "SwarmingIntegration",
                "algorithm": "Parallel Task Distribution",
                "step": "swarm_execution",
                "description": "Executing swarm tasks in parallel"
            }
            result["results"]["engines_used"].append(engine_info)
            # Send engine update during processing
            if progress_callback:
                await progress_callback({
                    "type": "engines",
                    "engines": [engine_info]
                })
            swarm_results = await self.swarming_integration.execute_swarm(swarm, parallel=True)
            if progress_callback:
                await progress_callback({
                    "type": "action",
                    "action": "swarm",
                    "status": "complete",
                    "description": "Swarm: Execution complete"
                })
            result["results"]["swarm_results"] = swarm_results
            result["results"]["algorithms_used"].append({
                "algorithm": "Micro-Agent Swarm",
                "method": "Parallel Task Distribution + Coordination",
                "step": 4
            })
        except Exception as e:
            result["results"]["swarm"] = None
            result["results"]["swarm_results"] = None
        
        # Step 6: Detect suppression if documents available
        # (Would need documents from swarm results)
        result["results"]["suppression_detected"] = False
        
        # Step 7: Generate insights
        try:
            if progress_callback:
                await progress_callback({
                    "type": "action",
                    "action": "insight_generator",
                    "status": "processing",
                    "description": "Insight Generator: Generating insights..."
                })
                await progress_callback({
                    "type": "thinking",
                    "content": "Generating insights using cross-domain synthesis"
                })
            engine_info = {
                "engine": "InsightGenerator",
                "algorithm": "Cross-Domain Synthesis",
                "step": "insight_generation",
                "description": "Generating insights using Enhanced Deliberation methodology"
            }
            result["results"]["engines_used"].append(engine_info)
            # Send engine update during processing
            if progress_callback:
                await progress_callback({
                    "type": "engines",
                    "engines": [engine_info]
                })
            if hasattr(self.curiosity_integration, 'insight_generator'):
                insights = self.curiosity_integration.insight_generator.generate_insights(
                    query,
                    domain=domain
                )
            else:
                # Use insight generator directly
                from ..research.insight_generator import InsightGenerator
                insight_generator = InsightGenerator()
                insights = insight_generator.generate_insights(
                    query=query,
                    domain=domain
                )
            result["results"]["insights"] = insights
            result["results"]["algorithms_used"].append({
                "algorithm": "Insight Generation",
                "method": "Cross-Domain Synthesis + Pattern Detection",
                "step": 7
            })
        except Exception as e:
            # Fallback insights
            result["results"]["insights"] = {
                "insights": [],
                "breakthroughs": [],
                "suppression_detected": False
            }
        
        # Step 8: Track sources and citations
        try:
            result["results"]["engines_used"].append({
                "engine": "SourceCitationTracker",
                "algorithm": "Citation Tracking",
                "step": "source_tracking",
                "description": "Tracking and managing source citations"
            })
            # Extract sources from results
            sources = []
            if swarm_results.get("agent_results"):
                for agent_result in swarm_results.get("agent_results", []):
                    if agent_result.get("sources"):
                        sources.extend(agent_result.get("sources", []))
            
            # Track citation
            if sources:
                citation_id = self.source_citation_tracker.track_citation(
                    query=query,
                    response=str(result.get("results", {})),
                    sources=sources,
                    metadata={"domain": domain, "methodology": "enhanced_deliberation"}
                )
                result["citation_id"] = citation_id
        except Exception:
            pass  # Citation tracking is optional
        
        # Step 9: Track emergent intelligence
        try:
            insights = result.get("results", {}).get("insights", {})
            if insights.get("insights"):
                for insight in insights.get("insights", []):
                    if isinstance(insight, dict):
                        content = insight.get("description", "") or insight.get("insight", "")
                        if content:
                            intelligence_id = self.emergent_intelligence_tracker.track_intelligence(
                                content=content,
                                domain=domain or "general",
                                intelligence_type="insight",
                                metadata={"query": query, "methodology": "enhanced_deliberation"}
                            )
        except Exception:
            pass  # Intelligence tracking is optional
        
        # Step 10: Publish to blackboard
        try:
            await self.blackboard_integration.publish_truth_discovery({
                "query": query,
                "results": result["results"]
            })
        except Exception:
            pass  # Blackboard publishing is optional
        
        # Ensure result always has content
        if not result.get("results", {}).get("content"):
            # Try to extract content from agent results
            agent_results = result.get("results", {}).get("agent_results", {})
            if agent_results:
                # Get first available agent result
                for agent_id, agent_result in agent_results.items():
                    if agent_result and isinstance(agent_result, str):
                        result["results"]["content"] = agent_result
                        break
                    elif agent_result and isinstance(agent_result, dict):
                        content = agent_result.get("content") or agent_result.get("response") or str(agent_result)
                        if content:
                            result["results"]["content"] = content
                            break
            
            # If still no content, provide fallback
            if not result.get("results", {}).get("content"):
                result["results"]["content"] = "I processed your query but couldn't generate a response. Please try rephrasing your question."
                logger.warning("No content generated for query, using fallback")
        
        return result
    
    async def generate_device_with_full_integration(
        self,
        device_type: str,
        requirements: Dict[str, Any],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate device with full system integration"""
        # Generate device using Enhanced Deliberation methodology
        device = await self.device_generator.generate_device(
            device_type=device_type,
            requirements=requirements,
            domain=domain
        )
        
        # Publish device discovery to blackboard
        await self.blackboard_integration.publish_autonomous_discovery({
            "type": "device_generation",
            "device": device
        })
        
        return device
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get integrated system status"""
        return {
            "integrated": self.integrated,
            "blackboard": self.blackboard_integration.get_blackboard_status(),
            "curiosity": self.curiosity_integration.get_curiosity_status(),
            "swarming": self.swarming_integration.get_swarm_capabilities(),
            "device_generation": {
                "capabilities": [
                    "General-purpose device generation",
                    "Schematic generation",
                    "Code generation",
                    "BOM generation",
                    "Assembly instructions"
                ]
            },
            "research_methodology": {
                "methodology": "enhanced_deliberation",
                "components": self.methodology_analyzer.get_methodology_components()
            }
        }

