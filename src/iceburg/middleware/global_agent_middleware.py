"""
Global Agent Middleware
Intercepts all agent outputs and applies hallucination detection and emergence tracking.
"""

import asyncio
import inspect
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from datetime import datetime
import logging

from .middleware_registry import MiddlewareRegistry
from ..validation.hallucination_detector import HallucinationDetector
from ..emergence_detector import EmergenceDetector
from ..global_workspace import GlobalWorkspace
from ..config import IceburgConfig

logger = logging.getLogger(__name__)


class GlobalAgentMiddleware:
    """
    Global middleware that intercepts all agent outputs.
    
    Features:
    - Automatic hallucination detection
    - Automatic emergence tracking
    - Pattern learning and sharing
    - Non-invasive agent wrapping
    - Backward compatible
    """
    
    def __init__(self, cfg: IceburgConfig):
        """
        Initialize global agent middleware.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        self.registry = MiddlewareRegistry()
        
        # Initialize detection systems
        self.hallucination_detector = None
        self.emergence_detector = None
        self.workspace = None
        
        if self.registry.config.get("enable_hallucination_detection", True):
            try:
                detector_config = {
                    "hallucination_threshold": self.registry.config.get("hallucination_threshold", 0.15),
                    "confidence_threshold": self.registry.config.get("confidence_threshold", 0.85),
                    "consistency_threshold": self.registry.config.get("consistency_threshold", 0.8)
                }
                self.hallucination_detector = HallucinationDetector(detector_config)
                logger.info("✅ Global hallucination detector initialized")
            except Exception as e:
                logger.warning(f"Could not initialize hallucination detector: {e}")
        
        if self.registry.config.get("enable_emergence_tracking", True):
            try:
                self.emergence_detector = EmergenceDetector()
                logger.info("✅ Global emergence detector initialized")
            except Exception as e:
                logger.warning(f"Could not initialize emergence detector: {e}")
        
        # Initialize GlobalWorkspace for sharing
        try:
            self.workspace = GlobalWorkspace()
            logger.info("✅ GlobalWorkspace initialized for pattern sharing")
        except Exception as e:
            logger.warning(f"Could not initialize GlobalWorkspace: {e}")
        
        # Initialize learning system (Phase 2)
        self.learning_system = None
        if self.registry.config.get("enable_learning", True):
            try:
                from .hallucination_learning import HallucinationLearning
                self.learning_system = HallucinationLearning(cfg)
                logger.info("✅ Hallucination learning system initialized")
            except Exception as e:
                logger.debug(f"Learning system not yet available: {e}")
        
        # Initialize emergence aggregator (Phase 3)
        self.emergence_aggregator = None
        if self.registry.config.get("enable_emergence_tracking", True):
            try:
                from .emergence_aggregator import EmergenceAggregator
                self.emergence_aggregator = EmergenceAggregator(cfg)
                logger.info("✅ Emergence aggregator initialized")
            except Exception as e:
                logger.debug(f"Emergence aggregator not yet available: {e}")
    
    async def execute_agent(
        self,
        agent_name: str,
        agent_func: Union[Callable, Awaitable],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an agent with middleware interception.
        
        Args:
            agent_name: Name of the agent
            agent_func: Agent function to execute
            *args: Positional arguments for agent
            **kwargs: Keyword arguments for agent
            
        Returns:
            Agent result (unchanged)
        """
        # Check if middleware is enabled for this agent
        if not self.registry.is_enabled(agent_name):
            # Execute agent directly without middleware
            return await self._execute_agent_direct(agent_func, *args, **kwargs)
        
        agent_config = self.registry.get_config(agent_name)
        
        # Extract query for pattern checking
        query = args[0] if args else kwargs.get('query', '')
        
        # Pre-execution: Check for known patterns (if learning system available)
        known_patterns = None
        if self.learning_system:
            try:
                known_patterns = self.learning_system.check_patterns(query, agent_name)
                if known_patterns:
                    logger.debug(f"Found {len(known_patterns)} known patterns for agent {agent_name}")
            except Exception as e:
                logger.debug(f"Could not check patterns: {e}")
        
        # Execute agent (unchanged)
        try:
            result = await self._execute_agent_direct(agent_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Agent {agent_name} execution failed: {e}")
            raise
        
        # Post-execution: Detection & Learning
        try:
            await self._process_agent_output(
                agent_name=agent_name,
                result=result,
                query=query,
                context=kwargs,
                agent_config=agent_config
            )
        except Exception as e:
            # Don't fail agent execution if middleware fails
            logger.warning(f"Middleware processing failed for {agent_name}: {e}")
        
        return result  # Return unchanged result
    
    async def _execute_agent_direct(
        self,
        agent_func: Union[Callable, Awaitable],
        *args,
        **kwargs
    ) -> Any:
        """Execute agent function directly, handling both sync and async."""
        if inspect.iscoroutinefunction(agent_func):
            return await agent_func(*args, **kwargs)
        elif inspect.isawaitable(agent_func):
            return await agent_func
        else:
            # Sync function - run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: agent_func(*args, **kwargs))
    
    async def _process_agent_output(
        self,
        agent_name: str,
        result: Any,
        query: str,
        context: Dict[str, Any],
        agent_config: Dict[str, Any]
    ):
        """
        Process agent output for detection and learning.
        
        Args:
            agent_name: Name of the agent
            result: Agent output
            query: Original query
            context: Execution context
            agent_config: Agent configuration
        """
        # Convert result to string if needed
        result_str = str(result) if not isinstance(result, str) else result
        
        # Hallucination detection
        if agent_config.get("enable_hallucination_detection", True) and self.hallucination_detector:
            try:
                # Extract sources if available
                sources = context.get('sources', [])
                evidence_level = context.get('evidence_level')
                
                hallucination_result = self.hallucination_detector.detect_hallucination(
                    content=result_str,
                    sources=sources if sources else None,
                    evidence_level=evidence_level,
                    context={
                        "agent": agent_name,
                        "query": query,
                        **context
                    }
                )
                
                # Learn from hallucination if detected
                if hallucination_result.get("hallucination_detected") and self.learning_system:
                    try:
                        self.learning_system.learn_pattern(
                            agent=agent_name,
                            pattern=hallucination_result,
                            query=query,
                            result=result_str,
                            context=context
                        )
                    except Exception as e:
                        logger.debug(f"Could not learn pattern: {e}")
                
                # Share via GlobalWorkspace
                if hallucination_result.get("hallucination_detected") and self.workspace:
                    try:
                        self.workspace.publish('hallucination/detected', {
                            'agent': agent_name,
                            'pattern': hallucination_result,
                            'query': query[:200],
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Could not publish to workspace: {e}")
                
            except Exception as e:
                logger.debug(f"Hallucination detection failed for {agent_name}: {e}")
        
        # Emergence detection
        if agent_config.get("enable_emergence_tracking", True) and self.emergence_detector:
            try:
                # Try to extract claims and evidence level from context
                claims = context.get('claims', [])
                evidence_level = context.get('evidence_level', 'B')
                
                # For non-Oracle agents, try to detect emergence from result
                if agent_name != "oracle":
                    # Simple emergence detection for text results
                    emergence_score = self._detect_emergence_simple(result_str, agent_name)
                    if emergence_score > agent_config.get("emergence_threshold", 0.6):
                        emergence_result = {
                            "emergence_score": emergence_score,
                            "emergence_type": "pattern_discovery",
                            "agent": agent_name,
                            "content": result_str[:500]
                        }
                    else:
                        emergence_result = None
                else:
                    # Oracle-specific emergence detection
                    emergence_result = self.emergence_detector.process(
                        oracle_output=result_str,
                        claims=claims,
                        evidence_level=evidence_level
                    )
                
                # Aggregate emergence if detected
                if emergence_result and self.emergence_aggregator:
                    try:
                        self.emergence_aggregator.record_emergence(
                            agent=agent_name,
                            emergence=emergence_result,
                            query=query,
                            context=context
                        )
                    except Exception as e:
                        logger.debug(f"Could not aggregate emergence: {e}")
                
                # Share via GlobalWorkspace
                if emergence_result and self.workspace:
                    try:
                        self.workspace.publish('emergence/detected', {
                            'agent': agent_name,
                            'emergence': emergence_result,
                            'query': query[:200],
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Could not publish emergence: {e}")
                
            except Exception as e:
                logger.debug(f"Emergence detection failed for {agent_name}: {e}")
    
    def _detect_emergence_simple(self, content: str, agent_name: str) -> float:
        """
        Simple emergence detection for non-Oracle agents.
        
        Args:
            content: Agent output content
            agent_name: Agent name
            
        Returns:
            Emergence score (0.0 to 1.0)
        """
        score = 0.0
        content_lower = content.lower()
        
        # Check for emergence indicators
        emergence_keywords = [
            "novel", "unprecedented", "breakthrough", "discovery",
            "emergent", "emergence", "paradigm", "revolutionary",
            "unexpected", "surprising", "counterintuitive",
            "cross-domain", "synthesis", "integration"
        ]
        
        for keyword in emergence_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # Check for multiple domains mentioned
        domain_indicators = ["quantum", "biological", "computational", "neural", "cognitive"]
        domains_mentioned = sum(1 for domain in domain_indicators if domain in content_lower)
        if domains_mentioned > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def execute_agent_sync(
        self,
        agent_name: str,
        agent_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Synchronous version of execute_agent for sync contexts.
        
        Args:
            agent_name: Name of the agent
            agent_func: Agent function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Agent result
        """
        # Check if middleware is enabled
        if not self.registry.is_enabled(agent_name):
            return agent_func(*args, **kwargs)
        
        agent_config = self.registry.get_config(agent_name)
        query = args[0] if args else kwargs.get('query', '')
        
        # Execute agent
        result = agent_func(*args, **kwargs)
        
        # Process output (async but we'll run it)
        try:
            # Run async processing in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule as task
                asyncio.create_task(self._process_agent_output(
                    agent_name=agent_name,
                    result=result,
                    query=query,
                    context=kwargs,
                    agent_config=agent_config
                ))
            else:
                # If no loop, run it
                loop.run_until_complete(self._process_agent_output(
                    agent_name=agent_name,
                    result=result,
                    query=query,
                    context=kwargs,
                    agent_config=agent_config
                ))
        except Exception as e:
            logger.debug(f"Could not process output synchronously: {e}")
        
        return result

