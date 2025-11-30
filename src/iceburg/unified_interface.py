"""
ICEBURG Unified Interface - Single entry point with mode auto-detection
Provides simplified access to all ICEBURG capabilities through intelligent routing.
"""

import asyncio
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import json

from .config import load_config
from .protocol_minimal import run_iceberg_protocol
from .agents.architect import Architect
from .memory.unified_memory import UnifiedMemory
from .global_workspace import GlobalWorkspace
from .gnosis.unified_gnosis_interface import UnifiedGnosisInterface

# Try to import Oracle, but make it optional
try:
    from .agents.oracle import run as oracle_run
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False
    oracle_run = None


class UnifiedICEBURG:
    """
    Unified interface for all ICEBURG capabilities with intelligent mode detection.
    
    Modes:
    - research: "Analyze quantum computing applications" → full protocol with emergence detection
    - chat: "What is X?" → reflexive fast path (30s)
    - software: "Build a calculator app" → Architect + Think Tank
    - science: "Design experiment for Y" → Oracle + hypothesis testing
    - civilization: "Simulate agent society with resource trading" → World model + MAS
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path."""
        self.config = load_config()  # load_config doesn't take arguments
        self.memory = UnifiedMemory(self.config)
        self.workspace = GlobalWorkspace()
        
        # Initialize components
        self.architect = Architect(self.config)
        self.oracle_available = ORACLE_AVAILABLE
        
        # Initialize gnosis interface
        try:
            self.gnosis_interface = UnifiedGnosisInterface(self.config)
        except Exception as e:
            print(f"Warning: Could not initialize gnosis interface: {e}")
            self.gnosis_interface = None
        
        # Mode detection patterns
        self.mode_patterns = {
            'research': [
                r'analyze|research|investigate|study|examine|explore',
                r'quantum|AI|machine learning|deep learning|neural',
                r'breakthrough|discovery|innovation|novel',
                r'comprehensive|detailed|thorough|systematic'
            ],
            'chat': [
                r'what is|how does|explain|define|describe',
                r'quick|simple|brief|fast',
                r'help|assist|guide|support'
            ],
            'software': [
                r'build|create|develop|generate|make.*app',
                r'calculator|IDE|game|database|web|mobile',
                r'code|programming|software|application'
            ],
            'science': [
                r'experiment|hypothesis|test|trial|study',
                r'design.*experiment|scientific|method',
                r'data|analysis|results|conclusion'
            ],
            'civilization': [
                r'simulate|agent.*society|multi.*agent',
                r'resource.*trading|cooperation|norms',
                r'AGI.*civilization|world.*model',
                r'emergent|social.*learning'
            ]
        }
    
    def _detect_mode(self, query: str) -> str:
        """
        Auto-detect the appropriate mode based on query content.
        
        Args:
            query: User input query
            
        Returns:
            Mode string: 'research', 'chat', 'software', 'science', or 'civilization'
        """
        query_lower = query.lower()
        
        # Score each mode based on pattern matches
        mode_scores = {}
        for mode, patterns in self.mode_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            mode_scores[mode] = score
        
        # Return mode with highest score, default to 'chat' if tie
        best_mode = max(mode_scores, key=mode_scores.get)
        
        # If no patterns match, default to chat for simple queries
        if mode_scores[best_mode] == 0:
            return 'chat'
        
        return best_mode
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calculate query complexity score (0-1 scale).
        
        Args:
            query: User input query
            
        Returns:
            Complexity score between 0 and 1
        """
        # Factors that increase complexity
        complexity_indicators = [
            len(query.split()) > 20,  # Long queries
            any(word in query.lower() for word in ['analyze', 'research', 'comprehensive', 'detailed']),
            any(word in query.lower() for word in ['multiple', 'several', 'various', 'different']),
            '?' in query and len(query.split('?')) > 2,  # Multiple questions
            any(word in query.lower() for word in ['complex', 'advanced', 'sophisticated'])
        ]
        
        # Calculate score
        score = sum(complexity_indicators) / len(complexity_indicators)
        return min(score, 1.0)
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query through the appropriate ICEBURG mode.
        
        Args:
            query: User input query
            context: Optional context dictionary
            
        Returns:
            Dictionary containing results and metadata
        """
        if context is None:
            context = {}
        
        # Detect mode and complexity
        mode = self._detect_mode(query)
        complexity = self._calculate_complexity(query)
        
        # Log interaction to unified memory
        run_id = context.get('run_id', 'default')
        agent_id = context.get('agent_id', 'unified_interface')
        task_id = context.get('task_id', f"task_{asyncio.get_event_loop().time()}")
        event_type = "query"
        text = f"Query: {query}\nMode: {mode}\nComplexity: {complexity}"
        # Flatten context for metadata (ChromaDB doesn't accept nested dicts)
        meta = {
            "query": str(query)[:200],  # Truncate long queries
            "mode": str(mode),
            "complexity": str(complexity),
            "user_id": str(context.get('user_id', 'default'))
        }
        self.memory.log_and_index(run_id, agent_id, task_id, event_type, text, meta)
        
        # Use gnosis interface if available
        gnosis_result = None
        if self.gnosis_interface:
            try:
                gnosis_result = self.gnosis_interface.process_query(query, context.get('user_id', 'default'))
                # Add gnosis knowledge to context
                if gnosis_result.get('gnosis_knowledge'):
                    context['gnosis_knowledge'] = gnosis_result['gnosis_knowledge']
                if gnosis_result.get('matrix_awareness'):
                    context['matrix_awareness'] = gnosis_result['matrix_awareness']
                if gnosis_result.get('computer_capabilities'):
                    context['computer_capabilities'] = gnosis_result['computer_capabilities']
            except Exception as e:
                print(f"Warning: Gnosis interface error: {e}")
        
        # Route to appropriate handler
        try:
            if mode == "civilization":
                result = await self._handle_civilization(query, context)
            elif mode == "software":
                result = await self._handle_software(query, context)
            elif mode == "science":
                result = await self._handle_science(query, context)
            elif mode == "research":
                result = await self._handle_research(query, context)
            else:  # chat mode
                result = await self._handle_chat(query, context)
            
            # Add metadata
            result.update({
                "mode": mode,
                "complexity": complexity,
                "processing_time": asyncio.get_event_loop().time() - context.get('start_time', 0)
            })
            
            # Add gnosis metadata if available
            if gnosis_result:
                result['gnosis'] = {
                    "knowledge_items": len(gnosis_result.get('gnosis_knowledge', {}).get('knowledge_items', [])),
                    "matrices_identified": len(gnosis_result.get('matrix_awareness', {}).get('matrices_identified', [])),
                    "tools_discovered": len(gnosis_result.get('computer_capabilities', {}).get('tools_used', []))
                }
            
            # Accumulate conversation to gnosis
            if self.gnosis_interface:
                try:
                    conversation = {
                        "query": query,
                        "response": str(result.get('result', result.get('message', ''))),
                        "metadata": {
                            "mode": mode,
                            "complexity": complexity,
                            "domains": context.get('domains', [])
                        }
                    }
                    self.gnosis_interface.orchestrator.accumulate_to_gnosis(conversation)
                    self.gnosis_interface.orchestrator.evolve_with_user(context.get('user_id', 'default'), conversation)
                except Exception as e:
                    print(f"Warning: Error accumulating to gnosis: {e}")
            
            # Log result to unified memory
            run_id = context.get('run_id', 'default')
            agent_id = context.get('agent_id', 'unified_interface')
            task_id = context.get('task_id', f"task_{asyncio.get_event_loop().time()}")
            event_type = "result"
            text = f"Result for query: {query}\nMode: {mode}\nResult: {str(result.get('result', ''))[:500]}"
            meta = {
                "mode": str(mode),
                "query": str(query)[:200],
                "user_id": str(context.get('user_id', 'default'))
            }
            self.memory.log_and_index(run_id, agent_id, task_id, event_type, text, meta)
            
            return result
            
        except Exception as e:
            # Log error to unified memory
            run_id = context.get('run_id', 'default')
            agent_id = context.get('agent_id', 'unified_interface')
            task_id = context.get('task_id', f"task_{asyncio.get_event_loop().time()}")
            event_type = "error"
            text = f"Error processing query: {query}\nError: {str(e)[:500]}"
            meta = {
                "mode": str(mode),
                "query": str(query)[:200],
                "error": str(e)[:500],
                "user_id": str(context.get('user_id', 'default'))
            }
            self.memory.log_and_index(run_id, agent_id, task_id, event_type, text, meta)
            raise
    
    async def _handle_civilization(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AGI civilization simulation requests."""
        # Civilization module integration
        try:
            from iceburg.civilization.world_model import WorldModel
            world_model = WorldModel()
            return world_model.get_civilization_state()
        except ImportError:
            # Fallback when civilization module not available
            return {
                "status": "civilization_module_not_available",
                "population": 0,
                "resources": {},
                "technologies": [],
                "social_structures": []
            }
        return {
            "type": "civilization_simulation",
            "message": "AGI civilization simulation not yet implemented",
            "query": query,
            "status": "pending"
        }
    
    async def _handle_software(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle software generation requests."""
        try:
            # Use architect for software generation
            app_request = {
                "description": query,
                "features": context.get("features", []),
                "platform": context.get("platform", "macos"),
                "sign": context.get("sign", False),
                "notarize": context.get("notarize", False),
                "dmg": context.get("dmg", False)
            }
            
            result = await self.architect.generate_application(app_request)
            
            return {
                "type": "software_generation",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "software_generation",
                "error": str(e),
                "status": "error"
            }
    
    async def _handle_science(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scientific experiment design requests."""
        try:
            if self.oracle_available and oracle_run:
                # Use oracle for scientific analysis
                from ..vectorstore import VectorStore
                vs = VectorStore(self.config)
                result = oracle_run(self.config, vs, query, verbose=False)
                
                return {
                    "type": "scientific_analysis",
                    "result": result,
                    "status": "success"
                }
            else:
                return {
                    "type": "scientific_analysis",
                    "result": "Oracle not available",
                    "status": "unavailable"
                }
        except Exception as e:
            return {
                "type": "scientific_analysis",
                "error": str(e),
                "status": "error"
            }
    
    async def _handle_research(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive research requests."""
        try:
            # Use full ICEBURG protocol for research
            result = await run_iceberg_protocol(query, self.config)
            
            return {
                "type": "research_analysis",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "research_analysis",
                "error": str(e),
                "status": "error"
            }
    
    async def _handle_chat(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple chat requests with fast path."""
        try:
            # Use reflexive routing for fast responses
            from .integration.reflexive_routing import ReflexiveRoutingSystem
            
            routing = ReflexiveRoutingSystem()
            result = await routing.route_query(query, self.config)
            
            return {
                "type": "chat_response",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "chat_response",
                "error": str(e),
                "status": "error"
            }
    
    def get_mode_info(self) -> Dict[str, str]:
        """Get information about available modes."""
        return {
            "research": "Comprehensive analysis with full ICEBURG protocol and emergence detection",
            "chat": "Fast responses for simple questions (30s target)",
            "software": "Application generation with Architect and Think Tank",
            "science": "Scientific experiment design with Oracle and hypothesis testing",
            "civilization": "AGI civilization simulation with world models and multi-agent systems"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "modes_available": list(self.mode_patterns.keys()),
            "memory_entries": await self.memory.get_stats(),
            "workspace_agents": len(self.workspace.get_agents()),
            "config_loaded": self.config is not None
        }


# Convenience functions for direct usage
async def process_query(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Process a query through the unified interface."""
    iceburg = UnifiedICEBURG()
    return await iceburg.process(query, context)


def detect_mode(query: str) -> str:
    """Detect the appropriate mode for a query."""
    iceburg = UnifiedICEBURG()
    return iceburg._detect_mode(query)


def calculate_complexity(query: str) -> float:
    """Calculate the complexity of a query."""
    iceburg = UnifiedICEBURG()
    return iceburg._calculate_complexity(query)
