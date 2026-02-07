"""
MoE (Mixture of Experts) Router for ICEBURG.
Routes queries to specialized models based on domain and complexity.
"""

import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .request_router import get_request_router, RoutingDecision

logger = logging.getLogger(__name__)

@dataclass
class MoEDecision:
    """Decision made by the MoE router"""
    model_id: str
    expert_domain: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class MoERouter:
    """
    Orchestrates Mixture-of-Experts routing.
    Detects domain and assigns the best model for the task.
    """
    
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.base_router = get_request_router()
        
        # Expert mappings
        # These can be overridden by environment variables
        self.experts = {
            "simple_chat": "phi3:mini",  # Ultra-fast for greetings
            "fast_chat": getattr(cfg, "surveyor_model", "qwen2.5:7b-instruct"),
            "deep_research": getattr(cfg, "synthesist_model", "deepseek-v2:16b"),
            "finance": "deepseek-v2:16b", 
            "code": "mistral:7b-instruct",
            "philosophical": getattr(cfg, "oracle_model", "deepseek-v2:16b"),
            "general": getattr(cfg, "surveyor_model", "qwen2.5:7b")
        }
        
        # M4 Hardware Optimization
        if getattr(cfg, "m4_optimization", False):
            logger.info("⚡ M4 Optimization detected: Preferring local MLX-optimized models")
            self.experts["fast_chat"] = "qwen2.5:7b-instruct" # Very fast on M4
            self.experts["code"] = "deepseek-v2:16b" # Better for coding on Mac

        
        # Domain patterns (extending base router)
        self.finance_patterns = [
            r"\b(price|market|trading|stock|crypto|btc|eth|sol|alpha|signal|portfolio|invest|finance)\b",
            r"\b(usd|usdc|yield|apy|volatility|liquidity|order book)\b"
        ]
        
        self.code_patterns = [
            r"\b(python|javascript|js|ts|typescript|java|c\+\+|rust|golang|bash|shell|script)\b",
            r"\b(function|method|class|variable|loop|async|await|promise|api|endpoint|debug|error|trace)\b",
            r"\b(implementation|refactor|test|unit test|integration test|mock|stub)\b"
        ]

    def route_to_expert(self, query: str, context: Optional[Dict[str, Any]] = None) -> MoEDecision:
        """Determines the best expert model for the query"""
        query_lower = query.lower()
        
        # 1. Use base RequestRouter for primary mode classification
        routing_decision = self.base_router.route(query, context)
        
        # 2. FAST PATH: Simple chat gets ultra-fast model
        if routing_decision.metadata.get("fast_path"):
            logger.info(f"⚡ Simple chat fast-path: {query[:30]}...")
            return MoEDecision(
                model_id=self.experts["simple_chat"],
                expert_domain="simple_chat",
                confidence=0.99,
                reasoning="Simple greeting/chat - using ultra-fast model",
                metadata={
                    "routing_mode": routing_decision.mode,
                    "fast_path": True
                }
            )
        
        # 3. Check for specialized domains
        domain = "general"
        model_id = self.experts["general"]
        
        # Finance Detection
        if any(re.search(p, query_lower) for p in self.finance_patterns):
            domain = "finance"
            model_id = self.experts["finance"]
            logger.info(f"💰 Finance domain detected: {query[:50]}...")
            
        # Code Detection
        elif any(re.search(p, query_lower) for p in self.code_patterns):
            domain = "code"
            model_id = self.experts["code"]
            logger.info(f"💻 Code domain detected: {query[:50]}...")
            
        # Deep Research Detection based on complexity
        elif routing_decision.mode == "web_research" and (len(query) > 200 or "?" in query):
            domain = "deep_research"
            model_id = self.experts["deep_research"]
            logger.info(f"🔬 Deep research domain detected: {query[:50]}...")
            
        # Philosophical/Abstract (not fast_path)
        elif routing_decision.mode == "pure_reasoning":
            domain = "philosophical"
            model_id = self.experts["philosophical"]
            logger.info(f"🧠 Philosophical domain detected: {query[:50]}...")

        # 4. Final decision
        return MoEDecision(
            model_id=model_id,
            expert_domain=domain,
            confidence=routing_decision.confidence,
            reasoning=f"Routed to {domain} expert based on patterns and {routing_decision.reasoning}",
            metadata={
                "routing_mode": routing_decision.mode,
                "confidence": routing_decision.confidence
            }
        )

# Global helper
_moe_router = None

def get_moe_router(cfg: Any) -> MoERouter:
    global _moe_router
    if _moe_router is None:
        _moe_router = MoERouter(cfg)
    return _moe_router
