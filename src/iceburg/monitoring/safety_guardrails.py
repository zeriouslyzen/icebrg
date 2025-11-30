"""
Safety Guardrails for Self-Healing
Runtime verification and rollback using ChromaDB memory snapshots
"""

import logging
import time
from typing import Dict, Any, Optional, List
from .verifiable_rollback import VerifiableRollback
from ..governance.human_oversight import HumanOversight
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """Safety guardrails for self-healing."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize safety guardrails.
        
        Args:
            config: Configuration for safety guardrails
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # Verifiable rollback using ChromaDB memory snapshots
        self.rollback = VerifiableRollback(config=self.config, cfg=cfg)
        
        # Human oversight for high-risk modifications
        self.human_oversight = HumanOversight(config=self.config, cfg=cfg)
        
        # Circuit breakers for cascade prevention
        self.circuit_breaker = CircuitBreaker(config=self.config)
        
        # Safety state
        self.safety_history = []
    
    async def verify_before_deployment(
        self,
        modification: Dict[str, Any],
        severity: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """
        Verify modification before deployment.
        
        Args:
            modification: Modification to verify
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            
        Returns:
            Verification results
        """
        logger.info(f"Verifying modification with severity: {severity}")
        
        # Create rollback snapshot
        snapshot = await self.rollback.create_snapshot()
        
        # Check if human oversight is required (severity >= HIGH)
        requires_oversight = severity in ["HIGH", "CRITICAL"]
        
        if requires_oversight:
            # Request human oversight
            oversight_result = await self.human_oversight.request_approval(modification, severity)
            
            if not oversight_result.get("approved", False):
                logger.warning("Modification rejected by human oversight")
                return {
                    "approved": False,
                    "reason": "human_oversight_rejected",
                    "snapshot": snapshot
                }
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, rejecting modification")
            return {
                "approved": False,
                "reason": "circuit_breaker_open",
                "snapshot": snapshot
            }
        
        # Verification passed
        logger.info("Modification verified and approved")
        
        return {
            "approved": True,
            "snapshot": snapshot,
            "oversight_required": requires_oversight
        }
    
    async def rollback_if_needed(
        self,
        modification: Dict[str, Any],
        snapshot: Dict[str, Any],
        metrics_after: Dict[str, Any]
    ) -> bool:
        """
        Rollback if modification causes issues.
        
        Args:
            modification: Modification that was applied
            snapshot: Rollback snapshot
            metrics_after: Metrics after modification
            
        Returns:
            True if rollback was performed
        """
        # Check if rollback is needed (e.g., performance regression > 15%)
        regression_threshold = 0.15  # 15% regression
        
        # Calculate regression (simplified - in real implementation, would compare metrics)
        regression = 0.0  # Placeholder
        
        if regression > regression_threshold:
            logger.warning(f"Performance regression detected: {regression:.1%}, rolling back")
            
            # Perform rollback
            rollback_success = await self.rollback.restore_snapshot(snapshot)
            
            if rollback_success:
                # Open circuit breaker to prevent further modifications
                self.circuit_breaker.open()
                logger.info("Rollback completed, circuit breaker opened")
                return True
            else:
                logger.error("Rollback failed")
                return False
        
        return False
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety status."""
        return {
            "rollback_available": self.rollback is not None,
            "human_oversight_enabled": self.human_oversight is not None,
            "circuit_breaker_status": self.circuit_breaker.get_status(),
            "safety_history": len(self.safety_history)
        }

