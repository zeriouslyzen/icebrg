"""
Human Oversight for High-Risk Modifications
Human-in-loop for high-risk changes (severity >= HIGH)
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class HumanOversight:
    """Human oversight for high-risk modifications."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize human oversight.
        
        Args:
            config: Configuration for human oversight
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # Approval queue
        self.approval_queue = []
        
        # Auto-approval threshold (for testing - in production, would require human approval)
        self.auto_approve = self.config.get("auto_approve", False)
    
    async def request_approval(
        self,
        modification: Dict[str, Any],
        severity: str = "HIGH"
    ) -> Dict[str, Any]:
        """
        Request human approval for modification.
        
        Args:
            modification: Modification to approve
            severity: Severity level
            
        Returns:
            Approval result
        """
        logger.info(f"Requesting approval for modification with severity: {severity}")
        
        # Create approval request
        approval_request = {
            "modification": modification,
            "severity": severity,
            "status": ApprovalStatus.PENDING,
            "timestamp": time.time()
        }
        
        # Add to queue
        self.approval_queue.append(approval_request)
        
        # For testing, auto-approve if enabled
        if self.auto_approve:
            approval_request["status"] = ApprovalStatus.APPROVED
            logger.info("Auto-approved modification (auto_approve enabled)")
            return {"approved": True, "reason": "auto_approved"}
        
        # In production, would wait for human approval
        # For now, return pending status
        logger.warning("Modification pending human approval (auto_approve disabled)")
        return {"approved": False, "reason": "pending_human_approval", "request": approval_request}
    
    def get_pending_approvals(self) -> list:
        """Get pending approval requests."""
        return [
            req for req in self.approval_queue
            if req["status"] == ApprovalStatus.PENDING
        ]
    
    def approve_request(self, request_id: int) -> bool:
        """Approve a request (for testing/manual approval)."""
        if 0 <= request_id < len(self.approval_queue):
            request = self.approval_queue[request_id]
            request["status"] = ApprovalStatus.APPROVED
            logger.info(f"Approved request {request_id}")
            return True
        return False
    
    def reject_request(self, request_id: int) -> bool:
        """Reject a request (for testing/manual approval)."""
        if 0 <= request_id < len(self.approval_queue):
            request = self.approval_queue[request_id]
            request["status"] = ApprovalStatus.REJECTED
            logger.info(f"Rejected request {request_id}")
            return True
        return False


# Import time for timestamp
import time

