"""
Blackboard Integration
Deep integration with global workspace/blackboard
"""

from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime
from ..global_workspace import GlobalWorkspace
from ..lab.virtual_physics_lab import VirtualPhysicsLab
from ..security.autonomous_red_team import AutonomousRedTeam
from ..autonomous.company_integration import CompanyIntegration
from ..micro_agent_swarm import MicroAgentSwarm
from ..curiosity.curiosity_engine import CuriosityEngine
from ..truth.suppression_detector import SuppressionDetector


class BlackboardIntegration:
    """Deep integration with blackboard/global workspace"""
    
    def __init__(self):
        self.global_workspace = GlobalWorkspace()
        self.subscriptions: Dict[str, List[callable]] = {}
        self._setup_integrations()
    
    def _setup_integrations(self):
        """Setup integrations with all systems"""
        # Subscribe to lab results
        try:
            self.global_workspace.subscribe(
                "lab_results",
                self._handle_lab_results
            )
        except Exception:
            pass
        
        # Subscribe to red team findings
        try:
            self.global_workspace.subscribe(
                "red_team_findings",
                self._handle_red_team_findings
            )
        except Exception:
            pass
        
        # Subscribe to autonomous learning discoveries
        try:
            self.global_workspace.subscribe(
                "autonomous_discoveries",
                self._handle_autonomous_discoveries
            )
        except Exception:
            pass
        
        # Subscribe to swarm coordination
        try:
            self.global_workspace.subscribe(
                "swarm_coordination",
                self._handle_swarm_coordination
            )
        except Exception:
            pass
        
        # Subscribe to curiosity queries
        try:
            self.global_workspace.subscribe(
                "curiosity_queries",
                self._handle_curiosity_queries
            )
        except Exception:
            pass
        
        # Subscribe to truth-finding discoveries
        try:
            self.global_workspace.subscribe(
                "truth_discoveries",
                self._handle_truth_discoveries
            )
        except Exception:
            pass
    
    async def _handle_lab_results(self, message: Dict[str, Any]):
        """Handle lab results from blackboard"""
        result = message.get("data", {})
        
        # Broadcast lab results
        await self.global_workspace.publish(
            "lab_results_broadcast",
            {
                "type": "lab_result",
                "result": result,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def _handle_red_team_findings(self, message: Dict[str, Any]):
        """Handle red team findings from blackboard"""
        finding = message.get("data", {})
        
        # Broadcast security findings
        await self.global_workspace.publish(
            "security_findings_broadcast",
            {
                "type": "security_finding",
                "finding": finding,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def _handle_autonomous_discoveries(self, message: Dict[str, Any]):
        """Handle autonomous learning discoveries"""
        discovery = message.get("data", {})
        
        # Broadcast discoveries
        await self.global_workspace.publish(
            "autonomous_discoveries_broadcast",
            {
                "type": "autonomous_discovery",
                "discovery": discovery,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def _handle_swarm_coordination(self, message: Dict[str, Any]):
        """Handle swarm coordination messages"""
        coordination = message.get("data", {})
        
        # Broadcast swarm coordination
        await self.global_workspace.publish(
            "swarm_coordination_broadcast",
            {
                "type": "swarm_coordination",
                "coordination": coordination,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def _handle_curiosity_queries(self, message: Dict[str, Any]):
        """Handle curiosity queries"""
        query = message.get("data", {})
        
        # Broadcast curiosity queries
        await self.global_workspace.publish(
            "curiosity_queries_broadcast",
            {
                "type": "curiosity_query",
                "query": query,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def _handle_truth_discoveries(self, message: Dict[str, Any]):
        """Handle truth-finding discoveries"""
        discovery = message.get("data", {})
        
        # Broadcast truth discoveries
        await self.global_workspace.publish(
            "truth_discoveries_broadcast",
            {
                "type": "truth_discovery",
                "discovery": discovery,
                "timestamp": message.get("timestamp")
            }
        )
    
    async def publish_lab_result(
        self,
        lab_result: Dict[str, Any]
    ) -> bool:
        """Publish lab result to blackboard"""
        await self.global_workspace.publish(
            "lab_results",
            {
                "data": lab_result,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    async def publish_red_team_finding(
        self,
        finding: Dict[str, Any]
    ) -> bool:
        """Publish red team finding to blackboard"""
        await self.global_workspace.publish(
            "red_team_findings",
            {
                "data": finding,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    async def publish_autonomous_discovery(
        self,
        discovery: Dict[str, Any]
    ) -> bool:
        """Publish autonomous discovery to blackboard"""
        await self.global_workspace.publish(
            "autonomous_discoveries",
            {
                "data": discovery,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    async def publish_swarm_coordination(
        self,
        coordination: Dict[str, Any]
    ) -> bool:
        """Publish swarm coordination to blackboard"""
        await self.global_workspace.publish(
            "swarm_coordination",
            {
                "data": coordination,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    async def publish_curiosity_query(
        self,
        query: Dict[str, Any]
    ) -> bool:
        """Publish curiosity query to blackboard"""
        await self.global_workspace.publish(
            "curiosity_queries",
            {
                "data": query,
                "timestamp": datetime.now().isoformat()
            }
        )
        return True
    
    async def publish_truth_discovery(
        self,
        discovery: Dict[str, Any]
    ) -> bool:
        """Publish truth discovery to blackboard"""
        try:
            if self.global_workspace:
                result = self.global_workspace.publish(
                    "truth_discoveries",
                    {
                        "data": discovery,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                if asyncio.iscoroutine(result):
                    await result
        except Exception:
            pass
        return True
    
    def get_blackboard_status(self) -> Dict[str, Any]:
        """Get blackboard status"""
        return {
            "subscriptions": len(self.subscriptions),
            "integrations": [
                "lab_results",
                "red_team_findings",
                "autonomous_discoveries",
                "swarm_coordination",
                "curiosity_queries",
                "truth_discoveries"
            ]
        }

