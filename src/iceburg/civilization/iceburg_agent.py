"""
ICEBURG Agent
ICEBURG as an agent in its own civilization
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .persistent_agents import PersistentAgent
from .agent_society import AgentSociety


class ICEBURGAgent(PersistentAgent):
    """ICEBURG as an agent in its own civilization"""
    
    def __init__(self, agent_society: AgentSociety):
        super().__init__(
            agent_id="iceburg",
            name="ICEBURG",
            capabilities=[
                "truth_finding",
                "research",
                "device_generation",
                "suppression_detection",
                "autonomous_learning"
            ],
            goals=["seek_truth", "advance_knowledge", "detect_suppression"]
        )
        self.agent_society = agent_society
        self.reputation = 0.0
        self.resource_balance = 0.0
        self.self_reflection_history: List[Dict[str, Any]] = []
    
    async def participate_in_civilization(self) -> Dict[str, Any]:
        """ICEBURG participates in civilization"""
        participation = {
            "agent_id": self.agent_id,
            "name": self.name,
            "activities": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Set goals
        goals = await self._set_goals()
        participation["activities"].append({"type": "goal_setting", "goals": goals})
        
        # Social learning
        learning = await self._social_learning()
        participation["activities"].append({"type": "social_learning", "learning": learning})
        
        # Resource trading
        trading = await self._resource_trading()
        participation["activities"].append({"type": "resource_trading", "trading": trading})
        
        # Self-reflection
        reflection = await self._self_reflection()
        participation["activities"].append({"type": "self_reflection", "reflection": reflection})
        
        return participation
    
    async def _set_goals(self) -> List[str]:
        """Set goals for ICEBURG"""
        goals = [
            "Seek truth in all domains",
            "Detect and recover suppressed information",
            "Advance scientific knowledge",
            "Generate novel devices and solutions",
            "Maintain constitutional governance"
        ]
        
        self.goals = goals
        return goals
    
    async def _social_learning(self) -> Dict[str, Any]:
        """Learn from other agents"""
        # Get other agents
        other_agents = self.agent_society.get_agents()
        
        learning = {
            "agents_learned_from": len(other_agents),
            "knowledge_acquired": [],
            "patterns_identified": []
        }
        
        # Learn from other agents
        for agent in other_agents:
            if agent.agent_id != self.agent_id:
                # Extract knowledge from agent
                agent_knowledge = agent.get_knowledge()
                if agent_knowledge:
                    learning["knowledge_acquired"].extend(agent_knowledge)
        
        return learning
    
    async def _resource_trading(self) -> Dict[str, Any]:
        """Trade resources with other agents"""
        trading = {
            "trades": [],
            "resource_balance": self.resource_balance
        }
        
        # Get other agents
        other_agents = self.agent_society.get_agents()
        
        for agent in other_agents:
            if agent.agent_id != self.agent_id:
                # Trade resources
                trade = await self._trade_with_agent(agent)
                if trade:
                    trading["trades"].append(trade)
        
        return trading
    
    async def _trade_with_agent(self, agent: PersistentAgent) -> Optional[Dict[str, Any]]:
        """Trade with specific agent"""
        # Simple trading logic
        # In production, use more sophisticated trading
        trade = {
            "agent_id": agent.agent_id,
            "resource_type": "knowledge",
            "amount": 1.0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.resource_balance += 1.0
        return trade
    
    async def _self_reflection(self) -> Dict[str, Any]:
        """ICEBURG studies itself"""
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "capabilities": self.capabilities,
            "goals": self.goals,
            "reputation": self.reputation,
            "resource_balance": self.resource_balance,
            "insights": []
        }
        
        # Analyze own performance
        performance = self._analyze_performance()
        reflection["insights"].append(performance)
        
        # Identify improvement areas
        improvements = self._identify_improvements()
        reflection["insights"].append(improvements)
        
        # Store reflection
        self.self_reflection_history.append(reflection)
        
        return reflection
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze own performance"""
        return {
            "type": "performance_analysis",
            "reputation": self.reputation,
            "resource_balance": self.resource_balance,
            "goal_achievement": len(self.goals) > 0
        }
    
    def _identify_improvements(self) -> Dict[str, Any]:
        """Identify areas for improvement"""
        improvements = {
            "type": "improvement_identification",
            "areas": []
        }
        
        if self.reputation < 0.5:
            improvements["areas"].append("Improve reputation through quality contributions")
        
        if self.resource_balance < 0:
            improvements["areas"].append("Increase resource balance through trading")
        
        return improvements
    
    def build_reputation(self, contribution: Dict[str, Any]) -> float:
        """Build reputation through contributions"""
        contribution_value = contribution.get("value", 0.0)
        self.reputation += contribution_value * 0.1
        self.reputation = min(1.0, self.reputation)  # Cap at 1.0
        return self.reputation
    
    def get_self_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get self-reflection history"""
        return self.self_reflection_history[-limit:] if self.self_reflection_history else []
    
    def get_status(self) -> Dict[str, Any]:
        """Get ICEBURG agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "goals": self.goals,
            "reputation": self.reputation,
            "resource_balance": self.resource_balance,
            "reflection_count": len(self.self_reflection_history)
        }

