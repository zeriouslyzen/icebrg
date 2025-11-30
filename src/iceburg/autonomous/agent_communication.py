"""
Agent Communication
Agent-to-agent communication during autonomous operation
"""

from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime
from collections import deque
from ..global_workspace import GlobalWorkspace


class AgentCommunication:
    """Agent-to-agent communication during autonomous operation"""
    
    def __init__(self):
        self.global_workspace = GlobalWorkspace()
        self.message_queue: Dict[str, deque] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: Dict[str, Any],
        priority: str = "normal"
    ) -> bool:
        """Send message between agents"""
        if to_agent not in self.message_queue:
            self.message_queue[to_agent] = deque(maxlen=1000)
        
        message_data = {
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to queue with priority
        if priority == "urgent":
            self.message_queue[to_agent].appendleft(message_data)
        else:
            self.message_queue[to_agent].append(message_data)
        
        # Broadcast to global workspace
        await self.global_workspace.publish(
            f"agent_message_{to_agent}",
            message_data
        )
        
        return True
    
    async def receive_messages(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Receive messages for agent"""
        if agent_id not in self.message_queue:
            return []
        
        messages = []
        queue = self.message_queue[agent_id]
        
        # Get urgent messages first
        urgent_messages = [m for m in queue if m.get("priority") == "urgent"]
        messages.extend(urgent_messages[:limit])
        
        # Get normal messages
        remaining = limit - len(messages)
        normal_messages = [m for m in queue if m.get("priority") != "urgent"]
        messages.extend(normal_messages[:remaining])
        
        # Remove processed messages
        for msg in messages:
            if msg in queue:
                queue.remove(msg)
        
        return messages
    
    def broadcast_status(
        self,
        agent_id: str,
        status: Dict[str, Any]
    ) -> bool:
        """Broadcast agent status"""
        self.agent_status[agent_id] = {
            **status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to global workspace
        asyncio.create_task(
            self.global_workspace.publish(
                f"agent_status_{agent_id}",
                self.agent_status[agent_id]
            )
        )
        
        return True
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        return self.agent_status.get(agent_id)
    
    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent statuses"""
        return self.agent_status.copy()
    
    async def collaborate_on_task(
        self,
        task: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """Collaborate on task with multiple agents"""
        collaboration = {
            "task": task,
            "agents": agents,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "contributions": {}
        }
        
        # Send task to all agents
        for agent_id in agents:
            await self.send_message(
                from_agent="coordinator",
                to_agent=agent_id,
                message={
                    "type": "collaboration_task",
                    "task": task,
                    "collaboration_id": f"collab_{int(datetime.now().timestamp())}"
                },
                priority="high"
            )
        
        # Record collaboration
        self.collaboration_history.append(collaboration)
        
        return collaboration
    
    async def route_urgent_message(
        self,
        message: Dict[str, Any],
        target_agent: Optional[str] = None
    ) -> bool:
        """Route urgent message immediately"""
        if target_agent:
            # Route to specific agent
            await self.send_message(
                from_agent=message.get("from", "system"),
                to_agent=target_agent,
                message=message.get("message", {}),
                priority="urgent"
            )
        else:
            # Broadcast to all agents
            for agent_id in self.agent_status.keys():
                await self.send_message(
                    from_agent=message.get("from", "system"),
                    to_agent=agent_id,
                    message=message.get("message", {}),
                    priority="urgent"
                )
        
        return True
    
    def get_collaboration_history(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get collaboration history"""
        if agent_id:
            return [
                c for c in self.collaboration_history
                if agent_id in c.get("agents", [])
            ]
        return self.collaboration_history
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        total_messages = sum(len(q) for q in self.message_queue.values())
        
        return {
            "total_agents": len(self.agent_status),
            "total_messages_queued": total_messages,
            "collaborations": len(self.collaboration_history),
            "agents_with_messages": len(self.message_queue)
        }

