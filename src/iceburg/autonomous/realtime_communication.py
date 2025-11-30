"""
Realtime Communication
Real-time communication during autonomous operation
"""

from typing import Any, Dict, Optional, List, AsyncGenerator
import asyncio
from datetime import datetime
from collections import deque
from ..services.streaming_handler import StreamingHandler


class RealtimeCommunication:
    """Real-time communication during autonomous operation"""
    
    def __init__(self):
        self.streaming_handler = StreamingHandler()
        self.message_queue: deque = deque(maxlen=1000)
        self.active_connections: Dict[str, bool] = {}
        self.status_updates: Dict[str, Dict[str, Any]] = {}
        self.notifications: List[Dict[str, Any]] = []
    
    async def send_status_update(
        self,
        update_type: str,
        data: Dict[str, Any],
        connection_id: Optional[str] = None
    ) -> bool:
        """Send status update"""
        update = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "connection_id": connection_id
        }
        
        self.status_updates[update_type] = update
        
        # Send to active connections
        if connection_id and connection_id in self.active_connections:
            await self._send_to_connection(connection_id, update)
        
        return True
    
    async def _send_to_connection(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ):
        """Send message to connection"""
        # In production, would use WebSocket or SSE
        # For now, just queue the message
        self.message_queue.append({
            "connection_id": connection_id,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def inject_query(
        self,
        query: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Inject query during autonomous operation"""
        injected_query = {
            "query": query,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "status": "queued"
        }
        
        # Add to message queue with priority
        if priority == "high":
            self.message_queue.appendleft(injected_query)
        else:
            self.message_queue.append(injected_query)
        
        return injected_query
    
    async def get_progress_updates(
        self,
        task_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Get progress updates for task"""
        while True:
            # Check for updates
            update = self.status_updates.get(task_id)
            if update:
                yield update
            
            await asyncio.sleep(1)
    
    def create_notification(
        self,
        notification_type: str,
        message: str,
        severity: str = "info"
    ) -> Dict[str, Any]:
        """Create notification"""
        notification = {
            "type": notification_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }
        
        self.notifications.append(notification)
        return notification
    
    def get_notifications(
        self,
        unread_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get notifications"""
        if unread_only:
            return [n for n in self.notifications if not n.get("read", False)]
        return self.notifications
    
    def mark_notification_read(self, notification_index: int) -> bool:
        """Mark notification as read"""
        if 0 <= notification_index < len(self.notifications):
            self.notifications[notification_index]["read"] = True
            return True
        return False
    
    async def balance_autonomous_and_queries(
        self,
        autonomous_task: callable,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Balance autonomous learning with user queries"""
        result = {
            "autonomous_task": None,
            "query_result": None,
            "balanced": True
        }
        
        # Check if query has high priority
        if query:
            # Process query first
            result["query_result"] = await self._process_query(query)
            result["balanced"] = True
        else:
            # Continue autonomous task
            result["autonomous_task"] = await autonomous_task()
        
        return result
    
    async def _process_query(self, query: str) -> Dict[str, Any]:
        """Process query"""
        # Placeholder for query processing
        return {
            "query": query,
            "result": "Query processed",
            "timestamp": datetime.now().isoformat()
        }
    
    def register_connection(self, connection_id: str) -> bool:
        """Register connection"""
        self.active_connections[connection_id] = True
        return True
    
    def unregister_connection(self, connection_id: str) -> bool:
        """Unregister connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            return True
        return False
    
    def get_active_connections(self) -> List[str]:
        """Get active connections"""
        return list(self.active_connections.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication status"""
        return {
            "active_connections": len(self.active_connections),
            "queued_messages": len(self.message_queue),
            "status_updates": len(self.status_updates),
            "notifications": len(self.notifications),
            "unread_notifications": len([n for n in self.notifications if not n.get("read", False)])
        }

