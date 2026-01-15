"""
WebSocket Manager
Manages WebSocket connections for the ICEBURG API
"""

from typing import Dict, Any, Optional, List
from fastapi import WebSocket
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and state"""
    
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self.connection_by_id: Dict[str, WebSocket] = {}
    
    async def register_connection(self, websocket: WebSocket, conversation_id: str) -> Dict[str, Any]:
        """Register a new WebSocket connection"""
        conn_info = {
            "conversation_id": conversation_id,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "is_connected": False,
            "ping_count": 0,
            "pong_count": 0,
        }
        self.active_connections[websocket] = conn_info
        self.connection_by_id[conversation_id] = websocket
        logger.info(f"WebSocket registered: {conversation_id}")
        return conn_info
    
    async def mark_connected(self, websocket: WebSocket):
        """Mark a connection as fully connected"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["is_connected"] = True
    
    async def mark_disconnected(self, websocket: WebSocket):
        """Mark a connection as disconnected"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["is_connected"] = False
    
    async def remove_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            conn_info = self.active_connections[websocket]
            conv_id = conn_info.get("conversation_id")
            if conv_id and conv_id in self.connection_by_id:
                del self.connection_by_id[conv_id]
            del self.active_connections[websocket]
            logger.info(f"WebSocket removed: {conv_id}")
    
    async def record_ping(self, websocket: WebSocket):
        """Record a ping from client"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["ping_count"] += 1
            self.active_connections[websocket]["last_activity"] = datetime.utcnow()
    
    async def record_pong(self, websocket: WebSocket):
        """Record a pong from client"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["pong_count"] += 1
            self.active_connections[websocket]["last_activity"] = datetime.utcnow()
    
    async def update_activity(self, websocket: WebSocket):
        """Update last activity timestamp"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["last_activity"] = datetime.utcnow()
    
    async def get_connection_health(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get health info for a connection"""
        if websocket in self.active_connections:
            info = self.active_connections[websocket]
            return {
                "is_connected": info.get("is_connected", False),
                "ping_count": info.get("ping_count", 0),
                "pong_count": info.get("pong_count", 0),
                "last_activity": info.get("last_activity"),
            }
        return {"is_connected": False}
    
    async def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get all active connections info"""
        return [
            {
                "conversation_id": info.get("conversation_id"),
                "is_connected": info.get("is_connected", False),
                "connected_at": str(info.get("connected_at")),
                "last_activity": str(info.get("last_activity")),
            }
            for info in self.active_connections.values()
        ]
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific connection"""
        await websocket.send_json(message)
        await self.update_activity(websocket)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Singleton instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the singleton WebSocket manager instance"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager

