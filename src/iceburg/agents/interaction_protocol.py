"""
ICEBURG Agent Interaction Protocol

Defines standardized protocol for agent-to-agent communication, coordination,
and data exchange. Ensures consistent, reliable, and efficient agent interactions.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
import logging
from uuid import uuid4

from .linguistic_intelligence import (
    get_linguistic_engine,
    get_metaphor_generator,
    get_anticliche_detector,
    LinguisticStyle
)

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in agent interaction protocol"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """
    Standardized message format for agent-to-agent communication.
    
    All agent interactions use this message format to ensure consistency
    and enable proper routing, logging, and error handling.
    """
    message_id: str = field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Agent identifiers
    sender_id: str = ""
    receiver_id: str = ""
    
    # Message content
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data["message_type"] = self.message_type.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary"""
        data = data.copy()
        data["message_type"] = MessageType(data["message_type"])
        data["priority"] = MessagePriority(data["priority"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """Deserialize message from JSON"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentRequest(AgentMessage):
    """Request message from one agent to another"""
    expected_output_types: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        self.message_type = MessageType.REQUEST


@dataclass
class AgentResponse(AgentMessage):
    """Response message from agent to requester"""
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    engines_used: List[str] = field(default_factory=list)
    algorithms_used: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.message_type = MessageType.RESPONSE


@dataclass
class AgentNotification(AgentMessage):
    """Notification message (one-way communication)"""
    notification_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.message_type = MessageType.NOTIFICATION


@dataclass
class AgentStatusUpdate(AgentMessage):
    """Status update message"""
    status: AgentStatus = AgentStatus.IDLE
    progress: float = 0.0
    current_step: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.message_type = MessageType.STATUS_UPDATE


class AgentInteractionProtocol:
    """
    Standardized protocol for agent-to-agent interactions.
    
    Provides:
    - Message routing and delivery
    - Request/response handling
    - Error handling and retries
    - Status tracking
    - Message queuing
    - Linguistic enhancement
    """
    
    def __init__(self, enable_linguistic_enhancement: bool = True):
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, AgentRequest] = {}
        self.message_queue: List[AgentMessage] = []
        self.agent_statuses: Dict[str, AgentStatus] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history: int = 1000
        self.enable_linguistic_enhancement = enable_linguistic_enhancement
        
        # Initialize linguistic intelligence components
        if self.enable_linguistic_enhancement:
            self.linguistic_engine = get_linguistic_engine()
            self.metaphor_generator = get_metaphor_generator()
            self.anticliche_detector = get_anticliche_detector()
    
    def register_handler(self, action: str, handler: Callable[[AgentMessage], Union[Any, Awaitable[Any]]]):
        """Register a message handler for a specific action"""
        self.message_handlers[action] = handler
        logger.info(f"Registered handler for action: {action}")
    
    def send_request(self, request: AgentRequest) -> str:
        """
        Send a request to another agent.
        
        Returns:
            Request ID for tracking
        """
        # Apply linguistic enhancement if enabled
        if self.enable_linguistic_enhancement:
            request = self._enhance_message(request)
        
        self.pending_requests[request.message_id] = request
        self.message_queue.append(request)
        self.message_history.append(request)
        self._trim_history()
        logger.debug(f"Sent request {request.message_id} from {request.sender_id} to {request.receiver_id}")
        return request.message_id
    
    def send_response(self, response: AgentResponse):
        """Send a response to a request"""
        if response.correlation_id:
            # Remove from pending requests
            self.pending_requests.pop(response.correlation_id, None)
        
        self.message_queue.append(response)
        self.message_history.append(response)
        self._trim_history()
        logger.debug(f"Sent response {response.message_id} from {response.sender_id} to {response.receiver_id}")
    
    def send_notification(self, notification: AgentNotification):
        """Send a notification (one-way message)"""
        self.message_queue.append(notification)
        self.message_history.append(notification)
        self._trim_history()
        logger.debug(f"Sent notification {notification.message_id} from {notification.sender_id} to {notification.receiver_id}")
    
    def send_status_update(self, update: AgentStatusUpdate):
        """Send a status update"""
        self.agent_statuses[update.sender_id] = update.status
        self.message_queue.append(update)
        self.message_history.append(update)
        self._trim_history()
        logger.debug(f"Status update from {update.sender_id}: {update.status.value}")
    
    def get_pending_request(self, request_id: str) -> Optional[AgentRequest]:
        """Get a pending request by ID"""
        return self.pending_requests.get(request_id)
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get current status of an agent"""
        return self.agent_statuses.get(agent_id)
    
    def process_message(self, message: AgentMessage) -> Optional[AgentResponse]:
        """
        Process a message and return response if applicable.
        
        This is the main message processing function that routes messages
        to appropriate handlers.
        """
        try:
            if message.message_type == MessageType.REQUEST:
                return self._handle_request(message)
            elif message.message_type == MessageType.RESPONSE:
                self._handle_response(message)
            elif message.message_type == MessageType.NOTIFICATION:
                self._handle_notification(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                self._handle_status_update(message)
            elif message.message_type == MessageType.ERROR:
                self._handle_error(message)
            elif message.message_type == MessageType.HEARTBEAT:
                self._handle_heartbeat(message)
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}", exc_info=True)
            return self._create_error_response(message, str(e))
        
        return None
    
    def _handle_request(self, request: AgentRequest) -> Optional[AgentResponse]:
        """Handle an incoming request"""
        handler = self.message_handlers.get(request.action)
        if not handler:
            logger.warning(f"No handler registered for action: {request.action}")
            return self._create_error_response(request, f"No handler for action: {request.action}")
        
        try:
            # Update agent status
            self.agent_statuses[request.receiver_id] = AgentStatus.PROCESSING
            
            # Call handler
            if isinstance(handler, Callable):
                result = handler(request)
                # Handle async handlers
                if hasattr(result, '__await__'):
                    # This is an async handler, but we're in sync context
                    # In real implementation, this would use asyncio
                    logger.warning("Async handler called in sync context")
                    result = None
            else:
                result = None
            
            # Create response
            response = AgentResponse(
                message_id=str(uuid4()),
                sender_id=request.receiver_id,
                receiver_id=request.sender_id,
                correlation_id=request.message_id,
                success=True,
                result=result,
                payload=request.payload,
                context=request.context
            )
            
            # Update status
            self.agent_statuses[request.receiver_id] = AgentStatus.COMPLETED
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling request {request.message_id}: {e}", exc_info=True)
            self.agent_statuses[request.receiver_id] = AgentStatus.FAILED
            return self._create_error_response(request, str(e))
    
    def _handle_response(self, response: AgentResponse):
        """Handle an incoming response"""
        # Response handling is typically done by the requester
        # This is a placeholder for logging and tracking
        logger.debug(f"Received response {response.message_id} for request {response.correlation_id}")
    
    def _handle_notification(self, notification: AgentNotification):
        """Handle an incoming notification"""
        logger.debug(f"Received notification {notification.message_id}: {notification.notification_type}")
    
    def _handle_status_update(self, update: AgentStatusUpdate):
        """Handle a status update"""
        self.agent_statuses[update.sender_id] = update.status
        logger.debug(f"Status update from {update.sender_id}: {update.status.value} ({update.progress:.1%})")
    
    def _handle_error(self, message: AgentMessage):
        """Handle an error message"""
        logger.error(f"Error message from {message.sender_id}: {message.payload.get('error', 'Unknown error')}")
        if message.sender_id in self.agent_statuses:
            self.agent_statuses[message.sender_id] = AgentStatus.FAILED
    
    def _handle_heartbeat(self, message: AgentMessage):
        """Handle a heartbeat message"""
        logger.debug(f"Heartbeat from {message.sender_id}")
    
    def _create_error_response(self, request: AgentRequest, error: str) -> AgentResponse:
        """Create an error response for a request"""
        return AgentResponse(
            message_id=str(uuid4()),
            sender_id=request.receiver_id,
            receiver_id=request.sender_id,
            correlation_id=request.message_id,
            success=False,
            error=error,
            payload=request.payload,
            context=request.context
        )
    
    def _trim_history(self):
        """Trim message history to max size"""
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
    
    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        """Get message history, optionally filtered by agent"""
        history = self.message_history
        if agent_id:
            history = [msg for msg in history if msg.sender_id == agent_id or msg.receiver_id == agent_id]
        return history[-limit:]
    
    def clear_queue(self):
        """Clear the message queue"""
        self.message_queue.clear()
        logger.info("Message queue cleared")
    
    def get_queue_size(self) -> int:
        """Get current message queue size"""
        return len(self.message_queue)
    
    def _enhance_message(self, message: AgentMessage) -> AgentMessage:
        """
        Apply linguistic enhancement to message.
        
        Args:
            message: Message to enhance
            
        Returns:
            Enhanced message
        """
        if not self.enable_linguistic_enhancement:
            return message
        
        # Enhance action description
        if hasattr(message, 'action') and message.action:
            enhanced_action = self._enhance_text(message.action)
            message.action = enhanced_action
        
        # Enhance payload text fields
        if hasattr(message, 'payload') and isinstance(message.payload, dict):
            enhanced_payload = {}
            for key, value in message.payload.items():
                if isinstance(value, str) and len(value) > 20:
                    # Enhance longer text fields
                    enhanced_value = self._enhance_text(value)
                    enhanced_payload[key] = enhanced_value
                else:
                    enhanced_payload[key] = value
            message.payload = enhanced_payload
        
        # Enhance context text fields
        if hasattr(message, 'context') and isinstance(message.context, dict):
            enhanced_context = {}
            for key, value in message.context.items():
                if isinstance(value, str) and len(value) > 20:
                    enhanced_value = self._enhance_text(value)
                    enhanced_context[key] = enhanced_value
                else:
                    enhanced_context[key] = value
            message.context = enhanced_context
        
        return message
    
    def _enhance_text(self, text: str) -> str:
        """
        Enhance text using linguistic intelligence.
        
        Args:
            text: Text to enhance
            
        Returns:
            Enhanced text
        """
        if not text or len(text) < 10:
            return text
        
        try:
            # Apply linguistic enhancement
            enhancement = self.linguistic_engine.enhance_text(
                text,
                style=LinguisticStyle.INTELLIGENT,
                verbosity_reduction=0.2,
                power_enhancement=0.5
            )
            
            # Apply anti-cliche detection and replacement
            enhanced_text, _ = self.anticliche_detector.detect_and_replace(
                enhancement.enhanced_text
            )
            
            return enhanced_text
        except Exception as e:
            logger.warning(f"Error enhancing text: {e}")
            return text


# Global protocol instance
_protocol: Optional[AgentInteractionProtocol] = None


def get_protocol() -> AgentInteractionProtocol:
    """Get or create global agent interaction protocol"""
    global _protocol
    if _protocol is None:
        _protocol = AgentInteractionProtocol()
    return _protocol


def create_request(sender_id: str, receiver_id: str, action: str, 
                   payload: Dict[str, Any] = None, 
                   context: Dict[str, Any] = None,
                   priority: MessagePriority = MessagePriority.NORMAL) -> AgentRequest:
    """Helper function to create a request message"""
    return AgentRequest(
        sender_id=sender_id,
        receiver_id=receiver_id,
        action=action,
        payload=payload or {},
        context=context or {},
        priority=priority
    )


def create_response(request: AgentRequest, result: Any = None, 
                   success: bool = True, error: Optional[str] = None) -> AgentResponse:
    """Helper function to create a response message"""
    return AgentResponse(
        sender_id=request.receiver_id,
        receiver_id=request.sender_id,
        correlation_id=request.message_id,
        success=success,
        result=result,
        error=error,
        payload=request.payload,
        context=request.context
    )


def create_notification(sender_id: str, receiver_id: str, 
                       notification_type: str,
                       data: Dict[str, Any] = None) -> AgentNotification:
    """Helper function to create a notification message"""
    return AgentNotification(
        sender_id=sender_id,
        receiver_id=receiver_id,
        notification_type=notification_type,
        data=data or {}
    )


def create_status_update(agent_id: str, status: AgentStatus,
                         progress: float = 0.0, current_step: str = "",
                         details: Dict[str, Any] = None) -> AgentStatusUpdate:
    """Helper function to create a status update"""
    return AgentStatusUpdate(
        sender_id=agent_id,
        status=status,
        progress=progress,
        current_step=current_step,
        details=details or {}
    )

