"""
Live Editor for Visual Generation
Enables real-time editing of generated UIs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pathlib import Path
import json
import asyncio
import websockets
import threading
import time
from datetime import datetime
import hashlib

from ..iir.visual_tsl import UISpec
from ..iir.visual_ir import VisualIRFunction
from ..iir.backends import BackendType
from ..storage.visual_storage import VisualArtifactStorage


class EditType(Enum):
    """Types of edits that can be made"""
    COMPONENT_ADD = "component_add"
    COMPONENT_REMOVE = "component_remove"
    COMPONENT_MODIFY = "component_modify"
    STYLE_CHANGE = "style_change"
    LAYOUT_CHANGE = "layout_change"
    TEXT_CHANGE = "text_change"
    PROPERTY_CHANGE = "property_change"


@dataclass
class EditOperation:
    """Represents an edit operation"""
    edit_type: EditType
    component_id: Optional[str]
    property_name: Optional[str]
    old_value: Any
    new_value: Any
    timestamp: datetime
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edit_type": self.edit_type.value,
            "component_id": self.component_id,
            "property_name": self.property_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id
        }


@dataclass
class LiveEditSession:
    """Represents a live editing session"""
    session_id: str
    artifact_id: str
    project_id: str
    connected_clients: List[str]
    edit_history: List[EditOperation]
    current_spec: UISpec
    current_ir: VisualIRFunction
    last_modified: datetime
    
    def add_edit(self, edit: EditOperation):
        """Add an edit to the session history"""
        self.edit_history.append(edit)
        self.last_modified = datetime.now()


class LiveEditor:
    """Live editor for real-time UI editing"""
    
    def __init__(self, storage_dir: Path = None):
        self.storage = VisualArtifactStorage(storage_dir)
        self.active_sessions: Dict[str, LiveEditSession] = {}
        self.websocket_server = None
        self.is_running = False
        
        # Callbacks for edit events
        self.on_edit_callbacks: List[Callable[[EditOperation], None]] = []
        self.on_save_callbacks: List[Callable[[str], None]] = []
    
    def start_live_editing(self, artifact_id: str, project_id: str = "default", port: int = 8765) -> str:
        """Start live editing session for an artifact"""
        try:
            # Load artifact
            artifact = self.storage.load_artifact(artifact_id, project_id)
            if not artifact:
                raise ValueError(f"Artifact {artifact_id} not found")
            
            # Create session
            session_id = f"session_{artifact_id}_{int(time.time())}"
            session = LiveEditSession(
                session_id=session_id,
                artifact_id=artifact_id,
                project_id=project_id,
                connected_clients=[],
                edit_history=[],
                current_spec=artifact.get("spec"),
                current_ir=artifact.get("ir"),
                last_modified=datetime.now()
            )
            
            self.active_sessions[session_id] = session
            
            # Start WebSocket server
            self._start_websocket_server(port)
            
            return session_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start live editing: {str(e)}")
    
    def stop_live_editing(self, session_id: str):
        """Stop live editing session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if not self.active_sessions and self.websocket_server:
            self.websocket_server.close()
            self.is_running = False
    
    def apply_edit(self, session_id: str, edit: EditOperation) -> bool:
        """Apply an edit to a live session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        try:
            # Apply edit to current IR
            success = self._apply_edit_to_ir(session.current_ir, edit)
            
            if success:
                # Add to history
                session.add_edit(edit)
                
                # Notify callbacks
                for callback in self.on_edit_callbacks:
                    callback(edit)
                
                # Broadcast to connected clients
                self._broadcast_edit(session_id, edit)
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def save_session(self, session_id: str) -> bool:
        """Save current session state"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        try:
            # Recompile artifacts with current IR
            from ..agents.visual_architect import VisualArchitect
            from ..config import load_config
            
            architect = VisualArchitect()
            cfg = load_config()
            
            # Create new result with updated IR
            updated_result = architect._compile_artifacts(session.current_ir, verbose=False)
            
            # Save updated artifacts
            self.storage.store_visual_generation(updated_result, session.project_id)
            
            # Notify save callbacks
            for callback in self.on_save_callbacks:
                callback(session_id)
            
            return True
            
        except Exception as e:
            return False
    
    def get_edit_history(self, session_id: str) -> List[EditOperation]:
        """Get edit history for a session"""
        if session_id not in self.active_sessions:
            return []
        
        return self.active_sessions[session_id].edit_history
    
    def undo_last_edit(self, session_id: str) -> bool:
        """Undo the last edit in a session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if not session.edit_history:
            return False
        
        # Get last edit
        last_edit = session.edit_history[-1]
        
        # Create reverse edit
        reverse_edit = EditOperation(
            edit_type=last_edit.edit_type,
            component_id=last_edit.component_id,
            property_name=last_edit.property_name,
            old_value=last_edit.new_value,
            new_value=last_edit.old_value,
            timestamp=datetime.now(),
            user_id=last_edit.user_id
        )
        
        # Apply reverse edit
        success = self.apply_edit(session_id, reverse_edit)
        
        if success:
            # Remove from history
            session.edit_history.pop()
        
        return success
    
    def _apply_edit_to_ir(self, visual_ir: VisualIRFunction, edit: EditOperation) -> bool:
        """Apply an edit operation to the visual IR"""
        try:
            if edit.edit_type == EditType.TEXT_CHANGE:
                # Change text content of a component
                if edit.component_id and edit.property_name == "text":
                    component = visual_ir.get_component(edit.component_id)
                    if component and "text" in component.props:
                        component.props["text"] = edit.new_value
                        return True
            
            elif edit.edit_type == EditType.STYLE_CHANGE:
                # Change style properties
                if edit.component_id and edit.property_name:
                    component = visual_ir.get_component(edit.component_id)
                    if component and component.style_ref:
                        style = visual_ir.style_graph.get_style(component.style_ref)
                        if style:
                            # Update style property
                            from ..iir.visual_ir import StyleProperty
                            from ..iir.ir import IRValue, ScalarType
                            
                            try:
                                prop = StyleProperty(edit.property_name)
                                style.properties[prop] = IRValue(edit.new_value, ScalarType("string"))
                                return True
                            except ValueError:
                                return False
            
            elif edit.edit_type == EditType.PROPERTY_CHANGE:
                # Change component properties
                if edit.component_id and edit.property_name:
                    component = visual_ir.get_component(edit.component_id)
                    if component:
                        from ..iir.ir import IRValue, ScalarType
                        component.props[edit.property_name] = IRValue(edit.new_value, ScalarType("string"))
                        return True
            
            elif edit.edit_type == EditType.COMPONENT_ADD:
                # Add new component
                from ..iir.visual_ir import ComponentNode
                from ..iir.visual_tsl import ComponentType
                from ..iir.ir import IRValue, ScalarType
                
                new_component = ComponentNode(
                    id=edit.new_value.get("id", f"comp_{len(visual_ir.ui_components)}"),
                    type=ComponentType(edit.new_value.get("type", "div")),
                    props={k: IRValue(v, ScalarType("string")) for k, v in edit.new_value.get("props", {}).items()}
                )
                
                visual_ir.add_component(new_component)
                return True
            
            elif edit.edit_type == EditType.COMPONENT_REMOVE:
                # Remove component
                if edit.component_id:
                    visual_ir.remove_component(edit.component_id)
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _start_websocket_server(self, port: int):
        """Start WebSocket server for real-time collaboration"""
        async def handle_client(websocket, path):
            """Handle WebSocket client connection"""
            session_id = None
            
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("type") == "join_session":
                        session_id = data.get("session_id")
                        if session_id in self.active_sessions:
                            self.active_sessions[session_id].connected_clients.append(str(websocket))
                            await websocket.send(json.dumps({
                                "type": "session_joined",
                                "session_id": session_id
                            }))
                    
                    elif data.get("type") == "edit":
                        if session_id:
                            edit_data = data.get("edit")
                            edit = EditOperation(
                                edit_type=EditType(edit_data["edit_type"]),
                                component_id=edit_data.get("component_id"),
                                property_name=edit_data.get("property_name"),
                                old_value=edit_data.get("old_value"),
                                new_value=edit_data.get("new_value"),
                                timestamp=datetime.fromisoformat(edit_data["timestamp"]),
                                user_id=edit_data.get("user_id")
                            )
                            
                            success = self.apply_edit(session_id, edit)
                            await websocket.send(json.dumps({
                                "type": "edit_result",
                                "success": success,
                                "edit_id": edit_data.get("id")
                            }))
                    
                    elif data.get("type") == "save":
                        if session_id:
                            success = self.save_session(session_id)
                            await websocket.send(json.dumps({
                                "type": "save_result",
                                "success": success
                            }))
            
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if session_id and session_id in self.active_sessions:
                    self.active_sessions[session_id].connected_clients = [
                        client for client in self.active_sessions[session_id].connected_clients
                        if client != str(websocket)
                    ]
        
        async def start_server():
            self.websocket_server = await websockets.serve(handle_client, "os.getenv("HOST", "localhost")", port)
            self.is_running = True
            await self.websocket_server.wait_closed()
        
        # Start server in background thread
        def run_server():
            asyncio.run(start_server())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def _broadcast_edit(self, session_id: str, edit: EditOperation):
        """Broadcast edit to all connected clients"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # This would broadcast to WebSocket clients
        # Implementation depends on WebSocket server setup
        pass
    
    def add_edit_callback(self, callback: Callable[[EditOperation], None]):
        """Add callback for edit events"""
        self.on_edit_callbacks.append(callback)
    
    def add_save_callback(self, callback: Callable[[str], None]):
        """Add callback for save events"""
        self.on_save_callbacks.append(callback)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a live editing session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "artifact_id": session.artifact_id,
            "project_id": session.project_id,
            "connected_clients": len(session.connected_clients),
            "edit_count": len(session.edit_history),
            "last_modified": session.last_modified.isoformat()
        }


def start_live_editing(artifact_id: str, project_id: str = "default", port: int = 8765) -> str:
    """Start live editing session for a visual artifact"""
    editor = LiveEditor()
    return editor.start_live_editing(artifact_id, project_id, port)
