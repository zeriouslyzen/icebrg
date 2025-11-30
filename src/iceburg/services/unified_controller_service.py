"""
Unified Controller Service for ICEBURG

This module exposes a single FastAPI application that combines chat, background
research jobs, control actions, approvals, and live telemetry streaming. It is
intended to back a unified dashboard so that users do not have to juggle
multiple CLIs or ad-hoc interfaces.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Set
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import IceburgConfig
from ..interface import smart_interface_main
from ..enhanced_tracking.thought_camera import ThoughtCamera, ThoughtFrame


@dataclass
class SessionState:
    """Holds state for a single controller session."""

    session_id: str
    config: IceburgConfig
    paused: bool = False
    tasks: Set[asyncio.Task] = field(default_factory=set)
    pending_artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    clients: Set[WebSocket] = field(default_factory=set)


class MessageRequest(BaseModel):
    message: str = Field(..., description="User message or instruction")
    mode: str = Field(default="chat", description="Processing mode to route the message through")
    verbose: bool = Field(default=False, description="Return verbose agent traces")


class JobRequest(BaseModel):
    query: str = Field(..., description="Background research task to launch")
    mode: str = Field(default="standard", description="Processing mode for the research job")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional job metadata for the frontend")


class ControlRequest(BaseModel):
    action: str = Field(..., description="One of pause, resume, cancel")


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="Whether the artifact is approved")
    feedback: Optional[str] = Field(default=None, description="Optional reviewer feedback")


class UnifiedControllerService:
    """Wraps ICEBURG orchestration logic behind a single FastAPI application."""

    def __init__(self) -> None:
        self.app = FastAPI(title="ICEBURG Unified Controller", docs_url=None, redoc_url=None)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.sessions: Dict[str, SessionState] = {}
        self.cfg = IceburgConfig()

        # Thought camera feeds live telemetry into session streams.
        self.thought_camera = ThoughtCamera()
        self.thought_camera.add_thought_callback(self._handle_thought_frame)

        self._register_routes()

        @self.app.on_event("startup")
        async def _startup() -> None:
            await self.thought_camera.start_camera()

        @self.app.on_event("shutdown")
        async def _shutdown() -> None:
            await self.thought_camera.stop_camera()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _create_session(self) -> SessionState:
        session_id = str(uuid4())
        state = SessionState(session_id=session_id, config=self.cfg)
        self.sessions[session_id] = state
        return state

    def _get_session(self, session_id: str) -> SessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:  # pragma: no cover - FastAPI handles this
            raise HTTPException(status_code=404, detail="Session not found") from exc

    async def _broadcast(self, session: SessionState, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", datetime.utcnow().isoformat())
        session.history.append(event)
        stale_clients: Set[WebSocket] = set()
        for client in session.clients:
            try:
                await client.send_json(event)
            except Exception:
                stale_clients.add(client)
        session.clients.difference_update(stale_clients)

    async def _record_artifact(self, session: SessionState, payload: Dict[str, Any]) -> str:
        artifact_id = str(uuid4())
        session.pending_artifacts[artifact_id] = payload
        await self._broadcast(session, {"type": "artifact_pending", "artifact_id": artifact_id, "payload": payload})
        return artifact_id

    async def _run_message(self, session: SessionState, request: MessageRequest) -> None:
        await self._broadcast(session, {"type": "message", "role": "user", "content": request.message})
        try:
            result = await asyncio.to_thread(
                smart_interface_main,
                request.message,
                mode=request.mode,
                verbose=request.verbose,
            )
            await self._broadcast(
                session,
                {
                    "type": "message",
                    "role": "assistant",
                    "mode": request.mode,
                    "content": result,
                },
            )
        except Exception as exc:
            await self._broadcast(
                session,
                {
                    "type": "error",
                    "scope": "message",
                    "detail": str(exc),
                },
            )

    async def _run_job(self, session: SessionState, request: JobRequest) -> None:
        job_id = str(uuid4())
        await self._broadcast(
            session,
            {
                "type": "job_started",
                "job_id": job_id,
                "query": request.query,
                "mode": request.mode,
                "metadata": request.metadata,
            },
        )
        try:
            result = await asyncio.to_thread(
                smart_interface_main,
                request.query,
                mode=request.mode,
                verbose=False,
            )
            artifact_payload = {
                "job_id": job_id,
                "mode": request.mode,
                "result": result,
            }
            await self._record_artifact(session, artifact_payload)
        except Exception as exc:
            await self._broadcast(
                session,
                {
                    "type": "error",
                    "scope": "job",
                    "job_id": job_id,
                    "detail": str(exc),
                },
            )

    def _schedule_task(self, session: SessionState, coro: asyncio.coroutine) -> None:
        task = asyncio.create_task(coro)
        session.tasks.add(task)
        task.add_done_callback(lambda t, s=session: s.tasks.discard(t))

    # ------------------------------------------------------------------
    # Thought camera integration
    # ------------------------------------------------------------------
    def _handle_thought_frame(self, frame: ThoughtFrame) -> None:
        event = {
            "type": "thought",
            "thought_id": frame.thought_id,
            "thought_type": frame.thought_type.value,
            "priority": frame.priority.value,
            "confidence": frame.confidence,
            "content": frame.content,
        }
        asyncio.create_task(self._broadcast_global(event))

    async def _broadcast_global(self, event: Dict[str, Any]) -> None:
        for session in list(self.sessions.values()):
            await self._broadcast(session, dict(event))

    # ------------------------------------------------------------------
    # FastAPI routes
    # ------------------------------------------------------------------
    def _register_routes(self) -> None:
        @self.app.post("/sessions")
        async def create_session() -> Dict[str, Any]:
            state = self._create_session()
            return {"session_id": state.session_id}

        @self.app.get("/sessions/{session_id}/history")
        async def get_history(session_id: str) -> Dict[str, Any]:
            session = self._get_session(session_id)
            return {"events": list(session.history)}

        @self.app.post("/sessions/{session_id}/messages")
        async def post_message(session_id: str, request: MessageRequest) -> Dict[str, Any]:
            session = self._get_session(session_id)
            if session.paused:
                raise HTTPException(status_code=409, detail="Session is paused")
            self._schedule_task(session, self._run_message(session, request))
            return {"status": "accepted"}

        @self.app.post("/sessions/{session_id}/jobs")
        async def post_job(session_id: str, request: JobRequest) -> Dict[str, Any]:
            session = self._get_session(session_id)
            if session.paused:
                raise HTTPException(status_code=409, detail="Session is paused")
            self._schedule_task(session, self._run_job(session, request))
            return {"status": "accepted"}

        @self.app.post("/sessions/{session_id}/control")
        async def control_session(session_id: str, request: ControlRequest) -> Dict[str, Any]:
            session = self._get_session(session_id)
            action = request.action.lower()
            if action == "pause":
                session.paused = True
            elif action == "resume":
                session.paused = False
            elif action == "cancel":
                for task in list(session.tasks):
                    task.cancel()
                session.tasks.clear()
            else:
                raise HTTPException(status_code=400, detail="Unsupported control action")
            await self._broadcast(session, {"type": "control", "action": action})
            return {"status": "ok", "paused": session.paused}

        @self.app.post("/sessions/{session_id}/approvals/{artifact_id}")
        async def approve_artifact(session_id: str, artifact_id: str, request: ApprovalRequest) -> Dict[str, Any]:
            session = self._get_session(session_id)
            artifact = session.pending_artifacts.pop(artifact_id, None)
            if artifact is None:
                raise HTTPException(status_code=404, detail="Artifact not found")
            event = {
                "type": "artifact_reviewed",
                "artifact_id": artifact_id,
                "approved": request.approved,
                "feedback": request.feedback,
                "payload": artifact,
            }
            await self._broadcast(session, event)
            return {"status": "recorded"}

        @self.app.websocket("/sessions/{session_id}/stream")
        async def stream_updates(session_id: str, websocket: WebSocket) -> None:
            session = self._get_session(session_id)
            await websocket.accept()
            session.clients.add(websocket)
            try:
                # Send history snapshot on connect
                for event in list(session.history):
                    await websocket.send_json(event)
                while True:
                    # Keep connection alive
                    await asyncio.sleep(30)
            except WebSocketDisconnect:
                session.clients.discard(websocket)
            except Exception:
                session.clients.discard(websocket)

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"status": "ok", "sessions": len(self.sessions)})


service = UnifiedControllerService()
app = service.app

__all__ = ["app", "service", "UnifiedControllerService"]
