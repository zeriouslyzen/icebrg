"""
ICEBURG Workflow Events
=======================

Helper module for emitting structured SSE events during agent execution.
These events power the frontend Agent Workflow Pill visualization.

Event Types:
- agent_start: When an agent begins processing
- agent_complete: When an agent finishes
- metacog_result: When metacognition checks complete
- workflow_complete: When the entire pipeline finishes
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    METACOG_RESULT = "metacog_result"
    WORKFLOW_COMPLETE = "workflow_complete"
    STEP_COMPLETE = "step_complete"  # Interactive step card with findings and route options


@dataclass
class AgentStartEvent:
    """Emitted when an agent begins processing."""
    agent: str
    index: int
    total: int
    description: str = ""
    
    def to_sse(self) -> str:
        return json.dumps({
            "type": EventType.AGENT_START.value,
            "agent": self.agent,
            "index": self.index,
            "total": self.total,
            "description": self.description,
            "timestamp": time.time()
        })


@dataclass 
class AgentCompleteEvent:
    """Emitted when an agent finishes processing."""
    agent: str
    duration_ms: float
    success: bool = True
    summary: str = ""
    
    def to_sse(self) -> str:
        return json.dumps({
            "type": EventType.AGENT_COMPLETE.value,
            "agent": self.agent,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "summary": self.summary,
            "timestamp": time.time()
        })


@dataclass
class MetacogResultEvent:
    """Emitted when metacognition checks complete."""
    alignment_score: float
    contradiction_count: int
    complexity: str  # "low", "medium", "high"
    quarantined: int = 0
    
    def to_sse(self) -> str:
        return json.dumps({
            "type": EventType.METACOG_RESULT.value,
            "alignment_score": self.alignment_score,
            "contradiction_count": self.contradiction_count,
            "complexity": self.complexity,
            "quarantined": self.quarantined,
            "timestamp": time.time()
        })


@dataclass
class WorkflowCompleteEvent:
    """Emitted when the entire workflow finishes."""
    total_duration_ms: float
    agents_run: List[str]
    mode: str
    
    def to_sse(self) -> str:
        return json.dumps({
            "type": EventType.WORKFLOW_COMPLETE.value,
            "total_duration_ms": self.total_duration_ms,
            "agents_run": self.agents_run,
            "mode": self.mode,
            "timestamp": time.time()
        })


@dataclass
class StepCompleteEvent:
    """
    Emitted when a workflow step completes - triggers interactive step card UI.
    Shows findings, time taken, and route options for user to choose next action.
    """
    step: str  # e.g. "surveyor", "deliberation", "synthesist"
    findings: List[str]  # Bullet points of what was discovered
    time_taken: float  # Seconds
    suggested_next: List[str]  # Suggested next actions
    options: List[Dict[str, str]]  # Route buttons: [{"action": "deep_dive", "label": "Deep Dive", "estimated_time": "30s"}]
    
    def to_sse(self) -> str:
        return json.dumps({
            "type": EventType.STEP_COMPLETE.value,
            "step": self.step,
            "report": {
                "findings": self.findings,
                "time_taken": self.time_taken,
                "suggested_next": self.suggested_next
            },
            "options": self.options,
            "timestamp": time.time()
        })


class WorkflowEventEmitter:
    """
    Manages workflow events for a single request.
    
    Usage:
        emitter = WorkflowEventEmitter(mode="research", agents=["surveyor", "deliberation", "synthesist"])
        
        # In your SSE generator:
        yield emitter.start_agent("surveyor")
        # ... run surveyor ...
        yield emitter.complete_agent("surveyor", duration_ms=1200)
        
        yield emitter.metacog_result(alignment=0.85, contradictions=1, complexity="medium")
        
        yield emitter.workflow_complete(total_ms=5000)
    """
    
    def __init__(self, mode: str, agents: List[str]):
        self.mode = mode
        self.agents = agents
        self.agents_run: List[str] = []
        self.start_time = time.time()
        
    def start_agent(self, agent: str, description: str = "") -> str:
        """Emit agent_start event and return SSE-formatted string."""
        index = self.agents.index(agent) if agent in self.agents else len(self.agents_run)
        event = AgentStartEvent(
            agent=agent,
            index=index,
            total=len(self.agents),
            description=description or f"Running {agent}..."
        )
        return f"data: {event.to_sse()}\n\n"
    
    def complete_agent(self, agent: str, duration_ms: float, success: bool = True, summary: str = "") -> str:
        """Emit agent_complete event and return SSE-formatted string."""
        self.agents_run.append(agent)
        event = AgentCompleteEvent(
            agent=agent,
            duration_ms=duration_ms,
            success=success,
            summary=summary
        )
        return f"data: {event.to_sse()}\n\n"
    
    def metacog_result(self, alignment: float, contradictions: int, complexity: str, quarantined: int = 0) -> str:
        """Emit metacog_result event and return SSE-formatted string."""
        event = MetacogResultEvent(
            alignment_score=alignment,
            contradiction_count=contradictions,
            complexity=complexity,
            quarantined=quarantined
        )
        return f"data: {event.to_sse()}\n\n"
    
    def workflow_complete(self, total_ms: Optional[float] = None) -> str:
        """Emit workflow_complete event and return SSE-formatted string."""
        if total_ms is None:
            total_ms = (time.time() - self.start_time) * 1000
        event = WorkflowCompleteEvent(
            total_duration_ms=total_ms,
            agents_run=self.agents_run,
            mode=self.mode
        )
        return f"data: {event.to_sse()}\n\n"
    
    def step_complete(
        self, 
        step: str, 
        findings: List[str], 
        time_taken: float,
        suggested_next: Optional[List[str]] = None,
        options: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Emit step_complete event for interactive step card UI.
        
        Args:
            step: Agent/step name (e.g., "surveyor", "deliberation")
            findings: List of bullet point findings
            time_taken: Time in seconds
            suggested_next: Suggested next actions
            options: Route buttons [{"action": "...", "label": "...", "estimated_time": "..."}]
        
        Returns:
            SSE-formatted string for the event
        """
        if suggested_next is None:
            suggested_next = ["continue"]
        
        if options is None:
            # Default options for each step type
            default_options = {
                "surveyor": [
                    {"action": "deep_dive", "label": "Deep Research", "estimated_time": "30s"},
                    {"action": "challenge", "label": "Challenge Findings", "estimated_time": "20s"},
                    {"action": "skip", "label": "Continue", "estimated_time": ""}
                ],
                "deliberation": [
                    {"action": "expand", "label": "Expand Analysis", "estimated_time": "25s"},
                    {"action": "verify", "label": "Verify Sources", "estimated_time": "15s"},
                    {"action": "skip", "label": "Continue", "estimated_time": ""}
                ],
                "dissident": [
                    {"action": "counter", "label": "Counter Arguments", "estimated_time": "20s"},
                    {"action": "accept", "label": "Accept Critique", "estimated_time": ""},
                    {"action": "skip", "label": "Continue", "estimated_time": ""}
                ],
                "synthesist": [
                    {"action": "refine", "label": "Refine Synthesis", "estimated_time": "20s"},
                    {"action": "skip", "label": "Finalize", "estimated_time": ""}
                ],
                "oracle": [
                    {"action": "principles", "label": "Extract Principles", "estimated_time": "15s"},
                    {"action": "skip", "label": "Complete", "estimated_time": ""}
                ]
            }
            options = default_options.get(step.lower(), [
                {"action": "continue", "label": "Continue", "estimated_time": ""},
                {"action": "skip", "label": "Skip", "estimated_time": ""}
            ])
        
        self.agents_run.append(step)
        event = StepCompleteEvent(
            step=step,
            findings=findings,
            time_taken=time_taken,
            suggested_next=suggested_next,
            options=options
        )
        return f"data: {event.to_sse()}\n\n"


# Mode Templates
# These define which agents run for each processing mode
MODE_TEMPLATES = {
    "fast": {
        "agents": ["secretary"],
        "metacognition": False,
        "description": "Quick chat responses"
    },
    "research": {
        "agents": ["surveyor", "deliberation", "synthesist"],
        "metacognition": True,
        "description": "Standard research with metacognition"
    },
    "deep_research": {
        "agents": ["surveyor", "dissident", "deliberation", "archaeologist", "synthesist", "oracle"],
        "metacognition": True,
        "description": "Comprehensive multi-perspective analysis"
    },
    "unbounded": {
        "agents": ["surveyor", "dissident", "deliberation", "archaeologist", "synthesist", "oracle", "self_redesign"],
        "metacognition": True,
        "self_modification": True,
        "description": "AGI mode with self-modification"
    }
}


def get_mode_template(mode: str) -> Dict[str, Any]:
    """Get the configuration for a processing mode."""
    return MODE_TEMPLATES.get(mode, MODE_TEMPLATES["research"])


def create_emitter_for_mode(mode: str) -> WorkflowEventEmitter:
    """Create a WorkflowEventEmitter configured for the given mode."""
    template = get_mode_template(mode)
    return WorkflowEventEmitter(mode=mode, agents=template["agents"])
