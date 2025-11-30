"""
Advanced Telemetry System
Model Context Protocol (MCP) integration for real-time monitoring
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TelemetryEvent:
    """Telemetry event structure"""
    event_type: str
    timestamp: float
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptMetrics:
    """Prompt performance metrics"""
    prompt_id: str
    prompt_text: str
    response_time: float
    token_count: int
    model_used: str
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


class AdvancedTelemetry:
    """Advanced telemetry system with MCP integration"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/telemetry")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.events: List[TelemetryEvent] = []
        self.prompt_metrics: List[PromptMetrics] = []
        self.trace_logs: List[Dict[str, Any]] = []
        self.version_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_stats = {
            "total_prompts": 0,
            "successful_prompts": 0,
            "failed_prompts": 0,
            "average_response_time": 0.0,
            "total_tokens": 0
        }
        
    def log_event(self, event: TelemetryEvent):
        """Log a telemetry event"""
        self.events.append(event)
        
        # Persist to file
        event_file = self.data_dir / "events.jsonl"
        with open(event_file, "a") as f:
            f.write(json.dumps({
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "task_id": event.task_id,
                "run_id": event.run_id,
                "payload": event.payload,
                "metadata": event.metadata
            }) + "\n")
            
    def track_prompt(self, metrics: PromptMetrics):
        """Track prompt metrics"""
        self.prompt_metrics.append(metrics)
        self.performance_stats["total_prompts"] += 1
        
        if metrics.success:
            self.performance_stats["successful_prompts"] += 1
        else:
            self.performance_stats["failed_prompts"] += 1
            
        self.performance_stats["total_tokens"] += metrics.token_count
        
        # Update average response time
        total_time = sum(m.response_time for m in self.prompt_metrics)
        count = len(self.prompt_metrics)
        if count > 0:
            self.performance_stats["average_response_time"] = total_time / count
            
        # Persist metrics
        metrics_file = self.data_dir / "prompt_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps({
                "prompt_id": metrics.prompt_id,
                "prompt_text": metrics.prompt_text[:200],  # Truncate for privacy
                "response_time": metrics.response_time,
                "token_count": metrics.token_count,
                "model_used": metrics.model_used,
                "success": metrics.success,
                "error_message": metrics.error_message,
                "quality_score": metrics.quality_score,
                "timestamp": time.time()
            }) + "\n")
            
    def log_trace(self, trace_data: Dict[str, Any]):
        """Log trace data for debugging"""
        trace = {
            "timestamp": time.time(),
            **trace_data
        }
        self.trace_logs.append(trace)
        
        # Persist trace
        trace_file = self.data_dir / "traces.jsonl"
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace) + "\n")
            
    def track_version(self, version_data: Dict[str, Any]):
        """Track version control for prompt iterations"""
        version = {
            "timestamp": time.time(),
            **version_data
        }
        self.version_history.append(version)
        
        # Persist version history
        version_file = self.data_dir / "versions.jsonl"
        with open(version_file, "a") as f:
            f.write(json.dumps(version) + "\n")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "success_rate": (
                self.performance_stats["successful_prompts"] / 
                self.performance_stats["total_prompts"]
                if self.performance_stats["total_prompts"] > 0 else 0.0
            ),
            "events_logged": len(self.events),
            "traces_logged": len(self.trace_logs),
            "versions_tracked": len(self.version_history)
        }
        
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent telemetry events"""
        return [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "agent_id": e.agent_id,
                "task_id": e.task_id,
                "payload": e.payload
            }
            for e in self.events[-limit:]
        ]
        
    def get_prompt_analytics(self) -> Dict[str, Any]:
        """Get prompt analytics"""
        if not self.prompt_metrics:
            return {"no_data": True}
            
        successful = [m for m in self.prompt_metrics if m.success]
        failed = [m for m in self.prompt_metrics if not m.success]
        
        return {
            "total_prompts": len(self.prompt_metrics),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.prompt_metrics) if self.prompt_metrics else 0.0,
            "average_response_time": (
                sum(m.response_time for m in successful) / len(successful)
                if successful else 0.0
            ),
            "average_tokens": (
                sum(m.token_count for m in self.prompt_metrics) / len(self.prompt_metrics)
                if self.prompt_metrics else 0
            ),
            "models_used": list(set(m.model_used for m in self.prompt_metrics)),
            "quality_scores": [
                m.quality_score for m in self.prompt_metrics 
                if m.quality_score is not None
            ]
        }

