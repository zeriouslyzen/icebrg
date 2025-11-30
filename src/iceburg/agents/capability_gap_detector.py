"""
ICEBURG Capability Gap Detector Agent
Identifies system limitations and capability gaps for autonomous improvement
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..config import IceburgConfig
from ..llm import chat_complete


CAPABILITY_GAP_SYSTEM = (
    "ROLE: System capability analyst who identifies gaps and limitations for autonomous improvement.\n"
    "TASK: Analyze reasoning chains and system outputs to identify capability gaps and improvement opportunities.\n"
    "CONSTRAINTS: Focus on technical limitations, knowledge gaps, and processing constraints. Provide actionable improvement recommendations.\n"
    "OUTPUT_FORMAT: JSON with keys: detected_gaps, improvement_recommendations, priority_levels, implementation_complexity, estimated_benefit."
)


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Safely parse JSON response"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        return json.loads(text)
    except Exception:
        return {}


def _is_valid_schema(d: dict[str, Any]) -> bool:
    """Validate capability gap analysis schema"""
    try:
        if not isinstance(d, dict):
            return False

        required = ["detected_gaps", "improvement_recommendations"]
        return all(key in d for key in required)
    except Exception:
        return False


class CapabilityGapDetector:
    """
    ICEBURG Capability Gap Detector Agent

    Identifies system limitations, capability gaps, and opportunities for autonomous improvement.
    """

    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.detected_gaps = []
        self.improvement_history = []

    def run(self, reasoning_chain: dict[str, Any], initial_query: str, verbose: bool = False) -> dict[str, Any]:
        """Run capability gap detection analysis"""

        try:
            # Analyze reasoning chain for gaps
            gaps_analysis = self._analyze_reasoning_gaps(reasoning_chain, initial_query)

            # Generate improvement recommendations
            recommendations = self._generate_improvements(gaps_analysis)

            # Assess implementation complexity and benefits
            assessment = self._assess_improvements(recommendations)

            return {
                "detected_gaps": gaps_analysis,
                "improvement_recommendations": recommendations,
                "implementation_assessment": assessment,
                "analysis_timestamp": str(os.times()),
                "query_context": initial_query[:200] + "..." if len(initial_query) > 200 else initial_query,
            }

        except Exception as e:
            if verbose:
                print(f"[CAPABILITY_GAP_DETECTOR] Error: {e}")
            return {
                "detected_gaps": [],
                "improvement_recommendations": [],
                "error": str(e),
            }

    def _analyze_reasoning_gaps(self, reasoning_chain: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Analyze reasoning chain for capability gaps"""
        gaps = []

        try:
            # Check for incomplete agent responses
            for agent_name, agent_output in reasoning_chain.items():
                if isinstance(agent_output, dict) and "error" in agent_output:
                    gaps.append({
                        "type": "agent_failure",
                        "agent": agent_name,
                        "description": f"Agent {agent_name} failed to complete analysis",
                        "severity": "high",
                        "impact": "Incomplete reasoning chain"
                    })

            # Check for evidence gaps
            if "scrutineer" in reasoning_chain:
                scrutineer_output = reasoning_chain["scrutineer"]
                if isinstance(scrutineer_output, dict) and "evidence_issues" in scrutineer_output:
                    gaps.append({
                        "type": "evidence_gap",
                        "description": "Insufficient evidence for conclusions",
                        "severity": "medium",
                        "impact": "Reduced confidence in analysis"
                    })

            # Check for knowledge gaps
            if "surveyor" in reasoning_chain:
                surveyor_output = reasoning_chain["surveyor"]
                if isinstance(surveyor_output, dict) and len(surveyor_output.get("sources", [])) < 3:
                    gaps.append({
                        "type": "knowledge_gap",
                        "description": "Limited source material for comprehensive analysis",
                        "severity": "medium",
                        "impact": "Narrower perspective on topic"
                    })

        except Exception:
            gaps.append({
                "type": "analysis_error",
                "description": "Error analyzing reasoning chain for gaps",
                "severity": "low",
                "impact": "Gap detection may be incomplete"
            })

        return gaps

    def _generate_improvements(self, gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate improvement recommendations based on detected gaps"""
        improvements = []

        for gap in gaps:
            if gap["type"] == "agent_failure":
                improvements.append({
                    "gap_type": gap["type"],
                    "recommendation": f"Improve {gap['agent']} agent reliability and error handling",
                    "implementation": "Add retry mechanisms and fallback processing",
                    "priority": "high",
                    "complexity": "medium"
                })

            elif gap["type"] == "evidence_gap":
                improvements.append({
                    "gap_type": gap["type"],
                    "recommendation": "Expand evidence gathering capabilities",
                    "implementation": "Enhance multi-source research integration",
                    "priority": "medium",
                    "complexity": "high"
                })

            elif gap["type"] == "knowledge_gap":
                improvements.append({
                    "gap_type": gap["type"],
                    "recommendation": "Broaden knowledge base integration",
                    "implementation": "Add more academic and research databases",
                    "priority": "medium",
                    "complexity": "medium"
                })

        # Add general system improvements
        improvements.extend([
            {
                "gap_type": "performance",
                "recommendation": "Optimize processing speed for complex queries",
                "implementation": "Implement parallel processing and caching",
                "priority": "medium",
                "complexity": "high"
            },
            {
                "gap_type": "memory",
                "recommendation": "Enhance memory management for large contexts",
                "implementation": "Implement advanced memory pooling and compression",
                "priority": "high",
                "complexity": "high"
            }
        ])

        return improvements

    def _assess_improvements(self, improvements: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess implementation complexity and benefits"""
        high_priority = sum(1 for imp in improvements if imp["priority"] == "high")
        medium_priority = sum(1 for imp in improvements if imp["priority"] == "medium")

        return {
            "total_improvements": len(improvements),
            "priority_breakdown": {
                "high": high_priority,
                "medium": medium_priority,
                "low": len(improvements) - high_priority - medium_priority
            },
            "estimated_implementation_effort": "4-8 weeks" if high_priority > 2 else "2-4 weeks",
            "expected_benefits": [
                "Improved system reliability",
                "Enhanced response quality",
                "Better evidence integration",
                "Optimized performance"
            ]
        }

    def save_analysis(self, analysis: dict[str, Any], filepath: str | None = None) -> None:
        """Save capability gap analysis"""
        if filepath is None:
            filepath = f"data/intelligence/capability_gap_analysis_{int(os.times().elapsed)}.json"

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
        except Exception:
            pass  # Silent fail for analysis saving
