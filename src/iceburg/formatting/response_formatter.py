"""
Response Formatter
Structured response formatting with thinking, informatics, and conclusions
"""

from typing import Any, Dict, Optional, List
import json


class ResponseFormatter:
    """Formats responses with structured thinking, informatics, and conclusions"""
    
    def __init__(self):
        pass
    
    def format_response(
        self,
        content: str,
        thinking: Optional[List[str]] = None,
        informatics: Optional[Dict[str, Any]] = None,
        conclusions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Format response with structured components"""
        response = {
            "content": content,
            "thinking": thinking or [],
            "informatics": informatics or {},
            "conclusions": conclusions or []
        }
        
        return response
    
    def format_from_analysis(
        self,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format response from analysis result"""
        thinking = []
        informatics = {}
        conclusions = []
        
        # Extract thinking from analysis
        if "thinking" in analysis_result:
            thinking = analysis_result["thinking"]
        elif "steps" in analysis_result:
            thinking = [f"Step {i+1}: {step}" for i, step in enumerate(analysis_result["steps"])]
        
        # Extract informatics
        if "informatics" in analysis_result:
            informatics = analysis_result["informatics"]
        else:
            # Extract from analysis result
            informatics = {
                "sources": analysis_result.get("sources", []),
                "confidence": analysis_result.get("confidence", 0.0),
                "processing_time": analysis_result.get("processing_time", 0.0)
            }
        
        # Extract conclusions
        if "conclusions" in analysis_result:
            conclusions = analysis_result["conclusions"]
        elif "summary" in analysis_result:
            conclusions = [analysis_result["summary"]]
        elif "insights" in analysis_result:
            conclusions = analysis_result["insights"]
        
        # Extract main content
        content = analysis_result.get("result", analysis_result.get("content", ""))
        
        return self.format_response(
            content=content,
            thinking=thinking,
            informatics=informatics,
            conclusions=conclusions
        )
    
    def format_suppression_detection(
        self,
        suppression_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format suppression detection result"""
        thinking = []
        informatics = {}
        conclusions = []
        
        # Extract thinking from steps
        steps = [
            "Classification delay detection",
            "Narrative rewriting identification",
            "Publication bottleneck analysis",
            "Funding misdirection patterns",
            "Timeline gap analysis",
            "Contradiction amplification",
            "Recovery evidence validation"
        ]
        
        for i, step_name in enumerate(steps, 1):
            step_key = f"step{i}_{step_name.lower().replace(' ', '_')}"
            step_result = suppression_result.get(step_key, {})
            
            if step_result.get("detected", False):
                thinking.append(f"Step {i}: {step_name} - Detected")
            else:
                thinking.append(f"Step {i}: {step_name} - Not detected")
        
        # Extract informatics
        informatics = {
            "suppression_score": suppression_result.get("overall_suppression_score", 0.0),
            "suppression_detected": suppression_result.get("suppression_detected", False),
            "steps_analyzed": 7,
            "steps_detected": sum(
                1 for i in range(1, 8)
                if suppression_result.get(f"step{i}_", {}).get("detected", False)
            )
        }
        
        # Extract conclusions
        if suppression_result.get("suppression_detected"):
            conclusions.append("Suppression patterns detected in analyzed documents")
            conclusions.append(f"Overall suppression score: {suppression_result.get('overall_suppression_score', 0.0):.2f}")
        else:
            conclusions.append("No significant suppression patterns detected")
        
        # Main content
        content = "Suppression detection analysis completed"
        
        return self.format_response(
            content=content,
            thinking=thinking,
            informatics=informatics,
            conclusions=conclusions
        )
    
    def to_json(self, response: Dict[str, Any]) -> str:
        """Convert response to JSON"""
        return json.dumps(response, indent=2)
    
    def to_markdown(self, response: Dict[str, Any]) -> str:
        """Convert response to markdown"""
        markdown = []
        
        # Main content
        markdown.append(response["content"])
        markdown.append("")
        
        # Thinking
        if response["thinking"]:
            markdown.append("## Thinking")
            for thought in response["thinking"]:
                markdown.append(f"- {thought}")
            markdown.append("")
        
        # Informatics
        if response["informatics"]:
            markdown.append("## Informatics")
            for key, value in response["informatics"].items():
                markdown.append(f"- **{key}**: {value}")
            markdown.append("")
        
        # Conclusions
        if response["conclusions"]:
            markdown.append("## Conclusions")
            for conclusion in response["conclusions"]:
                markdown.append(f"- {conclusion}")
            markdown.append("")
        
        return "\n".join(markdown)

