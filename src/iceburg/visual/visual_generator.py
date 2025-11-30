from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class VisualElement:
    """Represents a visual element (figure, table, panel)"""
    element_type: str
    title: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime


class VisualGenerator:
    """Generates scientific visualizations and figures"""
    
    def __init__(self):
        self.visual_counter = 0
        self.available_types = [
            "figure", "table", "panel", "caption", "diagram"
        ]
    
    def generate_figure(self, title: str, data: Dict[str, Any], 
                       figure_type: str = "general") -> VisualElement:
        """Generate a scientific figure"""
        
        figure_id = self._generate_visual_id()
        
        # Create figure structure
        figure_content = {
            "figure_id": figure_id,
            "title": title,
            "type": figure_type,
            "panels": [],
            "data_source": data.get("source", "generated"),
            "creation_method": "AI-generated",
            "recommended_format": "PNG/SVG for publication"
        }
        
        # Add panels if data contains multiple components
        if "components" in data:
            for i, component in enumerate(data["components"]):
                panel = self._create_panel(f"Panel {chr(65+i)}", component)
                figure_content["panels"].append(panel)
        
        metadata = {
            "generator": "Iceberg Visual Generator",
            "version": "1.0",
            "recommendations": [
                "Use high-resolution format for publication",
                "Include proper axis labels and legends",
                "Ensure color accessibility standards"
            ]
        }
        
        return VisualElement(
            element_type="figure",
            title=title,
            content=figure_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_table(self, title: str, data: Dict[str, Any]) -> VisualElement:
        """Generate a formatted data table"""
        
        table_id = self._generate_visual_id()
        
        # Extract table structure from data
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        
        table_content = {
            "table_id": table_id,
            "title": title,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers) if headers else 0,
            "format": "Publication-ready table",
            "recommended_style": "Professional academic format"
        }
        
        metadata = {
            "generator": "Iceberg Visual Generator",
            "version": "1.0",
            "recommendations": [
                "Use consistent formatting",
                "Include proper column headers",
                "Ensure numerical alignment"
            ]
        }
        
        return VisualElement(
            element_type="table",
            title=title,
            content=table_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_panel(self, title: str, content: Dict[str, Any]) -> VisualElement:
        """Generate a panel for complex figures"""
        
        panel_id = self._generate_visual_id()
        
        panel_content = {
            "panel_id": panel_id,
            "title": title,
            "content_type": content.get("type", "general"),
            "data": content.get("data", {}),
            "layout": content.get("layout", "standard"),
            "recommended_size": "300x300 pixels for publication"
        }
        
        metadata = {
            "generator": "Iceberg Visual Generator",
            "version": "1.0",
            "recommendations": [
                "Maintain consistent panel sizing",
                "Use clear labeling",
                "Ensure readability at small sizes"
            ]
        }
        
        return VisualElement(
            element_type="panel",
            title=title,
            content=panel_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
    
    def generate_caption(self, visual_element: VisualElement) -> str:
        """Generate a caption for a visual element"""
        
        if visual_element.element_type == "figure":
            caption = f"Figure {visual_element.content.get('figure_id', 'X')}: {visual_element.title}"
            if visual_element.content.get("panels"):
                caption += f" showing {len(visual_element.content['panels'])} panels"
            caption += "."
            
        elif visual_element.element_type == "table":
            caption = f"Table {visual_element.content.get('table_id', 'X')}: {visual_element.title}"
            caption += f" with {visual_element.content.get('row_count', 0)} rows and {visual_element.content.get('column_count', 0)} columns."
            
        elif visual_element.element_type == "panel":
            caption = f"Panel {visual_element.content.get('panel_id', 'X')}: {visual_element.title}"
            caption += f" showing {visual_element.content.get('content_type', 'data')}."
            
        else:
            caption = f"{visual_element.element_type.title()}: {visual_element.title}"
        
        return caption
    
    def create_publication_visuals(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete set of publication-ready visuals"""
        
        visuals = {
            "figures": [],
            "tables": [],
            "panels": [],
            "captions": [],
            "recommendations": []
        }
        
        # Generate main figure
        if "main_finding" in research_data:
            main_figure = self.generate_figure(
                title=research_data.get("title", "Research Findings"),
                data=research_data["main_finding"],
                figure_type="main"
            )
            visuals["figures"].append(main_figure)
            visuals["captions"].append(self.generate_caption(main_figure))
        
        # Generate comparison table
        if "comparison_data" in research_data:
            comparison_table = self.generate_table(
                title="Comparison of Methods/Results",
                data=research_data["comparison_data"]
            )
            visuals["tables"].append(comparison_table)
            visuals["captions"].append(self.generate_caption(comparison_table))
        
        # Generate methodology diagram
        if "methodology" in research_data:
            method_panel = self.generate_panel(
                title="Methodology Overview",
                content={
                    "type": "methodology",
                    "data": research_data["methodology"],
                    "layout": "flowchart"
                }
            )
            visuals["panels"].append(method_panel)
            visuals["captions"].append(self.generate_caption(method_panel))
        
        # Add publication recommendations
        visuals["recommendations"] = [
            "All figures should be 300 DPI minimum for print",
            "Use vector formats (SVG) when possible for scalability",
            "Ensure color schemes meet accessibility standards",
            "Include proper axis labels and legends",
            "Use consistent font families throughout"
        ]
        
        return visuals
    
    def _create_panel(self, title: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a panel structure"""
        return {
            "title": title,
            "content": content,
            "size": "300x300",
            "type": content.get("type", "data")
        }
    
    def _generate_visual_id(self) -> str:
        """Generate unique visual ID"""
        self.visual_counter += 1
        return f"visual_{self.visual_counter}_{int(datetime.utcnow().timestamp())}"
    
    def get_generator_status(self) -> Dict[str, Any]:
        """Get current generator status and capabilities"""
        return {
            "status": "operational",
            "available_types": self.available_types,
            "total_visuals_generated": self.visual_counter,
            "capabilities": [
                "Scientific figure generation",
                "Publication-ready table creation",
                "Panel and caption generation",
                "Publication recommendations"
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create visual generator
    generator = VisualGenerator()
    
    # Test with sample research data
    sample_research = {
        "title": "Quantum-Classical Hybrid Algorithm Performance",
        "main_finding": {
            "source": "simulation_results",
            "components": [
                {"type": "performance_graph", "data": {"x": [1, 2, 3], "y": [10, 20, 30]}},
                {"type": "efficiency_chart", "data": {"categories": ["QFOA", "QAGD++", "QCHF"], "values": [85, 92, 78]}}
            ]
        },
        "comparison_data": {
            "headers": ["Algorithm", "Efficiency", "Speed", "Accuracy"],
            "rows": [
                ["QFOA", "85%", "Fast", "High"],
                ["QAGD++", "92%", "Medium", "Very High"],
                ["QCHF", "78%", "Slow", "Medium"]
            ]
        },
        "methodology": {
            "type": "workflow",
            "data": {"steps": ["Quantum Exploration", "Classical Refinement", "Integration"]}
        }
    }
    
    # Generate publication visuals
    visuals = generator.create_publication_visuals(sample_research)
    
    print("Visual Generator Status:")
    print(generator.get_generator_status())
    
    print("\nGenerated Visuals:")
    print(f"Figures: {len(visuals['figures'])}")
    print(f"Tables: {len(visuals['tables'])}")
    print(f"Panels: {len(visuals['panels'])}")
    print(f"Captions: {len(visuals['captions'])}")
    
    print("\nSample Caption:")
    if visuals['captions']:
        print(visuals['captions'][0])
