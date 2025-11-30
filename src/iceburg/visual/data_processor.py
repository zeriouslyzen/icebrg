"""
Data Processor
Processes data for chart generation
"""

from typing import Any, Dict, Optional, List
import json
import numpy as np


class DataProcessor:
    """Processes data for visualization"""
    
    def __init__(self):
        pass
    
    def process_data_for_chart(
        self,
        data: Any,
        chart_type: str
    ) -> Dict[str, Any]:
        """Process data for specific chart type"""
        processed = {
            "chart_type": chart_type,
            "data": {},
            "metadata": {}
        }
        
        if chart_type == "bar":
            processed["data"] = self._process_bar_data(data)
        elif chart_type == "line":
            processed["data"] = self._process_line_data(data)
        elif chart_type == "pie":
            processed["data"] = self._process_pie_data(data)
        elif chart_type == "scatter":
            processed["data"] = self._process_scatter_data(data)
        elif chart_type == "heatmap":
            processed["data"] = self._process_heatmap_data(data)
        else:
            processed["data"] = self._process_generic_data(data)
        
        return processed
    
    def _process_bar_data(self, data: Any) -> Dict[str, Any]:
        """Process data for bar chart"""
        if isinstance(data, dict):
            return {
                "x": data.get("x", []),
                "y": data.get("y", []),
                "title": data.get("title", "Bar Chart"),
                "xlabel": data.get("xlabel", "X Axis"),
                "ylabel": data.get("ylabel", "Y Axis")
            }
        elif isinstance(data, list):
            # Assume list of [x, y] pairs
            x = [item[0] for item in data]
            y = [item[1] for item in data]
            return {
                "x": x,
                "y": y,
                "title": "Bar Chart",
                "xlabel": "X Axis",
                "ylabel": "Y Axis"
            }
        else:
            return {"x": [], "y": [], "title": "Bar Chart"}
    
    def _process_line_data(self, data: Any) -> Dict[str, Any]:
        """Process data for line chart"""
        if isinstance(data, dict):
            return {
                "x": data.get("x", []),
                "y": data.get("y", []),
                "title": data.get("title", "Line Chart"),
                "xlabel": data.get("xlabel", "X Axis"),
                "ylabel": data.get("ylabel", "Y Axis")
            }
        elif isinstance(data, list):
            x = [item[0] for item in data]
            y = [item[1] for item in data]
            return {
                "x": x,
                "y": y,
                "title": "Line Chart",
                "xlabel": "X Axis",
                "ylabel": "Y Axis"
            }
        else:
            return {"x": [], "y": [], "title": "Line Chart"}
    
    def _process_pie_data(self, data: Any) -> Dict[str, Any]:
        """Process data for pie chart"""
        if isinstance(data, dict):
            return {
                "labels": data.get("labels", []),
                "values": data.get("values", []),
                "title": data.get("title", "Pie Chart")
            }
        elif isinstance(data, list):
            # Assume list of [label, value] pairs
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            return {
                "labels": labels,
                "values": values,
                "title": "Pie Chart"
            }
        else:
            return {"labels": [], "values": [], "title": "Pie Chart"}
    
    def _process_scatter_data(self, data: Any) -> Dict[str, Any]:
        """Process data for scatter chart"""
        if isinstance(data, dict):
            return {
                "x": data.get("x", []),
                "y": data.get("y", []),
                "title": data.get("title", "Scatter Chart"),
                "xlabel": data.get("xlabel", "X Axis"),
                "ylabel": data.get("ylabel", "Y Axis")
            }
        elif isinstance(data, list):
            x = [item[0] for item in data]
            y = [item[1] for item in data]
            return {
                "x": x,
                "y": y,
                "title": "Scatter Chart",
                "xlabel": "X Axis",
                "ylabel": "Y Axis"
            }
        else:
            return {"x": [], "y": [], "title": "Scatter Chart"}
    
    def _process_heatmap_data(self, data: Any) -> Dict[str, Any]:
        """Process data for heatmap"""
        if isinstance(data, dict):
            return {
                "matrix": data.get("matrix", []),
                "title": data.get("title", "Heatmap"),
                "xlabel": data.get("xlabel", "X Axis"),
                "ylabel": data.get("ylabel", "Y Axis")
            }
        elif isinstance(data, list):
            # Assume 2D list/matrix
            return {
                "matrix": data,
                "title": "Heatmap",
                "xlabel": "X Axis",
                "ylabel": "Y Axis"
            }
        else:
            return {"matrix": [], "title": "Heatmap"}
    
    def _process_generic_data(self, data: Any) -> Dict[str, Any]:
        """Process generic data"""
        return {
            "data": data,
            "title": "Chart"
        }
    
    def validate_data(self, data: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
        """Validate data for chart type"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if chart_type == "bar" or chart_type == "line" or chart_type == "scatter":
            if "x" not in data or "y" not in data:
                validation["valid"] = False
                validation["errors"].append("Missing x or y data")
            elif len(data["x"]) != len(data["y"]):
                validation["valid"] = False
                validation["errors"].append("x and y data must have same length")
        
        elif chart_type == "pie":
            if "labels" not in data or "values" not in data:
                validation["valid"] = False
                validation["errors"].append("Missing labels or values")
            elif len(data["labels"]) != len(data["values"]):
                validation["valid"] = False
                validation["errors"].append("labels and values must have same length")
        
        elif chart_type == "heatmap":
            if "matrix" not in data:
                validation["valid"] = False
                validation["errors"].append("Missing matrix data")
        
        return validation

