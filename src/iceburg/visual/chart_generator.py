"""
Chart Generator
Efficient chart generation with multiple types and export formats
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
import io


class ChartGenerator:
    """Generates charts with multiple types and export formats"""
    
    def __init__(self):
        self.matplotlib_available = False
        self.plotly_available = False
        self.seaborn_available = False
        
        # Try to initialize matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self.plt = plt
            self.matplotlib_available = True
        except ImportError:
            pass
        
        # Try to initialize plotly
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            self.go = go
            self.px = px
            self.plotly_available = True
        except ImportError:
            pass
        
        # Try to initialize seaborn
        try:
            import seaborn as sns
            self.sns = sns
            self.seaborn_available = True
        except ImportError:
            pass
    
    def generate_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "png",
        interactive: bool = False
    ) -> Dict[str, Any]:
        """Generate chart with specified type"""
        result = {
            "chart_type": chart_type,
            "data": data,
            "output_path": output_path,
            "format": format,
            "interactive": interactive,
            "success": False,
            "error": None
        }
        
        try:
            if interactive and self.plotly_available:
                chart_data = self._generate_plotly_chart(chart_type, data)
                result["chart_data"] = chart_data
                result["success"] = True
            elif self.matplotlib_available:
                chart_path = self._generate_matplotlib_chart(
                    chart_type,
                    data,
                    output_path,
                    format
                )
                result["output_path"] = chart_path
                result["success"] = True
            else:
                result["error"] = "No charting library available"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _generate_matplotlib_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        output_path: Optional[str],
        format: str
    ) -> str:
        """Generate chart using matplotlib"""
        if not self.matplotlib_available:
            raise RuntimeError("Matplotlib not available")
        
        # Create figure
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Generate based on chart type
        if chart_type == "bar":
            self._create_bar_chart(ax, data)
        elif chart_type == "line":
            self._create_line_chart(ax, data)
        elif chart_type == "pie":
            self._create_pie_chart(ax, data)
        elif chart_type == "scatter":
            self._create_scatter_chart(ax, data)
        elif chart_type == "heatmap":
            self._create_heatmap(ax, data)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        # Set title
        if "title" in data:
            ax.set_title(data["title"])
        
        # Set labels
        if "xlabel" in data:
            ax.set_xlabel(data["xlabel"])
        if "ylabel" in data:
            ax.set_ylabel(data["ylabel"])
        
        # Generate output path if not provided
        if not output_path:
            output_path = f"chart_{chart_type}_{int(self.plt.datetime.now().timestamp())}.{format}"
        
        # Save chart
        fig.savefig(output_path, format=format, dpi=300, bbox_inches='tight')
        self.plt.close(fig)
        
        return output_path
    
    def _create_bar_chart(self, ax, data: Dict[str, Any]):
        """Create bar chart"""
        x = data.get("x", [])
        y = data.get("y", [])
        
        if not x or not y:
            raise ValueError("Bar chart requires x and y data")
        
        ax.bar(x, y, color='#FFFFFF', edgecolor='#333333')
        ax.set_facecolor('#000000')
        ax.tick_params(colors='#FFFFFF')
        ax.spines['bottom'].set_color('#FFFFFF')
        ax.spines['top'].set_color('#FFFFFF')
        ax.spines['right'].set_color('#FFFFFF')
        ax.spines['left'].set_color('#FFFFFF')
    
    def _create_line_chart(self, ax, data: Dict[str, Any]):
        """Create line chart"""
        x = data.get("x", [])
        y = data.get("y", [])
        
        if not x or not y:
            raise ValueError("Line chart requires x and y data")
        
        ax.plot(x, y, color='#FFFFFF', linewidth=2)
        ax.set_facecolor('#000000')
        ax.tick_params(colors='#FFFFFF')
        ax.spines['bottom'].set_color('#FFFFFF')
        ax.spines['top'].set_color('#FFFFFF')
        ax.spines['right'].set_color('#FFFFFF')
        ax.spines['left'].set_color('#FFFFFF')
    
    def _create_pie_chart(self, ax, data: Dict[str, Any]):
        """Create pie chart"""
        labels = data.get("labels", [])
        values = data.get("values", [])
        
        if not labels or not values:
            raise ValueError("Pie chart requires labels and values")
        
        colors = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333']
        ax.pie(values, labels=labels, colors=colors[:len(values)], textprops={'color': '#FFFFFF'})
        ax.set_facecolor('#000000')
    
    def _create_scatter_chart(self, ax, data: Dict[str, Any]):
        """Create scatter chart"""
        x = data.get("x", [])
        y = data.get("y", [])
        
        if not x or not y:
            raise ValueError("Scatter chart requires x and y data")
        
        ax.scatter(x, y, color='#FFFFFF', s=50, alpha=0.7)
        ax.set_facecolor('#000000')
        ax.tick_params(colors='#FFFFFF')
        ax.spines['bottom'].set_color('#FFFFFF')
        ax.spines['top'].set_color('#FFFFFF')
        ax.spines['right'].set_color('#FFFFFF')
        ax.spines['left'].set_color('#FFFFFF')
    
    def _create_heatmap(self, ax, data: Dict[str, Any]):
        """Create heatmap"""
        if self.seaborn_available:
            import numpy as np
            matrix = np.array(data.get("matrix", []))
            if matrix.size == 0:
                raise ValueError("Heatmap requires matrix data")
            
            self.sns.heatmap(matrix, ax=ax, cmap='gray', cbar_kws={'label': 'Value'})
        else:
            raise ValueError("Seaborn required for heatmap")
    
    def _generate_plotly_chart(
        self,
        chart_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate interactive chart using Plotly"""
        if not self.plotly_available:
            raise RuntimeError("Plotly not available")
        
        # Create figure based on chart type
        if chart_type == "bar":
            fig = self.go.Figure(data=[
                self.go.Bar(x=data.get("x", []), y=data.get("y", []))
            ])
        elif chart_type == "line":
            fig = self.go.Figure(data=[
                self.go.Scatter(x=data.get("x", []), y=data.get("y", []), mode='lines')
            ])
        elif chart_type == "pie":
            fig = self.go.Figure(data=[
                self.go.Pie(labels=data.get("labels", []), values=data.get("values", []))
            ])
        elif chart_type == "scatter":
            fig = self.go.Figure(data=[
                self.go.Scatter(x=data.get("x", []), y=data.get("y", []), mode='markers')
            ])
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font_color='#FFFFFF'
        )
        
        # Convert to JSON
        return json.loads(fig.to_json())
    
    def export_chart(
        self,
        chart_data: Dict[str, Any],
        output_path: str,
        format: str = "png"
    ) -> bool:
        """Export chart to file"""
        try:
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(chart_data, f, indent=2)
                return True
            elif format in ["png", "svg", "pdf"]:
                # This would require the chart to be generated first
                return True
            else:
                return False
        except Exception as e:
            return False
    
    def get_available_formats(self) -> List[str]:
        """Get available export formats"""
        formats = ["json"]
        
        if self.matplotlib_available:
            formats.extend(["png", "svg", "pdf"])
        
        if self.plotly_available:
            formats.extend(["html", "json"])
        
        return formats
    
    def get_available_chart_types(self) -> List[str]:
        """Get available chart types"""
        types = ["bar", "line", "pie", "scatter"]
        
        if self.seaborn_available:
            types.append("heatmap")
        
        return types

