"""
ICEBURG Geospatial Financial Anthropological Agent
Analyzes geospatial, financial, and anthropological data patterns
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

from ..config import IceburgConfig
from ..global_workspace import GlobalWorkspace
from ..vectorstore import VectorStore

try:
    from ..vision.computer_vision_engine import ComputerVisionEngine
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GeospatialFinancialAnthropological:
    """
    Analyzes geospatial, financial, and anthropological data patterns
    """

    def __init__(self, cfg: IceburgConfig, global_workspace: Optional[GlobalWorkspace] = None, 
                 vision_engine: Optional[ComputerVisionEngine] = None, 
                 vectorstore: Optional[VectorStore] = None):
        """
        Initialize geospatial agent
        
        Args:
            cfg: ICEBURG config
            global_workspace: Optional GlobalWorkspace for blackboard publishing
            vision_engine: Optional ComputerVisionEngine for image analysis
            vectorstore: Optional VectorStore for storing analysis results
        """
        self.cfg = cfg
        self.global_workspace = global_workspace
        self.vision_engine = vision_engine
        self.vectorstore = vectorstore

    def _fetch_google_earth_image(self, coordinates: Tuple[float, float], 
                                   zoom: int = 15, size: Tuple[int, int] = (640, 640)) -> Optional[bytes]:
        """
        Fetch satellite imagery from Google Earth/Static Maps API
        
        Args:
            coordinates: (latitude, longitude) tuple
            zoom: Zoom level (1-20)
            size: Image size (width, height)
            
        Returns:
            Image bytes or None if failed
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            lat, lon = coordinates
            api_key = self.cfg.google_earth_api_key if hasattr(self.cfg, 'google_earth_api_key') else None
            
            if not api_key:
                # Fallback: Use static map without API key (limited functionality)
                url = f"https://maps.googleapis.com/maps/api/staticmap"
                params = {
                    "center": f"{lat},{lon}",
                    "zoom": zoom,
                    "size": f"{size[0]}x{size[1]}",
                    "maptype": "satellite",
                    "format": "png"
                }
                if api_key:
                    params["key"] = api_key
            else:
                # Use Google Earth Engine API if available
                project_id = getattr(self.cfg, 'google_earth_project_id', None)
                if project_id:
                    url = f"https://earthengine.googleapis.com/v1alpha/projects/{project_id}/image"
                else:
                    url = "https://maps.googleapis.com/maps/api/staticmap"
                params = {
                    "center": f"{lat},{lon}",
                    "zoom": zoom,
                    "size": f"{size[0]}x{size[1]}",
                    "maptype": "satellite",
                    "format": "png"
                }
                if api_key:
                    params["key"] = api_key
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception:
            pass
        
        return None

    def _analyze_satellite_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze satellite imagery using vision engine
        
        Args:
            image_data: Image bytes
            
        Returns:
            Analysis results
        """
        if not self.vision_engine or not VISION_AVAILABLE:
            return {
                "objects": [],
                "analysis_type": "satellite",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_data))
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Use vision engine to analyze
            objects = self.vision_engine._detect_objects(frame) if hasattr(self.vision_engine, '_detect_objects') else []
            
            return {
                "objects": objects,
                "analysis_type": "satellite",
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {
                "objects": [],
                "analysis_type": "satellite",
                "timestamp": datetime.now().isoformat()
            }

    def _detect_environmental_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect environmental patterns from analysis
        
        Args:
            analysis: Analysis results
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Simple pattern detection based on objects
        objects = analysis.get("objects", [])
        if objects:
            # Group objects by type
            object_types = {}
            for obj in objects:
                obj_type = obj.get("type", "unknown")
                if obj_type not in object_types:
                    object_types[obj_type] = []
                object_types[obj_type].append(obj)
            
            # Detect patterns
            for obj_type, objs in object_types.items():
                if len(objs) > 1:
                    patterns.append({
                        "type": "clustering",
                        "object_type": obj_type,
                        "count": len(objs),
                        "description": f"Multiple {obj_type} objects detected"
                    })
        
        return patterns

    def analyze_environment(self, coordinates: Tuple[float, float], 
                           zoom: int = 15) -> Dict[str, Any]:
        """
        Analyze environment at coordinates using Google Earth and VLMs
        
        Args:
            coordinates: (latitude, longitude) tuple
            zoom: Zoom level
            
        Returns:
            Analysis results
        """
        try:
            # 1. Fetch Google Earth/satellite imagery
            image_data = self._fetch_google_earth_image(coordinates, zoom=zoom)
            
            if not image_data:
                return {
                    "coordinates": coordinates,
                    "error": "Failed to fetch satellite imagery",
                    "patterns": [],
                    "analysis": {}
                }
            
            # 2. Use VLM to analyze imagery
            analysis = self._analyze_satellite_image(image_data)
            
            # 3. Detect patterns (vegetation, structures, changes)
            patterns = self._detect_environmental_patterns(analysis)
            
            # 4. Store results in VectorStore
            if self.vectorstore:
                try:
                    analysis_text = f"Environmental analysis at {coordinates}: {json.dumps(patterns)}"
                    self.vectorstore.add(
                        [analysis_text],
                        metadatas=[{
                            "type": "geospatial",
                            "coordinates": f"{coordinates[0]},{coordinates[1]}",
                            "timestamp": datetime.now().isoformat(),
                            "source": "geospatial_agent"
                        }],
                        ids=[f"geospatial_{coordinates[0]}_{coordinates[1]}_{datetime.now().timestamp()}"]
                    )
                except Exception:
                    pass
            
            # 5. Publish to blackboard
            if self.global_workspace:
                try:
                    self.global_workspace.publish(
                        "geospatial/analysis",
                        {
                            "type": "environmental_analysis",
                            "coordinates": coordinates,
                            "patterns": patterns,
                            "analysis": analysis,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                except Exception:
                    pass
            
            return {
                "coordinates": coordinates,
                "patterns": patterns,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "coordinates": coordinates,
                "error": str(e),
                "patterns": [],
                "analysis": {}
            }

    def run(self, cfg: IceburgConfig, query: str, context: Any = None, verbose: bool = False) -> Dict[str, Any]:
        """Run geospatial financial anthropological analysis"""

        try:
            # Check if query contains coordinates
            import re
            coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
            matches = re.findall(coord_pattern, query)
            
            if matches:
                # Extract coordinates
                lat, lon = float(matches[0][0]), float(matches[0][1])
                coordinates = (lat, lon)
                
                # Analyze environment
                results = self.analyze_environment(coordinates)
                results["query"] = query
                results["analysis_type"] = "geospatial_financial_anthropological"
                
                return results
            else:
                # Simulate analysis for non-coordinate queries
                results = {
                    "query": query,
                    "analysis_type": "geospatial_financial_anthropological",
                    "results": [],
                    "processing_time": "simulated",
                    "note": "No coordinates found in query. Use format: 'analyze environment at 37.7749, -122.4194'"
                }

                return results

        except Exception as e:
            if verbose:
                print(f"[GEOSPATIAL_FINANCIAL_ANTHROPOLOGICAL] Error: {e}")
            return {
                "error": str(e),
                "results": []
            }
