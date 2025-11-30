"""
Image Analyzer
Image understanding with OCR, object detection, and scene analysis
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
from PIL import Image
import io


class ImageAnalyzer:
    """Analyzes images with multiple methods"""
    
    def __init__(self):
        self.ocr_enabled = False
        self.object_detection_enabled = False
        
        # Try to initialize OCR
        try:
            import pytesseract
            self.ocr_enabled = True
        except ImportError:
            pass
        
        # Try to initialize OpenCV
        try:
            import cv2
            self.object_detection_enabled = True
        except ImportError:
            pass
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        if not self.ocr_enabled:
            return ""
        
        try:
            import pytesseract
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"OCR error: {str(e)}"
    
    def detect_objects(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        if not self.object_detection_enabled:
            return []
        
        try:
            import cv2
            import numpy as np
            
            # Load image
            image = Image.open(image_path)
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for contour in contours[:20]:  # Limit to 20 objects
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "type": "object",
                        "bbox": [x, y, w, h],
                        "area": area,
                        "confidence": 0.7
                    })
            
            return objects
        except Exception as e:
            return []
    
    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        """Analyze scene in image"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Basic scene analysis
            analysis = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "format": image.format,
                "mode": image.mode,
                "description": self._describe_scene(image)
            }
            
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    def _describe_scene(self, image: Image.Image) -> str:
        """Describe scene in image"""
        width, height = image.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            return "Wide landscape or chart"
        elif aspect_ratio < 0.7:
            return "Portrait or document"
        else:
            return "Square or balanced composition"
    
    def analyze_chart(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Analyze chart in image"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Basic chart detection
            if width > height * 1.2:
                return {
                    "type": "bar_chart",
                    "confidence": 0.6,
                    "dimensions": [width, height]
                }
            elif abs(width - height) < 50:
                return {
                    "type": "pie_chart",
                    "confidence": 0.6,
                    "dimensions": [width, height]
                }
            
            return None
        except Exception as e:
            return None
    
    def analyze_diagram(self, image_path: str) -> List[Dict[str, Any]]:
        """Analyze diagram elements"""
        if not self.object_detection_enabled:
            return []
        
        try:
            import cv2
            import numpy as np
            
            image = Image.open(image_path)
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
            
            # Detect lines
            lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
            
            elements = []
            if lines is not None:
                for line in lines[:20]:  # Limit to 20 lines
                    x1, y1, x2, y2 = line[0]
                    elements.append({
                        "type": "line",
                        "start": [x1, y1],
                        "end": [x2, y2]
                    })
            
            return elements
        except Exception as e:
            return []

