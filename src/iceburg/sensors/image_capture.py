"""
Image Capture
Image capture and processing utilities
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import time
import base64
from PIL import Image
import io


class ImageCapture:
    """Image capture and processing utilities"""
    
    def __init__(self):
        self.captured_images: List[Dict[str, Any]] = []
    
    def capture_from_camera(
        self,
        camera_interface: Any,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Capture image from camera interface"""
        if not camera_interface or not camera_interface.camera_active:
            return None
        
        try:
            image_data = camera_interface.capture_image(output_path)
            if image_data:
                self.captured_images.append({
                    "path": output_path or "base64",
                    "timestamp": time.time(),
                    "data": image_data
                })
                return image_data
        except Exception as e:
            return None
        
        return None
    
    def process_image(
        self,
        image_path: str,
        operations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process image with various operations"""
        if not operations:
            operations = ["resize", "enhance"]
        
        result = {
            "image_path": image_path,
            "operations": operations,
            "processed": False,
            "output_path": None
        }
        
        try:
            image = Image.open(image_path)
            
            # Resize
            if "resize" in operations:
                image = image.resize((800, 600), Image.Resampling.LANCZOS)
            
            # Enhance
            if "enhance" in operations:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
            
            # Save processed image
            output_path = image_path.replace(".jpg", "_processed.jpg")
            image.save(output_path)
            
            result["processed"] = True
            result["output_path"] = output_path
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_captured_images(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get captured images"""
        return self.captured_images[-limit:] if self.captured_images else []
    
    def clear_captured_images(self) -> int:
        """Clear captured images"""
        count = len(self.captured_images)
        self.captured_images.clear()
        return count

