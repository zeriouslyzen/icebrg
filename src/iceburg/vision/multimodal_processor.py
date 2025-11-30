"""
Multimodal Processor
Processes multimodal input (text, images, audio)
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
from PIL import Image
import base64
import io


class MultimodalProcessor:
    """Processes multimodal input"""
    
    def __init__(self):
        self.image_analyzer = None
        self.audio_processor = None
        
        # Initialize image analyzer
        try:
            from .image_analyzer import ImageAnalyzer
            self.image_analyzer = ImageAnalyzer()
        except ImportError:
            pass
    
    def process_input(
        self,
        text: Optional[str] = None,
        images: Optional[List[str]] = None,
        audio: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process multimodal input"""
        result = {
            "text": text,
            "images": [],
            "audio": [],
            "combined_analysis": {}
        }
        
        # Process images
        if images:
            for img_path in images:
                img_analysis = self.process_image(img_path)
                result["images"].append(img_analysis)
        
        # Process audio
        if audio:
            for audio_path in audio:
                audio_analysis = self.process_audio(audio_path)
                result["audio"].append(audio_analysis)
        
        # Combine analysis
        result["combined_analysis"] = self._combine_analysis(
            text,
            result["images"],
            result["audio"]
        )
        
        return result
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image"""
        if not self.image_analyzer:
            return {"error": "Image analyzer not available"}
        
        try:
            analysis = {
                "path": image_path,
                "ocr_text": self.image_analyzer.extract_text(image_path),
                "objects": self.image_analyzer.detect_objects(image_path),
                "scene": self.image_analyzer.analyze_scene(image_path),
                "chart": self.image_analyzer.analyze_chart(image_path),
                "diagram": self.image_analyzer.analyze_diagram(image_path)
            }
            
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio"""
        # Placeholder for audio processing
        # In production, use proper audio processing
        return {
            "path": audio_path,
            "error": "Audio processing not yet implemented"
        }
    
    def _combine_analysis(
        self,
        text: Optional[str],
        images: List[Dict[str, Any]],
        audio: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine analysis from different modalities"""
        combined = {
            "text_available": text is not None,
            "images_count": len(images),
            "audio_count": len(audio),
            "modalities": []
        }
        
        if text:
            combined["modalities"].append("text")
        
        if images:
            combined["modalities"].append("image")
            # Extract text from images
            image_texts = [img.get("ocr_text", "") for img in images if img.get("ocr_text")]
            if image_texts:
                combined["extracted_text"] = " ".join(image_texts)
        
        if audio:
            combined["modalities"].append("audio")
        
        return combined
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            return ""
    
    def decode_image(self, base64_data: str) -> Optional[Image.Image]:
        """Decode base64 image"""
        try:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            return None

