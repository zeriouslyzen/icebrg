"""
Camera Interface
Mac M1 camera access for real-time image capture
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import time
import base64


class CameraInterface:
    """Interface for Mac M1 camera access"""
    
    def __init__(self):
        self.camera_available = False
        self.camera_active = False
        self.opencv_available = False
        
        # Try to initialize OpenCV
        try:
            import cv2
            self.cv2 = cv2
            self.opencv_available = True
            self.camera_available = self._check_camera_available()
        except ImportError:
            pass
    
    def _check_camera_available(self) -> bool:
        """Check if camera is available"""
        if not self.opencv_available:
            return False
        
        try:
            # Try to open camera
            cap = self.cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                return True
        except Exception:
            pass
        
        return False
    
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start camera"""
        if not self.opencv_available:
            return False
        
        try:
            self.cap = self.cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.camera_active = True
                return True
        except Exception:
            pass
        
        return False
    
    def stop_camera(self) -> bool:
        """Stop camera"""
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
                self.camera_active = False
                return True
            except Exception:
                pass
        
        return False
    
    def capture_image(self, output_path: Optional[str] = None) -> Optional[str]:
        """Capture image from camera"""
        if not self.camera_active or not hasattr(self, 'cap'):
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                if output_path:
                    self.cv2.imwrite(output_path, frame)
                    return output_path
                else:
                    # Return base64 encoded image
                    _, buffer = self.cv2.imencode('.jpg', frame)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    return image_base64
        except Exception as e:
            return None
        
        return None
    
    def get_camera_feed(self) -> Optional[Any]:
        """Get camera feed frame"""
        if not self.camera_active or not hasattr(self, 'cap'):
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
        except Exception:
            pass
        
        return None
    
    def record_video(
        self,
        output_path: str,
        duration: int = 10,
        fps: int = 30
    ) -> bool:
        """Record video from camera"""
        if not self.camera_active or not hasattr(self, 'cap'):
            return False
        
        try:
            fourcc = self.cv2.VideoWriter_fourcc(*'mp4v')
            out = self.cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
            
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                if ret:
                    out.write(frame)
                else:
                    break
            
            out.release()
            return True
        except Exception as e:
            return False
    
    def detect_faces(self, frame: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Detect faces in frame"""
        if not self.opencv_available:
            return []
        
        try:
            if frame is None:
                frame = self.get_camera_feed()
            
            if frame is None:
                return []
            
            # Convert to grayscale
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            face_cascade = self.cv2.CascadeClassifier(
                self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return [
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": 0.8
                }
                for (x, y, w, h) in faces
            ]
        except Exception:
            return []
    
    def get_camera_status(self) -> Dict[str, Any]:
        """Get camera status"""
        return {
            "camera_available": self.camera_available,
            "camera_active": self.camera_active,
            "opencv_available": self.opencv_available
        }

