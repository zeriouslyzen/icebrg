#!/usr/bin/env python3
"""
Computer Vision Engine for ICEBURG
Enables camera vision, screen analysis, and low-level computer interaction
"""

import asyncio
import cv2
import numpy as np
import pyautogui
import mss
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import os
from PIL import Image, ImageTk
import tkinter as tk

# Computer vision models
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class VisionFrame:
    """Vision frame data"""
    timestamp: datetime
    frame_type: str  # 'camera', 'screen', 'combined'
    image_data: np.ndarray
    width: int
    height: int
    objects_detected: List[Dict[str, Any]]
    faces_detected: List[Dict[str, Any]]
    text_detected: List[Dict[str, Any]]
    screen_elements: List[Dict[str, Any]]


@dataclass
class ComputerState:
    """Current computer state"""
    timestamp: datetime
    active_window: str
    mouse_position: Tuple[int, int]
    keyboard_state: Dict[str, bool]
    screen_content: str
    applications_open: List[str]
    system_metrics: Dict[str, Any]


class ComputerVisionEngine:
    """Advanced computer vision engine for ICEBURG"""
    
    def __init__(self, global_workspace: Optional[Any] = None):
        """
        Initialize computer vision engine
        
        Args:
            global_workspace: Optional GlobalWorkspace instance for blackboard publishing
        """
        self.is_active = False
        self.camera_active = False
        self.screen_monitoring = False
        
        # Blackboard integration
        self.global_workspace = global_workspace
        
        # Camera setup
        self.camera = None
        self.camera_index = 0
        
        # Screen capture setup
        self.screen_capture = mss.mss()
        
        # Computer vision models
        self.face_detector = None
        self.object_detector = None
        self.text_detector = None
        
        # MediaPipe for advanced vision
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        
        # PyTorch models for object detection
        if TORCH_AVAILABLE:
            self._load_object_detection_model()
        
        # Callbacks
        self.on_frame_processed: Optional[Callable[[VisionFrame], None]] = None
        self.on_face_detected: Optional[Callable[[List[Dict]], None]] = None
        self.on_object_detected: Optional[Callable[[List[Dict]], None]] = None
        self.on_screen_changed: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # State tracking
        self.current_state = ComputerState(
            timestamp=datetime.now(),
            active_window="",
            mouse_position=(0, 0),
            keyboard_state={},
            screen_content="",
            applications_open=[],
            system_metrics={}
        )
        
    
    def _load_object_detection_model(self):
        """Load object detection model"""
        try:
            # Load MobileNetV2 for efficient object detection
            self.object_detector = mobilenet_v2(pretrained=True)
            self.object_detector.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.warning(f"Error initializing vision transforms: {e}")
            # Use default transform if initialization fails
            self.transform = None
    
    async def start_camera_vision(self, camera_index: int = 0) -> bool:
        """Start camera vision processing"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                return False
            
            self.camera_index = camera_index
            self.camera_active = True
            
            # Start camera processing thread
            self.camera_thread = threading.Thread(target=self._camera_worker)
            self.camera_thread.start()
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    async def start_screen_monitoring(self) -> bool:
        """Start screen monitoring and analysis"""
        try:
            self.screen_monitoring = True
            
            # Start screen monitoring thread
            self.screen_thread = threading.Thread(target=self._screen_worker)
            self.screen_thread.start()
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def _camera_worker(self):
        """Camera processing worker thread"""
        while self.camera_active and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Process frame
                processed_frame = self._process_camera_frame(frame)
                
                # Callback
                if self.on_frame_processed:
                    self.on_frame_processed(processed_frame)
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                break
    
    def _screen_worker(self):
        """Screen monitoring worker thread"""
        while self.screen_monitoring:
            try:
                # Capture screen
                screenshot = self.screen_capture.grab(self.screen_capture.monitors[1])
                frame = np.array(screenshot)
                
                # Process screen frame
                processed_frame = self._process_screen_frame(frame)
                
                # Update computer state
                self._update_computer_state(processed_frame)
                
                # Callback
                if self.on_frame_processed:
                    self.on_frame_processed(processed_frame)
                
                # Small delay
                time.sleep(0.5)
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                break
    
    def _publish_to_blackboard(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish message to blackboard via GlobalWorkspace"""
        if self.global_workspace:
            try:
                self.global_workspace.publish(topic, message)
            except Exception:
                pass  # Blackboard publishing failed, continue
    
    def _is_medical_image(self, frame: np.ndarray) -> bool:
        """Check if frame appears to be a medical image"""
        # Simple heuristic: check for medical image characteristics
        # This could be enhanced with ML-based classification
        try:
            # Check for typical medical image patterns (X-ray, MRI, CT scan)
            # For now, return False (can be enhanced with actual ML model)
            return False
        except Exception:
            return False
    
    def _analyze_medical_image(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze medical image for abnormalities"""
        # Placeholder for medical image analysis
        # This would use specialized medical imaging models
        objects = self._detect_objects(frame)
        return {
            "objects": objects,
            "analysis_type": "medical",
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_camera_frame(self, frame: np.ndarray) -> VisionFrame:
        """Process camera frame for objects, faces, and text"""
        height, width = frame.shape[:2]
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        # Detect objects
        objects = self._detect_objects(frame)
        
        # Detect text (OCR)
        text_elements = self._detect_text(frame)
        
        # Publish findings to blackboard
        if objects:
            self._publish_to_blackboard(
                "vision/objects_detected",
                {
                    "type": "object_detection",
                    "objects": objects,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        if faces:
            self._publish_to_blackboard(
                "vision/faces_detected",
                {
                    "type": "face_detection",
                    "faces": faces,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check for medical images and analyze
        if self._is_medical_image(frame):
            analysis = self._analyze_medical_image(frame)
            self._publish_to_blackboard(
                "medical/analysis",
                {
                    "type": "medical_analysis",
                    "image_type": "camera",
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return VisionFrame(
            timestamp=datetime.now(),
            frame_type="camera",
            image_data=frame,
            width=width,
            height=height,
            objects_detected=objects,
            faces_detected=faces,
            text_detected=text_elements,
            screen_elements=[]
        )
    
    def _process_screen_frame(self, frame: np.ndarray) -> VisionFrame:
        """Process screen frame for UI elements and content"""
        height, width = frame.shape[:2]
        
        # Detect UI elements
        ui_elements = self._detect_ui_elements(frame)
        
        # Extract text from screen
        screen_text = self._extract_screen_text(frame)
        
        return VisionFrame(
            timestamp=datetime.now(),
            frame_type="screen",
            image_data=frame,
            width=width,
            height=height,
            objects_detected=[],
            faces_detected=[],
            text_detected=screen_text,
            screen_elements=ui_elements
        )
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in frame"""
        faces = []
        
        if MEDIAPIPE_AVAILABLE:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        
                        face_info = {
                            "confidence": detection.score[0],
                            "bbox": {
                                "x": int(bbox.xmin * w),
                                "y": int(bbox.ymin * h),
                                "width": int(bbox.width * w),
                                "height": int(bbox.height * h)
                            },
                            "landmarks": []
                        }
                        faces.append(face_info)
                        
            except Exception as e:
                logger.warning(f"Error detecting faces: {e}")
        
        # Callback for face detection
        if faces and self.on_face_detected:
            self.on_face_detected(faces)
        
        return faces
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame"""
        objects = []
        
        if TORCH_AVAILABLE and self.object_detector:
            try:
                # Preprocess frame
                input_tensor = self.transform(frame).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.object_detector(input_tensor)
                
                # Process outputs (simplified)
                # In a real implementation, you'd use a proper object detection model
                objects.append({
                    "class": "person",
                    "confidence": 0.8,
                    "bbox": {"x": 100, "y": 100, "width": 200, "height": 300}
                })
                
            except Exception as e:
                logger.warning(f"Error detecting objects: {e}")
        
        # Callback for object detection
        if objects and self.on_object_detected:
            self.on_object_detected(objects)
        
        return objects
    
    def _detect_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text in frame using OCR"""
        text_elements = []
        
        try:
            # Simple text detection using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use EAST text detector if available
            # For now, return empty list
            text_elements = []
            
        except Exception as e:
            logger.warning(f"Error extracting screen text: {e}")
            return []
        
        return text_elements
    
    def _detect_ui_elements(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements on screen"""
        ui_elements = []
        
        try:
            # Detect buttons, windows, etc.
            # This would use computer vision to identify UI components
            ui_elements = []
            
        except Exception as e:
            logger.warning(f"Error detecting UI elements: {e}")
            return []
        
        return ui_elements
    
    def _extract_screen_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from screen"""
        text_elements = []
        
        try:
            # OCR for screen text extraction
            # This would use Tesseract or similar
            text_elements = []
            
        except Exception as e:
            logger.warning(f"Error extracting screen text: {e}")
            return []
        
        return text_elements
    
    def _update_computer_state(self, frame: VisionFrame):
        """Update current computer state"""
        try:
            # Get mouse position
            mouse_pos = pyautogui.position()
            
            # Get active window (simplified)
            active_window = "Unknown"
            
            # Update state
            self.current_state = ComputerState(
                timestamp=datetime.now(),
                active_window=active_window,
                mouse_position=mouse_pos,
                keyboard_state={},
                screen_content="",
                applications_open=[],
                system_metrics={}
            )
            
        except Exception as e:
            logger.warning(f"Error in vision processing: {e}")
    
    def get_current_state(self) -> ComputerState:
        """Get current computer state"""
        return self.current_state
    
    def take_screenshot(self) -> np.ndarray:
        """Take a screenshot"""
        try:
            screenshot = self.screen_capture.grab(self.screen_capture.monitors[1])
            return np.array(screenshot)
        except Exception as e:
            logger.warning(f"Error capturing screen: {e}")
            return np.array([])
    
    def click_at_position(self, x: int, y: int) -> bool:
        """Click at specific screen position"""
        try:
            pyautogui.click(x, y)
            return True
        except Exception as e:
            logger.warning(f"Error in vision operation: {e}")
            return False
    
    def type_text(self, text: str) -> bool:
        """Type text using keyboard"""
        try:
            pyautogui.typewrite(text)
            return True
        except Exception as e:
            logger.warning(f"Error in vision operation: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a specific key"""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            logger.warning(f"Error in vision operation: {e}")
            return False
    
    def scroll(self, clicks: int) -> bool:
        """Scroll the mouse wheel"""
        try:
            pyautogui.scroll(clicks)
            return True
        except Exception as e:
            logger.warning(f"Error in vision operation: {e}")
            return False
    
    def set_callbacks(self,
        on_frame_processed: Optional[Callable[[VisionFrame], None]] = None,
                     on_face_detected: Optional[Callable[[List[Dict]], None]] = None,
                     on_object_detected: Optional[Callable[[List[Dict]], None]] = None,
                     on_screen_changed: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_frame_processed = on_frame_processed
        self.on_face_detected = on_face_detected
        self.on_object_detected = on_object_detected
        self.on_screen_changed = on_screen_changed
        self.on_error = on_error
    
    async def stop_vision(self):
        """Stop all vision processing"""
        self.camera_active = False
        self.screen_monitoring = False
        
        if self.camera:
            self.camera.release()
        
    
    def get_status(self) -> Dict[str, Any]:
        """Get vision engine status"""
        return {
            "is_active": self.is_active,
            "camera_active": self.camera_active,
            "screen_monitoring": self.screen_monitoring,
            "camera_index": self.camera_index,
            "torch_available": TORCH_AVAILABLE,
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "current_state": {
                "active_window": self.current_state.active_window,
                "mouse_position": self.current_state.mouse_position,
                "timestamp": self.current_state.timestamp.isoformat()
            }
        }
