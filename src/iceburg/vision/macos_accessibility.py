#!/usr/bin/env python3
"""
macOS Accessibility Integration for 2025
Full screen analysis, desktop automation, and UI element detection
"""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

# macOS specific imports
try:
    import Quartz
    import AppKit
    from Foundation import NSObject, NSRunLoop, NSDefaultRunLoopMode
    from Cocoa import NSApplication, NSWorkspace
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False

# Computer vision
import cv2
from PIL import Image
import pytesseract

# Screen capture
import mss


@dataclass
class UIElement:
    """UI element data"""
    element_type: str  # button, text, window, etc.
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    attributes: Dict[str, Any]
    confidence: float
    timestamp: datetime


@dataclass
class ScreenAnalysis:
    """Screen analysis data"""
    timestamp: datetime
    screenshot: np.ndarray
    ui_elements: List[UIElement]
    text_content: str
    active_window: str
    mouse_position: Tuple[int, int]
    window_list: List[Dict[str, Any]]


class MacOSAccessibilityEngine:
    """Advanced macOS accessibility and screen analysis engine"""
    
    def __init__(self):
        self.is_active = False
        self.screen_monitoring = False
        
        # Screen capture
        self.screen_capture = mss.mss()
        
        # State
        self.current_analysis = None
        self.ui_elements_cache = []
        
        # Threads
        self.monitoring_thread = None
        self.analysis_queue = queue.Queue()
        
        # Callbacks
        self.on_screen_changed: Optional[Callable[[ScreenAnalysis], None]] = None
        self.on_ui_element_detected: Optional[Callable[[UIElement], None]] = None
        self.on_window_changed: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # OCR engine
        self.ocr_engine = pytesseract
        
    
    def initialize(self) -> bool:
        """Initialize the accessibility engine"""
        try:
            if not MACOS_AVAILABLE:
                return False
            
            # Request accessibility permissions
            if not self._request_accessibility_permissions():
                return False
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def _request_accessibility_permissions(self) -> bool:
        """Request accessibility permissions"""
        try:
            # Check if we have accessibility permissions
            # This is a simplified check - in practice, you'd need to handle the permission request
            return True  # Assume permissions are granted for now
            
        except Exception as e:
            return False
    
    async def start_screen_monitoring(self) -> bool:
        """Start screen monitoring and analysis"""
        try:
            self.screen_monitoring = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def _monitoring_worker(self):
        """Screen monitoring worker thread"""
        while self.screen_monitoring:
            try:
                # Capture screen
                screenshot = self.screen_capture.grab(self.screen_capture.monitors[1])
                frame = np.array(screenshot)
                
                # Analyze screen
                analysis = self._analyze_screen(frame)
                
                # Update current analysis
                self.current_analysis = analysis
                
                # Callback
                if self.on_screen_changed:
                    self.on_screen_changed(analysis)
                
                # Small delay
                time.sleep(0.5)
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                break
    
    def _analyze_screen(self, screenshot: np.ndarray) -> ScreenAnalysis:
        """Analyze screen content"""
        try:
            # Extract text using OCR
            text_content = self._extract_text_from_screenshot(screenshot)
            
            # Detect UI elements
            ui_elements = self._detect_ui_elements(screenshot)
            
            # Get active window
            active_window = self._get_active_window()
            
            # Get mouse position
            mouse_position = self._get_mouse_position()
            
            # Get window list
            window_list = self._get_window_list()
            
            return ScreenAnalysis(
                timestamp=datetime.now(),
                screenshot=screenshot,
                ui_elements=ui_elements,
                text_content=text_content,
                active_window=active_window,
                mouse_position=mouse_position,
                window_list=window_list
            )
            
        except Exception as e:
            return ScreenAnalysis(
                timestamp=datetime.now(),
                screenshot=screenshot,
                ui_elements=[],
                text_content="",
                active_window="Unknown",
                mouse_position=(0, 0),
                window_list=[]
            )
    
    def _extract_text_from_screenshot(self, screenshot: np.ndarray) -> str:
        """Extract text from screenshot using OCR"""
        try:
            # Convert to PIL Image
            image = Image.fromarray(screenshot)
            
            # Extract text using Tesseract
            text = self.ocr_engine.image_to_string(image)
            
            return text.strip()
            
        except Exception as e:
            return ""
    
    def _detect_ui_elements(self, screenshot: np.ndarray) -> List[UIElement]:
        """Detect UI elements in screenshot"""
        try:
            ui_elements = []
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Detect buttons (rectangular shapes)
            buttons = self._detect_buttons(gray)
            ui_elements.extend(buttons)
            
            # Detect text areas
            text_areas = self._detect_text_areas(gray)
            ui_elements.extend(text_areas)
            
            # Detect windows
            windows = self._detect_windows(gray)
            ui_elements.extend(windows)
            
            return ui_elements
            
        except Exception as e:
            return []
    
    def _detect_buttons(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect button-like elements"""
        try:
            buttons = []
            
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's rectangular (button-like)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by size (reasonable button size)
                    if 20 < w < 300 and 20 < h < 100:
                        button = UIElement(
                            element_type="button",
                            bounds=(x, y, w, h),
                            text="",
                            attributes={"area": w * h, "aspect_ratio": w / h},
                            confidence=0.7,
                            timestamp=datetime.now()
                        )
                        buttons.append(button)
            
            return buttons
            
        except Exception as e:
            return []
    
    def _detect_text_areas(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect text areas"""
        try:
            text_areas = []
            
            # Use EAST text detector if available
            # For now, use simple contour detection
            
            # Threshold to get text regions
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (reasonable text size)
                if 10 < w < 500 and 10 < h < 50:
                    text_area = UIElement(
                        element_type="text",
                        bounds=(x, y, w, h),
                        text="",
                        attributes={"area": w * h},
                        confidence=0.6,
                        timestamp=datetime.now()
                    )
                    text_areas.append(text_area)
            
            return text_areas
            
        except Exception as e:
            return []
    
    def _detect_windows(self, gray_image: np.ndarray) -> List[UIElement]:
        """Detect window-like elements"""
        try:
            windows = []
            
            # Simple window detection based on large rectangular areas
            edges = cv2.Canny(gray_image, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (window-like size)
                if w > 200 and h > 200:
                    window = UIElement(
                        element_type="window",
                        bounds=(x, y, w, h),
                        text="",
                        attributes={"area": w * h},
                        confidence=0.8,
                        timestamp=datetime.now()
                    )
                    windows.append(window)
            
            return windows
            
        except Exception as e:
            return []
    
    def _get_active_window(self) -> str:
        """Get active window information"""
        try:
            if MACOS_AVAILABLE:
                # Get frontmost application
                workspace = NSWorkspace.sharedWorkspace()
                front_app = workspace.frontmostApplication()
                
                if front_app:
                    return front_app.localizedName()
            
            return "Unknown"
            
        except Exception as e:
            return "Unknown"
    
    def _get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        try:
            if MACOS_AVAILABLE:
                # Get mouse position using Quartz
                mouse_pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
                return (int(mouse_pos.x), int(mouse_pos.y))
            
            return (0, 0)
            
        except Exception as e:
            return (0, 0)
    
    def _get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of open windows"""
        try:
            windows = []
            
            if MACOS_AVAILABLE:
                # Get window list using Quartz
                window_list = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionOnScreenOnly,
                    Quartz.kCGNullWindowID
                )
                
                for window in window_list:
                    window_info = {
                        "name": window.get("kCGWindowName", "Unknown"),
                        "owner": window.get("kCGWindowOwnerName", "Unknown"),
                        "bounds": window.get("kCGWindowBounds", {}),
                        "layer": window.get("kCGWindowLayer", 0)
                    }
                    windows.append(window_info)
            
            return windows
            
        except Exception as e:
            return []
    
    # Desktop automation methods
    def click_at_position(self, x: int, y: int) -> bool:
        """Click at specific screen position"""
        try:
            if MACOS_AVAILABLE:
                # Create mouse click event
                click_event = Quartz.CGEventCreateMouseEvent(
                    None,
                    Quartz.kCGEventLeftMouseDown,
                    (x, y),
                    Quartz.kCGMouseButtonLeft
                )
                
                # Post the event
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, click_event)
                
                # Mouse up event
                click_event_up = Quartz.CGEventCreateMouseEvent(
                    None,
                    Quartz.kCGEventLeftMouseUp,
                    (x, y),
                    Quartz.kCGMouseButtonLeft
                )
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, click_event_up)
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def type_text(self, text: str) -> bool:
        """Type text using keyboard"""
        try:
            if MACOS_AVAILABLE:
                # Create keyboard event
                for char in text:
                    # Convert character to key code
                    key_code = ord(char.upper())
                    
                    # Key down event
                    key_down = Quartz.CGEventCreateKeyboardEvent(None, key_code, True)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
                    
                    # Key up event
                    key_up = Quartz.CGEventCreateKeyboardEvent(None, key_code, False)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
                    
                    time.sleep(0.01)  # Small delay between keystrokes
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a specific key"""
        try:
            if MACOS_AVAILABLE:
                # Map common keys to key codes
                key_map = {
                    "enter": 36,
                    "space": 49,
                    "tab": 48,
                    "escape": 53,
                    "delete": 51,
                    "return": 36
                }
                
                key_code = key_map.get(key.lower(), ord(key.upper()))
                
                # Key down
                key_down = Quartz.CGEventCreateKeyboardEvent(None, key_code, True)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
                
                # Key up
                key_up = Quartz.CGEventCreateKeyboardEvent(None, key_code, False)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> bool:
        """Scroll at position"""
        try:
            if MACOS_AVAILABLE:
                # Create scroll event
                scroll_event = Quartz.CGEventCreateScrollWheelEvent(
                    None,
                    Quartz.kCGScrollEventUnitPixel,
                    2,  # Number of wheels
                    delta_y,
                    delta_x
                )
                
                # Set position
                Quartz.CGEventSetLocation(scroll_event, (x, y))
                
                # Post event
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, scroll_event)
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def set_callbacks(self,
                     on_screen_changed: Optional[Callable[[ScreenAnalysis], None]] = None,
                     on_ui_element_detected: Optional[Callable[[UIElement], None]] = None,
                     on_window_changed: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_screen_changed = on_screen_changed
        self.on_ui_element_detected = on_ui_element_detected
        self.on_window_changed = on_window_changed
        self.on_error = on_error
    
    def get_current_analysis(self) -> Optional[ScreenAnalysis]:
        """Get current screen analysis"""
        return self.current_analysis
    
    def get_status(self) -> Dict[str, Any]:
        """Get accessibility engine status"""
        return {
            "is_active": self.is_active,
            "screen_monitoring": self.screen_monitoring,
            "macos_available": MACOS_AVAILABLE,
            "current_analysis": self.current_analysis is not None,
            "ui_elements_cached": len(self.ui_elements_cache)
        }
    
    async def stop_monitoring(self):
        """Stop screen monitoring"""
        self.screen_monitoring = False
