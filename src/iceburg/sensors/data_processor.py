"""
Sensor Data Processor
Intelligent sensor data processing and pattern detection
"""

from typing import Any, Dict, Optional, List
import time
from collections import deque
import statistics


class SensorDataProcessor:
    """Processes sensor data intelligently"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_buffer: Dict[str, deque] = {}
        self.patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    def process_data(
        self,
        sensor_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process sensor data"""
        if sensor_type not in self.data_buffer:
            self.data_buffer[sensor_type] = deque(maxlen=self.window_size)
        
        self.data_buffer[sensor_type].append(data)
        
        # Analyze patterns
        analysis = self._analyze_patterns(sensor_type)
        
        return {
            "sensor_type": sensor_type,
            "data": data,
            "analysis": analysis,
            "timestamp": time.time()
        }
    
    def _analyze_patterns(self, sensor_type: str) -> Dict[str, Any]:
        """Analyze patterns in sensor data"""
        if sensor_type not in self.data_buffer:
            return {}
        
        data = list(self.data_buffer[sensor_type])
        if not data:
            return {}
        
        analysis = {
            "sample_count": len(data),
            "patterns": []
        }
        
        # Detect movement patterns
        if sensor_type == "accelerometer":
            analysis["patterns"] = self._detect_movement_patterns(data)
        elif sensor_type == "gyroscope":
            analysis["patterns"] = self._detect_rotation_patterns(data)
        elif sensor_type == "temperature":
            analysis["patterns"] = self._detect_thermal_patterns(data)
        elif sensor_type == "fan_speed":
            analysis["patterns"] = self._detect_fan_patterns(data)
        
        return analysis
    
    def _detect_movement_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect movement patterns"""
        patterns = []
        
        # Extract values
        x_values = [d.get("x", 0.0) for d in data]
        y_values = [d.get("y", 0.0) for d in data]
        z_values = [d.get("z", 0.0) for d in data]
        
        # Detect vibration
        x_std = statistics.stdev(x_values) if len(x_values) > 1 else 0.0
        y_std = statistics.stdev(y_values) if len(y_values) > 1 else 0.0
        z_std = statistics.stdev(z_values) if len(z_values) > 1 else 0.0
        
        if x_std > 0.1 or y_std > 0.1 or z_std > 0.1:
            patterns.append({
                "type": "vibration",
                "intensity": max(x_std, y_std, z_std),
                "confidence": 0.8
            })
        
        # Detect orientation change
        z_mean = statistics.mean(z_values) if z_values else 0.0
        if abs(z_mean - 9.8) > 1.0:
            patterns.append({
                "type": "orientation_change",
                "deviation": abs(z_mean - 9.8),
                "confidence": 0.7
            })
        
        return patterns
    
    def _detect_rotation_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect rotation patterns"""
        patterns = []
        
        # Extract values
        x_values = [d.get("x", 0.0) for d in data]
        y_values = [d.get("y", 0.0) for d in data]
        z_values = [d.get("z", 0.0) for d in data]
        
        # Detect rotation
        x_std = statistics.stdev(x_values) if len(x_values) > 1 else 0.0
        y_std = statistics.stdev(y_values) if len(y_values) > 1 else 0.0
        z_std = statistics.stdev(z_values) if len(z_values) > 1 else 0.0
        
        if x_std > 0.05 or y_std > 0.05 or z_std > 0.05:
            patterns.append({
                "type": "rotation",
                "intensity": max(x_std, y_std, z_std),
                "confidence": 0.8
            })
        
        return patterns
    
    def _detect_thermal_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect thermal patterns"""
        patterns = []
        
        # Extract temperatures
        cpu_temps = [d.get("cpu", 0.0) for d in data]
        gpu_temps = [d.get("gpu", 0.0) for d in data]
        
        if cpu_temps:
            cpu_mean = statistics.mean(cpu_temps)
            if cpu_mean > 70.0:
                patterns.append({
                    "type": "high_cpu_temperature",
                    "temperature": cpu_mean,
                    "confidence": 0.9
                })
        
        if gpu_temps:
            gpu_mean = statistics.mean(gpu_temps)
            if gpu_mean > 65.0:
                patterns.append({
                    "type": "high_gpu_temperature",
                    "temperature": gpu_mean,
                    "confidence": 0.9
                })
        
        return patterns
    
    def _detect_fan_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect fan patterns"""
        patterns = []
        
        # Extract fan speeds
        fan1_speeds = [d.get("fan1", 0) for d in data]
        fan2_speeds = [d.get("fan2", 0) for d in data]
        
        if fan1_speeds:
            fan1_mean = statistics.mean(fan1_speeds)
            if fan1_mean > 3000:
                patterns.append({
                    "type": "high_fan_speed",
                    "speed": fan1_mean,
                    "confidence": 0.8
                })
        
        return patterns
    
    def get_patterns(self, sensor_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get detected patterns"""
        if sensor_type:
            return {sensor_type: self.patterns.get(sensor_type, [])}
        return self.patterns
    
    def clear_data(self, sensor_type: Optional[str] = None) -> int:
        """Clear sensor data"""
        if sensor_type:
            if sensor_type in self.data_buffer:
                count = len(self.data_buffer[sensor_type])
                self.data_buffer[sensor_type].clear()
                return count
        else:
            count = sum(len(buffer) for buffer in self.data_buffer.values())
            self.data_buffer.clear()
            return count
        
        return 0

