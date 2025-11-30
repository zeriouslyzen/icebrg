"""
Accelerometer
Mac M1 accelerometer data access
"""

from typing import Any, Dict, Optional, List
import time
from ..physiological_interface.real_sensor_interface import get_real_sensors


class Accelerometer:
    """Mac M1 accelerometer interface"""
    
    def __init__(self):
        self.sensors = get_real_sensors()
        self.is_monitoring = False
    
    async def start_monitoring(self) -> bool:
        """Start accelerometer monitoring"""
        try:
            await self.sensors.start_monitoring()
            self.is_monitoring = True
            return True
        except Exception:
            return False
    
    async def stop_monitoring(self) -> bool:
        """Stop accelerometer monitoring"""
        try:
            await self.sensors.stop_monitoring()
            self.is_monitoring = False
            return True
        except Exception:
            return False
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest accelerometer data"""
        latest = self.sensors.get_latest_data()
        return latest.get("accelerometer")
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all accelerometer data"""
        all_data = self.sensors.get_all_data()
        return all_data.get("accelerometer", [])
    
    def get_movement_pattern(self) -> Dict[str, Any]:
        """Analyze movement pattern"""
        data = self.get_all_data()
        
        if not data:
            return {"error": "No data available"}
        
        # Calculate movement statistics
        x_values = [d.get("x", 0.0) for d in data]
        y_values = [d.get("y", 0.0) for d in data]
        z_values = [d.get("z", 0.0) for d in data]
        
        return {
            "x_mean": sum(x_values) / len(x_values) if x_values else 0.0,
            "y_mean": sum(y_values) / len(y_values) if y_values else 0.0,
            "z_mean": sum(z_values) / len(z_values) if z_values else 0.0,
            "x_std": self._calculate_std(x_values),
            "y_std": self._calculate_std(y_values),
            "z_std": self._calculate_std(z_values),
            "movement_detected": self._detect_movement(data),
            "sample_count": len(data)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _detect_movement(self, data: List[Dict[str, Any]]) -> bool:
        """Detect if movement is present"""
        if len(data) < 2:
            return False
        
        # Check for significant changes
        x_values = [d.get("x", 0.0) for d in data]
        y_values = [d.get("y", 0.0) for d in data]
        z_values = [d.get("z", 0.0) for d in data]
        
        x_range = max(x_values) - min(x_values) if x_values else 0.0
        y_range = max(y_values) - min(y_values) if y_values else 0.0
        z_range = max(z_values) - min(z_values) if z_values else 0.0
        
        # Movement detected if range > threshold
        threshold = 0.1
        return x_range > threshold or y_range > threshold or z_range > threshold
    
    def get_status(self) -> Dict[str, Any]:
        """Get accelerometer status"""
        return {
            "is_monitoring": self.is_monitoring,
            "latest_data": self.get_latest_data(),
            "sample_count": len(self.get_all_data())
        }

