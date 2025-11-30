"""
ICEBURG Sensors Module
Mac M1 sensor integration
"""

from .camera_interface import CameraInterface
from .image_capture import ImageCapture
from .accelerometer import Accelerometer
from .gyroscope import Gyroscope

__all__ = [
    "CameraInterface",
    "ImageCapture",
    "Accelerometer",
    "Gyroscope",
]

