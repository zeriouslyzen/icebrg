"""
Visual IR Backends
Multi-target compilation from Visual IR to various UI frameworks
"""

from enum import Enum


class BackendType(Enum):
    """Supported backend types"""
    HTML5 = "html5"
    REACT = "react"
    SWIFTUI = "swiftui"
    FLUTTER = "flutter"


__all__ = ['BackendType']

