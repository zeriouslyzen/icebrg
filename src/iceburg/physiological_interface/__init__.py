"""
ICEBURG Physiological Interface
Direct sensor access for physiological state detection and frequency synthesis
"""

from .sensor_interface import MacSensorInterface
from .frequency_synthesizer import PhysiologicalFrequencySynthesizer
# from .physiological_detector_legacy import BrainwaveDetector  # Legacy - use physiological_detector instead
from .earth_connection import EarthConnectionInterface
from .physiological_amplifier import PhysiologicalStateDetector
from .physiological_detector import PhysiologicalStateDetector as PhysiologicalDetector
from .icberg_integration import ICEBURGPhysiologicalIntegration

__all__ = [
    'MacSensorInterface',
    'PhysiologicalFrequencySynthesizer', 
    # 'BrainwaveDetector',  # Legacy - use PhysiologicalDetector instead
    'EarthConnectionInterface',
    'PhysiologicalStateDetector',
    'PhysiologicalDetector',
    'ICEBURGPhysiologicalIntegration'
]
