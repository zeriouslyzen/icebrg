"""
ICEBURG Compliance Module
GDPR and regulatory compliance
"""

from .gdpr_compliance import GDPRCompliance
from .data_minimization import DataMinimization
from .privacy_by_design import PrivacyByDesign

__all__ = [
    "GDPRCompliance",
    "DataMinimization",
    "PrivacyByDesign",
]

