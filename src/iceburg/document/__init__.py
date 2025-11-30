"""
Iceberg Protocol - Document Workflow Module

This module provides publication-ready document generation capabilities
for creating research manuscripts, abstracts, and pre-registrations.
"""

from .document_workflow import DocumentWorkflow, DocumentElement

__all__ = [
    "DocumentWorkflow",
    "DocumentElement"
]

__version__ = "1.0.0"
__author__ = "Iceberg Protocol Team"
__description__ = "Publication-ready document generation for research outputs"
