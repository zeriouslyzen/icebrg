"""
Investigation storage and management for ICEBURG dossiers.
Provides persistence, retrieval, and context tracking for investigations.
"""

from .storage import (
    Investigation,
    InvestigationMetadata,
    InvestigationStore,
    get_investigation_store
)
from .context import (
    InvestigationContext,
    get_active_context,
    set_active_context,
    clear_active_context
)
from .pdf_export import (
    DossierPDFExporter,
    export_investigation_to_pdf,
    REPORTLAB_AVAILABLE
)
from .model_router import (
    ModelRouter,
    get_model_router,
    execute_with_fallback,
    StagedPipeline,
    TaskComplexity
)

__all__ = [
    "Investigation",
    "InvestigationMetadata", 
    "InvestigationStore",
    "get_investigation_store",
    "InvestigationContext",
    "get_active_context",
    "set_active_context",
    "clear_active_context",
    "DossierPDFExporter",
    "export_investigation_to_pdf",
    "REPORTLAB_AVAILABLE",
    "ModelRouter",
    "get_model_router",
    "execute_with_fallback",
    "StagedPipeline",
    "TaskComplexity"
]
