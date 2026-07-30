"""Offline-only routing diagnostics for TASK-RAG-002A.

Nothing in this package is imported by the production answer pipeline.
"""

from reg_rag.shadow_routing.document_family_resolver import DocumentFamilyResolver
from reg_rag.shadow_routing.query_anchor_parser import QueryAnchorParser
from reg_rag.shadow_routing.shadow_router import ShadowRouter
from reg_rag.shadow_routing.table_template_fingerprint import TableTemplateFingerprinter
from reg_rag.shadow_routing.version_period_resolver import VersionPeriodResolver

__all__ = [
    "DocumentFamilyResolver",
    "QueryAnchorParser",
    "ShadowRouter",
    "TableTemplateFingerprinter",
    "VersionPeriodResolver",
]
