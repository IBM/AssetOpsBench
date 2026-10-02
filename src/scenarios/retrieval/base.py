"""Protocol for evidence metadata executors (ArXiv, Semantic Scholar, etc.)."""

from __future__ import annotations

from typing import Protocol

from ..models import EvidenceCandidate, PdfTextOutcome


class EvidenceMetadataExecutor(Protocol):
    def fetch_metadata(self, query: str, max_results: int = 6) -> list[EvidenceCandidate]:
        ...

    def fetch_pdf_text_detail(
        self, pdf_url: str, max_pages: int = 10
    ) -> tuple[str, PdfTextOutcome]:
        ...

    def fetch_pdf_text(self, pdf_url: str, max_pages: int = 10) -> str:
        ...
