"""ArXiv HTTP client and retrieval keyword constants."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..models import EvidenceCandidate, PdfTextOutcome
from .pdf_http import fetch_pdf_bytes

_log = logging.getLogger(__name__)

_ARXIV_COOLDOWN_SECONDS = 3.1
_ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
_ARXIV_HEADERS = {"User-Agent": "AssetOpsBench/1.0 (mailto:admin@example.com)"}
_MAX_METADATA_RESULTS = 6
_MAX_PDF_PAGES = 10

_STOPWORDS = {
    "a",
    "an",
    "and",
    "asset",
    "class",
    "equipment",
    "for",
    "in",
    "industrial",
    "machine",
    "of",
    "on",
    "or",
    "plant",
    "system",
    "the",
    "to",
}

_DIAGNOSTIC_KEYWORDS = [
    "condition monitoring",
    "diagnostic",
    "diagnostics",
    "fault",
    "fault diagnosis",
    "failure",
    "failure analysis",
    "degradation",
    "health",
    "inspection",
    "maintenance",
    "monitoring",
    "prognostic",
    "prognostics",
    "reliability",
]

_STANDARD_KEYWORDS = [
    "iec",
    "ieee",
    "iso",
    "condition assessment",
    "industry 4.0",
    "iiot",
    "maintenance strategy",
    "reliability centered maintenance",
]

_ML_NEGATIVE_KEYWORDS = [
    "attention mechanism",
    "bert",
    "deep learning",
    "federated learning",
    "foundation model",
    "language model",
    "llm",
    "mamba",
    "neural network",
    "nlp",
    "time series transformer",
    "transformer architecture",
    "vision transformer",
]

_OFF_TARGET_KEYWORDS = [
    "communications",
    "control system",
    "control systems",
    "cyber attack",
    "cyber security",
    "cyberattack",
    "cybersecurity",
    "data architecture",
    "false data injection",
    "market",
    "networking",
    "smart grid",
]

_PHYSICAL_REASON_MARKERS = [
    "physical asset focused",
    "physical equipment focused",
]

_OFF_TARGET_REASON_MARKERS = [
    "not physical asset focused",
    "communications",
    "control system",
    "cyberattack",
    "cybersecurity",
    "data architecture",
    "generic ml",
    "indirect relevance",
    "market",
    "networking",
    "not directly",
    "not entirely physical asset focused",
    "other asset family",
    "somewhat indirect",
    "somewhat relevant",
    "smart-grid",
    "system paper",
    "transformer architecture",
]


class _ArxivExecutor:
    def __init__(
        self,
        cooldown_seconds: float = _ARXIV_COOLDOWN_SECONDS,
        *,
        log_writer: Callable[[str, str], None] | None = None,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_request_at: float | None = None
        self.metadata_requests = 0
        self.pdf_requests = 0
        self._ctx = ssl.create_default_context()
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE
        self._log_writer = log_writer

    def _wait_for_cooldown(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.cooldown_seconds:
            time.sleep(self.cooldown_seconds - elapsed)

    def _open(self, url: str, timeout: int) -> bytes:
        self._wait_for_cooldown()
        req = urllib.request.Request(url, headers=_ARXIV_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as response:
            data = response.read()
        self._last_request_at = time.monotonic()
        return data

    def fetch_metadata(
        self,
        query: str,
        max_results: int = _MAX_METADATA_RESULTS,
    ) -> list[EvidenceCandidate]:
        safe_query = urllib.parse.quote(query)
        url = f"{_ARXIV_BASE_URL}search_query={safe_query}&start=0&max_results={max_results}"
        self.metadata_requests += 1

        try:
            data = self._open(url, timeout=10)
        except urllib.error.HTTPError as exc:
            _log.warning("HTTP error fetching ArXiv metadata for %r: %s", query, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch ArXiv metadata for %r: %s", query, exc)
            return []

        if self._log_writer:
            self._log_writer(
                "02_retrieval/paper_search/raw_arxiv.json",
                json.dumps(
                    {
                        "backend": "arxiv",
                        "query": query,
                        "request_url": url,
                        "content_type": "application/atom+xml",
                        "raw_response": data.decode("utf-8", errors="replace"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            _log.warning("Failed to parse ArXiv XML for %r: %s", query, exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        candidates: list[EvidenceCandidate] = []
        for entry in root.findall("atom:entry", ns):
            raw_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
            paper_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
            title = entry.findtext("atom:title", default="No Title", namespaces=ns)
            summary = entry.findtext("atom:summary", default="No Summary", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns).strip() or None

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                href = link.attrib.get("href")
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = href
                    if pdf_url and not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    break

            candidates.append(
                EvidenceCandidate(
                    paper_id=paper_id or title.strip(),
                    title=title.strip().replace("\n", " "),
                    summary=summary.strip().replace("\n", " "),
                    query=query,
                    pdf_url=pdf_url,
                    published=published,
                )
            )
        return candidates

    def fetch_pdf_text_detail(
        self, pdf_url: str, max_pages: int = _MAX_PDF_PAGES
    ) -> tuple[str, PdfTextOutcome]:
        self.pdf_requests += 1
        try:
            pdf_bytes = fetch_pdf_bytes(pdf_url, timeout=15)
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages: list[str] = []
            for index, page in enumerate(reader.pages):
                if index >= max_pages:
                    break
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages)
            if text.strip():
                return text, "ok"
            return "", "empty_text"
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch or parse ArXiv PDF %s: %s", pdf_url, exc)
            return "", "fetch_failed"

    def fetch_pdf_text(self, pdf_url: str, max_pages: int = _MAX_PDF_PAGES) -> str:
        text, _ = self.fetch_pdf_text_detail(pdf_url, max_pages)
        return text
