"""Semantic Scholar Graph API client (paper search, PDF fetch; 1 Graph API req/s in-process)."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import certifi

from ..models import EvidenceCandidate, PdfTextOutcome
from .arxiv import _MAX_METADATA_RESULTS, _MAX_PDF_PAGES
from .pdf_http import fetch_pdf_bytes

_log = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_API_HOST = "api.semanticscholar.org"
_S2_MIN_INTERVAL_SECONDS = 1.0
_S2_USER_AGENT = "AssetOpsBench/1.0 (mailto:admin@example.com)"

_SEARCH_FIELDS = "paperId,title,abstract,year,openAccessPdf,externalIds"

_s2_api_rate_lock = threading.Lock()
_s2_last_api_request_at: float | None = None


def _is_doi_or_non_pdf_landing_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:  # noqa: BLE001
        return True
    if host in {"doi.org", "dx.doi.org"} or host.endswith(".doi.org"):
        return True
    if "link.springer.com" in host and "/article/" in (parsed.path or ""):
        return True
    if "nature.com" in host and "/articles/" in (parsed.path or "") and not (parsed.path or "").lower().endswith(".pdf"):
        return True
    return False


def _is_likely_direct_pdf_url(url: str) -> bool:
    if _is_doi_or_non_pdf_landing_url(url):
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        path = (parsed.path or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if path.endswith(".pdf"):
        return True
    if "arxiv.org" in host and "/pdf/" in path:
        return True
    if host == "pdfs.semanticscholar.org":
        return True
    if "pmc.ncbi.nlm.nih.gov" in host and "pdf" in path:
        return True
    return False


def _arxiv_external_id_to_pdf_url(raw: str) -> str | None:
    aid = raw.strip()
    if not aid:
        return None
    lower = aid.lower()
    if lower.startswith("arxiv:"):
        aid = aid.split(":", 1)[1].strip()
    if aid.startswith("http"):
        try:
            parsed = urllib.parse.urlparse(aid)
            if "arxiv.org" in (parsed.hostname or "").lower() and "/abs/" in (parsed.path or ""):
                tail = (parsed.path or "").split("/abs/", 1)[-1].rstrip("/")
                if tail:
                    return f"https://arxiv.org/pdf/{tail}.pdf"
        except Exception:  # noqa: BLE001
            return None
        return None
    return f"https://arxiv.org/pdf/{aid}.pdf"


def _pdf_url_from_external_ids(ext: object) -> str | None:
    if not isinstance(ext, dict):
        return None
    for key in ("ArXiv", "arXiv", "arxiv"):
        raw = ext.get(key)
        if isinstance(raw, str) and raw.strip():
            return _arxiv_external_id_to_pdf_url(raw)
    return None


def _normalize_open_access_url(url: str) -> str:
    u = url.strip().split("#")[0]
    if "arxiv.org/abs/" not in u:
        return u
    tail = u.split("/abs/", 1)[-1].rstrip("/")
    if not tail:
        return u
    return f"https://arxiv.org/pdf/{tail}.pdf"


def _resolve_pdf_url_for_paper(row: dict) -> str | None:
    oa = row.get("openAccessPdf")
    oa_url: str | None = None
    if isinstance(oa, dict):
        u = oa.get("url")
        if isinstance(u, str) and u.strip():
            oa_url = _normalize_open_access_url(u.strip())

    if oa_url and _is_doi_or_non_pdf_landing_url(oa_url):
        oa_url = None

    ext_pdf = _pdf_url_from_external_ids(row.get("externalIds"))

    if oa_url and _is_likely_direct_pdf_url(oa_url):
        return oa_url

    if ext_pdf:
        return ext_pdf

    if oa_url:
        return oa_url

    return None


def _normalize_s2_query(query: str) -> str:
    return re.sub(r"\s*-\s*", " ", query).strip()


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _is_semantic_scholar_api_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001 — urlparse is defensive for odd strings
        return False
    return host == _S2_API_HOST


def _http_get(url: str, *, timeout: int, ctx: ssl.SSLContext, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        return response.read()


def _semantic_scholar_api_get(url: str, *, timeout: int, ctx: ssl.SSLContext, headers: dict[str, str]) -> bytes:
    global _s2_last_api_request_at
    with _s2_api_rate_lock:
        if _s2_last_api_request_at is not None:
            elapsed = time.monotonic() - _s2_last_api_request_at
            if elapsed < _S2_MIN_INTERVAL_SECONDS:
                time.sleep(_S2_MIN_INTERVAL_SECONDS - elapsed)
        data = _http_get(url, timeout=timeout, ctx=ctx, headers=headers)
        _s2_last_api_request_at = time.monotonic()
        return data


class SemanticScholarExecutor:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        log_writer: Callable[[str, str], None] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self.metadata_requests = 0
        self.pdf_requests = 0
        self._ctx = _ssl_context()
        self._log_writer = log_writer

    def _s2_headers(self) -> dict[str, str]:
        headers = {"User-Agent": _S2_USER_AGENT, "Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _open(self, url: str, timeout: int) -> bytes:
        if not _is_semantic_scholar_api_url(url):
            raise ValueError(f"Expected Semantic Scholar API URL, got {url!r}")
        return _semantic_scholar_api_get(url, timeout=timeout, ctx=self._ctx, headers=self._s2_headers())

    def fetch_metadata(
        self,
        query: str,
        max_results: int = _MAX_METADATA_RESULTS,
    ) -> list[EvidenceCandidate]:
        q = _normalize_s2_query(query)
        limit = max(1, min(max_results, 100))
        params = urllib.parse.urlencode(
            {
                "query": q,
                "limit": str(limit),
                "fields": _SEARCH_FIELDS,
            }
        )
        search_url = f"{_S2_SEARCH_URL}?{params}"
        self.metadata_requests += 1

        try:
            raw = self._open(search_url, timeout=15)
        except urllib.error.HTTPError as exc:
            _log.warning("HTTP error fetching Semantic Scholar paper search for %r: %s", query, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch Semantic Scholar paper search for %r: %s", query, exc)
            return []

        payload_text = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            if self._log_writer:
                self._log_writer(
                    "02_retrieval/paper_search/raw_semantic_scholar.json",
                    json.dumps(
                        {
                            "backend": "semantic_scholar",
                            "query": query,
                            "request_url": search_url,
                            "raw_response": payload_text,
                            "json_decode_error": str(exc),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            _log.warning("Failed to parse Semantic Scholar search JSON for %r: %s", query, exc)
            return []

        if self._log_writer:
            self._log_writer(
                "02_retrieval/paper_search/raw_semantic_scholar.json",
                json.dumps(
                    {
                        "backend": "semantic_scholar",
                        "query": query,
                        "request_url": search_url,
                        "raw_response": payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            return []

        candidates: list[EvidenceCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("paperId") or "").strip()
            title = str(row.get("title") or "No Title").strip().replace("\n", " ")
            abstract_raw = row.get("abstract")
            abstract = (
                str(abstract_raw).strip().replace("\n", " ")
                if abstract_raw is not None
                else ""
            )
            year = row.get("year")
            published = str(year) if year is not None else None

            pdf_url = _resolve_pdf_url_for_paper(row)

            if not pid:
                pid = title or "unknown"

            candidates.append(
                EvidenceCandidate(
                    paper_id=pid,
                    title=title,
                    summary=abstract or "No Summary",
                    query=query,
                    pdf_url=pdf_url,
                    published=published,
                )
            )
            if len(candidates) >= max_results:
                break

        return candidates

    def fetch_pdf_text_detail(
        self, pdf_url: str, max_pages: int = _MAX_PDF_PAGES
    ) -> tuple[str, PdfTextOutcome]:
        self.pdf_requests += 1
        try:
            pdf_bytes = fetch_pdf_bytes(pdf_url, timeout=20)
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
            _log.warning("Failed to fetch or parse PDF %s: %s", pdf_url, exc)
            return "", "fetch_failed"

    def fetch_pdf_text(self, pdf_url: str, max_pages: int = _MAX_PDF_PAGES) -> str:
        text, _ = self.fetch_pdf_text_detail(pdf_url, max_pages)
        return text
