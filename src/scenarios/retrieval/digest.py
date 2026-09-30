"""Synthesize merged research digest from retrieved papers."""

from __future__ import annotations

import logging
from collections.abc import Callable

from llm import LLMBackend

from ..models import EvidenceBundle, EvidenceCandidate, RetrieverMode
from ..text import truncate_title_one_line
from ..prompts.research_digest import (
    RESEARCH_DIGEST_HEADINGS_MARKDOWN,
    RESEARCH_DIGEST_MERGE_HEADINGS_MARKDOWN,
    RESEARCH_MERGE_PROMPT,
    RESEARCH_PER_PAPER_PROMPT,
)
from .arxiv import _MAX_PDF_PAGES
from .pipeline import _make_executor

_log = logging.getLogger(__name__)

_MAX_DIGEST_BODY_CHARS = 28_000
_DIGEST_PER_PAPER_MAX_TOKENS = 4096
_DIGEST_MERGE_MAX_TOKENS = 4096


def _candidates_for_digest(bundle: EvidenceBundle) -> list[EvidenceCandidate]:
    by_id = {c.paper_id: c for c in bundle.candidates}
    ordered: list[EvidenceCandidate] = []
    for aid in bundle.selected_candidate_ids:
        if aid in by_id:
            ordered.append(by_id[aid])
    if ordered:
        return ordered
    return list(bundle.candidates[:3])


def _truncate_body(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_DIGEST_BODY_CHARS:
        return text
    return text[:_MAX_DIGEST_BODY_CHARS] + "\n\n[... truncated for digest input ...]"


def _body_text_for_candidate(
    executor,
    candidate: EvidenceCandidate,
    *,
    on_pdf_fetched: Callable[[EvidenceCandidate], None] | None = None,
    on_pdf_workflow: Callable[[EvidenceCandidate, str, str], None] | None = None,
) -> tuple[str, str]:
    pdf_unavailable = False
    if not candidate.pdf_url:
        if on_pdf_workflow:
            on_pdf_workflow(candidate, "digest", "no_pdf_url")
    else:
        pdf_text, pdf_outcome = executor.fetch_pdf_text_detail(
            candidate.pdf_url, max_pages=_MAX_PDF_PAGES
        )
        if pdf_outcome == "ok":
            if on_pdf_fetched:
                on_pdf_fetched(candidate)
            if on_pdf_workflow:
                on_pdf_workflow(candidate, "digest", "ok")
            return "pdf", _truncate_body(pdf_text)
        pdf_unavailable = True
        if on_pdf_workflow:
            on_pdf_workflow(candidate, "digest", pdf_outcome)

    summary = (candidate.summary or "").strip()
    if summary:
        if pdf_unavailable:
            _log.info(
                "Using abstract text (PDF unavailable) — %s | %s",
                candidate.paper_id,
                truncate_title_one_line(candidate.title),
            )
        return "abstract", _truncate_body(summary)

    if pdf_unavailable:
        _log.info(
            "No extractable text (PDF unavailable, empty abstract) — %s | %s",
            candidate.paper_id,
            truncate_title_one_line(candidate.title),
        )
    else:
        _log.info(
            "No extractable text (no PDF URL, empty abstract) — %s | %s",
            candidate.paper_id,
            truncate_title_one_line(candidate.title),
        )
    return "abstract", "(no extractable text)"


def synthesize_research_digest(
    bundle: EvidenceBundle,
    llm: LLMBackend,
    *,
    retriever: RetrieverMode = "arxiv",
    log_writer: Callable[[str, str], None] | None = None,
    on_pdf_fetched: Callable[[EvidenceCandidate], None] | None = None,
    on_pdf_workflow: Callable[[EvidenceCandidate, str, str], None] | None = None,
) -> str:
    executor = _make_executor(retriever)
    asset_name = bundle.asset_name
    canonical = bundle.canonical_asset_name
    candidates = _candidates_for_digest(bundle)

    if not candidates:
        merged = (
            "## Note\n\nNo ranked papers were available after retrieval; "
            "the research digest is empty. Stay conservative in the asset profile."
        )
        if log_writer:
            log_writer("02_retrieval/paper_digest/merged.txt", merged)
        return merged

    per_paper_digests: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        source_kind, body_text = _body_text_for_candidate(
            executor,
            candidate,
            on_pdf_fetched=on_pdf_fetched,
            on_pdf_workflow=on_pdf_workflow,
        )
        prompt = RESEARCH_PER_PAPER_PROMPT.format(
            asset_name=asset_name,
            canonical_asset_name=canonical,
            headings_markdown=RESEARCH_DIGEST_HEADINGS_MARKDOWN,
            body_text=body_text,
        )
        digest = llm.generate(prompt, max_tokens=_DIGEST_PER_PAPER_MAX_TOKENS)
        block = (
            f"### Paper {index} (source: {source_kind})\n{digest.strip()}"
        )
        per_paper_digests.append(block)
        if log_writer:
            log_writer(f"02_retrieval/paper_digest/per_paper_{index:02d}.txt", digest.strip())

    merge_prompt = RESEARCH_MERGE_PROMPT.format(
        asset_name=asset_name,
        canonical_asset_name=canonical,
        merge_headings_markdown=RESEARCH_DIGEST_MERGE_HEADINGS_MARKDOWN,
        per_paper_digests="\n\n".join(per_paper_digests),
    )
    merged = llm.generate(merge_prompt, max_tokens=_DIGEST_MERGE_MAX_TOKENS).strip()
    if log_writer:
        log_writer("02_retrieval/paper_digest/merged.txt", merged)
    return merged
