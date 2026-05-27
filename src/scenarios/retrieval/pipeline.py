"""LLM-orchestrated evidence retrieval (academic search engine backends)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable

from llm import LLMBackend

from ..models import (
    EvidenceBundle,
    EvidenceCandidate,
    EvidenceSnippet,
    RetrievalAction,
    RetrieverMode,
)
from ..prompts import (
    RETRIEVAL_METADATA_JUDGE_PROMPT,
    RETRIEVAL_QUERY_PLAN_PROMPT_SECTION,
)
from ..prompts.research_digest import RESEARCH_DIGEST_MERGE_SECTION_HEADINGS
from ..text import collapse_whitespace_lower, truncate_title_one_line
from ..utils import parse_llm_json
from .arxiv import (
    _ARXIV_COOLDOWN_SECONDS,
    _ArxivExecutor,
    _DIAGNOSTIC_KEYWORDS,
    _ML_NEGATIVE_KEYWORDS,
    _OFF_TARGET_KEYWORDS,
    _OFF_TARGET_REASON_MARKERS,
    _PHYSICAL_REASON_MARKERS,
    _STANDARD_KEYWORDS,
    _STOPWORDS,
)
from .base import EvidenceMetadataExecutor
from .pdf_http import probe_pdf_url
from .semantic_scholar import SemanticScholarExecutor

_MAX_STEPS = 5
_MAX_QUERIES_PER_STEP = 2
_MAX_CANDIDATE_POOL = 8
_TOP_PDF_DOWNLOADS = 3
_MAX_SNIPPETS_PER_DOC = 3
_RETRIEVAL_LLM_MAX_TOKENS = 4096

_log = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _default_canonical_asset_name(asset_name: str) -> str:
    return collapse_whitespace_lower(asset_name.replace("_", " "))


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            ordered.append(clean)
            seen.add(clean)
    return ordered


def _asset_tokens(asset_name: str) -> list[str]:
    return [
        token
        for token in _tokenise(asset_name)
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _render_queries(queries: list[str]) -> str:
    return "\n".join(f"- {query}" for query in queries) if queries else "(none yet)"


def _render_results_summary(candidates: list[EvidenceCandidate]) -> str:
    if not candidates:
        return "(no judged results yet)"
    lines = []
    for candidate in candidates[:5]:
        lines.append(
            f"- [{candidate.judge_score}/10] paper_id={candidate.paper_id} | {candidate.title} | {candidate.judge_reason}"
        )
    return "\n".join(lines)


def _summarise_metadata_for_judge(candidates: list[EvidenceCandidate]) -> str:
    payload = [
        {
            "paper_id": candidate.paper_id,
            "title": candidate.title,
            "summary": candidate.summary,
            "query": candidate.query,
            "published": candidate.published,
        }
        for candidate in candidates
    ]
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _coerce_action(parsed: object) -> RetrievalAction:
    if not isinstance(parsed, dict):
        raise ValueError("Retrieval planner response could not be parsed as an object.")

    action = str(parsed.get("action", "")).strip().lower()
    if action not in {"search", "finish"}:
        raise ValueError(f"Retrieval planner returned invalid action: {action!r}")

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        raise ValueError("Retrieval planner is missing required non-empty field: 'reason'")

    canonical = str(parsed.get("canonical_asset_name", "")).strip()
    if not canonical:
        raise ValueError("Retrieval planner is missing required non-empty field: 'canonical_asset_name'")

    action_obj = RetrievalAction(
        action=action,
        reason=reason,
        canonical_asset_name=canonical,
        queries=[
            str(query).strip()
            for query in parsed.get("queries", [])
            if str(query).strip()
        ][:_MAX_QUERIES_PER_STEP],
        selected_ids=[
            str(pid).strip()
            for pid in parsed.get("selected_ids", [])
            if str(pid).strip()
        ],
    )

    if action_obj.action == "finish":
        action_obj.queries = []

    return action_obj


def _judge_fallback(
    candidate: EvidenceCandidate,
    canonical_asset_name: str,
) -> tuple[int, str]:
    text = collapse_whitespace_lower(f"{candidate.title} {candidate.summary}")
    asset_phrase = collapse_whitespace_lower(canonical_asset_name)
    asset_tokens = _asset_tokens(canonical_asset_name)

    score = 2
    reasons: list[str] = []

    if asset_phrase and asset_phrase in text:
        score += 4
        reasons.append("physical asset focused; mentions canonical asset")
    else:
        hits = sum(1 for token in asset_tokens if token in text)
        if hits:
            score += min(3, hits)
            reasons.append("physical asset focused; partial asset overlap")

    if any(keyword in text for keyword in _DIAGNOSTIC_KEYWORDS):
        score += 2
        reasons.append("physical asset focused; diagnostics or reliability terms")

    if any(keyword in text for keyword in _OFF_TARGET_KEYWORDS):
        score -= 3
        reasons.append("not physical asset focused; system, cyber, networking, or smart-grid context")

    if any(keyword in text for keyword in _ML_NEGATIVE_KEYWORDS):
        score -= 2
        reasons.append("not physical asset focused; contains generic ML framing")

    score = max(1, min(10, score))
    if not reasons:
        reasons.append("not physical asset focused; fallback metadata relevance estimate")
    reason = "; ".join(reasons)
    return score, reason


def _judge_metadata_batch(
    asset_name: str,
    canonical_asset_name: str,
    candidates: list[EvidenceCandidate],
    llm: LLMBackend,
) -> list[EvidenceCandidate]:
    if not candidates:
        return []

    prompt = RETRIEVAL_METADATA_JUDGE_PROMPT.format(
        asset_name=asset_name,
        canonical_asset_name=canonical_asset_name,
        metadata_entries_json=_summarise_metadata_for_judge(candidates),
    )
    response = llm.generate_with_usage(
        prompt, max_tokens=_RETRIEVAL_LLM_MAX_TOKENS
    )
    parsed, _ = parse_llm_json(response.text)

    scored_by_id: dict[str, tuple[int, str]] = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            paper_id = str(entry.get("paper_id", "")).strip()
            if not paper_id:
                continue
            raw_score = entry.get("score_1_to_10", 0)
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                score = 0
            score = max(1, min(10, score)) if score else 0
            reason = str(entry.get("reason", "")).strip() or "No judge reason provided."
            if score:
                scored_by_id[paper_id] = (score, reason)

    judged: list[EvidenceCandidate] = []
    for candidate in candidates:
        score_reason = scored_by_id.get(candidate.paper_id)
        if score_reason is None:
            score_reason = _judge_fallback(candidate, canonical_asset_name)
        candidate.judge_score = score_reason[0]
        candidate.judge_reason = score_reason[1]
        judged.append(candidate)
    return judged


def _merge_pool(
    pool: dict[str, EvidenceCandidate],
    judged_candidates: list[EvidenceCandidate],
) -> None:
    for candidate in judged_candidates:
        existing = pool.get(candidate.paper_id)
        if existing is None or candidate.judge_score > existing.judge_score:
            pool[candidate.paper_id] = candidate


def _sorted_candidates(pool: dict[str, EvidenceCandidate]) -> list[EvidenceCandidate]:
    return sorted(
        pool.values(),
        key=lambda candidate: (candidate.judge_score, candidate.title.lower()),
        reverse=True,
    )


def _section_slug(section_heading: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", section_heading.lower()).strip("_")
    return slug[:80] if slug else "section"


def _merge_global_pools(
    global_pool: dict[str, EvidenceCandidate],
    section_pool: dict[str, EvidenceCandidate],
) -> None:
    for paper_id, candidate in section_pool.items():
        existing = global_pool.get(paper_id)
        if existing is None or candidate.judge_score > existing.judge_score:
            global_pool[paper_id] = candidate


def _filter_fetchable_candidates(raw: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    kept: list[EvidenceCandidate] = []
    for candidate in raw:
        if not candidate.pdf_url:
            continue
        if probe_pdf_url(candidate.pdf_url):
            kept.append(candidate)
        else:
            _log.debug(
                "Dropped candidate (PDF probe failed): %s | %s",
                candidate.paper_id,
                (candidate.pdf_url or "")[:120],
            )
    return kept


def _fallback_retry_queries_section(
    canonical_asset_name: str,
    query_history: list[str],
    section_heading: str,
) -> list[str]:
    canonical = _default_canonical_asset_name(canonical_asset_name)
    snippet = section_heading.split(",")[0].strip().lower()
    if len(snippet) > 48:
        snippet = snippet[:45] + "..."
    candidates = [
        f"{canonical} condition monitoring {snippet}",
        f"{canonical} fault diagnosis {snippet}",
        f"{canonical} condition assessment {snippet}",
        f"{canonical} maintenance {snippet}",
        f"{canonical} degradation {snippet}",
        f"{canonical} physical measurements {snippet}",
        f"{canonical} sensor monitoring {snippet}",
        f"{canonical} IIoT {snippet}",
    ]
    return [
        query
        for query in _unique_preserve_order(candidates)
        if query not in query_history
    ][:_MAX_QUERIES_PER_STEP]


def _candidate_reason_flags(reason: str) -> tuple[bool, bool]:
    normalized = collapse_whitespace_lower(reason)
    is_off_target = any(marker in normalized for marker in _OFF_TARGET_REASON_MARKERS)
    is_physical = (
        not is_off_target
        and any(marker in normalized for marker in _PHYSICAL_REASON_MARKERS)
    )
    return is_physical, is_off_target


def _candidate_focus_label(candidate: EvidenceCandidate) -> str:
    is_physical, is_off_target = _candidate_reason_flags(candidate.judge_reason)
    if candidate.judge_score >= 8 and is_physical:
        return "accepted: physical asset focused"
    if is_off_target or candidate.judge_score <= 4:
        return "rejected: not physical asset focused"
    return "mixed: uncertain physical asset focus"


def _evaluate_candidate_pool(candidates: list[EvidenceCandidate]) -> tuple[bool, str]:
    if not candidates:
        return False, "No judged candidates are available yet."

    top_candidates = candidates[:5]
    strong_physical = 0
    off_target = 0

    for candidate in top_candidates:
        is_physical, is_off_target = _candidate_reason_flags(candidate.judge_reason)
        if candidate.judge_score >= 8 and is_physical:
            strong_physical += 1
        elif is_off_target or candidate.judge_score <= 4:
            off_target += 1

    if strong_physical < 2:
        return (
            False,
            f"Only {strong_physical} top candidates are clearly physical asset focused; need at least 2 before finishing early.",
        )

    if off_target >= strong_physical:
        return (
            False,
            "Top candidates are still dominated by system, cyber, networking, or ML-context papers.",
        )

    return (
        True,
        f"Top pool contains {strong_physical} clearly physical-asset-focused papers.",
    )


def _extract_snippet_text(text: str, canonical_asset_name: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    keywords = _unique_preserve_order(
        [canonical_asset_name, *_DIAGNOSTIC_KEYWORDS, *_STANDARD_KEYWORDS]
    )
    lowered = compact.lower()
    windows: list[tuple[int, int]] = []
    for keyword in keywords:
        norm = collapse_whitespace_lower(keyword)
        if not norm:
            continue
        idx = lowered.find(norm)
        if idx == -1:
            continue
        start = max(0, idx - 220)
        end = min(len(compact), idx + len(norm) + 220)
        windows.append((start, end))
        if len(windows) >= _MAX_SNIPPETS_PER_DOC:
            break

    if not windows:
        return compact[:900]

    snippets = [compact[start:end].strip() for start, end in windows if compact[start:end].strip()]
    return "\n...\n".join(snippets[:_MAX_SNIPPETS_PER_DOC])


def _select_final_candidates(
    pool: dict[str, EvidenceCandidate],
    preferred_ids: list[str],
) -> tuple[list[EvidenceCandidate], list[str]]:
    ordered_pool = _sorted_candidates(pool)
    ordered_by_id = {candidate.paper_id: candidate for candidate in ordered_pool}

    selected: list[EvidenceCandidate] = []
    selected_ids: list[str] = []
    for pid in preferred_ids:
        candidate = ordered_by_id.get(pid)
        if candidate and pid not in selected_ids:
            selected.append(candidate)
            selected_ids.append(pid)

    for candidate in ordered_pool:
        if candidate.paper_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.append(candidate.paper_id)
        if len(selected) >= _TOP_PDF_DOWNLOADS:
            break

    return selected[:_TOP_PDF_DOWNLOADS], selected_ids[:_TOP_PDF_DOWNLOADS]


def _render_step_log(
    step_number: int,
    action: RetrievalAction,
    new_queries: list[str],
    fetched_candidates: list[EvidenceCandidate],
    pool: dict[str, EvidenceCandidate],
    canonical_asset_name: str,
    *,
    section_heading: str | None = None,
    metadata_rows_before_probe: int | None = None,
) -> str:
    lines = []
    if section_heading:
        lines.append(f"Section: {section_heading}")
    lines.extend(
        [
            f"Step: {step_number}/{_MAX_STEPS}",
            f"Action: {action.action}",
            f"Canonical Asset: {canonical_asset_name}",
            f"Reason: {action.reason}",
            "",
            "Queries:",
        ]
    )
    if new_queries:
        lines.extend(f"- {query}" for query in new_queries)
    else:
        lines.append("(no new queries)")

    lines.extend(["", "Fetched Metadata:"])
    if metadata_rows_before_probe is not None:
        lines.append(
            f"(metadata rows: {metadata_rows_before_probe}, "
            f"fetchable PDF after probe: {len(fetched_candidates)})"
        )
    if fetched_candidates:
        for candidate in fetched_candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] {candidate.title}"
            )
            lines.append(f"  paper_id: {candidate.paper_id}")
            lines.append(f"  query: {candidate.query}")
            lines.append(f"  assessment: {_candidate_focus_label(candidate)}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(no metadata fetched)")

    lines.extend(["", "Current Top Candidates:"])
    top_candidates = _sorted_candidates(pool)[:5]
    if top_candidates:
        for candidate in top_candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] paper_id={candidate.paper_id} | {candidate.title}"
            )
            lines.append(f"  assessment: {_candidate_focus_label(candidate)}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(no judged candidates in pool)")

    focused, focus_reason = _evaluate_candidate_pool(top_candidates)
    lines.extend(
        [
            "",
            f"Physical Asset Focused: {'yes' if focused else 'no'}",
            f"Focus Summary: {focus_reason}",
        ]
    )

    return "\n".join(lines)


def _render_final_log(
    bundle: EvidenceBundle,
    selected_candidates: list[EvidenceCandidate],
) -> str:
    focused, focus_reason = _evaluate_candidate_pool(bundle.candidates)
    lines = [
        f"Asset: {bundle.asset_name}",
        f"Canonical Asset: {bundle.canonical_asset_name}",
        f"Physical Asset Focused: {'yes' if focused else 'no'}",
        "",
        "Query History:",
    ]
    if bundle.query_history:
        lines.extend(f"- {query}" for query in bundle.query_history)
    else:
        lines.append("(none)")

    lines.extend(["", "Top Judged Candidates:"])
    if bundle.candidates:
        for candidate in bundle.candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] paper_id={candidate.paper_id} | {candidate.title}"
            )
            lines.append(f"  query: {candidate.query}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(none)")

    lines.extend(["", "Selected PDFs:"])
    if selected_candidates:
        for candidate in selected_candidates:
            lines.append(
                f"- paper_id={candidate.paper_id} | {candidate.title}"
            )
            lines.append(f"  pdf: {candidate.pdf_url or '(no pdf url)'}")
    else:
        lines.append("(none)")

    lines.extend(["", f"Final Pool Assessment: {focus_reason}"])

    return "\n".join(lines)


def _retrieve_for_section(
    asset_name: str,
    canonical_asset_name: str,
    section_heading: str,
    section_slug: str,
    server_desc: dict[str, str],
    llm: LLMBackend,
    executor: EvidenceMetadataExecutor,
    log_writer: Callable[[str, str], None] | None,
    on_academic_query: Callable[[str, int], None] | None,
) -> tuple[str, dict[str, EvidenceCandidate], list[str]]:
    query_history: list[str] = []
    candidate_pool: dict[str, EvidenceCandidate] = {}
    log_prefix = f"02_retrieval/paper_search/{section_slug}"

    for step_number in range(1, _MAX_STEPS + 1):
        current_summary = _render_results_summary(_sorted_candidates(candidate_pool))
        previous_queries = _render_queries(query_history)
        prompt = RETRIEVAL_QUERY_PLAN_PROMPT_SECTION.format(
            asset_name=asset_name,
            step_number=step_number,
            max_steps=_MAX_STEPS,
            canonical_asset_name=canonical_asset_name,
            tool_descriptions=json.dumps(server_desc, indent=2),
            previous_queries=previous_queries,
            current_results_summary=current_summary,
            section_heading=section_heading,
        )
        response = llm.generate_with_usage(
            prompt, max_tokens=_RETRIEVAL_LLM_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        action = _coerce_action(parsed)
        canonical_asset_name = action.canonical_asset_name

        if action.action == "finish" and candidate_pool:
            pool_focused, focus_reason = _evaluate_candidate_pool(
                _sorted_candidates(candidate_pool)
            )
            if pool_focused or step_number == _MAX_STEPS:
                if log_writer:
                    log_writer(
                        f"{log_prefix}/step_{step_number:02d}.txt",
                        _render_step_log(
                            step_number=step_number,
                            action=action,
                            new_queries=[],
                            fetched_candidates=[],
                            pool=candidate_pool,
                            canonical_asset_name=canonical_asset_name,
                            section_heading=section_heading,
                        ),
                    )
                break

            action = RetrievalAction(
                action="search",
                reason=f"Early finish blocked. {focus_reason}",
                canonical_asset_name=canonical_asset_name,
                queries=_fallback_retry_queries_section(
                    canonical_asset_name, query_history, section_heading
                ),
            )

        new_queries = [
            query
            for query in _unique_preserve_order(action.queries)
            if query not in query_history
        ][: _MAX_QUERIES_PER_STEP]

        if not new_queries:
            if log_writer:
                log_writer(
                    f"{log_prefix}/step_{step_number:02d}.txt",
                    _render_step_log(
                        step_number=step_number,
                        action=RetrievalAction(
                            action="finish",
                            reason=action.reason
                            or (
                                "No new queries remained; keeping the best-so-far pool."
                                if candidate_pool
                                else "No new queries from planner and no evidence pool yet."
                            ),
                            canonical_asset_name=canonical_asset_name,
                        ),
                        new_queries=[],
                        fetched_candidates=[],
                        pool=candidate_pool,
                        canonical_asset_name=canonical_asset_name,
                        section_heading=section_heading,
                    ),
                )
            break

        query_history.extend(new_queries)
        merged_raw: list[EvidenceCandidate] = []
        for query in new_queries:
            rows = executor.fetch_metadata(query)
            merged_raw.extend(rows)
            if on_academic_query:
                on_academic_query(query, len(rows))

        raw_count = len(merged_raw)
        fetchable = _filter_fetchable_candidates(merged_raw)

        judged_candidates = _judge_metadata_batch(
            asset_name=asset_name,
            canonical_asset_name=canonical_asset_name,
            candidates=fetchable,
            llm=llm,
        )
        _merge_pool(candidate_pool, judged_candidates)

        if log_writer:
            log_writer(
                f"{log_prefix}/step_{step_number:02d}.txt",
                _render_step_log(
                    step_number=step_number,
                    action=action,
                    new_queries=new_queries,
                    fetched_candidates=judged_candidates,
                    pool=candidate_pool,
                    canonical_asset_name=canonical_asset_name,
                    section_heading=section_heading,
                    metadata_rows_before_probe=raw_count,
                ),
            )

    return canonical_asset_name, candidate_pool, query_history


def _make_executor(
    retriever: RetrieverMode,
    log_writer: Callable[[str, str], None] | None = None,
) -> EvidenceMetadataExecutor:
    if retriever == "semantic_scholar":
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        return SemanticScholarExecutor(api_key=key, log_writer=log_writer)
    return _ArxivExecutor(cooldown_seconds=_ARXIV_COOLDOWN_SECONDS, log_writer=log_writer)


def retrieve_asset_evidence(
    asset_name: str,
    server_desc: dict[str, str],
    llm: LLMBackend,
    log_writer: Callable[[str, str], None] | None = None,
    *,
    retriever: RetrieverMode = "arxiv",
    on_pdf_fetched: Callable[[EvidenceCandidate], None] | None = None,
    on_pdf_workflow: Callable[[EvidenceCandidate, str, str], None] | None = None,
    on_academic_query: Callable[[str, int], None] | None = None,
) -> EvidenceBundle:
    executor = _make_executor(retriever, log_writer=log_writer)
    canonical_asset_name = _default_canonical_asset_name(asset_name)
    query_history: list[str] = []
    candidate_pool: dict[str, EvidenceCandidate] = {}

    for section_heading in RESEARCH_DIGEST_MERGE_SECTION_HEADINGS:
        section_slug = _section_slug(section_heading)
        canonical_asset_name, section_pool, section_queries = _retrieve_for_section(
            asset_name=asset_name,
            canonical_asset_name=canonical_asset_name,
            section_heading=section_heading,
            section_slug=section_slug,
            server_desc=server_desc,
            llm=llm,
            executor=executor,
            log_writer=log_writer,
            on_academic_query=on_academic_query,
        )
        for query in section_queries:
            query_history.append(f"[{section_heading}] {query}")
        _merge_global_pools(candidate_pool, section_pool)

    top_candidates = _sorted_candidates(candidate_pool)[:_MAX_CANDIDATE_POOL]
    selected_candidates, selected_candidate_ids = _select_final_candidates(
        candidate_pool,
        [],
    )

    snippets: list[EvidenceSnippet] = []
    for candidate in selected_candidates:
        source = "summary"
        text = candidate.summary
        if candidate.pdf_url:
            pdf_text, pdf_outcome = executor.fetch_pdf_text_detail(candidate.pdf_url)
            if pdf_outcome == "ok":
                source = "pdf"
                text = pdf_text
                if on_pdf_workflow:
                    on_pdf_workflow(candidate, "snippet", "ok")
                if on_pdf_fetched:
                    on_pdf_fetched(candidate)
            else:
                _log.info(
                    "Using abstract/summary for evidence snippet (PDF unavailable) — %s | %s",
                    candidate.paper_id,
                    truncate_title_one_line(candidate.title),
                )
                if on_pdf_workflow:
                    on_pdf_workflow(candidate, "snippet", pdf_outcome)
        elif on_pdf_workflow:
            on_pdf_workflow(candidate, "snippet", "no_pdf_url")
        snippet_text = _extract_snippet_text(text, canonical_asset_name)
        if not snippet_text:
            continue
        snippets.append(
            EvidenceSnippet(
                paper_id=candidate.paper_id,
                title=candidate.title,
                url=candidate.pdf_url,
                source=source,
                text=snippet_text,
            )
        )

    bundle = EvidenceBundle(
        asset_name=asset_name,
        canonical_asset_name=canonical_asset_name,
        query_history=query_history,
        selected_candidate_ids=selected_candidate_ids,
        candidates=top_candidates,
        snippets=snippets,
    )

    if log_writer:
        log_writer(
            "02_retrieval/paper_search/summary.txt",
            _render_final_log(bundle, selected_candidates),
        )

    return bundle
