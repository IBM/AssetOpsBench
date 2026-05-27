"""Research digest section headings and merge/per-paper prompts."""

from __future__ import annotations

RESEARCH_DIGEST_SECTION_HEADINGS: tuple[str, ...] = (
    "Condition monitoring, diagnostics, and degradation indicators",
    "Maintenance practices, scheduling, and work-order–relevant context",
    "Sensor modalities, measurements, and signal interpretation",
    "Failure modes, faults, and operational risks",
    "Applicable standards, norms, or industry conventions (when mentioned)",
    "Operator- and manager-style tasks phrased in natural language",
    "Gaps / not supported by this paper",
)

RESEARCH_DIGEST_HEADINGS_MARKDOWN = "\n".join(f"## {title}\n" for title in RESEARCH_DIGEST_SECTION_HEADINGS)

RESEARCH_DIGEST_MERGE_SECTION_HEADINGS: tuple[str, ...] = RESEARCH_DIGEST_SECTION_HEADINGS[:-1]
RESEARCH_DIGEST_MERGE_HEADINGS_MARKDOWN = "\n".join(f"## {title}\n" for title in RESEARCH_DIGEST_MERGE_SECTION_HEADINGS)

RESEARCH_PER_PAPER_PROMPT = """\
You are extracting industrial asset lifecycle management (IALM) knowledge from one academic paper
(condition monitoring, maintenance, sensors, work orders, failure modes, standards) for downstream
scenario generation.

Asset class name: {asset_name}
Canonical asset name: {canonical_asset_name}

Use ONLY the source text below. Do not invent citations, paper titles, or external paper/repository IDs.
If a section has no relevant content in the source, write a short line such as "Not supported by this
paper." under that heading.

Output MUST use exactly these section headings (Markdown ##), in this order:

{headings_markdown}

Under each heading, use short bullet points or 1–2 sentences. No JSON.

--- SOURCE TEXT (may be truncated) ---
{body_text}
--- END SOURCE TEXT ---
"""

RESEARCH_MERGE_PROMPT = """\
You merge several per-paper research digests into one coherent research brief for the same asset class.

Asset class name: {asset_name}
Canonical asset name: {canonical_asset_name}

Goals:
- Deduplicate overlapping facts; reconcile minor contradictions conservatively (prefer explicit wording
  from the digests; if sources conflict, say so briefly).
- Every section below must have substantive content. Where the per-paper digests are empty, sparse, or
  only say things like "Not supported by this paper" for a heading, fill that section to the best of
  your general knowledge for this asset class and typical industrial IALM practice. Do not leave
  sections blank or stub-only when you can reasonably supply domain-appropriate detail; do not
  contradict what the digests do state. When you supplement beyond the papers, keep wording factual
  and typical (not speculative edge cases); you may briefly note that a point is general domain
  context if it was not clearly grounded in the digests.
- Keep content focused on physical industrial assets, operations, and IALM (not generic ML methods,
  cyber-only, or unrelated domains).
- Produce text that will help generate realistic operator/manager natural-language scenarios
  (including plausible sensor interpretations, tasks, and failure contexts).

Output format: Markdown with exactly these top-level section headings (##), in this order:

{merge_headings_markdown}

Do not include paper titles, database IDs, or DOIs in the output.

--- PER-PAPER DIGESTS ---
{per_paper_digests}
--- END PER-PAPER DIGESTS ---
"""
