"""Prompts used by the bounded academic-search retrieval workflow."""

RETRIEVAL_QUERY_PLAN_PROMPT = """\
You are a bounded ReAct retrieval agent for industrial asset research.

You do not call an academic search engine directly. The host runs searches for you and returns paper metadata summaries.
At each step, decide whether to search again or finish with the best candidates found so far.

Asset Name: {asset_name}
Current Step: {step_number} of {max_steps}
Current Canonical Asset Name: {canonical_asset_name}

Available Subagent Tools:
{tool_descriptions}

Previous Queries:
{previous_queries}

Current Top Judged Results:
{current_results_summary}

Return ONLY a raw JSON object in this shape:
{{
    "action": "search" or "finish",
    "reason": "brief explanation",
    "canonical_asset_name": "canonical industrial equipment term",
    "queries": ["1 to 2 new search queries for the academic search engine if action=search, otherwise []"],
    "selected_ids": ["preferred paper_id values if action=finish, otherwise []"]
}}

Constraints:
- Keep the process generic for any industrial asset class.
- An industrial asset class is a physical piece of equipment or subsystem that is monitored, maintained, and can fail.
- Examples of industrial asset classes: chiller, air handling unit, pump, motor, power transformer.
- Context words like smart grid, plant, facility, building, line, or substation describe the deployment environment, not the asset class itself.
- If the asset name is ambiguous, resolve it toward the physical equipment noun phrase, not an ML concept and not a broader system context.
- Normalize queries toward the physical equipment itself. For example, "smart grid transformer" should be treated as a transformer or power transformer in a smart-grid context, not as the smart grid itself.
- Queries should focus on physical asset monitoring, diagnostics, failures, maintenance, condition assessment, degradation, or reliability.
- When it fits the asset class, prefer query ingredients such as: Industry 4.0 or IIoT in a plant context; physical sensors (e.g. vibration, temperature, pressure, current, partial discharge); and relevant standards (ISO, IEC, IEEE) only when tied to that equipment—do not stuff acronyms into every query.
- Prefer equipment-specific queries such as "power transformer condition monitoring" or "power transformer fault diagnosis".
- Avoid broad or ambiguous queries like "transformer reliability" or "smart grid transformer monitoring" when a more equipment-specific query is possible.
- Do not drift into smart-grid cyberattacks, communications, networking, markets, data architecture, control systems, or generic ML "Transformer" papers.
- When action is "search", propose 1 to 2 short queries only.
- When action is "finish", prefer the strongest already judged metadata and do not propose new queries.
- Only finish early if the current top results already contain at least two clearly physical-asset-focused papers.
- Do not use advanced boolean syntax.
- Do not invent paper_id values that are not present in Current Top Judged Results.
- Output ONLY raw JSON.
"""

RETRIEVAL_QUERY_PLAN_PROMPT_SECTION = """\
You are a bounded ReAct retrieval agent for industrial asset research.

You do not call an academic search engine directly. The host runs searches for you and returns paper metadata summaries.
At each step, decide whether to search again or finish with the best candidates found so far.

This retrieval pass is scoped to ONE subsection of a downstream research digest. Bias every search query toward
that facet of knowledge (not generic background). You will run again for other subsections separately.

Section focus (this pass only): {section_heading}

Asset Name: {asset_name}
Current Step: {step_number} of {max_steps}
Current Canonical Asset Name: {canonical_asset_name}

Available Subagent Tools:
{tool_descriptions}

Previous Queries (this section only):
{previous_queries}

Current Top Judged Results (this section only):
{current_results_summary}

Return ONLY a raw JSON object in this shape:
{{
    "action": "search" or "finish",
    "reason": "brief explanation",
    "canonical_asset_name": "canonical industrial equipment term",
    "queries": ["1 to 2 new search queries for the academic search engine if action=search, otherwise []"],
    "selected_ids": ["preferred paper_id values if action=finish, otherwise []"]
}}

Constraints:
- Keep the process generic for any industrial asset class.
- An industrial asset class is a physical piece of equipment or subsystem that is monitored, maintained, and can fail.
- Examples of industrial asset classes: chiller, air handling unit, pump, motor, power transformer.
- Context words like smart grid, plant, facility, building, line, or substation describe the deployment environment, not the asset class itself.
- If the asset name is ambiguous, resolve it toward the physical equipment noun phrase, not an ML concept and not a broader system context.
- Normalize queries toward the physical equipment itself. For example, "smart grid transformer" should be treated as a transformer or power transformer in a smart-grid context, not as the smart grid itself.
- Queries in this pass must align with the section focus above (e.g. maintenance scheduling vs sensor modalities vs failure modes vs standards). Still describe the physical asset explicitly.
- When it fits the asset class, prefer query ingredients such as: Industry 4.0 or IIoT in a plant context; physical sensors (e.g. vibration, temperature, pressure, current, partial discharge); and relevant standards (ISO, IEC, IEEE) only when tied to that equipment—do not stuff acronyms into every query.
- Prefer equipment-specific queries such as "power transformer condition monitoring" or "power transformer fault diagnosis".
- Avoid broad or ambiguous queries like "transformer reliability" or "smart grid transformer monitoring" when a more equipment-specific query is possible.
- Do not drift into smart-grid cyberattacks, communications, networking, markets, data architecture, control systems, or generic ML "Transformer" papers.
- When action is "search", propose 1 to 2 short queries only.
- When action is "finish", prefer the strongest already judged metadata and do not propose new queries.
- Only finish early if the current top results already contain at least two clearly physical-asset-focused papers.
- Do not use advanced boolean syntax.
- Do not invent paper_id values that are not present in Current Top Judged Results.
- Output ONLY raw JSON.
"""

RETRIEVAL_METADATA_JUDGE_PROMPT = """\
You are a relevance judge for paper metadata from an academic search engine, retrieved for industrial asset scenario generation.

Your task is to score how useful each metadata entry is for building an asset profile for the physical industrial asset below.
Judge from the title and summary only.

Asset Name: {asset_name}
Canonical Asset Name: {canonical_asset_name}

Metadata Entries:
{metadata_entries_json}

Return ONLY a raw JSON array. One object per metadata entry:
[
  {{
    "paper_id": "entry id (must match the paper_id in the metadata entry)",
    "score_1_to_10": 8,
    "reason": "short reason"
  }}
]

Scoring guide:
- Core question: would this paper help define the physical asset class itself and its maintenance, monitoring, sensing, degradation, diagnostics, or failure behavior?
- High score when the paper is clearly about the physical asset and would help populate sensor mappings (modalities, placement, or signal interpretation), known failure modes, relevant tools, maintenance workflows, or applicable standards for that equipment.
- 9-10: directly about the physical asset and clearly useful for condition monitoring, fault diagnosis, degradation, insulation or thermal behavior, reliability, maintenance, standards (ISO/IEC/IEEE where relevant), physical measurements, or industrial sensing/IIoT for the asset—not generic ML method papers
- 6-8: relevant to the physical asset but somewhat indirect, narrower than ideal, or missing strong maintenance or diagnostics detail
- 3-5: mixed relevance, generic system context, or only weakly connected to the physical asset itself
- 1-2: not about the physical asset itself, clearly about a different asset family, or dominated by unrelated ML/system context
- Low-score examples:
  - smart-grid cyberattacks or false-data-injection papers
  - communications or networking papers
  - grid data architecture, market, or control-system papers
  - generic ML "Transformer" architecture papers
  - papers about other asset families such as rotating machinery unless the target asset is actually that equipment
- High-score examples:
  - condition monitoring
  - fault diagnosis
  - degradation, insulation, or thermal behavior
  - reliability, maintenance, or standards
  - sensors and measurements for the physical equipment (including vibration, thermal, electrical, or process variables as they relate to the asset)
  - Industry 4.0 or IIoT when the work centers on the physical asset and its monitoring, not on abstract IT architecture alone
- In every reason, explicitly include either the phrase "physical asset focused" or the phrase "not physical asset focused".
- If the paper is off-target, say why in physical-asset terms.

Be strict about generic smart-grid, networking, cybersecurity, market, or ML-architecture papers when the asset is a physical piece of equipment.
Output ONLY raw JSON.
"""
