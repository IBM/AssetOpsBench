"""Prompt for constructing study-backed asset profile enrichment."""

PROFILE_BUILDER_PROMPT = """\
You are an expert reliability engineer preparing study-backed enrichment for an Asset Profile used in scenario generation.

The system will compose the final Asset Profile in code. Your job is to return only the study-derived fields that are not already deterministically grounded from CouchDB discovery.

Asset Name: {asset_name}
Generation Mode: {generation_mode}

Grounded Coverage Summary:
{grounding_summary_json}

Research Digest:
{research_digest}

Available Subagent Tools:
{tool_descriptions}

Task:
Produce a JSON object matching the following schema exactly.
CRITICAL: Output ONLY the raw JSON object. Do NOT include markdown code blocks, Python code, or any conversational preamble.

Rules:
- Keep the response concise. Do NOT restate asset instances, asset_name, or generation_mode.
- In open_form mode, treat grounded live coverage as the source of truth for concrete site names, asset ids, sensor keys (listed under `iot_sensors` / `vibration_sensors` in the summary), per-asset time ranges, and timestamps.
- In closed_form mode, assume scenarios must be self-contained: the query text should list explicit sensor readings (sensor name, numeric value, unit), and may include rules or dataset names when relevant.
- Use the research digest as the primary source of truth for asset-class understanding, operator workflows, monitoring practices, and standards.
- Do NOT return `failure_sensor_mapping` or `sensor_failure_mapping`.
- Use `iot_sensors` and `vibration_sensors` as separate lists of objects, each with `name` and `description` (what it measures). The system merges these with sensor name strings from the grounding summary in code (every IoT name from the summary must appear under `iot_sensors`; every vibration name from the summary under `vibration_sensors`).
- Use `failure_modes` the same way as before: list of `{{"key": "...", "description": "..."}}`. The system merges with grounded failure modes and F2S/S2F mappings in code (mappings use keys only).
- Include multiple relevant tools per focus when the toolset clearly supports more than one important action for that focus.
- Use an empty array [] for `vibration_sensors` when vibration sensing does not apply; use [] for `iot_sensors` only when the asset truly has no IoT channels in the grounded summary.
- Only include sensor descriptions, failure mode descriptions, and task types that are clearly supported by the research digest, grounded coverage summary, or tool definitions.
- If the research digest is sparse, stay conservative rather than inventing highly specific details.
- The profile must represent the physical industrial asset class itself, not surrounding digital, networking, cyber, or facility infrastructure.
- Add realistic tasks from the point of view of industrial asset operator and the operator's manager.
- Include vibration tools only when vibration analysis is relevant to the asset class or grounded data; otherwise use [] for "vibration".

{{
    "description": "Short summary of the asset class (1-2 sentences).",
    "iot_sensors": [
        {{"name": "sensor_name", "description": "what it measures"}}
    ],
    "vibration_sensors": [
        {{"name": "sensor_name", "description": "what it measures"}}
    ],
    "failure_modes": [
        {{"key": "failure_mode_name", "description": "why it matters for this asset"}}
    ],
    "relevant_tools": {{
        "iot": [
            {{"name": "tool_name_a", "reason": "why this tool matters for the asset"}},
            {{"name": "tool_name_b", "reason": "why this tool matters for the asset"}}
        ],
        "fmsr": [
            {{"name": "tool_name_a", "reason": "why this tool matters for the asset"}},
            {{"name": "tool_name_b", "reason": "why this tool matters for the asset"}}
        ],
        "tsfm": [
            {{"name": "tool_name_a", "reason": "why this tool matters for the asset"}},
            {{"name": "tool_name_b", "reason": "why this tool matters for the asset"}}
        ],
        "wo": [
            {{"name": "tool_name_a", "reason": "why this tool matters for the asset"}},
            {{"name": "tool_name_b", "reason": "why this tool matters for the asset"}}
        ],
        "vibration": [
            {{"name": "tool_name_a", "reason": "why this tool matters for the asset"}},
            {{"name": "tool_name_b", "reason": "why this tool matters for the asset"}}
        ]
    }},
    "operator_tasks": ["task phrased from the operator point of view"],
    "manager_tasks": ["task phrased from the manager point of view"]
}}
"""
