"""Prompt for allocating scenario counts across subagents."""

BUDGET_ALLOCATOR_PROMPT = """\
You are an expert Scenario Strategy Consultant for AssetOps Bench.

Given an Asset Profile and a total number of scenarios to generate ({total_scenarios}), your task is to allocate this budget across the following six focus areas:
1. iot: Focusing on sensor data and basic telemetry.
2. fmsr: Focusing on failure modes and structural reliability.
3. tsfm: Focusing on time-series analysis and technical maintenance.
4. wo: Focusing on actual maintenance execution and work orders.
5. vibration: Focusing on vibration diagnostics, severity assessment, FFT/envelope workflows, and bearing-related reasoning.
6. multiagent: Complex, multi-stage workflows involving orchestration of multiple agents.

Asset Profile:
{asset_profile_json}

Allocation Strategy:
- Prioritize agents that have more "relevant_tools" or richer "failure_modes" / "iot_sensors" / "vibration_sensors" entries in the profile.
- If the asset mentions complex standards (ISO 14224, etc.), lean towards analytical categories (tsfm, fmsr, vibration where appropriate).
- It is perfectly acceptable for a category to have 0 scenarios if the Asset Profile doesn't warrant it.
- Cap "multiagent" at a maximum of 75% of the total budget (e.g., max 37 if total is 50).
- The sum of all allocations MUST exactly equal {total_scenarios}.

Output Format:
CRITICAL: Return ONLY a raw JSON object. Do NOT include markdown code blocks, Python code, or any conversational preamble.

{{
    "reasoning": "A brief explanation of why you chose this distribution based on the asset details.",
    "allocation": {{
        "iot": int,
        "fmsr": int,
        "tsfm": int,
        "wo": int,
        "vibration": int,
        "multiagent": int
    }}
}}
"""
