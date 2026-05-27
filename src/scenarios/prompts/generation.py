"""Prompts for valid and negative scenario generation."""

SCENARIO_GENERATOR_PROMPT = """\
You are an advanced industrial agent Scenario Architect for AssetOps Bench.
We need you to generate {count} evaluation scenarios with primary focus '{subagent_name}' for the asset class: {asset_name}.

Generation Mode:
{generation_mode}

Asset Profile (full JSON):
{asset_profile_json}

Available Focus Tools:
{tool_definitions}

{few_shot_examples_section}

When generating scenarios, use the Generation Mode and Mode-specific grounding rules (not the few-shot rows) to decide whether each scenario must embed readings, rules, summaries, or dataset identifiers in the query text.

Suggested category values for this focus:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Hard-scenario contract for this batch:
- At least {hard_target_count} of the {count} scenarios should satisfy the hard rubric.
- Hard rubric: ask for at least two distinct outputs or actions, include at least one explicit constraint, and include an if/else, fallback, or missing-data instruction.
- Focus-specific hard patterns:
{hardness_guidance}

Avoid these bad patterns:
{forbidden_patterns}

Mode-specific grounding rules:
{mode_requirements}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Constraints:
1. Every scenario must read like a realistic direct request from an industrial operator or the operator's manager.
2. Prefer end-user-centric wording such as "Will my transformer's health be okay tomorrow?" over tool-centric or benchmark-centric wording such as "predict transformer health".
3. The scenario may involve supporting work from other agents, but the main burden should stay on the primary focus '{subagent_name}'.
4. Every scenario must be highly specific, having a clear 'text', a 'category', and a 'characteristic_form'.
5. The characteristic_form must explicitly mention the concrete MCP tool names needed to solve the task.
6. The 'text' field must stay natural operator language only: do not name MCP tools, API or function identifiers, or add parenthetical hints such as "e.g. get_failure_modes tool". Reserve every concrete tool name for 'characteristic_form' only.
7. Most scenarios should be multi-part and instruction-following rather than short one-liners.
8. Closed-form scenarios must embed explicit inline sensor readings in the query text: for each measurement, sensor name (or label), numeric value, and unit (e.g. ppm, %, Hz, mm/s). You may also embed rule text, summaries, or dataset identifiers when the task requires them.
9. Open-form scenarios must use only grounded identifiers present in the Asset Profile.
10. Do not output Unsupported.

Task: Generate a JSON array of {count} scenarios.
CRITICAL: Output ONLY the raw JSON array. Do NOT include markdown code blocks, Python code, or any conversational preamble.

Format exactly (raw JSON only):
[
    {{
        "text": "...",
        "category": "...",
        "characteristic_form": "..."
    }}
]
"""

VALIDATE_REPAIR_PROMPT = """\
You are the Validator and Repair agent for the AssetOps Bench scenario generator.

You are given a list of JSON scenarios and the corresponding Asset Profile.
Your job is to validate and repair them based on:
1. Do they fit exactly the JSON schema (text, category, characteristic_form)?
2. Does the characteristic_form name the concrete MCP tool names needed to solve the task?
3. Does the 'text' field contain only natural operator language, with no MCP tool names, function identifiers, or parenthetical hints like "e.g. get_failure_modes tool"? If any appear in 'text', remove or rewrite them and keep tool names in characteristic_form only.
4. Does each scenario clearly satisfy the prompt rules for primary focus '{subagent_name}' even if it uses supporting cross-agent steps?
5. In closed_form mode, does each scenario text include explicit inline sensor readings (name, value, unit) per Mode-specific grounding rules? Are there duplicates, near-duplicates, or (for open_form) ungrounded identifiers?
6. Are the scenarios phrased like direct end-user requests rather than internal benchmark instructions?
7. Does this batch satisfy the hard-scenario contract: at least {hard_target_count} of the {batch_size} scenarios should ask for at least two outputs or actions, include an explicit constraint, and include an if/else, fallback, or missing-data instruction?
8. Fix every deterministic validation failure listed below.

Target Focus:
{subagent_name}

Suggested Categories:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Hard-scenario contract:
- At least {hard_target_count} of the {batch_size} scenarios should satisfy the hard rubric.
- Hard rubric: ask for at least two distinct outputs or actions, include at least one explicit constraint, and include an if/else, fallback, or missing-data instruction.
- Focus-specific hard patterns:
{hardness_guidance}

Avoid these bad patterns:
{forbidden_patterns}

Mode-specific grounding rules:
{mode_requirements}

Asset Profile:
{asset_profile_json}

Input Scenarios:
{input_scenarios_json}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Deterministic validation failures that MUST be fixed:
{validation_failures_json}

Always review the full batch for end-user-centric phrasing and hardness, even when the deterministic failure list is empty.
Return ONLY the corrected and repaired JSON array. If a scenario is perfectly fine, keep it as is.
CRITICAL: Do not include any explanation, markdown formatting, or Python code. Output MUST be raw JSON text only.
"""

NEGATIVE_SCENARIO_GENERATOR_PROMPT = """\
You are an advanced industrial agent Scenario Architect for AssetOps Bench.
We need you to generate {count} intentionally unanswerable evaluation scenarios with primary focus '{subagent_name}' for the asset class: {asset_name}.

Generation Mode:
{generation_mode}

Asset Profile (full JSON):
{asset_profile_json}

Available Focus Tools:
{tool_definitions}

Suggested category values for this focus:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Mode-specific grounding rules:
{mode_requirements}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Task:
Generate realistic operator- or manager-facing requests that cannot be answered safely from the available grounded data and tools.

Negative-pattern guidance:
- Use grounded-adjacent realism: the question should sound like a plausible user request, not an adversarial puzzle.
- Make the scenario unanswerable because of one of these causes: missing or non-existent asset/site/sensor identifiers, time windows outside the available data range, unsupported external data joins, or contradictory instructions that prevent a safe grounded answer.
- It is acceptable for open-form negative scenarios to intentionally mention an ungrounded identifier or an out-of-range timestamp when that is exactly what makes the request unanswerable.
- The 'text' field must stay natural operator language only: do not name MCP tools, API or function identifiers.
- The characteristic_form must describe the expected insufficiency/refusal behavior: it should explicitly say that a correct answer explains why the request cannot be answered from the available data/tools and does not hallucinate missing information.
- Do not output a scenario that is secretly answerable from the Asset Profile and available tools.

Task: Generate a JSON array of {count} scenarios.
CRITICAL: Output ONLY the raw JSON array. Do NOT include markdown code blocks, Python code, or any conversational preamble.

Format exactly (raw JSON only):
[
    {{
        "text": "...",
        "category": "...",
        "characteristic_form": "..."
    }}
]
"""

NEGATIVE_VALIDATE_REPAIR_PROMPT = """\
You are the Validator and Repair agent for negative AssetOps Bench scenarios.

You are given a list of intentionally unanswerable JSON scenarios and the corresponding Asset Profile.
Your job is to validate and repair them based on:
1. Do they fit exactly the JSON schema (text, category, characteristic_form)?
2. Does the 'text' field stay in natural operator or manager language, with no MCP tool names or API identifiers?
3. Does each scenario remain realistically user-centric while still being unanswerable from the available data/tools?
4. Does each characteristic_form explicitly require an insufficiency/refusal-style answer that explains what is missing or unsupported?
5. Fix every deterministic validation failure listed below.

Target Focus:
{subagent_name}

Suggested Categories:
{category_options}

Primary-focus requirements:
{specialization_requirements}

Mode-specific grounding rules:
{mode_requirements}

Asset Profile:
{asset_profile_json}

Input Scenarios:
{input_scenarios_json}

Already accepted scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Deterministic validation failures that MUST be fixed:
{validation_failures_json}

Always preserve the fact that these scenarios are intentionally unanswerable.
Return ONLY the corrected and repaired JSON array. If a scenario is perfectly fine, keep it as is.
CRITICAL: Do not include any explanation, markdown formatting, or Python code. Output MUST be raw JSON text only.
"""

MULTIAGENT_COMBINER_PROMPT = """\
You are the Multiagent Scenario Combiner.
You generate complex multi-agent workflows that span across multiple focused capabilities (e.g. IoT + WO + TSFM + vibration).
We need {count} multi-agent scenarios for {asset_name}.

Generation Mode:
{generation_mode}

Asset Profile:
{asset_profile_json}

Available MCP Function Definitions (these are ALL the tools across all subagents, but you MUST prioritize the tools explicitly mentioned in the Asset Profile's "relevant_tools" section):
{mcp_function_definitions}

Here are some valid single-agent scenarios we generated earlier:
{single_agent_scenarios_json}

Follow Generation Mode: closed-form runs must inline explicit sensor readings (name, value, unit) in operator-facing text, plus any rule text, summaries, or dataset references the task needs.

Already accepted multiagent scenario texts. Avoid duplicates or near-duplicates of any of them:
{accepted_scenario_texts}

Hard-scenario contract for this batch:
- At least {hard_target_count} of the {count} scenarios should satisfy the hard rubric.
- Hard rubric: ask for at least two distinct outputs or actions, include at least one explicit constraint, and include an if/else, fallback, or missing-data instruction.
- Focus-specific hard patterns:
{hardness_guidance}

Mode-specific grounding rules:
{mode_requirements}

Avoid these bad patterns:
{forbidden_patterns}

Task: Use these single-agent pieces to construct cohesive multi-agent scenarios from the point of view of an operator or the operator's manager.
A multiagent scenario tests agent communication, workflow orchestration, and decision-making.
For instance, detecting an anomaly using IoT, confirming a failure signature with FMSR or vibration, projecting impact with TSFM, and planning action with WO.

Requirements:
- The 'text' field must stay natural operator language only: do not name MCP tools, API or function identifiers, or add parenthetical hints such as "e.g. get_failure_modes tool". Put concrete tool names only in characteristic_form.
- Phrase the text like a direct end-user request, not an internal workflow description.
- Most scenarios should be multi-part and instruction-following rather than short one-liners.
- Every multiagent characteristic_form must mention at least two distinct namespaces chosen from iot, fmsr, tsfm, wo, and vibration.
- Every multiagent characteristic_form must mention concrete MCP tool names from those namespaces.
- Open-form scenarios must only use grounded identifiers from the Asset Profile.
- Closed-form scenarios must contain explicit inline sensor readings (name, value, unit) in the text, and may include summaries, rule text, or dataset identifiers when needed.
- Do not output Unsupported.

Return ONLY a JSON array of length {count}.
CRITICAL: Do NOT include markdown code blocks, Python code, or any conversational preamble.

[
    {{
        "text": "Detect a temperature anomaly on the chiller, verify its past occurrences, and schedule a work order.",
        "category": "Knowledge Query",
        "characteristic_form": "The expected response should involve first utilizing the IoT tool to confirm the anomaly, then querying the WO tool to fetch history, before finalizing a work order creation."
    }}
]
"""
