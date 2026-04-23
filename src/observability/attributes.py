"""Canonical OpenTelemetry span attribute names used by agent runners.

Follows the OpenTelemetry GenAI semantic conventions where applicable
(https://opentelemetry.io/docs/specs/semconv/gen-ai/).  Runner-specific
attributes live under the ``agent.*`` namespace to avoid colliding with
future semconv additions.
"""

from __future__ import annotations

# ---- GenAI semantic conventions (v1.28) ------------------------------------

GEN_AI_SYSTEM = "gen_ai.system"
"""Provider family, e.g. ``"anthropic"``, ``"openai"``, ``"watsonx"``."""

GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
"""Requested model ID (post-``litellm_proxy/`` stripping)."""

GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
"""Total prompt/input tokens across the run."""

GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
"""Total completion/output tokens across the run."""


# ---- Agent-namespaced attributes -------------------------------------------

AGENT_RUNNER = "agent.runner"
"""Runner identifier: ``"plan-execute"``, ``"claude-agent"``, etc."""

AGENT_QUESTION_LENGTH = "agent.question.length"
"""Character length of the incoming question (avoids putting PII on spans)."""

AGENT_ANSWER_LENGTH = "agent.answer.length"
"""Character length of the final answer."""

AGENT_TURNS = "agent.turns"
"""Number of turns recorded in the trajectory."""

AGENT_TOOL_CALLS = "agent.tool_calls"
"""Total number of tool calls across the trajectory."""
