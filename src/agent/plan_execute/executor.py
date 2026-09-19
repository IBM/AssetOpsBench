"""MCP-based step executor for the plan-execute orchestrator.

The planner produces steps with no pre-filled arguments. For every step that
calls a tool the executor makes one LLM call to generate the concrete argument
dict from the task description, original question, and prior step results.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from llm import LLMBackend
from ..runner import DEFAULT_SERVER_PATHS
from .models import Plan, PlanStep, RetrySafety, StepFailureKind, StepResult

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SOURCE_ROOT = _REPO_ROOT / "src"

_PLACEHOLDER_RE = re.compile(r"\{step_(\d+)\}")

# Matches a ```-fenced block whether the language tag and content share a
# line with the fence markers or sit on lines of their own.
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\r?\n?(.*?)\r?\n?```$", re.DOTALL)

_ARG_RESOLUTION_PROMPT = """\
Generate the JSON arguments for the tool call below.

Question: {question}
Tool: {tool}
Tool description: {tool_description}
Tool parameters: {tool_schema}
Task: {task}

Prior step results:
{context}

{repair_feedback}

YOUR RESPONSE MUST BE A SINGLE RAW JSON OBJECT AND NOTHING ELSE.
Do not write any explanation, reasoning, or prose — output only the JSON object.
Use EXACTLY the parameter names listed in "Tool parameters" above.
Use the task description and prior step results to determine the correct argument values.
If a value comes from a list, use the first relevant element.
Treat optional parameters as filters: omit them when the question requests all values.
Never invent a placeholder value for an optional filter. Include a filter only when
its exact value is supported by the question, tool description, or prior evidence.
For identifiers, use the exact canonical value from prior results; do not paraphrase it.

JSON:"""

_SIMPLE_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ArgumentResolutionError(ValueError):
    """The argument model did not return a usable JSON object."""


class ToolArgumentValidationError(ValueError):
    """Generated arguments do not satisfy the advertised input schema."""


class Executor:
    """Executes plan steps by routing tool calls to MCP servers."""

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = (
            DEFAULT_SERVER_PATHS if server_paths is None else server_paths
        )

    async def get_server_descriptions(self) -> dict[str, str]:
        """Query each registered MCP server and return formatted tool signatures."""
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                lines = []
                for t in tools:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    lines.append(f"  - {t['name']}({params}): {t['description']}")
                descriptions[name] = "\n".join(lines)
            except Exception as exc:  # noqa: BLE001
                descriptions[name] = f"  (unavailable: {exc})"
        return descriptions

    async def execute_plan(
        self,
        plan: Plan,
        question: str,
        *,
        adaptive_recovery: bool = False,
        max_recovery_attempts: int = 2,
    ) -> list[StepResult]:
        """Execute all plan steps in dependency order.

        Adaptive recovery is opt-in. It permits at most one retry per failed
        step and also enforces the run-wide ``max_recovery_attempts`` budget.
        """
        ordered = plan.resolved_order()
        total = len(ordered)

        # Pre-fetch tool schemas for all servers referenced in the plan so that
        # _resolve_args_with_llm can include exact parameter names in its prompt.
        server_names = {step.server for step in ordered}
        tool_specs: dict[str, dict[str, dict[str, Any]]] = {}
        for name in server_names:
            path = self._server_paths.get(name)
            if path is None:
                continue
            try:
                tools = await _list_tools(path)
                tool_specs[name] = {t["name"]: t for t in tools}
            except Exception:  # noqa: BLE001
                tool_specs[name] = {}

        context: dict[int, StepResult] = {}
        results: list[StepResult] = []
        recovery_attempts = 0
        for step in ordered:
            _log.info(
                "Step %d/%d [%s]: %s",
                step.step_number,
                total,
                step.server,
                step.task,
            )
            step_started = time.perf_counter()
            unavailable_dependencies = [
                dependency
                for dependency in step.dependencies
                if dependency not in context or not context[dependency].success
            ]
            tool_spec = tool_specs.get(step.server, {}).get(step.tool)
            schema = _format_tool_schema(tool_spec)
            description = str((tool_spec or {}).get("description") or "").strip()

            if adaptive_recovery and unavailable_dependencies:
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=(
                        "Blocked by unavailable dependencies: "
                        + ", ".join(str(n) for n in unavailable_dependencies)
                    ),
                    tool=step.tool,
                    tool_args=step.tool_args,
                    failure_kind=StepFailureKind.FAILED_DEPENDENCY,
                    attempt_count=0,
                )
            elif (
                adaptive_recovery
                and step.tool
                and step.tool.lower() not in {"none", "null"}
                and step.server in self._server_paths
                and tool_spec is None
            ):
                result = StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    server=step.server,
                    response="",
                    error=f"Tool '{step.tool}' is not advertised by server '{step.server}'",
                    tool=step.tool,
                    tool_args=step.tool_args,
                    failure_kind=StepFailureKind.UNSUPPORTED_CAPABILITY,
                )
            else:
                result = await self.execute_step(
                    step,
                    context,
                    question,
                    tool_schema=schema,
                    tool_description=description,
                    tool_spec=tool_spec,
                    detect_failures=adaptive_recovery,
                )

            if adaptive_recovery and not result.success:
                retry_safety = _retry_safety(result, tool_spec)
                result.retry_safety = retry_safety
                may_retry = retry_safety in {
                    RetrySafety.SAFE_PRE_CALL,
                    RetrySafety.READ_ONLY,
                }
                if may_retry and recovery_attempts < max_recovery_attempts:
                    recovery_attempts += 1
                    retry = await self.execute_step(
                        step,
                        context,
                        question,
                        tool_schema=schema,
                        tool_description=description,
                        tool_spec=tool_spec,
                        detect_failures=True,
                        repair_feedback=_repair_feedback(result),
                    )
                    retry.attempt_count = result.attempt_count + 1
                    retry.recovery_attempted = True
                    retry.initial_error = result.error
                    retry.recovery_succeeded = retry.success
                    retry.retry_safety = (
                        retry_safety
                        if retry.success
                        else _retry_safety(retry, tool_spec)
                    )
                    retry.retry_exhausted = not retry.success
                    result = retry
                elif may_retry:
                    result.retry_blocked = True
                    result.retry_exhausted = True
                elif retry_safety in {RetrySafety.MUTATING, RetrySafety.UNKNOWN}:
                    result.retry_blocked = True

            result.duration_ms = (time.perf_counter() - step_started) * 1000
            if result.success:
                _log.info("Step %d OK.", step.step_number)
            else:
                _log.warning("Step %d FAILED: %s", step.step_number, result.error)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, StepResult],
        question: str,
        tool_schema: str = "",
        tool_description: str = "",
        tool_spec: dict[str, Any] | None = None,
        detect_failures: bool = False,
        repair_feedback: str = "",
    ) -> StepResult:
        """Execute a single plan step.

        1. Resolve the MCP server assigned to this step.
        2. If no tool is specified, return expected_output directly.
        3. Call the LLM to generate tool arguments from the task and prior results.
        4. Call the tool and return its result.
        """
        if not step.tool or step.tool.lower() in ("none", "null"):
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=step.expected_output,
                tool=step.tool,
                tool_args=step.tool_args,
            )

        server_path = self._server_paths.get(step.server)
        if server_path is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=(
                    f"Unknown server '{step.server}'. "
                    f"Registered servers: {list(self._server_paths)}"
                ),
                tool=step.tool,
                tool_args=step.tool_args,
                failure_kind=StepFailureKind.UNSUPPORTED_CAPABILITY,
            )

        try:
            _log.info("Step %d: calling LLM to resolve args.", step.step_number)
            resolved_args = await _resolve_args_with_llm(
                question,
                step.task,
                step.tool,
                tool_schema,
                context,
                self._llm,
                tool_description=tool_description,
                repair_feedback=repair_feedback,
                require_valid_json=detect_failures,
            )
            # Optional MCP arguments should be omitted rather than sent as
            # JSON null, which overrides server defaults and can fail schema
            # validation for typed parameters.
            resolved_args = {
                key: value for key, value in resolved_args.items() if value is not None
            }
            if detect_failures:
                _validate_tool_args(resolved_args, tool_spec)
        except Exception as exc:  # noqa: BLE001
            failure_kind = (
                StepFailureKind.ARGUMENT_VALIDATION
                if isinstance(exc, ToolArgumentValidationError)
                else StepFailureKind.ARGUMENT_RESOLUTION
            )
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=locals().get("resolved_args", step.tool_args),
                failure_kind=failure_kind,
                retry_safety=RetrySafety.SAFE_PRE_CALL,
            )

        try:
            response = await _call_tool(server_path, step.tool, resolved_args)
            if detect_failures:
                response_error = _structured_tool_error(response)
                if response_error:
                    raise RuntimeError(response_error)
                if not response.strip():
                    return StepResult(
                        step_number=step.step_number,
                        task=step.task,
                        server=step.server,
                        response="",
                        error="Tool returned empty output",
                        tool=step.tool,
                        tool_args=resolved_args,
                        failure_kind=StepFailureKind.EMPTY_OUTPUT,
                    )
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response=response,
                tool=step.tool,
                tool_args=resolved_args,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                server=step.server,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=resolved_args,
                failure_kind=StepFailureKind.TOOL_ERROR,
            )


# ── arg resolution ────────────────────────────────────────────────────────────


async def _resolve_args_with_llm(
    question: str,
    task: str,
    tool: str,
    tool_schema: str,
    context: dict[int, StepResult],
    llm: LLMBackend,
    *,
    tool_description: str = "",
    repair_feedback: str = "",
    require_valid_json: bool = False,
) -> dict:
    """Generate tool arguments from the task description and prior step results."""
    context_text = "\n".join(
        f"Step {n}: {r.response}" for n, r in sorted(context.items())
    )
    prompt = (
        _ARG_RESOLUTION_PROMPT.replace("{question}", question)
        .replace("{task}", task)
        .replace("{tool}", tool)
        .replace("{tool_description}", tool_description or "(none)")
        .replace("{tool_schema}", tool_schema or "(unknown)")
        .replace("{context}", context_text or "(none)")
        .replace(
            "{repair_feedback}",
            repair_feedback or "No previous failed attempt is being repaired.",
        )
    )
    raw = llm.generate(prompt)
    resolved = _parse_json(raw)
    if resolved is None:
        _log.warning(
            "Tool '%s': arg resolution returned no parseable JSON (response: %r…)",
            tool,
            raw[:120],
        )
        if require_valid_json:
            raise ArgumentResolutionError(
                "Argument generation returned no parseable JSON object"
            )
        return {}
    return resolved


def _parse_json(raw: str) -> dict | None:
    """Extract a JSON object from an LLM response, with markdown fence handling.

    Returns the parsed dict, or None if no JSON object could be extracted.
    An empty dict ``{}`` is a valid successful parse (e.g. for no-arg tools).
    """
    text = raw.strip()
    if text.startswith("```"):
        match = _FENCE_RE.match(text)
        if match:
            text = match.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    _log.debug("_parse_json: could not extract a JSON object from: %r…", raw[:120])
    return None


def _format_tool_schema(tool_spec: dict[str, Any] | None) -> str:
    if not tool_spec:
        return ""
    return ", ".join(
        f"{parameter['name']}: {parameter['type']}"
        f"{'?' if not parameter['required'] else ''}"
        for parameter in tool_spec.get("parameters", [])
    )


def _validate_tool_args(
    args: dict[str, Any], tool_spec: dict[str, Any] | None
) -> None:
    """Validate required fields and simple JSON types before a tool is called."""
    if not tool_spec:
        return
    parameter_names = {
        parameter["name"] for parameter in tool_spec.get("parameters", [])
    }
    unexpected = sorted(set(args) - parameter_names)
    if unexpected:
        raise ToolArgumentValidationError(
            "Unexpected argument(s): " + ", ".join(unexpected)
        )
    for parameter in tool_spec.get("parameters", []):
        name = parameter["name"]
        if parameter.get("required") and name not in args:
            raise ToolArgumentValidationError(f"Missing required argument '{name}'")
        if name not in args:
            continue
        expected = _SIMPLE_JSON_TYPES.get(parameter.get("type", ""))
        value = args[name]
        wrong_type = expected is not None and not isinstance(value, expected)
        if parameter.get("type") in {"integer", "number"} and isinstance(value, bool):
            wrong_type = True
        if wrong_type:
            raise ToolArgumentValidationError(
                f"Argument '{name}' must be {parameter['type']}"
            )


def _structured_tool_error(response: str) -> str | None:
    """Return an application error carried by a transport-successful JSON result."""
    payload = _parse_json(response)
    if not payload:
        return None
    error = payload.get("error")
    if error in (None, "", False):
        return None
    if isinstance(error, str):
        return error
    return json.dumps(error, sort_keys=True, default=str)


def _retry_safety(
    result: StepResult, tool_spec: dict[str, Any] | None
) -> RetrySafety:
    """Classify one failed attempt conservatively for automatic replay."""
    if result.failure_kind in {
        StepFailureKind.ARGUMENT_RESOLUTION,
        StepFailureKind.ARGUMENT_VALIDATION,
    }:
        return RetrySafety.SAFE_PRE_CALL
    if result.failure_kind in {
        StepFailureKind.FAILED_DEPENDENCY,
        StepFailureKind.UNSUPPORTED_CAPABILITY,
    }:
        return RetrySafety.NOT_APPLICABLE

    annotations = (tool_spec or {}).get("annotations", {})
    if annotations.get("destructive") is True or annotations.get("read_only") is False:
        return RetrySafety.MUTATING
    if annotations.get("read_only") is True:
        return RetrySafety.READ_ONLY
    return RetrySafety.UNKNOWN


def _repair_feedback(result: StepResult) -> str:
    return (
        "The previous attempt failed. Generate corrected arguments; do not repeat "
        "the same mistake.\n"
        f"Previous arguments: {json.dumps(result.tool_args, sort_keys=True, default=str)}\n"
        f"Failure: {result.error or 'unknown'}"
    )


# ── MCP protocol helpers ──────────────────────────────────────────────────────


def _make_stdio_params(server: Path | str) -> "StdioServerParameters":
    """Build StdioServerParameters for a server spec.

    - str  → entry-point name; invoked as ``uv run <name>`` from the repo root.
    - Path → invoked as ``python -m module.path`` when under the repo root
             (supports relative imports), or directly otherwise.
    """
    from mcp import StdioServerParameters

    if isinstance(server, str):
        return StdioServerParameters(
            command="uv",
            args=["run", server],
            cwd=str(_REPO_ROOT),
            # MCP's stdio transport deliberately starts from a restricted
            # environment that omits PYTHONPATH. Keep checkout entry points
            # importable even when the editable-install .pth is ignored.
            env={"PYTHONPATH": str(_SOURCE_ROOT)},
        )
    try:
        rel = server.relative_to(_REPO_ROOT)
        module = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        return StdioServerParameters(
            command="python",
            args=["-m", module],
            cwd=str(_REPO_ROOT),
        )
    except ValueError:
        return StdioServerParameters(command="python", args=[str(server)])


async def _list_tools(server_path: Path | str) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools with parameter info."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": parameters,
                        "annotations": _tool_annotations(t),
                    }
                )
            return tools


def _tool_annotations(tool: Any) -> dict[str, bool | None]:
    annotations = getattr(tool, "annotations", None)
    return {
        "read_only": getattr(annotations, "readOnlyHint", None),
        "destructive": getattr(annotations, "destructiveHint", None),
        "idempotent": getattr(annotations, "idempotentHint", None),
    }


async def _call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
    """Connect to an MCP server via stdio and call a tool."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_tool_result(result)


def _extract_tool_result(result: Any) -> str:
    """Return tool text, raising when MCP marks the result as an error."""
    text = _extract_content(result.content)
    if getattr(result, "isError", False):
        raise RuntimeError(text or "MCP tool call failed")
    return text


def _extract_content(content: list[Any]) -> str:
    """Extract text from MCP tool call result content."""
    return "\n".join(getattr(item, "text", str(item)) for item in content)


def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Simple string substitution of {{step_N}} placeholders (kept for tests)."""
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str):

            def _sub(m: re.Match) -> str:
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)

            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved


def _parse_tool_call(raw: str) -> dict:
    """Parse LLM output into a {tool, args} dict (utility, not used in main path)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"tool": None, "answer": text}
