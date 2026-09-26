"""Stirrup as a Harbor agent.

Written against AssetOpsBench main (81265cb). Harbor's agent factory imports
any ``module.path:ClassName`` passed to ``--agent`` directly, bypassing its
built-in name enum, so this class needs no upstream registration and lives in
the AssetOpsBench repo:

    PYTHONPATH=agent harbor run -p datasets/assetopsbench-open \
      --agent assetops_harbor.stirrup:StirrupAgent \
      --model watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
      --ak code_enabled=false \
      --n-concurrent 16

The agent process runs INSIDE the task container, which is what keeps this
phase small: the six MCP servers stay stdio children of Stirrup exactly as
``mcphub.DEFAULT_SERVERS`` launches them today (``uv run <name>-mcp-server``),
and they inherit this trial's COUCHDB_URL through mcphub's
``{**os.environ, **(env or {})}`` merge. Nothing about the transport changes.

Three facts about main shape this file:

* ``stirrup-agent`` takes the question as a REQUIRED POSITIONAL argument
  (``_cli_common.add_common_args``). There is no stdin path, so the question
  is uploaded to a file and passed as ``"$(cat ...)"``, which survives
  multi-line prose without the argv quoting hazards of inlining it.
* There is no ``--topology`` flag. The arms main actually exposes are
  ``--code-enabled`` / ``--no-code``, ``--code-backend``, ``--max-turns``,
  ``--temperature`` and ``--reasoning-effort``.
* ``--code-backend`` defaults to ``docker``, which spawns a sibling container
  from ``STIRRUP_CODE_IMAGE``. There is no Docker daemon inside a Harbor task
  container, so that default would fail every run. ``local`` is the right
  backend here: the Harbor container is already a per-trial sandbox, so the
  isolation the docker backend buys on a laptop is redundant.
"""

from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

AOB_HOME = "/opt/aob"
REMOTE_QUESTION_PATH = "/tmp/aob_instruction.txt"

# Mirrors PROXY_ROUTERS in src/llm/routers.py, which is the source of truth.
# Copied rather than imported because `llm/__init__` pulls in the LiteLLM and
# OpenAI backends, and this module runs host-side inside Harbor. The test suite
# asserts the two stay in step.
ROUTER_CREDENTIALS: dict[str, tuple[str, str]] = {
    "litellm_proxy/": ("LITELLM_BASE_URL", "LITELLM_API_KEY"),
    "tokenrouter/": ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"),
}

# Forwarded from the Harbor process into the agent container when present.
# Harbor scopes them to the agent phase, so the verifier and build steps never
# see them. `--ae KEY=VALUE` still takes precedence over the host environment.
CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "TOKENROUTER_BASE_URL",
    "TOKENROUTER_API_KEY",
    "WATSONX_APIKEY",
    "WATSONX_URL",
    "WATSONX_PROJECT_ID",
    "WATSONX_DEPLOYMENT_SPACE_ID",
    "WATSONX_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_REGION_NAME",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GEMINI_API_KEY",
)


class StirrupAgent(BaseInstalledAgent):
    """Runs the AssetOpsBench Stirrup CLI inside the task environment."""

    @staticmethod
    def name() -> str:
        return "stirrup"

    def __init__(
        self,
        *args: Any,
        code_enabled: bool = True,
        code_backend: str = "local",
        allow_docker_backend: bool = False,
        max_turns: int = 30,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        if code_backend not in {"local", "docker"}:
            raise ValueError("code_backend must be 'local' or 'docker'")
        if code_backend == "docker" and not allow_docker_backend:
            raise ValueError(
                "code_backend='docker' spawns a sibling container from "
                "STIRRUP_CODE_IMAGE, and a Harbor task container has no Docker "
                "daemon. Use code_backend='local' (the Harbor container is "
                "already a per-trial sandbox), or mount a Docker socket into "
                "the task's compose file and pass allow_docker_backend=true."
            )

        # Harbor records every agent kwarg in the trial's config.json, so the
        # arm is visible in the result rather than inferred from a directory
        # name. These are the arms main exposes; there is no topology flag yet.
        self.code_enabled = _as_bool(code_enabled)
        self.code_backend = code_backend
        self.max_turns = int(max_turns)
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        super().__init__(*args, **kwargs)
        self._require_router_credentials()

    def _require_router_credentials(self) -> None:
        """Fail before Harbor builds anything if the router creds are missing.

        llm.routers.resolve_router_creds raises inside the container otherwise,
        which costs an image build and a container per trial to learn that a
        variable is unset.
        """
        for prefix, (base_env, key_env) in ROUTER_CREDENTIALS.items():
            if not (self.model_name or "").startswith(prefix):
                continue
            missing = [name for name in (base_env, key_env) if not self._get_env(name)]
            if missing:
                raise ValueError(
                    f"{' and '.join(missing)} must be set for the {prefix!r} model "
                    f"prefix. Export them, or pass them per run with "
                    f"--ae {missing[0]}=... ."
                )

    def _credential_env(self) -> dict[str, str]:
        """Credentials to forward into the agent container.

        _get_env reads resolved env vars, then --ae overrides, then the Harbor
        process environment, so an exported shell variable reaches the agent
        without being named on the command line.
        """
        found = {}
        for name in CREDENTIAL_ENV_VARS:
            value = self._get_env(name)
            if value:
                found[name] = value
        return found

    async def install(self, environment: BaseEnvironment) -> None:
        """No-op: the repo and its uv environment are baked into the task image."""

    def get_version_command(self) -> str | None:
        """Record the AssetOpsBench commit as the agent version.

        Harbor writes this into result.json, so every trial carries the exact
        repo state it ran against. Detection is best-effort in Harbor, so a
        task image built without git history simply reports no version.
        """
        return f"git -C {AOB_HOME} rev-parse --short HEAD"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        run_id = self.run_id

        # The question is a required positional on main and scenario questions
        # are multi-line prose, so stage it as a file and let bash read it back
        # inside double quotes. upload_file rather than the /logs bind mount,
        # because not every Harbor provider mounts logs the same way.
        staged = self.logs_dir / "instruction.txt"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(instruction, encoding="utf-8")
        await environment.upload_file(staged, REMOTE_QUESTION_PATH)

        flags: list[str] = [
            "--model-id",
            shlex.quote(self.model_name or ""),
            "--run-id",
            shlex.quote(run_id),
            "--code-backend",
            self.code_backend,
            "--max-turns",
            str(self.max_turns),
        ]
        flags.append("--code-enabled" if self.code_enabled else "--no-code")
        if self.temperature is not None:
            flags += ["--temperature", str(self.temperature)]
        if self.reasoning_effort is not None:
            flags += ["--reasoning-effort", shlex.quote(self.reasoning_effort)]

        # AOB_SCENARIO_ID comes from the task's [environment].env, which Harbor
        # injects into the main service. It lands on the root OTEL span.
        command = (
            f"uv run stirrup-agent {' '.join(flags)} "
            f'--scenario-id "${{AOB_SCENARIO_ID:-}}" '
            f'"$(cat {REMOTE_QUESTION_PATH})" '
            f"2>&1 | tee {shlex.quote(f'/logs/agent/{run_id}.stdout.txt')}"
        )

        await self.exec_as_agent(
            environment, command=command, cwd=AOB_HOME, env=self._credential_env()
        )

    @property
    def run_id(self) -> str:
        """Harbor's trial name, so AGENT_TRAJECTORY_DIR files never collide.

        observability.persistence writes ``{run_id}.json`` into
        AGENT_TRAJECTORY_DIR, which the task config points at /logs/agent. The
        old ``{agent}_{scenario}`` convention would collide across arms sharing
        a log directory, so pass Harbor's own name through instead.
        """
        return self.logs_dir.parent.name or "stirrup-run"

    # ------------------------------------------------------------------ #
    # ATIF conversion
    # ------------------------------------------------------------------ #

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Write trajectory.json and fill the run's token counts.

        Harbor never calls convert_trajectory() on its own. Each agent invokes it
        from here and persists the result, exactly as harbor.agents.installed
        .claude_code and .codex do. Trial._sync_agent_output then reads
        logs_dir/trajectory.json back to populate model usage, so without this
        hook there is no trajectory file and no token accounting.
        """
        try:
            trajectory = self.convert_trajectory(self.logs_dir)
        except Exception as exc:  # noqa: BLE001 - never fail a scored run over telemetry
            self.logger.debug("Failed to convert the Stirrup trajectory: %s", exc)
            return
        if trajectory is None:
            self.logger.debug(
                "No AssetOpsBench record found in %s; "
                "check that AGENT_TRAJECTORY_DIR pointed at it",
                self.logs_dir,
            )
            return

        path = self.logs_dir / "trajectory.json"
        try:
            path.write_text(
                json.dumps(trajectory.to_json_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            self.logger.debug("Failed to write %s: %s", path, exc)

        metrics = trajectory.final_metrics
        if metrics is not None:
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.cost_usd = metrics.total_cost_usd

    def convert_trajectory(self, logs_dir: Path) -> Trajectory | None:
        """Map the AssetOpsBench persisted record onto Harbor's ATIF schema.

        This lives in the agent adapter, not in src/observability/, so the
        telemetry code stays untouched. Harbor reads token and cost totals back
        out of the ATIF trajectory to populate AgentContext, so a correct
        mapping also buys the run's token accounting for free.

        Handles both shapes persistence._serialize_trajectory emits: the SDK
        runners' Trajectory dataclass (a dict with "turns") and plan-execute's
        list[StepResult].
        """
        record = self._load_record(logs_dir)
        if record is None:
            return None

        raw = record.get("trajectory")
        if isinstance(raw, dict):
            steps, prompt_total, completion_total = self._steps_from_sdk(raw, record)
        elif isinstance(raw, list):
            steps, prompt_total, completion_total = self._steps_from_plan_execute(
                raw, record
            )
        else:
            steps, prompt_total, completion_total = ([], 0, 0)

        steps = [
            self._question_step(record),
            *steps,
            self._answer_step(record, len(steps) + 2),
        ]

        return Trajectory(
            session_id=record.get("run_id"),
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=record.get("model"),
                extra={
                    "runner": record.get("runner"),
                    "scenario_id": record.get("scenario_id"),
                    "code_enabled": self.code_enabled,
                    "code_backend": self.code_backend,
                },
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=prompt_total,
                total_completion_tokens=completion_total,
                total_steps=len(steps),
            ),
        )

    def _load_record(self, logs_dir: Path) -> dict | None:
        exact = logs_dir / f"{self.run_id}.json"
        candidates = (
            [exact]
            if exact.exists()
            else [
                path
                for path in sorted(logs_dir.glob("*.json"))
                if path.name != "trajectory.json"
            ]
        )
        for path in candidates:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        return None

    def _steps_from_sdk(self, raw: dict, record: dict) -> tuple[list[Step], int, int]:
        """agent.models.Trajectory: turns[] of TurnRecord.

        TurnRecord fields are index, text, tool_calls, input_tokens,
        output_tokens, duration_ms. ToolCall fields are name, input, id
        (default ""), output, duration_ms.
        """
        steps: list[Step] = []
        prompt_total = 0
        completion_total = 0
        started_at = raw.get("started_at")

        for offset, turn in enumerate(raw.get("turns") or []):
            step_id = offset + 2  # step 1 is the question
            prompt = int(turn.get("input_tokens") or 0)
            completion = int(turn.get("output_tokens") or 0)
            prompt_total += prompt
            completion_total += completion

            calls: list[ToolCall] = []
            results: list[ObservationResult] = []
            for n, call in enumerate(turn.get("tool_calls") or []):
                # ToolCall.id defaults to "", so synthesize a stable id when
                # the runner did not supply one. ATIF requires every
                # observation result to match a tool_call_id in the same step.
                call_id = call.get("id") or f"call_{step_id}_{n}"
                calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=call.get("name") or "unknown",
                        arguments=call.get("input") or {},
                    )
                )
                if call.get("output") is not None:
                    results.append(
                        ObservationResult(
                            source_call_id=call_id,
                            content=_as_text(call.get("output")),
                        )
                    )

            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=started_at or _now(),
                    source="agent",
                    message=turn.get("text") or "",
                    model_name=record.get("model"),
                    tool_calls=calls or None,
                    observation=Observation(results=results) if results else None,
                    metrics=Metrics(prompt_tokens=prompt, completion_tokens=completion),
                )
            )

        return steps, prompt_total, completion_total

    def _steps_from_plan_execute(
        self, raw: list, record: dict
    ) -> tuple[list[Step], int, int]:
        """plan_execute.models.StepResult: step_number, task, server, response,
        error, tool, tool_args, duration_ms. No token counts on this shape, so
        FinalMetrics stays zero and Harbor falls back to whatever the run
        reported elsewhere.
        """
        steps: list[Step] = []

        for offset, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            step_id = offset + 2
            call_id = f"call_{step_id}_0"
            tool_name = item.get("tool") or item.get("server") or "unknown"
            output = item.get("error") or item.get("response") or ""

            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=_now(),
                    source="agent",
                    message=item.get("task") or "",
                    model_name=record.get("model"),
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=tool_name,
                            arguments=item.get("tool_args") or {},
                        )
                    ],
                    observation=Observation(
                        results=[
                            ObservationResult(
                                source_call_id=call_id, content=_as_text(output)
                            )
                        ]
                    ),
                )
            )

        return steps, 0, 0

    @staticmethod
    def _question_step(record: dict) -> Step:
        return Step(
            step_id=1,
            timestamp=_now(),
            source="user",
            message=record.get("question") or "",
        )

    @staticmethod
    def _answer_step(record: dict, step_id: int) -> Step:
        return Step(
            step_id=step_id,
            timestamp=_now(),
            source="agent",
            message=record.get("answer") or "",
            model_name=record.get("model"),
            llm_call_count=0,
        )


def _as_bool(value: Any) -> bool:
    """Harbor passes --ak values through as strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", ""}


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()
