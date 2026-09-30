"""Codex subscription agent with benchmark MCP tools and standard trajectories."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from tempfile import TemporaryDirectory

from llm.codex import CodexBackend, _DISABLED_FEATURES
from observability import agent_run_span, persist_trajectory
from observability.benchmark_trace import emit, stream_process, tool_result_error
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner


class CodexAgentRunner(AgentRunner):
    def __init__(self, llm=None, server_paths=None, model="gpt-6-astra", *,
                 executable="codex", timeout_seconds=840):
        super().__init__(llm, server_paths)
        self.model = model
        self.executable = os.environ.get("BENCHMARK_CODEX_EXECUTABLE", executable)
        self.timeout_seconds = timeout_seconds
        self.project = Path(__file__).resolve().parents[3]

    def _execute(self, question):
        with TemporaryDirectory(prefix="assetops-codex-agent-") as directory:
            root = Path(directory)
            instructions = root / "instructions.txt"
            instructions.write_text(AGENT_SYSTEM_PROMPT)
            output = root / "answer.txt"
            command = [self.executable, "exec", "--json", "--ephemeral",
                       "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only",
                       "--color", "never", "--model", self.model,
                       "--output-last-message", str(output), "--enable", "skip_host_skill_discovery"]
            config = {"approval_policy": "never", "web_search": "disabled",
                      "tools.view_image": False, "agents.enabled": False,
                      "project_doc_max_bytes": 0, "model_instructions_file": str(instructions)}
            for name, spec in self._server_paths.items():
                prefix = f"mcp_servers.{name}"
                config.update({f"{prefix}.command": "uv",
                               f"{prefix}.args": ["run", "--project", str(self.project), str(spec)],
                               f"{prefix}.cwd": str(self.project),
                               f"{prefix}.env_vars": ["COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD"],
                               f"{prefix}.required": True,
                               f"{prefix}.default_tools_approval_mode": "approve",
                               f"{prefix}.startup_timeout_sec": 60})
            for key, value in config.items():
                command.extend(["-c", f"{key}={json.dumps(value)}"])
            for feature in _DISABLED_FEATURES:
                command.extend(["--disable", feature])
            command.append("-")
            tool_starts = {}
            def receive(line):
                event = json.loads(line)
                emit('codex_event', payload=event)
                item = event.get('item', {})
                identifier = item.get('id')
                if item.get('type') == 'mcp_tool_call':
                    data = {'id':identifier, 'name':item.get('tool'), 'server':item.get('server'),
                            'arguments':item.get('arguments')}
                    if event.get('type') == 'item.started':
                        tool_starts[identifier] = time.perf_counter()
                        emit('tool_start', **data)
                        emit('model_response_observed', payload=item)
                    elif event.get('type') == 'item.completed':
                        start = tool_starts.pop(identifier, None)
                        emit('tool_end', **data, output=item.get('result'), error=item.get('error') or tool_result_error(item.get('result')),
                             duration_ms=(time.perf_counter()-start)*1000 if start is not None else None)
                if event.get('type') == 'item.completed' and item.get('type') == 'agent_message':
                    emit('message', payload=item)
                if event.get('type') == 'turn.completed':
                    emit('provider_summary', payload={'usage':event.get('usage')})
            process = stream_process(command, prompt=question, cwd=directory,
                                     timeout=self.timeout_seconds, on_line=receive)
            result = CodexBackend._read_result(process, output)
            trajectory = self._trajectory(process.stdout)
            # CLI usage is cumulative over the completed turn, not per item.
            if not trajectory.turns:
                trajectory.turns.append(TurnRecord(index=0, text=result.text))
            trajectory.turns[-1].input_tokens = result.input_tokens
            trajectory.turns[-1].output_tokens = result.output_tokens
            return result.text, trajectory

    @staticmethod
    def _trajectory(events):
        trajectory = Trajectory()
        for line in events.splitlines():
            event = json.loads(line)
            if event.get("type") != "item.completed":
                continue
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                trajectory.turns.append(TurnRecord(index=len(trajectory.turns), text=item.get("text", "")))
            elif item.get("type") == "mcp_tool_call":
                result = item.get("result")
                if item.get("error") is not None:
                    result = {"result": result, "error": item["error"]}
                call = ToolCall(name=f"mcp__{item['server']}__{item['tool']}",
                                input=item.get("arguments", {}), id=item.get("id", ""), output=result)
                trajectory.turns.append(TurnRecord(index=len(trajectory.turns), text="", tool_calls=[call]))
        return trajectory

    async def run(self, question):
        emit('capabilities', tool_events=True, request_timings=False, compaction_events=False)
        started = dt.datetime.now(dt.UTC).isoformat()
        with agent_run_span("codex-agent", model=self.model, question=question):
            answer, trajectory = await asyncio.to_thread(self._execute, question)
            emit("final_answer", answer=answer)
            trajectory.started_at = started
            persist_trajectory(runner_name="codex-agent", model=self.model, question=question,
                               answer=answer, trajectory=trajectory)
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
