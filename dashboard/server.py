"""Local web/API server that exposes the repository's real AssetOpsBench modules."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
import logging
import math
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src"
DASHBOARD = Path(__file__).resolve().parent
SCENARIO_ROOTS = (SOURCE / "scenarios", SOURCE / "couchdb" / "scenarios_data")
DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"
MAX_REQUEST_BYTES = 2_000_000
sys.path.insert(0, str(SOURCE))

logging.basicConfig(level=os.environ.get("DASHBOARD_LOG_LEVEL", "INFO"))
LOG = logging.getLogger("assetopsbench.dashboard")

RUNS: list[dict[str, Any]] = []
RUNS_LOCK = threading.Lock()
TOOL_ALLOWLIST = {
    "iot": None,
    "tsfm": {
        "list_tasks", "profile_series", "characterize_series", "data_quality",
        "list_features", "list_models", "search_models", "find_models",
        "describe_candidates", "describe_models", "count_models", "list_domains",
        "get_model_lineage", "resolve_model", "hf_model_stats", "count_features",
        "describe_features", "extract_features", "select_features",
        "search_feature_catalog", "get_feature", "get_feature_lineage",
        "recipe_template", "run_recipe", "run_tabular_recipe", "evaluate",
        "get_result", "list_results", "get_run", "list_runs",
    },
    "fmsr": {"get_failure_modes", "generate_failure_modes"},
    "wo": {
        "list_workorders", "get_workorder", "get_workorder_tasks",
        "get_workorder_costs", "get_workorder_actuals_vs_planned",
        "get_workorder_kpis", "get_schedule_calendar",
        "get_my_assigned_workorders", "get_failure_codes",
    },
    "vibration": None,
}
MAX_TOOL_RESULT_CHARS = 1_000_000


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _scenario_from_record(record: dict[str, Any], source_path: Path) -> dict[str, Any] | None:
    text = record.get("text") or record.get("question")
    if not text or record.get("id") is None:
        return None
    raw_assets = record.get("assets", record.get("asset", []))
    if isinstance(raw_assets, str):
        raw_assets = [raw_assets]
    if not isinstance(raw_assets, list):
        raw_assets = []
    return {
        "id": str(record["id"]),
        "title": str(record.get("title") or text.splitlines()[0][:110]),
        "text": str(text),
        "category": str(record.get("category") or "Repository scenario"),
        "type": str(record.get("type") or ""),
        "agents": [str(agent) for agent in record.get("agents", [])],
        "assets": [str(asset) for asset in raw_assets if asset],
        "characteristic_form": record.get("characteristic_form") or record.get("expected_answer") or "",
        "source": source_path.relative_to(ROOT).as_posix(),
    }


def load_scenarios() -> list[dict[str, Any]]:
    """Read scenario definitions from their checked-in JSON and question files."""
    found: dict[str, dict[str, Any]] = {}
    for root in SCENARIO_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            try:
                content = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                LOG.warning("Skipping unreadable scenario candidate %s: %s", path, exc)
                continue
            records = content if isinstance(content, list) else [content]
            for record in records:
                if not isinstance(record, dict):
                    continue
                scenario = _scenario_from_record(record, path)
                if scenario:
                    found[scenario["id"]] = scenario

        for question_path in sorted(root.glob("scenario_*/question.txt")):
            folder = question_path.parent
            manifest_path = folder / "manifest.json"
            try:
                question = question_path.read_text(encoding="utf-8").strip()
                manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
                groundtruth_path = folder / "groundtruth.txt"
                groundtruth = groundtruth_path.read_text(encoding="utf-8").strip() if groundtruth_path.is_file() else ""
            except (OSError, json.JSONDecodeError) as exc:
                LOG.warning("Skipping incomplete scenario %s: %s", folder, exc)
                continue
            if not question:
                continue
            scenario_id = folder.name.removeprefix("scenario_")
            found.setdefault(scenario_id, {
                "id": scenario_id,
                "title": question.splitlines()[0][:110],
                "text": question,
                "category": "CouchDB scenario",
                "type": "Repository scenario",
                "agents": [],
                "assets": [],
                "characteristic_form": groundtruth,
                "manifest": manifest,
                "source": question_path.relative_to(ROOT).as_posix(),
            })
    return sorted(found.values(), key=lambda scenario: (scenario["category"].lower(), scenario["id"]))


def _manifest_files() -> list[tuple[Path, dict[str, Any]]]:
    output = []
    data_root = SOURCE / "couchdb" / "scenarios_data"
    for manifest_path in sorted(data_root.glob("*/manifest.json")):
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Skipping unreadable data manifest %s: %s", manifest_path, exc)
            continue
        output.append((manifest_path.parent, manifest))
    return output


def _data_path(base: Path, relative: str) -> Path | None:
    allowed_root = (SOURCE / "couchdb" / "scenarios_data").resolve()
    for root in (base, base.parent):
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(allowed_root) and candidate.is_file():
            return candidate
    return None


def _iter_manifest_paths(manifest: dict[str, Any], key: str) -> list[str]:
    value = manifest.get(key, [])
    return [value] if isinstance(value, str) else [item for item in value if isinstance(item, str)]


def _json_rows(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    records = value if isinstance(value, list) else [value]
    return [record for record in records if isinstance(record, dict)]


def load_data_overview() -> dict[str, Any]:
    """Read the repository's seeded data and derive truthful dashboard metrics."""
    manifests = _manifest_files()
    workorder_paths: set[Path] = set()
    iot_paths: set[Path] = set()
    failure_paths: set[Path] = set()
    missing_sources: list[str] = []
    for base, manifest in manifests:
        for field, paths in (("workorder", workorder_paths), ("iot", iot_paths), ("failure_mode", failure_paths)):
            for value in _iter_manifest_paths(manifest, field):
                path = _data_path(base, value)
                if path:
                    paths.add(path)
                else:
                    missing_sources.append(f"{base.relative_to(ROOT) / value}")
    if missing_sources:
        raise FileNotFoundError(f"Scenario manifests reference missing repository data: {', '.join(missing_sources)}")
    if not iot_paths or not workorder_paths or not failure_paths:
        missing = [
            name for name, files in (
                ("IoT telemetry", iot_paths),
                ("work-order", workorder_paths),
                ("FMEA", failure_paths),
            ) if not files
        ]
        raise FileNotFoundError(f"No repository sample source found for: {', '.join(missing)}")

    readings = 0
    chart_values: list[float] = []
    telemetry_records = 0
    for path in sorted(iot_paths):
        rows = _json_rows(path)
        telemetry_records += len(rows)
        for row in rows:
            measurements = [
                value for key, value in row.items()
                if key not in {"asset_id", "assetnum", "timestamp", "siteid"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ]
            readings += len(measurements)
            if not chart_values and measurements:
                chart_values.append(float(measurements[0]))
            elif len(chart_values) < 48 and measurements:
                chart_values.append(float(measurements[0]))

    failure_records = 0
    failure_modes = 0
    for path in sorted(failure_paths):
        for row in _json_rows(path):
            failure_records += 1
            modes = row.get("failure_modes")
            failure_modes += len(modes) if isinstance(modes, list) else 1

    workorders: list[dict[str, str]] = []
    for path in sorted(workorder_paths):
        with path.open(encoding="utf-8", newline="") as source:
            workorders.extend(dict(row) for row in csv.DictReader(source))
    statuses: dict[str, int] = {}
    for order in workorders:
        status = (order.get("status") or "Unknown").strip().upper()
        statuses[status] = statuses.get(status, 0) + 1
    completed = sum(count for status, count in statuses.items() if status in {"COMP", "CLOSE", "CLOSED"})
    completion_rate = round(100 * completed / len(workorders), 1) if workorders else None

    asset_paths = {
        path for base, manifest in manifests
        for value in _iter_manifest_paths(manifest, "asset")
        if (path := _data_path(base, value))
    }
    assets = sum(len(_json_rows(path)) for path in asset_paths)
    return {
        "telemetry": {
            "records": telemetry_records,
            "observations": readings,
            "series": chart_values,
            "sources": [path.relative_to(ROOT).as_posix() for path in sorted(iot_paths)],
        },
        "fmea": {
            "records": failure_records,
            "failure_modes": failure_modes,
            "names": [
                name
                for path in sorted(failure_paths)
                for row in _json_rows(path)
                for name in row.get("failure_modes", [])
                if isinstance(name, str)
            ],
            "sources": [path.relative_to(ROOT).as_posix() for path in sorted(failure_paths)],
        },
        "workorders": {
            "records": len(workorders),
            "statuses": statuses,
            "sources": [path.relative_to(ROOT).as_posix() for path in sorted(workorder_paths)],
        },
        "kpis": {
            "completed": completed,
            "total": len(workorders),
            "completion_rate": completion_rate,
            "open": sum(count for status, count in statuses.items() if status in {"WAPPR", "APPR"}),
            "sources": [path.relative_to(ROOT).as_posix() for path in sorted(workorder_paths)],
        },
        "assets": {"records": assets},
        "alerts": {"available": False, "message": "No separate alert collection is included in the checked-in sample manifests."},
        "source": "Checked-in scenario manifests and sample data files",
    }


def _required_environment(model: str) -> list[str]:
    if model.startswith("watsonx/"):
        names = ("WATSONX_APIKEY", "WATSONX_PROJECT_ID")
    elif model.startswith("tokenrouter/"):
        names = ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY")
    else:
        names = ("LITELLM_BASE_URL", "LITELLM_API_KEY")
    return [name for name in names if not os.environ.get(name)]


def get_status() -> dict[str, Any]:
    model = os.environ.get("ASSETOPSBENCH_DASHBOARD_MODEL", DEFAULT_MODEL)
    judge_model = os.environ.get("ASSETOPSBENCH_DASHBOARD_JUDGE_MODEL", model)
    missing = sorted(set(_required_environment(model) + _required_environment(judge_model)))
    try:
        from agent.runner import DEFAULT_SERVER_PATHS
        servers = sorted(DEFAULT_SERVER_PATHS)
    except ImportError as exc:
        servers = []
        import_error = str(exc)
    else:
        import_error = None
    return {
        "repo_root": str(ROOT),
        "model": model,
        "judge_model": judge_model,
        "missing_environment": missing,
        "runners": [
            {"id": "plan-execute", "label": "Plan-Execute"},
            {"id": "claude-agent", "label": "Claude Agent"},
            {"id": "openai-agent", "label": "OpenAI Agent"},
            {"id": "deep-agent", "label": "Deep Agent"},
        ],
        "servers": servers,
        "agent_modules_available": import_error is None,
        "agent_import_error": import_error,
        "scenarios": len(load_scenarios()),
        "ready": not missing and import_error is None,
    }


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _trajectory_as_text(value: Any) -> str:
    return json.dumps(value, default=_plain, ensure_ascii=False, indent=2)


def _run_real_agent(question: str, runner_name: str, model: str) -> tuple[str, Any, str]:
    from llm import make_backend

    os.environ["AOB_READONLY"] = "1"
    os.environ.setdefault("FMSR_MODEL_ID", model)
    if runner_name == "plan-execute":
        from agent.plan_execute.runner import PlanExecuteRunner

        runner = PlanExecuteRunner(llm=make_backend(model))
        result = asyncio.run(runner.run(question))
        return result.answer, result, model
    if runner_name == "claude-agent":
        from agent.claude_agent.runner import ClaudeAgentRunner

        runner = ClaudeAgentRunner(model=model)
        result = asyncio.run(runner.run(question))
        return result.answer, result, model
    if runner_name == "openai-agent":
        from agent.openai_agent.runner import OpenAIAgentRunner

        runner = OpenAIAgentRunner(model=model)
        result = asyncio.run(runner.run(question))
        return result.answer, result, model
    if runner_name == "deep-agent":
        from agent.deep_agent.runner import DeepAgentRunner

        runner = DeepAgentRunner(model=model)
        result = asyncio.run(runner.run(question))
        return result.answer, result, model
    raise ValueError("runner must be one of: plan-execute, claude-agent, openai-agent, deep-agent.")


def evaluate_result(scenario: dict[str, Any], answer: str, trajectory: Any, model: str) -> dict[str, Any]:
    from evaluation.models import Scenario
    from evaluation.scorers.llm_judge import LLMJudgeScorer
    from llm import make_backend

    expected = scenario.get("characteristic_form", "")
    evaluation_scenario = Scenario.from_raw({
        "id": scenario["id"],
        "text": scenario["text"],
        "category": scenario.get("category", ""),
        "type": scenario.get("type", ""),
        "characteristic_form": expected,
    })
    if not expected:
        return {
            "scorer": "llm_judge",
            "passed": False,
            "error": "This repository scenario has no characteristic_form or expected_answer for LLM-As-Judge evaluation.",
        }
    judge_model = os.environ.get("ASSETOPSBENCH_DASHBOARD_JUDGE_MODEL", model)
    missing = _required_environment(judge_model)
    if missing:
        return {"scorer": "llm_judge", "passed": False, "error": f"Missing judge-model environment: {', '.join(missing)}"}
    score = LLMJudgeScorer(make_backend(judge_model))(evaluation_scenario, answer, _trajectory_as_text(trajectory))
    result = score.model_dump(mode="json")
    result["model"] = judge_model
    if score.rationale.startswith(("judge backend error:", "judge returned unparseable JSON")):
        result.pop("score", None)
        result["error"] = score.rationale
    return result


def run_scenario(scenario_id: str, runner_name: str, model: str | None = None) -> dict[str, Any]:
    scenario = next((item for item in load_scenarios() if item["id"] == scenario_id), None)
    if scenario is None:
        raise LookupError(f"Scenario '{scenario_id}' is not present in the repository scenario files.")
    selected_model = (model or os.environ.get("ASSETOPSBENCH_DASHBOARD_MODEL", DEFAULT_MODEL)).strip()
    if not selected_model or len(selected_model) > 200:
        raise ValueError("A valid model ID is required.")
    missing = _required_environment(selected_model)
    if missing:
        raise RuntimeError(f"Configure the model credentials in the server environment: {', '.join(missing)}")

    run_id = uuid.uuid4().hex
    from observability.runspan import set_run_context

    set_run_context(run_id=run_id, scenario_id=scenario_id)
    answer, result, used_model = _run_real_agent(scenario["text"], runner_name, selected_model)
    trajectory = result.trajectory if hasattr(result, "trajectory") else None
    serialized = _plain(trajectory) if dataclasses.is_dataclass(trajectory) else (
        [_plain(step) for step in trajectory] if isinstance(trajectory, list) else trajectory
    )
    try:
        score = evaluate_result(scenario, answer, serialized, used_model)
    except Exception as exc:
        LOG.exception("Evaluation failed for dashboard run, scenario=%s", scenario_id)
        score = {"scorer": "llm_judge", "passed": False, "error": str(exc)}
    run = {
        "run_id": run_id,
        "scenario_id": scenario["id"],
        "scenario_title": scenario["title"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": runner_name,
        "model": used_model,
        "question": scenario["text"],
        "answer": answer,
        "trajectory": serialized,
        "plan": _plain(result.plan) if hasattr(result, "plan") else None,
        "evaluation": score,
        "is_real_run": True,
    }
    with RUNS_LOCK:
        RUNS.insert(0, run)
        del RUNS[20:]
    return run


def evaluate_uploaded_log(scenario_id: str, record: dict[str, Any]) -> dict[str, Any]:
    scenario = next((item for item in load_scenarios() if item["id"] == scenario_id), None)
    if scenario is None:
        raise LookupError(f"Scenario '{scenario_id}' is not present in the repository scenario files.")
    model = str(record.get("model") or os.environ.get("ASSETOPSBENCH_DASHBOARD_MODEL", DEFAULT_MODEL))
    missing = _required_environment(model)
    if missing:
        raise RuntimeError(f"Configure the model credentials in the server environment: {', '.join(missing)}")
    answer = record.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The evaluation log must contain an 'answer' string.")
    trajectory = record.get("trajectory")
    if trajectory is None:
        raise ValueError("The evaluation log must contain a 'trajectory'.")
    return evaluate_result(scenario, answer, trajectory, model)


def discover_tools(server_name: str) -> list[dict[str, Any]]:
    from agent.runner import DEFAULT_SERVER_PATHS
    from agent.plan_execute.executor import _list_tools

    if server_name not in TOOL_ALLOWLIST or server_name not in DEFAULT_SERVER_PATHS:
        raise ValueError(f"Unknown or unsupported read-only MCP server: {server_name}")
    os.environ["AOB_READONLY"] = "1"
    tools = asyncio.run(_list_tools(DEFAULT_SERVER_PATHS[server_name]))
    allowed = TOOL_ALLOWLIST[server_name]
    return [
        tool for tool in tools
        if allowed is None or tool["name"] in allowed
    ]


def call_repository_tool(server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from agent.runner import DEFAULT_SERVER_PATHS
    from agent.plan_execute.executor import _call_tool

    if server_name not in TOOL_ALLOWLIST or server_name not in DEFAULT_SERVER_PATHS:
        raise ValueError(f"Unknown or unsupported read-only MCP server: {server_name}")
    allowed = TOOL_ALLOWLIST[server_name]
    if allowed is not None and tool_name not in allowed:
        raise ValueError(f"Tool '{tool_name}' is not enabled in the dashboard's read-only tool policy.")
    available = {tool["name"] for tool in discover_tools(server_name)}
    if tool_name not in available:
        raise LookupError(f"Tool '{tool_name}' is not registered by the repository's {server_name} MCP server.")
    os.environ["AOB_READONLY"] = "1"
    output = asyncio.run(_call_tool(DEFAULT_SERVER_PATHS[server_name], tool_name, arguments))
    return {"server": server_name, "tool": tool_name, "arguments": arguments, "output": output[:MAX_TOOL_RESULT_CHARS]}


def benchmark_profiles() -> list[dict[str, Any]]:
    from benchmark.scenario_suite_runner import SCENARIO_PROFILE_PATHS, scenario_ids_for_selector

    available = {scenario["id"] for scenario in load_scenarios()}
    profiles = []
    for name, path in SCENARIO_PROFILE_PATHS.items():
        ids = scenario_ids_for_selector(name)
        profiles.append({
            "id": name,
            "source": path.relative_to(ROOT).as_posix(),
            "scenario_count": len(ids),
            "scenario_ids": ids,
            "locally_available": [scenario_id for scenario_id in ids if scenario_id in available],
            "requires_data_reset": True,
        })
    return profiles


def run_comparison(scenario_id: str, runners: list[str], model: str | None = None) -> dict[str, Any]:
    selected = list(dict.fromkeys(runners))
    if len(selected) < 2 or len(selected) > 4:
        raise ValueError("Select between two and four repository agent runners to compare.")
    completed = []
    failures = []
    for runner in selected:
        try:
            completed.append(run_scenario(scenario_id, runner, model))
        except Exception as exc:
            LOG.exception("Agent comparison run failed: runner=%s scenario=%s", runner, scenario_id)
            failures.append({"runner": runner, "error": str(exc)})
    if not completed and failures:
        raise RuntimeError("; ".join(f"{item['runner']}: {item['error']}" for item in failures))
    return {"runs": completed, "failures": failures}


def run_local_profile(profile_name: str, runner_name: str, model: str | None = None) -> dict[str, Any]:
    profiles = {profile["id"]: profile for profile in benchmark_profiles()}
    if profile_name not in profiles:
        raise ValueError("Choose a repository benchmark profile: mini, lite, open, or all.")
    profile = profiles[profile_name]
    available_ids = profile["locally_available"]
    if not available_ids:
        raise ValueError(
            f"The {profile_name} profile has no scenario definitions in the local repository checkout. "
            "The remaining profile scenarios require the benchmark's dataset and scenario loader."
        )
    if len(available_ids) > 20:
        raise ValueError(
            f"The {profile_name} profile matches {len(available_ids)} local scenarios. "
            "This dashboard limits one launch to 20 scenario runs; choose a smaller profile."
        )
    results = []
    failures = []
    for scenario_id in available_ids:
        try:
            results.append(run_scenario(scenario_id, runner_name, model))
        except Exception as exc:
            LOG.exception("Benchmark scenario failed: profile=%s scenario=%s", profile_name, scenario_id)
            failures.append({"scenario_id": scenario_id, "error": str(exc)})
    return {
        "profile": profile_name,
        "source": profile["source"],
        "runner": runner_name,
        "runs": results,
        "failures": failures,
        "skipped_scenario_ids": [item for item in profile["scenario_ids"] if item not in available_ids],
        "data_reset_performed": False,
    }


def load_run_history() -> list[dict[str, Any]]:
    history_dir = Path(os.environ.get("AGENT_TRAJECTORY_DIR", str(ROOT / "traces" / "dashboard"))).resolve()
    if not history_dir.is_relative_to(ROOT / "traces") or not history_dir.is_dir():
        return []
    persisted_runs = []
    for path in sorted(history_dir.glob("*.json"), key=lambda candidate: candidate.stat().st_mtime, reverse=True):
        try:
            record = _read_json(path)
            if isinstance(record, dict) and record.get("run_id"):
                persisted_runs.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Skipping unreadable saved run %s: %s", path, exc)
    with RUNS_LOCK:
        current_runs = [dict(record) for record in RUNS]
    known_ids = {record.get("run_id") for record in current_runs}
    runs = current_runs + [record for record in persisted_runs if record.get("run_id") not in known_ids]
    return runs[:100]


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/status":
                self._send_json(200, get_status())
            elif route == "/api/scenarios":
                self._send_json(200, {"scenarios": load_scenarios()})
            elif route == "/api/overview":
                self._send_json(200, load_data_overview())
            elif route == "/api/results":
                with RUNS_LOCK:
                    self._send_json(200, {"results": list(RUNS)})
            elif route == "/api/tools":
                server_name = parse_qs(parsed.query).get("server", [""])[0]
                self._send_json(200, {"server": server_name, "tools": discover_tools(server_name)})
            elif route == "/api/benchmark/profiles":
                self._send_json(200, {"profiles": benchmark_profiles()})
            elif route == "/api/history":
                self._send_json(200, {"runs": load_run_history()})
            else:
                super().do_GET()
        except Exception as exc:
            LOG.exception("GET %s failed", route)
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/run", "/api/evaluate-log", "/api/tool-call", "/api/compare", "/api/benchmark/run"}:
            self._send_json(404, {"error": "Unknown API endpoint."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes.")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("The request body must be a JSON object.")
            if route == "/api/run":
                scenario_id = str(payload.get("scenario_id") or "").strip()
                if not scenario_id or len(scenario_id) > 100:
                    raise ValueError("A valid repository scenario_id is required.")
                runner = str(payload.get("runner") or "plan-execute")
                model = payload.get("model")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string.")
                result = run_scenario(scenario_id, runner, model)
            elif route == "/api/evaluate-log":
                scenario_id = str(payload.get("scenario_id") or "").strip()
                if not scenario_id or len(scenario_id) > 100:
                    raise ValueError("A valid repository scenario_id is required.")
                record = payload.get("record")
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object.")
                result = evaluate_uploaded_log(scenario_id, record)
            elif route == "/api/tool-call":
                server_name = payload.get("server")
                tool_name = payload.get("tool")
                arguments = payload.get("arguments", {})
                if not isinstance(server_name, str) or not isinstance(tool_name, str):
                    raise ValueError("server and tool must be strings.")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object.")
                result = call_repository_tool(server_name, tool_name, arguments)
            elif route == "/api/compare":
                scenario_id = str(payload.get("scenario_id") or "").strip()
                runners = payload.get("runners")
                model = payload.get("model")
                if not scenario_id or len(scenario_id) > 100:
                    raise ValueError("A valid repository scenario_id is required.")
                if not isinstance(runners, list) or any(not isinstance(item, str) for item in runners):
                    raise ValueError("runners must be a list of agent runner IDs.")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string.")
                result = run_comparison(scenario_id, runners, model)
            else:
                profile_name = payload.get("profile")
                runner_name = payload.get("runner")
                model = payload.get("model")
                if not isinstance(profile_name, str) or not isinstance(runner_name, str):
                    raise ValueError("profile and runner must be strings.")
                if model is not None and not isinstance(model, str):
                    raise ValueError("model must be a string.")
                result = run_local_profile(profile_name, runner_name, model)
            self._send_json(200, result)
        except (json.JSONDecodeError, ValueError, LookupError, RuntimeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            LOG.exception("POST %s failed", route)
            self._send_json(500, {"error": f"AssetOpsBench execution failed: {exc}"})

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=_plain, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        LOG.warning("python-dotenv is unavailable; using the current process environment.")
    os.chdir(ROOT)
    os.environ.setdefault("AGENT_TRAJECTORY_DIR", str(ROOT / "traces" / "dashboard"))
    host = os.environ.get("ASSETOPSBENCH_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("ASSETOPSBENCH_DASHBOARD_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    LOG.info("AssetOpsBench dashboard: http://%s:%s", host, port)
    LOG.info("Repository root: %s", ROOT)
    LOG.info("Live agent calls require configured model credentials and available MCP data services.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Stopping dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
