#!/usr/bin/env python
"""Run a controlled Plan-Execute adaptive-escalation experiment.

The four conditions share the same runner, model, tools, and verifier:

* ``baseline`` disables verification;
* ``original`` replays the PR #432 critique-only routing policy;
* ``redesigned`` uses direct execution evidence and bounded safe recovery;
* ``always`` enables recovery and forces the expensive verification behavior.

Example:

    PYTHONPATH=src .venv/bin/python benchmarks/adaptive_escalation_experiment.py \
      --scenario-root src/couchdb/scenarios_data \
      --scenario-ids 1,2,3 \
      --acknowledge-external-llm \
      --output-dir /tmp/adaptive-escalation

The acknowledgement is deliberately required because scenario questions, plans,
tool arguments, and tool responses are sent to the configured LLM provider.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv

import agent.plan_execute.runner as runner_module
from agent.plan_execute.escalation import (
    EscalationAction,
    EscalationDecision,
    extract_escalation_signals,
    should_escalate,
)
from agent.plan_execute.runner import PlanExecuteRunner
from evaluation.evaluator import Evaluator
from evaluation.metrics import _estimate_cost
from llm import LLMBackend, LLMResult, LiteLLMBackend
from observability import init_tracing, set_run_context

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
CONDITIONS = ("baseline", "original", "redesigned", "always")


@dataclass(frozen=True)
class CallMetric:
    kind: str
    input_tokens: int
    output_tokens: int
    duration_ms: float


class RecordingBackend(LLMBackend):
    """Collect usage and latency without changing model behavior."""

    def __init__(self, model_id: str) -> None:
        self._inner = LiteLLMBackend(model_id)
        self.calls: list[CallMetric] = []

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0
    ) -> LLMResult:
        started = time.perf_counter()
        result = self._inner.generate_with_usage(prompt, temperature)
        self.calls.append(
            CallMetric(
                kind=classify_prompt(prompt),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return result


def classify_prompt(prompt: str) -> str:
    if prompt.startswith("You are reviewing the evidence"):
        return "verification"
    if "You are summarizing" in prompt:
        return "summarization"
    if "Generate the JSON arguments" in prompt:
        return "argument_resolution"
    return "planning"


def parse_scenario_ids(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--scenario-ids must contain at least one id")
    if len(set(values)) != len(values):
        raise ValueError("--scenario-ids contains duplicates")
    return values


def parse_conditions(raw: str) -> list[str]:
    """Return a validated, ordered subset of the experimental conditions."""
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--conditions must contain at least one condition")
    if len(set(values)) != len(values):
        raise ValueError("--conditions contains duplicates")
    unknown = [value for value in values if value not in CONDITIONS]
    if unknown:
        raise ValueError(
            "--conditions contains unknown values: " + ", ".join(unknown)
        )
    return values


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    estimate = _estimate_cost(model_id, input_tokens, output_tokens)
    if estimate is not None:
        return estimate
    if "llama-4-maverick" in model_id.lower():
        return round((input_tokens * 0.27 + output_tokens * 0.85) / 1_000_000, 6)
    return None


def prepare_scenario(scenario_root: Path, scenario_id: str) -> None:
    env = os.environ.copy()
    env["SCENARIOS_DATA_DIR"] = str(scenario_root.resolve())
    subprocess.run(
        [sys.executable, "src/couchdb/init_data.py", "--reset-only"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, "src/couchdb/init_data.py", scenario_id],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
    )


def _routing_context(condition: str):
    if condition == "original":
        return patch.object(runner_module, "should_escalate", side_effect=_original_policy)
    if condition == "always":
        return patch.object(
            runner_module,
            "should_escalate",
            return_value=EscalationDecision(
                action=EscalationAction.VERIFY,
                reasons=["always verify experimental condition"],
            ),
        )
    return nullcontext()


def _original_policy(signals) -> EscalationDecision:
    """Reproduce PR #432 routing without enabling redesigned recovery."""
    reasons = []
    if signals.any_step_failed:
        reasons.append("failed step")
    if signals.dependency_depth >= 3:
        reasons.append("dependency depth >= 3")
    if signals.uses_specialist_servers:
        reasons.append("specialist server used")
    if signals.has_domain_terms:
        reasons.append("domain escalation term matched")
    return EscalationDecision(
        action=EscalationAction.VERIFY if reasons else EscalationAction.NONE,
        reasons=reasons,
    )


def _condition_decision(condition: str, signals) -> EscalationDecision:
    """Record the decision made by the selected experimental condition."""
    if condition == "baseline":
        return EscalationDecision(action=EscalationAction.NONE, reasons=[])
    if condition == "original":
        return _original_policy(signals)
    if condition == "always":
        return EscalationDecision(
            action=EscalationAction.VERIFY,
            reasons=["always verify experimental condition"],
        )
    return should_escalate(signals)


async def run_one(
    *,
    condition: str,
    scenario_id: str,
    question: str,
    model_id: str,
    trajectory_dir: Path,
) -> dict[str, Any]:
    backend = RecordingBackend(model_id)
    run_id = f"plan-execute-{condition}-{scenario_id}"
    os.environ["AGENT_TRAJECTORY_DIR"] = str(trajectory_dir)
    set_run_context(run_id=run_id, scenario_id=scenario_id)

    started = time.perf_counter()
    with _routing_context(condition):
        result = await PlanExecuteRunner(
            backend,
            adaptive_escalation=condition != "baseline",
            adaptive_recovery=condition in {"redesigned", "always"},
        ).run(question)
    duration_ms = (time.perf_counter() - started) * 1000

    signals = extract_escalation_signals(question, result.plan, result.trajectory)
    policy_decision = _condition_decision(condition, signals)
    calls = [asdict(call) for call in backend.calls]
    input_tokens = sum(call.input_tokens for call in backend.calls)
    output_tokens = sum(call.output_tokens for call in backend.calls)

    return {
        "scenario_id": scenario_id,
        "condition": condition,
        "run_id": run_id,
        "model": model_id,
        "question": question,
        "answer": result.answer,
        "plan_steps": len(result.plan.steps),
        "failed_steps": sum(not step.success for step in result.trajectory),
        "tool_steps": sum(
            bool(step.tool and step.tool.lower() not in {"none", "null"})
            for step in result.trajectory
        ),
        "policy_action": policy_decision.action.value,
        "policy_reasons": policy_decision.reasons,
        "runner_action": result.escalation_action,
        "routed_to_verification": any(call.kind == "verification" for call in backend.calls),
        "routed_to_recovery": any(
            step.recovery_attempted for step in result.trajectory
        ),
        "signals": asdict(signals),
        "llm_calls": len(backend.calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": estimate_cost(model_id, input_tokens, output_tokens),
        "duration_ms": duration_ms,
        "llm_duration_ms": sum(call.duration_ms for call in backend.calls),
        "planning_duration_ms": sum(
            call.duration_ms for call in backend.calls if call.kind == "planning"
        ),
        "summarization_duration_ms": sum(
            call.duration_ms for call in backend.calls if call.kind == "summarization"
        ),
        "verification_duration_ms": sum(
            call.duration_ms for call in backend.calls if call.kind == "verification"
        ),
        "execution_duration_ms": sum(
            step.duration_ms or 0.0 for step in result.trajectory
        ),
        "calls": calls,
    }


def evaluate_runs(
    records: list[dict[str, Any]],
    *,
    scenario_root: Path,
    scenario_ids: list[str],
    trajectory_root: Path,
    conditions: list[str] | tuple[str, ...] = CONDITIONS,
) -> None:
    evaluator = Evaluator(default_scorer="static_json")
    by_key = {(record["condition"], record["scenario_id"]): record for record in records}
    for condition in conditions:
        report = evaluator.evaluate(
            trajectories_path=trajectory_root / condition,
            scenarios_paths=[scenario_root],
            scenario_ids=scenario_ids,
        )
        for result in report.results:
            record = by_key[(condition, result.scenario_id)]
            record["passed"] = result.score.passed
            record["score"] = result.score.score
            record["score_rationale"] = result.score.rationale
            record["score_details"] = result.score.details


def _condition_summary(
    records: list[dict[str, Any]],
    conditions: list[str] | tuple[str, ...] = CONDITIONS,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        selected = [record for record in records if record["condition"] == condition]
        summary[condition] = {
            "scenarios": len(selected),
            "passed": sum(record["passed"] for record in selected),
            "pass_rate": (
                sum(record["passed"] for record in selected) / len(selected)
                if selected
                else 0.0
            ),
            "mean_score": (
                statistics.fmean(record["score"] for record in selected)
                if selected
                else 0.0
            ),
            "input_tokens": sum(record["input_tokens"] for record in selected),
            "output_tokens": sum(record["output_tokens"] for record in selected),
            "total_tokens": sum(record["total_tokens"] for record in selected),
            "llm_calls": sum(record["llm_calls"] for record in selected),
            "estimated_cost_usd": (
                sum(record["estimated_cost_usd"] for record in selected)
                if selected
                and all(record["estimated_cost_usd"] is not None for record in selected)
                else None
            ),
            "median_duration_ms": (
                statistics.median(record["duration_ms"] for record in selected)
                if selected
                else None
            ),
            "verification_rate": (
                sum(record["routed_to_verification"] for record in selected)
                / len(selected)
                if selected
                else 0.0
            ),
            "recovery_rate": (
                sum(record["routed_to_recovery"] for record in selected) / len(selected)
                if selected
                else 0.0
            ),
        }
    return summary


def write_outputs(
    records: list[dict[str, Any]],
    output_dir: Path,
    conditions: list[str] | tuple[str, ...] = CONDITIONS,
) -> dict[str, Any]:
    summary = {
        "conditions": _condition_summary(records, conditions),
        "comparison_status": (
            "complete"
            if set(CONDITIONS).issubset(conditions)
            else "partial_mechanism_check"
        ),
        "interpretation_warning": (
            "Conditions use independently planned single runs; score differences "
            "are diagnostic associations, not causal effects of escalation."
        ),
    }
    (output_dir / "runs.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fields = [
        "scenario_id",
        "condition",
        "passed",
        "score",
        "plan_steps",
        "failed_steps",
        "tool_steps",
        "policy_action",
        "runner_action",
        "routed_to_verification",
        "routed_to_recovery",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "duration_ms",
        "llm_duration_ms",
        "planning_duration_ms",
        "summarization_duration_ms",
        "verification_duration_ms",
        "execution_duration_ms",
    ]
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    return summary


async def run_experiment(args: argparse.Namespace) -> int:
    if not args.acknowledge_external_llm:
        raise ValueError(
            "--acknowledge-external-llm is required because benchmark content "
            "and tool evidence are sent to the configured model provider"
        )
    scenario_ids = parse_scenario_ids(args.scenario_ids)
    conditions = parse_conditions(args.conditions)
    scenario_root = args.scenario_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_root = output_dir / "trajectories"
    for condition in conditions:
        (trajectory_root / condition).mkdir(parents=True)

    os.environ["OTEL_TRACES_FILE"] = str(output_dir / "traces.jsonl")
    init_tracing("adaptive-escalation-experiment")

    records: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        question_path = scenario_root / f"scenario_{scenario_id}" / "question.txt"
        if not question_path.exists():
            raise FileNotFoundError(f"missing scenario question: {question_path}")
        question = question_path.read_text(encoding="utf-8").strip()
        for condition in conditions:
            print(f"running scenario={scenario_id} condition={condition}", flush=True)
            prepare_scenario(scenario_root, scenario_id)
            records.append(
                await run_one(
                    condition=condition,
                    scenario_id=scenario_id,
                    question=question,
                    model_id=args.model_id,
                    trajectory_dir=trajectory_root / condition,
                )
            )
            (output_dir / "runs.partial.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )

    evaluate_runs(
        records,
        scenario_root=scenario_root,
        scenario_ids=scenario_ids,
        trajectory_root=trajectory_root,
        conditions=conditions,
    )
    summary = write_outputs(records, output_dir, conditions)
    print(json.dumps(summary, indent=2))
    print(f"results={output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--scenario-ids", default="1")
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="comma-separated subset for gated runs (default: all four)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-external-llm",
        action="store_true",
        help="confirm that benchmark prompts and tool evidence may leave this host",
    )
    return parser


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    try:
        return asyncio.run(run_experiment(build_parser().parse_args()))
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
