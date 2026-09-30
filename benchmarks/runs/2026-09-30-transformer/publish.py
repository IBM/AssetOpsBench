# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Export a measured comparison, or rebuild its tables and graphs without model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from benchmark.measurement import summarize, suite_hash

MODELS = {
    "opus-5-5": "Opus 5.5",
    "gpt-6-astra": "GPT-6 Astra",
    "glm-5-3-low": "GLM 5.3 (low)",
    "gpt-6-1-sol": "GPT-6.1 Sol",
    "fable-5-1": "Fable 5.1",
}
COLORS = ["#42684f", "#60816a", "#90a98c", "#b2bc9c", "#d4b477"]
RUBRICS = {
    "task_completion": "Task completion",
    "data_retrieval_accuracy": "Data accuracy",
    "generalized_result_verification": "Result verification",
    "agent_sequence_correct": "Tool sequence",
    "clarity_and_justification": "Clarity",
    "hallucinations": "No hallucinations",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export(source, suite, env_file):
    """Copy only ledger-linked evidence; remove credentials and local path prefixes."""
    if (HERE / "models").exists():
        raise SystemExit("Evidence already exists; omit --source to rebuild the report.")
    source, suite = source.resolve(), suite.resolve()
    secrets = []
    if env_file and env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", line)
            if match and re.search(r"KEY|TOKEN|SECRET|PASSWORD", match[1], re.I):
                value = match[2].strip().strip("\"'")
                if value and not value.startswith("${"):
                    secrets.append((match[1], value))
    redactions = Counter()
    paths = {}
    for key in MODELS:
        target = source / key
        for record_file in sorted((target / "measurements").glob("*.json")):
            record = json.loads(record_file.read_text())
            refs = [record["trace_file"]]
            grade = record.get("grading") or {}
            refs.extend(a["trace_file"] for a in grade.get("attempts", []))
            if grade.get("trace_file"):
                refs.append(grade["trace_file"])
            for ref in refs:
                path = Path(ref)
                if not path.is_file():
                    raise ValueError(f"Missing evidence: {path.name}")
                paths[str(path)] = f"models/{key}/events/{path.name}.gz"

    def portable(text):
        # Longer paths first, before replacing their shared directory prefixes.
        for old, new in sorted(paths.items(), key=lambda pair: -len(pair[0])):
            text = text.replace(old, new)
        text = text.replace(str(suite), "suite").replace(str(source), "models")
        text = text.replace(str(REPO), "<workspace>")
        text = text.replace(str(Path.home()), "<home>")
        for name, secret in secrets:
            # Short passwords are replaced only when visibly used as credentials.
            if len(secret) >= 8:
                redactions[name] += text.count(secret)
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)
        return text

    for old, new in paths.items():
        dest = HERE / new
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(gzip.compress(portable(Path(old).read_text()).encode(), mtime=0))
    for key in MODELS:
        target = source / key
        for name in ("target.json", "settings.json", "environment.json"):
            write_json(HERE / "models" / key / name, json.loads(portable((target / name).read_text())))
        for path in sorted((target / "measurements").glob("*.json")):
            record = json.loads(path.read_text())
            write_json(HERE / "models" / key / "measurements" / path.name, json.loads(portable(path.read_text())))
            if record["status"] == "completed":
                trajectory = target / "trajectories" / f"{record['run_id']}.json"
                write_json(HERE / "models" / key / "trajectories" / trajectory.name,
                           json.loads(portable(trajectory.read_text())))
    # Preserve the scenario JSON bytes so the recorded suite hash remains verifiable.
    for name in ("run.json", "scenarios.json", "negative_scenarios.json"):
        path = suite / name
        if portable(path.read_text()) != path.read_text():
            raise ValueError("Scenario source needs redaction; its original hash cannot be retained.")
        dest = HERE / "suite" / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
    for path in sorted((suite / "logs").rglob("*")):
        if path.is_file():
            dest = HERE / "suite" / "generation-evidence" / path.relative_to(suite / "logs")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(portable(path.read_text()))
    (HERE / "comparison.html").write_text(portable((source / "comparison.html").read_text()))
    (HERE / "preview.jpg").write_bytes((source / "preview.jpg").read_bytes())
    write_json(HERE / "manifest.json", {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "implementation_commit_at_publication": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_run": source.name,
        "source_generation_run": suite.name,
        "judge": "claude-code/claude-fable-5-1",
        "evidence_policy": "Only retained measurement-linked execution and judge traces; all observed payloads, compressed JSONL.",
        "redaction_policy": "Credential values and local home/workspace path prefixes removed; timestamps, usage, outcomes and durations unchanged.",
        "credential_redactions": dict(redactions),
        "database_snapshot_included": False,
    })


def load_and_validate():
    scenarios = []
    for name in ("scenarios.json", "negative_scenarios.json"):
        rows = json.loads((HERE / "suite" / name).read_text())
        if name.startswith("negative"):
            rows = [{**row, "type": "negative"} for row in rows]
        scenarios.extend(rows)
    ids = {str(s["id"]) for s in scenarios}
    digest = suite_hash([HERE / "suite" / name for name in ("scenarios.json", "negative_scenarios.json")])
    groups, sessions = {}, []
    for key in MODELS:
        records = [json.loads(p.read_text()) for p in sorted((HERE / "models" / key / "measurements").glob("*.json"))]
        latest = {}
        for record in records:
            sid = record["scenario_id"]
            if sid not in latest or record["attempt"] > latest[sid]["attempt"]:
                latest[sid] = record
            assert record["execution_start"] and record["execution_end"]
            assert record["execution_duration_ms"] > 0
            assert record["settings"]["suite_sha256"] == digest
            for ref in [record["trace_file"], *[a["trace_file"] for a in (record.get("grading") or {}).get("attempts", [])]]:
                events = [json.loads(line) for line in gzip.decompress((HERE / ref).read_bytes()).decode().splitlines()]
                assert all(event.get("timestamp") for event in events)
                if ".judge" in ref:
                    sessions.extend(e["payload"]["session_id"] for e in events if e["kind"] == "judge_result")
        assert set(latest) == ids
        assert all(r["status"] == "completed" and r["grading"]["status"] == "completed" for r in latest.values())
        groups[key] = {"records": records, "latest": sorted(latest.values(), key=lambda r: r["execution_index"])}
    assert len(sessions) == 260 and len(set(sessions)) == 260 and all(sessions)
    assert sum(len(g["records"]) for g in groups.values()) == 263
    return scenarios, groups, digest


def report(scenarios, groups, digest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#d8dfd9", "text.color": "#24382a",
                         "axes.labelcolor": "#526356", "xtick.color": "#526356", "ytick.color": "#526356",
                         "figure.facecolor": "white", "axes.facecolor": "white", "svg.hashsalt": "transformer-20260930"})
    summaries = {}
    for key, group in groups.items():
        cases, attempts = group["latest"], group["records"]
        case_summary, attempt_summary = summarize(cases), summarize(attempts)
        total_tools = attempt_summary["tool_call_count"]["total"]
        total_errors = attempt_summary["tool_errors"]["total"]
        summaries[key] = {
            "name": MODELS[key], "cases": case_summary, "attempts": attempt_summary,
            "passed": sum(r["grading"]["result"]["score"]["passed"] for r in cases),
            "total_attempt_execution_ms": sum(r["execution_duration_ms"] for r in attempts),
            "tool_error_rate_all_attempts": total_errors / total_tools if total_tools is not None and total_errors is not None and total_tools else None,
        }
    write_json(HERE / "summary.json", summaries)
    keys, labels = list(MODELS), list(MODELS.values())
    graphs = HERE / "graphs"
    graphs.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(graphs / f"{name}.png", dpi=180, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        svg = graphs / f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    rates = [summaries[k]["cases"]["pass_rate"] * 100 for k in keys]
    ax.barh(labels, rates, color=COLORS, height=.55)
    for i, key in enumerate(keys):
        ax.text(rates[i] + 1, i, f"{summaries[key]['passed']}/52  ·  {rates[i]:.1f}%", va="center", fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 70); ax.set_xlabel("Pass rate (%) · independent Fable 5.1 judge")
    ax.set_title("Transformer · 52 open-form scenarios", loc="left", pad=14, fontweight="bold")
    save(fig, "pass-rate")

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for key, color in zip(keys, COLORS):
        seconds = sorted(r["execution_duration_ms"] / 1000 for r in groups[key]["latest"])
        ax.step(seconds, np.arange(1, len(seconds) + 1) / len(seconds) * 100,
                where="post", color=color, linewidth=2, label=MODELS[key])
    ax.set_xlabel("Entire agent invocation (seconds)"); ax.set_ylabel("Scenarios completed (%)")
    ax.set_ylim(0, 102); ax.grid(axis="y", alpha=.15); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Execution time · latest completed attempt per scenario", loc="left", pad=14, fontweight="bold")
    save(fig, "execution-time")

    fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
    matrix = np.array([[summaries[k]["cases"]["rubric_success_rates"][r]["success_rate"] * 100 for r in RUBRICS] for k in keys])
    from matplotlib.colors import LinearSegmentedColormap
    ax.imshow(matrix, vmin=0, vmax=100, cmap=LinearSegmentedColormap.from_list("success", ["#faf9f2", "#cfdbc9", "#42684f"]), aspect="auto")
    ax.set_xticks(range(len(RUBRICS)), list(RUBRICS.values()), fontsize=9)
    ax.set_yticks(range(len(keys)), labels)
    for i in range(len(keys)):
        for j in range(len(RUBRICS)):
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center", color="white" if matrix[i,j] > 75 else "#24382a")
    ax.set_title("Rubric success · 52 judgments per cell", loc="left", pad=14, fontweight="bold")
    save(fig, "rubric-success")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for ax, metric, divisor, title in zip(axes, ["input_tokens", "output_tokens", "tool_call_count"],
                                         [1000, 1000, 1], ["Mean input · k tokens", "Mean output · k tokens", "Mean tool calls"]):
        values = [summaries[k]["cases"][metric]["mean"] / divisor for k in keys]
        ax.barh(labels, values, color=COLORS, height=.55); ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.set_xlim(0, max(values) * 1.3)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .03, i, f"{value:,.1f}", va="center", fontsize=9)
    save(fig, "resources")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for ax, values, title in [
        (axes[0], [summaries[k]["tool_error_rate_all_attempts"] * 100 for k in keys], "Tool calls with errors · all attempts (%)"),
        (axes[1], [summaries[k]["attempts"]["run_error_rate"] * 100 for k in keys], "Failed / timed-out invocations (%)"),
    ]:
        ax.barh(labels, values, color=COLORS, height=.55); ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.set_xlim(0, max(max(values) * 1.4, 1))
        for i, value in enumerate(values):
            ax.text(value + ax.get_xlim()[1] * .03, i, f"{value:.1f}%", va="center", fontsize=9)
    save(fig, "reliability")

    fields = ["model", "scenario_id", "type", "execution_index", "attempt", "status", "passed", "score",
              "execution_start", "execution_end", "execution_duration_ms", "grading_duration_ms",
              "time_to_first_response_ms", "time_to_final_answer_ms", "input_tokens", "output_tokens", "reasoning_tokens",
              "cache_read_tokens", "cache_write_tokens", "tool_call_count", "tool_errors", "tool_timeouts", "repeated_identical_calls",
              "model_turns", "context_compactions", "turn_limit_hit", "database_writes_attempted", "database_writes_succeeded",
              "database_records_created", "database_records_changed", "actual_api_cost_usd", "estimated_cost_usd", *RUBRICS, "judge_rationale"]
    types = {str(s["id"]): s["type"] for s in scenarios}
    with (HERE / "cases.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for key in keys:
            for r in groups[key]["latest"]:
                grade = r["grading"]; score = grade["result"]["score"]
                row = {f: r.get(f, r["metrics"].get(f)) for f in fields}
                row.update(model=MODELS[key], type=types[r["scenario_id"]], passed=score["passed"], score=score["score"],
                           grading_duration_ms=grade["duration_ms"], judge_rationale=score["details"].get("suggestions", score["rationale"]))
                row.update({k: score["details"].get(k) for k in RUBRICS})
                writer.writerow(row)

    table = []
    token_table = []
    for key in keys:
        item = summaries[key]; c = item["cases"]; a = item["attempts"]
        table.append(f"| {MODELS[key]} | {item['passed']}/52 | {c['pass_rate']:.1%} | {c['mean_score']:.3f} | {c['median_execution_ms']/1000:.1f} | {c['p95_execution_ms']/1000:.1f} | {c['median_grading_ms']/1000:.1f} | {c['tool_call_count']['mean']:.1f} | {a['attempted']-a['completed']}/{a['attempted']} ({a['run_error_rate']:.1%}) |")
        fmt = lambda metric: "—" if c[metric]["total"] is None else f"{c[metric]['total']:,}"
        token_table.append(f"| {MODELS[key]} | {fmt('input_tokens')} | {fmt('output_tokens')} | {fmt('reasoning_tokens')} | {fmt('cache_read_tokens')} | {fmt('cache_write_tokens')} |")
    settings = [json.loads((HERE / "models" / k / "settings.json").read_text()) for k in keys]
    policy = settings[0]["database_policy"]
    first = min(r["execution_start"] for g in groups.values() for r in g["records"])
    last = max(r["grading"]["end"] for g in groups.values() for r in g["latest"])
    document = f"""# Transformer comparison · 2026-09-30

[Offline HTML](comparison.html) · [Per-scenario CSV](cases.csv) · [Summary JSON](summary.json) · [Run manifest](manifest.json)

Five models completed the same **52 scenarios** (50 positive, 2 negative), with **260 independent Fable 5.1 judgments**. Opus 5.5 generated the suite in open form using Semantic Scholar research; the model chose the positive distribution: 9 IoT, 11 FMSR, 6 TSFM, 7 work-order and 17 multiagent scenarios.

![Pass rates](graphs/pass-rate.png)

| Model | Passes | Pass rate | Mean score | Median exec (s) | p95 exec (s) | Median grade (s) | Mean tools | Failed attempts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Timing and aggregation

![Execution time distribution](graphs/execution-time.png)

Execution timing surrounds the **entire agent invocation**, including process/SDK startup, MCP setup, model and tool work, answer persistence, cleanup and exit. Grading is timed separately. The first-response metric is the first observed assistant/protocol response, not first-token latency. Request-level timing is available for GLM; Claude and Codex do not expose every underlying request boundary.

Pass rates, score, rubric rates, resource means and the execution distribution use the latest completed attempt for each scenario: 52 observations per model. p95 uses the nearest-rank method. Reliability uses **all 263 retained attempts**: three GLM attempts hit the 30-turn limit and subsequently succeeded on retry. `summary.json` keeps case and attempt aggregates separately, including total invocation time across retries. These sums exclude scheduling gaps and are not elapsed suite time. Excluded obsolete results and invalid infrastructure runs are not part of this publication.

Retained execution/judgment window (UTC): `{first}` through `{last}`. Execution targets ran concurrently, with configured concurrency 5 and serial suite order within each target. Independent grading workers ran alongside execution. Actual overlap varied, including a resumed GLM phase; timestamps preserve the observed order.

## Rubric

![Rubric success rates](graphs/rubric-success.png)

The existing AssetOpsBench LLM judge checks task completion, data accuracy, result verification, tool sequence, clarity and hallucinations against each generated characteristic answer. A pass requires all five positive criteria and no hallucinations. Score is the fraction of the first five criteria satisfied, with a 0.2 hallucination penalty. The last heatmap column means **no hallucinations**; raw CSV/JSON `hallucinations=true` means the adverse finding.

Judge: `claude-code/claude-fable-5-1`, Claude CLI 2.1.286. Each judgment used a fresh, tool-free session independent of the executor; all 260 successful judge session IDs are distinct. Fable execution was judged by the same model in a separate session, as requested. The existing evaluator caps trajectory evidence at **8,000 characters**; final answer and characteristic behavior are provided separately. Full untruncated observed execution and judge traces are included. Independent sessions do not eliminate same-model preference bias.

## Tokens and tools

![Mean resource use](graphs/resources.png)

| Model | Input tokens | Output tokens | Reasoning tokens | Cache read | Cache write |
|---|---:|---:|---:|---:|---:|
{chr(10).join(token_table)}

Totals above cover the latest 52 execution attempts, excluding judging. Input includes provider-reported cached input; output uses provider totals. Reasoning and cache subdivisions remain missing when the runtime does not report them. JSON uses `null`, CSV uses empty cells, and tables use “—” for unavailable values. Availability denominators and retry-inclusive totals are in `summary.json`. Actual billed costs are unavailable; CLI estimates are retained separately in each measurement and must not be treated as subscription charges.

![Tool and invocation reliability](graphs/reliability.png)

Tool errors include explicit MCP errors and benchmark error payloads. The error fraction uses all retained tool calls, including failed invocation attempts. Timeout and internal retry counts remain missing where they cannot be reliably distinguished. Repeated identical calls are recorded separately and are not assumed to be retries. Database auditing records actual attempted/succeeded writes and created/changed record IDs in the measurement ledger.

## Environment and limits

| Model ID | Provider / harness | CLI / SDK | Reasoning | Invocation timeout | Turn cap |
|---|---|---|---|---:|---:|
| `claude-opus-5-5` | Anthropic / Claude Agent SDK | CLI 2.1.286 / SDK 0.1.56 | Provider default, not reported | 900 s | 30 |
| `gpt-6-astra` | OpenAI / Codex CLI subscription | CLI 0.159.0 | Provider default, not reported | 900 s (inner CLI 840 s) | Not exposed |
| `zai/glm-5.3` | z.ai / OpenAI Agents SDK | SDK 0.13.6 | low | 900 s | 30 |
| `gpt-6.1-sol` | OpenAI / Codex CLI subscription | CLI 0.159.0 | Provider default, not reported | 900 s (inner CLI 840 s) | Not exposed |
| `claude-fable-5-1` | Anthropic / Claude Agent SDK | CLI 2.1.286 / SDK 0.1.56 | Provider default, not reported | 900 s | 30 |

Execution temperature and output-token caps were not explicitly set; unknown effective defaults are missing. Generation requests use a 131,072-token ceiling clipped to the known model output limit; CLI runtimes do not expose a controllable effective token cap. GLM's provider SDK allows two internal retries; whole-invocation retries are bounded at three in the current runner. Grading allows three attempts on backend/parse failure. Historical per-record settings are preserved, including fields introduced during the run; absence is not backfilled.

All targets started from the same database snapshot: **{policy['database_count']} databases, {policy['document_count']:,} documents**. Each model had an isolated namespace; state persisted across scenarios in suite order with **no per-scenario reset**. Partial writes from retries also persist. The raw starting database snapshot is not bundled; its hash and policy are recorded, so rebuilding these charts is reproducible from the published evidence, while rerunning models requires an equivalent database environment.

**Shared tool limitation:** FMSR LLM-dependent tools returned “LLM unavailable” because their default Watsonx model (`watsonx/meta-llama/llama-3-3-70b-instruct`) was not configured. This affected DGA interpretation and related workflows for all models. The grader still applies the original expected tool behavior. These scores compare the recorded model/harness combinations in this environment; they are not an isolated model-ability ranking.

- Suite SHA-256: `{digest}`
- Rubric SHA-256: `{settings[0]['rubric_sha256']}`
- Initial database snapshot SHA-256: `{policy['snapshot_sha256']}`

## Evidence and reproduction

`suite/` contains the exact scenario files, generation manifest and research/generation evidence. `models/<target>/settings.json` records model/provider/runtime versions and limits; `environment.json` records the snapshot/reset policy. `measurements/` contains every retained attempt, outcome, rubric, rationale, timestamps, token/tool metrics, database audit and cost availability. `trajectories/` contains successful evaluator inputs. `events/*.jsonl.gz` contains full observed execution messages, tool arguments/outputs/errors and judge inputs/results. Trace references are relative to this run folder. Credential values and local path prefixes are removed during publication; measured numbers and timestamps are preserved.

Open `comparison.html` locally for the self-contained interactive comparison. GitHub's file view does not execute HTML; download it or serve it locally. Static graphs above are also available as SVG for export. `checksums.sha256` covers published evidence and artifacts.

Rebuild the graphs, CSV, summary, README and checksums without invoking any model:

```bash
uv run benchmarks/runs/2026-09-30-transformer/publish.py
```

Launch a new five-model run against the published suite after configuring the model authentication and CouchDB environment:

```bash
# The configuration points to the committed transformer suite.
PYTHONPATH=src .venv/bin/python tools/run_generated_comparison.py \\
  --output-dir generated/comparisons/transformer-new
```

See [runner setup](../../generated-scenarios.md) for authentication, isolation and live grading. No model calls are made by this publication script. The implementation commit at publication is recorded in `manifest.json`; this is a code reference, not a claim that the runs originally executed from a clean committed tree.
"""
    (HERE / "README.md").write_text(document)
    files = sorted(p for p in HERE.rglob("*") if p.is_file() and p.name != "checksums.sha256" and "__pycache__" not in p.parts)
    (HERE / "checksums.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE)}\n" for p in files))
    print(f"Validated {len(scenarios)} scenarios, 263 attempts, 260 unique judge sessions; wrote five graphs and report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="First publication only: measured comparison directory")
    parser.add_argument("--suite", type=Path, help="Original completed generation directory")
    parser.add_argument("--redact-env-file", type=Path, help="Private credential file to scan without copying it")
    args = parser.parse_args()
    if args.source:
        if not args.suite:
            parser.error("--source requires --suite")
        export(args.source, args.suite, args.redact_env_file)
    scenarios, groups, digest = load_and_validate()
    report(scenarios, groups, digest)


if __name__ == "__main__":
    main()
