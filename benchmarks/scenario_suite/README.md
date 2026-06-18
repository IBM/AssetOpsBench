# Benchmark Runner

This folder contains the scenario list and usage notes for the benchmark.

In the benchmark, users can add the scenario IDs they want to execute.

The benchmark runner executes each scenario sequentially, saves trajectories, and then invokes the existing evaluation pipeline to generate per-scenario and aggregate reports.

## Scenario ID file

The benchmark registry is a plain text file:

```text
benchmarks/scenario_suite/scenarios.txt
```

Each line contains one scenario id:

```text
11
12
14
15
```

Blank lines and lines starting with `#` are ignored, so you can also use comments:

```text
# User 1
11
12
14
15

# User 2
21
22
23
```

## Expected scenario folder layout

The runner expects a scenario root directory containing folders like:

```text
scenarios_data/
  scenario_11/
    question.txt
    manifest.json
    groundtruth.txt
  scenario_12/
    question.txt
    manifest.json
    groundtruth.txt
```

For each scenario:

- `question.txt` is passed to the agent
- `manifest.json` is used by couchdb to load the data
- `groundtruth.txt` is used by the evaluator

The scenario folder name must match the id from `scenarios.txt`:

- `11` → `scenario_11`
- `12` → `scenario_12`

## Run direct LLM

Run the direct LLM baseline sequentially over the listed scenarios:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name direct_llm --model-id tokenrouter/MiniMax-M3
```

This writes trajectories to:

```text
traces/trajectories/scenario_suite/direct_llm/
```

and reports to:

```text
reports/scenario_suite/direct_llm/
```

## Run Stirrup agent

Run the Stirrup agent sequentially over the listed scenarios:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name stirrup_agent
```

Run the Stirrup agent sequentially over the listed scenarios using the MiniMax model

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids benchmarks/scenario_suite/scenarios.txt \
  --scenario-root /.../scenarios_data \
  --agent_name stirrup_agent \
  --model-id tokenrouter/MiniMax-M3
```

This writes trajectories to:

```text
traces/trajectories/scenario_suite/stirrup_agent/
```

and reports to:

```text
reports/scenario_suite/stirrup_agent/
```

## Run all agents

Run all supported agents one after the other:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name all
```

## Useful options

### Dry run

Print the commands without executing them:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name direct_llm   --dry-run
```

### Skip existing trajectories

Skip scenarios whose trajectory files already exist:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name direct_llm   --skip-existing
```

### Continue after errors

Keep running later scenarios even if one fails:

```bash
uv run python -m benchmark.scenario_suite_runner   --scenario-ids benchmarks/scenario_suite/scenarios.txt   --scenario-root /.../scenarios_data   --agent_name direct_llm   --continue-on-error
```

## Environment variables

The direct LLM baseline uses TokenRouter by default. Set these before running:

```bash
export TOKENROUTER_API_KEY=your_tokenrouter_key
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
```

If you use a different model or backend, set the corresponding environment variables required by that backend.

## Output layout

Typical outputs look like this:

```text
traces/trajectories/scenario_suite/
  direct_llm/
    direct_llm_11.json
    direct_llm_12.json
    direct_llm_14.json
    direct_llm_15.json
  stirrup_agent/
    stirrup_agent_11.json
    stirrup_agent_12.json
```

```text
reports/scenario_suite/
  direct_llm/
    direct_llm_11.json
    direct_llm_12.json
    _aggregate.json
  stirrup_agent/
    stirrup_agent_11.json
    stirrup_agent_12.json
    _aggregate.json
```

Each per-scenario report contains the final answer, score, and operational metrics. The aggregate report summarizes the full batch.

## Tests

Run the benchmark runner tests with:

```bash
uv run pytest src/benchmark/tests/test_scenario_suite_runner.py -v
```

Run all benchmark tests with:

```bash
uv run pytest src/benchmark/tests -v
```
