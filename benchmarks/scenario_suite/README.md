# Scenario Suite Runner

The runner executes selected scenarios sequentially, saves each trajectory, and
runs evaluation unless `--no-evaluate` is set.

## Select scenarios

`--scenario-ids` accepts a named selector:

```text
<category>[+<category>...]_<all|lite|mini|open>
```

Categories are `car`, `fcc`, `fmea`, `fmsr`, `health`, `tsfm`, and `wosr`.
The `all`, `lite`, `mini`, and `open` shorthands select every category from the
corresponding profile.

Profiles are loaded from `all.yaml`, `lite.yaml`, `mini.yaml`, and `open.yaml`
in this directory. The Mini profile contains:

| Category | Scenario IDs |
| -------- | ------------ |
| CAR | 152, 157, 158, 159, 165, 166, 168, 191 |
| FCC | 303, 318, 322, 324 |
| FMEA | 9001, 9003, 9027 |
| FMSR | 902, 906, 913, 914, 915 |
| Health | 402, 409 |
| TSFM | 1008, 1014, 1015, 1019 |
| WOSR | 10, 14, 18, 21, 22, 28, 36, 45, 49 |

The Lite profile contains:

| Category | Scenario IDs |
| -------- | ------------ |
| CAR | 151, 152, 153, 157, 159, 160, 178, 180, 182, 183, 189, 193 |
| FCC | 303, 304, 305, 308, 312, 320, 322, 325 |
| FMEA | `[]` (none) |
| FMSR | 902, 904, 905, 906, 908, 915, 916, 917, 920, 923, 928 |
| Health | 401–410 |
| TSFM | 1001–1025 |
| WOSR | 11, 13, 16, 20, 24, 31, 43, 50, 61, 66 |

Individual categories may use an empty list, such as `tsfm: []`; the profile as
a whole must contain at least one scenario ID. Profile shorthands such as `lite`
simply skip empty categories.

Examples:

```bash
--scenario-ids fcc_lite
--scenario-ids fcc_mini
--scenario-ids fmea_all
--scenario-ids fcc+fmsr_all
--scenario-ids mini
--scenario-ids lite
--scenario-ids all
```

A profile YAML file can also be passed directly:

```bash
--scenario-ids benchmarks/scenario_suite/lite.yaml
```

Plain-text files are supported too. Put one scenario ID on each line; blank
lines and `#` comments are ignored.

## Scenario data layout

The scenario root must contain one directory per selected ID:

```text
scenarios_data/
  scenario_151/
    question.txt
    manifest.json
    groundtruth.txt
```

`question.txt` is passed to the agent, `manifest.json` loads the scenario into
CouchDB, and `groundtruth.txt` is required by evaluation.

## Run scenarios

Direct LLM baseline:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids lite \
  --scenario-root /path/to/scenarios_data \
  --agent_name direct_llm \
  --model-id tokenrouter/MiniMax-M3
```

Available agent names are `direct_llm`, `stirrup_agent`, `opencode_agent`,
`gemini_cli_agent`, `openclaw_cli_agent`, and `all`.

## Useful options

| Option | Behavior |
| ------ | -------- |
| `--dry-run` | Print commands without executing them. |
| `--skip-existing` | Skip a scenario when its expected trajectory already exists; default is false. |
| `--continue-on-error` | Continue after a scenario fails. |
| `--no-evaluate` | Save trajectories without running evaluation. |
| `--preserve-workspaces` | Keep existing per-run workspaces. |

With `--skip-existing`, the runner checks:

```text
<trajectory-root>/<agent>/<model-slug>/<agent>_<scenario-id>.json
```

For example:

```bash
uv run python -m benchmark.scenario_suite_runner \
  --scenario-ids fcc+fmsr_all \
  --scenario-root /path/to/scenarios_data \
  --agent_name direct_llm \
  --model-id tokenrouter/MiniMax-M3 \
  --skip-existing
```

## Environment

For `tokenrouter/*` models, set:

```bash
export TOKENROUTER_API_KEY=your_tokenrouter_key
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1
```

For `litellm_proxy/*` models, set `LITELLM_API_KEY` and `LITELLM_BASE_URL`.

## Output layout

Outputs are nested by agent and model slug:

```text
traces/trajectories/scenario_suite/
  direct_llm/
    tokenrouter-MiniMax-M3/
      direct_llm_151.json

reports/scenario_suite/
  direct_llm/
    tokenrouter-MiniMax-M3/
      _aggregate.json
```

Each aggregate report contains matched scenario results, operational metrics,
and score summaries for that agent/model pair.

## Tests

```bash
uv run pytest src/benchmark/tests/test_scenario_suite_runner.py -q
```
