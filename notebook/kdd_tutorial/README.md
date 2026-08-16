# AssetOpsBench KDD Tutorial — MCP Tools, Agents, and Evaluation

Hands-on materials for a three-hour tutorial that moves from running a first industrial agent query, to inspecting the MCP tools behind its answer, to evaluating the resulting trajectory with AssetOpsBench.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| [`0_environment_setup.ipynb`](./0_environment_setup.ipynb) | Beginner setup for cloning AssetOpsBench, installing Python and `uv`, configuring the single `KDD_MODEL_ID`, installing Docker, loading the bundled CouchDB sample data, and selecting the correct Jupyter kernel. |
| [`00_mcp_agents_architecture.ipynb`](./00_mcp_agents_architecture.ipynb) | Introduces industrial asset operations, AssetOpsBench, MCP architecture, tools, agents, execution trajectories, safety, and evaluation through beginner-friendly explanations and direct examples. |
| [`000_agent_hello_world.ipynb`](./000_agent_hello_world.ipynb) | Runs the first read-only agent queries with `plan-execute` and Stirrup, checks model access, persists a trajectory, and reveals the executed tool calls. |
| [`01_mcp_utilities.ipynb`](./01_mcp_utilities.ipynb) | Provides a fast first MCP success using dependency-free utility tools such as the current date and time, while teaching the MCP request/response pattern. |
| [`02_mcp_iot.ipynb`](./02_mcp_iot.ipynb) | Explores sites and assets, compares installed sensor metadata with measured telemetry, and demonstrates the IoT and asset-registry MCP tools. |
| [`03_mcp_fmsr_fixed.ipynb`](./03_mcp_fmsr_fixed.ipynb) | Discovers the FMSR contract, retrieves stored failure modes, and optionally generates failure-mode suggestions using the same `KDD_MODEL_ID` without writing them to CouchDB. |
| [`04_mcp_workorders_final.ipynb`](./04_mcp_workorders_final.ipynb) | Exercises the complete read-only work-order contract, including listing, filtering, record details, costs, schedules, KPIs, assignments, tasks, and failure codes. |
| [`05_mcp_tsfm_final.ipynb`](./05_mcp_tsfm_final.ipynb) | Demonstrates TSFM tasks, time-series profiling and characterization, data-quality analysis, model and feature catalogs, recipe contracts, optional forecasting, and run provenance. |
| [`06_mcp_vibration.ipynb`](./06_mcp_vibration.ipynb) | Applies vibration-domain calculations and severity assessment, then shows how deterministic diagnostic tools connect with telemetry evidence. |
| [`07_end_to_end_stirrup.ipynb`](./07_end_to_end_stirrup.ipynb) | Runs a complete read-only Stirrup scenario that retrieves a real work order, assigns a grounded failure-code description, persists the trajectory, and audits tool use and output safety. |
| [`07b_end_to_end_stirrup_advanced.ipynb`](./07b_end_to_end_stirrup_advanced.ipynb) | Loads the repository's `scenario_kdd_fcc` data into local CouchDB and runs the advanced `TST-WO00032` failure-code classification scenario. |
| [`08_evaluation_static_json.ipynb`](./08_evaluation_static_json.ipynb) | Evaluates the saved Stirrup work-order trajectory with AssetOpsBench's deterministic `static_json` scorer and reports pass rate and strict exact match. |
| [`09_leaderboard.ipynb`](./09_leaderboard.ipynb) | Aggregates evaluator outputs into a reproducible leaderboard while keeping tools-only and code-enabled agent tracks separate. |

## Three-hour tutorial path

| Time | Theme | Recommended notebooks |
| --- | --- | --- |
| Hour 1 | Basic setup, AssetOpsBench introduction, and agent hello world | [`0_environment_setup.ipynb`](./0_environment_setup.ipynb), [`00_mcp_agents_architecture.ipynb`](./00_mcp_agents_architecture.ipynb), [`000_agent_hello_world.ipynb`](./000_agent_hello_world.ipynb) |
| Hour 2 | MCP tools and ecosystem, focusing on IoT and TSFM | [`02_mcp_iot.ipynb`](./02_mcp_iot.ipynb), [`05_mcp_tsfm_final.ipynb`](./05_mcp_tsfm_final.ipynb) |
| Hour 3 | Scenario, trajectory, and deterministic evaluation | [`07_end_to_end_stirrup.ipynb`](./07_end_to_end_stirrup.ipynb), [`08_evaluation_static_json.ipynb`](./08_evaluation_static_json.ipynb) |

The remaining notebooks are optional references or extensions. The tutorial does not require running every notebook.

## Quick start

Run these commands from the AssetOpsBench repository root:

```bash
# 1. Install the project environment
uv sync

# 2. Create the private configuration file
cp .env.public .env

# 3. In .env, configure the tutorial model and TokenRouter credentials
# KDD_MODEL_ID=tokenrouter/MiniMax-M3

# 4. Start CouchDB; the bundled sample data is loaded automatically
docker compose -f src/couchdb/docker-compose.yaml up -d

# 5. Start Jupyter
uv run jupyter lab
```

All notebooks load the repository `.env` with `override=True`, so `KDD_MODEL_ID` is the single model selection used throughout the tutorial. Restart the kernel after changing `.env`.

## Example questions

### Single-server

```text
What sensors are on Chiller 6?
Is LSTM model supported in TSFM?
Get the work order of equipment CWC04013 for year 2017.
```

### Multi-step and multi-server

```text
What is the current date and time? Also list assets at site MAIN. Also get the sensor list and failure-mode list for any chiller at site MAIN.
```

## Data and outputs

- Tutorial CouchDB data is bundled under `src/couchdb/scenarios_data/`; participants should not upload private operational data.
- Generated logs, trajectories, evaluation results, and leaderboard artifacts are written under `artifacts/kdd_tutorial/`.
- Do not commit `.env`, print API keys, or include credentials in screenshots.
- Keep Stirrup tools-only and code-enabled results on separate leaderboard tracks.
