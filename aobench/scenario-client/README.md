# scenario-client

A Python client library for interacting with scenario servers, with integrated MLflow tracking for benchmarking and evaluation workflows.

## Installation

### Using uv (recommended)

```bash
uv pip install "https://github.com/IBM/AssetOpsBench.git#subdirectory=aobench/scenario-client"
```

### Using pip

```bash
pip install "https://github.com/IBM/AssetOpsBench.git#subdirectory=aobench/scenario-client"
```

## Requirements

- Python >= 3.12
- httpx >= 0.28.1
- load-dotenv >= 0.1.0
- mlflow >= 3.4.0

## Quick Start

```python
from scenario_client.client import AOBench

# Initialize the client
client = AOBench(
    scenario_uri="https://your-scenario-server.com",
    tracking_uri="https://your-mlflow-server.com"  # Optional
)

# Get available scenario types
types = client.scenario_types()
print(types)

# Load a scenario set
scenario_set, tracking_context = client.scenario_set(
    scenario_set_id="my-scenario-set",
    tracking=True  # Enable MLflow tracking
)

# Run scenarios
answers = []
for scenario in scenario_set["scenarios"]:
    answer = client.run(
        func=your_function,
        scenario_id=scenario["id"],
        tracking_context=tracking_context,
        **scenario["inputs"]
    )
    answers.append(answer)

# Grade the results
results = client.grade(
    scenario_set_id="my-scenario-set",
    answers=answers,
    tracking_context=tracking_context
)
print(results)
```

## Features

### Synchronous and Asynchronous Execution

The client supports both sync and async workflows:

```python
# Synchronous
answer = client.run(
    func=my_sync_function,
    scenario_id="scenario-1",
    **kwargs
)

# Asynchronous
answer = await client.arun(
    afunc=my_async_function,
    scenario_id="scenario-1",
    **kwargs
)
```

### Configuration

Configure SSL verification with custom settings:

```python
from scenario_client import AOBench, SSLConfig

# Use default configuration (from environment variables)
client = AOBench(scenario_uri="https://scenarios.example.com")

# Custom SSL configuration
config = SSLConfig(ssl_verify=False)  # Disable SSL verification
client = AOBench(
    scenario_uri="https://scenarios.example.com",
    config=config
)

# Load configuration from environment
config = SSLConfig.from_env()
client = AOBench(scenario_uri="https://scenarios.example.com", config=config)

# Use custom CA certificate
config = SSLConfig(ssl_verify="/path/to/ca-bundle.crt")
client = AOBench(scenario_uri="https://scenarios.example.com", config=config)
```

### MLflow Integration

Automatically track experiments, runs, and traces:

```python
# Enable tracking when loading scenarios
scenario_set, tracking_context = client.scenario_set(
    scenario_set_id="my-set",
    tracking=True
)

# Tracking context is automatically used in run/arun
answer = client.run(
    func=my_function,
    scenario_id="scenario-1",
    run_name="My Experiment Run",  # Optional custom name
    tracking_context=tracking_context,
    **kwargs
)
```

### Post-Processing

Apply transformations to results before submission:

```python
def extract_answer(result):
    return result["output"]["answer"]

answer = client.run(
    func=my_function,
    scenario_id="scenario-1",
    post_process=extract_answer,
    **kwargs
)
```

### Deferred Grading

For long-running evaluations, use deferred grading:

```python
# Submit for grading
response = client.deferred_grading(
    scenario_set_id="my-set",
    answers=answers,
    tracking_context=tracking_context
)
grading_id = response["grading_id"]

# Check status
status = client.deferred_grading_status(grading_id)
print(status["status"])  # "pending", "processing", "completed", "failed"

# Get results when ready
if status["status"] == "completed":
    results = client.deferred_grading_result(grading_id)
    print(results)
```

## Configuration

### SSL Certificate Verification

Configure SSL verification via the `SSL_CERT_FILE` environment variable:

```bash
# Use default verification
export SSL_CERT_FILE=true

# Disable verification (not recommended for production)
export SSL_CERT_FILE=false

# Use custom CA bundle
export SSL_CERT_FILE=/path/to/ca-bundle.crt
```

Or use a `.env` file:

```
SSL_CERT_FILE=/path/to/ca-bundle.crt
```

### Environment Variables

The client automatically loads environment variables from a `.env` file in your working directory using `python-dotenv`.

## API Reference

### `AOBench`

Main client class for interacting with scenario servers.

#### `__init__(scenario_uri: str, tracking_uri: str = "")`

Initialize the client.

**Parameters:**
- `scenario_uri`: Base URL of the scenario server
- `tracking_uri`: (Optional) MLflow tracking server URL. If provided, overrides server-provided tracking URI.

#### `scenario_types() -> dict`

Retrieve available scenario types from the server.

**Returns:** Dictionary of scenario types

#### `scenario_set(scenario_set_id: str, tracking: bool) -> tuple[dict, dict | None]`

Load a scenario set with optional tracking.

**Parameters:**
- `scenario_set_id`: ID of the scenario set to load
- `tracking`: Enable MLflow tracking

**Returns:** Tuple of (scenario_set, tracking_context)

#### `run(func, scenario_id, run_name: str = "", post_process=None, tracking_context: dict | None = None, **kwargs)`

Execute a synchronous function for a scenario.

**Parameters:**
- `func`: Function to execute
- `scenario_id`: ID of the scenario
- `run_name`: (Optional) Custom name for the MLflow run
- `post_process`: (Optional) Function to transform the result
- `tracking_context`: (Optional) Tracking context from `scenario_set()`
- `**kwargs`: Arguments passed to `func`

**Returns:** Dictionary with `scenario_id` and `answer`

#### `arun(afunc, scenario_id, run_name: str = "", post_process=None, tracking_context: dict | None = None, **kwargs)`

Execute an asynchronous function for a scenario.

**Parameters:** Same as `run()` but with async function

**Returns:** Dictionary with `scenario_id` and `answer`

#### `grade(scenario_set_id: str, answers, tracking_context: dict | None) -> dict`

Submit answers for immediate grading.

**Parameters:**
- `scenario_set_id`: ID of the scenario set
- `answers`: List of answer dictionaries
- `tracking_context`: (Optional) Tracking context

**Returns:** Grading results

#### `deferred_grading(scenario_set_id: str, answers, tracking_context: dict | None) -> dict`

Submit answers for deferred grading.

**Parameters:** Same as `grade()`

**Returns:** Dictionary with `grading_id`

#### `deferred_grading_status(grading_id) -> dict`

Check the status of a deferred grading job.

**Parameters:**
- `grading_id`: ID from `deferred_grading()`

**Returns:** Status information

#### `deferred_grading_result(grading_id) -> dict`

Retrieve results of a completed grading job.

**Parameters:**
- `grading_id`: ID from `deferred_grading()`

**Returns:** Grading results

## Example Workflow

```python
import asyncio
from scenario_client.client import AOBench

async def my_ai_function(prompt: str) -> str:
    # Your AI/ML logic here
    return f"Response to: {prompt}"

async def main():
    # Initialize client
    client = AOBench(scenario_uri="https://scenarios.example.com")

    # Load scenarios with tracking
    scenario_set, tracking_context = client.scenario_set(
        scenario_set_id="qa-benchmark-v1",
        tracking=True
    )

    print(f"Running {len(scenario_set['scenarios'])} scenarios...")

    # Run all scenarios
    answers = []
    for scenario in scenario_set["scenarios"]:
        answer = await client.arun(
            afunc=my_ai_function,
            scenario_id=scenario["id"],
            tracking_context=tracking_context,
            **scenario["inputs"]
        )
        answers.append(answer)

    # Grade results
    results = client.grade(
        scenario_set_id="qa-benchmark-v1",
        answers=answers,
        tracking_context=tracking_context
    )

    print(f"Score: {results['score']}")
    print(f"Details: {results['details']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tests

```bash
uv run python -m pytest -v
```