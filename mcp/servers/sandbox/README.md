# Python Sandbox MCP Server

A secure Model Context Protocol (MCP) server that provides isolated execution of arbitrary Python code using container technology (Docker or Podman).

## Features

- **Maximum Isolation**: Executes code in Docker containers with no network access
- **Resource Limits**: CPU and memory constraints to prevent resource exhaustion
- **Secure by Default**: Runs as non-root user with minimal privileges
- **Flexible Input/Output**: Support for input files and output file retrieval
- **Package Installation**: Install pip packages on-demand within the sandbox
- **Timeout Protection**: Configurable execution timeouts

## Installation

### Prerequisites

You need either **Docker** or **Podman** installed on your system:

- **Docker**: [Installation Guide](https://docs.docker.com/get-docker/)
- **Podman**: [Installation Guide](https://podman.io/getting-started/installation)

The server will automatically detect which container runtime is available and use it.

### Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

1. The container image will be built automatically on first use, or you can build it manually:

**With Docker:**

```bash
docker build -t python-sandbox:latest .
```

**With Podman:**

```bash
podman build -t python-sandbox:latest .
```

## Usage

### As an MCP Server

Run the server:

```bash
python main.py
```

The server exposes two tools:

#### 1. `execute_python_code`

Execute arbitrary Python code with full control over the environment.

**Parameters:**

- `code` (str, required): Python code to execute
- `requirements` (list[str], optional): Pip packages to install
- `input_files` (dict[str, str], optional): Input files as filename -> content mapping
- `timeout` (int, optional): Execution timeout in seconds (default: 60)
- `output_files` (list[str], optional): Output filenames to retrieve

**Example:**

```python
{
    "code": "import pandas as pd\ndf = pd.DataFrame({'a': [1,2,3]})\nprint(df)",
    "requirements": ["pandas"],
    "timeout": 30
}
```

#### 2. `execute_python_script`

Simplified interface for scripts that read from `data.json` and write to `output.json`.

**Parameters:**

- `script_content` (str, required): Python script code
- `input_data` (str, optional): JSON string saved as data.json
- `requirements` (list[str], optional): Pip packages to install
- `timeout` (int, optional): Execution timeout in seconds (default: 60)

**Example:**

```python
{
    "script_content": "import json\nwith open('data.json') as f:\n    data = json.load(f)\nresult = {'count': len(data)}\nwith open('output.json', 'w') as f:\n    json.dump(result, f)",
    "input_data": "{\"items\": [1, 2, 3]}"
}
```

## Container Runtime

The server supports both Docker and Podman as container runtimes:

### Automatic Detection

By default, the server automatically detects which container runtime is available:

1. First checks for Docker
2. Falls back to Podman if Docker is not available
3. Raises an error if neither is found

### Manual Selection

You can explicitly specify which runtime to use via the `CONTAINER_RUNTIME` environment variable:

```bash
# Use Docker
export CONTAINER_RUNTIME=docker

# Use Podman
export CONTAINER_RUNTIME=podman
```

Or in your `.env` file:

```
CONTAINER_RUNTIME=podman
```

### Runtime Differences

Both runtimes provide equivalent security and isolation. Key differences:

- **Docker**: Uses the Docker Python SDK for container management
- **Podman**: Uses CLI commands (no daemon required, rootless by default)

## Security Features

1. **Network Isolation**: Containers run with no network access
2. **Resource Limits**:
   - Memory: 512MB
   - CPU: 50% of one core
3. **Non-root User**: Code runs as user `sandbox` (UID 1000)
4. **Filesystem Isolation**: Only mounted workspace directory is accessible
5. **Timeout Protection**: Execution is terminated after timeout expires

## Configuration

Environment variables (see `.env.example`):

- `LOG_LEVEL`: Logging level (default: WARNING)
- `CONTAINER_RUNTIME`: Force specific runtime - "docker" or "podman" (default: auto-detect)
- `DOCKER_HOST`: Docker daemon socket (default: unix:///var/run/docker.sock)

## Architecture

The sandbox uses container technology (Docker or Podman) for maximum isolation:

```
┌─────────────────────────────────────┐
│         MCP Server (Host)           │
│  ┌───────────────────────────────┐  │
│  │   FastMCP Server              │  │
│  │   - execute_python_code       │  │
│  │   - execute_python_script     │  │
│  └───────────┬───────────────────┘  │
│              │                       │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │   Container Runtime           │  │
│  │   (Docker or Podman)          │  │
│  └───────────┬───────────────────┘  │
└──────────────┼───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│     Container (Isolated)             │
│  ┌────────────────────────────────┐  │
│  │  Python 3.11 Runtime           │  │
│  │  - No network access           │  │
│  │  - Limited CPU/Memory          │  │
│  │  - Non-root user               │  │
│  │  - Temporary filesystem        │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```
