"""Python Sandbox MCP Server.

Provides secure execution of arbitrary Python code in isolated containers.
Supports both Docker and Podman for maximum isolation from the host operating system.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from servers.sandbox.container_runtime import (ContainerRuntime,
                                               get_container_runtime)

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("sandbox-mcp-server")

# Container configuration
CONTAINER_IMAGE_NAME = "python-sandbox"
CONTAINER_IMAGE_TAG = "latest"
CONTAINER_FULL_IMAGE = f"{CONTAINER_IMAGE_NAME}:{CONTAINER_IMAGE_TAG}"

# Global container runtime instance
_runtime: Optional[ContainerRuntime] = None


def _get_runtime() -> ContainerRuntime:
    """Get or initialize the container runtime."""
    global _runtime
    if _runtime is None:
        _runtime = get_container_runtime()
    return _runtime


def _ensure_container_image() -> None:
    """Ensure the container image exists, build if necessary."""
    runtime = _get_runtime()
    dockerfile_path = Path(__file__).parent / "container" / "Containerfile"
    runtime.ensure_image(CONTAINER_FULL_IMAGE, dockerfile_path)


# Initialize FastMCP server
mcp = FastMCP("PythonSandbox")


class ErrorResult(BaseModel):
    """Error result model."""

    error: str


class ExecutionResult(BaseModel):
    """Successful execution result."""

    success: bool = Field(description="Whether execution was successful")
    stdout: str = Field(description="Standard output from the execution")
    stderr: str = Field(description="Standard error from the execution")
    exit_code: int = Field(description="Exit code from the execution")
    output_files: Optional[Dict[str, str]] = Field(
        default=None, description="Dictionary of output files (filename -> content)"
    )


def _execute_in_container(
    code: str,
    requirements: Optional[List[str]] = None,
    input_files: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    output_files: Optional[List[str]] = None,
) -> ExecutionResult:
    """Execute Python code in an isolated container.

    Args:
        code: Python code to execute
        requirements: Optional list of pip packages to install
        input_files: Optional dict of filename -> content for input files
        timeout: Execution timeout in seconds
        output_files: Optional list of output filenames to retrieve

    Returns:
        ExecutionResult with execution details
    """
    runtime = _get_runtime()

    try:
        # Create temporary directory for files
        # On macOS with Podman, use home directory since /tmp isn't mounted in the VM
        if os.name == 'posix' and os.path.exists('/Users'):
            # macOS - use home directory
            base_tmp = Path.home() / '.cache' / 'sandbox'
            base_tmp.mkdir(parents=True, exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir=base_tmp)
        else:
            tmpdir = tempfile.mkdtemp()
        
        try:
            tmppath = Path(tmpdir)

            # Write main Python script
            script_path = tmppath / "script.py"
            script_path.write_text(code)

            # Write requirements if provided
            if requirements:
                req_path = tmppath / "requirements.txt"
                req_path.write_text("\n".join(requirements))

            # Write input files if provided
            if input_files:
                for filename, content in input_files.items():
                    file_path = tmppath / filename
                    file_path.write_text(content)

            # Run container
            exit_code, stdout, stderr = runtime.run_container(
                image=CONTAINER_FULL_IMAGE,
                command=["python", "/workspace/script.py"],
                volumes={str(tmppath): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                network_mode="none",  # No network access for security
                mem_limit="512m",  # Memory limit
                cpu_quota=50000,  # CPU limit (50% of one core)
                timeout=timeout,
            )

            # Retrieve output files if specified
            output_file_contents = {}
            if output_files:
                for filename in output_files:
                    output_path = tmppath / filename
                    if output_path.exists():
                        output_file_contents[filename] = output_path.read_text()

            return ExecutionResult(
                success=(exit_code == 0),
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                output_files=output_file_contents if output_file_contents else None,
            )
        finally:
            # Clean up temporary directory
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error executing code in container: {e}")
        raise


@mcp.tool()
def execute_python_code(
    code: str,
    requirements: Optional[List[str]] = None,
    input_files: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    output_files: Optional[List[str]] = None,
) -> Union[ExecutionResult, ErrorResult]:
    """Execute arbitrary Python code in a secure, isolated container.

    This tool provides maximum isolation from the host operating system by running
    code in a container (Docker or Podman) with:
    - No network access
    - Limited CPU and memory resources
    - Isolated filesystem
    - Restricted process capabilities

    Args:
        code: Python code to execute (as a string)
        requirements: Optional list of pip package names to install before execution
        input_files: Optional dictionary mapping filenames to their content (as strings)
        timeout: Maximum execution time in seconds (default: 60)
        output_files: Optional list of output filenames to retrieve after execution

    Returns:
        ExecutionResult containing stdout, stderr, exit code, and any output files

    Example:
        ```python
        result = execute_python_code(
            code="print('Hello, World!')",
            requirements=["pandas"],
            timeout=30
        )
        ```
    """
    try:
        # Ensure container image exists
        _ensure_container_image()

        # Execute code
        result = _execute_in_container(
            code=code,
            requirements=requirements,
            input_files=input_files,
            timeout=timeout,
            output_files=output_files,
        )

        return result

    except Exception as e:
        logger.error(f"Error in execute_python_code: {e}")
        return ErrorResult(error=str(e))


@mcp.tool()
def execute_python_script(
    script_content: str,
    input_data: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    timeout: int = 60,
) -> Union[ExecutionResult, ErrorResult]:
    """Execute a Python script with optional input data in a secure sandbox.

    This is a simplified interface for executing Python scripts that read from
    stdin or a data.json file and write results to output.json.

    Args:
        script_content: The Python script code to execute
        input_data: Optional JSON string to provide as input (saved as data.json)
        requirements: Optional list of pip packages to install
        timeout: Maximum execution time in seconds (default: 60)

    Returns:
        ExecutionResult with execution details and output.json content if created

    Example:
        ```python
        script = '''
        import json
        with open('data.json') as f:
            data = json.load(f)
        result = {'processed': len(data)}
        with open('output.json', 'w') as f:
            json.dump(result, f)
        '''
        result = execute_python_script(
            script_content=script,
            input_data='{"items": [1, 2, 3]}'
        )
        ```
    """
    try:
        # Prepare input files
        input_files = {}
        if input_data:
            input_files["data.json"] = input_data

        # Ensure container image exists
        _ensure_container_image()

        # Execute script
        result = _execute_in_container(
            code=script_content,
            requirements=requirements,
            input_files=input_files if input_files else None,
            timeout=timeout,
            output_files=["output.json"],
        )

        return result

    except Exception as e:
        logger.error(f"Error in execute_python_script: {e}")
        return ErrorResult(error=str(e))


def main():
    """Run the MCP server."""
    logger.info("Starting MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
