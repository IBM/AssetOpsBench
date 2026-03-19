"""Python Sandbox MCP Server.

Provides secure, isolated execution of Python code in containers (Docker/Podman).
Offers tools for executing code strings, Python files, and simplified script interfaces
with automatic resource management and security constraints.
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
    """Get or initialize the singleton container runtime instance.
    
    Returns:
        Initialized ContainerRuntime (Docker or Podman).
    """
    global _runtime
    if _runtime is None:
        _runtime = get_container_runtime()
    return _runtime


def _ensure_container_image() -> None:
    """Verify container image exists, building if necessary.
    
    Locates the Containerfile and ensures the sandbox image is built and ready.
    """
    runtime = _get_runtime()
    dockerfile_path = Path(__file__).parent / "container" / "Containerfile"
    runtime.ensure_image(CONTAINER_FULL_IMAGE, dockerfile_path)


# Initialize FastMCP server
mcp = FastMCP("PythonSandbox")


class ErrorResult(BaseModel):
    """Error result returned when execution fails.
    
    Attributes:
        error: Error message describing what went wrong.
    """
    error: str


class ExecutionResult(BaseModel):
    """Successful execution result with output and metadata.
    
    Attributes:
        success: Whether execution completed successfully (exit code 0).
        stdout: Standard output captured from execution.
        stderr: Standard error captured from execution.
        exit_code: Process exit code.
        output_files: Optional dictionary of output files (filename -> content).
    """
    success: bool = Field(description="Whether execution was successful")
    stdout: str = Field(description="Standard output from the execution")
    stderr: str = Field(description="Standard error from the execution")
    exit_code: int = Field(description="Exit code from the execution")
    output_files: Optional[Dict[str, str]] = Field(
        default=None, description="Dictionary of output files (filename -> content)"
    )


def _create_temp_directory() -> str:
    """Create temporary directory for container execution.
    
    On macOS with Podman, uses home directory since /tmp isn't mounted in the VM.
    
    Returns:
        Path to created temporary directory.
    """
    if os.name == 'posix' and os.path.exists('/Users'):
        base_tmp = Path.home() / '.cache' / 'sandbox'
        base_tmp.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(dir=base_tmp)
    return tempfile.mkdtemp()


def _resolve_file_path(file_path: str) -> Path:
    """Resolve file path relative to workspace if needed.
    
    Args:
        file_path: File path (absolute or relative to workspace).
        
    Returns:
        Resolved absolute Path object.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If path is not a file.
    """
    source_file = Path(file_path)
    if not source_file.is_absolute():
        source_file = Path.cwd() / source_file
    
    if not source_file.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not source_file.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    return source_file


def _prepare_script(tmppath: Path, code: Optional[str], file_path: Optional[str]) -> str:
    """Prepare Python script in temporary directory.
    
    Args:
        tmppath: Temporary directory path.
        code: Python code string (if provided).
        file_path: Path to Python file (if provided).
        
    Returns:
        Name of the script file.
    """
    if code is not None:
        script_name = "script.py"
        (tmppath / script_name).write_text(code)
    else:
        source_file = _resolve_file_path(file_path)  # type: ignore
        script_name = source_file.name
        shutil.copy2(source_file, tmppath / script_name)
    return script_name


def _prepare_input_files(
    tmppath: Path,
    input_files: Optional[Dict[str, str]],
    input_file_paths: Optional[Dict[str, str]]
) -> None:
    """Prepare input files in temporary directory.
    
    Args:
        tmppath: Temporary directory path.
        input_files: Input files as filename -> content mappings.
        input_file_paths: Input files as destination -> source path mappings.
    """
    if input_files:
        for filename, content in input_files.items():
            (tmppath / filename).write_text(content)
    
    if input_file_paths:
        for dest_filename, source_path in input_file_paths.items():
            source_file = _resolve_file_path(source_path)
            shutil.copy2(source_file, tmppath / dest_filename)


def _retrieve_output_files(tmppath: Path, output_files: Optional[List[str]]) -> Dict[str, str]:
    """Retrieve output files from temporary directory.
    
    Args:
        tmppath: Temporary directory path.
        output_files: List of output filenames to retrieve.
        
    Returns:
        Dictionary mapping filenames to their content.
    """
    output_file_contents = {}
    if output_files:
        for filename in output_files:
            output_path = tmppath / filename
            if output_path.exists():
                output_file_contents[filename] = output_path.read_text()
    return output_file_contents


def _execute_in_container(
    code: Optional[str] = None,
    file_path: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    input_files: Optional[Dict[str, str]] = None,
    input_file_paths: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    output_files: Optional[List[str]] = None,
) -> ExecutionResult:
    """Execute Python code or file in an isolated container with resource limits.
    
    Core execution function that handles both code strings and file paths,
    managing temporary directories, input/output files, and container execution.

    Args:
        code: Python code string to execute (mutually exclusive with file_path).
        file_path: Path to Python file to execute (mutually exclusive with code).
        requirements: Pip packages to install before execution.
        input_files: Input files as filename -> content mappings.
        input_file_paths: Input files as destination -> source path mappings.
        timeout: Maximum execution time in seconds.
        output_files: Output filenames to retrieve after execution.

    Returns:
        ExecutionResult containing stdout, stderr, exit code, and output files.
        
    Raises:
        ValueError: If both or neither code and file_path are provided.
        FileNotFoundError: If specified file_path doesn't exist.
    """
    if code is None and file_path is None:
        raise ValueError("Either code or file_path must be provided")
    if code is not None and file_path is not None:
        raise ValueError("Cannot provide both code and file_path")
    
    runtime = _get_runtime()

    try:
        tmpdir = _create_temp_directory()
        
        try:
            tmppath = Path(tmpdir)
            script_name = _prepare_script(tmppath, code, file_path)

            if requirements:
                (tmppath / "requirements.txt").write_text("\n".join(requirements))

            _prepare_input_files(tmppath, input_files, input_file_paths)

            exit_code, stdout, stderr = runtime.run_container(
                image=CONTAINER_FULL_IMAGE,
                command=["python", f"/workspace/{script_name}"],
                volumes={str(tmppath): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                network_mode="none",
                mem_limit="512m",
                cpu_quota=50000,
                timeout=timeout,
            )

            output_file_contents = _retrieve_output_files(tmppath, output_files)

            return ExecutionResult(
                success=(exit_code == 0),
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                output_files=output_file_contents if output_file_contents else None,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error executing in container: {e}")
        raise


@mcp.tool()
def execute_python_file(
    file_path: str,
    requirements: Optional[List[str]] = None,
    input_files: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    output_files: Optional[List[str]] = None,
) -> Union[ExecutionResult, ErrorResult]:
    """Execute a Python file from workspace in a secure, isolated container.
    
    Copies the specified Python file into the container and executes it with
    optional dependencies, input files, and output file retrieval.

    Args:
        file_path: Path to Python file relative to workspace directory.
        requirements: Pip package names to install before execution.
        input_files: Input files as filename -> content mappings.
        timeout: Maximum execution time in seconds (default: 60).
        output_files: Output filenames to retrieve after execution.

    Returns:
        ExecutionResult with stdout, stderr, exit code, and output files,
        or ErrorResult if execution fails.
    """
    _ensure_container_image()

    try:
        return _execute_in_container(
            file_path=file_path,
            requirements=requirements,
            input_files=input_files,
            timeout=timeout,
            output_files=output_files,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"File error: {e}")
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.exception("Error executing Python file")
        return ErrorResult(error=str(e))


@mcp.tool()
def execute_python_code(
    code: str,
    requirements: Optional[List[str]] = None,
    input_files: Optional[Dict[str, str]] = None,
    input_file_paths: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    output_files: Optional[List[str]] = None,
) -> Union[ExecutionResult, ErrorResult]:
    """Execute arbitrary Python code in a secure, isolated container.
    
    Provides maximum isolation with no network access, limited resources,
    isolated filesystem, and restricted process capabilities.

    Security Features:
        - No network access
        - Memory limit: 512MB
        - CPU limit: 50% of one core
        - Isolated filesystem
        - Non-root execution

    Pre-installed Libraries:
        matplotlib 3.9.3, numpy 2.2.1, pandas 2.2.3, pyarrow 18.1.0,
        pydantic 2.10.5, pympler 1.1, scikit-learn 1.6.1, seaborn 0.13.2

    Args:
        code: Python code string to execute.
        requirements: Pip package names to install before execution.
        input_files: Input files as filename -> content mappings.
        input_file_paths: Input files as destination -> source path mappings from workspace.
        timeout: Maximum execution time in seconds (default: 60).
        output_files: Output filenames to retrieve after execution.

    Returns:
        ExecutionResult with stdout, stderr, exit code, and output files,
        or ErrorResult if execution fails.
    """
    try:
        _ensure_container_image()
        return _execute_in_container(
            code=code,
            requirements=requirements,
            input_files=input_files,
            input_file_paths=input_file_paths,
            timeout=timeout,
            output_files=output_files,
        )
    except Exception as e:
        logger.error(f"Error executing Python code: {e}")
        return ErrorResult(error=str(e))


@mcp.tool()
def execute_python_script(
    script_content: str,
    input_data: Optional[str] = None,
    input_file_paths: Optional[Dict[str, str]] = None,
    requirements: Optional[List[str]] = None,
    timeout: int = 60,
) -> Union[ExecutionResult, ErrorResult]:
    """Execute a Python script with simplified input/output interface.
    
    Simplified interface for scripts that read from data.json and write to output.json.
    Automatically handles JSON input data and retrieves output.json if created.

    Pre-installed Libraries:
        matplotlib 3.9.3, numpy 2.2.1, pandas 2.2.3, pyarrow 18.1.0,
        pydantic 2.10.5, pympler 1.1, scikit-learn 1.6.1, seaborn 0.13.2

    Args:
        script_content: Python script code to execute.
        input_data: JSON string saved as data.json for script input.
        input_file_paths: Input files as destination -> source path mappings from workspace.
        requirements: Pip package names to install before execution.
        timeout: Maximum execution time in seconds (default: 60).

    Returns:
        ExecutionResult with stdout, stderr, exit code, and output.json content if created,
        or ErrorResult if execution fails.
    """
    try:
        _ensure_container_image()
        
        input_files = {"data.json": input_data} if input_data else None
        
        return _execute_in_container(
            code=script_content,
            requirements=requirements,
            input_files=input_files,
            input_file_paths=input_file_paths,
            timeout=timeout,
            output_files=["output.json"],
        )
    except Exception as e:
        logger.error(f"Error executing Python script: {e}")
        return ErrorResult(error=str(e))


def main():
    """Run the MCP server.
    
    Initializes and starts the FastMCP server, making sandbox tools available
    for secure Python code execution.
    """
    logger.info("Starting Python Sandbox MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
