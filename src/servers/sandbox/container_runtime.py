"""Container runtime abstraction layer.

Provides a unified interface for Docker and Podman container runtimes,
enabling secure Python code execution in isolated containers.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("sandbox-mcp-server")


class ContainerRuntime(ABC):
    """Abstract base class for container runtime implementations.
    
    Defines the interface for Docker and Podman runtime implementations,
    providing methods for image management and container execution.
    """

    @abstractmethod
    def from_env(self) -> Any:
        """Initialize runtime client from environment variables.
        
        Returns:
            Initialized runtime instance.
        """
        pass

    @abstractmethod
    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Verify container image exists, building it if necessary.
        
        Args:
            image_name: Name and tag of the container image.
            dockerfile_path: Path to the Containerfile/Dockerfile.
            
        Raises:
            FileNotFoundError: If Containerfile doesn't exist.
            RuntimeError: If image build fails.
        """
        pass

    @abstractmethod
    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        working_dir: str,
        network_mode: str,
        mem_limit: str,
        cpu_quota: int,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Execute code in an isolated container with resource limits.
        
        Args:
            image: Container image name and tag.
            command: Command to execute in the container.
            volumes: Volume mounts mapping host paths to container paths.
            working_dir: Working directory inside the container.
            network_mode: Network isolation mode (typically "none").
            mem_limit: Memory limit (e.g., "512m").
            cpu_quota: CPU quota in microseconds per 100ms period.
            timeout: Maximum execution time in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr).
            
        Raises:
            subprocess.TimeoutExpired: If execution exceeds timeout.
            RuntimeError: If container execution fails.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release runtime client resources."""
        pass


class DockerRuntime(ContainerRuntime):
    """Docker container runtime implementation using Docker Python SDK."""

    def __init__(self):
        self.client: Any = None

    def from_env(self) -> DockerRuntime:
        """Initialize Docker client from environment variables.
        
        Returns:
            Self with initialized Docker client.
        """
        import docker

        self.client = docker.from_env()
        return self

    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Verify Docker image exists, building if necessary.
        
        Args:
            image_name: Name and tag of the Docker image.
            dockerfile_path: Path to the Containerfile/Dockerfile.
            
        Raises:
            FileNotFoundError: If Containerfile doesn't exist.
        """
        from docker.errors import ImageNotFound

        try:
            self.client.images.get(image_name)
            logger.info(f"Docker image {image_name} already exists")
        except ImageNotFound:
            self._build_image(image_name, dockerfile_path)

    def _build_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Build Docker image from Containerfile.
        
        Args:
            image_name: Name and tag for the built image.
            dockerfile_path: Path to the Containerfile/Dockerfile.
            
        Raises:
            FileNotFoundError: If Containerfile doesn't exist.
        """
        logger.info(f"Building Docker image {image_name}...")
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        self.client.images.build(
            path=str(dockerfile_path.parent), tag=image_name, rm=True, forcerm=True
        )
        logger.info(f"Successfully built Docker image {image_name}")

    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        working_dir: str,
        network_mode: str,
        mem_limit: str,
        cpu_quota: int,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Execute code in a Docker container with resource limits.
        
        Args:
            image: Docker image name and tag.
            command: Command to execute in the container.
            volumes: Volume mounts mapping host paths to container paths.
            working_dir: Working directory inside the container.
            network_mode: Network isolation mode.
            mem_limit: Memory limit string.
            cpu_quota: CPU quota in microseconds.
            timeout: Maximum execution time in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        container = self.client.containers.create(
            image,
            command=command,
            volumes=volumes,
            working_dir=working_dir,
            network_mode=network_mode,
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            detach=True,
            auto_remove=False,
        )

        try:
            container.start()
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", -1)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            return exit_code, stdout, stderr
        finally:
            container.remove()

    def close(self) -> None:
        """Release Docker client resources."""
        if self.client:
            self.client.close()


class PodmanRuntime(ContainerRuntime):
    """Podman container runtime implementation using CLI commands.
    
    Uses subprocess to execute Podman CLI commands for container operations.
    Suitable for rootless container execution without requiring a daemon.
    """

    def __init__(self):
        self.podman_cmd: str = ""

    def from_env(self) -> PodmanRuntime:
        """Initialize Podman runtime by verifying CLI availability.
        
        Returns:
            Self with initialized Podman command path.
            
        Raises:
            RuntimeError: If Podman is not found or not functional.
        """
        podman_path = shutil.which("podman")
        if not podman_path:
            raise RuntimeError("Podman executable not found in PATH")
        self.podman_cmd = podman_path
        self._verify_podman()
        return self

    def _verify_podman(self) -> None:
        """Verify Podman is functional by checking version.
        
        Raises:
            RuntimeError: If Podman version check fails or times out.
        """
        try:
            result = subprocess.run(
                [self.podman_cmd, "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Podman version check failed: {result.stderr}")
            logger.info(f"Using Podman: {result.stdout.split()[0]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Podman version check timed out")

    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Verify Podman image exists, building if necessary.
        
        Args:
            image_name: Name and tag of the Podman image.
            dockerfile_path: Path to the Containerfile/Dockerfile.
            
        Raises:
            FileNotFoundError: If Containerfile doesn't exist.
            RuntimeError: If image build fails.
        """
        if self._image_exists(image_name):
            logger.info(f"Podman image {image_name} already exists")
            return
        self._build_image(image_name, dockerfile_path)

    def _image_exists(self, image_name: str) -> bool:
        """Check if Podman image exists.
        
        Args:
            image_name: Name and tag of the image to check.
            
        Returns:
            True if image exists, False otherwise.
        """
        result = subprocess.run(
            [self.podman_cmd, "image", "exists", image_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _build_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Build Podman image from Containerfile.
        
        Args:
            image_name: Name and tag for the built image.
            dockerfile_path: Path to the Containerfile/Dockerfile.
            
        Raises:
            FileNotFoundError: If Containerfile doesn't exist.
            RuntimeError: If build fails.
        """
        logger.info(f"Building Podman image {image_name}...")
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        build_result = subprocess.run(
            [
                self.podman_cmd,
                "build",
                "-t",
                image_name,
                "-f",
                str(dockerfile_path),
                str(dockerfile_path.parent),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build Podman image: {build_result.stderr}")

        logger.info(f"Successfully built Podman image {image_name}")

    def _build_volume_args(self, volumes: Dict[str, Dict[str, str]]) -> List[str]:
        """Build volume mount arguments for Podman CLI.
        
        Args:
            volumes: Volume mounts mapping host paths to container paths.
            
        Returns:
            List of volume arguments for Podman CLI.
        """
        volume_args = []
        for host_path, mount_info in volumes.items():
            bind_path = mount_info["bind"]
            mode = mount_info.get("mode", "rw")
            volume_args.extend(["-v", f"{host_path}:{bind_path}:{mode}"])
        return volume_args

    def run_container(
        self,
        image: str,
        command: List[str],
        volumes: Dict[str, Dict[str, str]],
        working_dir: str,
        network_mode: str,
        mem_limit: str,
        cpu_quota: int,
        timeout: int,
    ) -> Tuple[int, str, str]:
        """Execute code in a Podman container with resource limits.
        
        Args:
            image: Podman image name and tag.
            command: Command to execute in the container.
            volumes: Volume mounts mapping host paths to container paths.
            working_dir: Working directory inside the container.
            network_mode: Network isolation mode.
            mem_limit: Memory limit string.
            cpu_quota: CPU quota in microseconds (converted to --cpus decimal).
            timeout: Maximum execution time in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr).
            
        Raises:
            subprocess.TimeoutExpired: If execution exceeds timeout.
        """
        volume_args = self._build_volume_args(volumes)
        cpu_limit = cpu_quota / 100000.0

        run_cmd = [
            self.podman_cmd,
            "run",
            "--rm",
            "-w", working_dir,
            "--network", network_mode,
            "--memory", mem_limit,
            "--cpus", str(cpu_limit),
        ] + volume_args + [image] + command

        logger.debug(f"Running Podman command: {' '.join(run_cmd)}")

        try:
            result = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Container execution timed out after {timeout}s")
            stdout = e.stdout.decode("utf-8") if e.stdout else ""
            stderr = e.stderr.decode("utf-8") if e.stderr else ""
            stderr += f"\n[Execution timed out after {timeout} seconds]"
            return -1, stdout, stderr

    def close(self) -> None:
        """Release Podman runtime resources (no-op for CLI-based runtime)."""
        pass


def detect_container_runtime() -> str:
    """Detect available container runtime (Docker or Podman).
    
    Checks CONTAINER_RUNTIME environment variable first, then auto-detects
    by attempting to connect to Docker, falling back to Podman if unavailable.

    Returns:
        Runtime identifier: "docker" or "podman".

    Raises:
        RuntimeError: If neither Docker nor Podman is available.
    """
    runtime_env = os.environ.get("CONTAINER_RUNTIME", "").lower()
    if runtime_env in ("docker", "podman"):
        logger.info(f"Using container runtime from CONTAINER_RUNTIME env: {runtime_env}")
        return runtime_env

    if _is_docker_available():
        logger.info("Detected Docker as container runtime")
        return "docker"

    if _is_podman_available():
        logger.info("Detected Podman as container runtime")
        return "podman"

    raise RuntimeError(
        "Neither Docker nor Podman is available. "
        "Please install Docker (https://docs.docker.com/get-docker/) "
        "or Podman (https://podman.io/getting-started/installation)"
    )


def _is_docker_available() -> bool:
    """Check if Docker is available and functional.
    
    Returns:
        True if Docker is available, False otherwise.
    """
    try:
        import docker
        client = docker.from_env()
        client.ping()
        client.close()
        return True
    except Exception as e:
        logger.debug(f"Docker not available: {e}")
        return False


def _is_podman_available() -> bool:
    """Check if Podman is available and functional.
    
    Returns:
        True if Podman is available, False otherwise.
    """
    podman_cmd = shutil.which("podman")
    if not podman_cmd:
        return False
    
    try:
        result = subprocess.run(
            [podman_cmd, "version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Podman not available: {e}")
        return False


def get_container_runtime() -> ContainerRuntime:
    """Initialize and return the appropriate container runtime.
    
    Detects available runtime (Docker or Podman) and returns an initialized instance.

    Returns:
        Initialized ContainerRuntime instance (DockerRuntime or PodmanRuntime).
        
    Raises:
        RuntimeError: If no container runtime is available or unknown runtime detected.
    """
    runtime_type = detect_container_runtime()

    if runtime_type == "docker":
        return DockerRuntime().from_env()
    elif runtime_type == "podman":
        return PodmanRuntime().from_env()
    else:
        raise RuntimeError(f"Unknown container runtime: {runtime_type}")
