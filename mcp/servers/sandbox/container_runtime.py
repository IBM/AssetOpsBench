"""Container runtime abstraction layer.

Provides a unified interface for both Docker and Podman container runtimes.
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
    """Abstract base class for container runtime implementations."""

    @abstractmethod
    def from_env(self) -> Any:
        """Create a client from environment variables."""
        pass

    @abstractmethod
    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Ensure the container image exists, build if necessary."""
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
        """Run a container and return exit code, stdout, stderr."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the client connection."""
        pass


class DockerRuntime(ContainerRuntime):
    """Docker container runtime implementation."""

    def __init__(self):
        self.client: Any = None

    def from_env(self) -> DockerRuntime:
        """Create a Docker client from environment variables."""
        import docker

        self.client = docker.from_env()
        return self

    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Ensure the Docker image exists, build if necessary."""
        from docker.errors import ImageNotFound

        try:
            self.client.images.get(image_name)
            logger.info(f"Docker image {image_name} already exists")
        except ImageNotFound:
            logger.info(f"Building Docker image {image_name}...")
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

            image, build_logs = self.client.images.build(
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
        """Run a Docker container and return exit code, stdout, stderr."""
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
        """Close the Docker client connection."""
        if self.client:
            self.client.close()


class PodmanRuntime(ContainerRuntime):
    """Podman container runtime implementation using CLI."""

    def __init__(self):
        self.podman_cmd: str = ""

    def from_env(self) -> PodmanRuntime:
        """Initialize Podman runtime."""
        # Check if podman is available
        podman_path = shutil.which("podman")
        if not podman_path:
            raise RuntimeError("Podman executable not found in PATH")
        self.podman_cmd = podman_path

        # Verify podman is working
        try:
            result = subprocess.run(
                [self.podman_cmd, "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Podman version check failed: {result.stderr}")
            logger.info(f"Using Podman: {result.stdout.split()[0]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Podman version check timed out")

        return self

    def ensure_image(self, image_name: str, dockerfile_path: Path) -> None:
        """Ensure the Podman image exists, build if necessary."""
        # Check if image exists
        result = subprocess.run(
            [self.podman_cmd, "image", "exists", image_name],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"Podman image {image_name} already exists")
            return

        logger.info(f"Building Podman image {image_name}...")
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

        # Build the image
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
            timeout=300,  # 5 minute timeout for build
        )

        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build Podman image: {build_result.stderr}")

        logger.info(f"Successfully built Podman image {image_name}")

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
        """Run a Podman container and return exit code, stdout, stderr."""
        # Build volume mounts
        volume_args = []
        for host_path, mount_info in volumes.items():
            bind_path = mount_info["bind"]
            mode = mount_info.get("mode", "rw")
            volume_args.extend(["-v", f"{host_path}:{bind_path}:{mode}"])

        # Convert CPU quota to CPU shares for Podman
        # Docker cpu_quota is in microseconds per 100ms period
        # Podman uses --cpus which is a decimal (e.g., 0.5 for 50%)
        cpu_limit = cpu_quota / 100000.0

        # Build the run command
        run_cmd = (
            [
                self.podman_cmd,
                "run",
                "--rm",  # Auto-remove container after execution
                "-w",
                working_dir,
                "--network",
                network_mode,
                "--memory",
                mem_limit,
                "--cpus",
                str(cpu_limit),
            ]
            + volume_args
            + [image]
            + command
        )

        logger.debug(f"Running Podman command: {' '.join(run_cmd)}")

        # Run the container
        try:
            result = subprocess.run(
                run_cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            # Container timed out, try to clean up
            logger.warning(f"Container execution timed out after {timeout}s")
            stdout = e.stdout.decode("utf-8") if e.stdout else ""
            stderr = e.stderr.decode("utf-8") if e.stderr else ""
            stderr += f"\n[Execution timed out after {timeout} seconds]"
            return -1, stdout, stderr

    def close(self) -> None:
        """Close the Podman runtime (no-op for CLI)."""
        pass


def detect_container_runtime() -> str:
    """Detect which container runtime is available.

    Returns:
        "docker" or "podman" based on what's available

    Raises:
        RuntimeError: If neither Docker nor Podman is available
    """
    # Check environment variable first
    runtime_env = os.environ.get("CONTAINER_RUNTIME", "").lower()
    if runtime_env in ("docker", "podman"):
        logger.info(
            f"Using container runtime from CONTAINER_RUNTIME env: {runtime_env}"
        )
        return runtime_env

    # Try Docker first
    try:
        import docker

        client = docker.from_env()
        client.ping()
        client.close()
        logger.info("Detected Docker as container runtime")
        return "docker"
    except Exception as e:
        logger.debug(f"Docker not available: {e}")

    # Try Podman
    podman_cmd = shutil.which("podman")
    if podman_cmd:
        try:
            result = subprocess.run(
                [podman_cmd, "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("Detected Podman as container runtime")
                return "podman"
        except Exception as e:
            logger.debug(f"Podman not available: {e}")

    raise RuntimeError(
        "Neither Docker nor Podman is available. "
        "Please install Docker (https://docs.docker.com/get-docker/) "
        "or Podman (https://podman.io/getting-started/installation)"
    )


def get_container_runtime() -> ContainerRuntime:
    """Get the appropriate container runtime based on what's available.

    Returns:
        ContainerRuntime instance (DockerRuntime or PodmanRuntime)
    """
    runtime_type = detect_container_runtime()

    if runtime_type == "docker":
        return DockerRuntime().from_env()
    elif runtime_type == "podman":
        return PodmanRuntime().from_env()
    else:
        raise RuntimeError(f"Unknown container runtime: {runtime_type}")
