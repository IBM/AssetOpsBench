"""Docker code execution with no runtime network interface or image pulls."""

import tempfile
from pathlib import Path

import docker
from anyio import to_thread
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider


class OfflineDockerCodeExecToolProvider(DockerCodeExecToolProvider):
    """Use a prebuilt local image, mounted workspace, and Docker network=none."""

    async def __aenter__(self):
        if self._is_dockerfile:
            raise ValueError("Build the offline code image before starting an agent")
        if self._temp_base_dir:
            self._temp_base_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(
            tempfile.mkdtemp(prefix="offline_code_", dir=self._temp_base_dir)
        )
        try:
            self._client = await to_thread.run_sync(docker.from_env)
            image = await to_thread.run_sync(
                lambda: self._client.images.get(str(self._source))
            )
            self._container = await to_thread.run_sync(
                lambda: self._client.containers.run(
                    image.id,
                    command="tail -f /dev/null",
                    detach=True,
                    volumes={
                        str(self._temp_dir): {"bind": self._working_dir, "mode": "rw"}
                    },
                    working_dir=self._working_dir,
                    network_mode="none",
                    cap_drop=["NET_ADMIN", "NET_RAW"],
                    environment=None,
                    remove=False,
                )
            )
            return self.get_code_exec_tool()
        except BaseException:
            await super().__aexit__(None, None, None)
            raise
