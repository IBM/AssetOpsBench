"""Borrow one execution environment across sequential actor/critic sessions."""

from contextlib import asynccontextmanager

from stirrup.tools.code_backends.base import CodeExecToolProvider


class BorrowedWorkspace(CodeExecToolProvider):
    """Expose an already-open provider without letting a session close it."""

    def __init__(self, provider, tool):
        super().__init__()
        self.provider = provider
        self.tool = tool

    @property
    def temp_dir(self):
        return self.provider.temp_dir

    async def __aenter__(self):
        return self.tool

    async def __aexit__(self, *args):
        return None

    async def run_command(self, cmd, *, timeout=None):
        return await self.provider.run_command(cmd, timeout=timeout)

    async def read_file_bytes(self, path):
        return await self.provider.read_file_bytes(path)

    async def write_file_bytes(self, path, content):
        return await self.provider.write_file_bytes(path, content)

    async def file_exists(self, path):
        return await self.provider.file_exists(path)

    async def is_directory(self, path):
        return await self.provider.is_directory(path)

    async def list_files(self, path):
        return await self.provider.list_files(path)

    async def view_image(self, path):
        return await self.provider.view_image(path)

    def output_source_roots(self):
        return self.provider.output_source_roots()

    async def resolve_output_source(self, source_path):
        return await self.provider.resolve_output_source(source_path)

    async def output_destination_identity(self, destination, output_root):
        return await self.provider.output_destination_identity(destination, output_root)


@asynccontextmanager
async def shared_workspace(provider):
    """Only the outer loop owns container startup, preservation and cleanup."""
    async with provider as tool:
        yield BorrowedWorkspace(provider, tool)
