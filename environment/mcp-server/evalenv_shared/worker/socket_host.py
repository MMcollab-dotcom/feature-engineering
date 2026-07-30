"""Trusted parent-side client for the isolated model-worker container."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

from evalenv_shared.worker import (
    WorkerProcessError,
    WorkerProtocolError,
    WorkerTimeoutError,
)
from evalenv_shared.worker.session import WorkerSession


class UnixSocketWorkerHost:
    def __init__(self, socket_path: str | Path) -> None:
        self.socket_path = Path(socket_path)

    async def start(
        self,
        *,
        timeout_s: float,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> UnixSocketWorkerSession:
        del env
        if cwd is None:
            raise ValueError("The isolated worker requires a shared working directory.")
        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
        except OSError as exc:
            raise WorkerProcessError(
                f"Could not connect to isolated model worker: {exc}"
            ) from exc
        writer.write(
            (
                json.dumps(
                    {"cwd": cwd},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
        try:
            async with asyncio.timeout(min(float(timeout_s), 30.0)):
                await writer.drain()
                ready_line = await reader.readline()
        except TimeoutError as exc:
            writer.close()
            await writer.wait_closed()
            raise WorkerTimeoutError(
                "Isolated model worker did not acknowledge the session.",
                timeout_s=min(float(timeout_s), 30.0),
            ) from exc
        try:
            ready = json.loads(ready_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            writer.close()
            await writer.wait_closed()
            raise WorkerProtocolError(
                "Isolated model worker returned an invalid session acknowledgement."
            ) from exc
        if ready != {"ok": True}:
            writer.close()
            await writer.wait_closed()
            raise WorkerProcessError("Isolated model worker rejected the session.")
        return UnixSocketWorkerSession(
            reader=reader,
            writer=writer,
            timeout_s=float(timeout_s),
        )


class UnixSocketWorkerSession(WorkerSession):
    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        timeout_s: float,
    ) -> None:
        super().__init__(reader=reader, writer=writer, timeout_s=timeout_s)

    async def _finish_transport(self) -> None:
        await self._close_socket()

    async def _close_transport(self) -> None:
        await self._close_socket()

    async def _close_socket(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


__all__ = ["UnixSocketWorkerHost"]
