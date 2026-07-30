"""Transport-independent parent-side worker session."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from evalenv_shared.worker import (
    WorkerProcessError,
    WorkerProtocolError,
    WorkerRemoteError,
    WorkerTimeoutError,
)
from evalenv_shared.worker.protocol import decode_message, encode_message


class WorkerSession(ABC):
    def __init__(
        self,
        *,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        timeout_s: float,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.timeout_s = float(timeout_s)
        self._next_request_id = 1
        self._request_lock = asyncio.Lock()
        self._closed = False

    @property
    def stderr_tail(self) -> str:
        return ""

    async def request(self, message: Mapping[str, Any]) -> Any:
        if self._closed:
            raise WorkerProcessError(
                "Worker session is closed.",
                stderr_tail=self.stderr_tail,
            )
        if self._request_lock.locked():
            raise WorkerProtocolError(
                "Worker session already has an active request.",
                stderr_tail=self.stderr_tail,
            )

        async with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            deadline = asyncio.get_running_loop().time() + self.timeout_s
            try:
                self.writer.write(
                    encode_message({"id": request_id, **dict(message)}).encode("utf-8")
                )
                async with asyncio.timeout_at(deadline):
                    await self.writer.drain()
                    line = await self.reader.readline()
                if line == b"":
                    raise WorkerProcessError(
                        "Worker exited unexpectedly.",
                        stderr_tail=self.stderr_tail,
                    )
                try:
                    response = decode_message(line.decode("utf-8"))
                except Exception as exc:
                    raise WorkerProtocolError(
                        "Worker emitted invalid JSON.",
                        stderr_tail=self.stderr_tail,
                    ) from exc
                if response.get("type") != "result":
                    raise WorkerProtocolError(
                        "Worker returned an unknown protocol message: "
                        f"{response.get('type')!r}.",
                        stderr_tail=self.stderr_tail,
                    )
                return self._handle_result(response, request_id)
            except TimeoutError as exc:
                await self.close()
                raise WorkerTimeoutError(
                    "Worker request exceeded timeout_s.",
                    timeout_s=self.timeout_s,
                    stderr_tail=self.stderr_tail,
                ) from exc
            except WorkerRemoteError:
                raise
            except BaseException:
                await self.close()
                raise

    async def shutdown(self) -> None:
        if self._closed:
            return
        await self.request({"type": "shutdown"})
        await self._finish_transport()
        self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        await self._close_transport()
        self._closed = True

    def _handle_result(self, response: Mapping[str, Any], request_id: int) -> Any:
        if response.get("id") != request_id:
            raise WorkerProtocolError(
                "Worker response id did not match the active request: "
                f"expected {request_id}, got {response.get('id')}.",
                stderr_tail=self.stderr_tail,
            )
        if response.get("ok"):
            return response.get("value")

        error = response.get("error")
        if isinstance(error, Mapping):
            raise WorkerRemoteError(error, stderr_tail=self.stderr_tail)
        raise WorkerProtocolError(
            f"Worker returned a malformed error: {response!r}.",
            stderr_tail=self.stderr_tail,
        )

    @abstractmethod
    async def _finish_transport(self) -> None:
        """Wait for a worker that accepted the shutdown request."""

    @abstractmethod
    async def _close_transport(self) -> None:
        """Stop or disconnect from the worker immediately."""


__all__ = ["WorkerSession"]
