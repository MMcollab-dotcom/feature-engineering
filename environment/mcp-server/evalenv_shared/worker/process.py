"""Subprocess-backed worker host."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping

from evalenv_shared.worker.session import WorkerSession

STDERR_TAIL_CHARS = 4000


class SubprocessWorkerHost:
    async def start(
        self,
        *,
        timeout_s: float,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SubprocessWorkerSession:
        process = await WorkerProcess.start(
            [
                sys.executable,
                "-m",
                "feature_engineering.submissions.worker",
            ],
            cwd=cwd,
            env=env,
        )
        return SubprocessWorkerSession(process=process, timeout_s=timeout_s)


class WorkerProcess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self._stderr_tail = ""
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    @classmethod
    async def start(
        cls,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> WorkerProcess:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=dict(env) if env is not None else None,
        )
        return cls(process)

    @property
    def stderr_tail(self) -> str:
        return self._stderr_tail[-STDERR_TAIL_CHARS:]

    async def terminate(self) -> None:
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
        await self._finish_stderr_task()

    async def wait_closed(self) -> None:
        await self.process.wait()
        await self._finish_stderr_task()

    async def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        while True:
            chunk = await self.process.stderr.read(1024)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            self._stderr_tail = (self._stderr_tail + text)[-STDERR_TAIL_CHARS:]

    async def _finish_stderr_task(self) -> None:
        if self._stderr_task.done():
            await self._stderr_task
            return
        self._stderr_task.cancel()
        try:
            await self._stderr_task
        except asyncio.CancelledError:
            pass


class SubprocessWorkerSession(WorkerSession):
    def __init__(self, *, process: WorkerProcess, timeout_s: float) -> None:
        assert process.process.stdout is not None
        assert process.process.stdin is not None
        self.process = process
        super().__init__(
            reader=process.process.stdout,
            writer=process.process.stdin,
            timeout_s=timeout_s,
        )

    @property
    def stderr_tail(self) -> str:
        return self.process.stderr_tail

    async def _finish_transport(self) -> None:
        await self.process.wait_closed()

    async def _close_transport(self) -> None:
        await self.process.terminate()
