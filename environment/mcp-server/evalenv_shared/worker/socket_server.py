"""Unix-socket broker that executes submitted models in a networkless container."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

MAX_BOOTSTRAP_BYTES = 64 * 1024


class WorkerBroker:
    def __init__(self, *, socket_path: Path, worker_root: Path) -> None:
        self.socket_path = socket_path
        self.worker_root = worker_root.resolve(strict=True)

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        process: asyncio.subprocess.Process | None = None
        try:
            bootstrap = await reader.readline()
            if not bootstrap or len(bootstrap) > MAX_BOOTSTRAP_BYTES:
                raise ValueError("Invalid worker bootstrap message.")
            request = json.loads(bootstrap)
            cwd = self._validate_bootstrap(request)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "feature_engineering.submissions.worker",
                cwd=str(cwd),
                env=self._worker_env(cwd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            writer.write(b'{"ok":true}\n')
            await writer.drain()
            await asyncio.gather(
                self._client_to_child(reader, process),
                self._child_to_client(process, writer),
                self._drain_stderr(process),
            )
        except (
            ConnectionError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            if process is None:
                writer.write(
                    (
                        json.dumps(
                            {"ok": False, "error": type(exc).__name__},
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                try:
                    await writer.drain()
                except ConnectionError:
                    pass
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _validate_bootstrap(self, request: Any) -> Path:
        if not isinstance(request, dict):
            raise TypeError("Worker bootstrap must be an object.")
        if set(request) != {"cwd"}:
            raise ValueError("Worker bootstrap contains unsupported fields.")
        cwd = Path(str(request.get("cwd") or "")).resolve(strict=True)
        if cwd != self.worker_root and self.worker_root not in cwd.parents:
            raise ValueError("Worker cwd is outside the shared exchange root.")
        return cwd

    @staticmethod
    def _worker_env(cwd: Path) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "TMPDIR": str(cwd),
            "TEMP": str(cwd),
            "TMP": str(cwd),
        }

    @staticmethod
    async def _client_to_child(
        reader: asyncio.StreamReader,
        process: asyncio.subprocess.Process,
    ) -> None:
        assert process.stdin is not None
        while line := await reader.readline():
            process.stdin.write(line)
            await process.stdin.drain()
        process.stdin.close()

    @staticmethod
    async def _child_to_client(
        process: asyncio.subprocess.Process,
        writer: asyncio.StreamWriter,
    ) -> None:
        assert process.stdout is not None
        while line := await process.stdout.readline():
            writer.write(line)
            await writer.drain()

    @staticmethod
    async def _drain_stderr(process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while chunk := await process.stderr.read(4096):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()


async def serve() -> None:
    socket_path = Path(os.environ.get("FEATURE_WORKER_SOCKET", "/exchange/worker.sock"))
    worker_root = Path(os.environ.get("FEATURE_WORKER_ROOT", "/exchange/runtime"))
    worker_root.mkdir(parents=True, exist_ok=True)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        raise RuntimeError(f"Worker socket already exists: {socket_path}")
    broker = WorkerBroker(socket_path=socket_path, worker_root=worker_root)
    server = await asyncio.start_unix_server(broker.handle, path=socket_path)
    os.chmod(socket_path, 0o660)
    async with server:
        await server.serve_forever()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
