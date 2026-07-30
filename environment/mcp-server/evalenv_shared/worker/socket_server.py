"""Unix-socket broker that executes submitted models in a networkless container."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import stat
import sys
from pathlib import Path
from typing import Any

MAX_BOOTSTRAP_BYTES = 64 * 1024
HEALTH_REQUEST = {"health": True}


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
            if request == HEALTH_REQUEST:
                writer.write(b'{"ok":true}\n')
                await writer.drain()
                return
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


async def _socket_is_live(socket_path: Path) -> bool:
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    try:
        writer.write(b'{"health":true}\n')
        await writer.drain()
        async with asyncio.timeout(2.0):
            return await reader.readline() == b'{"ok":true}\n'
    except (ConnectionError, OSError, TimeoutError):
        return False
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def serve() -> None:
    socket_path = Path(os.environ.get("FEATURE_WORKER_SOCKET", "/exchange/worker.sock"))
    worker_root = Path(os.environ.get("FEATURE_WORKER_ROOT", "/exchange/runtime"))
    worker_root.mkdir(parents=True, exist_ok=True)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not stat.S_ISSOCK(socket_path.stat().st_mode):
            raise RuntimeError(f"Worker socket path is not a socket: {socket_path}")
        if await _socket_is_live(socket_path):
            raise RuntimeError(f"Worker socket is already serving: {socket_path}")
        socket_path.unlink()
    broker = WorkerBroker(socket_path=socket_path, worker_root=worker_root)
    server = await asyncio.start_unix_server(broker.handle, path=socket_path)
    socket_identity = (socket_path.stat().st_dev, socket_path.stat().st_ino)
    os.chmod(socket_path, 0o660)
    try:
        async with server:
            await server.serve_forever()
    finally:
        try:
            current_identity = (
                socket_path.stat().st_dev,
                socket_path.stat().st_ino,
            )
        except FileNotFoundError:
            current_identity = None
        if current_identity == socket_identity:
            socket_path.unlink()


def socket_is_healthy() -> bool:
    socket_path = os.environ.get("FEATURE_WORKER_SOCKET", "/exchange/worker.sock")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2.0)
            client.connect(socket_path)
            client.sendall(b'{"health":true}\n')
            return client.recv(1024) == b'{"ok":true}\n'
    except OSError:
        return False


async def _serve_until_stopped() -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await serve()
    except asyncio.CancelledError:
        pass
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    if sys.argv[1:] == ["--healthcheck"]:
        raise SystemExit(0 if socket_is_healthy() else 1)
    if sys.argv[1:]:
        raise SystemExit("Usage: feature-engineering-worker-server [--healthcheck]")
    asyncio.run(_serve_until_stopped())


if __name__ == "__main__":
    main()
