#!/usr/bin/env python3
"""Build and exercise the repository's native-platform Docker contracts."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEST_COMPOSE = ROOT / "tests" / "docker-compose.yaml"
ENVIRONMENT_COMPOSE = ROOT / "environment" / "docker-compose.yaml"
PROJECT_SUFFIX = f"{os.getpid()}-{secrets.token_hex(3)}"
ENVIRONMENT_PROJECT = f"feature-engineering-env-test-{PROJECT_SUFFIX}"
VERIFIER_PROJECT = f"feature-engineering-verifier-test-{PROJECT_SUFFIX}"


def run(
    *args: str,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(args)}", flush=True)
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def output(*args: str) -> str:
    return run(*args, capture=True).stdout.strip()


def compose(
    project: str, compose_file: Path, *args: str, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    return run(
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(compose_file),
        *args,
        **kwargs,
    )


def compose_output(project: str, compose_file: Path, *args: str) -> str:
    return output(
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(compose_file),
        *args,
    )


def inspect(object_id: str) -> dict[str, Any]:
    return json.loads(output("docker", "inspect", object_id))[0]


def image_id(project: str, service: str) -> str:
    image = f"{project}-{service}:latest"
    identifier = output("docker", "image", "inspect", "--format", "{{.Id}}", image)
    if not identifier:
        raise AssertionError(f"Compose did not produce an image for {service!r}.")
    return identifier


def container_id(service: str) -> str:
    identifier = compose_output(
        VERIFIER_PROJECT,
        TEST_COMPOSE,
        "ps",
        "--all",
        "--quiet",
        service,
    )
    if not identifier:
        raise AssertionError(f"Compose did not create the {service!r} container.")
    return identifier


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def normalize_capabilities(capabilities: list[str]) -> list[str]:
    return sorted(capability.removeprefix("CAP_") for capability in capabilities)


def assert_compose_contract() -> None:
    print("\n== Compose security and dependency contract ==", flush=True)
    config = json.loads(
        compose_output(VERIFIER_PROJECT, TEST_COMPOSE, "config", "--format", "json")
    )
    services = config["services"]
    environment_config = json.loads(
        compose_output(
            ENVIRONMENT_PROJECT,
            ENVIRONMENT_COMPOSE,
            "config",
            "--format",
            "json",
        )
    )
    environment_services = environment_config["services"]
    agent = environment_services["main"]
    mcp_server = environment_services["mcp-server"]
    assert_equal(
        agent["depends_on"]["mcp-server"]["condition"],
        "service_healthy",
        "agent MCP dependency condition",
    )
    assert_equal(
        mcp_server["healthcheck"]["test"],
        [
            "CMD",
            "python",
            "-c",
            (
                "import socket; "
                "s=socket.create_connection(('localhost',8000),timeout=2); "
                "s.close()"
            ),
        ],
        "MCP readiness probe",
    )
    assert_equal(
        mcp_server["healthcheck"]["start_period"],
        "30m0s",
        "MCP initialization allowance",
    )
    main = services["main"]
    worker = services["model-worker"]
    initializer = services["worker-volume-init"]

    assert_equal(main["user"], "10001:10001", "verifier user")
    assert_equal(
        main["depends_on"]["model-worker"]["condition"],
        "service_healthy",
        "verifier dependency condition",
    )

    assert_equal(worker["user"], "10001:10001", "worker user")
    assert_equal(worker["network_mode"], "none", "worker network mode")
    assert_equal(worker["read_only"], True, "worker read-only root")
    assert_equal(worker["cap_drop"], ["ALL"], "worker dropped capabilities")
    assert "no-new-privileges:true" in worker["security_opt"]
    assert_equal(worker["pids_limit"], 64, "worker PID limit")
    assert_equal(
        worker["depends_on"]["worker-volume-init"]["condition"],
        "service_completed_successfully",
        "worker initializer dependency condition",
    )

    assert_equal(initializer["user"], "0:0", "initializer user")
    assert_equal(initializer["network_mode"], "none", "initializer network mode")
    assert_equal(initializer["read_only"], True, "initializer read-only root")
    assert_equal(initializer["cap_drop"], ["ALL"], "initializer dropped capabilities")
    assert_equal(
        sorted(initializer["cap_add"]),
        ["CHOWN", "DAC_OVERRIDE", "FOWNER"],
        "initializer capabilities",
    )
    assert "no-new-privileges:true" in initializer["security_opt"]
    assert_equal(initializer["pids_limit"], 32, "initializer PID limit")


def assert_python_image_contract(image: str, script: str, *, user: str = "0:0") -> None:
    run(
        "docker",
        "run",
        "--rm",
        "--user",
        user,
        "--entrypoint",
        "python",
        image,
        "-c",
        script,
    )


def assert_image_contracts() -> None:
    print("\n== Built-image contracts ==", flush=True)
    agent = image_id(ENVIRONMENT_PROJECT, "main")
    tools = image_id(ENVIRONMENT_PROJECT, "mcp-server")
    verifier = image_id(VERIFIER_PROJECT, "main")
    worker = image_id(VERIFIER_PROJECT, "model-worker")

    assert_equal(inspect(worker)["Config"]["User"], "10001:10001", "worker image user")

    no_uv = """
from pathlib import Path
assert not Path('/usr/local/bin/uv').exists()
assert not Path('/root/.cache/uv').exists()
"""
    assert_python_image_contract(
        agent,
        no_uv
        + """
import stat
path = Path('/app/data/public_train.parquet')
assert path.is_file()
assert stat.S_IMODE(path.stat().st_mode) == 0o444
assert not Path('/app/tests/hidden_data').exists()
""",
    )
    assert_python_image_contract(
        tools,
        no_uv
        + """
import stat
path = Path('/app/data/runtime_public.parquet')
metadata = path.stat()
assert path.is_file()
assert (metadata.st_uid, metadata.st_gid) == (10001, 10001)
assert stat.S_IMODE(metadata.st_mode) == 0o600
assert not Path('/app/tests/hidden_data').exists()
assert not Path('/tests/test.sh').exists()
""",
    )
    assert_python_image_contract(
        worker,
        no_uv
        + """
assert not Path('/app/data/runtime_public.parquet').exists()
assert not Path('/app/tests/hidden_data').exists()
assert not Path('/tests/test.sh').exists()
""",
    )
    assert_python_image_contract(
        worker,
        "import joblib, numpy, pandas, sklearn",
        user="10001:10001",
    )
    assert_python_image_contract(
        verifier,
        no_uv
        + """
import stat
public = Path('/app/data/runtime_public.parquet').stat()
assert (public.st_uid, public.st_gid) == (10001, 10001)
assert stat.S_IMODE(public.st_mode) == 0o600
assert Path('/app/tests/hidden_data/hidden.parquet').is_file()
assert Path('/app/tests/hidden_data/hidden.parquet').stat().st_size > 1024
assert Path('/tests/test.sh').is_file()
for directory in ('/app/runtime', '/app/submission', '/logs/verifier'):
    metadata = Path(directory).stat()
    assert (metadata.st_uid, metadata.st_gid) == (10001, 10001)
""",
    )


def wait_for_worker_health(timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    worker = container_id("model-worker")
    while time.monotonic() < deadline:
        state = inspect(worker)["State"]
        if state.get("Health", {}).get("Status") == "healthy":
            return
        if not state.get("Running"):
            raise AssertionError(f"Worker stopped before becoming healthy: {state}")
        time.sleep(1.0)
    raise AssertionError("Worker did not become healthy before the deadline.")


def assert_runtime_security() -> None:
    print("\n== Runtime security contract ==", flush=True)
    main = inspect(container_id("main"))
    worker = inspect(container_id("model-worker"))
    initializer = inspect(container_id("worker-volume-init"))

    assert_equal(initializer["State"]["Status"], "exited", "initializer state")
    assert_equal(initializer["State"]["ExitCode"], 0, "initializer exit code")
    assert_equal(main["State"]["Running"], True, "verifier running state")
    assert_equal(worker["State"]["Health"]["Status"], "healthy", "worker health")

    worker_host = worker["HostConfig"]
    assert_equal(worker_host["ReadonlyRootfs"], True, "runtime worker read-only root")
    assert_equal(worker_host["NetworkMode"], "none", "runtime worker network")
    assert_equal(worker_host["CapDrop"], ["ALL"], "runtime worker capabilities")
    assert "no-new-privileges:true" in worker_host["SecurityOpt"]
    assert_equal(worker_host["PidsLimit"], 64, "runtime worker PID limit")

    initializer_host = initializer["HostConfig"]
    assert_equal(
        initializer_host["ReadonlyRootfs"], True, "runtime initializer read-only root"
    )
    assert_equal(initializer_host["NetworkMode"], "none", "runtime initializer network")
    assert_equal(
        initializer_host["CapDrop"], ["ALL"], "runtime initializer dropped capabilities"
    )
    assert_equal(
        normalize_capabilities(initializer_host["CapAdd"]),
        ["CHOWN", "DAC_OVERRIDE", "FOWNER"],
        "runtime initializer capabilities",
    )


def exercise_worker_session(label: str) -> None:
    print(f"\n== Worker session: {label} ==", flush=True)
    script = """
import asyncio
import os
import stat
import tempfile
from pathlib import Path
from evalenv_shared.worker.socket_host import UnixSocketWorkerHost

assert (os.getuid(), os.getgid()) == (10001, 10001)
exchange = Path('/exchange')
runtime = exchange / 'runtime'
socket_path = exchange / 'worker.sock'
assert (exchange.stat().st_uid, exchange.stat().st_gid) == (10001, 10001)
assert stat.S_IMODE(exchange.stat().st_mode) == 0o770
assert (runtime.stat().st_uid, runtime.stat().st_gid) == (10001, 10001)
assert (socket_path.stat().st_uid, socket_path.stat().st_gid) == (10001, 10001)
assert stat.S_IMODE(socket_path.stat().st_mode) == 0o660

async def main():
    workdir = Path(tempfile.mkdtemp(dir=runtime))
    metadata = workdir.stat()
    assert (metadata.st_uid, metadata.st_gid) == (10001, 10001)
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    session = await UnixSocketWorkerHost(socket_path).start(
        timeout_s=10,
        cwd=str(workdir),
    )
    await session.shutdown()

asyncio.run(main())
print('worker-session-ok')
"""
    compose(
        VERIFIER_PROJECT,
        TEST_COMPOSE,
        "exec",
        "--no-TTY",
        "main",
        "python",
        "-c",
        script,
    )


def assert_compose_lifecycle() -> None:
    print("\n== Compose lifecycle ==", flush=True)
    compose(
        VERIFIER_PROJECT,
        TEST_COMPOSE,
        "up",
        "--detach",
        "--wait",
        "--wait-timeout",
        "180",
    )
    wait_for_worker_health()
    assert_runtime_security()
    exercise_worker_session("cold volume")

    compose(VERIFIER_PROJECT, TEST_COMPOSE, "restart", "model-worker")
    wait_for_worker_health()
    exercise_worker_session("graceful restart with reused volume")

    compose(
        VERIFIER_PROJECT, TEST_COMPOSE, "kill", "--signal", "SIGKILL", "model-worker"
    )
    compose(
        VERIFIER_PROJECT,
        TEST_COMPOSE,
        "up",
        "--detach",
        "--wait",
        "--wait-timeout",
        "180",
        "model-worker",
    )
    wait_for_worker_health()
    exercise_worker_session("unclean restart with stale socket")


def cleanup() -> None:
    for project, compose_file in (
        (VERIFIER_PROJECT, TEST_COMPOSE),
        (ENVIRONMENT_PROJECT, ENVIRONMENT_COMPOSE),
    ):
        compose(
            project,
            compose_file,
            "down",
            "--volumes",
            "--remove-orphans",
            "--rmi",
            "local",
            check=False,
        )


def main() -> int:
    run("docker", "info", capture=True)
    assert_compose_contract()
    try:
        compose(ENVIRONMENT_PROJECT, ENVIRONMENT_COMPOSE, "build")
        compose(VERIFIER_PROJECT, TEST_COMPOSE, "build")
        assert_image_contracts()
        assert_compose_lifecycle()
    except BaseException:
        compose(
            VERIFIER_PROJECT,
            TEST_COMPOSE,
            "logs",
            "--no-color",
            check=False,
        )
        raise
    finally:
        cleanup()
    print("\nDocker contracts: OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
