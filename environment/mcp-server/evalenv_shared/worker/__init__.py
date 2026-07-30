"""Shared worker host primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, stderr_tail: str = "") -> None:
        self.stderr_tail = stderr_tail
        super().__init__(message)


class WorkerProtocolError(WorkerError):
    """Raised when the private local worker envelope is violated."""


class WorkerProcessError(WorkerError):
    """Raised when a worker process exits or cannot be reached."""


class WorkerRemoteError(WorkerError):
    def __init__(
        self,
        error: Mapping[str, Any],
        *,
        stderr_tail: str = "",
    ) -> None:
        self.error = dict(error)
        super().__init__(str(self.error), stderr_tail=stderr_tail)


class WorkerTimeoutError(WorkerError):
    def __init__(
        self,
        message: str,
        *,
        timeout_s: float,
        stderr_tail: str = "",
    ) -> None:
        self.timeout_s = float(timeout_s)
        super().__init__(message, stderr_tail=stderr_tail)


