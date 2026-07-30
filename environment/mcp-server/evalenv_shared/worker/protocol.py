"""Compact newline-delimited JSON envelope for local workers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def encode_message(message: Mapping[str, Any]) -> str:
    return json.dumps(dict(message), allow_nan=False, separators=(",", ":")) + "\n"


def decode_message(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Worker protocol message must be a JSON object.")
    return value
