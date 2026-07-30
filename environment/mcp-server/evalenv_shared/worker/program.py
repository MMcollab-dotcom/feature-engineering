"""Generic child-side worker program loop."""

from __future__ import annotations

import sys
from typing import Any, TextIO

from evalenv_shared.worker.protocol import decode_message, encode_message


class WorkerProgram:
    def __init__(self, handler: Any) -> None:
        self.handler = handler

    def run(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        input_stream = stdin or sys.stdin
        output_stream = stdout or sys.__stdout__
        for line in input_stream:
            message = decode_message(line)
            request_id = int(message["id"])
            if message.get("type") == "shutdown":
                output_stream.write(
                    encode_message(
                        {
                            "id": request_id,
                            "type": "result",
                            "ok": True,
                            "value": None,
                        }
                    )
                )
                output_stream.flush()
                return
            response = self.handler.handle(message)
            output_stream.write(
                encode_message(
                    {
                        "id": request_id,
                        "type": "result",
                        **dict(response),
                    }
                )
            )
            output_stream.flush()
