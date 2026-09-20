"""Standalone TCP server for cross-process GraphKV state sharing."""

from __future__ import annotations

import argparse
import socketserver
from typing import Any

from .protocol import ProtocolError, recv_frame, send_frame
from .store import MemoryStateStore


class StateRequestHandler(socketserver.BaseRequestHandler):
    """Serve one request per TCP connection."""

    server: StateTcpServer

    def handle(self) -> None:
        try:
            request, payload = recv_frame(self.request, self.server.max_payload_bytes)
            response, response_payload = self._dispatch(request, payload)
            send_frame(self.request, {"status": "ok", **response}, response_payload)
        except Exception as exc:  # noqa: BLE001 - RPC errors must reach the client.
            error_type = type(exc).__name__
            try:
                send_frame(
                    self.request,
                    {
                        "status": "error",
                        "error_type": error_type,
                        "error": str(exc),
                    },
                )
            except OSError:
                return

    def _dispatch(
        self, request: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        operation = request.get("op")
        if operation == "ping":
            return {}, b""
        if operation == "stats":
            return {"stats": self.server.store.stats()}, b""

        key = request.get("key")
        if not isinstance(key, str) or not key:
            raise ProtocolError("a non-empty string key is required")

        if operation == "get":
            record = self.server.store.get(key)
            if record is None:
                return {"found": False}, b""
            return (
                {
                    "found": True,
                    "version": record.version,
                    "metadata": record.metadata,
                    "created_ns": record.created_ns,
                },
                record.payload,
            )
        if operation == "put":
            expected = request.get("expected_version")
            record = self.server.store.put(
                key,
                payload,
                metadata=request.get("metadata"),
                expected_version=int(expected) if expected is not None else None,
            )
            return {
                "version": record.version,
                "created_ns": record.created_ns,
            }, b""
        if operation == "delete":
            expected = request.get("expected_version")
            deleted = self.server.store.delete(
                key,
                expected_version=int(expected) if expected is not None else None,
            )
            return {"deleted": deleted}, b""
        raise ProtocolError(f"unsupported operation: {operation!r}")


class StateTcpServer(socketserver.ThreadingTCPServer):
    """Thread-per-connection server around a versioned in-memory store."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: MemoryStateStore,
        max_payload_bytes: int = 2 << 30,
    ) -> None:
        self.store = store
        self.max_payload_bytes = max_payload_bytes
        super().__init__(address, StateRequestHandler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7648)
    parser.add_argument(
        "--max-gib",
        type=float,
        default=4.0,
        help="maximum bytes retained by the server",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    max_bytes = int(args.max_gib * (1 << 30))
    with StateTcpServer(
        (args.host, args.port), MemoryStateStore(max_bytes=max_bytes)
    ) as server:
        print(
            f"GraphKV state server listening on {args.host}:{args.port} "
            f"with {args.max_gib:g} GiB capacity",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
