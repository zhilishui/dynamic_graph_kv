"""Length-prefixed protocol shared by the TCP state server and client."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

HEADER = struct.Struct("!Q")
MAX_HEADER_BYTES = 1 << 20


class ProtocolError(RuntimeError):
    """Raised when a peer sends a malformed GraphKV message."""


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket, header: dict[str, Any], payload: bytes = b""
) -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise ProtocolError("header is too large")
    sock.sendall(HEADER.pack(len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def recv_frame(
    sock: socket.socket, max_payload_bytes: int
) -> tuple[dict[str, Any], bytes]:
    (header_size,) = HEADER.unpack(recv_exact(sock, HEADER.size))
    if header_size > MAX_HEADER_BYTES:
        raise ProtocolError("header is too large")
    try:
        header = json.loads(recv_exact(sock, header_size))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON header") from exc
    payload_size = int(header.get("payload_bytes", 0))
    if payload_size < 0 or payload_size > max_payload_bytes:
        raise ProtocolError(f"payload size {payload_size} is not allowed")
    return header, recv_exact(sock, payload_size) if payload_size else b""
