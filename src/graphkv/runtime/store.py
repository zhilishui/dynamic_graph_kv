"""Runtime stores for opaque, versioned KV-state blobs."""

from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from graphkv.runtime.protocol import recv_frame, send_frame


@dataclass(frozen=True, slots=True)
class StateRecord:
    """An immutable state object returned by a state store."""

    key: str
    version: int
    payload: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    created_ns: int = field(default_factory=time.time_ns)


class StateStore(Protocol):
    def get(self, key: str) -> StateRecord | None: ...

    def put(
        self,
        key: str,
        payload: bytes,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> StateRecord: ...

    def delete(self, key: str, expected_version: int | None = None) -> bool: ...


class VersionConflict(RuntimeError):
    """Raised when a conditional state update observes another version."""


class MemoryStateStore:
    """Thread-safe byte-bounded LRU store used locally and by the TCP server."""

    def __init__(self, max_bytes: int = 1 << 30) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._records: OrderedDict[str, StateRecord] = OrderedDict()
        self._bytes = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> StateRecord | None:
        with self._lock:
            record = self._records.get(key)
            if record is not None:
                self._records.move_to_end(key)
            return record

    def put(
        self,
        key: str,
        payload: bytes,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> StateRecord:
        if not key:
            raise ValueError("key cannot be empty")
        payload = bytes(payload)
        if len(payload) > self.max_bytes:
            raise ValueError("payload is larger than the entire store")
        with self._lock:
            previous = self._records.get(key)
            previous_version = previous.version if previous else 0
            if expected_version is not None and previous_version != expected_version:
                raise VersionConflict(
                    f"expected version {expected_version}, found {previous_version}"
                )
            if previous is not None:
                self._bytes -= len(previous.payload)
            record = StateRecord(
                key=key,
                version=previous_version + 1,
                payload=payload,
                metadata=dict(metadata or {}),
            )
            self._records[key] = record
            self._records.move_to_end(key)
            self._bytes += len(payload)
            self._evict_to_limit(exclude=key)
            return record

    def delete(self, key: str, expected_version: int | None = None) -> bool:
        with self._lock:
            previous = self._records.get(key)
            if previous is None:
                return False
            if expected_version is not None and previous.version != expected_version:
                raise VersionConflict(
                    f"expected version {expected_version}, found {previous.version}"
                )
            del self._records[key]
            self._bytes -= len(previous.payload)
            return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._records),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "evictions": self._evictions,
            }

    def _evict_to_limit(self, exclude: str) -> None:
        while self._bytes > self.max_bytes and self._records:
            oldest_key = next(iter(self._records))
            if oldest_key == exclude and len(self._records) == 1:
                break
            if oldest_key == exclude:
                self._records.move_to_end(oldest_key)
                oldest_key = next(iter(self._records))
            evicted = self._records.pop(oldest_key)
            self._bytes -= len(evicted.payload)
            self._evictions += 1


class TcpStateStoreClient:
    """Network client for an independent GraphKV state-server process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7648,
        timeout_s: float = 10.0,
        max_payload_bytes: int = 2 << 30,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_payload_bytes = max_payload_bytes

    def get(self, key: str) -> StateRecord | None:
        header, payload = self._request({"op": "get", "key": key})
        if not header["found"]:
            return None
        return StateRecord(
            key=key,
            version=int(header["version"]),
            payload=payload,
            metadata=dict(header.get("metadata", {})),
            created_ns=int(header["created_ns"]),
        )

    def put(
        self,
        key: str,
        payload: bytes,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> StateRecord:
        request: dict[str, Any] = {
            "op": "put",
            "key": key,
            "metadata": dict(metadata or {}),
        }
        if expected_version is not None:
            request["expected_version"] = expected_version
        header, _ = self._request(request, bytes(payload))
        return StateRecord(
            key=key,
            version=int(header["version"]),
            payload=bytes(payload),
            metadata=dict(metadata or {}),
            created_ns=int(header["created_ns"]),
        )

    def delete(self, key: str, expected_version: int | None = None) -> bool:
        request: dict[str, Any] = {"op": "delete", "key": key}
        if expected_version is not None:
            request["expected_version"] = expected_version
        header, _ = self._request(request)
        return bool(header["deleted"])

    def stats(self) -> dict[str, int]:
        header, _ = self._request({"op": "stats"})
        return {str(k): int(v) for k, v in header["stats"].items()}

    def ping(self) -> bool:
        header, _ = self._request({"op": "ping"})
        return header.get("status") == "ok"

    def _request(
        self, header: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes]:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as sock:
            sock.settimeout(self.timeout_s)
            send_frame(sock, header, payload)
            response, response_payload = recv_frame(sock, self.max_payload_bytes)
        if response.get("status") != "ok":
            error = response.get("error", "unknown remote error")
            if response.get("error_type") == "VersionConflict":
                raise VersionConflict(str(error))
            raise RuntimeError(str(error))
        return response, response_payload
