"""Policy and metrics for layout-aware local/remote state decisions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .identity import LayoutIdentity
from .store import StateRecord, StateStore


class Policy(str, Enum):
    RESET = "reset"
    WHOLE_GRAPH = "whole_graph"
    LOCAL_LAYOUT = "local_layout"


class Decision(str, Enum):
    LOCAL_HIT = "local_hit"
    REMOTE_HIT = "remote_hit"
    RECOMPUTE = "recompute"
    REMOTE_REJECTED = "remote_rejected"


@dataclass(slots=True)
class AccessResult:
    decision: Decision
    key: str
    payload: bytes
    elapsed_ns: int
    transfer_bytes: int
    version: int


@dataclass(slots=True)
class ManagerStats:
    local_hits: int = 0
    remote_hits: int = 0
    recomputes: int = 0
    remote_rejected: int = 0
    bytes_loaded: int = 0
    bytes_stored: int = 0
    load_ns: int = 0
    compute_ns: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


class LayoutStateManager:
    """Choose local reuse, remote fetch, or safe recomputation.

    The caller supplies the expensive state computation. The manager only
    reasons about identity, location, versions, and an optional admission rule.
    """

    def __init__(
        self,
        remote_store: StateStore | None = None,
        *,
        policy: Policy = Policy.LOCAL_LAYOUT,
        max_local_entries: int = 64,
        fetch_admission: Callable[[StateRecord, int | None], bool] | None = None,
    ) -> None:
        if max_local_entries <= 0:
            raise ValueError("max_local_entries must be positive")
        self.remote_store = remote_store
        self.policy = policy
        self.max_local_entries = max_local_entries
        self.fetch_admission = fetch_admission
        self._local: dict[str, StateRecord] = {}
        self._local_order: list[str] = []
        self.stats = ManagerStats()

    def state_key(
        self,
        layout: LayoutIdentity,
        *,
        request_id: str,
        graph_digest: str,
    ) -> str:
        if self.policy == Policy.RESET:
            return f"request:{request_id}:{layout.spec.consumer_role}"
        if self.policy == Policy.WHOLE_GRAPH:
            return f"graph:{graph_digest}:{layout.spec.consumer_role}"
        return f"layout:{layout.digest}"

    def acquire(
        self,
        layout: LayoutIdentity,
        *,
        request_id: str,
        graph_digest: str,
        compute: Callable[[], bytes],
        estimated_compute_ns: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AccessResult:
        key = self.state_key(layout, request_id=request_id, graph_digest=graph_digest)
        started = time.perf_counter_ns()
        if self.policy == Policy.RESET:
            compute_started = time.perf_counter_ns()
            payload = bytes(compute())
            compute_ns = time.perf_counter_ns() - compute_started
            self.stats.recomputes += 1
            self.stats.compute_ns += compute_ns
            return AccessResult(
                Decision.RECOMPUTE,
                key,
                payload,
                time.perf_counter_ns() - started,
                0,
                0,
            )

        local = self._local.get(key)
        if local is not None:
            self._touch(key)
            self.stats.local_hits += 1
            return AccessResult(
                Decision.LOCAL_HIT,
                key,
                local.payload,
                time.perf_counter_ns() - started,
                0,
                local.version,
            )

        remote = self.remote_store.get(key) if self.remote_store is not None else None
        if remote is not None:
            accepted = self.fetch_admission is None or self.fetch_admission(
                remote, estimated_compute_ns
            )
            if accepted:
                elapsed = time.perf_counter_ns() - started
                self._remember(remote)
                self.stats.remote_hits += 1
                self.stats.bytes_loaded += len(remote.payload)
                self.stats.load_ns += elapsed
                return AccessResult(
                    Decision.REMOTE_HIT,
                    key,
                    remote.payload,
                    elapsed,
                    len(remote.payload),
                    remote.version,
                )
            self.stats.remote_rejected += 1
            rejected = True
        else:
            rejected = False

        compute_started = time.perf_counter_ns()
        payload = bytes(compute())
        compute_ns = time.perf_counter_ns() - compute_started
        record = StateRecord(
            key=key, version=1, payload=payload, metadata=metadata or {}
        )
        if self.remote_store is not None:
            record = self.remote_store.put(key, payload, metadata=metadata)
            self.stats.bytes_stored += len(payload)
        self._remember(record)
        self.stats.recomputes += 1
        self.stats.compute_ns += compute_ns
        decision = Decision.REMOTE_REJECTED if rejected else Decision.RECOMPUTE
        return AccessResult(
            decision,
            key,
            payload,
            time.perf_counter_ns() - started,
            0,
            record.version,
        )

    def clear_local(self) -> None:
        self._local.clear()
        self._local_order.clear()

    def _remember(self, record: StateRecord) -> None:
        self._local[record.key] = record
        self._touch(record.key)
        while len(self._local_order) > self.max_local_entries:
            evicted_key = self._local_order.pop(0)
            self._local.pop(evicted_key, None)

    def _touch(self, key: str) -> None:
        try:
            self._local_order.remove(key)
        except ValueError:
            pass
        self._local_order.append(key)
