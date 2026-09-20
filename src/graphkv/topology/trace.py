"""Topology-generator boundary: validated request-specific graph traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphkv.topology.model import GraphSpec


@dataclass(frozen=True, slots=True)
class TopologyRequest:
    """One generator output consumed by the dynamic runtime."""

    request_id: str
    graph: GraphSpec

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> TopologyRequest:
        request_id = value.get("request_id")
        active_roles = value.get("active_roles")
        edges = value.get("edges")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(active_roles, list) or not all(
            isinstance(role, str) and role for role in active_roles
        ):
            raise ValueError("active_roles must be a list of non-empty strings")
        if not isinstance(edges, list):
            raise TypeError("edges must be a list")
        normalized_edges: list[tuple[str, str]] = []
        for edge in edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(role, str) and role for role in edge)
            ):
                raise ValueError("each edge must contain two non-empty role strings")
            normalized_edges.append((edge[0], edge[1]))
        return cls(
            request_id=request_id,
            graph=GraphSpec.create(active_roles, normalized_edges),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "active_roles": list(self.graph.active_roles),
            "edges": [list(edge) for edge in self.graph.edges],
        }


def load_topology_trace(path: Path) -> tuple[TopologyRequest, ...]:
    """Load the JSONL contract emitted by a topology generator."""

    requests: list[TopologyRequest] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        request = TopologyRequest.from_mapping(value)
        if request.request_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate request_id")
        seen_ids.add(request.request_id)
        requests.append(request)
    if not requests:
        raise ValueError(f"topology trace is empty: {path}")
    return tuple(requests)
