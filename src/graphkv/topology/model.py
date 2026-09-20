"""Validated request-specific multi-agent graph descriptions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


def graph_identity(
    active_roles: Iterable[str], edges: Iterable[tuple[str, str]]
) -> str:
    """Return a stable identity for a complete graph baseline."""

    value = {
        "active_roles": sorted(active_roles),
        "edges": sorted([list(edge) for edge in edges]),
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """One request's active roles and directed communication edges."""

    active_roles: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    @classmethod
    def create(
        cls,
        active_roles: Iterable[str],
        edges: Iterable[tuple[str, str]],
    ) -> GraphSpec:
        graph = cls(tuple(active_roles), tuple(edges))
        graph.topological_order()  # Validate eagerly.
        return graph

    @property
    def digest(self) -> str:
        return graph_identity(self.active_roles, self.edges)

    def predecessors(self, role: str) -> tuple[str, ...]:
        if role not in self.active_roles:
            raise KeyError(f"inactive role: {role}")
        return tuple(sorted(source for source, target in self.edges if target == role))

    def topological_order(self) -> tuple[str, ...]:
        roles = set(self.active_roles)
        if len(roles) != len(self.active_roles):
            raise ValueError("active roles must be unique")
        indegree = {role: 0 for role in roles}
        successors: dict[str, list[str]] = {role: [] for role in roles}
        seen_edges: set[tuple[str, str]] = set()
        for source, target in self.edges:
            if source not in roles or target not in roles:
                raise ValueError(
                    f"edge {(source, target)!r} references an inactive role"
                )
            if source == target:
                raise ValueError("self edges are not supported")
            if (source, target) in seen_edges:
                raise ValueError(f"duplicate edge: {(source, target)!r}")
            seen_edges.add((source, target))
            successors[source].append(target)
            indegree[target] += 1

        ready = sorted(role for role, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            role = ready.pop(0)
            order.append(role)
            for target in sorted(successors[role]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(roles):
            raise ValueError("agent graph must be acyclic")
        return tuple(order)
