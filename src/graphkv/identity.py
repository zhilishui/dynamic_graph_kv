"""Stable identities for graph-local prompt layouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """The structural inputs that determine whether an agent state is reusable.

    Request values are deliberately absent. A predecessor's message may change
    while its placeholder and surrounding prompt layout remain stable.
    """

    model: str
    template_version: str
    consumer_role: str
    predecessor_roles: tuple[str, ...]
    placeholder_schema: tuple[str, ...]
    token_structure_version: str = "v1"

    @classmethod
    def create(
        cls,
        *,
        model: str,
        template_version: str,
        consumer_role: str,
        predecessor_roles: Iterable[str],
        placeholder_schema: Iterable[str],
        token_structure_version: str = "v1",
    ) -> LayoutSpec:
        return cls(
            model=model,
            template_version=template_version,
            consumer_role=consumer_role,
            predecessor_roles=tuple(predecessor_roles),
            placeholder_schema=tuple(placeholder_schema),
            token_structure_version=token_structure_version,
        )


@dataclass(frozen=True, slots=True)
class LayoutIdentity:
    """A content-addressed identity for one agent's local prompt structure."""

    digest: str
    spec: LayoutSpec

    @classmethod
    def from_spec(cls, spec: LayoutSpec) -> LayoutIdentity:
        return cls(digest=_digest(asdict(spec)), spec=spec)

    @property
    def short(self) -> str:
        return self.digest[:12]


def graph_identity(
    active_roles: Iterable[str], edges: Iterable[tuple[str, str]]
) -> str:
    """Return a stable identity for a complete graph baseline."""

    value = {
        "active_roles": sorted(active_roles),
        "edges": sorted([list(edge) for edge in edges]),
    }
    return _digest(value)
