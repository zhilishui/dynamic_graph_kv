"""Dynamic KV runtime control plane over a topology-generator trace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from graphkv.runtime.identity import LayoutIdentity, layouts_for_graph
from graphkv.runtime.manager import AccessResult, LayoutStateManager, Policy
from graphkv.runtime.store import StateStore
from graphkv.topology.trace import TopologyRequest


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    """The graph-derived inputs required for one agent invocation."""

    request_id: str
    graph_digest: str
    role: str
    instance: str
    layout: LayoutIdentity


class DynamicKVRuntime:
    """Map a per-request graph to layout-scoped state decisions.

    Model execution remains in the caller so this control plane can wrap
    KVCOMM, a Transformers smoke model, or a CPU test without owning either.
    """

    def __init__(
        self,
        store: StateStore,
        *,
        model: str,
        policy: Policy = Policy.LOCAL_LAYOUT,
        placement: str = "round_robin",
        instances: tuple[str, ...] = ("instance-0", "instance-1"),
    ) -> None:
        if not instances or len(set(instances)) != len(instances):
            raise ValueError("instances must be non-empty and unique")
        if placement not in ("stable", "round_robin"):
            raise ValueError("placement must be 'stable' or 'round_robin'")
        self.model = model
        self.policy = policy
        self.placement = placement
        self.instances = instances
        self.managers = {
            instance: LayoutStateManager(store, policy=policy) for instance in instances
        }

    def plan(
        self,
        request: TopologyRequest,
        *,
        request_position: int,
        repetition: int = 0,
    ) -> tuple[InvocationPlan, ...]:
        layouts = layouts_for_graph(request.graph, model=self.model)
        plans: list[InvocationPlan] = []
        for role in request.graph.topological_order():
            base = sum(role.encode("utf-8")) % len(self.instances)
            if self.placement == "round_robin":
                base = (base + request_position + repetition) % len(self.instances)
            plans.append(
                InvocationPlan(
                    request_id=request.request_id,
                    graph_digest=request.graph.digest,
                    role=role,
                    instance=self.instances[base],
                    layout=layouts[role],
                )
            )
        return tuple(plans)

    def acquire(
        self,
        invocation: InvocationPlan,
        *,
        compute: Callable[[], bytes],
        estimated_compute_ns: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AccessResult:
        invocation_metadata = {
            "request_id": invocation.request_id,
            "role": invocation.role,
            "instance": invocation.instance,
            "layout": invocation.layout.digest,
            **(metadata or {}),
        }
        return self.managers[invocation.instance].acquire(
            invocation.layout,
            request_id=invocation.request_id,
            graph_digest=invocation.graph_digest,
            compute=compute,
            estimated_compute_ns=estimated_compute_ns,
            metadata=invocation_metadata,
        )

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            instance: manager.stats.as_dict()
            for instance, manager in self.managers.items()
        }
