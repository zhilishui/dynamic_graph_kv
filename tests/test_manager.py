import unittest

from graphkv.identity import LayoutIdentity, LayoutSpec
from graphkv.manager import Decision, LayoutStateManager, Policy
from graphkv.store import MemoryStateStore


def layout(predecessors: tuple[str, ...]) -> LayoutIdentity:
    return LayoutIdentity.from_spec(
        LayoutSpec.create(
            model="model",
            template_version="v1",
            consumer_role="C",
            predecessor_roles=predecessors,
            placeholder_schema=tuple(f"message:{item}" for item in predecessors),
        )
    )


class ManagerTests(unittest.TestCase):
    def test_layout_policy_reuses_across_complete_graphs(self) -> None:
        calls = 0

        def compute() -> bytes:
            nonlocal calls
            calls += 1
            return b"state"

        manager = LayoutStateManager(MemoryStateStore(), policy=Policy.LOCAL_LAYOUT)
        first = manager.acquire(
            layout(("A",)), request_id="1", graph_digest="graph-1", compute=compute
        )
        second = manager.acquire(
            layout(("A",)), request_id="2", graph_digest="graph-2", compute=compute
        )
        self.assertEqual(first.decision, Decision.RECOMPUTE)
        self.assertEqual(second.decision, Decision.LOCAL_HIT)
        self.assertEqual(calls, 1)

    def test_graph_policy_isolates_complete_graphs(self) -> None:
        calls = 0

        def compute() -> bytes:
            nonlocal calls
            calls += 1
            return b"state"

        manager = LayoutStateManager(MemoryStateStore(), policy=Policy.WHOLE_GRAPH)
        for graph in ("graph-1", "graph-2"):
            manager.acquire(
                layout(("A",)), request_id=graph, graph_digest=graph, compute=compute
            )
        self.assertEqual(calls, 2)

    def test_second_manager_observes_remote_hit(self) -> None:
        store = MemoryStateStore()
        first = LayoutStateManager(store)
        second = LayoutStateManager(store)
        identity = layout(("A",))
        first.acquire(
            identity, request_id="1", graph_digest="g", compute=lambda: b"state"
        )
        result = second.acquire(
            identity,
            request_id="2",
            graph_digest="g2",
            compute=lambda: self.fail("remote state should avoid recomputation"),
        )
        self.assertEqual(result.decision, Decision.REMOTE_HIT)

    def test_admission_can_choose_recompute(self) -> None:
        store = MemoryStateStore()
        identity = layout(("A",))
        first = LayoutStateManager(store)
        first.acquire(
            identity, request_id="1", graph_digest="g", compute=lambda: b"old"
        )
        second = LayoutStateManager(
            store, fetch_admission=lambda _record, _estimate: False
        )
        result = second.acquire(
            identity, request_id="2", graph_digest="g", compute=lambda: b"new"
        )
        self.assertEqual(result.decision, Decision.REMOTE_REJECTED)
        self.assertEqual(result.payload, b"new")

    def test_reset_does_not_publish_unreusable_state(self) -> None:
        store = MemoryStateStore()
        manager = LayoutStateManager(store, policy=Policy.RESET)
        result = manager.acquire(
            layout(("A",)), request_id="1", graph_digest="g", compute=lambda: b"state"
        )
        self.assertEqual(result.decision, Decision.RECOMPUTE)
        self.assertEqual(result.version, 0)
        self.assertEqual(store.stats()["entries"], 0)
        self.assertEqual(manager.stats.bytes_stored, 0)


if __name__ == "__main__":
    unittest.main()
