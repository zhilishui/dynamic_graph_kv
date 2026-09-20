import unittest

from graphkv.runtime import LayoutIdentity, LayoutSpec, layouts_for_graph
from graphkv.topology import GraphSpec


class IdentityTests(unittest.TestCase):
    def test_same_local_layout_survives_unrelated_graph_change(self) -> None:
        first = GraphSpec.create(("A", "B", "C"), (("A", "C"),))
        second = GraphSpec.create(("A", "C", "D"), (("A", "C"),))
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(
            layouts_for_graph(first, model="m")["C"].digest,
            layouts_for_graph(second, model="m")["C"].digest,
        )

    def test_predecessor_change_changes_identity(self) -> None:
        first = GraphSpec.create(("A", "C"), (("A", "C"),))
        second = GraphSpec.create(("B", "C"), (("B", "C"),))
        self.assertNotEqual(
            layouts_for_graph(first, model="m")["C"].digest,
            layouts_for_graph(second, model="m")["C"].digest,
        )

    def test_order_is_part_of_explicit_layout_identity(self) -> None:
        left = LayoutSpec.create(
            model="m",
            template_version="v1",
            consumer_role="C",
            predecessor_roles=("A", "B"),
            placeholder_schema=("a", "b"),
        )
        right = LayoutSpec.create(
            model="m",
            template_version="v1",
            consumer_role="C",
            predecessor_roles=("B", "A"),
            placeholder_schema=("b", "a"),
        )
        self.assertNotEqual(
            LayoutIdentity.from_spec(left).digest,
            LayoutIdentity.from_spec(right).digest,
        )

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "acyclic"):
            GraphSpec.create(("A", "B"), (("A", "B"), ("B", "A")))


if __name__ == "__main__":
    unittest.main()
