import json
import tempfile
import unittest
from pathlib import Path

from graphkv.metrics import JsonlMetricWriter, summarize_result_file
from graphkv.runtime import Decision, DynamicKVRuntime, MemoryStateStore, Policy
from graphkv.topology import TopologyRequest, load_topology_trace


class TopologyTraceTests(unittest.TestCase):
    def test_example_trace_is_the_generator_contract(self) -> None:
        trace = load_topology_trace(Path("configs/example_trace.jsonl"))
        self.assertEqual(len(trace), 5)
        self.assertEqual(trace[0].request_id, "0")
        self.assertEqual(trace[0].graph.edges, (("A", "C"),))

    def test_duplicate_request_ids_are_rejected(self) -> None:
        row = {
            "request_id": "same",
            "active_roles": ["A"],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n")
            with self.assertRaisesRegex(ValueError, "duplicate request_id"):
                load_topology_trace(path)


class DynamicRuntimeTests(unittest.TestCase):
    def test_runtime_maps_trace_to_layout_reuse(self) -> None:
        store = MemoryStateStore()
        runtime = DynamicKVRuntime(
            store,
            model="model",
            policy=Policy.LOCAL_LAYOUT,
            placement="stable",
        )
        request = TopologyRequest.from_mapping(
            {
                "request_id": "request-1",
                "active_roles": ["A", "C"],
                "edges": [["A", "C"]],
            }
        )
        first_plan = runtime.plan(request, request_position=0)
        second_plan = runtime.plan(request, request_position=1)
        first = runtime.acquire(first_plan[0], compute=lambda: b"state")
        second = runtime.acquire(
            second_plan[0],
            compute=lambda: self.fail("stable recurring layout should be local"),
        )
        self.assertEqual(first.decision, Decision.RECOMPUTE)
        self.assertEqual(second.decision, Decision.LOCAL_HIT)


class MetricSystemTests(unittest.TestCase):
    def test_writer_and_summary_share_one_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            with JsonlMetricWriter(path) as writer:
                writer.write(
                    {
                        "request_id": "request-1",
                        "role": "A",
                        "instance": "instance-0",
                        "layout": "layout",
                        "decision": "local_hit",
                        "observed_state_ready_ms": 1.0,
                        "modeled_state_ready_ms": 0.5,
                        "payload_bytes": 10,
                        "verified": True,
                    }
                )
            summary = summarize_result_file(path)
            self.assertEqual(summary["samples"], 1)
            self.assertTrue(summary["all_verified"])


if __name__ == "__main__":
    unittest.main()
