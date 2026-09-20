"""CPU-only demonstration of policies over a dynamic graph trace."""

from __future__ import annotations

import argparse
import json
import time

from .manager import LayoutStateManager, Policy
from .store import MemoryStateStore, TcpStateStoreClient
from .topology import GraphSpec

TRACE = (
    GraphSpec.create(("A", "C"), (("A", "C"),)),
    GraphSpec.create(("B", "C"), (("B", "C"),)),
    GraphSpec.create(("A", "C"), (("A", "C"),)),
    GraphSpec.create(("A", "B", "C"), (("A", "C"), ("B", "C"))),
    GraphSpec.create(("B", "C"), (("B", "C"),)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=[policy.value for policy in Policy], default="local_layout"
    )
    parser.add_argument("--server", help="optional HOST:PORT for cross-process state")
    parser.add_argument("--compute-ms", type=float, default=5.0)
    parser.add_argument("--state-kib", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.server:
        host, port = args.server.rsplit(":", 1)
        store = TcpStateStoreClient(host, int(port))
    else:
        store = MemoryStateStore(max_bytes=64 << 20)
    manager = LayoutStateManager(store, policy=Policy(args.policy))

    for request_index, graph in enumerate(TRACE):
        layouts = graph.layouts(model="demo-model")
        for role in graph.topological_order():

            def compute(role: str = role) -> bytes:
                time.sleep(args.compute_ms / 1000)
                marker = f"state:{role}".encode()
                repeats = max(1, (args.state_kib << 10) // len(marker))
                return (marker * repeats)[: args.state_kib << 10]

            result = manager.acquire(
                layouts[role],
                request_id=str(request_index),
                graph_digest=graph.digest,
                compute=compute,
                metadata={"role": role, "layout": layouts[role].digest},
            )
            print(
                json.dumps(
                    {
                        "request": request_index,
                        "role": role,
                        "layout": layouts[role].short,
                        "decision": result.decision.value,
                        "elapsed_ms": result.elapsed_ns / 1e6,
                        "bytes": len(result.payload),
                    }
                )
            )
    print(json.dumps({"summary": manager.stats.as_dict()}))


if __name__ == "__main__":
    main()
