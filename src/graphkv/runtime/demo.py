"""CPU-only demonstration of policies over a dynamic graph trace."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graphkv.runtime.manager import Policy
from graphkv.runtime.orchestrator import DynamicKVRuntime
from graphkv.runtime.store import MemoryStateStore, TcpStateStoreClient
from graphkv.topology.trace import load_topology_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=[policy.value for policy in Policy], default="local_layout"
    )
    parser.add_argument("--server", help="optional HOST:PORT for cross-process state")
    parser.add_argument("--compute-ms", type=float, default=5.0)
    parser.add_argument("--state-kib", type=int, default=64)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("configs/example_trace.jsonl"),
        help="JSONL emitted by the topology-generator boundary",
    )
    parser.add_argument(
        "--placement", choices=("stable", "round_robin"), default="round_robin"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.server:
        host, port = args.server.rsplit(":", 1)
        store = TcpStateStoreClient(host, int(port))
    else:
        store = MemoryStateStore(max_bytes=64 << 20)
    runtime = DynamicKVRuntime(
        store,
        model="demo-model",
        policy=Policy(args.policy),
        placement=args.placement,
    )
    trace = load_topology_trace(args.trace)

    for request_index, request in enumerate(trace):
        for invocation in runtime.plan(request, request_position=request_index):
            role = invocation.role

            def compute(role: str = role) -> bytes:
                time.sleep(args.compute_ms / 1000)
                marker = f"state:{role}".encode()
                repeats = max(1, (args.state_kib << 10) // len(marker))
                return (marker * repeats)[: args.state_kib << 10]

            result = runtime.acquire(
                invocation,
                compute=compute,
            )
            print(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "role": role,
                        "instance": invocation.instance,
                        "layout": invocation.layout.short,
                        "decision": result.decision.value,
                        "elapsed_ms": result.elapsed_ns / 1e6,
                        "bytes": len(result.payload),
                    }
                )
            )
    print(json.dumps({"summary": runtime.stats()}))


if __name__ == "__main__":
    main()
