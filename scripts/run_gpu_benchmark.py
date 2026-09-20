#!/usr/bin/env python3
"""Run real small-model prefills and GraphKV transfers on a CUDA GPU.

This is an integration benchmark, not a simulator. It alternates roles between
two logical serving instances. When ``--server`` is supplied, both managers
share an independent GraphKV server, exercising serialization and TCP transfer.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from graphkv.codec import DynamicCacheCodec
from graphkv.manager import LayoutStateManager, Policy
from graphkv.store import MemoryStateStore, TcpStateStoreClient
from graphkv.topology import GraphSpec

TRACE = (
    GraphSpec.create(("A", "C"), (("A", "C"),)),
    GraphSpec.create(("B", "C"), (("B", "C"),)),
    GraphSpec.create(("A", "C"), (("A", "C"),)),
    GraphSpec.create(("A", "B", "C"), (("A", "C"), ("B", "C"))),
    GraphSpec.create(("B", "C"), (("B", "C"),)),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    result.add_argument(
        "--model-revision",
        help="optional Hugging Face commit or tag used for reproducible downloads",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--prompt-tokens", type=int, default=512)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--server", help="HOST:PORT of graphkv-server")
    result.add_argument(
        "--placement",
        choices=("stable", "round_robin"),
        default="round_robin",
        help="round_robin forces recurring layouts to cross instance boundaries",
    )
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--quiet", action="store_true", help="print only the summary")
    result.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help="eager avoids partial sliding-window support in Transformers SDPA",
    )
    result.add_argument(
        "--policy", choices=[item.value for item in Policy], default="local_layout"
    )
    result.add_argument("--output", type=Path, default=Path("results/gpu.jsonl"))
    return result


def synchronize(torch: object, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    args = parser().parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install GPU dependencies: pip install -e '.[gpu]'") from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not visible; verify the NVIDIA driver and CUDA-enabled PyTorch"
        )
    if args.prompt_tokens <= 0:
        raise SystemExit("--prompt-tokens must be positive")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    revision_args = {"revision": args.model_revision} if args.model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **revision_args)
    config = AutoConfig.from_pretrained(args.model, **revision_args)
    # Qwen2.5 publishes a sliding_window value even when use_sliding_window is
    # false. Transformers 4.50 warns on the value alone; clearing the disabled
    # field avoids a misleading warning without changing model semantics.
    if not getattr(config, "use_sliding_window", False):
        config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        **revision_args,
    )
    model.to(args.device).eval()
    codec = DynamicCacheCodec()

    warmup_ids = torch.ones(
        (1, min(64, args.prompt_tokens)), dtype=torch.long, device=args.device
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids=warmup_ids, use_cache=True)
    synchronize(torch, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    if args.server:
        host, port = args.server.rsplit(":", 1)
        store = TcpStateStoreClient(host, int(port), timeout_s=60)
    else:
        store = MemoryStateStore(max_bytes=4 << 30)

    managers = {
        "instance-0": LayoutStateManager(store, policy=Policy(args.policy)),
        "instance-1": LayoutStateManager(store, policy=Policy(args.policy)),
    }
    gpu_states: dict[str, dict[str, dict[str, object]]] = {
        "instance-0": {},
        "instance-1": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []

    with args.output.open("w", encoding="utf-8") as output:
        for repetition in range(args.repetitions):
            for request_index, graph in enumerate(TRACE):
                layouts = graph.layouts(model=args.model)
                for role in graph.topological_order():
                    base_instance = sum(role.encode()) % 2
                    if args.placement == "round_robin":
                        base_instance = (base_instance + request_index + repetition) % 2
                    instance = f"instance-{base_instance}"
                    manager = managers[instance]
                    layout = layouts[role]
                    structural_prompt = (
                        f"role={role}; predecessors={','.join(layout.spec.predecessor_roles)}; "
                        f"schema={','.join(layout.spec.placeholder_schema)}. "
                    )
                    base_ids = tokenizer.encode(
                        structural_prompt, add_special_tokens=True
                    )
                    repeats = (args.prompt_tokens + len(base_ids) - 1) // len(base_ids)
                    ids = (base_ids * repeats)[: args.prompt_tokens]
                    input_ids = torch.tensor([ids], device=args.device)

                    dense_ns = 0
                    computed_state: dict[str, object] | None = None

                    def compute(
                        current_ids: object = input_ids,
                        current_layout: object = layout,
                    ) -> bytes:
                        nonlocal dense_ns, computed_state
                        synchronize(torch, args.device)
                        started = time.perf_counter_ns()
                        with torch.inference_mode():
                            result = model(input_ids=current_ids, use_cache=True)
                        synchronize(torch, args.device)
                        dense_ns = time.perf_counter_ns() - started
                        computed_state = {
                            "past_key_values": result.past_key_values,
                            "layout_digest": current_layout.digest,
                            "input_ids": current_ids,
                        }
                        if args.policy == Policy.RESET.value:
                            return b""
                        return codec.encode(computed_state)

                    state_ready_started = time.perf_counter_ns()
                    acquired = manager.acquire(
                        layout,
                        request_id=f"{repetition}-{request_index}",
                        graph_digest=graph.digest,
                        compute=compute,
                        metadata={"role": role, "instance": instance},
                    )
                    restore_ns = 0
                    if acquired.decision.value in ("recompute", "remote_rejected"):
                        assert computed_state is not None
                        active_state = computed_state
                        gpu_states[instance][acquired.key] = active_state
                        modeled_state_ready_ns = dense_ns
                    elif acquired.decision.value == "local_hit":
                        active_state = gpu_states[instance][acquired.key]
                        modeled_state_ready_ns = acquired.elapsed_ns
                    else:
                        restore_started = time.perf_counter_ns()
                        active_state = codec.decode(
                            acquired.payload, device=args.device
                        )
                        synchronize(torch, args.device)
                        restore_ns = time.perf_counter_ns() - restore_started
                        gpu_states[instance][acquired.key] = active_state
                        modeled_state_ready_ns = acquired.elapsed_ns + restore_ns

                    observed_state_ready_ns = (
                        time.perf_counter_ns() - state_ready_started
                    )

                    if active_state["layout_digest"] != layout.digest:
                        raise RuntimeError("loaded state belongs to the wrong layout")
                    if not torch.equal(active_state["input_ids"], input_ids):
                        raise RuntimeError(
                            "loaded state belongs to different prompt tokens"
                        )

                    sample = {
                        "repetition": repetition,
                        "request": request_index,
                        "role": role,
                        "instance": instance,
                        "placement": args.placement,
                        "attn_implementation": args.attn_implementation,
                        "layout": layout.short,
                        "decision": acquired.decision.value,
                        "prompt_tokens": len(ids),
                        "payload_bytes": len(acquired.payload),
                        "acquire_ms": acquired.elapsed_ns / 1e6,
                        "dense_prefill_ms": dense_ns / 1e6,
                        "restore_ms": restore_ns / 1e6,
                        "observed_state_ready_ms": observed_state_ready_ns / 1e6,
                        "modeled_state_ready_ms": modeled_state_ready_ns / 1e6,
                        "synchronous_publication_ms": max(
                            0.0, (acquired.elapsed_ns - dense_ns) / 1e6
                        )
                        if dense_ns
                        else 0.0,
                        "verified": True,
                    }
                    samples.append(sample)
                    output.write(json.dumps(sample) + "\n")
                    output.flush()
                    if not args.quiet:
                        print(json.dumps(sample))

    summary = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "model": args.model,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": getattr(config, "_commit_hash", None),
        },
        "samples": len(samples),
        "decisions": {
            decision: sum(sample["decision"] == decision for sample in samples)
            for decision in sorted({str(sample["decision"]) for sample in samples})
        },
        "median_acquire_ms": statistics.median(
            float(sample["acquire_ms"]) for sample in samples
        ),
        "median_restore_ms": statistics.median(
            float(sample["restore_ms"]) for sample in samples
        ),
        "median_observed_state_ready_ms_by_decision": {
            decision: statistics.median(
                float(sample["observed_state_ready_ms"])
                for sample in samples
                if sample["decision"] == decision
            )
            for decision in sorted({str(sample["decision"]) for sample in samples})
        },
        "median_modeled_state_ready_ms_by_decision": {
            decision: statistics.median(
                float(sample["modeled_state_ready_ms"])
                for sample in samples
                if sample["decision"] == decision
            )
            for decision in sorted({str(sample["decision"]) for sample in samples})
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else "cpu",
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1 << 20)
            if args.device.startswith("cuda")
            else 0.0,
        },
        "manager_stats": {
            name: manager.stats.as_dict() for name, manager in managers.items()
        },
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
