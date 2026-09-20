# GraphKV

GraphKV is a research prototype for **layout-aware KV-state management in
request-specific multi-agent graphs**. It addresses the gap between a topology
optimizer, which changes agents and edges, and a KV-reusing runtime, which must
decide whether cached prompt state remains valid.

The key observation is that neither an `agent_id` nor a complete graph ID is
the right state key. An agent's state is reusable when its local prompt layout
is unchanged: model/template version, consumer role, ordered predecessors, and
placeholder schema must match. GraphKV derives a content-addressed layout ID
from that structure and uses it consistently across serving instances.

## Scope

This repository implements the project mechanism, not a new distributed object
store:

- validated request-specific DAGs and deterministic local layouts;
- layout, complete-graph, and reset state policies;
- a versioned, byte-bounded local LRU store;
- a real length-prefixed TCP store for independent serving processes;
- local-hit, remote-hit, miss, and remote-rejected decisions;
- a codec for KVCOMM's Hugging Face `DynamicCache` tensors;
- a CPU demo and an RTX-4060-sized real-model benchmark.

LMCache and Mooncake remain candidate production data planes. The first
prototype uses a small explicit store so the layout-validity mechanism can be
tested without RDMA, a cluster, or a particular inference engine. The
`DynamicCacheCodec` is the integration boundary that a later LMCache backend
can reuse.

## Architecture

```text
request-specific graph
          |
          v
Topology adapter ---> active roles + ordered predecessors + placement
          |
          v
Layout identity ----> SHA-256(model, template, role, predecessors, schema)
          |
          v
State manager ------> local hit | remote hit | recompute
          |                              |
          |                    TCP / LMCache / Mooncake
          v                              v
DynamicCache codec <-------------- versioned state bytes
```

The safety rule is simple: an invocation may only load state stored under its
exact layout identity. Unknown or incompatible layouts recompute.

## Quick start: CPU control plane

The package itself has no mandatory dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
graphkv-demo --policy local_layout
```

To exercise an independent state-server process:

```bash
# Terminal 1
graphkv-server --host 127.0.0.1 --port 7648 --max-gib 1

# Terminal 2
graphkv-demo --server 127.0.0.1:7648 --policy local_layout
```

Compare all three policies:

```bash
graphkv-demo --policy reset
graphkv-demo --policy whole_graph
graphkv-demo --policy local_layout
```

The demo uses sleeps and opaque bytes only to validate policy behavior. It is
not reported as a performance result.

## RTX 4060 benchmark

The default benchmark uses Qwen2.5-0.5B in FP16, a 512-token prompt, batch size
one, and one shared model. This is intentionally small enough for an 8-GB RTX
4060 while still executing real transformer prefill and moving real KV tensors.

Install a CUDA-enabled PyTorch build appropriate for the host first, then:

```bash
python -m pip install -e '.[gpu]'

# Terminal 1: independent state process
graphkv-server --port 7648 --max-gib 4

# Terminal 2: real model prefill + TCP state movement
python scripts/run_gpu_benchmark.py \
  --server 127.0.0.1:7648 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-tokens 512 \
  --repetitions 3 \
  --placement round_robin \
  --policy local_layout
```

Run `reset` and `whole_graph` with identical arguments for baselines. Results
are written as JSON Lines under `results/` and include the decision, layout,
payload size, dense-prefill time, acquisition time, and CPU-to-GPU restore
time. `round_robin` changes a role's logical serving instance across requests,
so a recurring layout exercises a real TCP remote hit; `stable` measures local
recurrence. The benchmark fails early if CUDA is not visible instead of
silently falling back to CPU.

`critical_path_ms` is defined according to the actual state location: a miss
uses the freshly computed GPU cache without decoding it again, a local hit uses
the instance's GPU-resident object, and a remote hit includes TCP acquisition
plus CPU-to-GPU restoration. Synchronous publication overhead is reported
separately rather than incorrectly charging serialization to current-state
readiness.

### What one 4060 can and cannot establish

A single 4060 can validate real KV generation, layout reuse, serialization,
TCP transfer between processes, and the local/remote/recompute control path.
Both logical instances share one physical model in this benchmark, so it does
**not** claim cross-GPU throughput. If a second GPU becomes available, run one
model process per GPU while keeping the same server protocol. Multi-node and
RDMA are optional extensions, not prerequisites for the layout-identity result.

## Experimental baselines

- `reset`: request-scoped keys; every request computes new state.
- `whole_graph`: state is isolated by complete graph and agent role.
- `local_layout`: state is shared whenever the agent-local structure matches,
  even if unrelated portions of the complete graph changed.

The required comparison uses the same graph trace for all policies. This keeps
task topology and token structure fixed while isolating the state-management
effect.

## Next integration step

KVCOMM currently stores initialized prefixes and anchors in process-global
dictionaries indexed primarily by node ID. The next adapter will replace those
lookups with GraphKV layout IDs and package the following state:

- prefix `DynamicCache` segments;
- prefix token IDs and placeholder positions;
- layout-scoped anchor tensors and their metadata;
- model/template and codec versions.

LMCache's public connector abstraction provides a model for moving tensors,
but its released connectors target vLLM, SGLang, and TensorRT-LLM cache
layouts. KVCOMM uses Transformers `DynamicCache`, which is why this repository
contains an explicit codec rather than claiming a zero-code LMCache
integration.

## Research questions

1. When do dynamic-graph token reductions overstate measured prefill/TTFT
   savings because reusable layouts are fragmented?
2. Does agent-local layout identity reduce dense prefill versus reset and
   complete-graph isolation without changing model output?
3. When state is remote, where is the break-even point between transferring it
   and recomputing it?

Negative cases---low layout recurrence, small prompts, or transfer cost larger
than prefill---are part of the result rather than excluded.
