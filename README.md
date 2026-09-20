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
topology generator / saved JSONL trace
                  |
                  v
        Dynamic KV runtime
        - graph adapter
        - layout identity
        - local / remote / recompute
        - memory or TCP backend
                  |
                  v
          JSONL metric system
```

The safety rule is simple: an invocation may only load state stored under its
exact layout identity. Unknown or incompatible layouts recompute.

The code follows the same three-component boundary:

- `graphkv.topology` loads the JSONL contract produced by a topology generator;
- `graphkv.runtime` turns each graph into ordered agent invocations and
  layout-scoped state decisions; and
- `graphkv.metrics` writes and summarizes execution records outside the
  runtime decision path.

`graphkv.runtime.manager`, the in-memory/TCP stores, and the `DynamicCache`
codec are internal runtime mechanisms. LMCache or Mooncake may replace the
state backend, but they are not required to exercise the architecture.

### Why use a TCP state server?

The TCP server is a mechanism-validation layer, not the research contribution
and not a performance model of a production network. Keeping all states in one
Python dictionary would avoid the main distributed-systems boundary: objects
would remain in one address space and would require neither a wire format nor
an explicit remote lookup. The independent server forces GraphKV to execute a
real cross-process state path:

```text
worker produces GPU KV state
        -> serialize tensors into a CPU byte payload
        -> publish the payload over TCP
        -> retain it in an independent process
        -> fetch it over TCP from another logical worker
        -> deserialize and copy tensors back to the GPU
```

This path validates state identity, remote lookup, versioned storage,
serialization, transfer, restoration, and the distinction between local hit,
remote hit, and recomputation. It also exposes costs that token-count analysis
cannot capture. In the corrected preliminary run, local-layout identity reduced
dense prefills from seven to five, but it did not reduce observed latency over
complete-graph isolation because the few miss, serialization, and synchronous
publication measurements were variable and expensive. Reuse is therefore not
automatically beneficial; a complete runtime must eventually compare remote
transfer cost with recomputation cost.

The current server runs on `127.0.0.1`, while the two logical serving instances
share one model process and one physical GPU. It proves that the control and
data path works across a process boundary, but it does not reproduce a physical
cluster network, independent GPU workers, congestion, RDMA, or GPUDirect. The
project's contribution is the layout-validity and reuse decision above this
transport. The TCP store can later be replaced by LMCache, Mooncake, or another
data plane without changing that decision boundary.

## Quick start: CPU control plane

The package itself has no mandatory dependencies:

```bash
git clone https://github.com/zhilishui/dynamic_graph_kv.git
cd dynamic_graph_kv
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

## CUDA GPU benchmark

The default benchmark uses Qwen2.5-0.5B in FP16, a 512-token prompt, batch size
one, and one shared model. It executes real transformer prefill and moves real
KV tensors; it is not a NumPy or sleep-based performance simulation.

### Hardware and software requirements

- Linux or WSL2 with Python 3.10 or newer;
- a CUDA-capable NVIDIA GPU and a working NVIDIA driver;
- approximately 8 GiB of free disk space for the environment and model;
- approximately 8 GiB of system memory recommended; and
- Internet access on the first run to download Python packages and the model.

The benchmark has been validated on an 8-GiB RTX 4060 Laptop GPU. Its measured
peak PyTorch-allocated GPU memory was 1,306 MiB, so the default 0.5B model and
512-token input are expected to fit on a typical 6-GiB RTX 4050. The 4050 has
not yet been measured by this project, and its absolute latency should not be
compared directly with the reported 4060 latency. If an out-of-memory error
occurs, retry with `GRAPHKV_PROMPT_TOKENS=256`.

The server's `--max-gib` option limits **CPU memory** used for stored payloads;
it is not a GPU-memory reservation.

### Fresh installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/zhilishui/dynamic_graph_kv.git
cd dynamic_graph_kv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build compatible with the machine's NVIDIA
driver, then install GraphKV's GPU dependencies. For the exact software stack
used for the reported results:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==4.50.2
python -m pip install -e .
```

Users with a different driver should select the appropriate CUDA wheel from
the official PyTorch installer rather than assuming that CUDA 12.6 is
compatible. Confirm that the environment can see the GPU and run the tests:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

The CUDA check must print `True`. On the first benchmark run, Transformers will
download Qwen2.5-0.5B-Instruct from Hugging Face.

### Reproduce all three policies

The supported launch command is:

```bash
./scripts/run_gpu_suite.sh
```

The script validates CUDA, runs `reset`, `whole_graph`, and `local_layout`, and
prints their combined summary. It intentionally starts an empty state server
for each policy, preventing state left by a previous run from turning initial
misses into remote hits. It also writes each policy to a different file:

```text
results/gpu-reset.jsonl
results/gpu-whole_graph.jsonl
results/gpu-local_layout.jsonl
```

The defaults reproduce the reported workload: 512 tokens, three repetitions,
round-robin placement, Qwen2.5-0.5B-Instruct, and model revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Optional environment variables
change the hardware-sized parameters without editing the script:

```bash
GRAPHKV_PROMPT_TOKENS=256 GRAPHKV_REPETITIONS=1 ./scripts/run_gpu_suite.sh
```

To replay a topology-generator output instead of the example trace:

```bash
GRAPHKV_TRACE=path/to/generated-trace.jsonl ./scripts/run_gpu_suite.sh
```

Each JSONL row must contain `request_id`, `active_roles`, and `edges`; see
`configs/example_trace.jsonl` for the generator/runtime contract.

To summarize previously generated result files without rerunning the model:

```bash
python scripts/summarize_results.py \
  results/gpu-reset.jsonl \
  results/gpu-whole_graph.jsonl \
  results/gpu-local_layout.jsonl
```

### Manual two-terminal launch

For debugging one policy, start a clean server in the first terminal:

```bash
# Terminal 1: independent state process
graphkv-server --host 127.0.0.1 --port 7648 --max-gib 1
```

Then run the benchmark in a second terminal:

```bash
python scripts/run_gpu_benchmark.py \
  --server 127.0.0.1:7648 \
  --trace configs/example_trace.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --prompt-tokens 512 \
  --repetitions 3 \
  --placement round_robin \
  --policy local_layout \
  --output results/gpu-local_layout.jsonl
```

Stop and restart the server before measuring another policy or repeating the
same policy. Results are JSON Lines and include the decision, layout, payload
size, dense-prefill time, acquisition time, CPU-to-GPU restore time, and two
clearly separated readiness metrics.
`round_robin` changes a role's logical serving instance across requests, so a
recurring layout exercises a real TCP remote hit; `stable` measures local
recurrence. The benchmark fails early if CUDA is not visible instead of
silently falling back to CPU.

`observed_state_ready_ms` is the primary implementation metric. It measures
wall-clock time from immediately before `manager.acquire()` until the selected
or reconstructed state is available to the caller on the GPU. In the current
synchronous prototype, a miss includes prefill, serialization, and publication
to the TCP server; a remote hit includes TCP retrieval, deserialization, and
CPU-to-GPU copy; and a local hit includes lookup and GPU-state selection.

`modeled_state_ready_ms` is a secondary analytical metric. For a miss it counts
only dense prefill, modeling a future implementation that makes publication
asynchronous. It must not be presented as the observed latency of the current
implementation. `synchronous_publication_ms` reports the miss-path work omitted
by that model. The summarizer computes p95 using the empirical nearest-rank
method, selecting rank `ceil(0.95 * N)`.

### Validated test environment

The preliminary result files were produced with the following environment:

| Component | Validated value |
|---|---|
| Operating system | Ubuntu 24.04.4 LTS under WSL2, Linux 6.18.33.2 |
| CPU | Intel Core i9-13900H, 20 logical CPUs visible to WSL2 |
| System memory | 7.6 GiB RAM and 2.0 GiB swap visible to WSL2 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB |
| NVIDIA driver | 566.26 |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| cuDNN | 9.5.1 |
| Transformers | 4.50.2 |
| Model | Qwen/Qwen2.5-0.5B-Instruct, FP16 |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Attention implementation | eager |
| Benchmark parameters | 512 tokens, batch 1, 2 warm-ups, 3 repetitions |
| State service | `127.0.0.1:7648`, localhost TCP, 1-GiB CPU-store limit |

The two logical serving instances shared one physical GPU and one loaded model
in this experiment. The table describes the machine used for the published
numbers, not a claim that every listed component is required.

### What one GPU can and cannot establish

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
