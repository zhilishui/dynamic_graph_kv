# Current Conclusions: Layout-Aware KV-State Reuse for Dynamic Multi-Agent Graphs

## Status and Scope

This report summarizes the conclusions currently supported by the GraphKV
prototype and its preliminary experiment on a single NVIDIA RTX 4060 Laptop
GPU. The experiment is a mechanism and feasibility study. It is not yet the
final evaluation of the course project and should not be interpreted as a
claim about production-scale multi-GPU performance.

## Research Problem

KVCOMM reduces repeated LLM computation by reusing KV state across agents.
However, a multi-agent system may select a different communication graph for
each request. Resetting state for every request is safe but discards reuse;
isolating structural state by complete graph also fragments it when an
unrelated part of the graph changes. Keying only by agent identifier is too
coarse: the same agent may receive a different ordered set of predecessors,
prompt template, or placeholder layout in the new graph.

The project therefore asks:

> Can an agent-local prompt-layout identity expose reusable state across
> dynamic graph changes, and can KVCOMM use that namespace without changing
> task outputs?

GraphKV constructs an identity from the state-relevant local structure,
including the model and template version, the consumer role, its ordered
predecessors, and the placeholder schema. In the current exact-cache benchmark,
prompt token IDs must also match; layout identity alone does not make a full KV
cache reusable across changed message values. The policy is compared with
request-level reset and complete-graph isolation.

## Implemented Prototype

The current prototype contains:

- validated request-specific DAGs and deterministic local-layout derivation;
- reset, complete-graph, and agent-local-layout state policies;
- a versioned and byte-bounded in-memory LRU state store;
- an independent TCP state-server process;
- local-hit, remote-hit, recompute, and rejected-state control paths;
- serialization and restoration of real Hugging Face `DynamicCache` tensors;
- embedded layout and token checks that reject incompatible state; and
- unit and integration tests for identity, topology, storage, networking, and
  cache serialization.

This is distributed-systems implementation work rather than an offline NumPy
optimization study: it defines state identity and validity, coordinates state
between independent processes, handles version conflicts, and makes online
reuse-versus-recompute decisions.

## Role of the TCP Prototype

The TCP state server is validation infrastructure, not the project's proposed
novelty. An in-process dictionary could demonstrate that two equal keys retrieve
the same Python object, but it would bypass the distributed-systems boundary.
An independent server requires the prototype to turn GPU state into an explicit
wire payload, publish it outside the worker's address space, locate it by a
stable identity, and reconstruct it for another logical worker:

```text
GPU DynamicCache
    -> CPU serialization
    -> TCP publication
    -> independent CPU-memory store
    -> TCP retrieval
    -> deserialization and CPU-to-GPU restoration
```

This implementation makes three system costs observable: local lookup, remote
transfer and restoration, and recomputation. That distinction matters because
a policy that increases the number of reusable states does not necessarily
reduce latency. In the current preliminary run, agent-local layout identity
reduced dense prefills from seven to five relative to complete-graph isolation,
but its summed state-readiness time was 8.2% higher and its p95 was also worse.
Measured latency varied substantially across preliminary runs. The result
motivates, but does not yet validate, a cost-aware choice between remote reuse
and recomputation.

The current experiment uses localhost TCP. Its two logical serving instances
share one model process and one physical RTX 4060; only the state server is an
independent process. The experiment therefore validates a real cross-process
protocol and real tensor movement, but it cannot establish cross-machine
bandwidth, independent-worker contention, RDMA performance, or cluster-scale
throughput. The proposed contribution remains the layout-scoped namespace and
reuse decision for dynamic graphs. TCP is a replaceable data plane that can
later be mapped to LMCache, Mooncake, or another state-transfer system.

## Preliminary Experimental Setup

The experiment used the following configuration:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU with 8 GiB of memory;
- model: Qwen2.5-0.5B-Instruct in FP16 with eager attention;
- context length: 512 input tokens, batch size one;
- workload: five controlled synthetic request-specific DAGs repeated three times;
- total agent invocations: 33 per policy;
- placement: two logical serving instances assigned round-robin by request;
- data path: an independent state server over localhost TCP; and
- serialized KV-state size: 6,309,443 bytes per state.

The maximum allocated GPU memory was approximately 1,246 MiB. The primary
latency metric measures the current synchronous implementation: a cache miss
includes real transformer prefill, serialization, and publication; a remote
hit includes TCP retrieval, deserialization, and CPU-to-GPU restoration; and a
local hit selects GPU-resident state.

The repository tracks the three raw JSONL files and their sidecar summaries.
Each summary records the benchmark source hash, Git revision, trace hash, model
revision, parameters, software environment, GPU peak, and aggregate metrics.

## Results

### How to read the results

Each policy runs exactly the same workload: 33 agent invocations. An agent
invocation is one point at which an agent needs a valid KV state before it can
continue inference. The state manager must handle every invocation in exactly
one of three ways:

- **Dense prefill:** No reusable state is available, so the model recomputes
  the KV state from the 512-token prompt. This is the most expensive path.
- **Remote hit:** A matching state exists in the independent state-server
  process. The system retrieves it over localhost TCP, deserializes it, and
  copies it back to the GPU.
- **Local hit:** A matching state is already resident in the logical serving
  instance's GPU cache. Only a lookup is needed.

For every row, `dense prefills + remote hits + local hits = 33`. The three
policies differ only in the rule used to decide whether an existing state is
valid:

- **Reset per request** forbids reuse across requests. Every invocation must
  recompute its state.
- **Complete-graph isolation** reuses state only when both the complete graph
  and the agent role match. A change anywhere in the graph creates a different
  key, even if this particular agent's input did not change.
- **Agent-local layout identity**, our policy, reuses state when this agent's
  own input layout matches. Unrelated changes elsewhere in the graph do not
  invalidate it.

### How often each path was used

| Policy | Recomputed from prompt | Retrieved over TCP | Reused on local GPU | Total invocations |
|---|---:|---:|---:|---:|
| Reset per request | 33 | 0 | 0 | 33 |
| Complete-graph isolation | 7 | 7 | 19 | 33 |
| Agent-local layout identity | 5 | 5 | 23 | 33 |

The key comparison is between the last two rows. Complete-graph isolation
performed seven dense prefills. The layout-aware policy recognized that two of
those states were still valid because the corresponding agents' local inputs
had not changed, even though another part of the graph had changed. It
therefore reduced dense prefills from seven to five and increased GPU-local
hits from 19 to 23.

### Observed time required to make the KV state ready

| Policy | Average per invocation | Nearest-rank p95 | Sum across 33 invocations |
|---|---:|---:|---:|
| Reset per request | 106.745 ms | 148.162 ms | 3,522.580 ms |
| Complete-graph isolation | 64.415 ms | 276.158 ms | 2,125.692 ms |
| Agent-local layout identity | 69.716 ms | 545.085 ms | 2,300.644 ms |

These columns measure **state-readiness time**, not full request latency:

- **Average** is the mean time required to obtain a valid state across all 33
  invocations.
- **Nearest-rank p95** sorts all observations and selects one-based rank
  `ceil(0.95 * 33) = 32`.
- **Sum** adds the state-readiness time of all 33 invocations and is useful for
  comparing how much work the complete trace required.

The timer starts immediately before `manager.acquire()` and ends after the
selected or reconstructed state is available to its caller on the GPU. For a
miss, it includes dense prefill, serialization, and synchronous publication to
the state server. For a remote hit, it includes TCP retrieval, deserialization,
and CPU-to-GPU restoration. For a local hit, it includes lookup and selection
of the GPU-resident state. It does not include subsequent token generation,
the rest of the multi-agent workflow, or final answer generation. It is
therefore neither full TTFT nor end-to-end request latency.

All 99 state acquisitions passed the layout-identity and exact prompt-token
checks. Under the layout-aware policy, the median observed miss path was
535.774 ms, while the median localhost remote-hit path was 43.716 ms. The
median GPU-local path was 0.029 ms. These checks establish exact-cache
compatibility for this fixed-token trace, not KVCOMM correctness under changing
message values.

The benchmark also records `modeled_state_ready_ms` as a secondary metric. For
a miss, this value counts only dense prefill and models a future design in
which state publication is asynchronous. It is not used as the primary latency
result because publication is synchronous in the current implementation.

For this trace, agent-local layout identity:

- eliminated two of the seven dense prefills required by complete-graph
  isolation;
- increased GPU-local hits from 19 to 23; and
- produced 8.2% higher summed observed state-readiness time than complete-graph
  isolation, although 34.7% lower than reset, in this run.

The latency result requires caution. Layout-aware reuse had higher mean, total,
and p95 time than complete-graph isolation in this run; its p95 was 545.085 ms
versus 276.158 ms. With only 33 invocations per policy, a fixed execution order,
and substantial variation across preliminary runs, these measurements cannot
separate policy effects from runtime variation. The data support a reduction
in repeated prefill work; they do not yet establish a general latency
improvement.

## Conclusions Supported by the Current Evidence

### 1. The mechanism is feasible on one RTX 4060

A single 8-GiB RTX 4060 is sufficient to build and exercise the mechanism with
real transformer KV tensors. It can validate layout identities, local and
remote state paths, serialization, compatibility checks, and recomputation.
It cannot by itself establish multi-GPU or production-network performance.

### 2. Complete-graph identity can discard valid reuse

The experiment includes graph transitions in which the complete graph changes
but some agents retain the same local prompt layout. Complete-graph isolation
treats these states as different, whereas agent-local layout identity recognizes
the unchanged structure. Because this controlled trace also keeps prompt tokens
identical for equal layouts, the reduction from seven to five dense prefills
demonstrates a reuse opportunity, not yet general cross-context KVCOMM reuse.

### 3. Agent identifier alone is not a sufficient correctness condition

An agent's state namespace depends on its local input structure, not only its
name. The prototype checks both the structural digest and exact prompt tokens
before inference. The future KVCOMM integration must additionally validate
that its approximation remains accurate when message values differ.

### 4. Remote reuse is not automatically beneficial

In the current localhost experiment, restoring state remotely was faster than
recomputing a 512-token prefix at the median. This relationship is not
universal. State size, prompt length, network bandwidth, serialization cost,
and model speed determine whether transfer is preferable to recomputation. A
complete system should therefore support a cost-aware reuse decision rather
than transferring every matching state unconditionally.

### 5. One GPU is enough for mechanism evaluation, but it limits the claim

One GPU can exercise the proposed layout namespace and distributed control
path using multiple logical instances and an independent state process. A
second GPU would materially strengthen the remote-worker evaluation. The
current experiment cannot establish production multi-GPU throughput, RDMA
performance, or data-center scalability because both logical model instances
share one physical GPU and one model process.

## What Has Not Yet Been Established

The present results do not yet show that GraphKV improves full application
TTFT, end-to-end latency, throughput, or task quality in general. In
particular:

- GraphKV has not yet replaced KVCOMM's actual prefix and anchor dictionaries;
- the workload uses structural prompts rather than a full multi-agent task;
- the graph trace is small and intentionally exposes reusable transitions;
- the test uses localhost TCP rather than cross-GPU or cross-machine traffic;
- both logical serving instances share a physical model process;
- task correctness and output quality have not yet been evaluated; and
- the reported measurements do not include confidence intervals across many
  independent runs.

Consequently, the defensible current claim is:

> On a controlled fixed-token trace, the prototype demonstrates that
> agent-local structural identity exposes reuse opportunities hidden by
> complete-graph isolation. It also validates real KV serialization and
> cross-process TCP restoration on one RTX 4060. It does not yet establish safe
> cross-context KVCOMM reuse or a general latency improvement; those claims
> require full KVCOMM integration, generated workloads, and repeated
> counterbalanced trials.

## Work Required for the Final Conclusion

The remaining project should integrate layout identities with KVCOMM's prefix
and anchor state, use realistic dynamically generated graph traces, and test
real multi-agent tasks. The evaluation should vary topology churn, prefix
length, cache capacity, placement, and simulated network conditions. It should
report dense-prefill count, reuse rate, transferred bytes, TTFT, end-to-end
latency, and task correctness, with repeated runs and confidence intervals.
It should also identify negative regimes, especially low layout recurrence,
short prefixes, and cases in which transfer costs more than recomputation.

If these experiments confirm the preliminary behavior without changing model
outputs, the final conclusion can be that agent-local layout identity is a
useful state-management namespace for KVCOMM under dynamic multi-agent graphs,
with explicit regimes in which local reuse, transfer, or recomputation wins.
