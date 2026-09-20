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
each request. Keying reusable state by request or by the complete graph is
safe, but it discards useful state when an unrelated part of the graph changes.
Keying state only by agent identifier can reuse more state, but it can be
unsafe: the same agent may receive a different ordered set of predecessors,
prompt template, or placeholder layout in the new graph.

The project therefore asks:

> Can an agent-local prompt-layout identity preserve safe KV-state reuse across
> dynamic graph changes and reduce unnecessary dense prefills?

GraphKV constructs an identity from the state-relevant local structure,
including the model and template version, the consumer role, its ordered
predecessors, and the placeholder schema. A cached state is reused only when
this identity matches. The policy is compared with request-level reset and
complete-graph isolation.

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
reduce latency. In the corrected preliminary run, agent-local layout identity
reduced dense prefills from seven to five relative to complete-graph isolation,
but its observed total and p95 state-readiness times were not lower. The small
number of synchronous serialization and publication observations varied enough
to outweigh the saved prefills in that run. The result motivates, but does not
yet validate, a cost-aware choice between remote reuse and recomputation.

The current experiment uses localhost TCP. Its two logical serving instances
share one model process and one physical RTX 4060; only the state server is an
independent process. The experiment therefore validates a real cross-process
protocol and real tensor movement, but it cannot establish cross-machine
bandwidth, independent-worker contention, RDMA performance, or cluster-scale
throughput. The research contribution remains the identity and validity rule
for dynamic graphs. TCP is a replaceable data plane that can later be mapped to
LMCache, Mooncake, or another state-transfer system.

## Preliminary Experimental Setup

The experiment used the following configuration:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU with 8 GiB of memory;
- model: Qwen2.5-0.5B-Instruct in FP16 with eager attention;
- context length: 512 input tokens, batch size one;
- workload: five request-specific DAGs repeated three times;
- total agent invocations: 33 per policy;
- placement: two logical serving instances assigned round-robin by request;
- data path: an independent state server over localhost TCP; and
- serialized KV-state size: 6,309,443 bytes per state.

The maximum allocated GPU memory was approximately 1,306 MiB. The primary
latency metric measures the current synchronous implementation: a cache miss
includes real transformer prefill, serialization, and publication; a remote
hit includes TCP retrieval, deserialization, and CPU-to-GPU restoration; and a
local hit selects GPU-resident state.

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
| Reset per request | 63.982 ms | 145.747 ms | 2,111.406 ms |
| Complete-graph isolation | 34.003 ms | 239.574 ms | 1,122.083 ms |
| Agent-local layout identity | 36.192 ms | 277.285 ms | 1,194.345 ms |

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

All 99 state acquisitions passed the embedded layout-identity and prompt-token
checks. Under the layout-aware policy, the median observed miss path was
197.885 ms, while the median localhost remote-hit path was 27.398 ms. The
median GPU-local path was 0.062 ms.

The benchmark also records `modeled_state_ready_ms` as a secondary metric. For
a miss, this value counts only dense prefill and models a future design in
which state publication is asynchronous. It is not used as the primary latency
result because publication is synchronous in the current implementation.

For this trace, agent-local layout identity:

- eliminated two of the seven dense prefills required by complete-graph
  isolation;
- increased GPU-local hits from 19 to 23; but
- produced 6.4% higher summed observed state-readiness time than complete-graph
  isolation in this run.

The latency result requires caution. Layout-aware reuse produced a higher p95
than complete-graph isolation: 277.285 ms versus 239.574 ms. Its relatively few
miss, serialization, and synchronous-publication observations also took longer
in this run. With only 33 invocations per policy and a fixed policy execution
order, these differences cannot separate policy effects from runtime
variation. The current data support a reduction in repeated prefill work, but
they do not yet demonstrate an observed mean, aggregate, or tail-latency
improvement over complete-graph isolation.

## Conclusions Supported by the Current Evidence

### 1. The mechanism is feasible on one RTX 4060

A single 8-GiB RTX 4060 is sufficient to build and exercise the core mechanism
with real transformer KV tensors. It can validate layout identities, local and
remote state paths, serialization, compatibility checks, and recomputation.
Neither RDMA nor a multi-node cluster is necessary to answer the core research
question.

### 2. Complete-graph identity can discard valid reuse

The experiment includes graph transitions in which the complete graph changes
but some agents retain the same local prompt layout. Complete-graph isolation
treats these states as different, whereas agent-local layout identity safely
recognizes the unchanged layouts. The reduction from seven to five dense
prefills directly demonstrates this otherwise hidden reuse opportunity in the
tested trace.

### 3. Agent identifier alone is not a sufficient correctness condition

An agent's reusable state depends on its local input structure, not only its
name. The implemented identity makes the validity condition explicit and
checks both the structural digest and prompt tokens before accepting restored
state. This separates safe reuse from accidental reuse across incompatible
graph layouts.

### 4. Remote reuse is not automatically beneficial

In the current localhost experiment, restoring state remotely was faster than
recomputing a 512-token prefix at the median. This relationship is not
universal. State size, prompt length, network bandwidth, serialization cost,
and model speed determine whether transfer is preferable to recomputation. A
complete system should therefore support a cost-aware reuse decision rather
than transferring every matching state unconditionally.

### 5. One GPU is enough for the course project, but it limits the claim

One GPU can support a complete course project if the contribution is framed as
a new state-validity and reuse mechanism for dynamic graphs. Multiple logical
instances and an independent state process are sufficient to test the
distributed control path. The experiment cannot establish production
multi-GPU throughput, RDMA performance, or data-center scalability because the
logical model instances currently share one physical GPU and one model
process.

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

> The prototype demonstrates that agent-local prompt layouts can remain safely
> reusable when the complete multi-agent graph changes. On one RTX 4060 and a
> small controlled trace, layout-granular state management reduced redundant
> dense prefills relative to complete-graph isolation, but it did not improve
> observed latency in the corrected single run. Repeated counterbalanced
> trials, full KVCOMM integration, and broader task-level evaluation are still
> required to determine the general end-to-end benefit.

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
safer and less wasteful state-management boundary than either request-level
reset, complete-graph isolation, or agent-ID-only reuse for dynamic
multi-agent inference.
