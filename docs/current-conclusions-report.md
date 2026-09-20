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

The maximum allocated GPU memory was approximately 1,306 MiB. A cache miss
measures real transformer prefill. A remote hit includes TCP retrieval,
deserialization, and CPU-to-GPU restoration. A local hit uses GPU-resident
state. Synchronous publication cost is recorded separately because it is not
on the current request's state-readiness path.

## Results

| Policy | Dense prefills | Remote hits | Local hits | Mean state-readiness time | p95 | Summed time |
|---|---:|---:|---:|---:|---:|---:|
| Reset per request | 33 | 0 | 0 | 48.085 ms | 67.262 ms | 1,586.812 ms |
| Complete-graph isolation | 7 | 7 | 19 | 14.686 ms | 44.951 ms | 484.648 ms |
| Agent-local layout identity | 5 | 5 | 23 | 11.315 ms | 47.707 ms | 373.405 ms |

All 99 state acquisitions passed the embedded layout-identity and prompt-token
checks. Under the layout-aware policy, the median dense-prefill path was
47.707 ms, while the median localhost remote-hit path was 20.890 ms. A
GPU-local hit took approximately 0.002 ms at the state-manager boundary.

For this trace, agent-local layout identity:

- eliminated two of the seven dense prefills required by complete-graph
  isolation;
- reduced summed state-readiness time by 23.0% relative to complete-graph
  isolation; and
- reduced summed state-readiness time by 76.5% relative to resetting state for
  every request.

The p95 result requires caution. Layout-aware reuse produced a slightly higher
p95 than complete-graph isolation in this small sample, even though it lowered
the number of prefills, mean time, and total time. The current data therefore
do not demonstrate a general tail-latency improvement.

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
> dense prefills and summed state-readiness time relative to complete-graph
> isolation. A full KVCOMM integration and broader task-level evaluation are
> still required to determine the general end-to-end benefit.

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
