# Preliminary RTX 4060 Results

These measurements are a smoke-scale validation of the GraphKV mechanism, not
the final course-project evaluation.

## Configuration

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB
- Model: Qwen2.5-0.5B-Instruct, FP16, eager attention
- Prompt: 512 tokens, batch size 1
- Trace: five request-specific DAGs repeated three times
- Agent invocations: 33 per policy
- Placement: two logical serving instances, round-robin across requests
- Data plane: an independent GraphKV TCP server on localhost
- KV payload: 6,309,443 bytes per 512-token state
- Peak allocated GPU memory: at most 1,306 MiB

`observed_state_ready_ms` measures the current implementation's wall-clock time
from immediately before state acquisition until the state is available to the
caller on the GPU. A miss includes prefill, serialization, and synchronous TCP
publication. A remote hit includes TCP fetch, deserialization, and CPU-to-GPU
restoration. A local hit uses the logical instance's GPU-resident state.

`modeled_state_ready_ms` is reported separately for analysis. On a miss, it
counts only dense prefill and therefore models a future asynchronous
publication path; it is not the observed latency of this prototype. Percentile
values use the empirical nearest-rank definition.

## Results

| Policy | Recompute | Remote hit | Local hit | Mean observed ready time | Total observed ready time |
|---|---:|---:|---:|---:|---:|
| Reset per request | 33 | 0 | 0 | 63.982 ms | 2,111.406 ms |
| Whole-graph isolation | 7 | 7 | 19 | 34.003 ms | 1,122.083 ms |
| Agent-local layout identity | 5 | 5 | 23 | 36.192 ms | 1,194.345 ms |

All 99 state acquisitions passed embedded layout-identity and prompt-token
checks. Under the layout policy, the median observed miss path was 197.885 ms,
the median localhost remote-hit path was 27.398 ms, and the median local-hit
path was 0.062 ms. On this short trace, local layout identity eliminated two
additional prefills relative to whole-graph isolation. However, its summed
observed state-ready time was 6.4% higher than whole-graph isolation because
the small number of miss and publication measurements varied substantially.

Nearest-rank observed p95 was 145.747 ms for reset, 239.574 ms for whole-graph
isolation, and 277.285 ms for local-layout identity. The layout policy therefore
did not improve observed mean, total, or p95 latency over whole-graph isolation
in this run. Multiple independent, counterbalanced trials are required before
making a latency claim.

## Interpretation and limits

This result establishes that the implementation works on the available 4060:
real transformer KV tensors cross a process boundary and return to the GPU,
while graph-local identity finds reuse that a complete-graph key misses. It
does **not** yet establish the final research claim:

- both logical model instances share one physical GPU and one model process;
- localhost TCP is not a multi-node network;
- prompts encode graph structure but do not yet run the full KVCOMM anchor
  approximation or downstream agent generation;
- the trace is intentionally small and favorable enough to exercise every
  transition;
- these are state-readiness measurements, not full TTFT or end-to-end latency.

The next experiments must integrate the identity into KVCOMM, use benchmark
tasks and generated graph traces, compare output accuracy, vary topology churn
and prompt length, and include confidence intervals across repeated trials.
