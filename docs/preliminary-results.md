# Preliminary RTX 4060 Results

These measurements are a smoke-scale validation of the GraphKV mechanism, not
the final course-project evaluation.

## Configuration

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB
- Model: Qwen2.5-0.5B-Instruct, FP16, eager attention
- Prompt: 512 tokens, batch size 1
- Trace: five controlled synthetic request-specific DAGs repeated three times
- Agent invocations: 33 per policy
- Placement: two logical serving instances, round-robin across requests
- Data plane: an independent GraphKV TCP server on localhost
- KV payload: 6,309,443 bytes per 512-token state
- Peak allocated GPU memory: at most 1,246 MiB

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
| Reset per request | 33 | 0 | 0 | 106.745 ms | 3,522.580 ms |
| Whole-graph isolation | 7 | 7 | 19 | 64.415 ms | 2,125.692 ms |
| Agent-local layout identity | 5 | 5 | 23 | 69.716 ms | 2,300.644 ms |

All 99 state acquisitions passed layout-identity and exact prompt-token checks.
Under the layout policy, the median observed miss path was 535.774 ms, the
median localhost remote-hit path was 43.716 ms, and the median local-hit path
was 0.029 ms. On this short trace, local layout identity eliminated two
additional prefills relative to whole-graph isolation. Its summed observed
state-ready time was nevertheless 8.2% higher than whole-graph isolation,
although 34.7% lower than reset, in this run.

Nearest-rank observed p95 was 148.162 ms for reset, 276.158 ms for whole-graph
isolation, and 545.085 ms for local-layout identity. Thus the additional reuse
did not improve mean, total, or p95 over whole-graph isolation in this run.
Large variation across our preliminary runs, the small sample, and fixed policy
order mean that these latency differences are observations, not a general
performance claim. Multiple independent, counterbalanced trials are required.

## Interpretation and limits

This result establishes that the implementation works on the available 4060:
real transformer KV tensors cross a process boundary and return to the GPU,
while graph-local identity finds a reuse opportunity that a complete-graph key
misses when prompt tokens remain identical. It does **not** yet establish the
final research claim:

- both logical model instances share one physical GPU and one model process;
- localhost TCP is not a multi-node network;
- prompts encode graph structure but do not yet run the full KVCOMM anchor
  approximation or downstream agent generation;
- the trace is intentionally small and favorable enough to exercise every
  transition;
- these are state-readiness measurements, not full TTFT or end-to-end latency.

The raw JSONL files and sidecar summaries are tracked under `results/`. Each
summary records the benchmark source hash, Git revision, trace hash, model
revision, parameters, environment, GPU peak, and aggregate metrics.

The next experiments must integrate the identity into KVCOMM, use benchmark
tasks and generated graph traces, compare output accuracy, vary topology churn
and prompt length, and include confidence intervals across repeated trials.
