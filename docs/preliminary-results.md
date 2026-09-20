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

For a cache miss, `critical_path_ms` is the actual dense prefill time because
the newly generated state is already on the GPU. Publication/serialization is
reported separately. For a remote hit, the critical path includes TCP fetch,
deserialization, and CPU-to-GPU restoration. A local hit uses the logical
instance's GPU-resident state.

## Results

| Policy | Recompute | Remote hit | Local hit | Mean critical path | Total critical path |
|---|---:|---:|---:|---:|---:|
| Reset per request | 33 | 0 | 0 | 48.085 ms | 1586.812 ms |
| Whole-graph isolation | 7 | 7 | 19 | 14.686 ms | 484.648 ms |
| Agent-local layout identity | 5 | 5 | 23 | 11.315 ms | 373.405 ms |

All 99 state acquisitions passed embedded layout-identity and prompt-token
checks. Under the layout policy, the median miss path was 47.707 ms and the
median localhost remote-hit path was 20.890 ms. On this short trace, local
layout identity eliminated two additional prefills relative to whole-graph
isolation and reduced summed critical-path time by 23.0%. It reduced summed
time by 76.5% relative to reset.

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

