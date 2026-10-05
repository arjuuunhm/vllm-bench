# Baseline — default engine, before the harness

`vllm bench serve`, random dataset, 1024-token prompts / 256 output tokens, default engine.

| Concurrency | Mean TTFT | Mean TPOT | Output tok/s |
|---|---|---|---|
| 1 | 112 ms | 12.6 ms | 77 |
| 8 | 354 ms | 15.1 ms | 487 |

Raw: `~/dev/bench-results/baseline_c{1,8}.json`
