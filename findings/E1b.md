# E1b — KV-read cost vs context length (2026-10-02)

**Setup:** `experiments/e1b_kv_context.py`, default config, c ∈ {1, 8} fixed, prompt length
64–2048, 256 output tokens (`--ignore-eos`). Zero preemptions. Analysis: `python analyze_e1b.py`
("batch ctx tok" = c × (L + 128), the mean context over the decode).

| c | L | batch ctx tok | ITL p50 | TPOT p50 | ITL p99 | out tok/s |
|---|---|---|---|---|---|---|
| 1 | 64 | 192 | 12.27 | 12.29 | 13.04 | 81 |
| 1 | 2048 | 2,176 | 12.78 | 12.81 | 13.56 | 75 |
| 8 | 64 | 1,536 | 12.59 | 12.62 | 13.55 | 625 |
| 8 | 512 | 5,120 | 13.22 | 13.30 | 14.26 | 561 |
| 8 | 1024 | 9,216 | 13.93 | 14.58 | 15.06 | 495 |
| 8 | 1536 | 13,312 | 14.71 | 17.00 | 153 | 420 |
| 8 | 2048 | 17,408 | 15.38 | 18.35 | 197 | 376 |

Linear fits (median ITL): **c=8: 0.177 ms per 1k batch tokens ⇒ 831 GB/s**; c=1: 0.257 ms per 1k ⇒ 574 GB/s.

## Findings
1. **KV reads are pure bandwidth at c=8.** Predicted +2.8 ms from L=64→2048; measured +2.79 ms
   median ITL. Implied 831 GB/s ≈ the 849 GB/s `bwtest.py` number. Each token in the batch costs
   144 KiB/step, no more. ⇒ FP8 KV (E5) should halve this term.
2. **One sequence doesn't saturate bandwidth in attention** (574 GB/s at c=1, +0.51 ms vs +0.35
   predicted) — likely too few CTAs for one sequence × 8 KV heads. Small in absolute terms.
3. **At long context, prefill interference costs more than KV reads.** c=8, L=2048: median ITL
   +2.8 ms but TPOT +5.7 ms, and p99 ITL = 197 ms (13× median). With `--ignore-eos` and equal
   lengths, requests finish in waves, so ~8 × 2k-token prefills land together and stall decode
   steps. Throughput falls 40% (625 → 376 tok/s); KV reads are only ~half the TPOT growth,
   the rest is prefill stalls. This is exactly E2's question.

## Caveat — prefix-cache hits in "random" workloads (11–73%)
The random dataset builds prompt *i* from `offset_i + arange(L)` with offsets drawn from
`--seed 0`, so every sweep point reuses the same offsets: the L=1024 prompts start with the
L=512 prompts' tokens (hit ≈ 512/1024 ≈ 60% observed). Doesn't affect decode ITL (attention
still reads the full KV), but **understates TTFT and prefill stalls** here and in E1. Must fix
before E2/E3: per-point seed or `--no-enable-prefix-caching` (or restart server per point).
