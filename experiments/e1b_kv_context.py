# E1b: Does KV-read traffic dominate TPOT at long context?
# Fixed concurrency, sweep prompt length. (Sweeping concurrency with 2k prompts doesn't
# work: ~2.3k tokens/request × 24,240-token KV cache ⇒ at most ~10 running, so c≥16
# would measure queueing/preemption, not batch size.)
#
# Prediction if attention KV reads are pure bandwidth: each token in the batch's
# context costs 144 KiB/step ⇒ ~0.18 ms per 1k context tokens at ~800 GB/s.
#   c=8: L=64 → 2048 adds ~8 × 1984 tokens ≈ +2.9 ms TPOT.
#   c=1: same sweep adds only ~0.36 ms (control: KV traffic scales with c × L).
# Analyze with: python analyze_e1b.py  (fits median ITL vs total batch context tokens
# and reports implied bandwidth). Use median ITL, not TPOT: TPOT includes steps where
# a new request's prefill is mixed into the decode batch.
#
# KV budget check at c=8, L=2048: 8 × (2048 + 256) = 18.4k < 24,240 tokens.

EXPERIMENT = {
    "name": "e1b_kv_context",
    "model": "Qwen/Qwen3-8B-FP8",
    "base_args": ["--max-model-len", "8192", "--gpu-memory-utilization", "0.88"],
    "configs": {
        "default": [],
    },
    "workloads": [
        {
            "name": f"ctx_c{c}",
            "args": [
                "--dataset-name", "random",
                "--random-output-len", "256",
                "--ignore-eos",
                "--num-warmups", "4",
                "--max-concurrency", str(c),
            ],
            "sweep": {"random-input-len": [64, 512, 1024, 1536, 2048]},
            # 4 full waves at c=8; 16 sequential requests at c=1.
            "num_prompts": lambda L, c=c: max(16, 4 * c),
        }
        for c in (1, 8)
    ],
}
