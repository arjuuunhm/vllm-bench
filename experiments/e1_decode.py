# E1: Where does decode time go?
# Decode-dominated workload (64-token prompts, 256 forced output tokens) swept over
# concurrency. Compares the default engine against no CUDA graphs / torch.compile
# (--enforce-eager) and against synchronous scheduling (--no-async-scheduling).
#
# Expectation: TPOT stays near-flat with batch size while decode is weight-bandwidth
# bound (8.8 GB weights / ~896 GB/s ≈ 9.8 ms floor), then rises once compute-bound.
# The gap between measured TPOT at c=1 and the floor is overhead (CPU, launches,
# non-GEMM kernels, KV reads).

EXPERIMENT = {
    "name": "e1_decode",
    "model": "Qwen/Qwen3-8B-FP8",
    "base_args": ["--max-model-len", "8192", "--gpu-memory-utilization", "0.88"],
    "configs": {
        "default": [],
        "no_async": ["--no-async-scheduling"],
        "eager": ["--enforce-eager"],
    },
    "workloads": [
        {
            "name": "decode",
            "args": [
                "--dataset-name", "random",
                "--random-input-len", "64",
                "--random-output-len", "256",
                "--ignore-eos",
                "--num-warmups", "4",
            ],
            "sweep": {"max-concurrency": [1, 2, 4, 8, 16, 32, 64]},
            # 4 full waves at each concurrency, at least 16 requests.
            "num_prompts": lambda c: max(16, 4 * c),
        },
    ],
}
