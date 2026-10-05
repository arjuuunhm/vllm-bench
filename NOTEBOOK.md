# vLLM engine optimization notebook

Local experiments on vLLM serving performance (TTFT, TPOT, chunked prefill,
prefix caching, KV offload, P/D disaggregation) on a single consumer GPU.

This file covers **how to bring up the model and run experiments**, and **the list of
experiments**. Results and findings live in [`findings/`](findings/), one file per experiment;
raw data lives in `results/<exp>/`.

---

## 1. Environment

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB GDDR7, 896 GB/s spec / **849 GB/s measured read** (`bwtest.py`; copy 770), compute capability 12.0 (sm_120) |
| Host | WSL2 (Ubuntu 26.04), 16 cores, 15 GB RAM visible to Linux, driver 591.86 |
| vLLM | `~/dev/vllm` @ `0cbac6cd13` (0.30.1rc1.dev558), editable install |
| Python env | `~/dev/vllm/.venv`, Python 3.12, torch 2.13.0+cu132 |
| Model | `Qwen/Qwen3-8B-FP8` (8.8 GiB weights) |
| Serve flags | `--max-model-len 8192 --gpu-memory-utilization 0.88` |
| KV cache | 3.33 GiB → **24,240 tokens** (~144 KiB/token: 36 layers × 8 KV heads × 128 dim × K+V × 2 B) |

---

## 2. Bringing up the model

```bash
cd ~/dev/vllm && source .venv/bin/activate
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 8192 --gpu-memory-utilization 0.88
```

- `activate` exports `CUDA_HOME=~/.local/cuda-13.2`, adds its `bin` to `PATH`, and sets
  `VLLM_USE_DEEP_GEMM=0` — always serve from the activated venv.
- Python changes under `vllm/` take effect on restart (no rebuild; kernels are precompiled).
- Startup with a warm compile cache: 20–56 s.
- Only one 8B-FP8 instance fits in VRAM. Before starting, check the GPU is idle
  (~0.8 GB used by Windows is normal).

### Setup decisions and fixes (in order encountered)
1. **Precompiled kernels** (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`). Planned work is all
   Python (`vllm/v1/...`); a full CUDA build would likely OOM with 15 GB RAM.
2. **FP8 model.** BF16 Qwen3-8B is ~16.4 GB of weights — doesn't fit in 16 GB VRAM.
3. **No C compiler** → Triton/inductor failed. Fixed with `sudo apt install build-essential python3-dev`
   (the "Failed to connect to system scope bus" message during install is a harmless WSL/systemd warning).
4. **DeepGEMM JIT failed** (needs a CUDA toolkit; targets sm90/sm100 anyway).
   Disabled with `VLLM_USE_DEEP_GEMM=0` → vLLM selects `CutlassFp8BlockScaledMMKernel`.
5. **FlashInfer sampler JIT needs nvcc.** No system CUDA toolkit, so used NVIDIA's pip wheels
   already in the venv. Two gotchas:
   - All compiler wheels must match torch's CUDA runtime (13.2): `nvidia-cuda-nvcc`, `nvidia-nvvm`,
     `nvidia-cuda-crt`, `nvidia-cuda-cccl` pinned to `13.2.*`. Mixed 13.2/13.4 gave
     "Unsupported .version 9.4" (ptxas) and "CUDA compiler and toolkit headers are incompatible".
   - The wheel layout lacks `lib64/libcudart.so` and `libcuda.so`. Built a symlink shim at
     `~/.local/cuda-13.2` (bin/include/nvvm → wheels; `lib64/libcudart.so` → `libcudart.so.13`;
     `lib64/stubs/libcuda.so` → `/usr/lib/wsl/lib/libcuda.so.1`).
6. `.venv/bin/activate` now exports `CUDA_HOME=~/.local/cuda-13.2`, adds its `bin` to `PATH`,
   and sets `VLLM_USE_DEEP_GEMM=0`.

> If the venv is ever recreated: re-pin the four compiler wheels to 13.2 and re-append the
> exports to `activate`. The shim links point into the venv and will work again.

---

## 3. Running experiments (harness)

```
~/dev/vllm-bench/
  harness.py              # launch server per config → run bench sweep → save → tear down
  summarize.py            # results/<exp> → markdown table + summary.csv
  experiments/<exp>.py    # EXPERIMENT dict: configs (server flags) × workloads (bench args + sweep)
  results/<exp>/<config>/ # <workload>__<key>=<val>.json/.log, server.log, meta.json
  findings/<ID>.md        # write-up per experiment: setup, tables, findings, caveats
```

```bash
source ~/dev/vllm/.venv/bin/activate && cd ~/dev/vllm-bench
python harness.py experiments/e1_decode.py [--only CONFIG ...] [--force]
python summarize.py results/e1_decode
```

- Each config gets a fresh server (launch-time flags can't change on a live server; only one
  instance fits in VRAM; fresh server = empty prefix cache = fair comparison).
- Teardown: SIGINT to the process group, SIGKILL after 60 s, then wait for GPU memory < 2 GB.
- Existing results are skipped unless `--force`, so interrupted runs resume.
- Every result JSON gets a `harness` block: bench command, wall time, and **deltas of `/metrics`
  counters** (prefix cache queries/hits, external (offload) cache hits, preemptions, token counts).
- Bench defaults: p50/p90/p99 for TTFT/TPOT/ITL/E2EL, `--seed 0`.

### Known harness issues
- **Cross-point prefix-cache hits.** The random dataset with a fixed `--seed 0` reuses prompt
  offsets across sweep points on the same server, so later points get 11–73% prefix hits.
  Understates TTFT/prefill cost. Fix before E2/E3 (per-point seed, `--no-enable-prefix-caching`,
  or restart per point). Details: [findings/E1b.md](findings/E1b.md#caveat--prefix-cache-hits-in-random-workloads-1173).

### Logging a new experiment
1. Add `experiments/<exp>.py`, run it through the harness.
2. Write `findings/<ID>.md`: date, setup (script, workload, configs), results table, findings, caveats.
3. Update the **Status** column in §4 and the next steps in §5.

---

## 4. Experiments

Constraints that shape everything: decode is **weight-bandwidth bound**, and the **KV cache is
small** (~24k tokens).

Workloads: **chat** (1k in / 256 out), **rag** (6k / 128), **shared-prefix**
(`--dataset-name prefix_repetition`, N × 4k prefixes + 200-token suffix), **mixed** (steady
short chats + occasional long prompts). Sweep `--request-rate` (+ `--burstiness`), report p50/p99.

| ID | Question | Knobs | Status |
|---|---|---|---|
| — | Baseline: default engine, chat workload | `vllm bench serve`, c ∈ {1, 8} | ✅ [findings](findings/baseline.md) |
| E1 | Where does decode time go? | `experiments/e1_decode.py`: concurrency 1–64; `--enforce-eager`; `--no-async-scheduling` | ✅ [findings](findings/E1.md) |
| E1b | Does KV-read traffic dominate TPOT at long context? | `experiments/e1b_kv_context.py`: c ∈ {1, 8} fixed, prompt len 64–2048 (a c-sweep at 2k can't run: cache fits only ~10 such requests). `analyze_e1b.py` fits median ITL vs batch context tokens → implied KV bandwidth; ~849 GB/s ⇒ pure bandwidth | ✅ [findings](findings/E1b.md) |
| E1p | What is the ~2.6 ms gap between measured TPOT (12.3 ms) and the 9.7 ms measured-bandwidth floor at c=1? | torch profiler trace: small kernels (norms, RoPE, sampling) vs idle time between steps | todo |
| E2 | Chunked prefill: how much does a bigger chunk cut TTFT, and how much does it hurt p99 TPOT of running requests? | `--max-num-batched-tokens` 512–8192, chunking off, `--long-prefill-token-threshold` on **mixed** | todo |
| E3 | Prefix caching: how fast does hit rate fall as the set of shared prefixes outgrows the ~24k-token cache? | prefixes = 0.5×/1×/2×/4× KV capacity, caching on/off | todo |
| E4 | CPU KV offload: at what prompt length does loading KV from CPU beat recomputing it? | `--kv-offloading-backend native --kv-offloading-size {2,4,8}` (keep ≤6 GB or raise WSL `memory=`) | todo |
| E5 | FP8 KV cache: 2× tokens in cache — what does it cost in accuracy? | `--kv-cache-dtype fp8`; watch preemptions; quick gsm8k | todo |
| E6 | Speculative decoding (optional): how much does it cut TPOT? | `--speculative-config` ngram, then EAGLE3 | todo |
| P/D | Disaggregation mechanics on one GPU | 2 instances via `--kv-transfer-config` + `examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`. **Two 8B-FP8 instances don't fit** — use Qwen3-4B-FP8 or 1.7B. Shared SMs ⇒ learning exercise, not a speedup | todo |

**Recommended order:** E1p → E2 → E3 → E4 → E5 (E6 and P/D optional). E2 is the main
TTFT experiment and feeds directly into the adaptive-chunk-budget project below.

### Engine-code projects (once the above give baselines)
- **Adaptive chunk budget** in `vllm/v1/core/sched/scheduler.py` (size per-step token budget from
  decode count + TPOT SLO); compare against E2's best static setting.
- **Popularity-aware eviction** in `vllm/v1/core/block_pool.py` (default is LRU); Zipf prefix
  workload at 4× capacity; baseline = E3 hit-rate curve.
- **Break-even offload policy** in `vllm/v1/kv_offload/`: only offload/load prefixes longer than
  E4's measured crossover.

---

## 5. Next steps (pick up here)

- [ ] **Fix cross-point prefix-cache hits** in the harness (§3 known issues) before E2/E3.
- [ ] **E1p** — profile the ~2.6 ms overhead at c=1 (torch profiler): non-GEMM kernels (norms,
      RoPE, sampling) vs inter-step gaps.
- [ ] **E2** — chunked prefill budget sweep on the mixed workload.
- [ ] Then E3 → E4 → E5 in order.

Current state: baseline, E1, E1b complete; no server or background jobs running; GPU idle.
