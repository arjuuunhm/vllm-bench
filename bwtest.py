#!/usr/bin/env python3
"""Measure achievable GPU memory bandwidth (run with no server on the GPU).

Two patterns: copy (read + write) and a read-dominated reduction. The read number is
the relevant floor for decode (weights and KV are read, outputs are tiny).
"""

import torch

N_BYTES = 2 * 1024**3  # 2 GiB per buffer — far larger than L2
ITERS = 20


def timed(fn) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS / 1e3  # seconds per iter


def main():
    src = torch.empty(N_BYTES // 2, dtype=torch.bfloat16, device="cuda").uniform_()
    dst = torch.empty_like(src)

    t = timed(lambda: dst.copy_(src))
    print(f"copy (read+write): {2 * N_BYTES / t / 1e9:6.0f} GB/s")

    t = timed(lambda: src.sum(dtype=torch.float32))
    print(f"read (sum):        {N_BYTES / t / 1e9:6.0f} GB/s")

    print(f"spec: 896 GB/s; decode floor for 8.2 GB/step at measured read BW: "
          f"{8.2e9 / (N_BYTES / t) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
