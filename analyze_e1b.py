#!/usr/bin/env python3
"""Fit E1b: median ITL vs total KV tokens read per decode step.

If attention is bandwidth-bound, ITL = base + slope × batch_ctx_tokens, and
144 KiB / slope ≈ effective bandwidth. Compare against bwtest.py's measured number.

Usage: python analyze_e1b.py [results/e1b_kv_context/default]
"""

import json
import sys
from pathlib import Path

KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2  # layers × KV heads × head dim × K+V × bf16


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/e1b_kv_context/default")
    rows = []
    for f in sorted(d.glob("ctx_c*__*.json")):
        r = json.loads(f.read_text())
        h = r["harness"]
        c = int(h["workload"].split("_c")[1])
        L = h["sweep_value"]
        out = 256
        # Average context per running sequence over its decode: prompt + half the output.
        ctx = c * (L + out / 2)
        rows.append((c, L, ctx, r["median_itl_ms"], r["median_tpot_ms"],
                     int(h["metrics_delta"].get("vllm:num_preemptions_total", 0))))

    print("|  c |    L | batch ctx tok | ITL p50 ms | TPOT p50 ms | preempt |")
    print("|---|---|---|---|---|---|")
    for c, L, ctx, itl, tpot, pre in sorted(rows):
        print(f"| {c} | {L} | {ctx:,.0f} | {itl:.2f} | {tpot:.2f} | {pre} |")

    for c in sorted({r[0] for r in rows}):
        pts = [(r[2], r[3]) for r in rows if r[0] == c]
        if len(pts) < 2:
            continue
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        slope = sum((x - mx) * (y - my) for x, y in pts) / sxx  # ms per token
        base = my - slope * mx
        bw = KV_BYTES_PER_TOKEN / (slope * 1e-3) / 1e9 if slope > 0 else float("inf")
        print(f"\nc={c}: ITL ≈ {base:.2f} ms + {slope * 1e3:.3f} ms per 1k batch tokens"
              f"  ⇒ implied KV bandwidth {bw:.0f} GB/s")


if __name__ == "__main__":
    main()
