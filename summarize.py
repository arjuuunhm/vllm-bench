#!/usr/bin/env python3
"""Print a markdown table (and write summary.csv) for one experiment's results.

Usage: python summarize.py results/e1_decode
"""

import csv
import json
import sys
from pathlib import Path

COLS = [
    ("config", None),
    ("workload", None),
    ("sweep", None),
    ("ttft_p50", "median_ttft_ms"),
    ("ttft_p99", "p99_ttft_ms"),
    ("tpot_p50", "median_tpot_ms"),
    ("tpot_p99", "p99_tpot_ms"),
    ("itl_p50", "median_itl_ms"),
    ("itl_p99", "p99_itl_ms"),
    ("out_tok_s", "output_throughput"),
    ("req_s", "request_throughput"),
    ("prefix_hit%", None),
    ("preempt", None),
]


def rows(exp_dir: Path):
    for cfg_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        for f in cfg_dir.glob("*.json"):
            if f.name == "meta.json":
                continue
            d = json.loads(f.read_text())
            h = d.get("harness", {})
            md = h.get("metrics_delta", {})
            q = md.get("vllm:prefix_cache_queries_total", 0)
            row = {
                "config": cfg_dir.name,
                "workload": h.get("workload", f.stem),
                "sweep": f"{h.get('sweep_key')}={h.get('sweep_value')}",
                "_sort": (cfg_dir.name, h.get("workload", ""), h.get("sweep_value", 0)),
                "prefix_hit%": round(100 * md.get("vllm:prefix_cache_hits_total", 0) / q, 1) if q else "",
                "preempt": int(md.get("vllm:num_preemptions_total", 0)),
            }
            for name, key in COLS:
                if key:
                    v = d.get(key)
                    row[name] = round(v, 2) if isinstance(v, float) else v
            yield row


def main():
    exp_dir = Path(sys.argv[1])
    data = sorted(rows(exp_dir), key=lambda r: r["_sort"])
    names = [c for c, _ in COLS]
    print("| " + " | ".join(names) + " |")
    print("|" + "---|" * len(names))
    for r in data:
        print("| " + " | ".join(str(r.get(c, "")) for c in names) + " |")
    with open(exp_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


if __name__ == "__main__":
    main()
