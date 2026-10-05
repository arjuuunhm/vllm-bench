#!/usr/bin/env python3
"""Benchmark harness: for each server config, launch `vllm serve`, run a sweep of
`vllm bench serve` workloads against it, save results + /metrics deltas, tear down.

Usage:
    python harness.py experiments/e1_decode.py [--only CONFIG ...] [--force]

An experiment file defines a module-level EXPERIMENT dict:

    EXPERIMENT = {
        "name": "e1_decode",
        "model": "Qwen/Qwen3-8B-FP8",
        "base_args": ["--max-model-len", "8192", ...],   # passed to every config
        "configs": {"default": [], "eager": ["--enforce-eager"]},
        "config_env": {"eager": {"SOME_VAR": "1"}},      # optional per-config env
        "workloads": [
            {
                "name": "decode",
                "args": ["--dataset-name", "random", ...],  # passed to every point
                "sweep": {"max-concurrency": [1, 2, 4]},    # one bench run per value
                "num_prompts": lambda v: max(16, 8 * v),    # optional, from sweep value
            },
        ],
    }

Results land in results/<experiment>/<config>/<workload>__<sweepkey>=<value>.json.
Existing results are skipped unless --force, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VLLM_DIR = Path.home() / "dev" / "vllm"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"

# Counters/gauges worth diffing around each bench run.
METRICS_OF_INTEREST = [
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]


def load_experiment(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXPERIMENT


def scrape_metrics() -> dict[str, float]:
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$", line)
        if m and m.group(1) in METRICS_OF_INTEREST:
            # Sum across label sets (e.g. per-engine / per-model labels).
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_mem_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().split()[0])


def wait_for_gpu_free(threshold_mib: int = 2000, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if gpu_mem_used_mib() < threshold_mib:
            return
        time.sleep(1)
    print(f"  ! GPU memory still >{threshold_mib} MiB after {timeout}s", flush=True)


def start_server(model: str, args: list[str], env_extra: dict, log_path: Path):
    if server_healthy():
        sys.exit(f"Something is already serving on port {PORT}; stop it first.")
    cmd = ["vllm", "serve", model, "--port", str(PORT), *args]
    env = {**os.environ, **env_extra}
    log = open(log_path, "w")
    log.write(f"$ {' '.join(cmd)}\n# env overrides: {env_extra}\n\n")
    log.flush()
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=VLLM_DIR,
        start_new_session=True,  # own process group so we can kill engine core too
    )
    t0 = time.time()
    while not server_healthy():
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {log_path}")
        if time.time() - t0 > 900:
            stop_server(proc)
            raise RuntimeError(f"server not healthy after 900s; see {log_path}")
        time.sleep(2)
    return proc, time.time() - t0


def stop_server(proc) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    wait_for_gpu_free()


def run_bench(model: str, wl: dict, key: str, value, out_dir: Path, fname: str) -> dict:
    num_prompts = wl.get("num_prompts", lambda v: 32)(value)
    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--base-url", BASE_URL,
        "--num-prompts", str(num_prompts),
        f"--{key}", str(value),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--seed", "0",
        "--disable-tqdm",
        "--save-result", "--result-dir", str(out_dir), "--result-filename", fname,
        *wl["args"],
    ]
    before = scrape_metrics()
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=VLLM_DIR)
    wall = time.time() - t0
    after = scrape_metrics()
    (out_dir / fname.replace(".json", ".log")).write_text(
        f"$ {' '.join(cmd)}\n\n{res.stdout}\n{res.stderr}"
    )
    if res.returncode != 0:
        raise RuntimeError(f"bench failed ({res.returncode}); see {out_dir / fname}.log")

    # Annotate the saved result with harness metadata and server-side counter deltas.
    path = out_dir / fname
    data = json.loads(path.read_text())
    data["harness"] = {
        "workload": wl["name"],
        "sweep_key": key,
        "sweep_value": value,
        "bench_cmd": cmd,
        "wall_s": wall,
        "metrics_delta": {k: after.get(k, 0) - before.get(k, 0) for k in METRICS_OF_INTEREST},
    }
    path.write_text(json.dumps(data, indent=2))
    return data


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(VLLM_DIR), "describe", "--always", "--dirty"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--only", nargs="*", help="run only these config names")
    ap.add_argument("--force", action="store_true", help="re-run existing results")
    a = ap.parse_args()

    exp = load_experiment(a.experiment)
    exp_dir = ROOT / "results" / exp["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    for cfg_name, cfg_args in exp["configs"].items():
        if a.only and cfg_name not in a.only:
            continue
        cfg_dir = exp_dir / cfg_name
        cfg_dir.mkdir(exist_ok=True)

        points = [
            (wl, key, v, f"{wl['name']}__{key}={v}.json")
            for wl in exp["workloads"]
            for key, values in wl["sweep"].items()
            for v in values
        ]
        todo = [p for p in points if a.force or not (cfg_dir / p[3]).exists()]
        if not todo:
            print(f"[{cfg_name}] all {len(points)} points done, skipping", flush=True)
            continue

        args = [*exp.get("base_args", []), *cfg_args]
        env_extra = exp.get("config_env", {}).get(cfg_name, {})
        print(f"[{cfg_name}] starting server: {' '.join(args)}", flush=True)
        proc, startup_s = start_server(exp["model"], args, env_extra, cfg_dir / "server.log")
        print(f"[{cfg_name}] healthy after {startup_s:.0f}s", flush=True)
        (cfg_dir / "meta.json").write_text(json.dumps({
            "model": exp["model"], "server_args": args, "env": env_extra,
            "vllm_git": git_rev(), "startup_s": startup_s,
        }, indent=2))
        try:
            for wl, key, v, fname in todo:
                d = run_bench(exp["model"], wl, key, v, cfg_dir, fname)
                print(
                    f"[{cfg_name}] {wl['name']} {key}={v}: "
                    f"TTFT p50 {d['median_ttft_ms']:.1f}ms  "
                    f"TPOT p50 {d['median_tpot_ms']:.2f}ms  "
                    f"out {d['output_throughput']:.0f} tok/s",
                    flush=True,
                )
        finally:
            stop_server(proc)
            print(f"[{cfg_name}] server stopped", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
