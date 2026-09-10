#!/usr/bin/env python3
"""modelwatch — live prefill / decode / queue / cache monitor for vLLM, SGLang and llama.cpp.

Reads the engine's own Prometheus endpoint and diffs the counters, so the numbers do not
depend on any engine's log format, and cumulative ratios never get confused with a single
idle logging window (an idle 10 s window legitimately reports a 0% hit rate).

P1 scope: engine detection, scraping, windowed rates, cumulative cache ratio, one line per
tick, optional GPU line.

Usage:
    python3 modelwatch.py                          # auto-detect on :8080, 2 s interval
    python3 modelwatch.py --interval 1 --gpu       # faster tick + nvidia-smi line
    python3 modelwatch.py --once                   # single snapshot, then exit
    python3 modelwatch.py --json                   # one JSON object per tick
    python3 modelwatch.py --url http://127.0.0.1:30000   # SGLang default port
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- engine maps
# Only the names differ between engines; the semantics are the same counters/gauges.
ENGINES = {
    "vllm": {
        "match": "vllm:prompt_tokens_total",
        "prompt_tokens": "vllm:prompt_tokens_total",
        "generation_tokens": "vllm:generation_tokens_total",
        "running": "vllm:num_requests_running",
        "waiting": "vllm:num_requests_waiting",
        "kv_usage": "vllm:kv_cache_usage_perc",
        "kv_usage_scale": 100.0,          # vLLM reports a 0..1 ratio
        "cache_hits": "vllm:prefix_cache_hits_total",
        "cache_queries": "vllm:prefix_cache_queries_total",
        "preemptions": "vllm:num_preemptions_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    },
    "sglang": {
        "match": "sglang:prompt_tokens_total",
        "prompt_tokens": "sglang:prompt_tokens_total",
        "generation_tokens": "sglang:generation_tokens_total",
        "running": "sglang:num_running_reqs",
        "waiting": "sglang:num_queue_reqs",
        "kv_usage": "sglang:token_usage",
        "kv_usage_scale": 100.0,
        "cache_hits": None,               # SGLang exposes a hit-rate gauge instead
        "cache_queries": None,
        "cache_rate_gauge": "sglang:cache_hit_rate",
        "preemptions": None,
        "draft_tokens": None,
        "accepted_tokens": None,
    },
    "llama": {
        "match": "llamacpp:prompt_tokens_total",
        "prompt_tokens": "llamacpp:prompt_tokens_total",
        "generation_tokens": "llamacpp:tokens_predicted_total",
        "running": "llamacpp:requests_processing",
        "waiting": "llamacpp:requests_deferred",
        "kv_usage": "llamacpp:kv_cache_usage_ratio",
        "kv_usage_scale": 100.0,
        "cache_hits": None,
        "cache_queries": None,
        "preemptions": None,
        "draft_tokens": None,
        "accepted_tokens": None,
    },
}


# --------------------------------------------------------------------------- prometheus
def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_prometheus(text: str):
    """name -> {"values": {labelstr: value}, "total": sum-of-all-label-sets}"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            labelstr, _, value = rest.rpartition("}")
            labelstr = labelstr.strip()
            value = value.strip()
        else:
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, value = parts[0], parts[1]
            labelstr = ""
        try:
            val = float(value)
        except ValueError:
            continue
        entry = out.setdefault(name.strip(), {"values": {}, "total": 0.0})
        entry["values"][labelstr] = val
        entry["total"] += val
    return out


class Sample:
    __slots__ = ("t", "metrics")

    def __init__(self, t, metrics):
        self.t = t
        self.metrics = metrics

    def total(self, name):
        if not name:
            return None
        e = self.metrics.get(name)
        return e["total"] if e else None


def rate(prev, cur, name, default=None):
    """Per-second rate of a counter between samples; reset- and None-safe."""
    if not name:
        return default
    a, b = prev.total(name), cur.total(name)
    if a is None or b is None:
        return default
    dt = cur.t - prev.t
    if dt <= 0:
        return default
    if b < a:                      # counter reset (engine restart)
        return default
    return (b - a) / dt


# --------------------------------------------------------------------------- gpu
def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None
    parts = []
    for line in out.splitlines():
        try:
            idx, util, power, mem = [x.strip() for x in line.split(",")]
            parts.append(f"GPU{idx} {util}% {float(power):.0f}W {int(mem)}MiB")
        except ValueError:
            continue
    return " | ".join(parts) if parts else None


def detect_engine(metrics):
    for name, spec in ENGINES.items():
        if spec["match"] in metrics:
            return name
    return None


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live prefill/decode monitor for vLLM, SGLang, llama.cpp")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--engine", choices=["auto", "vllm", "sglang", "llama"], default="auto")
    ap.add_argument("--once", action="store_true", help="print one sample and exit")
    ap.add_argument("--json", action="store_true", help="emit one JSON object per tick")
    ap.add_argument("--gpu", action="store_true", help="append an nvidia-smi line each tick")
    ap.add_argument("--ticks", type=int, default=0, help="stop after N ticks (0 = forever)")
    args = ap.parse_args()

    engine = spec = None
    prev = None
    tick = 0

    if not args.json:
        print(f"modelwatch  url={args.url}  interval={args.interval}s   (ctrl-c to stop)", flush=True)

    while True:
        tick += 1
        now = time.time()
        try:
            metrics = parse_prometheus(fetch(args.url))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(json.dumps({"t": round(now, 3), "error": str(exc)}) if args.json
                  else f"  !! cannot read {args.url}/metrics: {exc}", flush=True)
            if args.once:
                return 2
            time.sleep(args.interval)
            continue

        if engine is None:
            engine = args.engine if args.engine != "auto" else detect_engine(metrics)
            if engine is None:
                msg = ("unrecognised metrics (no vllm:/sglang:/llamacpp: counters) — "
                       "is /metrics enabled on this engine?")
                print(json.dumps({"t": round(now, 3), "error": msg, "metric_names": len(metrics)})
                      if args.json else f"  !! {msg}  (saw {len(metrics)} metric names)", flush=True)
                if args.once:
                    return 2
                time.sleep(args.interval)
                continue
            spec = ENGINES[engine]
            if not args.json:
                print(f"detected engine: {engine}", flush=True)
                print(f"{'time':>8} {'run':>4} {'wait':>4} {'prefill':>9} {'decode':>8} "
                      f"{'KV%':>6} {'cache%':>7} {'accept%':>8} {'preempt':>7}", flush=True)

        cur = Sample(now, metrics)

        prefill = rate(prev, cur, spec["prompt_tokens"]) if prev else None
        decode = rate(prev, cur, spec["generation_tokens"]) if prev else None
        running = cur.total(spec["running"])
        waiting = cur.total(spec["waiting"])
        kv = cur.total(spec["kv_usage"])
        if kv is not None and spec.get("kv_usage_scale"):
            kv *= spec["kv_usage_scale"]

        # cumulative ratio over the whole server lifetime — never a single idle window
        hits, queries = cur.total(spec.get("cache_hits")), cur.total(spec.get("cache_queries"))
        if hits is not None and queries:
            cache_pct, cache_kind = 100.0 * hits / queries, "cumulative"
        elif spec.get("cache_rate_gauge"):
            g = cur.total(spec["cache_rate_gauge"])
            cache_pct, cache_kind = (g * 100.0 if g is not None else None), "gauge"
        else:
            cache_pct, cache_kind = None, None

        if prev is not None:
            d = rate(prev, cur, spec.get("draft_tokens"))
            a = rate(prev, cur, spec.get("accepted_tokens"))
            accept_pct = (100.0 * a / d) if (d and a is not None and d > 0) else None
        else:
            accept_pct = None

        preempt = cur.total(spec.get("preemptions"))
        quiet = (running or 0) == 0 and (waiting or 0) == 0 and not prefill and not decode

        def f(v, w, p=1):
            return "-".rjust(w) if v is None else f"{v:.{p}f}".rjust(w)

        row = {
            "t": round(now, 3),
            "engine": engine,
            "running": running,
            "waiting": waiting,
            "prefill_tok_s": round(prefill, 1) if prefill is not None else None,
            "decode_tok_s": round(decode, 1) if decode is not None else None,
            "kv_cache_pct": round(kv, 1) if kv is not None else None,
            "cache_hit_pct": round(cache_pct, 1) if cache_pct is not None else None,
            "cache_hit_kind": cache_kind,
            "cache_hits_total": int(hits) if hits is not None else None,
            "cache_queries_total": int(queries) if queries is not None else None,
            "accept_pct": round(accept_pct, 1) if accept_pct is not None else None,
            "preemptions_total": int(preempt) if preempt is not None else None,
        }
        if args.gpu:
            row["gpu"] = gpu_line()

        if args.json:
            print(json.dumps(row), flush=True)
        else:
            stamp = time.strftime("%H:%M:%S")
            print(f"{stamp:>8} {f(running,4,0)} {f(waiting,4,0)} {f(prefill,9)} {f(decode,8)} "
                  f"{f(kv,6)} {f(cache_pct,7)} {f(accept_pct,8)} {f(preempt,7,0)}"
                  + ("   [quiet: no requests in this window]" if quiet else ""), flush=True)
            if args.gpu and row.get("gpu"):
                print(f"{'':>8} {row['gpu']}", flush=True)

        prev = cur
        if args.once or (args.ticks and tick >= args.ticks):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
