#!/usr/bin/env python3
"""Context sweep: prefill and decode vs prompt length, one cold request per tier.

Every prompt starts with a unique nonce, so prefix caching cannot turn a repeat into a
cache hit — each tier really pays for its own prefill.

Reports three sources of truth side by side:
  * client-side timing (TTFT from the stream, decode from inter-token times)
  * server-side per-request metrics (vLLM >= 0.29 `metrics` block), when enabled
  * speculative-decoding acceptance for the same request

NOTE (vLLM 0.29): with stream=true the response `metrics` block carries ONLY
`speculative_decoding`. The timing fields (`time_to_first_token_ms`,
`generation_time_ms`, `queue_time_ms`, `mean_itl_ms`, `tokens_per_second`) are
returned for non-streaming requests only, so they stay "-" here.

Usage:
    python3 context_sweep.py                       # default tiers 8K..128K
    python3 context_sweep.py 8192 32768 131072
    python3 context_sweep.py --max-tokens 512
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import urllib.request

FILLER = ("The quarterly infrastructure report covers storage, networking, and compute "
          "capacity. Each section lists measured values and their provenance.\n")
QUESTION = "\nSummarise in one sentence what this document is about."
DEFAULT_TIERS = [8192, 16384, 32768, 65536, 98304, 131072]


def build_prompt(nonce: str, lines: int) -> str:
    # the nonce must lead: it changes the first block, so no prefix reuse is possible
    return f"Request nonce {nonce}. Ignore it.\n\n" + FILLER * lines + QUESTION


def stream_once(base_url, model, prompt, max_tokens, timeout=1800):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    t0 = time.monotonic()
    ttft = None
    last = None
    n_stream = 0
    usage = {}
    server_metrics = {}
    response_timings = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("metrics"):
                server_metrics = chunk["metrics"]
            if chunk.get("timings"):
                response_timings = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - t0
                    last = now
                    n_stream += 1
    wall = time.monotonic() - t0
    timings = response_timings or {}          # llama.cpp returns native timings
    pt = usage.get("prompt_tokens") or timings.get("prompt_n")
    ct = usage.get("completion_tokens") or n_stream
    decode_window = (last - t0 - ttft) if (last and ttft) else None
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct or timings.get("predicted_n"),
        "ttft_s": round(ttft, 3) if ttft else None,
        "wall_s": round(wall, 2),
        "client_prefill_tok_s": round(pt / ttft, 1) if pt and ttft else None,
        "client_decode_tok_s": (round(max(ct - 1, 0) / decode_window, 1)
                                if decode_window and decode_window > 0 and ct > 1 else None),
        "native_prefill_tok_s": timings.get("prompt_per_second"),
        "native_decode_tok_s": timings.get("predicted_per_second"),
        "native_draft_n": timings.get("draft_n"),
        "native_draft_accepted": timings.get("draft_n_accepted"),
        "server_metrics": server_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiers", nargs="*", type=int, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--log", default="/home/lcy/context-sweep.log")
    args = ap.parse_args()
    tiers = args.tiers or DEFAULT_TIERS

    if not args.model:
        models = json.load(urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30))
        args.model = models["data"][0]["id"]

    out = open(args.log, "w", buffering=1)
    def emit(line=""):
        print(line, flush=True)
        out.write(line + "\n")

    emit(f"model={args.model}  max_tokens={args.max_tokens}  tiers={tiers}")
    emit(f"{'target':>8} {'prompt':>8} {'TTFT s':>8} {'prefill':>9} {'decode':>8} "
         f"{'srv_TTFT':>9} {'srv_decode':>10} {'accept_len':>10} {'accept%':>8}")
    rows = []
    for target in tiers:
        nonce = secrets.token_hex(8)
        lines = max(1, target // 28)
        r = stream_once(args.base_url, args.model, build_prompt(nonce, lines), args.max_tokens)
        # re-scale to hit the target more precisely, then re-measure
        if r["prompt_tokens"]:
            lines = max(1, int(lines * target / r["prompt_tokens"]))
            r = stream_once(args.base_url, args.model, build_prompt(secrets.token_hex(8), lines),
                            args.max_tokens)
        sm = r.get("server_metrics") or {}
        sd = sm.get("speculative_decoding") or {}
        rows.append({"target": target, **r})
        if r.get("native_prefill_tok_s"):
            sm = {"time_to_first_token_ms": None}
            emit(f"{'  native':>8} {'':>8} {'':>8} {r['native_prefill_tok_s']:>9.1f} "
                 f"{r['native_decode_tok_s']:>8.1f}  (llama.cpp timings)"
                 + (f"  draft {r['native_draft_accepted']}/{r['native_draft_n']}"
                    if r.get('native_draft_n') else ""))
        emit(f"{target:>8} {str(r['prompt_tokens']):>8} {str(r['ttft_s']):>8} "
             f"{str(r['client_prefill_tok_s']):>9} {str(r['client_decode_tok_s']):>8} "
             f"{str(round(sm['time_to_first_token_ms'] / 1000, 3)) if sm.get('time_to_first_token_ms') else '-':>9} "
             f"{str(sm.get('tokens_per_second')):>10} "
             f"{str(round(sd['mean_acceptance_length'], 2)) if sd else '-':>10} "
             f"{str(round(sd['draft_acceptance_rate'] * 100, 1)) if sd else '-':>8}")
    with open(args.log.replace('.log', '.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    emit(f"\nwritten: {args.log.replace('.log', '.json')}")


if __name__ == "__main__":
    sys.exit(main())
