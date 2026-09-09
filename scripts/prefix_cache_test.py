#!/usr/bin/env python3
"""Prefix-cache A/B test / 前缀缓存效果实测.

Sends the SAME long prompt twice (no nonce, so the second request can reuse the
prefix) and reports TTFT / prefill speed for each, plus what the server reports.

    python3 prefix_cache_test.py --context-tokens 30000
    python3 prefix_cache_test.py --context-tokens 60000 --base-url http://127.0.0.1:8080/v1

Exit code 0 if the second request was faster, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

FILLER = ("The quarterly infrastructure report covers storage, networking, and compute capacity. "
          "Each section lists measured values and their provenance for the audit trail.\n")
QUESTION = "\nSummarise in one sentence what this document is about."


def stream_once(base_url, model, prompt, max_tokens, timeout):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    started = time.monotonic()
    ttft = None
    usage = {}
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
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.monotonic() - started
    wall = time.monotonic() - started
    pt = usage.get("prompt_tokens")
    return {"ttft": round(ttft, 3) if ttft else None,
            "wall": round(wall, 3),
            "prompt_tokens": pt,
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_tok_s": round(pt / ttft, 1) if pt and ttft else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="")
    p.add_argument("--context-tokens", type=int, default=30000)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    if not args.model:
        models = json.load(urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30))
        args.model = models["data"][0]["id"]
    print(f"model: {args.model}  target context: {args.context_tokens} tokens")

    # 28 chars per filler line ~= 28 tokens observed; calibrate on the first request
    lines = max(1, args.context_tokens // 28)
    prompt = FILLER * lines + QUESTION

    print("\n--- request 1 (cold, populates the prefix cache) ---")
    first = stream_once(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
    print(json.dumps(first, indent=2))

    # re-scale filler so request 2 uses the same real prompt size
    if first["prompt_tokens"]:
        lines = max(1, int(lines * args.context_tokens / first["prompt_tokens"]))
        prompt = FILLER * lines + QUESTION
    print("\n--- request 2 (identical prompt, should hit the prefix cache) ---")
    second = stream_once(args.base_url, args.model, prompt, args.max_tokens, args.timeout)
    print(json.dumps(second, indent=2))

    print("\n=== result ===")
    print(f"prompt tokens : {first['prompt_tokens']} -> {second['prompt_tokens']}")
    print(f"TTFT          : {first['ttft']}s -> {second['ttft']}s")
    print(f"prefill tok/s : {first['prefill_tok_s']} -> {second['prefill_tok_s']}")
    if first["ttft"] and second["ttft"]:
        speedup = first["ttft"] / second["ttft"]
        print(f"TTFT speedup  : {speedup:.1f}x")
        return 0 if speedup > 1.5 else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
