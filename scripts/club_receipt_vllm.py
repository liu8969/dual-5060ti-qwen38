#!/usr/bin/env python3
"""club-5060ti protocol receipt for vLLM — streaming, TTFT-aware.

This is a faithful port of club-5060ti's `scripts/run_high_context_profile.py`
workload and pass/fail rules, adapted for vLLM so the numbers are
protocol-comparable instead of null.

Kept identical to the upstream runner:
  - same NEEDLE / FILLER text and the unique leading request nonce
    (so prompt caching cannot make the prefill look artificially cheap)
  - same calibration loop (probe at 32 filler lines, then adjust to hit the
    target prompt size from real server token counts)
  - retrieval: 2 repeats, 512 output tokens, final answer must equal the needle
  - sustained: 2 repeats, 3072 output tokens, >=1076 generated tokens and
    >=800 client-visible characters (reasoning alone does not qualify)
  - same profile JSON (`data/benchmark-profiles/high-context.json`)

Deviations, and why (documented for the maintainers):
  1. metrics come from the streaming SSE stream: TTFT, prefill tok/s
     (prompt_tokens / TTFT) and decode tok/s (tokens after TTFT / decode window).
     Upstream reads `response["timings"]`, which only llama.cpp emits.
  2. the context ceiling is read from vLLM's `/v1/models` `max_model_len`
     instead of llama.cpp's `status.args --ctx-size/--parallel`.

Usage:
    python3 club_receipt_vllm.py \
        --base-url http://127.0.0.1:8080/v1 \
        --model Qwen3.8-27B-Q6-dual-5060ti \
        --preset vllm-qwen38-27b-dflash2-dual-5060ti \
        --context-tokens 131072 --disable-thinking \
        --output receipt-131k.json
"""
from __future__ import annotations

import argparse
import json
import math
import secrets
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NEEDLE = "CLUB-5060TI-HIGH-CONTEXT-NEEDLE-48291"
FILLER = ("Benchmark context filler. Preserve operational facts, configuration fields, "
          "and evidence boundaries.\n")


# --------------------------------------------------------------------------- http
def _headers(args):
    h = {"content-type": "application/json"}
    if args.api_key:
        h["authorization"] = f"Bearer {args.api_key}"
    return h


def get_json(url, timeout, api_key):
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_chat(args, payload, timeout):
    """POST /chat/completions with stream=true; return (usage, visible, reasoning, timing)."""
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=_headers(args),
        method="POST",
    )
    started = time.monotonic()
    ttft = None
    last_token_at = None
    streamed_tokens = 0
    visible: list[str] = []
    reasoning: list[str] = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
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
                text = delta.get("content")
                if isinstance(text, str) and text:
                    if ttft is None:
                        ttft = time.monotonic() - started
                    last_token_at = time.monotonic()
                    streamed_tokens += 1
                    visible.append(text)
                for field in ("reasoning_content", "reasoning"):
                    value = delta.get(field)
                    if isinstance(value, str) and value:
                        if ttft is None:
                            ttft = time.monotonic() - started
                        last_token_at = time.monotonic()
                        reasoning.append(value)
    wall = time.monotonic() - started
    decode_window = (last_token_at - started - (ttft or 0)) if last_token_at and ttft else None
    completion = usage.get("completion_tokens") or streamed_tokens
    prompt_tokens = usage.get("prompt_tokens")
    timing = {
        "ttft_seconds": round(ttft, 3) if ttft is not None else None,
        "wall_seconds": round(wall, 3),
        "streamed_tokens": streamed_tokens,
        "prompt_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens and ttft else None,
        "decode_tok_s": (round(max(completion - 1, 0) / decode_window, 1)
                         if decode_window and decode_window > 0 and completion > 1 else None),
    }
    return usage, "".join(visible), "".join(reasoning), timing


# ------------------------------------------------------------------------ workload
def prompt(kind, lines, nonce):
    prefix = "Benchmark request nonce: " + nonce + ". Do not repeat it.\n\n"
    if kind == "retrieval":
        suffix = "\nQuestion: reply with only the exact key."
        prefix += "Remember this exact key: " + NEEDLE + "\n\n"
    else:
        suffix = ("\nWrite a detailed, practical guide to validating a local inference preset. "
                  "Cover context fit, retrieval, sustained generation, caveats, and reproducibility.")
        prefix += "Use the following local benchmark context as source material.\n\n"
    return prefix + FILLER * lines + suffix


def call(args, kind, lines, output_tokens):
    nonce = secrets.token_hex(12)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt(kind, lines, nonce)}],
        "temperature": 0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,          # ignored by vLLM, kept for protocol parity
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        usage, visible, reasoning, timing = stream_chat(args, payload, args.timeout)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the run
        return {"kind": kind, "lines": lines, "passed": False, "error": f"{type(exc).__name__}: {exc}"}

    completion = usage.get("completion_tokens") or timing["streamed_tokens"]
    minimum_completion = (min(output_tokens, args.minimum_generated_tokens)
                          if args.minimum_generated_tokens is not None
                          else math.ceil(output_tokens * args.minimum_output_fraction))
    passed = (visible.strip() == NEEDLE if kind == "retrieval"
              else (completion >= minimum_completion
                    and len(visible.strip()) >= args.minimum_visible_output_characters))
    return {
        "kind": kind, "lines": lines, "request_nonce": nonce, "passed": passed,
        "actual_prompt_tokens": usage.get("prompt_tokens"),
        "generated_tokens": completion,
        "prompt_tok_s": timing["prompt_tok_s"],
        "decode_tok_s": timing["decode_tok_s"],
        "ttft_seconds": timing["ttft_seconds"],
        "wall_seconds": timing["wall_seconds"],
        "response": visible[:500],
        "reasoning_response": reasoning[:500],
        "visible_output_characters": len(visible.strip()),
        "reasoning_output_characters": len(reasoning.strip()),
        "failure": None if passed else (
            "retrieval final-answer mismatch" if kind == "retrieval"
            else "generation stopped before required visible output"),
    }


def calibrate(args, kind, target_tokens, output_tokens):
    probe = call(args, kind, 32, output_tokens)
    observed = probe.get("actual_prompt_tokens")
    if not observed:
        return None, [probe]
    per_line = max(0.01, observed / 32)
    lines = max(1, math.ceil(target_tokens / per_line))
    attempts = [probe]
    for _ in range(args.max_calibration_attempts):
        measured = call(args, kind, lines, output_tokens)
        attempts.append(measured)
        actual = measured.get("actual_prompt_tokens")
        if not actual:
            lines = max(1, math.floor(lines * 0.80))
            continue
        if actual >= target_tokens and actual <= args.context_tokens - output_tokens:
            return lines, attempts
        ratio = target_tokens / actual
        lines = max(1, math.ceil(lines * min(1.25, max(0.70, ratio))))
    return None, attempts


def median(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    return round(statistics.median(values), 3) if values else None


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="club-5060ti high-context profile, vLLM adapter.")
    p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--context-tokens", required=True, type=int)
    p.add_argument("--profile", default="data/benchmark-profiles/high-context.json")
    p.add_argument("--retrieval-repeats", type=int, default=None)
    p.add_argument("--sustained-repeats", type=int, default=None)
    p.add_argument("--retrieval-output-tokens", type=int, default=None)
    p.add_argument("--sustained-output-tokens", type=int, default=None)
    p.add_argument("--minimum-prompt-fraction", type=float, default=None)
    p.add_argument("--minimum-output-fraction", type=float, default=None)
    p.add_argument("--minimum-generated-tokens", type=int, default=None)
    p.add_argument("--minimum-visible-output-characters", type=int, default=None)
    p.add_argument("--minimum-decode-tok-s", type=float, default=None)
    p.add_argument("--max-calibration-attempts", type=int, default=4)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output", default="")
    args = p.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    fit = profile["fit"]
    args.retrieval_repeats = args.retrieval_repeats or fit["retrieval_repeats"]
    args.sustained_repeats = args.sustained_repeats or fit["sustained_repeats"]
    args.retrieval_output_tokens = args.retrieval_output_tokens or fit["retrieval_output_tokens"]
    args.sustained_output_tokens = args.sustained_output_tokens or fit["sustained_output_tokens"]
    args.minimum_prompt_fraction = args.minimum_prompt_fraction or fit["minimum_prompt_fraction"]
    args.minimum_visible_output_characters = (args.minimum_visible_output_characters
                                              if args.minimum_visible_output_characters is not None
                                              else fit["minimum_visible_output_characters"])
    args.minimum_generated_tokens = (args.minimum_generated_tokens
                                     if args.minimum_generated_tokens is not None
                                     else fit.get("minimum_generated_tokens"))
    args.minimum_output_fraction = (args.minimum_output_fraction
                                    if args.minimum_output_fraction is not None
                                    else fit.get("minimum_output_fraction", 0.35))
    args.minimum_decode_tok_s = (args.minimum_decode_tok_s
                                 if args.minimum_decode_tok_s is not None
                                 else fit["minimum_decode_tok_s"])
    calibration_margin = fit.get("calibration_margin_tokens", 64)

    # vLLM: read the context ceiling from /v1/models instead of llama.cpp status.args
    models = get_json(args.base_url.rstrip("/") + "/models", 60, args.api_key).get("data", [])
    entry = next((m for m in models if m.get("id") == args.model), None)
    if entry is None:
        raise SystemExit(f"model {args.model!r} is not listed by /models")
    max_model_len = entry.get("max_model_len")
    if max_model_len and args.context_tokens > max_model_len:
        raise SystemExit(f"server max_model_len={max_model_len} < requested tier {args.context_tokens}")

    targets = {
        "retrieval": min(int(args.context_tokens * fit["prompt_fraction"]) + calibration_margin,
                         args.context_tokens - args.retrieval_output_tokens),
        "sustained": min(int(args.context_tokens * fit["sustained_prompt_fraction"]) + calibration_margin,
                         args.context_tokens - args.sustained_output_tokens),
    }
    calibration, cases = {}, []
    for kind, output in (("retrieval", args.retrieval_output_tokens),
                         ("sustained", args.sustained_output_tokens)):
        print(f"[calibrate] {kind} -> {targets[kind]} prompt tokens, {output} output tokens", flush=True)
        lines, attempts = calibrate(args, kind, targets[kind], output)
        calibration[kind] = {"target_prompt_tokens": targets[kind], "filler_lines": lines,
                             "attempts": attempts}
        if lines is None:
            print(f"[calibrate] {kind}: FAILED to reach target", flush=True)
            continue
        repeats = args.retrieval_repeats if kind == "retrieval" else args.sustained_repeats
        for i in range(repeats):
            case = call(args, kind, lines, output)
            cases.append(case)
            print(f"[{kind} {i + 1}/{repeats}] passed={case.get('passed')} "
                  f"prompt={case.get('actual_prompt_tokens')} gen={case.get('generated_tokens')} "
                  f"ttft={case.get('ttft_seconds')}s prefill={case.get('prompt_tok_s')} tok/s "
                  f"decode={case.get('decode_tok_s')} tok/s", flush=True)

    minimum = math.ceil(args.context_tokens * args.minimum_prompt_fraction)
    retrieval = [c for c in cases if c["kind"] == "retrieval"]
    sustained = [c for c in cases if c["kind"] == "sustained"]
    coverage = (all((c.get("actual_prompt_tokens") or 0) >= minimum for c in retrieval)
                and all((c.get("actual_prompt_tokens") or 0) >= int(args.context_tokens * fit["sustained_prompt_fraction"])
                        for c in sustained))
    retrieval_ok = len(retrieval) == args.retrieval_repeats and all(c.get("passed") for c in retrieval)
    sustained_ok = len(sustained) == args.sustained_repeats and all(c.get("passed") for c in sustained)
    decode = median(c.get("decode_tok_s") for c in sustained)
    useful = coverage and retrieval_ok and sustained_ok and decode is not None and decode >= args.minimum_decode_tok_s

    receipt = {
        "schema_version": "1.0",
        "kind": "raw-high-context-profile-vllm-adapter",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "model": args.model,
        "context_tokens": args.context_tokens,
        "active_server_max_model_len": max_model_len,
        "requested_context_matches_server": (max_model_len is None or args.context_tokens <= max_model_len),
        "policy": {
            "prompt_cache_disabled": True,
            "unique_leading_request_nonce": True,
            "disable_thinking": args.disable_thinking,
            "minimum_decode_tok_s": args.minimum_decode_tok_s,
            "sustained_output_tokens": args.sustained_output_tokens,
            "minimum_generated_tokens": args.minimum_generated_tokens,
            "minimum_visible_output_characters": args.minimum_visible_output_characters,
            "calibration_margin_tokens": calibration_margin,
            "profile": args.profile,
            "deviations": [
                "metrics from streaming SSE (TTFT, prefill=prompt_tokens/TTFT, decode=(completion-1)/decode_window); "
                "upstream reads llama.cpp response['timings']",
                "context ceiling from /v1/models max_model_len instead of llama.cpp status.args",
            ],
        },
        "calibration": calibration,
        "summary": {
            "useful": useful,
            "retrieval_passed": retrieval_ok,
            "sustained_passed": sustained_ok,
            "prompt_coverage_passed": coverage,
            "median_sustained_decode_tok_s": decode,
            "median_prompt_tok_s": median(c.get("prompt_tok_s") for c in cases),
            "median_ttft_seconds": median(c.get("ttft_seconds") for c in cases),
        },
        "cases": cases,
    }
    out = Path(args.output) if args.output else Path(f"receipt-ctx{args.context_tokens}.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote receipt to {out}")
    print("USEFUL" if useful else "NOT USEFUL", f"at {args.context_tokens} context")
    return 0 if useful else 2


if __name__ == "__main__":
    raise SystemExit(main())
