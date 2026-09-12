#!/usr/bin/env python3
"""Windowed per-draft-position acceptance — answers "is DFlash paying off right now?"

Samples the engine metrics twice and diffs `spec_decode_num_accepted_tokens_per_pos_total`
and the draft counters, so the numbers describe the CURRENT workload, not the server lifetime.
"""
import json
import re
import sys
import time
import urllib.request


def grab(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=15) as r:
        txt = r.read().decode()

    def total(name):
        vals = [float(m.group(1)) for m in
                re.finditer(r"^%s\{[^}]*\}\s+([0-9.eE+]+)" % re.escape(name), txt, re.M)]
        return sum(vals) if vals else None

    per_pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+]+)', txt, re.M):
        per_pos[int(m.group(1))] = float(m.group(2))
    others = {n: total("vllm:" + n) for n in (
        "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total", "prompt_tokens_total",
        "generation_tokens_total", "num_requests_running")}
    return per_pos, others, time.time()


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    p0, o0, t0 = grab(url)
    print(f"sampling {gap:.0f}s of live traffic …", flush=True)
    time.sleep(gap)
    p1, o1, t1 = grab(url)
    dt = t1 - t0

    drafts = (o1["spec_decode_num_drafts_total"] or 0) - (o0["spec_decode_num_drafts_total"] or 0)
    accepted = (o1["spec_decode_num_accepted_tokens_total"] or 0) - (o0["spec_decode_num_accepted_tokens_total"] or 0)
    gen = (o1["generation_tokens_total"] or 0) - (o0["generation_tokens_total"] or 0)
    prompt = (o1["prompt_tokens_total"] or 0) - (o0["prompt_tokens_total"] or 0)

    print(f"\nwindow: {dt:.1f}s   running={o1['num_requests_running']}")
    print(f"prompt tokens processed : {prompt:,.0f}  ({prompt/dt:,.1f} tok/s)")
    print(f"generated tokens        : {gen:,.0f}  ({gen/dt:,.1f} tok/s)  <-- decode")
    if drafts:
        print(f"spec steps              : {drafts:,.0f}")
        print(f"accepted draft tokens   : {accepted:,.0f}")
        print(f"draft acceptance        : {100*accepted/(drafts*10):.1f}%  (per draft token)")
        per_step = accepted / drafts
        print(f"accepted per step       : {per_step:.2f}")
        print(f"tokens per step         : {1 + per_step:.2f}   <-- the multiplier you actually get")
    print("\nper-position acceptance (of the drafts that reached that position):")
    print(f"  {'pos':>3} {'accepted':>9} {'of drafts':>10} {'rate':>7}  {'marginal tokens':>16}")
    prev = drafts
    for pos in sorted(p1):
        a = p1[pos] - p0.get(pos, 0.0)
        rate = (a / prev) if prev else 0.0
        print(f"  {pos:>3} {a:>9,.0f} {prev:>10,.0f} {100*rate:>6.1f}%  {a/drafts if drafts else 0:>16.3f}")
        prev = a      # only tokens accepted at pos n can attempt pos n+1
    if drafts and gen:
        print(f"\n=> {gen/drafts:.2f} generated tokens per spec step; without speculation it would be 1.00")


if __name__ == "__main__":
    main()
