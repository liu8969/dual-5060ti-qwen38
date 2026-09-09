import json, time, urllib.request, sys

BASE = "http://127.0.0.1:8080"
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=15))["data"][0]["id"]
print("model:", MODEL, flush=True)

PROMPT = ("Write a Python function `lru_cache_ttl(maxsize, ttl)` implementing an LRU cache "
          "with per-entry TTL expiry, thread-safe. Include docstring and a short usage example.")
FILLER = ("The quarterly infrastructure report covers storage, networking, and compute capacity. "
          "Each section lists measured values and their provenance for the audit trail. ")
NEEDLES = ["ZEBRA-ALPHA-8842", "ZEBRA-BETA-3391", "ZEBRA-GAMMA-7705",
           "ZEBRA-DELTA-6617", "ZEBRA-OMEGA-2054"]
POS = [0.05, 0.25, 0.50, 0.75, 0.95]


def post(payload, timeout=3600):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0


def build(n_tokens):
    reps = max(1, n_tokens // 28)
    parts, placed = [], set()
    for i in range(reps):
        parts.append(FILLER)
        frac = i / max(reps - 1, 1)
        for k, p in enumerate(POS):
            if k not in placed and frac >= p:
                parts.append(f" NOTE: the secret code {NEEDLES[k]} was recorded at section {i}. ")
                placed.add(k)
    for k in range(len(POS)):
        if k not in placed:
            parts.append(f" NOTE: the secret code {NEEDLES[k]} was recorded at the end. ")
    return "".join(parts)


def short_bench():
    for label, thinking in [("thinking-off", False), ("thinking-on", True)]:
        try:
            body, dt = post({"model": MODEL,
                             "messages": [{"role": "user", "content": PROMPT}],
                             "max_tokens": 512, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": thinking}})
            u = body.get("usage", {})
            ct, pt = u.get("completion_tokens", 0), u.get("prompt_tokens", 0)
            print(f"{label}: prompt={pt} completion={ct} wall={dt:.2f}s decode={ct/dt:.1f} tok/s",
                  flush=True)
        except Exception as e:
            print(f"{label} FAIL {type(e).__name__} {str(e)[:200]}", flush=True)


def needle(n_tokens):
    doc = build(n_tokens)
    q = ("List every secret code you found in this document, one per line, exactly as written. "
         "If you found none, reply NONE.")
    try:
        body, dt = post({"model": MODEL,
                         "messages": [{"role": "user", "content": doc + "\n\n" + q}],
                         "max_tokens": 256, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})
        u = body.get("usage", {})
        txt = body["choices"][0]["message"].get("content") or ""
        hits = [m for m in NEEDLES if m in txt]
        pt = u.get("prompt_tokens", 0)
        print(f"needle target={n_tokens} prompt_tokens={pt} wall={dt:.1f}s "
              f"prefill={pt/dt:.0f} tok/s recall {len(hits)}/{len(NEEDLES)}: {hits}", flush=True)
    except Exception as e:
        print(f"needle {n_tokens} FAIL {type(e).__name__} {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    short_bench()
    for n in [int(x) for x in sys.argv[1:]] or [100000]:
        needle(n)
